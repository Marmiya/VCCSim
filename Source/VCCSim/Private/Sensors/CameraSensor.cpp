/*
* Copyright (C) 2025 Visual Computing Research Center, Shenzhen University
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

DEFINE_LOG_CATEGORY_STATIC(LogCameraSensor, Log, All);

#include "Sensors/CameraSensor.h"
#include "DataStructures/RecordData.h"
#include "RenderingThread.h"
#include "Async/AsyncWork.h"
#include "Windows/WindowsHWrapper.h"
#include "Utils/InsMeshHolder.h"
#include "Components/InstancedStaticMeshComponent.h"

URGBCameraComponent::URGBCameraComponent()
{
}

void URGBCameraComponent::Configure(const FSensorConfig& Config)
{
    if (!bBPConfigured)
    {
        const auto RGBConfig = static_cast<const FRGBCameraConfig&>(Config);
        FOV = RGBConfig.FOV;
        Width = RGBConfig.Width;
        Height = RGBConfig.Height;
        bOrthographic = RGBConfig.bOrthographic;
        OrthoWidth = RGBConfig.OrthoWidth;
    }
    
    ComputeIntrinsics();
    InitializeRenderTargets();
    SetCaptureComponent();
}

TFuture<FSensorDataPacket> URGBCameraComponent::CaptureDataAsync()
{
    TSharedPtr<TPromise<FSensorDataPacket>> Promise = MakeShared<TPromise<FSensorDataPacket>>();
    TFuture<FSensorDataPacket> Future = Promise->GetFuture();

    AsyncTask(ENamedThreads::GameThread, [this, Promise]()
    {
        FSensorDataPacket Packet;
        Packet.Type = ESensorType::RGBCamera;
        Packet.SensorIndex = GetSensorIndex();
        Packet.OwnerActor = GetOwnerActor();
        Packet.Timestamp = FPlatformTime::Seconds();

        if (!CheckComponentAndRenderTarget())
        {
            Packet.bValid = false;
            Promise->SetValue(Packet);
            return;
        }

        CaptureRGBScene();

        ProcessRGBTextureParam([this, Promise, Packet](const TArray<FColor>& CapturedData) mutable
        {
            Async(EAsyncExecution::TaskGraph, [Promise, Packet, CapturedData, Width = this->Width, Height = this->Height]() mutable
            {
                auto RGBData = MakeShared<FRGBCameraData>();
                RGBData->Timestamp = Packet.Timestamp;
                RGBData->SensorIndex = Packet.SensorIndex;
                RGBData->Width = Width;
                RGBData->Height = Height;
                RGBData->Data = CapturedData;

                Packet.Data = RGBData;
                Packet.bValid = true;
                Promise->SetValue(Packet);
            });
        });
    });

    return Future;
}

void URGBCameraComponent::SetIgnoreLidar(UInsMeshHolder* MeshHolder)
{
    CaptureComponent->HideComponent(MeshHolder->GetInstancedMeshComponent());
    CaptureComponent->HideComponent(MeshHolder->GetInstancedMeshComponentColor());
}

void URGBCameraComponent::SetCaptureComponent() const
{
    Super::SetCaptureComponent();

    if (CaptureComponent)
    {
        CaptureComponent->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
        CaptureComponent->bCaptureEveryFrame = false;
        CaptureComponent->bCaptureOnMovement = false;
        CaptureComponent->bAlwaysPersistRenderingState = true;
        CaptureComponent->PrimaryComponentTick.bCanEverTick = true;

        FEngineShowFlags& ShowFlags = CaptureComponent->ShowFlags;
        ShowFlags.EnableAdvancedFeatures();
        ShowFlags.SetPostProcessing(true);
        ShowFlags.SetTonemapper(true);
        ShowFlags.SetBloom(true);
        ShowFlags.SetLumenGlobalIllumination(true);
        ShowFlags.SetLumenReflections(true);
    }
    else
    {
        UE_LOG(LogCameraSensor, Error, TEXT("Capture component not initialized!"));
    }
}

void URGBCameraComponent::InitializeRenderTargets()
{
    RGBRenderTarget = NewObject<UTextureRenderTarget2D>(this);

    RGBRenderTarget->TargetGamma = GEngine->GetDisplayGamma();
    RGBRenderTarget->InitCustomFormat(Width, Height,
        PF_B8G8R8A8, true);
    RGBRenderTarget->RenderTargetFormat = ETextureRenderTargetFormat::RTF_RGBA8;
    RGBRenderTarget->bGPUSharedFlag = true;
    RGBRenderTarget->bAutoGenerateMips = true;
    
    RGBRenderTarget->UpdateResource();
}

void URGBCameraComponent::CaptureRGBScene()
{
    if (!CheckComponentAndRenderTarget())
    {
        UE_LOG(LogCameraSensor, Error, TEXT("Component or RenderTarget not valid!"));
        return;
    }
    
    if (IsInGameThread())
    {
        CaptureComponent->CaptureScene();
    }
    else
    {
        AsyncTask(ENamedThreads::GameThread, [this]()
        {
            CaptureComponent->CaptureScene();
        });
    }
}

void URGBCameraComponent::AsyncGetRGBImageData(
    TFunction<void(const TArray<FColor>&)> Callback)
{
    AsyncTask(ENamedThreads::GameThread, [this, Callback = MoveTemp(Callback)]()
    {
        if (!CheckComponentAndRenderTarget())
        {
            UE_LOG(LogCameraSensor, Error, TEXT("Component or RenderTarget not valid!"));
            return;
        }
        
        CaptureComponent->CaptureScene();
        
        ProcessRGBTextureParam([Callback](const TArray<FColor>& ColorData)
        {
            Callback(ColorData);
        });
    });
}

void URGBCameraComponent::ProcessRGBTexture(TFunction<void()> OnComplete)
{
    ProcessRGBTextureTemplate(std::move(OnComplete));
}

void URGBCameraComponent::ProcessRGBTextureParam(
    TFunction<void(const TArray<FColor>&)> OnComplete)
{
    ProcessRGBTextureTemplate(std::move(OnComplete));
}