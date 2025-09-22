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

DEFINE_LOG_CATEGORY_STATIC(LogNormalCamera, Log, All);

#include "Sensors/NormalCamera.h"
#include "DataStructures/RecordData.h"
#include "RenderingThread.h"
#include "Async/AsyncWork.h"
#include "Windows/WindowsHWrapper.h"
#include "RHI.h"

UNormalCameraComponent::UNormalCameraComponent()
{
}

void UNormalCameraComponent::Configure(const FSensorConfig& Config)
{
    if (!bBPConfigured)
    {       
        const auto NormalConfig = static_cast<const FNormalCameraConfig&>(Config);
        FOV = NormalConfig.FOV;
        Width = NormalConfig.Width;
        Height = NormalConfig.Height;
    }

    ComputeIntrinsics();
    InitializeRenderTargets();
    SetCaptureComponent();
}

TFuture<FSensorDataPacket> UNormalCameraComponent::CaptureDataAsync()
{
    TSharedPtr<TPromise<FSensorDataPacket>> Promise = MakeShared<TPromise<FSensorDataPacket>>();
    TFuture<FSensorDataPacket> Future = Promise->GetFuture();

    AsyncTask(ENamedThreads::GameThread, [this, Promise]()
    {
        FSensorDataPacket Packet;
        Packet.Type = ESensorType::NormalCamera;
        Packet.SensorIndex = GetSensorIndex();
        Packet.OwnerActor = GetOwnerActor();
        Packet.Timestamp = FPlatformTime::Seconds();

        if (!CheckComponentAndRenderTarget())
        {
            Packet.bValid = false;
            Promise->SetValue(Packet);
            return;
        }

        CaptureNormalScene();

        ProcessNormalTextureParam([this, Promise, Packet](const TArray<FLinearColor>& CapturedNormalData) mutable
        {
            Async(EAsyncExecution::TaskGraph, [Promise, Packet, CapturedNormalData, Width = this->Width, Height = this->Height]() mutable
            {
                auto NormalData = MakeShared<FNormalCameraData>();
                NormalData->Timestamp = Packet.Timestamp;
                NormalData->SensorIndex = Packet.SensorIndex;
                NormalData->Width = Width;
                NormalData->Height = Height;
                NormalData->Data = CapturedNormalData;

                Packet.Data = NormalData;
                Packet.bValid = true;
                Promise->SetValue(Packet);
            });
        });
    });

    return Future;
}

void UNormalCameraComponent::SetCaptureComponent() const
{
    Super::SetCaptureComponent();

    if (CaptureComponent)
    {
        CaptureComponent->CaptureSource = ESceneCaptureSource::SCS_Normal;
    }
}

void UNormalCameraComponent::InitializeRenderTargets()
{
    NormalRenderTarget = NewObject<UTextureRenderTarget2D>(this);
    NormalRenderTarget->InitCustomFormat(Width, Height,
        PF_A32B32G32R32F, true);
    NormalRenderTarget->UpdateResource();
}

void UNormalCameraComponent::CaptureNormalScene()
{
    if (!CheckComponentAndRenderTarget())
    {
        UE_LOG(LogNormalCamera, Error, TEXT("Component or RenderTarget not valid!"));
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

void UNormalCameraComponent::CaptureNormalSceneAndProcess()
{
    CaptureNormalScene();
    ProcessNormalTexture([]()
    {
    });
}

void UNormalCameraComponent::ProcessNormalTexture(TFunction<void()> OnComplete)
{
    ProcessNormalTextureTemplate(std::move(OnComplete));    
}

void UNormalCameraComponent::ProcessNormalTextureParam(
    TFunction<void(const TArray<FLinearColor>&)> OnComplete)
{
    ProcessNormalTextureTemplate(std::move(OnComplete));
}

void UNormalCameraComponent::AsyncGetNormalImageData(
    TFunction<void(const TArray<FLinearColor>&)> Callback)
{
    AsyncTask(ENamedThreads::GameThread,
        [this, Callback = MoveTemp(Callback)]() {
        if (!CheckComponentAndRenderTarget())
        {
            UE_LOG(LogNormalCamera, Error, TEXT("Component or RenderTarget not valid!"));
            Callback({});
            return;
        }
        
        CaptureComponent->CaptureScene();
        ProcessNormalTextureParam(Callback);
    });
}