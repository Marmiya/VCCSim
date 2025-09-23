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

DEFINE_LOG_CATEGORY_STATIC(LogSegmentCamera, Log, All);

#include "Sensors/SegmentCamera.h"
#include "DataStructures/RecordData.h"
#include "RenderingThread.h"
#include "Async/AsyncWork.h"
#include "Windows/WindowsHWrapper.h"

USegmentationCameraComponent::USegmentationCameraComponent()
{
}

void USegmentationCameraComponent::Configure(const FSensorConfig& Config)
{
    if (!bBPConfigured)
    {
        auto SegConfig = static_cast<const FSegmentationCameraConfig&>(Config);
        FOV = SegConfig.FOV;
        Width = SegConfig.Width;
        Height = SegConfig.Height;
        MaxRange = SegConfig.MaxRange;
    }

    ComputeIntrinsics();
    InitializeRenderTargets();
    SetCaptureComponent();

    if (CheckComponentAndRenderTarget())
    {
        AsyncTask(ENamedThreads::GameThread, [this]()
        {
            GetWorld()->GetTimerManager().SetTimerForNextTick([this]()
            {
                if (CaptureComponent)
                {
                    CaptureComponent->ShowOnlyActors.Empty();
                }
            });
        });
    }
}

TFuture<FSensorDataPacket> USegmentationCameraComponent::CaptureDataAsync()
{
    TSharedPtr<TPromise<FSensorDataPacket>> Promise = MakeShared<TPromise<FSensorDataPacket>>();
    TFuture<FSensorDataPacket> Future = Promise->GetFuture();

    AsyncTask(ENamedThreads::GameThread, [this, Promise]()
    {
        FSensorDataPacket Packet;
        Packet.Type = ESensorType::SegmentationCamera;
        Packet.SensorIndex = GetSensorIndex();
        Packet.OwnerActor = GetOwnerActor();
        Packet.Timestamp = FPlatformTime::Seconds();

        if (!CheckComponentAndRenderTarget())
        {
            Packet.bValid = false;
            Promise->SetValue(Packet);
            return;
        }

        CaptureSegmentationScene();

        ProcessSegmentationTextureParam([this, Promise, Packet](const TArray<FColor>& CapturedSegmentationData) mutable
        {
            Async(EAsyncExecution::TaskGraph, [Promise, Packet, CapturedSegmentationData, Width = this->Width, Height = this->Height]() mutable
            {
                auto SegmentationData = MakeShared<FSegmentationCameraData>();
                SegmentationData->Timestamp = Packet.Timestamp;
                SegmentationData->Width = Width;
                SegmentationData->Height = Height;
                SegmentationData->Data = CapturedSegmentationData;

                Packet.Data = SegmentationData;
                Packet.bValid = true;
                Promise->SetValue(Packet);
            });
        });
    });

    return Future;
}

void USegmentationCameraComponent::InitializeRenderTargets()
{
    SegmentationRenderTarget = NewObject<UTextureRenderTarget2D>(this);
    SegmentationRenderTarget->InitCustomFormat(Width, Height, PF_B8G8R8A8, true);
}

void USegmentationCameraComponent::SetCaptureComponent() const
{
    Super::SetCaptureComponent();

    if (CaptureComponent)
    {
        CaptureComponent->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives;
        CaptureComponent->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
        CaptureComponent->PostProcessSettings.bOverride_AutoExposureMinBrightness = true;
        CaptureComponent->PostProcessSettings.bOverride_AutoExposureMaxBrightness = true;
        CaptureComponent->PostProcessSettings.AutoExposureMinBrightness = 1.0f;
        CaptureComponent->PostProcessSettings.AutoExposureMaxBrightness = 1.0f;
        CaptureComponent->TextureTarget = SegmentationRenderTarget;
    }
}

void USegmentationCameraComponent::CaptureSegmentationScene()
{
    if (!CheckComponentAndRenderTarget())
    {
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

void USegmentationCameraComponent::AsyncGetSegmentationImageData(TFunction<void(const TArray<FColor>&)> Callback)
{
    AsyncTask(ENamedThreads::GameThread, [this, Callback = MoveTemp(Callback)]()
    {
        if (!CheckComponentAndRenderTarget())
        {
            return;
        }

        CaptureComponent->CaptureScene();

        ProcessSegmentationTextureParam([Callback](const TArray<FColor>& ColorData)
        {
            Callback(ColorData);
        });
    });
}

void USegmentationCameraComponent::ProcessSegmentationTexture(TFunction<void()> OnComplete)
{
    ProcessSegTextureTemplate(std::move(OnComplete));
}

void USegmentationCameraComponent::ProcessSegmentationTextureParam(
    TFunction<void(const TArray<FColor>&)> OnComplete)
{
    ProcessSegTextureTemplate(std::move(OnComplete));
}

void USegmentationCameraComponent::ContributeToRDGPass(FSensorViewInfo& OutViewInfo)
{
    OutViewInfo.SensorType = ESensorType::SegmentationCamera;
    OutViewInfo.MRTSlot = GetMRTSlot();

    FVector CameraLocation = GetComponentLocation();
    FRotator CameraRotation = GetComponentRotation();
    OutViewInfo.ViewMatrix = FInverseRotationMatrix(CameraRotation) * FTranslationMatrix(-CameraLocation);

    float FOVRadians = FMath::DegreesToRadians(FOV);
    float AspectRatio = (float)Width / (float)Height;
    OutViewInfo.ProjectionMatrix = FReversedZPerspectiveMatrix(FOVRadians, AspectRatio, GNearClippingPlane, MaxRange);

    OutViewInfo.CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
    OutViewInfo.Resolution = FIntPoint(Width, Height);
    OutViewInfo.Provider = this;
}

