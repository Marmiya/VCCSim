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
    }

    RecordInterval = Config.RecordInterval;

    ComputeIntrinsics();
    InitializeRenderTargets();
    SetCaptureComponent();

    if (CheckComponentAndRenderTarget())
    {
        AsyncTask(ENamedThreads::GameThread, [this]()
        {
            GetWorld()->GetTimerManager().SetTimerForNextTick([this]()
            {
                CaptureSegmentationScene();
            });
        });
    }
}

void USegmentationCameraComponent::InitializeRenderTargets()
{
    RenderTarget = NewObject<UTextureRenderTarget2D>(this);
    RenderTarget->InitCustomFormat(Width, Height, PF_B8G8R8A8, true);
    RenderTarget->RenderTargetFormat = RTF_RGBA8;
    RenderTarget->bAutoGenerateMips = false;
    RenderTarget->UpdateResource();
}

void USegmentationCameraComponent::SetCaptureComponent() const
{
    Super::SetCaptureComponent();

    CaptureComponent->CaptureSource = SCS_FinalColorLDR;
    CaptureComponent->bAlwaysPersistRenderingState = true;
    
    CaptureComponent->ShowFlags.DisableFeaturesForUnlit();
    CaptureComponent->ShowFlags.SetPostProcessing(true); 
    CaptureComponent->ShowFlags.SetPostProcessMaterial(true);
    CaptureComponent->ShowFlags.SetAntiAliasing(false);
    CaptureComponent->ShowFlags.SetMotionBlur(false);
    
    if (SegmentationMaterial)
    {
        CaptureComponent->PostProcessSettings.WeightedBlendables.Array.Empty();
        FWeightedBlendable WeightedBlendable;
        WeightedBlendable.Object = SegmentationMaterial;
        WeightedBlendable.Weight = 1.f;
        CaptureComponent->PostProcessSettings.WeightedBlendables.Array.Add(WeightedBlendable);
    }
    else
    {
        UE_LOG(LogSegmentCamera, Error, TEXT("Segmentation material not set!"));
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

void USegmentationCameraComponent::CaptureSegmentationSceneDeferred()
{
    if (!CheckComponentAndRenderTarget())
    {
        UE_LOG(LogSegmentCamera, Error, TEXT("Component or RenderTarget not valid!"));
        return;
    }

    LastCaptureTimestamp = FPlatformTime::Seconds();
    CaptureComponent->CaptureSceneDeferred();
}

void USegmentationCameraComponent::AsyncGetSegmentationImageData(
    TFunction<void(const TArray<FColor>&)> Callback)
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