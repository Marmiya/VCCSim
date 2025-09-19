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
#include "Simulation/Recorder.h"
#include "RenderingThread.h"
#include "Async/AsyncWork.h"
#include "Windows/WindowsHWrapper.h"

USegmentationCameraComponent::USegmentationCameraComponent()
{
}


void USegmentationCameraComponent::OnRecordTick()
{
    CaptureSegmentationScene();

    ProcessSegmentationTextureAsyncRaw([this]
    {
        Dirty = true;
    });

    if (RecorderPtr)
    {
        FSegmentationCameraData CameraData;
        CameraData.Timestamp = FPlatformTime::Seconds();
        CameraData.Width = Width;
        CameraData.Height = Height;
        while(!Dirty)
        {
            FPlatformProcess::Sleep(0.01f);
        }
        CameraData.Data = SegmentationData;
        Dirty = false;
        RecorderPtr->SubmitSegmentationData(ParentActor, MoveTemp(CameraData));
    }
}

void USegmentationCameraComponent::RConfigure(
    const FSegmentationCameraConfig& Config, ARecorder* Recorder)
{
    FOV = Config.FOV;
    Width = Config.Width;
    Height = Config.Height;

    ComputeIntrinsics();
    InitializeRenderTargets();
    SetCaptureComponent();
    
    if (Config.RecordInterval > 0)
    {
        RecordInterval = Config.RecordInterval;
        SetupRecorder(Recorder);
        RecordState = Recorder->RecordState;
        Recorder->OnRecordStateChanged.AddDynamic(this,
            &USegmentationCameraComponent::SetRecordState);
        SetComponentTickEnabled(true);
        bRecorded = true;
    }
    else
    {
        SetComponentTickEnabled(false);
    }
    
    bBPConfigured = true;
}

void USegmentationCameraComponent::SetCaptureComponent() const
{
    Super::SetCaptureComponent();

    if (CaptureComponent)
    {
        CaptureComponent->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives;
        CaptureComponent->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
        CaptureComponent->bAlwaysPersistRenderingState = true;

        // Apply the segmentation post-process material if available
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
}

void USegmentationCameraComponent::InitializeRenderTargets()
{
    SegmentationRenderTarget = NewObject<UTextureRenderTarget2D>(this);
    SegmentationRenderTarget->RenderTargetFormat = ETextureRenderTargetFormat::RTF_RGBA8;
    SegmentationRenderTarget->InitCustomFormat(Width, Height,
        PF_R8G8B8A8, true);
    SegmentationRenderTarget->TargetGamma = GEngine->GetDisplayGamma();
    SegmentationRenderTarget->bGPUSharedFlag = true;
    SegmentationRenderTarget->bAutoGenerateMips = false;
    
    SegmentationRenderTarget->UpdateResource();
}

void USegmentationCameraComponent::CaptureSegmentationScene()
{
    if (!CheckComponentAndRenderTarget())
    {
        return;
    }
    
    // Check if we're on the game thread
    if (IsInGameThread())
    {
        // We're already on the game thread, proceed normally
        ExecuteCaptureOnGameThread();
    }
    else
    {
        // We're not on the game thread, so we need to dispatch to it
        AsyncTask(ENamedThreads::GameThread, [this]()
        {
            ExecuteCaptureOnGameThread();
        });
    }
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
        
        ProcessSegmentationTextureAsync([Callback](const TArray<FColor>& ColorData)
        {
            Callback(ColorData);
        });
    });
}

void USegmentationCameraComponent::ProcessSegmentationTextureAsyncRaw(TFunction<void()> OnComplete)
{
    FTextureRenderTargetResource* RenderTargetResource = 
        SegmentationRenderTarget->GameThread_GetRenderTargetResource();
    
    if (!RenderTargetResource)
    {
        UE_LOG(LogSegmentCamera, Error, TEXT("Failed to get render target resource!"));
        return;
    }

    // Ensure SegmentationData has correct size
    if (SegmentationData.Num() != Width * Height)
    {
        SegmentationData.SetNumUninitialized(Width * Height);
    }
    
    FReadSurfaceContext Context =
    {
        &SegmentationData,
        RenderTargetResource,
        FIntRect(0, 0, Width, Height),
        FReadSurfaceDataFlags(RCM_UNorm, CubeFace_MAX)
    };

    auto SharedCallback = MakeShared<TFunction<void()>>(OnComplete);
    
    ENQUEUE_RENDER_COMMAND(ReadSurfaceCommand)(
        [Context, SharedCallback](FRHICommandListImmediate& RHICmdList)
        {
            RHICmdList.ReadSurfaceData(
                Context.RenderTarget->GetRenderTargetTexture(),
                Context.Rect,
                *Context.OutData,
                Context.Flags
            );
            (*SharedCallback)();
        });
}

void USegmentationCameraComponent::ProcessSegmentationTextureAsync(
    TFunction<void(const TArray<FColor>&)> OnComplete)
{
    FTextureRenderTargetResource* RenderTargetResource = 
        SegmentationRenderTarget->GameThread_GetRenderTargetResource();
    
    if (!RenderTargetResource)
    {
        UE_LOG(LogSegmentCamera, Error, TEXT("Failed to get render target resource!"));
        return;
    }

    // Ensure SegmentationData has correct size
    if (SegmentationData.Num() != Width * Height)
    {
        SegmentationData.SetNumUninitialized(Width * Height);
    }
    
    FReadSurfaceContext Context =
    {
        &SegmentationData,
        RenderTargetResource,
        FIntRect(0, 0, Width, Height),
        FReadSurfaceDataFlags(RCM_UNorm, CubeFace_MAX)
    };

    auto SharedCallback = MakeShared<TFunction<void(const TArray<FColor>&)>>(MoveTemp(OnComplete));
    // Capture the OnComplete callback in the render command
    ENQUEUE_RENDER_COMMAND(ReadSurfaceCommand)(
        [Context, SharedCallback](FRHICommandListImmediate& RHICmdList)
        {            
            RHICmdList.ReadSurfaceData(
                Context.RenderTarget->GetRenderTargetTexture(),
                Context.Rect,
                *Context.OutData,
                Context.Flags
            );
            
            (*SharedCallback)(*Context.OutData);
        });
}


void USegmentationCameraComponent::ExecuteCaptureOnGameThread()
{
    check(IsInGameThread());
    
    CaptureComponent->CaptureScene();
}