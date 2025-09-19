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
#include "Simulation/Recorder.h"
#include "RenderingThread.h"
#include "Async/AsyncWork.h"
#include "Windows/WindowsHWrapper.h"
#include "Utils/InsMeshHolder.h"
#include "Components/InstancedStaticMeshComponent.h"

URGBCameraComponent::URGBCameraComponent()
{
}

void URGBCameraComponent::OnRecordTick()
{
    UE_LOG(LogCameraSensor, Log, TEXT("Recording camera %s"), *CameraName);
    CaptureRGBScene();

    ProcessRGBTextureAsyncRaw([this]
    {
        Dirty = true;
    });

    if (RecorderPtr)
    {
        FRGBCameraData CameraData;
        CameraData.Timestamp = FPlatformTime::Seconds();
        CameraData.SensorIndex = GetSensorIndex();
        CameraData.Width = Width;
        CameraData.Height = Height;
        while(!Dirty)
        {
            FPlatformProcess::Sleep(0.01f);
        }
        CameraData.Data = RGBData;
        Dirty = false;
        RecorderPtr->SubmitRGBData(ParentActor, MoveTemp(CameraData));
    }
}


void URGBCameraComponent::RConfigure(
    const FRGBCameraConfig& Config, ARecorder* Recorder)
{
    FOV = Config.FOV;
    Width = Config.Width;
    Height = Config.Height;
    bOrthographic = Config.bOrthographic;
    OrthoWidth = Config.OrthoWidth;

    ComputeIntrinsics();

    InitializeRenderTargets();
    SetCaptureComponent();

    if (Config.RecordInterval > 0)
    {
        RecordInterval = Config.RecordInterval;
        SetupRecorder(Recorder);
        RecordState = Recorder->RecordState;
        Recorder->OnRecordStateChanged.AddDynamic(this,
            &URGBCameraComponent::SetRecordState);
        SetComponentTickEnabled(true);
        bRecorded = true;
    }
    else
    {
        SetComponentTickEnabled(false);
    }
    
    bBPConfigured = true;
}

void URGBCameraComponent::SetIgnoreLidar(
    UInsMeshHolder* MeshHolder)
{
    if (CaptureComponent)
    {
        CaptureComponent->PrimitiveRenderMode =
            ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives;
        CaptureComponent->HideComponent(MeshHolder->GetInstancedMeshComponent());
        CaptureComponent->HideComponent(MeshHolder->GetInstancedMeshComponentColor());
    }
    else
    {
        UE_LOG(LogCameraSensor, Error, TEXT("Capture component not initialized!"));
    }
}

void URGBCameraComponent::SetCaptureComponent() const
{
    Super::SetCaptureComponent();

    if (CaptureComponent)
    {
        CaptureComponent->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives;
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

void URGBCameraComponent::AsyncGetRGBImageData(
    TFunction<void(const TArray<FColor>&)> Callback)
{
    AsyncTask(ENamedThreads::GameThread, [this, Callback = MoveTemp(Callback)]()
    {
        if (CheckComponentAndRenderTarget())
        {
            return;
        }
        
        CaptureComponent->CaptureScene();
        
        ProcessRGBTextureAsync([Callback](const TArray<FColor>& ColorData)
        {
            Callback(ColorData);
        });
    });
}

void URGBCameraComponent::ProcessRGBTextureAsyncRaw(TFunction<void()> OnComplete)
{
    FTextureRenderTargetResource* RenderTargetResource = 
        RGBRenderTarget->GameThread_GetRenderTargetResource();
    
    if (!RenderTargetResource)
    {
        UE_LOG(LogCameraSensor, Error, TEXT("Failed to get render target resource!"));
        return;
    }

    // Ensure RGBData has correct size
    if (RGBData.Num() != Width * Height)
    {
        RGBData.SetNumUninitialized(Width * Height);
    }
    
    FReadSurfaceContext Context =
    {
        &RGBData,
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

void URGBCameraComponent::ProcessRGBTextureAsync(
    TFunction<void(const TArray<FColor>&)> OnComplete)
{
    FTextureRenderTargetResource* RenderTargetResource = 
        RGBRenderTarget->GameThread_GetRenderTargetResource();
    
    if (!RenderTargetResource)
    {
        UE_LOG(LogCameraSensor, Error, TEXT("Failed to get render target resource!"));
        return;
    }

    // Ensure RGBData has correct size
    if (RGBData.Num() != Width * Height)
    {
        RGBData.SetNumUninitialized(Width * Height);
    }
    
    FReadSurfaceContext Context =
    {
        &RGBData,
        RenderTargetResource,
        FIntRect(0, 0, Width, Height),
        FReadSurfaceDataFlags(RCM_UNorm, CubeFace_MAX)
    };

    auto SharedCallback = MakeShared<
        TFunction<void(const TArray<FColor>&)>>(MoveTemp(OnComplete));
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


void URGBCameraComponent::ExecuteCaptureOnGameThread()
{
    check(IsInGameThread());
    
    CaptureComponent->CaptureScene();
    if (OnKeyPointCaptured.IsBound())
    {
        OnKeyPointCaptured.Execute(this->GetComponentTransform(), CameraName);
    }
}
