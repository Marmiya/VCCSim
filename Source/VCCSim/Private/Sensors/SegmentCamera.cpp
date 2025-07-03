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

#include "Sensors/SegmentCamera.h"
#include "Simulation/Recorder.h"
#include "RenderingThread.h"
#include "Async/AsyncWork.h"
#include "Windows/WindowsHWrapper.h"

USegmentationCameraComponent::USegmentationCameraComponent()
    : FOV(90.0f)
    , Width(512)
    , Height(512)
    , TimeSinceLastCapture(0.0f)
{
    PrimaryComponentTick.bCanEverTick = true;
}

void USegmentationCameraComponent::BeginPlay()
{
    Super::BeginPlay();
    
    InitializeRenderTargets();
    SetCaptureComponent();
    
    SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
    SetCollisionResponseToAllChannels(ECR_Ignore);
    SetSimulatePhysics(true);
}

void USegmentationCameraComponent::OnComponentCreated()
{
    Super::OnComponentCreated();
    
    CaptureComponent = NewObject<USceneCaptureComponent2D>(this);
    CaptureComponent->AttachToComponent(this,
        FAttachmentTransformRules::SnapToTargetIncludingScale);
    
    SetCaptureComponent();
}

void USegmentationCameraComponent::TickComponent(float DeltaTime, ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
    
    if (bRecorded && RecordState)
    {
        TimeSinceLastCapture += DeltaTime;
        if (TimeSinceLastCapture >= RecordInterval)
        {
            TimeSinceLastCapture = 0.0f;
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
    }
}

void USegmentationCameraComponent::RConfigure(
    const FSegmentationCameraConfig& Config, ARecorder* Recorder)
{ 
    FOV = Config.FOV;
    Width = Config.Width;
    Height = Config.Height;
    
    float HorizontalFOVRad = FMath::DegreesToRadians(FOV);
    float fx = (Width / 2.0f) / FMath::Tan(HorizontalFOVRad / 2.0f);

    // Compute vertical FOV from horizontal FOV and aspect ratio.
    float verticalFOVRad = 2.0f * FMath::Atan((static_cast<float>(Height) / Width) *
        FMath::Tan(HorizontalFOVRad / 2.0f));
    float fy = (Height / 2.0f) / FMath::Tan(verticalFOVRad / 2.0f);

    float cx = Width / 2.0f;
    float cy = Height / 2.0f;

    CameraIntrinsics = FMatrix44f::Identity;
    CameraIntrinsics.M[0][0] = fx;  // focal length in x
    CameraIntrinsics.M[1][1] = fy;  // focal length in y
    CameraIntrinsics.M[0][2] = cx;  // principal point x
    CameraIntrinsics.M[1][2] = cy;  // principal point y
    
    InitializeRenderTargets();
    SetCaptureComponent();
    
    if (Config.RecordInterval > 0)
    {
        ParentActor = GetOwner();
        RecorderPtr = Recorder;
        RecordInterval = Config.RecordInterval;
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
    if (CaptureComponent)
    {
        // Basic camera settings
        CaptureComponent->ProjectionType = ECameraProjectionMode::Perspective;
        CaptureComponent->FOVAngle = FOV;

        // For segmentation, we want to capture the base color/diffuse
        CaptureComponent->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
        CaptureComponent->bCaptureEveryFrame = false;
        CaptureComponent->bCaptureOnMovement = false;
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
            UE_LOG(LogTemp, Error, TEXT("Segmentation material not set!"));
        }
    }
    else 
    {
        UE_LOG(LogTemp, Error, TEXT("Capture component not initialized!"));
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
    CaptureComponent->TextureTarget = SegmentationRenderTarget;
}

void USegmentationCameraComponent::CaptureSegmentationScene()
{
    if (CheckComponentAndRenderTarget())
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
        if (CheckComponentAndRenderTarget())
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
        UE_LOG(LogTemp, Error, TEXT("Failed to get render target resource!"));
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
        UE_LOG(LogTemp, Error, TEXT("Failed to get render target resource!"));
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

bool USegmentationCameraComponent::CheckComponentAndRenderTarget() const
{
    if (!CaptureComponent || !SegmentationRenderTarget)
    {
        UE_LOG(LogTemp, Error, TEXT("USegmentationCameraComponent::CheckComponentAndRenderTarget: "
                                    "Capture component or render target not initialized!"));
        return true;
    }
    return false;
}

void USegmentationCameraComponent::ExecuteCaptureOnGameThread()
{
    check(IsInGameThread());
    
    CaptureComponent->CaptureScene();
}