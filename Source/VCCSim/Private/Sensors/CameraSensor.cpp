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

#include "Sensors/CameraSensor.h"
#include "Simulation/Recorder.h"
#include "RenderingThread.h"
#include "Async/AsyncWork.h"
#include "Windows/WindowsHWrapper.h"

URGBCameraComponent::URGBCameraComponent()
    : FOV(90.0f)
    , Width(512)
    , Height(512)
    , bOrthographic(false)
    , OrthoWidth(512.0f)
    , TimeSinceLastCapture(0.0f)
{
    PrimaryComponentTick.bCanEverTick = true;
}

void URGBCameraComponent::BeginPlay()
{
    Super::BeginPlay();
    
    InitializeRenderTargets();
    SetCaptureComponent();
    
    SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
    SetCollisionResponseToAllChannels(ECR_Ignore);
    SetSimulatePhysics(true);
}

void URGBCameraComponent::OnComponentCreated()
{
    Super::OnComponentCreated();
    
    CaptureComponent = NewObject<USceneCaptureComponent2D>(this);
    CaptureComponent->AttachToComponent(this,
        FAttachmentTransformRules::SnapToTargetIncludingScale);
    
    SetCaptureComponent();
}

void URGBCameraComponent::TickComponent(float DeltaTime, ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
    
    if (bRecorded && RecordState)
    {
        TimeSinceLastCapture += DeltaTime;
        if (TimeSinceLastCapture >= RecordInterval)
        {
            TimeSinceLastCapture = 0.0f;
            CaptureRGBScene();
            
            ProcessRGBTextureAsyncRaw([this]
            {
                Dirty = true;
            });
            
            if (RecorderPtr)
            {
                FRGBCameraData CameraData;
                CameraData.Timestamp = FPlatformTime::Seconds();
                CameraData.SensorIndex = CameraIndex;
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
    }
}

void URGBCameraComponent::ComputeIntrinsics()
{
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
        ParentActor = GetOwner();
        RecorderPtr = Recorder;
        RecordInterval = Config.RecordInterval;
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

void URGBCameraComponent::SetCaptureComponent() const
{
    if (CaptureComponent)
    {
        // Basic camera settings
        CaptureComponent->ProjectionType = bOrthographic ?
            ECameraProjectionMode::Orthographic : ECameraProjectionMode::Perspective;
        CaptureComponent->FOVAngle = FOV;
        CaptureComponent->OrthoWidth = OrthoWidth;

        // Change the capture source to HDR for better quality
        CaptureComponent->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
        CaptureComponent->bCaptureEveryFrame = false;
        CaptureComponent->bCaptureOnMovement = true;
        CaptureComponent->bAlwaysPersistRenderingState = true;
        CaptureComponent->PrimaryComponentTick.bCanEverTick = true;
        
        FEngineShowFlags& ShowFlags = CaptureComponent->ShowFlags;
        // ShowFlags.SetTemporalAA(true);
        ShowFlags.EnableAdvancedFeatures();
        ShowFlags.SetPostProcessing(true);
        ShowFlags.SetTonemapper(true);
        ShowFlags.SetBloom(true);

        ShowFlags.SetLumenGlobalIllumination(true);
        ShowFlags.SetLumenReflections(true);
    }
    else 
    {
        UE_LOG(LogTemp, Error, TEXT("Capture component not initialized!"));
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
    CaptureComponent->TextureTarget = RGBRenderTarget;
}

void URGBCameraComponent::CaptureRGBScene()
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
        UE_LOG(LogTemp, Error, TEXT("Failed to get render target resource!"));
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
        UE_LOG(LogTemp, Error, TEXT("Failed to get render target resource!"));
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

bool URGBCameraComponent::CheckComponentAndRenderTarget() const
{
    if (!CaptureComponent || !RGBRenderTarget)
    {
        UE_LOG(LogTemp, Error, TEXT("URGBCameraComponent::CheckComponentAndRenderTarget: "
                                    "Capture component or render target not initialized!"));
        return true;
    }
    return false;
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
