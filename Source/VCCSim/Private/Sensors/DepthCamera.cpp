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

#include "Sensors/DepthCamera.h"
#include "Simulation/Recorder.h"
#include "RenderingThread.h"
#include "Async/AsyncWork.h"
#include "Windows/WindowsHWrapper.h"
#include "RHI.h"

UDepthCameraComponent::UDepthCameraComponent()
    : FOV(90.0f)
    , MaxRange(2000.0f)
    , MinRange(.0f)
    , Width(512)
    , Height(512)
    , bOrthographic(false)
    , OrthoWidth(512.0f)
    , TimeSinceLastCapture(0.0f)
{
    PrimaryComponentTick.bCanEverTick = true;
}

void UDepthCameraComponent::RConfigure(
    const FDepthCameraConfig& Config, ARecorder* Recorder)
{  
    FOV = Config.FOV;
    MaxRange = Config.MaxRange;
    MinRange = Config.MinRange;
    Width = Config.Width;
    Height = Config.Height;
    bOrthographic = Config.bOrthographic;
    OrthoWidth = Config.OrthoWidth;
    InitializeRenderTargets();
    SetCaptureComponent();

    if (Config.RecordInterval > 0)
    {
        ParentActor = GetOwner();
        RecorderPtr = Recorder;
        RecordInterval = Config.RecordInterval;
        RecordState = Recorder->RecordState;
        // UE_LOG(LogTemp, Display, TEXT("RecordState: %d"),RecordState);
        
        Recorder->OnRecordStateChanged.AddDynamic(this,
            &UDepthCameraComponent::SetRecordState);
        SetComponentTickEnabled(true);
        bRecorded = true;
    }
    else
    {
        SetComponentTickEnabled(false);
    }
    bBPConfigured = true;
}

void UDepthCameraComponent::SetCaptureComponent() const
{
    if (CaptureComponent)
    {
        CaptureComponent->ProjectionType = bOrthographic ?
            ECameraProjectionMode::Orthographic : ECameraProjectionMode::Perspective;
        CaptureComponent->FOVAngle = FOV;
        CaptureComponent->OrthoWidth = OrthoWidth;
        CaptureComponent->CaptureSource = ESceneCaptureSource::SCS_SceneDepth;
        CaptureComponent->bCaptureEveryFrame = false;
        CaptureComponent->bCaptureOnMovement = false;
    }
    else 
    {
        UE_LOG(LogTemp, Error, TEXT("Capture component not initialized!"));
    }
}

void UDepthCameraComponent::BeginPlay()
{
    Super::BeginPlay();
    InitializeRenderTargets();
    SetCaptureComponent();
    
    SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
    SetCollisionResponseToAllChannels(ECR_Ignore);
    SetSimulatePhysics(true);
}

void UDepthCameraComponent::OnComponentCreated()
{
    Super::OnComponentCreated();
    
    // Initialize capture component
    CaptureComponent = NewObject<USceneCaptureComponent2D>(this);
    CaptureComponent->AttachToComponent(this,
        FAttachmentTransformRules::SnapToTargetIncludingScale);
    
    SetCaptureComponent();
}

void UDepthCameraComponent::TickComponent(float DeltaTime, ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    if (bRecorded && RecordState)
    {
        TimeSinceLastCapture += DeltaTime;
        if (TimeSinceLastCapture >= RecordInterval)
        {
            TimeSinceLastCapture = 0.0f;
            CaptureDepthScene();
            if (RecorderPtr)
            {
                FDepthCameraData DepthCameraData;
                DepthCameraData.Timestamp = FPlatformTime::Seconds();
                DepthCameraData.SensorIndex = CameraIndex;
                DepthCameraData.Width = Width;
                DepthCameraData.Height = Height;
                while(!Dirty)
                {
                    FPlatformProcess::Sleep(0.01f);
                }
                DepthCameraData.Data = GetDepthImage();
                Dirty = false;
                RecorderPtr->SubmitDepthData(ParentActor, MoveTemp(DepthCameraData));
            }
        }
    }
}

void UDepthCameraComponent::InitializeRenderTargets()
{
    DepthRenderTarget = NewObject<UTextureRenderTarget2D>(this);
    DepthRenderTarget->InitCustomFormat(Width, Height,
        PF_FloatRGBA, true);
    
    DepthRenderTarget->UpdateResource();
    CaptureComponent->TextureTarget = DepthRenderTarget;
    
    if (CaptureComponent==nullptr)
    {
        UE_LOG(LogTemp, Error, TEXT("Capture component not initialized!"));
    }
}

bool UDepthCameraComponent::CheckComponentAndRenderTarget() const
{
    if (!CaptureComponent || !DepthRenderTarget)
    {
        UE_LOG(LogTemp, Error, TEXT("Capture component or "
                   "render target not initialized!"));
        return true;
    }
    return false;
}

void UDepthCameraComponent::CaptureDepthScene()
{
    if (CheckComponentAndRenderTarget())
    {
        UE_LOG(LogTemp, Error, TEXT("UDepthCameraComponent: "
                                    "Capture component or render target not initialized!"));
        return;
    }
    
    CaptureComponent->CaptureScene();
    
    ProcessDepthTexture([this]()
        {
            Dirty = true;
        });
}

void UDepthCameraComponent::OnlyCaptureDepthScene()
{
    if (CheckComponentAndRenderTarget())
    {
        UE_LOG(LogTemp, Error, TEXT("UDepthCameraComponent: "
                                    "Capture component or render target not initialized!"));
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

void UDepthCameraComponent::ProcessDepthTexture(TFunction<void()> OnComplete)
{
    // Get the render target resource
    FTextureRenderTargetResource* RenderTargetResource = 
        DepthRenderTarget->GameThread_GetRenderTargetResource();
    
    if (!RenderTargetResource)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to get render target resource!"));
        return;
    }

    // Prepare depth data array
    DepthData.Empty(Width * Height);
    DepthData.SetNumUninitialized(Width * Height);

    // Define context for reading surface data
    struct FReadSurfaceContext
    {
        TArray<FFloat16Color>* OutData;
        FTextureRenderTargetResource* RenderTarget;
        FIntRect Rect;
        FReadSurfaceDataFlags Flags;
    };

    FReadSurfaceContext Context = {
        &DepthData,
        RenderTargetResource,
        FIntRect(0, 0, Width, Height),
        FReadSurfaceDataFlags(RCM_MinMax, CubeFace_MAX)
    };

    auto SharedCallback = MakeShared<TFunction<void()>>(OnComplete);
    // Submit the render thread command
    ENQUEUE_RENDER_COMMAND(ReadSurfaceCommand)(
        [Context, SharedCallback](FRHICommandListImmediate& RHICmdList)
    {
        // The render thread performs the data read
        RHICmdList.ReadSurfaceFloatData(
            Context.RenderTarget->GetRenderTargetTexture(),
            Context.Rect,
            *Context.OutData,
            ECubeFace::CubeFace_PosX,
            0,
            0
        );
        (*SharedCallback)();
    });
}

void UDepthCameraComponent::ProcessDepthTextureParam(
    TFunction<void(const TArray<FFloat16Color>&)> OnComplete)
{
    // Get the render target resource
    FTextureRenderTargetResource* RenderTargetResource = 
        DepthRenderTarget->GameThread_GetRenderTargetResource();
    
    if (!RenderTargetResource)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to get render target resource!"));
        return;
    }

    // Prepare depth data array
    DepthData.Empty(Width * Height);
    DepthData.SetNumUninitialized(Width * Height);

    // Define context for reading surface data
    struct FReadSurfaceContext
    {
        TArray<FFloat16Color>* OutData;
        FTextureRenderTargetResource* RenderTarget;
        FIntRect Rect;
        FReadSurfaceDataFlags Flags;
    };

    FReadSurfaceContext Context = {
        &DepthData,
        RenderTargetResource,
        FIntRect(0, 0, Width, Height),
        FReadSurfaceDataFlags(RCM_MinMax, CubeFace_MAX)
    };

    auto SharedCallback = MakeShared<TFunction<void(const TArray<FFloat16Color>&)>>(OnComplete);
    // Submit the render thread command
    
    
    ENQUEUE_RENDER_COMMAND(ReadSurfaceCommand)(
        [Context, SharedCallback](FRHICommandListImmediate& RHICmdList)
    {
        // The render thread performs the data read
        RHICmdList.ReadSurfaceFloatData(
            Context.RenderTarget->GetRenderTargetTexture(),
            Context.Rect,
            *Context.OutData,
            ECubeFace::CubeFace_PosX,
            0,
            0
        );
        (*SharedCallback)(*Context.OutData);
    });
}

TArray<FDCPoint> UDepthCameraComponent::GeneratePointCloud()
{
    if (DepthData.Num() == 0) 
    {
        UE_LOG(LogTemp, Error, TEXT("GeneratePointCloud: No depth data available!"));
        return {};
    }

    TArray<FDCPoint> tPointCloudData;

    // Get camera transform
    const FTransform CameraTransform = GetComponentTransform();
    UE_LOG(LogTemp, Log, TEXT("Camera Transform: %s"), *CameraTransform.ToString());
    const float AspectRatio = static_cast<float>(Width) / Height;
    
    // Get the actual FOV from the capture component
    float ActualFOV = CaptureComponent ? CaptureComponent->FOVAngle : FOV;
    UE_LOG(LogTemp, Log, TEXT("Actual FOV: %f"), ActualFOV);
    float HalfFOVRad = FMath::DegreesToRadians(ActualFOV * 0.5f);

    for (int32 Y = 0; Y < Height; ++Y)
    {
        for (int32 X = 0; X < Width; ++X)
        {
            const int32 Index = Y * Width + X;
            const float Depth = DepthData[Index].R.GetFloat();

            // Skip invalid depth values
            if (Depth < MinRange || Depth > MaxRange) continue;

            FDCPoint Point;
            FVector WorldPos;

            if (bOrthographic)
            {
                // Convert pixel coordinates to world space
                const float WorldX =
                    (static_cast<float>(X) / Width - 0.5f) * OrthoWidth;
                const float WorldY =
                    (static_cast<float>(Y) / Height - 0.5f) * (OrthoWidth / AspectRatio);

                // Create point in camera space (Forward, Right, Up)
                FVector CameraSpacePos(Depth, WorldX, -WorldY);

                // Transform to world space
                WorldPos = CameraTransform.TransformPosition(CameraSpacePos);
            }
            else
            {
                float NDC_X = (2.0f * X / (Width - 1)) - 1.0f;
                float NDC_Y = 1.0f - (2.0f * Y / (Height - 1));
                
                float TanHalfHorizontalFOV = FMath::Tan(HalfFOVRad);
                float TanHalfVerticalFOV = TanHalfHorizontalFOV / AspectRatio;
                
                float ViewX = NDC_X * TanHalfHorizontalFOV * Depth;
                float ViewY = NDC_Y * TanHalfVerticalFOV * Depth;
                
                FVector CameraSpacePos(
                    Depth,      // Forward
                    ViewX,      // Right
                    ViewY       // Up
                );
                
                WorldPos = CameraTransform.TransformPosition(CameraSpacePos);
            }

            Point.Location = WorldPos;
            tPointCloudData.Add(Point);
        }
    }

    DepthData.Empty(Width * Height);
    return tPointCloudData;
}

TArray<float> UDepthCameraComponent::GetDepthImage()
{
    TArray<float> DepthImage;
    DepthImage.Empty(Width * Height);

    // Wait if depth data hasn't been processed yet
    if (DepthData.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("GetDepthImage: No depth data available!"));
        return DepthImage;
    }
    
    for (const FFloat16Color& Color : DepthData)
    {
        DepthImage.Add(Color.R.GetFloat());
    }

    return DepthImage;
}

void UDepthCameraComponent::VisualizePointCloud()
{
    // Draw the point cloud. Debug only
    UWorld* World = GetWorld();
    if (!World)
    {
        UE_LOG(LogTemp, Error, TEXT("VisualizePointCloud: World is null!"));
        return;
    }
    int i = 0;
    for (const FDCPoint& Point : PointCloudData)
    {
        if (i % 100 == 0) // Draw every 100th point for performance
        {
            i++;
            continue;
        }
        DrawDebugPoint(World, Point.Location, 5.0f, FColor::Red,
            false, -1.0f);
    }
}

void UDepthCameraComponent::AsyncGetPointCloudData(
    TFunction<void()> Callback)
{
    AsyncTask(ENamedThreads::GameThread, [this, Callback = MoveTemp(Callback)]() {
        if (CheckComponentAndRenderTarget())
        {
            Callback();
            return;
        }
        
        CaptureComponent->CaptureScene();
        ProcessDepthTexture(Callback);
    });
}

void UDepthCameraComponent::AsyncGetDepthImageData(
    TFunction<void(const TArray<FFloat16Color>&)> Callback)
{
    AsyncTask(ENamedThreads::GameThread, [this, Callback = MoveTemp(Callback)]() {
        if (CheckComponentAndRenderTarget())
        {
            Callback({});
            return;
        }
        CaptureComponent->CaptureScene();
        ProcessDepthTextureParam(Callback);
    });
}