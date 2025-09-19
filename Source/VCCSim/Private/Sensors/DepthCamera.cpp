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

DEFINE_LOG_CATEGORY_STATIC(LogDepthCamera, Log, All);

#include "Sensors/DepthCamera.h"
#include "Simulation/Recorder.h"
#include "RenderingThread.h"
#include "Async/AsyncWork.h"
#include "Windows/WindowsHWrapper.h"
#include "RHI.h"

UDepthCameraComponent::UDepthCameraComponent()
{
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

    ComputeIntrinsics();
    InitializeRenderTargets();
    SetCaptureComponent();

    if (Config.RecordInterval > 0)
    {
        RecordInterval = Config.RecordInterval;
        SetupRecorder(Recorder);
        RecordState = Recorder->RecordState;

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
    Super::SetCaptureComponent();

    if (CaptureComponent)
    {
        CaptureComponent->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives;
        CaptureComponent->CaptureSource = ESceneCaptureSource::SCS_SceneDepth;
    }
}

void UDepthCameraComponent::OnRecordTick()
{
    double CaptureStartTime = FPlatformTime::Seconds();
    CaptureDepthScene();
    if (RecorderPtr)
    {
        FDepthCameraData DepthCameraData;
        DepthCameraData.Timestamp = FPlatformTime::Seconds();
        DepthCameraData.SensorIndex = GetSensorIndex();
        DepthCameraData.Width = Width;
        DepthCameraData.Height = Height;

        double WaitStartTime = FPlatformTime::Seconds();
        while(!Dirty)
        {
            FPlatformProcess::Sleep(0.01f);
        }
        double WaitEndTime = FPlatformTime::Seconds();

        DepthCameraData.Data = GetDepthImage();
        Dirty = false;
        RecorderPtr->SubmitDepthData(ParentActor, MoveTemp(DepthCameraData));

        double CaptureEndTime = FPlatformTime::Seconds();
        double TotalCaptureTime = CaptureEndTime - CaptureStartTime;
        double WaitTime = WaitEndTime - WaitStartTime;

        UE_LOG(LogDepthCamera, Log, TEXT("Depth capture - RecordInterval: %.6f, Total time: %.6f, Wait time: %.6f, Actual FPS: %.2f"),
            RecordInterval, TotalCaptureTime, WaitTime, 1.0 / TotalCaptureTime);
    }
}

void UDepthCameraComponent::InitializeRenderTargets()
{
    DepthRenderTarget = NewObject<UTextureRenderTarget2D>(this);
    DepthRenderTarget->InitCustomFormat(Width, Height,
        PF_FloatRGBA, true);
    
    DepthRenderTarget->UpdateResource();
    
    if (CaptureComponent==nullptr)
    {
        UE_LOG(LogDepthCamera, Error, TEXT("Capture component not initialized!"));
    }
}


void UDepthCameraComponent::CaptureDepthScene()
{
    if (!CheckComponentAndRenderTarget())
    {
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

void UDepthCameraComponent::ProcessDepthTexture(TFunction<void()> OnComplete)
{
    // Get the render target resource
    FTextureRenderTargetResource* RenderTargetResource = 
        DepthRenderTarget->GameThread_GetRenderTargetResource();
    
    if (!RenderTargetResource)
    {
        UE_LOG(LogDepthCamera, Error, TEXT("Failed to get render target resource!"));
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
        UE_LOG(LogDepthCamera, Error, TEXT("Failed to get render target resource!"));
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
        UE_LOG(LogDepthCamera, Error, TEXT("GeneratePointCloud: No depth data available!"));
        return {};
    }

    TArray<FDCPoint> tPointCloudData;

    // Get camera transform
    const FTransform CameraTransform = GetComponentTransform();
    UE_LOG(LogDepthCamera, Log, TEXT("Camera Transform: %s"), *CameraTransform.ToString());
    const float AspectRatio = static_cast<float>(Width) / Height;
    
    // Get the actual FOV from the capture component
    float ActualFOV = CaptureComponent ? CaptureComponent->FOVAngle : FOV;
    UE_LOG(LogDepthCamera, Log, TEXT("Actual FOV: %f"), ActualFOV);
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
        UE_LOG(LogDepthCamera, Warning, TEXT("GetDepthImage: No depth data available!"));
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
        UE_LOG(LogDepthCamera, Error, TEXT("VisualizePointCloud: World is null!"));
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