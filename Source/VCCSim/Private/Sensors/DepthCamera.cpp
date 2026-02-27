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
#include "RenderingThread.h"
#include "Async/AsyncWork.h"
#include "Windows/WindowsHWrapper.h"
#include "RHI.h"


void UDepthCameraComponent::Configure(const FSensorConfig& Config)
{
    if (!bBPConfigured)
    {
        const auto DepthConfig = static_cast<const FDepthCameraConfig&>(Config);
        FOV = DepthConfig.FOV;
        Width = DepthConfig.Width;
        Height = DepthConfig.Height;
        MaxRange = DepthConfig.MaxRange;
        MinRange = DepthConfig.MinRange;
    }

    RecordInterval = Config.RecordInterval;

    ComputeIntrinsics();
    InitializeRenderTargets();
    SetCaptureComponent();
}

void UDepthCameraComponent::SetCaptureComponent() const
{
    Super::SetCaptureComponent();
    
    CaptureComponent->CaptureSource = SCS_SceneDepth;
    
    CaptureComponent->ShowFlags.DisableFeaturesForUnlit();
    CaptureComponent->ShowFlags.SetAntiAliasing(false);
    CaptureComponent->ShowFlags.SetMotionBlur(false);
}

void UDepthCameraComponent::InitializeRenderTargets()
{
    RenderTarget = NewObject<UTextureRenderTarget2D>(this);
    RenderTarget->InitCustomFormat(Width, Height,
        PF_FloatRGBA, true);
    RenderTarget->bAutoGenerateMips = false;
    RenderTarget->UpdateResource();
}

void UDepthCameraComponent::CaptureDepthScene()
{
    if (!CheckComponentAndRenderTarget())
    {
        UE_LOG(LogDepthCamera, Error, TEXT("Component or RenderTarget not valid!"));
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

void UDepthCameraComponent::CaptureDepthSceneDeferred()
{
    if (!CheckComponentAndRenderTarget())
    {
        UE_LOG(LogDepthCamera, Error, TEXT("Component or RenderTarget not valid!"));
        return;
    }

    LastCaptureTimestamp = FPlatformTime::Seconds();
    CaptureComponent->CaptureSceneDeferred();
}

void UDepthCameraComponent::CaptureDepthSceneAndProcess()
{
    CaptureDepthScene();
    ProcessDepthTexture([]()
    {
    });
}

void UDepthCameraComponent::ProcessDepthTexture(TFunction<void()> OnComplete)
{
    ProcessDepthTextureTemplate(std::move(OnComplete));    
}

void UDepthCameraComponent::ProcessDepthTextureParam(
    TFunction<void(const TArray<FFloat16Color>&)> OnComplete)
{
    ProcessDepthTextureTemplate(std::move(OnComplete));
}

void UDepthCameraComponent::AsyncGetDepthImageData(
    TFunction<void(const TArray<FFloat16Color>&)> Callback)
{
    AsyncTask(ENamedThreads::GameThread,
        [this, Callback = MoveTemp(Callback)]() {
        if (!CheckComponentAndRenderTarget())
        {
            UE_LOG(LogDepthCamera, Error, TEXT("Component or RenderTarget not valid!"));
            Callback({});
            return;
        }
        
        CaptureComponent->CaptureScene();
        ProcessDepthTextureParam(Callback);
    });
}

void UDepthCameraComponent::AsyncGetPointCloudData(TFunction<void()> Callback)
{
    if (!CheckComponentAndRenderTarget())
    {
        UE_LOG(LogDepthCamera, Error, TEXT("Component or RenderTarget not valid!"));
        return;
    }

    CaptureDepthScene();

    ProcessDepthTexture([this, Callback]()
    {
        PointCloudData = GeneratePointCloud();
        Callback();
    });
}

TArray<FDCPoint> UDepthCameraComponent::GeneratePointCloud()
{
    TArray<FDCPoint> Points;
    if (DepthData.Num() != Width * Height)
    {
        return Points;
    }

    Points.Reserve(Width * Height);

    const float FocalLengthX = (Width * 0.5f) / FMath::Tan(FMath::DegreesToRadians(FOV * 0.5f));
    const float FocalLengthY = (Height * 0.5f) / FMath::Tan(FMath::DegreesToRadians(FOV * 0.5f));
    const float CenterX = Width * 0.5f;
    const float CenterY = Height * 0.5f;

    for (int32 Y = 0; Y < Height; ++Y)
    {
        for (int32 X = 0; X < Width; ++X)
        {
            const int32 Index = Y * Width + X;
            const float Depth = DepthData[Index].R;

            if (Depth > MinRange && Depth < MaxRange)
            {
                FDCPoint Point;
                Point.Location.X = ((X - CenterX) / FocalLengthX) * Depth;
                Point.Location.Y = ((Y - CenterY) / FocalLengthY) * Depth;
                Point.Location.Z = Depth;
                Points.Add(Point);
            }
        }
    }

    return Points;
}

void UDepthCameraComponent::VisualizePointCloud()
{
    PointCloudData = GeneratePointCloud();
}
