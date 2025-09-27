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
#include "Async/AsyncWork.h"
#include "Windows/WindowsHWrapper.h"
#include "Utils/InsMeshHolder.h"
#include "Components/InstancedStaticMeshComponent.h"

URGBDCameraComponent::URGBDCameraComponent()
{
}

void URGBDCameraComponent::Configure(const FSensorConfig& Config)
{
    if (!bBPConfigured)
    {
        const auto RGBDepthConfig = static_cast<const FRGBDCameraConfig&>(Config);
        FOV = RGBDepthConfig.FOV;
        Width = RGBDepthConfig.Width;
        Height = RGBDepthConfig.Height;
        MaxRange = RGBDepthConfig.MaxRange;
        MinRange = RGBDepthConfig.MinRange;
        bSaveRGB = RGBDepthConfig.bSaveRGB;
        bSaveDepth = RGBDepthConfig.bSaveDepth;
    }

    ComputeIntrinsics();
    InitializeRenderTargets();
    SetCaptureComponent();
}

void URGBDCameraComponent::SetIgnoreLidar(UInsMeshHolder* MeshHolder)
{
    CaptureComponent->HideComponent(MeshHolder->GetInstancedMeshComponent());
    CaptureComponent->HideComponent(MeshHolder->GetInstancedMeshComponentColor());
}

void URGBDCameraComponent::SetCaptureComponent() const
{
    Super::SetCaptureComponent();

    CaptureComponent->CaptureSource = SCS_FinalColorLDR;
    CaptureComponent->bAlwaysPersistRenderingState = true;
    CaptureComponent->PrimaryComponentTick.bCanEverTick = true;
    
    FEngineShowFlags& ShowFlags = CaptureComponent->ShowFlags;
    ShowFlags.EnableAdvancedFeatures();
    ShowFlags.SetPostProcessing(true);
    ShowFlags.SetTonemapper(true);
    ShowFlags.SetBloom(true);
    ShowFlags.SetLumenGlobalIllumination(true);
    ShowFlags.SetLumenReflections(true);
    ShowFlags.SetAntiAliasing(true);

    if (RGBDMaterial)
    {
        CaptureComponent->PostProcessSettings.WeightedBlendables.Array.Empty();
        FWeightedBlendable WeightedBlendable;
        WeightedBlendable.Object = RGBDMaterial;
        WeightedBlendable.Weight = 1.f;
        CaptureComponent->PostProcessSettings.WeightedBlendables.Array.Add(WeightedBlendable);
    }
    else
    {
        UE_LOG(LogCameraSensor, Error, TEXT("RGBDMaterial material not set!"));
    }
}

void URGBDCameraComponent::InitializeRenderTargets()
{
    RenderTarget = NewObject<UTextureRenderTarget2D>(this);

    RenderTarget->TargetGamma = GEngine->GetDisplayGamma();
    RenderTarget->InitCustomFormat(Width, Height,
        PF_A32B32G32R32F, true);
    RenderTarget->RenderTargetFormat = RTF_RGBA32f;
    RenderTarget->bGPUSharedFlag = true;
    RenderTarget->bAutoGenerateMips = false;

    RenderTarget->UpdateResource();
}

void URGBDCameraComponent::CaptureRGBDScene()
{
    if (!CheckComponentAndRenderTarget())
    {
        UE_LOG(LogCameraSensor, Error, TEXT("Component or RenderTarget not valid!"));
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

void URGBDCameraComponent::AsyncGetRGBDImageData(
    TFunction<void(const TArray<FLinearColor>&)> Callback)
{
    if (!CheckComponentAndRenderTarget())
    {
        UE_LOG(LogCameraSensor, Error, TEXT("Component or RenderTarget not valid!"));
        return;
    }

    CaptureRGBDScene();

    ProcessRGBDTextureParam([Callback](const TArray<FLinearColor>& CombinedData)
    {
        Callback(CombinedData);
    });
}


void URGBDCameraComponent::AsyncGetPointCloudData(TFunction<void()> Callback)
{
    if (!CheckComponentAndRenderTarget())
    {
        UE_LOG(LogCameraSensor, Error, TEXT("Component or RenderTarget not valid!"));
        return;
    }

    CaptureComponent->CaptureScene();

    ProcessRGBDTexture([this, Callback]()
    {
        PointCloudData = GeneratePointCloud();
        Callback();
    });
}

void URGBDCameraComponent::ProcessRGBDTexture(TFunction<void()> OnComplete)
{
    ProcessRGBDepthTextureTemplate(std::move(OnComplete));
}

void URGBDCameraComponent::ProcessRGBDTextureParam(
TFunction<void(const TArray<FLinearColor>&)> OnComplete)
{
    ProcessRGBDepthTextureTemplate(std::move(OnComplete));
}

TArray<FDCPoint> URGBDCameraComponent::GeneratePointCloud()
{
    TArray<FDCPoint> Points;
    if (CombinedData.Num() != Width * Height)
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
            const float Depth = CombinedData[Index].A * MaxRange;

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

void URGBDCameraComponent::VisualizePointCloud()
{
    PointCloudData = GeneratePointCloud();
}
