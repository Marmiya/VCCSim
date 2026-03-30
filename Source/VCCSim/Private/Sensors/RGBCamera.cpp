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

#include "Sensors/RGBCamera.h"
#include "Async/AsyncWork.h"
#include "Windows/WindowsHWrapper.h"
#include "Utils/InsMeshHolder.h"
#include "Components/InstancedStaticMeshComponent.h"

void URGBCameraComponent::Configure(const FSensorConfig& Config)
{
    if (!bBPConfigured)
    {
        const auto RGBDepthConfig = static_cast<const FRGBCameraConfig&>(Config);
        FOV = RGBDepthConfig.FOV;
        Width = RGBDepthConfig.Width;
        Height = RGBDepthConfig.Height;
    }

    RecordInterval = Config.RecordInterval;

    ComputeIntrinsics();
    InitializeRenderTargets();
    SetCaptureComponent();
}

void URGBCameraComponent::SetIgnoreLidar(UInsMeshHolder* MeshHolder)
{
    CaptureComponent->HideComponent(MeshHolder->GetInstancedMeshComponent());
    CaptureComponent->HideComponent(MeshHolder->GetInstancedMeshComponentColor());
}

void URGBCameraComponent::SetCaptureComponent() const
{
    Super::SetCaptureComponent();

    CaptureComponent->CaptureSource = SCS_FinalColorHDR;
    CaptureComponent->bAlwaysPersistRenderingState = true;
    CaptureComponent->PrimaryComponentTick.bCanEverTick = true;
    CaptureComponent->bRenderInMainRenderer = true;
    
    auto& SF = CaptureComponent->ShowFlags;
    SF.SetMotionBlur(false);
    SF.SetDepthOfField(false);
    SF.SetLensFlares(false);
    SF.SetVignette(false);
    SF.SetGrain(false);
}

void URGBCameraComponent::InitializeRenderTargets()
{
    RenderTarget = NewObject<UTextureRenderTarget2D>(this);

    RenderTarget->TargetGamma = GEngine->GetDisplayGamma();
    RenderTarget->InitCustomFormat(Width, Height,
        PF_B8G8R8A8, false);
    RenderTarget->bForceLinearGamma = false;
    RenderTarget->SRGB = true;
    RenderTarget->RenderTargetFormat = RTF_RGBA8;
    RenderTarget->bGPUSharedFlag = true;
    RenderTarget->bAutoGenerateMips = false;

    RenderTarget->UpdateResource();
}

void URGBCameraComponent::CaptureRGBScene()
{
    
    if (!RenderTarget)
    {
        InitializeRenderTargets();
        SetCaptureComponent();
    }

    
    if (!CheckComponentAndRenderTarget())
    {
        UE_LOG(LogCameraSensor, Error, TEXT("Component or RenderTarget not valid!"));
        return;
    }

    if (IsInGameThread())
    {
        CaptureComponent->CaptureSceneDeferred();
    }
    else
    {
        AsyncTask(ENamedThreads::GameThread, [this]()
        {
            CaptureComponent->CaptureSceneDeferred();
        });
    }
}

void URGBCameraComponent::AsyncGetRGBImageData(
    TFunction<void(const TArray<FColor>&)> Callback)
{
    if (!RenderTarget)
    {
        InitializeRenderTargets();
        SetCaptureComponent();
    }
    
    if (!CheckComponentAndRenderTarget())
    {
        UE_LOG(LogCameraSensor, Error, TEXT("Component or RenderTarget not valid!"));
        return;
    }

    CaptureRGBScene();

    ProcessRGBTextureParam([Callback](const TArray<FColor>& CombinedData)
    {
        Callback(CombinedData);
    });
}


void URGBCameraComponent::ProcessRGBTexture(TFunction<void()> OnComplete)
{
    ProcessRGBTextureTemplate(std::move(OnComplete));
}

void URGBCameraComponent::ProcessRGBTextureParam(
TFunction<void(const TArray<FColor>&)> OnComplete)
{
    ProcessRGBTextureTemplate(std::move(OnComplete));
}