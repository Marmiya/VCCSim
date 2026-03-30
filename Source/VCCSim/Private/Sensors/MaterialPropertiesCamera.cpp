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

DEFINE_LOG_CATEGORY_STATIC(LogMaterialPropertiesCamera, Log, All);

#include "Sensors/MaterialPropertiesCamera.h"
#include "RenderingThread.h"
#include "Async/AsyncWork.h"
#include "Windows/WindowsHWrapper.h"
#include "RHI.h"

void UMaterialPropertiesCameraComponent::Configure(const FSensorConfig& Config)
{
    if (!bBPConfigured)
    {
        const auto& CamConfig = static_cast<const FMaterialPropertiesCameraConfig&>(Config);
        FOV    = CamConfig.FOV;
        Width  = CamConfig.Width;
        Height = CamConfig.Height;
    }

    RecordInterval = Config.RecordInterval;

    ComputeIntrinsics();
    InitializeRenderTargets();
    SetCaptureComponent();
}

void UMaterialPropertiesCameraComponent::SetCaptureComponent() const
{
    Super::SetCaptureComponent();

    CaptureComponent->CaptureSource = SCS_FinalColorHDR;
    CaptureComponent->bAlwaysPersistRenderingState = true;

    UMaterialInterface* MatPropsMaterial = LoadObject<UMaterialInterface>(
        nullptr, TEXT("/VCCSim/Materials/M_MatPropsCapture"));
    if (MatPropsMaterial)
    {
        CaptureComponent->PostProcessSettings.WeightedBlendables.Array.Empty();
        CaptureComponent->PostProcessSettings.WeightedBlendables.Array.Add(
            FWeightedBlendable(1.0f, MatPropsMaterial));
    }
    else
    {
        UE_LOG(LogMaterialPropertiesCamera, Error,
            TEXT("M_MatPropsCapture not found at /VCCSim/Materials/M_MatPropsCapture"));
    }

    auto& SF = CaptureComponent->ShowFlags;
    SF.SetMotionBlur(false);
    SF.SetDepthOfField(false);
    SF.SetLensFlares(false);
    SF.SetVignette(false);
    SF.SetGrain(false);
    SF.SetAntiAliasing(false);
}

void UMaterialPropertiesCameraComponent::InitializeRenderTargets()
{
    RenderTarget = NewObject<UTextureRenderTarget2D>(this);
    RenderTarget->InitCustomFormat(Width, Height, PF_B8G8R8A8, false);
    RenderTarget->bForceLinearGamma = true;
    RenderTarget->SRGB = false;
    RenderTarget->RenderTargetFormat = RTF_RGBA8;
    RenderTarget->bGPUSharedFlag = true;
    RenderTarget->bAutoGenerateMips = false;
    RenderTarget->UpdateResource();
}

void UMaterialPropertiesCameraComponent::CaptureMaterialPropertiesScene()
{
    if (!CheckComponentAndRenderTarget())
    {
        UE_LOG(LogMaterialPropertiesCamera, Warning, 
            TEXT("Component or RenderTarget not valid! Try to initialize them."));
        InitializeRenderTargets();
        SetCaptureComponent();
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

void UMaterialPropertiesCameraComponent::ProcessMaterialPropertiesTexture(TFunction<void()> OnComplete)
{
    ProcessMaterialPropertiesTextureTemplate(std::move(OnComplete));
}

void UMaterialPropertiesCameraComponent::ProcessMaterialPropertiesTextureParam(
    TFunction<void(const TArray<FColor>&)> OnComplete)
{
    ProcessMaterialPropertiesTextureTemplate(std::move(OnComplete));
}

void UMaterialPropertiesCameraComponent::AsyncGetMaterialPropertiesImageData(
    TFunction<void(const TArray<FColor>&)> Callback)
{
    AsyncTask(ENamedThreads::GameThread,
        [this, Callback = MoveTemp(Callback)]()
        {
            if (!CheckComponentAndRenderTarget())
            {
                UE_LOG(LogMaterialPropertiesCamera, Warning, 
                    TEXT("Component or RenderTarget not valid! Try to initialize them."));
                InitializeRenderTargets();
                SetCaptureComponent();
            }

            CaptureComponent->CaptureScene();
            ProcessMaterialPropertiesTextureParam(Callback);
        });
}
