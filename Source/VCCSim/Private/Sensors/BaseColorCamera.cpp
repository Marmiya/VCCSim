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

DEFINE_LOG_CATEGORY_STATIC(LogBaseColorCamera, Log, All);

#include "Sensors/BaseColorCamera.h"
#include "RenderingThread.h"
#include "Async/AsyncWork.h"
#include "Windows/WindowsHWrapper.h"
#include "RHI.h"

void UBaseColorCameraComponent::Configure(const FSensorConfig& Config)
{
    if (!bBPConfigured)
    {
        const auto& CamConfig = static_cast<const FBaseColorCameraConfig&>(Config);
        FOV    = CamConfig.FOV;
        Width  = CamConfig.Width;
        Height = CamConfig.Height;
    }

    RecordInterval = Config.RecordInterval;

    ComputeIntrinsics();
    InitializeRenderTargets();
    SetCaptureComponent();
}

void UBaseColorCameraComponent::SetCaptureComponent() const
{
    Super::SetCaptureComponent();

    CaptureComponent->CaptureSource = SCS_BaseColor;
    CaptureComponent->bAlwaysPersistRenderingState = true;

    auto& SF = CaptureComponent->ShowFlags;
    SF.SetMotionBlur(false);
    SF.SetDepthOfField(false);
    SF.SetLensFlares(false);
    SF.SetVignette(false);
    SF.SetGrain(false);
    SF.SetAntiAliasing(false);
}

void UBaseColorCameraComponent::InitializeRenderTargets()
{
    RenderTarget = NewObject<UTextureRenderTarget2D>(this);
    RenderTarget->InitCustomFormat(Width, Height, PF_B8G8R8A8, false);
    RenderTarget->bForceLinearGamma = false;
    RenderTarget->SRGB = true;
    RenderTarget->RenderTargetFormat = RTF_RGBA8;
    RenderTarget->bGPUSharedFlag = true;
    RenderTarget->bAutoGenerateMips = false;
    RenderTarget->UpdateResource();
}

void UBaseColorCameraComponent::CaptureBaseColorScene()
{
    if (!CheckComponentAndRenderTarget())
    {
        UE_LOG(LogBaseColorCamera, Warning, 
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

void UBaseColorCameraComponent::ProcessBaseColorTexture(TFunction<void()> OnComplete)
{
    ProcessBaseColorTextureTemplate(std::move(OnComplete));
}

void UBaseColorCameraComponent::ProcessBaseColorTextureParam(
    TFunction<void(const TArray<FColor>&)> OnComplete)
{
    ProcessBaseColorTextureTemplate(std::move(OnComplete));
}

void UBaseColorCameraComponent::AsyncGetBaseColorImageData(
    TFunction<void(const TArray<FColor>&)> Callback)
{
    AsyncTask(ENamedThreads::GameThread,
        [this, Callback = MoveTemp(Callback)]()
        {
            if (!CheckComponentAndRenderTarget())
            {
                UE_LOG(LogBaseColorCamera, Warning, 
                    TEXT("Component or RenderTarget not valid! Try to initialize them."));
                InitializeRenderTargets();
                SetCaptureComponent();
            }

            CaptureComponent->CaptureScene();
            ProcessBaseColorTextureParam(Callback);
        });
}
