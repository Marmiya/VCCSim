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

DEFINE_LOG_CATEGORY_STATIC(LogNormalCamera, Log, All);

#include "Sensors/NormalCamera.h"
#include "RenderingThread.h"
#include "Async/AsyncWork.h"
#include "Windows/WindowsHWrapper.h"
#include "RHI.h"

UNormalCameraComponent::UNormalCameraComponent()
{
}

void UNormalCameraComponent::Configure(const FSensorConfig& Config)
{
    if (!bBPConfigured)
    {
        const auto NormalConfig = static_cast<const FNormalCameraConfig&>(Config);
        FOV = NormalConfig.FOV;
        Width = NormalConfig.Width;
        Height = NormalConfig.Height;
    }

    RecordInterval = Config.RecordInterval;

    ComputeIntrinsics();
    InitializeRenderTargets();
    SetCaptureComponent();
}

void UNormalCameraComponent::SetCaptureComponent() const
{
    Super::SetCaptureComponent();
    
    CaptureComponent->CaptureSource = SCS_Normal;
    CaptureComponent->bAlwaysPersistRenderingState = true;
    
    CaptureComponent->ShowFlags.DisableFeaturesForUnlit();
    CaptureComponent->ShowFlags.SetAntiAliasing(false);
    CaptureComponent->ShowFlags.SetMotionBlur(false);
}

void UNormalCameraComponent::InitializeRenderTargets()
{
    RenderTarget = NewObject<UTextureRenderTarget2D>(this);
    RenderTarget->InitCustomFormat(Width, Height,
        PF_FloatRGBA, true);
    RenderTarget->RenderTargetFormat = RTF_RGBA16f;
    RenderTarget->UpdateResource();
}

void UNormalCameraComponent::CaptureNormalScene()
{
    if (!RenderTarget)
    {
        InitializeRenderTargets();
        SetCaptureComponent();
    }

    if (!CheckComponentAndRenderTarget())
    {
        UE_LOG(LogNormalCamera, Error, TEXT("Component or RenderTarget not valid!"));
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

void UNormalCameraComponent::CaptureNormalSceneAndProcess()
{
    CaptureNormalScene();
    ProcessNormalTexture([]()
    {
    });
}

void UNormalCameraComponent::ProcessNormalTexture(TFunction<void()> OnComplete)
{
    ProcessNormalTextureTemplate(std::move(OnComplete));    
}

void UNormalCameraComponent::ProcessNormalTextureParam(
    TFunction<void(const TArray<FFloat16Color>&)> OnComplete)
{
    ProcessNormalTextureTemplate(std::move(OnComplete));
}

void UNormalCameraComponent::AsyncGetNormalImageData(
    TFunction<void(const TArray<FFloat16Color>&)> Callback)
{
    AsyncTask(ENamedThreads::GameThread,
        [this, Callback = MoveTemp(Callback)]() {
        if (!RenderTarget)
        {
            InitializeRenderTargets();
            SetCaptureComponent();
        }

        if (!CheckComponentAndRenderTarget())
        {
            UE_LOG(LogNormalCamera, Error, TEXT("Component or RenderTarget not valid!"));
            Callback({});
            return;
        }
        
        CaptureComponent->CaptureScene();
        ProcessNormalTextureParam(Callback);
    });
}