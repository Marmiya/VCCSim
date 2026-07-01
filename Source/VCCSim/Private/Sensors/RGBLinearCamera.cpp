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

DEFINE_LOG_CATEGORY_STATIC(LogRGBLinearCamera, Log, All);

#include "Sensors/RGBLinearCamera.h"
#include "RenderingThread.h"
#include "Async/AsyncWork.h"
#include "Windows/WindowsHWrapper.h"
#include "RHI.h"

URGBLinearCameraComponent::URGBLinearCameraComponent()
{
}

void URGBLinearCameraComponent::Configure(const FSensorConfig& Config)
{
    if (!bBPConfigured)
    {
        const auto RGBLinearConfig = static_cast<const FRGBLinearCameraConfig&>(Config);
        FOV = RGBLinearConfig.FOV;
        Width = RGBLinearConfig.Width;
        Height = RGBLinearConfig.Height;
    }

    RecordInterval = Config.RecordInterval;

    ComputeIntrinsics();
    InitializeRenderTargets();
    SetCaptureComponent();
}

void URGBLinearCameraComponent::SetCaptureComponent() const
{
    Super::SetCaptureComponent();

    CaptureComponent->CaptureSource = SCS_SceneColorHDR;
    CaptureComponent->bAlwaysPersistRenderingState = true;

    CaptureComponent->ShowFlags.SetAntiAliasing(false);
    CaptureComponent->ShowFlags.SetMotionBlur(false);

    FPostProcessSettings& PP = CaptureComponent->PostProcessSettings;
    PP.bOverride_AutoExposureMethod = true;
    PP.AutoExposureMethod = AEM_Manual;
    PP.bOverride_AutoExposureBias = true;
    PP.AutoExposureBias = 0.0f;
}

void URGBLinearCameraComponent::InitializeRenderTargets()
{
    RenderTarget = NewObject<UTextureRenderTarget2D>(this);
    RenderTarget->InitCustomFormat(Width, Height, PF_FloatRGBA, true);
    RenderTarget->RenderTargetFormat = RTF_RGBA16f;
    RenderTarget->bForceLinearGamma = true;
    RenderTarget->SRGB = false;
    RenderTarget->UpdateResource();
}

void URGBLinearCameraComponent::AsyncGetRGBLinearImageData(
    TFunction<void(const TArray<FFloat16Color>&)> Callback)
{
    AsyncTask(ENamedThreads::GameThread,
        [this, Callback = MoveTemp(Callback)]() mutable {
            if (!CheckComponentAndRenderTarget())
            {
                UE_LOG(LogRGBLinearCamera, Warning,
                    TEXT("Component or RenderTarget not valid! Try to initialize them."));
                InitializeRenderTargets();
                SetCaptureComponent();
            }

        CaptureComponent->CaptureScene();

        const int32 W = Width;
        const int32 H = Height;
        EnqueueReadback(
            [W, H, Callback = MoveTemp(Callback)](const void* Mapped, int32 RowPitchInPixels)
            {
                TArray<FFloat16Color> Data;
                Data.SetNumUninitialized(W * H);
                const uint8* Src = static_cast<const uint8*>(Mapped);
                uint8* Dst = reinterpret_cast<uint8*>(Data.GetData());
                const int32 SrcRowBytes = RowPitchInPixels * sizeof(FFloat16Color);
                const int32 DstRowBytes = W * sizeof(FFloat16Color);
                for (int32 Row = 0; Row < H; ++Row)
                {
                    FMemory::Memcpy(Dst + Row * DstRowBytes, Src + Row * SrcRowBytes, DstRowBytes);
                }
                Callback(Data);
            });
    });
}
