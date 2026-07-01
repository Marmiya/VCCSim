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
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program. If not, see <https://www.gnu.org/licenses/>.
*/

DEFINE_LOG_CATEGORY_STATIC(LogSkyHDRICapture, Log, All);

#include "Utils/SkyHDRICapture.h"
#include "Sensors/DepthCamera.h"
#include "Utils/ImageProcesser.h"
#include "Engine/SceneCapture2D.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/DirectionalLight.h"
#include "Components/DirectionalLightComponent.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "TextureResource.h"
#include "RenderingThread.h"
#include "Async/Async.h"
#include "Async/AsyncWork.h"

namespace
{
    struct FFaceData
    {
        TArray<FFloat16Color> Pixels;
        FVector Forward;
        FVector Right;
        FVector Up;
    };

    FLinearColor SampleFaceBilinear(const FFaceData& Face, int32 FaceSize, float U, float V)
    {
        const float Fx = FMath::Clamp(U, 0.0f, 1.0f) * (FaceSize - 1);
        const float Fy = FMath::Clamp(V, 0.0f, 1.0f) * (FaceSize - 1);
        const int32 X0 = FMath::FloorToInt(Fx);
        const int32 Y0 = FMath::FloorToInt(Fy);
        const int32 X1 = FMath::Min(X0 + 1, FaceSize - 1);
        const int32 Y1 = FMath::Min(Y0 + 1, FaceSize - 1);
        const float Tx = Fx - X0;
        const float Ty = Fy - Y0;

        auto Px = [&](int32 X, int32 Y) -> FLinearColor
        {
            const FFloat16Color& C = Face.Pixels[Y * FaceSize + X];
            return FLinearColor(C.R.GetFloat(), C.G.GetFloat(), C.B.GetFloat());
        };

        const FLinearColor C00 = Px(X0, Y0);
        const FLinearColor C10 = Px(X1, Y0);
        const FLinearColor C01 = Px(X0, Y1);
        const FLinearColor C11 = Px(X1, Y1);
        return FMath::Lerp(FMath::Lerp(C00, C10, Tx), FMath::Lerp(C01, C11, Tx), Ty);
    }
}

bool FSkyHDRICapture::CaptureSkyEquirect(
    UWorld* World, const FString& OutExrPath, int32 EquirectWidth, int32 FaceSize)
{
    if (!World)
    {
        UE_LOG(LogSkyHDRICapture, Warning, TEXT("CaptureSkyEquirect: null world; skip"));
        return false;
    }

    const int32 W = FMath::Max(EquirectWidth, 2);
    const int32 H = W / 2;

    UTextureRenderTarget2D* RT = NewObject<UTextureRenderTarget2D>();
    RT->InitCustomFormat(FaceSize, FaceSize, PF_FloatRGBA, true);
    RT->RenderTargetFormat = RTF_RGBA16f;
    RT->bForceLinearGamma = true;
    RT->SRGB = false;
    RT->UpdateResource();

    ASceneCapture2D* CaptureActor =
        World->SpawnActor<ASceneCapture2D>(FVector(0.0f, 0.0f, 1000.0f), FRotator::ZeroRotator);
    if (!CaptureActor)
    {
        UE_LOG(LogSkyHDRICapture, Error, TEXT("CaptureSkyEquirect: failed to spawn capture actor"));
        return false;
    }

    USceneCaptureComponent2D* Capture = CaptureActor->GetCaptureComponent2D();
    Capture->TextureTarget = RT;
    Capture->CaptureSource = SCS_SceneColorHDR;
    Capture->FOVAngle = 90.0f;
    Capture->ProjectionType = ECameraProjectionMode::Perspective;
    Capture->bCaptureEveryFrame = false;
    Capture->bCaptureOnMovement = false;
    Capture->bAlwaysPersistRenderingState = true;
    Capture->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_UseShowOnlyList;
    Capture->ShowOnlyComponents.Empty();

    Capture->ShowFlags.SetAtmosphere(true);
    Capture->ShowFlags.SetCloud(true);
    Capture->ShowFlags.SetFog(false);
    Capture->ShowFlags.SetVolumetricFog(false);
    Capture->ShowFlags.SetMotionBlur(false);
    Capture->ShowFlags.SetAntiAliasing(false);

    Capture->PostProcessSettings.bOverride_AutoExposureMethod = true;
    Capture->PostProcessSettings.AutoExposureMethod = AEM_Manual;
    Capture->PostProcessSettings.bOverride_AutoExposureBias = true;
    Capture->PostProcessSettings.AutoExposureBias = 0.0f;

    TArray<TPair<UDirectionalLightComponent*, float>> SavedSunAngles;
    for (TActorIterator<ADirectionalLight> It(World); It; ++It)
    {
        if (ADirectionalLight* Light = *It)
        {
            if (UDirectionalLightComponent* LC =
                Cast<UDirectionalLightComponent>(Light->GetLightComponent()))
            {
                SavedSunAngles.Emplace(LC, LC->LightSourceAngle);
                LC->LightSourceAngle = 0.0f;
                LC->MarkRenderStateDirty();
            }
        }
    }

    const FVector FaceForwards[6] = {
        FVector(1, 0, 0), FVector(-1, 0, 0),
        FVector(0, 1, 0), FVector(0, -1, 0),
        FVector(0, 0, 1), FVector(0, 0, -1)
    };

    FFaceData Faces[6];
    for (int32 f = 0; f < 6; ++f)
    {
        const FRotator FaceRot = FRotationMatrix::MakeFromX(FaceForwards[f]).Rotator();
        CaptureActor->SetActorRotation(FaceRot);

        const FRotationMatrix RotM(FaceRot);
        Faces[f].Forward = RotM.GetScaledAxis(EAxis::X);
        Faces[f].Right   = RotM.GetScaledAxis(EAxis::Y);
        Faces[f].Up      = RotM.GetScaledAxis(EAxis::Z);

        Capture->CaptureScene();
        FlushRenderingCommands();

        FTextureRenderTargetResource* RTRes = RT->GameThread_GetRenderTargetResource();
        if (!RTRes || !RTRes->ReadFloat16Pixels(Faces[f].Pixels) ||
            Faces[f].Pixels.Num() != FaceSize * FaceSize)
        {
            UE_LOG(LogSkyHDRICapture, Error,
                TEXT("CaptureSkyEquirect: face %d readback failed"), f);
            for (const TPair<UDirectionalLightComponent*, float>& S : SavedSunAngles)
            {
                S.Key->LightSourceAngle = S.Value;
                S.Key->MarkRenderStateDirty();
            }
            CaptureActor->Destroy();
            return false;
        }
    }

    for (const TPair<UDirectionalLightComponent*, float>& S : SavedSunAngles)
    {
        S.Key->LightSourceAngle = S.Value;
        S.Key->MarkRenderStateDirty();
    }
    CaptureActor->Destroy();

    TArray<FFloat16Color> Equirect;
    Equirect.SetNumUninitialized(W * H);

    for (int32 Py = 0; Py < H; ++Py)
    {
        const double V01 = (Py + 0.5) / H;
        const double Theta = V01 * PI;
        const double SinT = FMath::Sin(Theta);
        const double CosT = FMath::Cos(Theta);

        for (int32 Px = 0; Px < W; ++Px)
        {
            const double U01 = (Px + 0.5) / W;
            const double Phi = U01 * 2.0 * PI;

            // RH world (+X east/right, +Y north/fwd, +Z up): yaw=0 -> +Y, increasing toward +X.
            const double XRh = SinT * FMath::Sin(Phi);
            const double YRh = SinT * FMath::Cos(Phi);
            const double ZRh = CosT;

            // RH world -> UE world: swap X<->Y.
            const FVector Dir(YRh, XRh, ZRh);

            int32 BestFace = 0;
            float BestDot = -FLT_MAX;
            for (int32 f = 0; f < 6; ++f)
            {
                const float D = static_cast<float>(Dir | Faces[f].Forward);
                if (D > BestDot) { BestDot = D; BestFace = f; }
            }

            const FFaceData& Face = Faces[BestFace];
            const float FwdDot = static_cast<float>(Dir | Face.Forward);
            const float Su = static_cast<float>(Dir | Face.Right) / FwdDot;
            const float Sv = static_cast<float>(Dir | Face.Up) / FwdDot;
            const float U = 0.5f + 0.5f * Su;
            const float Vv = 0.5f - 0.5f * Sv;

            const FLinearColor Sampled = SampleFaceBilinear(Face, FaceSize, U, Vv);
            FFloat16Color& Out = Equirect[Py * W + Px];
            Out.R = Sampled.R;
            Out.G = Sampled.G;
            Out.B = Sampled.B;
            Out.A = FFloat16(1.0f);
        }
    }

    const FIntPoint Size(W, H);
    Async(EAsyncExecution::ThreadPool,
        [Equirect = MoveTemp(Equirect), Size, OutExrPath]()
        {
            FAsyncEXRSaveTask(Equirect, Size, OutExrPath).DoWork();
        });

    UE_LOG(LogSkyHDRICapture, Log,
        TEXT("CaptureSkyEquirect: wrote %dx%d sky panorama to %s"), W, H, *OutExrPath);
    return true;
}
