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

#include "Utils/ImageCaptureService.h"
#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Pawns/FlashPawn.h"
#include "Sensors/RGBCamera.h"
#include "Sensors/DepthCamera.h"
#include "Sensors/SegmentationCamera.h"
#include "Sensors/NormalCamera.h"
#include "Sensors/BaseColorCamera.h"
#include "Sensors/MaterialPropertiesCamera.h"
#include "Utils/ImageProcesser.h"
#include "LevelEditorViewport.h"
#include "Editor.h"
#include "Async/Async.h"
#include "Slate/SceneViewport.h"
#include "UnrealClient.h"

DEFINE_LOG_CATEGORY_STATIC(LogImageCaptureService, Log, All);

FImageCaptureService::FImageCaptureService(TSharedPtr<FVCCSimPanelSelection> InSelectionManager)
    : SelectionManager(InSelectionManager)
{
    JobNum = MakeShared<std::atomic<int32>>(0);
    SaveJobNum = MakeShared<std::atomic<int32>>(0);
}

void FImageCaptureService::CaptureImageFromCurrentPose(
    int32 PoseIndex,
    const FString& InSaveDirectory,
    bool& bAnyCaptured,
    bool bDatasetChannelsOnly)
{
    TSharedPtr<FVCCSimPanelSelection> SelectionManagerPin = SelectionManager.Pin();
    if (!SelectionManagerPin.IsValid())
    {
        UE_LOG(LogImageCaptureService, Warning, TEXT("SelectionManager is not valid."));
        return;
    }

    AFlashPawn* SelectedFlashPawn = SelectionManagerPin->GetSelectedFlashPawn().Get();
    if (!SelectedFlashPawn)
    {
        UE_LOG(LogImageCaptureService, Warning, TEXT("No FlashPawn selected."));
        return;
    }

    if (SelectedFlashPawn->IsReady())
    {
        if (bDatasetChannelsOnly)
        {
            if (SelectionManagerPin->HasRGBCamera())
            {
                SaveRGB(SelectedFlashPawn, PoseIndex, InSaveDirectory, bAnyCaptured);
            }
            if (SelectionManagerPin->HasNormalCamera())
            {
                SaveNormal(SelectedFlashPawn, PoseIndex, InSaveDirectory, bAnyCaptured);
            }
            if (SelectionManagerPin->HasBaseColorCamera())
            {
                SaveBaseColor(SelectedFlashPawn, PoseIndex, InSaveDirectory, bAnyCaptured);
            }
            if (SelectionManagerPin->HasMaterialPropertiesCamera())
            {
                SaveMaterialProperties(SelectedFlashPawn, PoseIndex, InSaveDirectory, bAnyCaptured);
            }
        }
        else
        {
            if (SelectionManagerPin->IsUsingRGBCamera() && SelectionManagerPin->HasRGBCamera())
            {
                SaveRGB(SelectedFlashPawn, PoseIndex, InSaveDirectory, bAnyCaptured);
            }
            if (SelectionManagerPin->IsUsingDepthCamera() && SelectionManagerPin->HasDepthCamera())
            {
                SaveDepth(SelectedFlashPawn, PoseIndex, InSaveDirectory, bAnyCaptured);
            }
            if (SelectionManagerPin->IsUsingSegmentationCamera() && SelectionManagerPin->HasSegmentationCamera())
            {
                SaveSeg(SelectedFlashPawn, PoseIndex, InSaveDirectory, bAnyCaptured);
            }
            if (SelectionManagerPin->IsUsingNormalCamera() && SelectionManagerPin->HasNormalCamera())
            {
                SaveNormal(SelectedFlashPawn, PoseIndex, InSaveDirectory, bAnyCaptured);
            }
            if (SelectionManagerPin->IsUsingBaseColorCamera() && SelectionManagerPin->HasBaseColorCamera())
            {
                SaveBaseColor(SelectedFlashPawn, PoseIndex, InSaveDirectory, bAnyCaptured);
            }
            if (SelectionManagerPin->IsUsingMaterialPropertiesCamera() && SelectionManagerPin->HasMaterialPropertiesCamera())
            {
                SaveMaterialProperties(SelectedFlashPawn, PoseIndex, InSaveDirectory, bAnyCaptured);
            }
        }

        if (!bAnyCaptured)
        {
            UE_LOG(LogImageCaptureService, Warning, TEXT("No images captured. Ensure cameras are enabled."));
        }
    }
    else
    {
        UE_LOG(LogImageCaptureService, Warning, TEXT("FlashPawn not ready for capture. Wait for it to reach position."));
    }
}

void FImageCaptureService::SaveRGB(AFlashPawn* SelectedFlashPawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured)
{
    auto SelectionManagerPin = SelectionManager.Pin();
    if (!SelectionManagerPin.IsValid()) return;

    if (SelectionManagerPin->ShouldUseRGBCameraClass())
    {
        TArray<URGBCameraComponent*> RGBCameras;
        SelectedFlashPawn->GetComponents<URGBCameraComponent>(RGBCameras);
        *JobNum += RGBCameras.Num();

        for (int32 i = 0; i < RGBCameras.Num(); ++i)
        {
            URGBCameraComponent* Camera = RGBCameras[i];
            if (Camera)
            {
                int32 CameraIndex = Camera->GetSensorIndex();
                if (CameraIndex < 0) CameraIndex = i;

                FString Filename = InSaveDirectory / FString::Printf(TEXT("RGB_Cam%02d_Pose%03d.png"), CameraIndex, PoseIndex);
                FIntPoint Size = {Camera->GetImageSize().first, Camera->GetImageSize().second};

                Camera->AsyncGetRGBImageData(
                    [Filename, Size, JobNum = this->JobNum, SaveJobNum = this->SaveJobNum](const TArray<FColor>& ImageData)
                    {
                        TArray<FColor> DataCopy = ImageData;
                        *JobNum -= 1;
                        (*SaveJobNum)++;
                        Async(EAsyncExecution::ThreadPool,
                            [DataCopy = MoveTemp(DataCopy), Size, Filename, SaveJobNum]()
                            {
                                FAsyncImageSaveTask(DataCopy, Size, Filename).DoWork();
                                (*SaveJobNum)--;
                            });
                    });
                bAnyCaptured = true;
            }
            else
            {
                *JobNum -= 1;
            }
        }
    }
    else
    {
        CaptureRGBFromViewport(SelectedFlashPawn, PoseIndex, InSaveDirectory, bAnyCaptured);
    }
}

void FImageCaptureService::SaveDepth(AFlashPawn* SelectedFlashPawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured)
{
    TArray<UDepthCameraComponent*> DepthCameras;
    SelectedFlashPawn->GetComponents<UDepthCameraComponent>(DepthCameras);
    *JobNum += DepthCameras.Num();

    for (int32 i = 0; i < DepthCameras.Num(); ++i)
    {
        UDepthCameraComponent* Camera = DepthCameras[i];
        if (Camera)
        {
            int32 CameraIndex = Camera->GetSensorIndex();
            if (CameraIndex < 0) CameraIndex = i;

            FString DepthFilename = InSaveDirectory / FString::Printf(TEXT("Depth16_Cam%02d_Pose%03d.png"), CameraIndex, PoseIndex);
            FIntPoint Size = {Camera->GetImageSize().first, Camera->GetImageSize().second};

            Camera->AsyncGetDepthImageData(
                [DepthFilename, Size, JobNum = this->JobNum, SaveJobNum = this->SaveJobNum](const TArray<FFloat16Color>& ImageData)
                {
                    TArray<float> DepthValues;
                    DepthValues.SetNum(ImageData.Num());
                    for (int32 idx = 0; idx < ImageData.Num(); ++idx)
                    {
                        DepthValues[idx] = ImageData[idx].R;
                    }

                    *JobNum -= 1;
                    (*SaveJobNum)++;
                    Async(EAsyncExecution::ThreadPool,
                        [DepthValues = MoveTemp(DepthValues), Size, DepthFilename, SaveJobNum]()
                        {
                            FAsyncDepthSaveTask(DepthValues, Size, DepthFilename).DoWork();
                            (*SaveJobNum)--;
                        });
                });
            bAnyCaptured = true;
        }
        else
        {
            *JobNum -= 1;
        }
    }
}

void FImageCaptureService::SaveSeg(AFlashPawn* SelectedFlashPawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured)
{
    TArray<USegCameraComponent*> SegmentationCameras;
    SelectedFlashPawn->GetComponents<USegCameraComponent>(SegmentationCameras);
    *JobNum += SegmentationCameras.Num();

    for (int32 i = 0; i < SegmentationCameras.Num(); ++i)
    {
        USegCameraComponent* Camera = SegmentationCameras[i];
        if (Camera)
        {
            int32 CameraIndex = Camera->GetSensorIndex();
            if (CameraIndex < 0) CameraIndex = i;

            FString Filename = InSaveDirectory / FString::Printf(TEXT("Seg_Cam%02d_Pose%03d.png"), CameraIndex, PoseIndex);
            FIntPoint Size = {Camera->GetImageSize().first, Camera->GetImageSize().second};

            Camera->AsyncGetSegmentationImageData(
                [Filename, Size, JobNum = this->JobNum, SaveJobNum = this->SaveJobNum](TArray<FColor> ImageData)
                {
                    *JobNum -= 1;
                    (*SaveJobNum)++;
                    Async(EAsyncExecution::ThreadPool,
                        [ImageData = MoveTemp(ImageData), Size, Filename, SaveJobNum]()
                        {
                            FAsyncImageSaveTask(ImageData, Size, Filename).DoWork();
                            (*SaveJobNum)--;
                        });
                });
            bAnyCaptured = true;
        }
        else
        {
            *JobNum -= 1;
        }
    }
}

void FImageCaptureService::SaveNormal(AFlashPawn* SelectedFlashPawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured)
{
    TArray<UNormalCameraComponent*> NormalCameras;
    SelectedFlashPawn->GetComponents<UNormalCameraComponent>(NormalCameras);
    *JobNum += NormalCameras.Num();

    for (int32 i = 0; i < NormalCameras.Num(); ++i)
    {
        UNormalCameraComponent* Camera = NormalCameras[i];
        if (Camera)
        {
            int32 CameraIndex = Camera->GetSensorIndex();
            if (CameraIndex < 0) CameraIndex = i;

            FString NormalEXRFilename = InSaveDirectory / FString::Printf(TEXT("Normal_Cam%02d_Pose%03d.exr"), CameraIndex, PoseIndex);
            FIntPoint Size = {Camera->GetImageSize().first, Camera->GetImageSize().second};
            
            Camera->AsyncGetNormalImageData(
                [NormalEXRFilename, Size, JobNum = this->JobNum, SaveJobNum = this->SaveJobNum](const TArray<FFloat16Color>& NormalData)
                {
                    TArray<FFloat16Color> DataCopy = NormalData;
                    *JobNum -= 1;
                    (*SaveJobNum)++;
                    Async(EAsyncExecution::ThreadPool,
                        [DataCopy = MoveTemp(DataCopy), Size, NormalEXRFilename, SaveJobNum]()
                        {
                            FAsyncNormalEXRSaveTask(DataCopy, Size, NormalEXRFilename).DoWork();
                            (*SaveJobNum)--;
                        });
                });
            bAnyCaptured = true;
        }
        else
        {
            *JobNum -= 1;
        }
    }
}

void FImageCaptureService::SaveBaseColor(AFlashPawn* SelectedFlashPawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured)
{
    TArray<UBaseColorCameraComponent*> Cameras;
    SelectedFlashPawn->GetComponents<UBaseColorCameraComponent>(Cameras);
    *JobNum += Cameras.Num();

    for (int32 i = 0; i < Cameras.Num(); ++i)
    {
        UBaseColorCameraComponent* Camera = Cameras[i];
        if (Camera)
        {
            int32 CameraIndex = Camera->GetSensorIndex();
            if (CameraIndex < 0) CameraIndex = i;

            FString Filename = InSaveDirectory / FString::Printf(TEXT("BaseColor_Cam%02d_Pose%03d.png"), CameraIndex, PoseIndex);
            FIntPoint Size = {Camera->GetImageSize().first, Camera->GetImageSize().second};

            Camera->AsyncGetBaseColorImageData(
                [Filename, Size, JobNum = this->JobNum, SaveJobNum = this->SaveJobNum](const TArray<FColor>& ImageData)
                {
                    TArray<FColor> DataCopy = ImageData;
                    for (FColor& Pixel : DataCopy) Pixel.A = 255;
                    *JobNum -= 1;
                    (*SaveJobNum)++;
                    Async(EAsyncExecution::ThreadPool,
                        [DataCopy = MoveTemp(DataCopy), Size, Filename, SaveJobNum]()
                        {
                            FAsyncImageSaveTask(DataCopy, Size, Filename).DoWork();
                            (*SaveJobNum)--;
                        });
                });
            bAnyCaptured = true;
        }
        else
        {
            *JobNum -= 1;
        }
    }
}

void FImageCaptureService::SaveMaterialProperties(AFlashPawn* SelectedFlashPawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured)
{
    TArray<UMaterialPropertiesCameraComponent*> Cameras;
    SelectedFlashPawn->GetComponents<UMaterialPropertiesCameraComponent>(Cameras);
    *JobNum += Cameras.Num();

    for (int32 i = 0; i < Cameras.Num(); ++i)
    {
        UMaterialPropertiesCameraComponent* Camera = Cameras[i];
        if (Camera)
        {
            int32 CameraIndex = Camera->GetSensorIndex();
            if (CameraIndex < 0) CameraIndex = i;

            FString Filename = InSaveDirectory / FString::Printf(TEXT("MatProps_Cam%02d_Pose%03d.png"), CameraIndex, PoseIndex);
            FIntPoint Size = {Camera->GetImageSize().first, Camera->GetImageSize().second};

            Camera->AsyncGetMaterialPropertiesImageData(
                [Filename, Size, JobNum = this->JobNum, SaveJobNum = this->SaveJobNum](const TArray<FFloat16Color>& ImageData)
                {
                    TArray<FFloat16Color> DataCopy = ImageData;
                    *JobNum -= 1;
                    (*SaveJobNum)++;
                    Async(EAsyncExecution::ThreadPool,
                        [DataCopy = MoveTemp(DataCopy), Size, Filename, SaveJobNum]()
                        {
                            FAsyncMatProps16SaveTask(DataCopy, Size, Filename).DoWork();
                            (*SaveJobNum)--;
                        });
                });
            bAnyCaptured = true;
        }
        else
        {
            *JobNum -= 1;
        }
    }
}

// ============================================================================
// DIRECT EDITOR-VIEWPORT RGB CAPTURE
// ============================================================================

FEditorViewportClient* FImageCaptureService::FindPerspectiveViewportClient()
{
    if (!GEditor) return nullptr;
    for (FLevelEditorViewportClient* LevelVC : GEditor->GetLevelViewportClients())
    {
        if (LevelVC && LevelVC->Viewport && !LevelVC->IsOrtho())
        {
            return LevelVC;
        }
    }
    return nullptr;
}

void FImageCaptureService::BeginViewportCaptureSession(AFlashPawn* Pawn)
{
    bViewportSizeFixed = false;

    TSharedPtr<FVCCSimPanelSelection> SelectionManagerPin = SelectionManager.Pin();
    if (!SelectionManagerPin.IsValid() || SelectionManagerPin->ShouldUseRGBCameraClass() || !Pawn)
    {
        return;  // RGBCamera path (or no pawn): the viewport is not used for RGB.
    }

    TArray<URGBCameraComponent*> RGBCameras;
    Pawn->GetComponents<URGBCameraComponent>(RGBCameras);
    if (RGBCameras.Num() == 0 || !RGBCameras[0]) return;

    FEditorViewportClient* VC = FindPerspectiveViewportClient();
    if (!VC || !VC->Viewport) return;

    const std::pair<int32, int32> Size = RGBCameras[0]->GetImageSize();
    static_cast<FSceneViewport*>(VC->Viewport)->SetFixedViewportSize(Size.first, Size.second);

    // The camera teleports between poses, so the motion-blur post-process would smear each
    // captured frame along that jump. Disable it for the session (restored in End...).
    bSavedMotionBlur = VC->EngineShowFlags.MotionBlur != 0;
    VC->EngineShowFlags.SetMotionBlur(false);

    bViewportSizeFixed = true;
    UE_LOG(LogImageCaptureService, Log,
        TEXT("Viewport locked to %dx%d for direct RGB capture"), Size.first, Size.second);
}

void FImageCaptureService::EndViewportCaptureSession()
{
    if (!bViewportSizeFixed) return;
    bViewportSizeFixed = false;

    FEditorViewportClient* VC = FindPerspectiveViewportClient();
    if (VC && VC->Viewport)
    {
        static_cast<FSceneViewport*>(VC->Viewport)->SetFixedViewportSize(0, 0);
        VC->EngineShowFlags.SetMotionBlur(bSavedMotionBlur);
    }
}

void FImageCaptureService::CaptureRGBFromViewport(
    AFlashPawn* Pawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured)
{
    FEditorViewportClient* VC = FindPerspectiveViewportClient();
    if (!VC || !VC->Viewport)
    {
        UE_LOG(LogImageCaptureService, Error, TEXT("No valid editor viewport found"));
        return;
    }

    TArray<URGBCameraComponent*> RGBCameras;
    Pawn->GetComponents<URGBCameraComponent>(RGBCameras);

    for (int32 i = 0; i < RGBCameras.Num(); ++i)
    {
        URGBCameraComponent* Camera = RGBCameras[i];
        if (!Camera) continue;

        int32 CameraIndex = Camera->GetSensorIndex();
        if (CameraIndex < 0) CameraIndex = i;

        const FString Filename =
            InSaveDirectory / FString::Printf(TEXT("RGB_Cam%02d_Pose%03d.png"), CameraIndex, PoseIndex);
        const FTransform CameraTransform = Camera->GetComponentTransform();

        VC->SetViewLocation(CameraTransform.GetLocation());
        VC->SetViewRotation(CameraTransform.GetRotation().Rotator());
        VC->ViewFOV = Camera->FOV;
        VC->Invalidate();
        VC->Viewport->Draw();

        // The viewport render target is often a float / 10-bit format; ReadPixels lets the
        // engine convert it to 8-bit sRGB FColor correctly (a raw GPU readback misinterprets
        // a non-BGRA8 target → colour moire). Synchronous read, async save.
        TArray<FColor> Pixels;
        if (!VC->Viewport->ReadPixels(Pixels))
        {
            UE_LOG(LogImageCaptureService, Warning, TEXT("Viewport ReadPixels failed; RGB pose skipped"));
            continue;
        }
        for (FColor& Px : Pixels) { Px.A = 255; }

        const FIntPoint Sz = VC->Viewport->GetSizeXY();
        TSharedPtr<std::atomic<int32>> SaveCounter = SaveJobNum;
        (*SaveCounter)++;
        Async(EAsyncExecution::ThreadPool,
            [Pixels = MoveTemp(Pixels), Sz, Filename, SaveCounter]()
            {
                FAsyncImageSaveTask(Pixels, Sz, Filename).DoWork();
                (*SaveCounter)--;
            });
        bAnyCaptured = true;
    }
}
