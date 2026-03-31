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
#include "HighResScreenshot.h"
#include "LevelEditorViewport.h"
#include "Editor.h"

DEFINE_LOG_CATEGORY_STATIC(LogImageCaptureService, Log, All);

FImageCaptureService::FImageCaptureService(TSharedPtr<FVCCSimPanelSelection> InSelectionManager)
    : SelectionManager(InSelectionManager)
{
    JobNum = MakeShared<std::atomic<int32>>(0);
}

void FImageCaptureService::CaptureImageFromCurrentPose(int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured)
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
                    [Filename, Size, JobNum = this->JobNum](const TArray<FColor>& ImageData)
                    {
                        (new FAutoDeleteAsyncTask<FAsyncImageSaveTask>(ImageData, Size, Filename))->StartBackgroundTask();
                        *JobNum -= 1;
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
        TArray<URGBCameraComponent*> RGBCameras;
        SelectedFlashPawn->GetComponents<URGBCameraComponent>(RGBCameras);
        *JobNum += RGBCameras.Num();
        
        FEditorViewportClient* ViewportClient = nullptr;
        for (FLevelEditorViewportClient* LevelVC : GEditor->GetLevelViewportClients())
        {
            if (LevelVC && LevelVC->Viewport && !LevelVC->IsOrtho())
            {
                ViewportClient = LevelVC;
                break;
            }
        }
        
        if (!ViewportClient)
        {
            UE_LOG(LogImageCaptureService, Error, TEXT("No valid editor viewport found"));
            *JobNum -= RGBCameras.Num();
            return;
        }
        
        for (int32 i = 0; i < RGBCameras.Num(); ++i)
        {
            URGBCameraComponent* Camera = RGBCameras[i];
            if (Camera)
            {
                int32 CameraIndex = Camera->GetSensorIndex();
                if (CameraIndex < 0) CameraIndex = i;
                
                FString Filename = InSaveDirectory / FString::Printf(TEXT("RGB_Cam%02d_Pose%03d.png"), CameraIndex, PoseIndex);
                FIntPoint CameraSize = {Camera->GetImageSize().first, Camera->GetImageSize().second};
                FTransform CameraTransform = Camera->GetComponentTransform();
                
                ViewportClient->SetViewLocation(CameraTransform.GetLocation());
                ViewportClient->SetViewRotation(CameraTransform.GetRotation().Rotator());
                ViewportClient->ViewFOV = Camera->FOV;
                ViewportClient->Invalidate();
                ViewportClient->Viewport->Draw();
                
                FHighResScreenshotConfig& HighResScreenshotConfig = GetHighResScreenshotConfig();
                HighResScreenshotConfig.SetResolution(CameraSize.X, CameraSize.Y);
                HighResScreenshotConfig.SetFilename(Filename);
                HighResScreenshotConfig.bMaskEnabled = false;
                HighResScreenshotConfig.bCaptureHDR = false;
                
                FScreenshotRequest::RequestScreenshot(Filename, false, false);
                *JobNum -= 1;
                
                bAnyCaptured = true;
            }
            else
            {
                *JobNum -= 1;
            }
        }
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
                [DepthFilename, Size, JobNum = this->JobNum](const TArray<FFloat16Color>& ImageData)
                {
                    TArray<float> DepthValues;
                    DepthValues.SetNum(ImageData.Num());
                    for (int32 idx = 0; idx < ImageData.Num(); ++idx)
                    {
                        DepthValues[idx] = ImageData[idx].A;
                    }

                    (new FAutoDeleteAsyncTask<FAsyncDepthSaveTask>(DepthValues, Size, DepthFilename))->StartBackgroundTask();
                    *JobNum -= 1;
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
                [Filename, Size, JobNum = this->JobNum](TArray<FColor> ImageData)
                {
                    (new FAutoDeleteAsyncTask<FAsyncImageSaveTask>(ImageData, Size, Filename))->StartBackgroundTask();
                    *JobNum -= 1;
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
                [NormalEXRFilename, Size, JobNum = this->JobNum](const TArray<FFloat16Color>& NormalData)
                {
                    (new FAutoDeleteAsyncTask<FAsyncNormalEXRSaveTask>(NormalData, Size, NormalEXRFilename))->StartBackgroundTask();
                    *JobNum -= 1;
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
                [Filename, Size, JobNum = this->JobNum](const TArray<FColor>& ImageData)
                {
                    TArray<FColor> DataCopy = ImageData;
                    for (FColor& Pixel : DataCopy) Pixel.A = 255;
                    (new FAutoDeleteAsyncTask<FAsyncImageSaveTask>(DataCopy, Size, Filename))->StartBackgroundTask();
                    *JobNum -= 1;
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
                [Filename, Size, JobNum = this->JobNum](const TArray<FColor>& ImageData)
                {
                    TArray<FColor> DataCopy = ImageData;
                    for (FColor& Pixel : DataCopy) Pixel.A = 255;
                    (new FAutoDeleteAsyncTask<FAsyncImageSaveTask>(DataCopy, Size, Filename))->StartBackgroundTask();
                    *JobNum -= 1;
                });
            bAnyCaptured = true;
        }
        else
        {
            *JobNum -= 1;
        }
    }
}
