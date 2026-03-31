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

#pragma once

#include "CoreMinimal.h"

class AFlashPawn;
class FVCCSimPanelSelection;

/**
 * Service class for handling image capture operations from various sensor components.
 */
class VCCSIMEDITOR_API FImageCaptureService
{
public:
    FImageCaptureService(TSharedPtr<FVCCSimPanelSelection> InSelectionManager);

    /**
     * Captures images from all active cameras on the selected FlashPawn for a given pose.
     *
     * @param PoseIndex The index of the current pose, used for filenames.
     * @param InSaveDirectory The directory where images will be saved.
     * @param bAnyCaptured Output parameter, set to true if at least one image was captured.
     */
    void CaptureImageFromCurrentPose(int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured);

private:
    /** Reference to the selection manager to get the selected pawn and camera states. */
    TWeakPtr<FVCCSimPanelSelection> SelectionManager;

    /** A shared counter for all asynchronous save operations. */
    TSharedPtr<std::atomic<int32>> JobNum;

    /**
     * Saves RGB image data.
     * @param SelectedFlashPawn The pawn to capture from.
     * @param PoseIndex The index of the current pose.
     * @param InSaveDirectory The directory to save to.
     * @param bAnyCaptured Set to true if an image is saved.
     */
    void SaveRGB(AFlashPawn* SelectedFlashPawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured);

    /**
     * Saves Depth image data.
     */
    void SaveDepth(AFlashPawn* SelectedFlashPawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured);

    /**
     * Saves Segmentation image data.
     */
    void SaveSeg(AFlashPawn* SelectedFlashPawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured);
    
    /**
     * Saves Normal vector image data.
     */
    void SaveNormal(AFlashPawn* SelectedFlashPawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured);

    /**
     * Saves BaseColor (albedo) image data.
     */
    void SaveBaseColor(AFlashPawn* SelectedFlashPawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured);

    /**
     * Saves Material Properties (Roughness, Metallic, etc.) image data.
     */
    void SaveMaterialProperties(AFlashPawn* SelectedFlashPawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured);
};
