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
#include <atomic>

class AFlashPawn;
class FVCCSimPanelSelection;
class FViewport;
class FEditorViewportClient;

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
     * @param bDatasetChannelsOnly When true, capture the fixed dataset channel set
     *        (RGB, Normal, BaseColor, MaterialProperties) regardless of panel camera toggles.
     * @param bRgbOnly When true (with bDatasetChannelsOnly), capture only RGB and skip the
     *        lighting-independent GT channels (Normal/BaseColor/MaterialProperties) — used when
     *        those are reused from another capture under different lighting.
     */
    void CaptureImageFromCurrentPose(
        int32 PoseIndex,
        const FString& InSaveDirectory,
        bool& bAnyCaptured,
        bool bDatasetChannelsOnly = false,
        bool bRgbOnly = false);

    /**
     * Locks the editor perspective viewport to the RGB camera resolution so the direct
     * viewport RGB capture renders at exactly the dataset resolution (matching the other
     * channels). Call once at session start. No-op when RGB uses the RGBCamera-class path.
     */
    void BeginViewportCaptureSession(AFlashPawn* Pawn);

    /** Restores the editor viewport's size. Call at session end. */
    void EndViewportCaptureSession();

    /** Throwaway viewport frames drawn before reading back each direct-viewport RGB capture, so the
     *  viewport's temporal occlusion culling (HZB), Lumen and streaming converge to the jumped-to pose
     *  before the pixels are read. Without this the first frame after a camera jump culls parts of
     *  buildings using the previous pose's depth and they go missing. */
    void SetViewportWarmupFrames(int32 N) { ViewportWarmupFrames = FMath::Max(0, N); }

    /** Per-pose completion mask for an existing capture directory, used by resume. Element i is true
     *  only when EVERY expected output file for pose i (each channel of the active set, for every
     *  camera of that type) exists on disk with non-zero size. The channel set mirrors
     *  CaptureImageFromCurrentPose's branch on (bDatasetChannelsOnly, bRgbOnly), and filenames mirror
     *  the Save* helpers, so "complete" matches exactly what a full capture would have written.
     *  Returns an all-false mask (capture everything) when no channels/cameras are known. */
    TArray<bool> ComputeCompletedPoses(
        AFlashPawn* Pawn,
        const FString& Dir,
        int32 PoseCount,
        bool bDatasetChannelsOnly,
        bool bRgbOnly) const;

    /** Number of capture readbacks still in flight (async save tasks not yet dispatched). */
    int32 GetPendingJobCount() const { return JobNum.IsValid() ? JobNum->load() : 0; }

    /** Sensor readbacks + save tasks — the total unfinished work used for capture
     *  backpressure and end-of-session draining. */
    int32 GetInFlightCount() const
    {
        return (JobNum.IsValid()     ? JobNum->load()     : 0)
             + (SaveJobNum.IsValid() ? SaveJobNum->load() : 0);
    }

private:
    /** Reference to the selection manager to get the selected pawn and camera states. */
    TWeakPtr<FVCCSimPanelSelection> SelectionManager;

    /** A shared counter for all asynchronous save operations. */
    TSharedPtr<std::atomic<int32>> JobNum;

    /** Save tasks (PNG/EXR encode + disk write) still running; separate from the
     *  readback counter so backpressure can bound queued full-res image copies. */
    TSharedPtr<std::atomic<int32>> SaveJobNum;

    void SaveRGB(AFlashPawn* SelectedFlashPawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured);
    void SaveDepth(AFlashPawn* SelectedFlashPawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured);
    void SaveSeg(AFlashPawn* SelectedFlashPawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured);
    void SaveNormal(AFlashPawn* SelectedFlashPawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured);
    void SaveRGBLinear(AFlashPawn* SelectedFlashPawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured);
    void SaveBaseColor(AFlashPawn* SelectedFlashPawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured);
    void SaveMaterialProperties(AFlashPawn* SelectedFlashPawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured);

    // Direct editor-viewport RGB capture (replaces the high-res-screenshot path): render
    // the level-editor viewport at the camera pose/resolution and read it back. The engine's
    // ReadPixels converts whatever pixel format the viewport target uses into 8-bit sRGB
    // FColor (a raw GPU readback would misinterpret a non-BGRA8 target); the save is async.
    void CaptureRGBFromViewport(AFlashPawn* Pawn, int32 PoseIndex, const FString& InSaveDirectory, bool& bAnyCaptured);
    static FEditorViewportClient* FindPerspectiveViewportClient();

    bool bViewportSizeFixed = false;
    bool bSavedMotionBlur = false;
    int32 ViewportWarmupFrames = 3;
};
