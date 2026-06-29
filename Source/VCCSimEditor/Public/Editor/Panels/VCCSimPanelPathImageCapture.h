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

#pragma once

#include "CoreMinimal.h"
#include "Widgets/SCompoundWidget.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SNumericEntryBox.h"
#include "Widgets/Input/SCheckBox.h"
#include "Widgets/Text/STextBlock.h"
#include "Engine/TimerHandle.h"
#include <atomic>

// Forward declarations
class AFlashPawn;
class FPathGenerator;
class FImageCaptureService;
class FVCCSimPanelSelection;

DECLARE_DELEGATE_OneParam(FOnCaptureSessionComplete, bool /*bSuccess*/);

/**
 * Dedicated panel for Path Configuration and Image Capture functionality
 * Combines path generation, pose configuration, and image capture operations
 */
class VCCSIMEDITOR_API FVCCSimPanelPathImageCapture
{
public:
    FVCCSimPanelPathImageCapture();
    ~FVCCSimPanelPathImageCapture();

    void Initialize();
    void Cleanup();
    TSharedRef<SWidget> CreatePathImageCapturePanel();

    // Integration with selection manager
    void SetSelectionManager(TSharedPtr<FVCCSimPanelSelection> InSelectionManager);

    // External access for main panel
    bool IsAutoCaptureInProgress() const { return bAutoCaptureInProgress; }
    bool IsPathVisualized() const { return bPathVisualized; }
    void UpdatePathNeedsUpdate(bool bNeedsUpdate) { bPathNeedsUpdate = bNeedsUpdate; }

    /**
     * Runs the full FlashPawn path capture into TargetDirectory.
     * Writes poses.txt + intrinsics.json + lighting.json, captures every pose,
     * waits for all async readbacks, then fires OnComplete on the game thread.
     *
     * @param TargetDirectory Capture output directory (created if missing).
     * @param bDatasetChannelsOnly Force the dataset channel set (RGB/Normal/BaseColor/MatProps).
     * @param bRgbOnly Capture only RGB and skip the lighting-independent GT channels
     *        (Normal/BaseColor/MatProps) — used when those are reused from another capture.
     * @param OnComplete Fired with true on success, false on cancel/failure.
     * @return false if a session is already running or prerequisites are missing.
     */
    bool StartCaptureSession(
        const FString& TargetDirectory,
        bool bDatasetChannelsOnly,
        bool bRgbOnly,
        FOnCaptureSessionComplete OnComplete);

    /** True when the selected FlashPawn has a path and the dataset cameras are present. */
    bool CanRunDatasetCapture(FString& OutReason) const;

    /** Stable hash of the selected FlashPawn's current path poses (the same poses written to
     *  poses.txt). Two captures along the same path share it — used to gate GT-channel reuse.
     *  Empty if no FlashPawn/path is available. */
    FString ComputePathPoseKey() const;

    /** Cancels the running capture session; fires the session delegate with false. */
    void StopAutoCapture();

    // UI state access for persistence
    bool IsPathImageCaptureSectionExpanded() const { return bPathImageCaptureSectionExpanded; }
    void SetPathImageCaptureSectionExpanded(bool bExpanded) { bPathImageCaptureSectionExpanded = bExpanded; }

    void LoadFromConfigManager();
    void SaveToConfigManager() const;

    // UI Construction (public for _UI.cpp)
    TSharedRef<SWidget> CreatePathConfigSection();
    TSharedRef<SWidget> CreateImageCaptureSection();
    TSharedRef<SWidget> CreatePoseFileButtons();
    TSharedRef<SWidget> CreatePoseActionButtons();
    TSharedRef<SWidget> CreateMovementButtons();
    TSharedRef<SWidget> CreateCaptureButtons();

private:
    // UI Callbacks
    FReply OnGeneratePosesClicked();
    FReply OnLoadPoseClicked();
    FReply OnSavePoseClicked();
    FReply OnTogglePathVisualizationClicked();
    FReply OnCaptureImagesClicked();
    
    // Path Configuration & Generation
    void GeneratePosesAroundTarget();
    
    // Pose File I/O
    void LoadPredefinedPose();
    void SaveGeneratedPose();
    void WritePosesToFile(const TArray<FVector>& Positions, const TArray<FRotator>& Rotations, const FString& FilePath);
    
    // Path Visualization
    void UpdatePathVisualization();
    void ShowPathVisualization();
    void HidePathVisualization();
    
    // Image Capture
    void CaptureImageFromCurrentPose();
    void StartAutoCapture();
    void TickCaptureSession();
    void FinishCaptureSession(bool bSuccess);

    // Utility
    static FString GetTimestampedFilename();

    // UI Widgets
    TSharedPtr<SNumericEntryBox<float>> OrbitMarginSpinBox;
    TSharedPtr<SNumericEntryBox<float>> OrbitStartHeightSpinBox;
    TSharedPtr<SNumericEntryBox<float>> OrbitCameraHFOVSpinBox;
    TSharedPtr<SNumericEntryBox<float>> OrbitHOverlapSpinBox;
    TSharedPtr<SNumericEntryBox<float>> OrbitVOverlapSpinBox;
    TSharedPtr<SNumericEntryBox<float>> OrbitSurveyOverlapSpinBox;
    TSharedPtr<SNumericEntryBox<float>> OrbitNadirAltSpinBox;
    TSharedPtr<SNumericEntryBox<float>> OrbitNadirTiltSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> OrbitObliqueRingsSpinBox;
    TSharedPtr<SNumericEntryBox<float>> CaptureTickIntervalSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> PoseWarmupFramesSpinBox;
    TSharedPtr<SButton> VisualizePathButton;
    TSharedPtr<SButton> AutoCaptureButton;
    
    // State Variables
    float OrbitMargin = 500.0f;
    float OrbitStartHeight = 200.0f;
    float OrbitCameraHFOV = 90.0f;
    float OrbitHOverlap = 0.60f;
    float OrbitVOverlap = 0.60f;
    float OrbitSurveyOverlap = 0.70f;
    float OrbitNadirAlt = 500.0f;
    float OrbitNadirTiltAngle = 45.0f;
    bool bOrbitIncludeOblique = false;
    int32 OrbitObliqueRings = 2;
    bool bOrbitSideOrbit = false;
    float CaptureTickInterval = 0.2f;

    TOptional<float> OrbitMarginValue;
    TOptional<float> OrbitStartHeightValue;
    TOptional<float> OrbitCameraHFOVValue;
    TOptional<float> OrbitHOverlapValue;
    TOptional<float> OrbitVOverlapValue;
    TOptional<float> OrbitSurveyOverlapValue;
    TOptional<float> OrbitNadirAltValue;
    TOptional<float> OrbitNadirTiltValue;
    TOptional<int32> OrbitObliqueRingsValue;
    TOptional<float> CaptureTickIntervalValue;
    TOptional<int32> PoseWarmupFramesValue;
    
    bool bPathVisualized = false;
    bool bPathNeedsUpdate = true;
    TWeakObjectPtr<AActor> PathVisualizationActor;
    
    bool bAutoCaptureInProgress = false;
    bool bGenerationInProgress = false;
    bool bGameViewChangedForCapture = false;
    bool bSessionDatasetChannelsOnly = false;
    bool bSessionRgbOnly = false;
    bool bDrainingCaptureJobs = false;
    bool bSessionCancelled = false;
    // Per-pose warm-up: ticks spent rendering throwaway frames at each pose before capture, so temporal
    // occlusion culling (HZB) / Lumen / exposure history converge to the jumped-to pose. This counts down
    // the SceneCapture channels' warm-up (BaseColor / Normal / Depth / MaterialProperties); direct-
    // viewport RGB warms up separately inside CaptureRGBFromViewport. Both are driven by PoseWarmupFrames,
    // which is exposed in the panel next to Tick and persisted to config.
    int32 PoseWarmupRemaining = 0;
    int32 PoseWarmupFrames = 3;
    FOnCaptureSessionComplete SessionCompleteDelegate;
    FTimerHandle AutoCaptureTimerHandle;
    FString SaveDirectory;
    
    bool bPathImageCaptureSectionExpanded = false;
    
    // Dependencies
    TWeakPtr<FVCCSimPanelSelection> SelectionManager;
    TSharedPtr<FPathGenerator> PathGenerator;
    TSharedPtr<FImageCaptureService> ImageCaptureService;
};
