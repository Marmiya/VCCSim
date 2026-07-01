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
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/Text/STextBlock.h"
#include "Engine/TimerHandle.h"
#include "Utils/CaptureReuseManifest.h"
#include "Utils/CaptureSessionCheckpoint.h"
#include <atomic>

// Forward declarations
class AFlashPawn;
class FPathGenerator;
class FImageCaptureService;
class FVCCSimPanelSelection;
class FLightingManager;
class FGTMaterialExporter;

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
     * @param bRgbOnly Capture only RGB and skip the lighting-independent GT channels
     *        (Normal/BaseColor/MatProps) — used when those are reused from another capture.
     * @param OnComplete Fired with true on success, false on cancel/failure.
     * @return false if a session is already running or prerequisites are missing.
     */
    bool StartCaptureSession(
        const FString& TargetDirectory,
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

    /** True when every pose of the selected FlashPawn's path already has all expected channel files
     *  present (non-empty) in Dir. Used by dataset-capture resume to skip windows that are fully done.
     *  bRgbOnly selects the channel set (RGB-only vs the full dataset GT channels). */
    bool IsCaptureWindowComplete(const FString& Dir, bool bRgbOnly) const;

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
    TSharedRef<SWidget> CreateDatasetConfigSection();
    TSharedRef<SWidget> CreateLightingScheduleSection();
    TSharedRef<SWidget> CreateLightingEntry(int32 Index);
    TSharedRef<SWidget> CreateSunPositionCalculatorWidget();
    TSharedRef<SWidget> CreateDatasetCaptureSection();

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

    // ============================================================================
    // DATASET CONFIGURATION
    // ============================================================================

    FReply OnBrowseOutputDirClicked();

    // ============================================================================
    // LIGHTING SCHEDULE
    // ============================================================================

    FReply OnApplyLightingClicked(int32 Index);
    FReply OnCalculateSunPositionClicked();
    FReply OnFillFromSunPositionClicked();

    // ============================================================================
    // DATASET CAPTURE
    // ============================================================================

    FReply OnCaptureDatasetClicked();
    bool DecideAndStartCapture(const FString& CaptureDir);
    void StartNextBatchCapture();
    void OnDatasetCaptureFinished(bool bSuccess, FString CaptureDirectory);
    FString GetDatasetCapturesRoot() const;
    FString MakeNextCaptureDirectory() const;

    /** Continue the last interrupted dataset capture (from <captures>/capture_session.json): skip
     *  windows/poses already on disk, re-shoot the last present pose, finish the rest. */
    FReply OnResumeCaptureClicked();
    /** True when a resumable checkpoint exists and no capture is currently running. */
    bool HasResumableCapture() const;

    /** Export the enabled target actors' GT mesh (gt_materials) for Dataset Capture's "Mesh"
     *  checkbox, reused across lighting windows via the capture reuse manifest. */
    bool StartGTMaterialExport(const FString& BaseDir);

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

    // Dataset Configuration widgets
    TSharedPtr<SEditableTextBox> OutputDirTextBox;
    TSharedPtr<SEditableTextBox> SceneNameTextBox;

    // Lighting Schedule widgets
    static constexpr int32 NumLightingConditions = 5;
    TSharedPtr<SNumericEntryBox<float>> LightingElevationSpinBox[NumLightingConditions];
    TSharedPtr<SNumericEntryBox<float>> LightingAzimuthSpinBox[NumLightingConditions];
    TOptional<float> LightingElevationValue[NumLightingConditions];
    TOptional<float> LightingAzimuthValue[NumLightingConditions];

    // Sun Position Calculator widgets
    TSharedPtr<SNumericEntryBox<float>> SunCalcLatSpinBox;
    TSharedPtr<SNumericEntryBox<float>> SunCalcLonSpinBox;
    TSharedPtr<SNumericEntryBox<float>> SunCalcTZSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> SunCalcYearSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> SunCalcMonthSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> SunCalcDaySpinBox;
    TSharedPtr<SNumericEntryBox<int32>> SunCalcHourSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> SunCalcMinuteSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> SunCalcFillSlotSpinBox;

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
    // Resume support: per-pose completion mask for the session's directory, computed at StartCaptureSession
    // from files already on disk. Poses marked true are skipped (no warm-up, no capture); the rest are
    // captured. Empty when not resuming (fresh directory => all poses captured).
    TArray<bool> SessionCompleted;
    FOnCaptureSessionComplete SessionCompleteDelegate;
    FTimerHandle AutoCaptureTimerHandle;
    FString SaveDirectory;
    
    bool bPathImageCaptureSectionExpanded = false;

    // ============================================================================
    // DATASET CONFIGURATION / LIGHTING SCHEDULE / DATASET CAPTURE STATE
    // (folded in from the retired TexEnhancer panel)
    // ============================================================================

    FString OutputDirectory;
    FString SceneName = TEXT("Scene_A");

    bool bLightingScheduleExpanded = false;
    float LightingElevation[NumLightingConditions] = { 20.f, 70.f, 35.f, 85.f, 15.f };
    float LightingAzimuth[NumLightingConditions]   = { 30.f, 110.f, 190.f, 250.f, 320.f };
    bool  bLightingSelected[NumLightingConditions] = {};

    float SunCalcLatitude  = 22.52933f;
    float SunCalcLongitude = 113.94092f;
    float SunCalcTimeZone  = 8.0f;
    int32 SunCalcYear      = 2026;
    int32 SunCalcMonth     = 3;
    int32 SunCalcDay       = 20;
    int32 SunCalcHour      = 10;
    int32 SunCalcMinute    = 0;
    int32 SunCalcFillSlot  = 1;
    float SunCalcElevation = 0.f;
    float SunCalcAzimuth   = 0.f;

    TOptional<float> SunCalcLatValue;
    TOptional<float> SunCalcLonValue;
    TOptional<float> SunCalcTZValue;
    TOptional<int32> SunCalcYearValue;
    TOptional<int32> SunCalcMonthValue;
    TOptional<int32> SunCalcDayValue;
    TOptional<int32> SunCalcHourValue;
    TOptional<int32> SunCalcMinuteValue;
    TOptional<int32> SunCalcFillSlotValue;

    int32 GTTextureResolution = 2048;
    bool  bOutputImages = true;
    bool  bOutputMesh   = true;
    bool  bUseCaptureReuse = true;

    bool bDatasetCaptureInProgress = false;
    bool bBatchCapture = false;
    FString BatchCaptureTimestamp;
    TArray<int32> LightingCaptureQueue;

    FString PendingCaptureName;
    FCaptureReuseEntry PendingReuseEntry;

    // In-memory copy of the on-disk resume checkpoint for the active dataset run. Written to
    // <captures>/capture_session.json when a run starts (and updated with each window's channel mode),
    // and cleared when the whole run finishes. Survives Stop; the on-disk copy survives an editor crash.
    FCaptureSessionCheckpoint ActiveCheckpoint;

    TSharedPtr<FLightingManager> LightingManager;
    TSharedPtr<FGTMaterialExporter> GTMaterialExporter;

    // Dependencies
    TWeakPtr<FVCCSimPanelSelection> SelectionManager;
    TSharedPtr<FPathGenerator> PathGenerator;
    TSharedPtr<FImageCaptureService> ImageCaptureService;
};
