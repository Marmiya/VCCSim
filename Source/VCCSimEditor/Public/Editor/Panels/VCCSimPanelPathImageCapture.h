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
#include "Widgets/Views/SListView.h"
#include "Engine/TimerHandle.h"
#include <atomic>

class AFlashPawn;
class FVCCSimPanelSelection;

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

    // UI state access for persistence
    bool IsPathImageCaptureSectionExpanded() const { return bPathImageCaptureSectionExpanded; }
    void SetPathImageCaptureSectionExpanded(bool bExpanded) { bPathImageCaptureSectionExpanded = bExpanded; }
    
private:
    // ============================================================================
    // UI ELEMENTS
    // ============================================================================
    
    // Target actor list for bounding-box orbit
    TSharedPtr<SListView<TSharedPtr<FString>>> OrbitActorListView;
    TArray<TSharedPtr<FString>>               OrbitActorListItems;

    // Orbit path parameter spinboxes
    TSharedPtr<SNumericEntryBox<float>> OrbitMarginSpinBox;
    TSharedPtr<SNumericEntryBox<float>> OrbitStartHeightSpinBox;
    TSharedPtr<SNumericEntryBox<float>> OrbitCameraHFOVSpinBox;
    TSharedPtr<SNumericEntryBox<float>> OrbitHOverlapSpinBox;
    TSharedPtr<SNumericEntryBox<float>> OrbitVOverlapSpinBox;
    TSharedPtr<SNumericEntryBox<float>> OrbitNadirAltSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> OrbitNadirCountSpinBox;
    
    // Path Visualization UI
    TSharedPtr<SButton> VisualizePathButton;
    
    // Image Capture UI
    TSharedPtr<SButton> AutoCaptureButton;
    
    // ============================================================================
    // STATE VARIABLES
    // ============================================================================
    
    // Orbit path configuration parameters
    float OrbitMargin       = 500.0f;
    float OrbitStartHeight  = 200.0f;
    float OrbitCameraHFOV   = 90.0f;
    float OrbitHOverlap     = 0.80f;
    float OrbitVOverlap     = 0.80f;
    float OrbitNadirAlt     = 500.0f;
    int32 OrbitNadirCount   = 4;
    bool  bOrbitIncludeNadir = true;

    // TOptional attributes for SpinBox values
    TOptional<float> OrbitMarginValue;
    TOptional<float> OrbitStartHeightValue;
    TOptional<float> OrbitCameraHFOVValue;
    TOptional<float> OrbitHOverlapValue;
    TOptional<float> OrbitVOverlapValue;
    TOptional<float> OrbitNadirAltValue;
    TOptional<int32> OrbitNadirCountValue;
    
    // Path visualization state
    bool bPathVisualized = false;
    bool bPathNeedsUpdate = true;
    TWeakObjectPtr<AActor> PathVisualizationActor;
    
    // Auto-capture state
    bool bAutoCaptureInProgress = false;
    bool bGameViewChangedForCapture = false;
    FTimerHandle AutoCaptureTimerHandle;
    TSharedPtr<std::atomic<int32>> JobNum;
    FString SaveDirectory;
    
    // Panel state
    bool bPathImageCaptureSectionExpanded = false;
    
    // Selection manager reference
    TWeakPtr<FVCCSimPanelSelection> SelectionManager;
    
    // ============================================================================
    // PATH CONFIGURATION OPERATIONS
    // ============================================================================

    FReply OnAddOrbitActorsClicked();
    FBox  ComputeCombinedBounds() const;
    FReply OnGeneratePosesClicked();
    void GeneratePosesAroundTarget();
    FReply OnLoadPoseClicked();
    FReply OnSavePoseClicked();
    void LoadPredefinedPose();
    void SaveGeneratedPose();
    
    // ============================================================================
    // PATH VISUALIZATION OPERATIONS
    // ============================================================================
    
    FReply OnTogglePathVisualizationClicked();
    void UpdatePathVisualization();
    void ShowPathVisualization();
    void HidePathVisualization();
    
    // ============================================================================
    // IMAGE CAPTURE OPERATIONS
    // ============================================================================
    
    FReply OnCaptureImagesClicked();
    void CaptureImageFromCurrentPose();
    void SaveRGB(int32 PoseIndex, bool& bAnyCaptured);
    void SaveDepth(int32 PoseIndex, bool& bAnyCaptured);
    void SaveSeg(int32 PoseIndex, bool& bAnyCaptured);
    void SaveNormal(int32 PoseIndex, bool& bAnyCaptured);
    void StartAutoCapture();
    void StopAutoCapture();
    
    // ============================================================================
    // UI CONSTRUCTION HELPERS
    // ============================================================================
    
    TSharedRef<SWidget> CreatePathConfigSection();
    TSharedRef<SWidget> CreateImageCaptureSection();
    
    // Button group creators
    TSharedRef<SWidget> CreatePoseFileButtons();
    TSharedRef<SWidget> CreatePoseActionButtons();
    TSharedRef<SWidget> CreateMovementButtons();
    TSharedRef<SWidget> CreateCaptureButtons();
    
    // Utility functions
    static FString GetTimestampedFilename();
    void SaveOrbitActorList();
    void LoadOrbitActorList();
};