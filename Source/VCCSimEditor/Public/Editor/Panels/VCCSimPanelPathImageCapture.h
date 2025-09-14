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
#include "Widgets/Text/STextBlock.h"
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
    
private:
    // ============================================================================
    // UI ELEMENTS
    // ============================================================================
    
    // Path Configuration UI
    TSharedPtr<SNumericEntryBox<int32>> NumPosesSpinBox;
    TSharedPtr<SNumericEntryBox<float>> RadiusSpinBox;
    TSharedPtr<SNumericEntryBox<float>> HeightOffsetSpinBox;
    TSharedPtr<SNumericEntryBox<float>> VerticalGapSpinBox;
    
    // Path Visualization UI
    TSharedPtr<SButton> VisualizePathButton;
    
    // Image Capture UI
    TSharedPtr<SButton> AutoCaptureButton;
    
    // ============================================================================
    // STATE VARIABLES
    // ============================================================================
    
    // Path configuration parameters
    int32 NumPoses = 50;
    float Radius = 500.0f;
    float HeightOffset = 0.0f;
    float VerticalGap = 50.0f;
    
    // TOptional attributes for SpinBox values
    TOptional<int32> NumPosesValue;
    TOptional<float> RadiusValue;
    TOptional<float> HeightOffsetValue;
    TOptional<float> VerticalGapValue;
    
    // Path visualization state
    bool bPathVisualized = false;
    bool bPathNeedsUpdate = true;
    TWeakObjectPtr<AActor> PathVisualizationActor;
    
    // Auto-capture state
    bool bAutoCaptureInProgress = false;
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
};