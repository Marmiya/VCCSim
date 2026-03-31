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

// Forward declarations
class AFlashPawn;
class FPathGenerator;
class FImageCaptureService;
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

    void LoadFromConfigManager();
    
    // UI Construction (public for _UI.cpp)
    TSharedRef<SWidget> CreatePathConfigSection();
    TSharedRef<SWidget> CreateImageCaptureSection();
    TSharedRef<SWidget> CreatePoseFileButtons();
    TSharedRef<SWidget> CreatePoseActionButtons();
    TSharedRef<SWidget> CreateMovementButtons();
    TSharedRef<SWidget> CreateCaptureButtons();

private:
    // UI Callbacks
    FReply OnAddOrbitActorsClicked();
    FReply OnGeneratePosesClicked();
    FReply OnLoadPoseClicked();
    FReply OnSavePoseClicked();
    FReply OnTogglePathVisualizationClicked();
    FReply OnCaptureImagesClicked();
    
    // Path Configuration & Generation
    FBox ComputeCombinedBounds() const;
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
    void StopAutoCapture();
    
    // Utility
    void SaveOrbitActorList();
    void LoadOrbitActorList();
    static FString GetTimestampedFilename();

    // UI Widgets
    TSharedPtr<SListView<TSharedPtr<FString>>> OrbitActorListView;
    TSharedPtr<SNumericEntryBox<float>> OrbitMarginSpinBox;
    TSharedPtr<SNumericEntryBox<float>> OrbitStartHeightSpinBox;
    TSharedPtr<SNumericEntryBox<float>> OrbitCameraHFOVSpinBox;
    TSharedPtr<SNumericEntryBox<float>> OrbitHOverlapSpinBox;
    TSharedPtr<SNumericEntryBox<float>> OrbitVOverlapSpinBox;
    TSharedPtr<SNumericEntryBox<float>> OrbitNadirAltSpinBox;
    TSharedPtr<SNumericEntryBox<float>> OrbitNadirTiltSpinBox;
    TSharedPtr<SButton> VisualizePathButton;
    TSharedPtr<SButton> AutoCaptureButton;
    
    // State Variables
    TArray<TSharedPtr<FString>> OrbitActorListItems;
    
    float OrbitMargin = 500.0f;
    float OrbitStartHeight = 200.0f;
    float OrbitCameraHFOV = 90.0f;
    float OrbitHOverlap = 0.60f;
    float OrbitVOverlap = 0.60f;
    float OrbitNadirAlt = 500.0f;
    float OrbitNadirTiltAngle = 45.0f;
    bool bOrbitIncludeNadir = true;

    TOptional<float> OrbitMarginValue;
    TOptional<float> OrbitStartHeightValue;
    TOptional<float> OrbitCameraHFOVValue;
    TOptional<float> OrbitHOverlapValue;
    TOptional<float> OrbitVOverlapValue;
    TOptional<float> OrbitNadirAltValue;
    TOptional<float> OrbitNadirTiltValue;
    
    bool bPathVisualized = false;
    bool bPathNeedsUpdate = true;
    TWeakObjectPtr<AActor> PathVisualizationActor;
    
    bool bAutoCaptureInProgress = false;
    bool bGenerationInProgress = false;
    bool bGameViewChangedForCapture = false;
    FTimerHandle AutoCaptureTimerHandle;
    FString SaveDirectory;
    
    bool bPathImageCaptureSectionExpanded = false;
    
    // Dependencies
    TWeakPtr<FVCCSimPanelSelection> SelectionManager;
    TSharedPtr<FPathGenerator> PathGenerator;
    TSharedPtr<FImageCaptureService> ImageCaptureService;
};
