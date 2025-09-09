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
#include "Utils/TriangleSplattingManager.h"

class FTriangleSplattingManager;
class FColmapManager;
class FVCCSimPanelSelection;
class SEditableTextBox;
class SButton;
class STextBlock;
template<typename NumericType> class SNumericEntryBox;
class SWidget;
class UStaticMesh;
struct FTriangleSplattingConfig;

/**
 * Triangle Splatting Panel - Modular panel for Triangle Splatting neural rendering functionality
 * Handles data input, camera parameters, training control, and COLMAP integration
 */
class VCCSIMEDITOR_API FVCCSimPanelTriangleSplatting
{
public:
    FVCCSimPanelTriangleSplatting();
    ~FVCCSimPanelTriangleSplatting();
    
    void Initialize(TSharedPtr<FVCCSimPanelSelection> InSelectionManager);
    void Cleanup();
    TSharedRef<SWidget> CreateTriangleSplattingPanel();
    void UpdateVisualization();
    
    // Getters for state access
    bool IsTriangleSplattingSectionExpanded() const { return bTriangleSplattingSectionExpanded; }
    void SetTriangleSplattingSectionExpanded(bool bExpanded) { bTriangleSplattingSectionExpanded = bExpanded; }
    bool IsTrainingInProgress() const { return bGSTrainingInProgress; }
    
private:
    // ============================================================================
    // UI ELEMENTS
    // ============================================================================
    
    // Panel state
    bool bTriangleSplattingSectionExpanded = true;
    
    // Data Input UI Controls
    TSharedPtr<SEditableTextBox> GSImageDirectoryTextBox;
    TSharedPtr<SEditableTextBox> GSCameraIntrinsicsFileTextBox;
    TSharedPtr<SEditableTextBox> GSPoseFileTextBox;
    TSharedPtr<SEditableTextBox> GSOutputDirectoryTextBox;
    TSharedPtr<SEditableTextBox> GSColmapDatasetTextBox;
    
    // Camera Parameter UI Controls
    TSharedPtr<SNumericEntryBox<float>> GSFOVSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> GSImageWidthSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> GSImageHeightSpinBox;
    TSharedPtr<SNumericEntryBox<float>> GSFocalLengthXSpinBox;
    TSharedPtr<SNumericEntryBox<float>> GSFocalLengthYSpinBox;
    
    // Training Parameter UI Controls
    TSharedPtr<SNumericEntryBox<int32>> GSMaxIterationsSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> GSInitPointCountSpinBox;
    
    // Training Control UI Controls
    TSharedPtr<SButton> GSTrainingToggleButton;
    TSharedPtr<SButton> GSColmapTrainingButton;
    TSharedPtr<STextBlock> GSTrainingStatusText;
    
    // ============================================================================
    // STATE VARIABLES
    // ============================================================================
    
    // Configuration
    FTriangleSplattingConfig GSConfig;
    
    // TOptional values for SpinBoxes
    TOptional<float> GSFOVValue;
    TOptional<int32> GSImageWidthValue;
    TOptional<int32> GSImageHeightValue;
    TOptional<float> GSFocalLengthXValue;
    TOptional<float> GSFocalLengthYValue;
    TOptional<int32> GSMaxIterationsValue;
    TOptional<int32> GSInitPointCountValue;
    
    // Training state
    bool bGSTrainingInProgress = false;
    bool bColmapPipelineInProgress = false;
    FString GSCurrentLoss;
    FTimerHandle GSStatusUpdateTimerHandle;
    
    // Managers
    TSharedPtr<FTriangleSplattingManager> GSTrainingManager;
    TSharedPtr<FColmapManager> ColmapManager;
    
    // Dependencies
    TWeakPtr<FVCCSimPanelSelection> SelectionManager;
    
    // ============================================================================
    // INITIALIZATION AND MANAGER OPERATIONS
    // ============================================================================
    
    void InitializeGSManager();
    void InitializeColmapManager();
    
    // ============================================================================
    // UI CONSTRUCTION METHODS
    // ============================================================================
    
    // Main UI section creators
    TSharedRef<SWidget> CreateGSDataInputSection();
    TSharedRef<SWidget> CreateGSCameraParamsSection();
    TSharedRef<SWidget> CreateGSTrainingParamsSection();
    TSharedRef<SWidget> CreateGSTrainingControlSection();
    
    // Style and layout helpers
    TSharedRef<SWidget> CreateCollapsibleSection(
        const FString& Title, TSharedRef<SWidget> Content, bool& bExpanded);
    TSharedRef<SWidget> CreateSectionHeader(const FString& Title);
    TSharedRef<SWidget> CreateSectionContent(TSharedRef<SWidget> Content);
    TSharedRef<SWidget> CreatePropertyRow(
        const FString& Label, TSharedRef<SWidget> Content);
    TSharedRef<SWidget> CreateSeparator();
    
    // Numeric property row creators
    template<typename T>
    TSharedRef<SWidget> CreateGSNumericPropertyRow(
        const FString& Label,
        TSharedPtr<SNumericEntryBox<T>>& SpinBox,
        TOptional<T>& Value,
        T MinValue,
        T MaxValue,
        T DeltaValue,
        TFunction<void(T)> OnValueChanged);
    
    // ============================================================================
    // EVENT HANDLERS
    // ============================================================================
    
    // Window management
    void* GetParentWindowHandle();
    
    // Browse dialog handlers
    FReply OnGSBrowseImageDirectoryClicked();
    FReply OnGSBrowseCameraIntrinsicsFileClicked();
    FReply OnGSBrowsePoseFileClicked();
    FReply OnGSBrowseOutputDirectoryClicked();
    FReply OnGSBrowseColmapDatasetClicked();
    
    // Camera intrinsics loading
    void OnGSCameraIntrinsicsLoaded();
    bool LoadCameraIntrinsicsFromColmap(const FString& FilePath);
    bool LoadCameraIntrinsicsFromColmapText(const FString& FilePath);
    bool LoadCameraIntrinsicsFromColmapBinary(const FString& FilePath);
    
    // Parameter change handlers
    void OnGSFOVChanged(float NewValue);
    void OnGSImageWidthChanged(int32 NewValue);
    void OnGSImageHeightChanged(int32 NewValue);
    void OnGSFocalLengthXChanged(float NewValue);
    void OnGSFocalLengthYChanged(float NewValue);
    void OnGSMaxIterationsChanged(int32 NewValue);
    void OnGSInitPointCountChanged(int32 NewValue);
    
    // Training control handlers
    FReply OnGSStartTrainingClicked();
    FReply OnGSStopTrainingClicked();
    FReply OnGSColmapTrainingClicked();
    FReply OnGSTestTransformationClicked();
    FReply OnGSExportColmapClicked();
    
    // ============================================================================
    // TRAINING OPERATIONS
    // ============================================================================
    
    void StartTriangleSplattingWithColmapData(const FString& ColmapDatasetPath);
    bool ValidateGSConfiguration();
    void ShowGSNotification(const FString& Message, bool bIsError = false);
    
    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================
    
    void ExportCamerasToPLY(const TArray<struct FCameraInfo>& CameraInfos, const FString& OutputPath);
    void SaveCameraInfoData(const TArray<struct FCameraInfo>& CameraInfos, const FString& OutputPath);
};