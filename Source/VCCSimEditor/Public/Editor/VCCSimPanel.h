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
#include "Widgets/Docking/SDockTab.h"
#include "Widgets/SCompoundWidget.h"
#include "Widgets/Layout/SExpandableArea.h"
#include "Editor/PropertyEditor/Public/IDetailsView.h"
#include "ProceduralMeshComponent.h"
#include "Misc/Paths.h"
#include "AssetRegistry/AssetData.h"
#include "VCCSimPanel.generated.h"

class AFlashPawn;
class AVCCSimPath;
class USplineMeshComponent;
class ASceneAnalysisManager;
class UStaticMeshComponent;
class FTriangleSplattingManager;
class FColmapManager;
class FVCCSimPanelPointCloud;
class FVCCSimPanelSelection;
class FVCCSimPanelPathImageCapture;
struct FCameraInfo;

/**
 * Triangle Splatting configuration structure
 */
USTRUCT()
struct VCCSIMEDITOR_API FTriangleSplattingConfig
{
    GENERATED_BODY()

    // Input paths
    FString ImageDirectory;
    FString PoseFilePath;
    FString OutputDirectory;
    FString ColmapDatasetPath;
    
    // Mesh configuration
    TWeakObjectPtr<UStaticMesh> SelectedMesh;
    bool bUseMeshInitialization = true;
    
    // Camera parameters (user inputs)
    float FOVDegrees = 90.0f;
    int32 ImageWidth = 1297;
    int32 ImageHeight = 840;
    
    // Camera intrinsics (optional - if provided,
    // fx/fy are used directly instead of FOV calculation)
    float FocalLengthX = 961.22f;  // fx - horizontal focal length in pixels
    float FocalLengthY = 963.089f;  // fy - vertical focal length in pixels
    
    // Training parameters
    int32 MaxIterations = 30000;
    int32 InitPointCount = 100000;
    
    // Constructor
    FTriangleSplattingConfig()
    {
        ImageDirectory = TEXT("");
        PoseFilePath = TEXT("");
        OutputDirectory = FPaths::ProjectSavedDir() / TEXT("TriangleSplatting");
        ColmapDatasetPath = TEXT("");
    }
};

class VCCSIMEDITOR_API SVCCSimPanel final : public SCompoundWidget
{
public:
    SLATE_BEGIN_ARGS(SVCCSimPanel) {}
    SLATE_END_ARGS()

    ~SVCCSimPanel();

    void Construct(const FArguments& InArgs);
    
    // Update signature to match UE's selection event
    void OnSelectionChanged(UObject* Object);

private:
    // ============================================================================
    // UI ELEMENTS
    // ============================================================================
    
    // Logo brushes
    TSharedPtr<FSlateDynamicImageBrush> VCCLogoBrush;
    TSharedPtr<FSlateDynamicImageBrush> SZULogoBrush;
    
    // Expandable area states
    bool bSceneAnalysisSectionExpanded = false;  
    bool bTriangleSplattingSectionExpanded = true;
    
    TSharedPtr<class SCheckBox> SelectUseLimitedToggle;
    
    // Configuration spinboxes
    TSharedPtr<class SNumericEntryBox<float>> SafeDistanceSpinBox;
    TSharedPtr<class SNumericEntryBox<float>> SafeHeightSpinBox;
    TSharedPtr<class SNumericEntryBox<float>> LimitedMinXSpinBox;
    TSharedPtr<class SNumericEntryBox<float>> LimitedMaxXSpinBox;
    TSharedPtr<class SNumericEntryBox<float>> LimitedMinYSpinBox;
    TSharedPtr<class SNumericEntryBox<float>> LimitedMaxYSpinBox;
    TSharedPtr<class SNumericEntryBox<float>> LimitedMinZSpinBox;
    TSharedPtr<class SNumericEntryBox<float>> LimitedMaxZSpinBox;
    
    
    // Visualization buttons
    TSharedPtr<class SButton> VisualizeSafeZoneButton;
    TSharedPtr<class SButton> VisualizeCoverageButton;
    TSharedPtr<class SButton> VisualizeComplexityButton;

    // Triangle Splatting UI elements (simplified with UE official asset picker)
    TSharedPtr<SEditableTextBox> GSImageDirectoryTextBox;
    TSharedPtr<SEditableTextBox> GSPoseFileTextBox;
    TSharedPtr<SEditableTextBox> GSOutputDirectoryTextBox;
    TSharedPtr<SEditableTextBox> GSColmapDatasetTextBox;
    TSharedPtr<SNumericEntryBox<float>> GSFOVSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> GSImageWidthSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> GSImageHeightSpinBox;
    TSharedPtr<SNumericEntryBox<float>> GSFocalLengthXSpinBox;
    TSharedPtr<SNumericEntryBox<float>> GSFocalLengthYSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> GSMaxIterationsSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> GSInitPointCountSpinBox;
    TSharedPtr<SButton> GSTrainingToggleButton;
    TSharedPtr<SButton> GSColmapTrainingButton;
    TSharedPtr<STextBlock> GSTrainingStatusText;

    // ============================================================================
    // STATE VARIABLES
    // ============================================================================
    
    // Path configuration state
    bool bUseLimited = false;
    
    // Scene analysis configuration
    float SafeDistance = 200.0f;
    float SafeHeight = 200.0f;
    float LimitedMinX = 0.0f;
    float LimitedMaxX = 5000.0f;
    float LimitedMinY = -9500.0f;
    float LimitedMaxY = -7000.0f;
    float LimitedMinZ = -20.0f;
    float LimitedMaxZ = 2000.0f;
    
    // TOptional attributes for SpinBox values
    TOptional<float> SafeDistanceValue;
    TOptional<float> SafeHeightValue;
    TOptional<float> LimitedMinXValue;
    TOptional<float> LimitedMaxXValue;
    TOptional<float> LimitedMinYValue;
    TOptional<float> LimitedMaxYValue;
    TOptional<float> LimitedMinZValue;
    TOptional<float> LimitedMaxZValue;
    
    
    // Scene analysis state
    TWeakObjectPtr<ASceneAnalysisManager> SceneAnalysisManager = nullptr;
    bool bNeedScan = true;
    bool bGenSafeZone = true;
    bool bSafeZoneVisualized = false;
    bool bInitCoverage = true;
    bool bGenCoverage = true;
    bool bCoverageVisualized = false;
    bool bAnalyzeComplexity = true;
    bool bComplexityVisualized = false;

    // Triangle Splatting state (simplified)
    FTriangleSplattingConfig GSConfig;
    bool bGSTrainingInProgress = false;
    TSharedPtr<FTriangleSplattingManager> GSTrainingManager;
    FTimerHandle GSStatusUpdateTimerHandle;
    
    // Training monitoring
    FString GSCurrentLoss;
    
    // COLMAP pipeline state
    bool bColmapPipelineInProgress = false;
    TSharedPtr<FColmapManager> ColmapManager;
    
    // Panel managers
    TSharedPtr<FVCCSimPanelPointCloud> PointCloudManager;
    TSharedPtr<FVCCSimPanelSelection> SelectionManager;
    TSharedPtr<FVCCSimPanelPathImageCapture> PathImageCaptureManager;
    
    // TOptional attributes for Triangle Splatting SpinBox values
    TOptional<float> GSFOVValue;
    TOptional<int32> GSImageWidthValue;
    TOptional<int32> GSImageHeightValue;
    TOptional<float> GSFocalLengthXValue;
    TOptional<float> GSFocalLengthYValue;
    TOptional<int32> GSMaxIterationsValue;
    TOptional<int32> GSInitPointCountValue;

    // ============================================================================
    // INITIALIZATION AND CLEANUP
    // ============================================================================
    
    void LoadLogoImages();
    void InitializeSceneAnalysisManager();
    void CreateMainLayout();
    void OnUseLimitedToggleChanged(ECheckBoxState NewState);
    

    // ============================================================================
    // SCENE ANALYSIS OPERATIONS
    // ============================================================================
    
    FReply OnToggleSafeZoneVisualizationClicked();
    FReply OnToggleCoverageVisualizationClicked();
    FReply OnToggleComplexityVisualizationClicked();

    // ============================================================================
    // TRIANGLE SPLATTING OPERATIONS (implemented in VCCSimPanel_gs.cpp)
    // ============================================================================
    
    // Initialization (simplified)
    void InitializeGSManager();
    void InitializeColmapManager();
    
    // UI event handlers (simplified - removed mesh management functions)
    FReply OnGSBrowseImageDirectoryClicked();
    FReply OnGSBrowsePoseFileClicked();
    FReply OnGSBrowseOutputDirectoryClicked();
    FReply OnGSBrowseColmapDatasetClicked();
    void OnGSFOVChanged(float NewValue);
    void OnGSImageWidthChanged(int32 NewValue);
    void OnGSImageHeightChanged(int32 NewValue);
    void OnGSFocalLengthXChanged(float NewValue);
    void OnGSFocalLengthYChanged(float NewValue);
    void OnGSMaxIterationsChanged(int32 NewValue);
    void OnGSInitPointCountChanged(int32 NewValue);
    
    // Training control
    FReply OnGSStartTrainingClicked();
    FReply OnGSStopTrainingClicked();
    FReply OnGSColmapTrainingClicked();
    void StartTriangleSplattingWithColmapData(const FString& ColmapDatasetPath);
    FReply OnGSTestTransformationClicked();
    FReply OnGSExportColmapClicked();
    bool ValidateGSConfiguration();
    void ShowGSNotification(const FString& Message, bool bIsError = false);
    
    // Test and validation helpers
    void ExportCamerasToPLY(const TArray<FCameraInfo>& CameraInfos, const FString& OutputPath);
    void SaveCameraInfoData(const TArray<FCameraInfo>& CameraInfos, const FString& OutputPath);
    
    // Window management helper
    void* GetParentWindowHandle();

    // ============================================================================
    // UI CONSTRUCTION HELPERS
    // ============================================================================
    
    // Main panel creators
    TSharedRef<SWidget> CreateLogoPanel();
    TSharedRef<SWidget> CreateSceneAnalysisPanel();
    TSharedRef<SWidget> CreatePointCloudPanel();
    TSharedRef<SWidget> CreateTriangleSplattingPanel();
    
    
    // Scene analysis UI helpers
    TSharedRef<SWidget> CreateLimitedRegionControls();
    TSharedRef<SWidget> CreateSceneOperationButtons();
    TSharedRef<SWidget> CreateSafeZoneButtons();
    TSharedRef<SWidget> CreateCoverageButtons();
    TSharedRef<SWidget> CreateComplexityButtons();
    
    
    // Triangle Splatting UI creators (implemented in VCCSimPanel_gs.cpp)
    TSharedRef<SWidget> CreateGSDataInputSection();
    TSharedRef<SWidget> CreateGSCameraParamsSection();
    TSharedRef<SWidget> CreateGSTrainingParamsSection();
    TSharedRef<SWidget> CreateGSTrainingControlSection();
    
    template<typename T>
    TSharedRef<SWidget> CreateGSNumericPropertyRow(
        const FString& Label,
        TSharedPtr<SNumericEntryBox<T>>& SpinBox,
        TOptional<T>& Value,
        T MinValue,
        T MaxValue,
        T DeltaValue,
        TFunction<void(T)> OnValueChanged);
    
    // Style and layout helpers
    TSharedRef<SWidget> CreateCollapsibleSection(
        const FString& Title, TSharedRef<SWidget> Content, bool& bExpanded);
    TSharedRef<SWidget> CreateSectionHeader(const FString& Title);
    TSharedRef<SWidget> CreateSectionContent(TSharedRef<SWidget> Content);
    TSharedRef<SWidget> CreatePropertyRow(
        const FString& Label, TSharedRef<SWidget> Content);
    TSharedRef<SWidget> CreateSeparator();
    
    // Numeric property row creators
    TSharedRef<SWidget> CreateNumericPropertyRowInt32(
        const FString& Label,
        TSharedPtr<SNumericEntryBox<int32>>& SpinBox,
        TOptional<int32>& Value,
        int32& ActualVariable,
        int32 MinValue,
        int32 DeltaValue);
        
    TSharedRef<SWidget> CreateNumericPropertyRowFloat(
        const FString& Label,
        TSharedPtr<SNumericEntryBox<float>>& SpinBox,
        TOptional<float>& Value,
        float& ActualVariable,
        float MinValue,
        float DeltaValue);

    template<typename T>
    TSharedRef<SWidget> CreateNumericPropertyRow(
        const FString& Label,
        TSharedPtr<SNumericEntryBox<T>>& SpinBox,
        TOptional<T>& Value,
        T MinValue,
        T DeltaValue);

    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================

    static FString GetTimestampedFilename();
};

namespace FVCCSimPanelFactory
{
    extern const FName TabId;
    void RegisterTabSpawner(FTabManager& TabManager);
}