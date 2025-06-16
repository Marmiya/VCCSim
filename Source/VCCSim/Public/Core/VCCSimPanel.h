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
#include "Editor/PropertyEditor/Public/IDetailsView.h"
#include "DataType/PointCloud.h"
#include "ProceduralMeshComponent.h"

class AFlashPawn;
class AVCCSimPath;
class USplineMeshComponent;
class ASceneAnalysisManager;

class VCCSIM_API SVCCSimPanel final : public SCompoundWidget
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
    
    // Selection UI
    TSharedPtr<class STextBlock> SelectedFlashPawnText;
    TSharedPtr<class STextBlock> SelectedTargetObjectText;
    TSharedPtr<class SCheckBox> SelectFlashPawnToggle;
    TSharedPtr<class SCheckBox> SelectTargetToggle;
    TSharedPtr<class SCheckBox> SelectUseLimitedToggle;
    
    // Configuration spinboxes
    TSharedPtr<class SNumericEntryBox<int32>> NumPosesSpinBox;
    TSharedPtr<class SNumericEntryBox<float>> RadiusSpinBox;
    TSharedPtr<class SNumericEntryBox<float>> HeightOffsetSpinBox;
    TSharedPtr<class SNumericEntryBox<float>> VerticalGapSpinBox;
    TSharedPtr<class SNumericEntryBox<float>> SafeDistanceSpinBox;
    TSharedPtr<class SNumericEntryBox<float>> SafeHeightSpinBox;
    TSharedPtr<class SNumericEntryBox<float>> LimitedMinXSpinBox;
    TSharedPtr<class SNumericEntryBox<float>> LimitedMaxXSpinBox;
    TSharedPtr<class SNumericEntryBox<float>> LimitedMinYSpinBox;
    TSharedPtr<class SNumericEntryBox<float>> LimitedMaxYSpinBox;
    TSharedPtr<class SNumericEntryBox<float>> LimitedMinZSpinBox;
    TSharedPtr<class SNumericEntryBox<float>> LimitedMaxZSpinBox;
    
    // Camera UI elements
    TSharedPtr<class SCheckBox> RGBCameraCheckBox;
    TSharedPtr<class SCheckBox> DepthCameraCheckBox;
    TSharedPtr<class SCheckBox> SegmentationCameraCheckBox;
    
    // Visualization buttons
    TSharedPtr<class SButton> VisualizePathButton;
    TSharedPtr<class SButton> VisualizeSafeZoneButton;
    TSharedPtr<class SButton> VisualizeCoverageButton;
    TSharedPtr<class SButton> VisualizeComplexityButton;
    
    // Point cloud UI elements
    TSharedPtr<SButton> LoadPointCloudButton;
    TSharedPtr<SButton> VisualizePointCloudButton;
    TSharedPtr<STextBlock> PointCloudStatusText;
    TSharedPtr<STextBlock> PointCloudColorStatusText;

    // ============================================================================
    // STATE VARIABLES
    // ============================================================================
    
    // Selection state
    bool bSelectingFlashPawn = false;
    bool bSelectingTarget = false;
    bool bUseLimited = false;
    
    // Selected objects
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    TWeakObjectPtr<AActor> SelectedTargetObject;
    
    // Path configuration
    int32 NumPoses = 50;
    float Radius = 500.0f;
    float HeightOffset = 0.0f;
    float VerticalGap = 50.0f;
    FString SaveDirectory;
    float SafeDistance = 200.0f;
    float SafeHeight = 200.0f;
    float LimitedMinX = 0.0f;
    float LimitedMaxX = 5000.0f;
    float LimitedMinY = -9500.0f;
    float LimitedMaxY = -7000.0f;
    float LimitedMinZ = -20.0f;
    float LimitedMaxZ = 2000.0f;
    
    // TOptional attributes for SpinBox values
    TOptional<int32> NumPosesValue;
    TOptional<float> RadiusValue;
    TOptional<float> HeightOffsetValue;
    TOptional<float> VerticalGapValue;
    TOptional<float> SafeDistanceValue;
    TOptional<float> SafeHeightValue;
    TOptional<float> LimitedMinXValue;
    TOptional<float> LimitedMaxXValue;
    TOptional<float> LimitedMinYValue;
    TOptional<float> LimitedMaxYValue;
    TOptional<float> LimitedMinZValue;
    TOptional<float> LimitedMaxZValue;
    
    // Camera settings
    bool bUseRGBCamera = true;
    bool bUseDepthCamera = false;
    bool bUseSegmentationCamera = false;
    
    // Available cameras on current FlashPawn
    bool bHasRGBCamera = false;
    bool bHasDepthCamera = false;
    bool bHasSegmentationCamera = false;
    
    // Auto-capture state
    bool bAutoCaptureInProgress = false;
    FTimerHandle AutoCaptureTimerHandle;
    TSharedPtr<std::atomic<int32>> JobNum;
    
    // Path visualization state
    bool bPathVisualized = false;
    bool bPathNeedsUpdate = true;
    TWeakObjectPtr<AActor> PathVisualizationActor;
    
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
    
    // Point cloud state
    TArray<FRatPoint> PointCloudData;
    TWeakObjectPtr<AActor> PointCloudActor;
    TWeakObjectPtr<UProceduralMeshComponent> PointCloudComponent;
    bool bPointCloudVisualized = false;
    bool bPointCloudLoaded = false;
    bool bPointCloudHasColors = false;
    FString LoadedPointCloudPath;
    int32 PointCloudCount = 0;
    
    // Point cloud settings
    FLinearColor DefaultPointColor = FLinearColor(1.0f, 0.5f, 0.0f, 1.0f);
    float PointSize = 100.f;

    // ============================================================================
    // INITIALIZATION AND CLEANUP
    // ============================================================================
    
    void LoadLogoImages();
    void InitializeSceneAnalysisManager();
    void CreateMainLayout();

    // ============================================================================
    // SELECTION MANAGEMENT
    // ============================================================================
    
    void OnSelectFlashPawnToggleChanged(ECheckBoxState NewState);
    void OnSelectTargetToggleChanged(ECheckBoxState NewState);
    void OnUseLimitedToggleChanged(ECheckBoxState NewState);

    // ============================================================================
    // CAMERA MANAGEMENT
    // ============================================================================
    
    void OnRGBCameraCheckboxChanged(ECheckBoxState NewState);
    void OnDepthCameraCheckboxChanged(ECheckBoxState NewState);
    void OnSegmentationCameraCheckboxChanged(ECheckBoxState NewState);
    void CheckCameraComponents();
    void UpdateActiveCameras();

    // ============================================================================
    // POSE GENERATION AND MANAGEMENT
    // ============================================================================
    
    FReply OnGeneratePosesClicked();
    void GeneratePosesAroundTarget();
    void LoadPredefinedPose();
    void SaveGeneratedPose();
    FReply OnLoadPoseClicked();
    FReply OnSavePoseClicked();

    // ============================================================================
    // PATH VISUALIZATION
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
    void StartAutoCapture();

    // ============================================================================
    // SCENE ANALYSIS OPERATIONS
    // ============================================================================
    
    FReply OnToggleSafeZoneVisualizationClicked();
    FReply OnToggleCoverageVisualizationClicked();
    FReply OnToggleComplexityVisualizationClicked();

    // ============================================================================
    // POINT CLOUD OPERATIONS
    // ============================================================================
    
    FReply OnLoadPointCloudClicked();
    FReply OnTogglePointCloudVisualizationClicked();
    void CreateProceduralPointCloudVisualization();
    void ClearPointCloudVisualization();
    void GeneratePointCloudMesh(TArray<FVector>& Vertices, 
                               TArray<int32>& Triangles, 
                               TArray<FVector>& Normals,
                               TArray<FVector2D>& UVs,
                               TArray<FColor>& VertexColors);
    void ApplyVertexColorMaterial(UProceduralMeshComponent* MeshComponent);
    bool TryApplyFallbackMaterial(UProceduralMeshComponent* MeshComponent);
    void CreateBasicPointCloudMaterial(UProceduralMeshComponent* MeshComponent);

    // ============================================================================
    // UI CONSTRUCTION HELPERS
    // ============================================================================
    
    // Main panel creators
    TSharedRef<SWidget> CreateLogoPanel();
    TSharedRef<SWidget> CreatePawnSelectPanel();
    TSharedRef<SWidget> CreateCameraSelectPanel();
    TSharedRef<SWidget> CreateTargetSelectPanel();
    TSharedRef<SWidget> CreatePoseConfigPanel();
    TSharedRef<SWidget> CreateCapturePanel();
    TSharedRef<SWidget> CreateSceneAnalysisPanel();
    TSharedRef<SWidget> CreatePointCloudPanel();
    
    // Camera UI helpers
    TSharedRef<SWidget> CreateCameraStatusRow();
    TSharedRef<SWidget> CreateCameraStatusBox(
        const FString& CameraName,
        TFunction<bool()> HasCameraFunc,
        TFunction<ECheckBoxState()> CheckBoxStateFunc,
        TFunction<void(ECheckBoxState)> OnCheckBoxChangedFunc);
    
    // Scene analysis UI helpers
    TSharedRef<SWidget> CreateLimitedRegionControls();
    TSharedRef<SWidget> CreateSceneOperationButtons();
    TSharedRef<SWidget> CreateSafeZoneButtons();
    TSharedRef<SWidget> CreateCoverageButtons();
    TSharedRef<SWidget> CreateComplexityButtons();
    
    // Button group creators
    TSharedRef<SWidget> CreatePoseFileButtons();
    TSharedRef<SWidget> CreatePoseActionButtons();
    TSharedRef<SWidget> CreateMovementButtons();
    TSharedRef<SWidget> CreateCaptureButtons();
    TSharedRef<SWidget> CreatePointCloudButtons();
    
    // Style and layout helpers
    TSharedRef<SWidget> CreateSectionHeader(const FString& Title);
    TSharedRef<SWidget> CreateSectionContent(TSharedRef<SWidget> Content);
    TSharedRef<SWidget> CreatePropertyRow(const FString& Label, TSharedRef<SWidget> Content);
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