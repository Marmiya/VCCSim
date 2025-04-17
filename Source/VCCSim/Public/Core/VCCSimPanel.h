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
    // UI Elements - Removed combo boxes, added selection state toggles
    TSharedPtr<FSlateDynamicImageBrush> VCCLogoBrush;
    TSharedPtr<FSlateDynamicImageBrush> SZULogoBrush;
    
    TSharedPtr<class STextBlock> SelectedFlashPawnText;
    TSharedPtr<class STextBlock> SelectedTargetObjectText;
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
    
    // Camera availability indicators
    TSharedPtr<class STextBlock> RGBCameraAvailableText;
    TSharedPtr<class STextBlock> DepthCameraAvailableText;
    TSharedPtr<class STextBlock> SegmentationCameraAvailableText;
    
    // Camera activation checkboxes
    TSharedPtr<class SCheckBox> RGBCameraCheckBox;
    TSharedPtr<class SCheckBox> DepthCameraCheckBox;
    TSharedPtr<class SCheckBox> SegmentationCameraCheckBox;
    
    // Selection state toggles
    TSharedPtr<class SCheckBox> SelectFlashPawnToggle;
    TSharedPtr<class SCheckBox> SelectTargetToggle;
    TSharedPtr<class SCheckBox> SelectUseLimitedToggle;
    bool bSelectingFlashPawn = false;
    bool bSelectingTarget = false;
    bool bUseLimited = false;
    
    // Selected objects
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    TWeakObjectPtr<AActor> SelectedTargetObject;
    
    // Configuration
    int32 NumPoses = 50;
    float Radius = 500.0f;
    float HeightOffset = 0.0f;
    float VerticalGap = 50.0f;
    FString SaveDirectory;
    float SafeDistance = 200.0f;
    float SafeHeight = 200.0f;
    float LimitedMinX = .0f;
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
    
    // UI creation helpers
    TSharedRef<SWidget> CreatePawnSelectPanel();
    TSharedRef<SWidget> CreateCameraSelectPanel();
    TSharedRef<SWidget> CreateTargetSelectPanel();
    TSharedRef<SWidget> CreatePoseConfigPanel();
    TSharedRef<SWidget> CreateCapturePanel();
    TSharedRef<SWidget> CreateSceneAnalysisPanel();
    
    // UI callbacks
    void OnSelectFlashPawnToggleChanged(ECheckBoxState NewState);
    void OnSelectTargetToggleChanged(ECheckBoxState NewState);
    void OnUseLimitedToggleChanged(ECheckBoxState NewState);
    
    void OnRGBCameraCheckboxChanged(ECheckBoxState NewState);
    void OnDepthCameraCheckboxChanged(ECheckBoxState NewState);
    void OnSegmentationCameraCheckboxChanged(ECheckBoxState NewState);
    
    FReply OnGeneratePosesClicked();
    FReply OnCaptureImagesClicked();
    void SaveRGB(int32 PoseIndex, bool& bAnyCaptured);
    void SaveDepth(int32 PoseIndex, bool& bAnyCaptured);
    void SaveSeg(int32 PoseIndex, bool& bAnyCaptured);
    void StartAutoCapture();
    TSharedPtr<std::atomic<int32>> JobNum;
    
    // Helper functions
    void GeneratePosesAroundTarget();
    void CaptureImageFromCurrentPose();
    void CheckCameraComponents();
    void UpdateActiveCameras();
    static FString GetTimestampedFilename();
    
    // New pose file functions
    void LoadPredefinedPose();
    void SaveGeneratedPose();
    FReply OnLoadPoseClicked();
    FReply OnSavePoseClicked();

    TSharedRef<SWidget> CreateSectionHeader(const FString& Title);
    TSharedRef<SWidget> CreateSectionContent(TSharedRef<SWidget> Content);
    TSharedRef<SWidget> CreatePropertyRow(const FString& Label,
        TSharedRef<SWidget> Content);

    bool bPathVisualized = false;
    bool bPathNeedsUpdate = true;
    TWeakObjectPtr<AActor> PathVisualizationActor;
    TSharedPtr<class SButton> VisualizePathButton;
    FReply OnTogglePathVisualizationClicked();
    void UpdatePathVisualization();
    void ShowPathVisualization();
    void HidePathVisualization();

    TWeakObjectPtr<ASceneAnalysisManager> SceneAnalysisManager = nullptr;
    bool bNeedScan = true;
    bool bGenSafeZone = true;
    bool bSafeZoneVisualized = false;
    TSharedPtr<class SButton> VisualizeSafeZoneButton;
    FReply OnToggleSafeZoneVisualizationClicked();

    bool bInitCoverage = true;
    bool bGenCoverage = true;
    bool bCoverageVisualized = false;
    TSharedPtr<class SButton> VisualizeCoverageButton;
    FReply OnToggleCoverageVisualizationClicked();

    bool bAnalyzeComplexity = true;
    bool bComplexityVisualized = false;
    TSharedPtr<class SButton> VisualizeComplexityButton;
    FReply OnToggleComplexityVisualizationClicked();
};

namespace FVCCSimPanelFactory
{
    extern const FName TabId;
    void RegisterTabSpawner(FTabManager& TabManager);
}