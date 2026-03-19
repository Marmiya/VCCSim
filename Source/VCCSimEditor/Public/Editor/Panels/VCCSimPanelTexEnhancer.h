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
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SNumericEntryBox.h"
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Views/SListView.h"
#include "Engine/TimerHandle.h"
#include "Dom/JsonObject.h"

class FVCCSimPanelSelection;
class AStaticMeshActor;
class UStaticMesh;
class UTexture2D;

class VCCSIMEDITOR_API FVCCSimPanelTexEnhancer
{
public:
    FVCCSimPanelTexEnhancer();
    ~FVCCSimPanelTexEnhancer();

    void Initialize();
    void Cleanup();
    TSharedRef<SWidget> CreateTexEnhancerPanel();

    void SetSelectionManager(TSharedPtr<FVCCSimPanelSelection> InSelectionManager);

    bool IsTexEnhancerSectionExpanded() const { return bSectionExpanded; }
    void SetTexEnhancerSectionExpanded(bool bExpanded) { bSectionExpanded = bExpanded; }

    void LoadFromConfigManager();

private:
    // ============================================================================
    // UI ELEMENTS
    // ============================================================================

    TSharedPtr<SEditableTextBox> OutputDirTextBox;
    TSharedPtr<SEditableTextBox> SceneNameTextBox;

    TSharedPtr<SNumericEntryBox<float>> SetAElevationSpinBox[4];
    TSharedPtr<SNumericEntryBox<float>> SetAAzimuthSpinBox[4];
    TSharedPtr<SNumericEntryBox<float>> SetBElevationSpinBox[4];
    TSharedPtr<SNumericEntryBox<float>> SetBAzimuthSpinBox[4];
    TOptional<float> SetAElevationValue[4];
    TOptional<float> SetAAzimuthValue[4];
    TOptional<float> SetBElevationValue[4];
    TOptional<float> SetBAzimuthValue[4];

    TSharedPtr<SNumericEntryBox<float>> SphereRadiusSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> SphereRingsSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> PosesPerRingSpinBox;
    TSharedPtr<SNumericEntryBox<float>> NadirAltitudeSpinBox;
    TSharedPtr<SNumericEntryBox<float>> FrontOverlapSpinBox;
    TSharedPtr<SNumericEntryBox<float>> SideOverlapSpinBox;
    TOptional<float> SphereRadiusValue;
    TOptional<int32> SphereRingsValue;
    TOptional<int32> PosesPerRingValue;
    TOptional<float> NadirAltitudeValue;
    TOptional<float> FrontOverlapValue;
    TOptional<float> SideOverlapValue;

    TSharedPtr<SNumericEntryBox<float>> CaptureFOVSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> CaptureWidthSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> CaptureHeightSpinBox;
    TOptional<float> CaptureFOVValue;
    TOptional<int32> CaptureWidthValue;
    TOptional<int32> CaptureHeightValue;

    TSharedPtr<SEditableTextBox> TexEnhancerScriptTextBox;
    TSharedPtr<SEditableTextBox> EstimatedMaterialsDirTextBox;
    TSharedPtr<STextBlock> LightingStatusTextBlock;
    TSharedPtr<STextBlock> StatusTextBlock;
    TSharedPtr<STextBlock> EvalResultsTextBlock;

    TSharedPtr<SListView<TSharedPtr<FString>>> GTActorListView;
    TArray<TSharedPtr<FString>> GTActorListItems;
    TSharedPtr<SNumericEntryBox<int32>> GTTexResSpinBox;
    TOptional<int32> GTTexResValue;

    TSharedPtr<SNumericEntryBox<float>> SunCalcLatSpinBox;
    TSharedPtr<SNumericEntryBox<float>> SunCalcLonSpinBox;
    TSharedPtr<SNumericEntryBox<float>> SunCalcTZSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> SunCalcYearSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> SunCalcMonthSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> SunCalcDaySpinBox;
    TSharedPtr<SNumericEntryBox<int32>> SunCalcHourSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> SunCalcMinuteSpinBox;
    TSharedPtr<SNumericEntryBox<int32>> SunCalcFillSlotSpinBox;
    TSharedPtr<SNumericEntryBox<float>> DayCycleSpeedSpinBox;
    TSharedPtr<STextBlock>              SunCalcResultTextBlock;
    TOptional<float>                    DayCycleSpeedValue;

    // ============================================================================
    // STATE VARIABLES
    // ============================================================================

    bool bSectionExpanded = false;

    static constexpr int32 MaxLightingEntries = 4;
    int32 NumLightingSetA = 3;
    int32 NumLightingSetB = 2;

    float SetAElevation[MaxLightingEntries] = { 35.f, 65.f, 50.f, 0.f };
    float SetAAzimuth[MaxLightingEntries]   = { 120.f, 150.f, 90.f, 0.f };

    float SetBElevation[MaxLightingEntries] = { 25.f, 75.f, 0.f, 0.f };
    float SetBAzimuth[MaxLightingEntries]   = { 200.f, 60.f, 0.f, 0.f };

    float SphereRadius  = 3000.f;
    int32 SphereRings   = 3;
    int32 PosesPerRing  = 18;
    float NadirAltitude = 5000.f;
    float FrontOverlap  = 0.80f;
    float SideOverlap   = 0.70f;

    float CaptureFOVDegrees = 90.f;
    int32 CaptureWidth      = 1920;
    int32 CaptureHeight     = 1080;

    FString OutputDirectory;
    FString SceneName = TEXT("Scene_A");
    FString TexEnhancerScriptPath;
    FString EstimatedMaterialsDir;
    FString StatusMessage;

    bool bCaptureInProgress  = false;
    bool bPipelineInProgress = false;
    bool bEvalInProgress     = false;
    bool bSetBLocked         = false;

    int32 GTTextureResolution = 1024;

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

    bool  bDayCycleActive   = false;
    float DayCycleSpeed     = 10.f;
    float DayCycleSimMinute = 0.f;
    FTimerHandle DayCycleTimerHandle;

    TOptional<float> SunCalcLatValue;
    TOptional<float> SunCalcLonValue;
    TOptional<float> SunCalcTZValue;
    TOptional<int32> SunCalcYearValue;
    TOptional<int32> SunCalcMonthValue;
    TOptional<int32> SunCalcDayValue;
    TOptional<int32> SunCalcHourValue;
    TOptional<int32> SunCalcMinuteValue;
    TOptional<int32> SunCalcFillSlotValue;

    FTimerHandle StatusTimerHandle;
    FProcHandle PipelineProcHandle;

    TWeakPtr<FVCCSimPanelSelection> SelectionManager;

    // ============================================================================
    // SECTION 1: DATASET CONFIGURATION
    // ============================================================================

    TSharedRef<SWidget> CreateDatasetConfigSection();
    FReply OnBrowseOutputDirClicked();

    // ============================================================================
    // SECTION 2: LIGHTING SCHEDULE
    // ============================================================================

    TSharedRef<SWidget> CreateLightingScheduleSection();
    TSharedRef<SWidget> CreateSetALightingEntry(int32 Index);
    TSharedRef<SWidget> CreateSetBLightingEntry(int32 Index);
    TSharedRef<SWidget> CreateSunPositionCalculatorWidget();
    FReply OnApplySetALightingClicked(int32 Index);
    FReply OnApplySetBLightingClicked(int32 Index);
    FReply OnCalculateSunPositionClicked();
    FReply OnFillSetAFromSunPositionClicked();
    FReply OnFillSetBFromSunPositionClicked();
    FReply OnToggleDayCycleClicked();
    void   TickDayCycle();
    void ApplyLightingCondition(float ElevationDeg, float AzimuthDeg, bool bMarkDirty = true);

    // ============================================================================
    // SECTION 3: CAPTURE PROTOCOL
    // ============================================================================

    TSharedRef<SWidget> CreateCaptureSection();
    TSharedRef<SWidget> CreateSemiSphericalParams();
    TSharedRef<SWidget> CreateNadirGridParams();
    TSharedRef<SWidget> CreateCameraIntrinsicsParams();
    FReply OnCheckCoverageClicked();
    FReply OnStartCaptureSetAClicked();
    FReply OnStartCaptureSetBClicked();
    void ExecuteCapturePipeline(bool bIsSetB);
    void GenerateCameraInfoFromFlashPawn(const FString& ImageDir);

    // ============================================================================
    // SECTION 4: GT MATERIAL EXPORT
    // ============================================================================

    TSharedRef<SWidget> CreateGTExportSection();
    FReply OnAddSelectedActorsClicked();
    FReply OnRemoveFromGTListClicked();
    FReply OnExportGTMaterialsClicked();
    void ExportGTMaterialsFromScene();
    void ExportSingleActorGT(AStaticMeshActor* Actor, const FString& BaseDir, TSharedPtr<FJsonObject> ActorJson);
    bool ExportMeshAsOBJ(UStaticMesh* SM, const FString& ObjPath);
    bool ExportTextureAsPNG(UTexture2D* Tex, const FString& PngPath, int32 Channel);
    bool ExportSolidColorPNG(float Value, int32 Resolution, const FString& PngPath);
    bool ExportMaterialSlotTextures(UMaterialInterface* Mat, int32 SlotIdx, const FString& ActorDir, TSharedPtr<FJsonObject> SlotJson);

    // ============================================================================
    // SECTION 6: TEXENHANCER PIPELINE
    // ============================================================================

    TSharedRef<SWidget> CreatePipelineSection();
    FReply OnBrowseScriptClicked();
    FReply OnRunTexEnhancerClicked();
    FReply OnStopTexEnhancerClicked();
    void PollPipelineProcess();

    // ============================================================================
    // SECTION 7: EVALUATION
    // ============================================================================

    TSharedRef<SWidget> CreateEvaluationSection();
    FReply OnBrowseEstimatedDirClicked();
    FReply OnRunEvaluationClicked();
    void RunBRDFEvaluation();

    // ============================================================================
    // UTILITIES
    // ============================================================================

    void UpdateStatus(const FString& Message);
    FString GetSetACaptureDir() const;
    FString GetSetBCaptureDir() const;
    FString GetGTMaterialsPath() const;
    FString GetEvaluationOutputDir() const;

    void SavePaths();
    void LoadPaths();
    FString GetPathConfigFilePath() const;
};
