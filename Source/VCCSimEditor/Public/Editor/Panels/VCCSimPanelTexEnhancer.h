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

#include "Widgets/Input/SCheckBox.h"

#include "Utils/LightingManager.h"
#include "Utils/GTMaterialExporter.h"
#include "Utils/CaptureReuseManifest.h"

class FVCCSimPanelSelection;
class FVCCSimPanelPathImageCapture;
class FLightingManager;

class VCCSIMEDITOR_API FVCCSimPanelTexEnhancer : public TSharedFromThis<FVCCSimPanelTexEnhancer>
{
public:
    FVCCSimPanelTexEnhancer();
    ~FVCCSimPanelTexEnhancer();

    void Initialize();
    void Cleanup();
    TSharedRef<SWidget> CreateTexEnhancerPanel();

    void SetSelectionManager(TSharedPtr<FVCCSimPanelSelection> InSelectionManager);
    void SetPathImageCaptureManager(TSharedPtr<FVCCSimPanelPathImageCapture> InPathImageCaptureManager);

    bool IsTexEnhancerSectionExpanded() const { return bSectionExpanded; }
    void SetTexEnhancerSectionExpanded(bool bExpanded) { bSectionExpanded = bExpanded; }

    void LoadFromConfigManager();
    void SaveToConfigManager();

private:
    // ============================================================================
    // UI ELEMENTS
    // ============================================================================

    TSharedPtr<SEditableTextBox> OutputDirTextBox;
    TSharedPtr<SEditableTextBox> SceneNameTextBox;

    TSharedPtr<SNumericEntryBox<float>> LightingElevationSpinBox[5];
    TSharedPtr<SNumericEntryBox<float>> LightingAzimuthSpinBox[5];
    TOptional<float> LightingElevationValue[5];
    TOptional<float> LightingAzimuthValue[5];

    TSharedPtr<SEditableTextBox> TexEnhancerScriptTextBox;
    TSharedPtr<SEditableTextBox> EstimatedMaterialsDirTextBox;

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
    TOptional<float>                    DayCycleSpeedValue;

    // ============================================================================
    // STATE VARIABLES
    // ============================================================================

    bool bSectionExpanded             = false;
    bool bLightingScheduleExpanded    = false;

    static constexpr int32 NumLightingConditions = 5;

    float LightingElevation[NumLightingConditions] = { 20.f, 70.f, 35.f, 85.f, 15.f };
    float LightingAzimuth[NumLightingConditions]   = { 30.f, 110.f, 190.f, 250.f, 320.f };
    bool  bLightingSelected[NumLightingConditions] = {};

    FString OutputDirectory;
    FString SceneName = TEXT("Scene_A");
    FString TexEnhancerScriptPath;
    FString EstimatedMaterialsDir;

    bool bPipelineInProgress    = false;
    bool bEvalInProgress        = false;
    bool bGTExportInProgress    = false;
    bool bDatasetCaptureInProgress = false;
    bool bBatchCapture          = false;
    bool bDayCycleActive        = false;

    FString BatchCaptureTimestamp;
    TArray<int32> LightingCaptureQueue;

    FString PendingCaptureName;
    FCaptureReuseEntry PendingReuseEntry;

    int32 GTTextureResolution = 2048;

    bool bOutputImages = true;
    bool bOutputMesh   = true;

    float SunCalcLatitude  = 22.52933f;
    float SunCalcLongitude = 113.94092f;
    float SunCalcTimeZone  = 8.0f;
    float DayCycleSpeed    = 10.f;
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

    FTimerHandle StatusTimerHandle;
    FProcHandle  PipelineProcHandle;

    TWeakPtr<FVCCSimPanelSelection> SelectionManager;
    TWeakPtr<FVCCSimPanelPathImageCapture> PathImageCaptureManager;
    TSharedPtr<FLightingManager>    LightingManager;

    // ============================================================================
    // SECTION 1: DATASET CONFIGURATION
    // ============================================================================

    TSharedRef<SWidget> CreateDatasetConfigSection();
    FReply OnBrowseOutputDirClicked();

    // ============================================================================
    // SECTION 2: LIGHTING SCHEDULE
    // ============================================================================

    TSharedRef<SWidget> CreateLightingScheduleSection();
    TSharedRef<SWidget> CreateLightingEntry(int32 Index);
    TSharedRef<SWidget> CreateSunPositionCalculatorWidget();
    FReply OnApplyLightingClicked(int32 Index);
    FReply OnCalculateSunPositionClicked();
    FReply OnFillFromSunPositionClicked();
    FReply OnToggleDayCycleClicked();

    // ============================================================================
    // SECTION 3: DATASET CAPTURE
    // ============================================================================

    TSharedRef<SWidget> CreateDatasetCaptureSection();
    FReply OnCaptureDatasetClicked();
    bool DecideAndStartCapture(const FString& CaptureDir);
    void StartNextBatchCapture();
    void OnDatasetCaptureFinished(bool bSuccess, FString CaptureDirectory);
    FString GetDatasetCapturesRoot() const;
    FString MakeNextCaptureDirectory() const;

    // ============================================================================
    // SECTION 4: GT MATERIAL EXPORT
    // ============================================================================

    TSharedRef<SWidget> CreateGTExportSection();
    FReply OnExportGTMaterialsClicked();
    bool StartGTMaterialExport(const FString& BaseDir);
    FString FindLatestCaptureDirectory() const;

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

    FString GetGTMaterialsPath() const;
    FString GetEvaluationOutputDir() const;

    void SavePaths();
    void LoadPaths();
    void LoadParamsFromConfig();
};

