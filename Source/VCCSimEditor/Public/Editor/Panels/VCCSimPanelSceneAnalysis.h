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

class ASceneAnalysisManager;
class FVCCSimPanelSelection;
class URGBCameraComponent;
class SCheckBox;
class SButton;
template<typename NumericType> class SNumericEntryBox;
class SWidget;

/**
 * Scene Analysis Panel - Modular panel for scene analysis functionality
 * Handles scene scanning, safe zone generation, coverage analysis, and complexity analysis
 */
class VCCSIMEDITOR_API FVCCSimPanelSceneAnalysis
{
public:
    FVCCSimPanelSceneAnalysis();
    ~FVCCSimPanelSceneAnalysis();
    
    void Initialize(TSharedPtr<FVCCSimPanelSelection> InSelectionManager);
    void Cleanup();
    TSharedRef<SWidget> CreateSceneAnalysisPanel();
    void UpdateVisualization();
    
    // Getters for state access
    TWeakObjectPtr<ASceneAnalysisManager> GetSceneAnalysisManager() const { return SceneAnalysisManager; }
    bool IsSceneAnalysisSectionExpanded() const { return bSceneAnalysisSectionExpanded; }
    void SetSceneAnalysisSectionExpanded(bool bExpanded) { bSceneAnalysisSectionExpanded = bExpanded; }
    
private:
    // ============================================================================
    // UI ELEMENTS
    // ============================================================================
    
    // Panel state
    bool bSceneAnalysisSectionExpanded = false;
    
    // UI Controls
    TSharedPtr<SCheckBox> SelectUseLimitedToggle;
    
    // Configuration spinboxes
    TSharedPtr<SNumericEntryBox<float>> SafeDistanceSpinBox;
    TSharedPtr<SNumericEntryBox<float>> SafeHeightSpinBox;
    TSharedPtr<SNumericEntryBox<float>> LimitedMinXSpinBox;
    TSharedPtr<SNumericEntryBox<float>> LimitedMaxXSpinBox;
    TSharedPtr<SNumericEntryBox<float>> LimitedMinYSpinBox;
    TSharedPtr<SNumericEntryBox<float>> LimitedMaxYSpinBox;
    TSharedPtr<SNumericEntryBox<float>> LimitedMinZSpinBox;
    TSharedPtr<SNumericEntryBox<float>> LimitedMaxZSpinBox;
    
    // Visualization buttons
    TSharedPtr<SButton> VisualizeSafeZoneButton;
    TSharedPtr<SButton> VisualizeCoverageButton;
    TSharedPtr<SButton> VisualizeComplexityButton;
    
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
    
    // Dependencies
    TWeakPtr<FVCCSimPanelSelection> SelectionManager;
    
    // ============================================================================
    // INITIALIZATION AND MANAGER OPERATIONS
    // ============================================================================
    
    void InitializeSceneAnalysisManager();
    void OnUseLimitedToggleChanged(ECheckBoxState NewState);
    
    // ============================================================================
    // SCENE ANALYSIS OPERATIONS
    // ============================================================================
    
    FReply OnToggleSafeZoneVisualizationClicked();
    FReply OnToggleCoverageVisualizationClicked();
    FReply OnToggleComplexityVisualizationClicked();
    
    // ============================================================================
    // UI CONSTRUCTION HELPERS
    // ============================================================================
    
    // Main UI creators
    TSharedRef<SWidget> CreateLimitedRegionControls();
    TSharedRef<SWidget> CreateSceneOperationButtons();
    TSharedRef<SWidget> CreateSafeZoneButtons();
    TSharedRef<SWidget> CreateCoverageButtons();
    TSharedRef<SWidget> CreateComplexityButtons();
    
    // Style and layout helpers
    TSharedRef<SWidget> CreateCollapsibleSection(
        const FString& Title, TSharedRef<SWidget> Content, bool& bExpanded);
    TSharedRef<SWidget> CreateSectionHeader(const FString& Title);
    TSharedRef<SWidget> CreateSectionContent(TSharedRef<SWidget> Content);
    TSharedRef<SWidget> CreatePropertyRow(
        const FString& Label, TSharedRef<SWidget> Content);
    TSharedRef<SWidget> CreateSeparator();
    
    // Numeric property row creators
    TSharedRef<SWidget> CreateNumericPropertyRowFloat(
        const FString& Label,
        TSharedPtr<SNumericEntryBox<float>>& SpinBox,
        TOptional<float>& Value,
        float& ActualVariable,
        float MinValue,
        float DeltaValue);
};