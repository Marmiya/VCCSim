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
#include "Misc/Paths.h"
#include "AssetRegistry/AssetData.h"

class FColmapManager;
class FVCCSimPanelPointCloud;
class FVCCSimPanelSelection;
class FVCCSimPanelPathImageCapture;
class FVCCSimPanelSceneAnalysis;
class FVCCSimPanelRatSplatting;


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
    
    // ============================================================================
    // STATE VARIABLES
    // ============================================================================
    
    // COLMAP pipeline state
    bool bColmapPipelineInProgress = false;
    TSharedPtr<FColmapManager> ColmapManager;
    
    // Panel managers
    TSharedPtr<FVCCSimPanelPointCloud> PointCloudManager;
    TSharedPtr<FVCCSimPanelSelection> SelectionManager;
    TSharedPtr<FVCCSimPanelPathImageCapture> PathImageCaptureManager;
    TSharedPtr<FVCCSimPanelSceneAnalysis> SceneAnalysisManager;
    TSharedPtr<FVCCSimPanelRatSplatting> RatSplattingManager;

    // ============================================================================
    // INITIALIZATION AND CLEANUP
    // ============================================================================
    
    void LoadLogoImages();
    void CreateMainLayout();
    
    // ============================================================================
    // UI CONSTRUCTION HELPERS
    // ============================================================================
    
    // Main panel creators
    TSharedRef<SWidget> CreateLogoPanel();
    TSharedRef<SWidget> CreateSceneAnalysisPanel();
    TSharedRef<SWidget> CreatePointCloudPanel();
    TSharedRef<SWidget> CreateRatSplattingPanel();
    
    // Style and layout helpers (now using FVCCSimUIHelpers)
    

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