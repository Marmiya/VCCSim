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
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program. If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "CoreMinimal.h"
#include "Widgets/SCompoundWidget.h"
#include "Widgets/SWidget.h"
#include "Widgets/Views/SListView.h"
#include "Engine/World.h"
#include "GameFramework/Actor.h"

class AFlashPawn;
class AVCCSimLookAtPath;
class URGBDCameraComponent;
class USegCameraComponent;
class UNormalCameraComponent;
class UBaseColorCameraComponent;
class UMaterialPropertiesCameraComponent;

/** One entry of the shared target actor list; disabled entries stay in the list but are skipped by tasks. */
struct FVCCSimTargetActorItem
{
    FString Label;
    bool bEnabled = true;
};

/**
 * Selection panel functionality manager
 * Handles Flash Pawn selection, Camera selection, and Target object selection
 */
class VCCSIMEDITOR_API FVCCSimPanelSelection : public TSharedFromThis<FVCCSimPanelSelection>
{
public:
    FVCCSimPanelSelection();
    ~FVCCSimPanelSelection();

    // ============================================================================
    // PUBLIC INTERFACE
    // ============================================================================
    
    /** Initialize the selection manager */
    void Initialize();
    
    /** Cleanup resources */
    void Cleanup();
    
    /** Create the selection UI panel */
    TSharedRef<SWidget> CreateSelectionPanel();
    
    /** Handle actor selection from viewport */
    void HandleActorSelection(AActor* Actor);

    // UI state access for persistence
    bool IsFlashPawnSectionExpanded() const { return bFlashPawnSectionExpanded; }
    void SetFlashPawnSectionExpanded(bool bExpanded) { bFlashPawnSectionExpanded = bExpanded; }
    
    /** Auto-select first available FlashPawn in the scene */
    void AutoSelectFlashPawn();

    /** Auto-select first available LookAtPath in the scene */
    void AutoSelectLookAtPath();

    // ============================================================================
    // SHARED TARGET ACTOR LIST
    // ============================================================================

    /** Labels of list entries whose checkbox is enabled (consumed by path generation and GT export). */
    TArray<FString> GetEnabledTargetActorLabels() const;

    /** True if at least one list entry is enabled. */
    bool HasEnabledTargetActors() const;

    /** Reload the target actor list from the centralized config manager. */
    void LoadFromConfigManager();

    // ============================================================================
    // GETTERS
    // ============================================================================
    
    /** Get currently selected FlashPawn */
    TWeakObjectPtr<AFlashPawn> GetSelectedFlashPawn() const { return SelectedFlashPawn; }

    /** Get currently selected LookAtPath */
    TWeakObjectPtr<AVCCSimLookAtPath> GetSelectedLookAtPath() const { return SelectedLookAtPath; }

    /** Check if FlashPawn selection mode is active */
    bool IsSelectingFlashPawn() const { return bSelectingFlashPawn; }

    /** Check if LookAtPath selection mode is active */
    bool IsSelectingLookAtPath() const { return bSelectingLookAtPath; }
    
    /** Check camera availability */
    bool HasRGBCamera() const { return bHasRGBCamera; }
    bool HasDepthCamera() const { return bHasDepthCamera; }
    bool HasSegmentationCamera() const { return bHasSegmentationCamera; }
    bool HasNormalCamera() const { return bHasNormalCamera; }
    bool HasBaseColorCamera() const { return bHasBaseColorCamera; }
    bool HasMaterialPropertiesCamera() const { return bHasMaterialPropertiesCamera; }

    /** Check camera usage state */
    bool IsUsingRGBCamera() const { return bUseRGBCamera; }
    bool IsUsingDepthCamera() const { return bUseDepthCamera; }
    bool IsUsingSegmentationCamera() const { return bUseSegmentationCamera; }
    bool IsUsingNormalCamera() const { return bUseNormalCamera; }
    bool IsUsingBaseColorCamera() const { return bUseBaseColorCamera; }
    bool IsUsingMaterialPropertiesCamera() const { return bUseMaterialPropertiesCamera; }
    
    /** Check if any camera is both available and active */
    bool HasAnyActiveCamera() const;

    /** Check if RGB capture should use RGBCamera class instead of screenshot */
    bool ShouldUseRGBCameraClass() const { return bUseRGBCameraClass; }

    /** Check if cameras have been warmed up */
    bool IsWarmedUp() const { return bIsWarmedUp; }

    /** Initialize and warmup all cameras on the selected FlashPawn */
    void WarmupCameras();

    /** Geometric building-detection thresholds (cm) used to pick which target clusters get facade
     *  orbits. Consumed by the path-image-capture panel so Highlight Targets and Generate Poses
     *  detect the same buildings. */
    float GetMinBuildingHeight() const { return MinBuildingHeight; }
    float GetMinBuildingFootprint() const { return MinBuildingFootprint; }
    float GetConnectGap() const { return ConnectGap; }

private:
    // ============================================================================
    // UI CREATION
    // ============================================================================
    
    /** Create Flash Pawn selection panel */
    TSharedRef<SWidget> CreatePawnSelectPanel();

    /** Create Camera selection panel */
    TSharedRef<SWidget> CreateCameraSelectPanel();

    /** Create LookAtPath selection panel */
    TSharedRef<SWidget> CreateLookAtSelectPanel();

    /** Create the shared target actor list panel */
    TSharedRef<SWidget> CreateTargetActorListPanel();

    /** Create the coordinate bounding-box selection sub-panel */
    TSharedRef<SWidget> CreateBoundsSelectPanel();

    /** Create camera status row showing available cameras */
    TSharedRef<SWidget> CreateCameraStatusRow();
    
    /** Create individual camera status box */
    TSharedRef<SWidget> CreateCameraStatusBox(
        const FString& CameraName,
        TFunction<bool()> HasCameraFunc,
        TFunction<ECheckBoxState()> IsCheckedFunc,
        TFunction<void(ECheckBoxState)> OnStateChangedFunc
    );
    
    
    // ============================================================================
    // EVENT HANDLERS
    // ============================================================================
    
    /** Handle FlashPawn selection toggle */
    void OnSelectFlashPawnToggleChanged(ECheckBoxState NewState);

    /** Handle LookAtPath selection toggle */
    void OnSelectLookAtToggleChanged(ECheckBoxState NewState);
    
    /** Target actor list handlers */
    FReply OnAddTargetActorsClicked();
    void SaveTargetActorsToConfig() const;

    /** Add every mesh actor whose bounds center falls inside the coordinate box (clutter-filtered). */
    FReply OnAddTargetActorsInBoundsClicked();

    /** Fill the coordinate box from the combined AABB of the current editor selection. */
    FReply OnFillBoundsFromSelectionClicked();

    /** Export the enabled target actors' GT mesh (geometry + is_glass) to a chosen folder. */
    FReply OnExportGTMeshClicked();

    /** Draw a labelled debug box around every list actor (enabled green / disabled gray). */
    FReply OnHighlightTargetsClicked();

    /** Rename actors that share a label so every actor label is unique (the whole target
     *  pipeline resolves actors by label, which UE does not globally enforce as unique). */
    FReply OnFixDuplicateLabelsClicked();

    /** Handle camera checkbox changes */
    void OnRGBCameraCheckboxChanged(ECheckBoxState NewState);
    void OnDepthCameraCheckboxChanged(ECheckBoxState NewState);
    void OnSegmentationCameraCheckboxChanged(ECheckBoxState NewState);
    void OnNormalCameraCheckboxChanged(ECheckBoxState NewState);
    void OnBaseColorCameraCheckboxChanged(ECheckBoxState NewState);
    void OnMaterialPropertiesCameraCheckboxChanged(ECheckBoxState NewState);
    void OnUseRGBCameraClassCheckboxChanged(ECheckBoxState NewState);
    
    // ============================================================================
    // SELECTION LOGIC
    // ============================================================================
    
    /** Update camera availability flags based on selected FlashPawn */
    void RefreshCameraAvailability();
    
    /** Clear current selections */
    void ClearSelections();

private:
    // ============================================================================
    // UI ELEMENTS
    // ============================================================================
    
    // FlashPawn Selection UI
    TSharedPtr<STextBlock> SelectedFlashPawnText;
    TSharedPtr<SCheckBox> SelectFlashPawnToggle;

    // LookAtPath Selection UI
    TSharedPtr<STextBlock> SelectedLookAtText;
    TSharedPtr<SCheckBox> SelectLookAtToggle;

    // Shared target actor list UI
    TSharedPtr<SListView<TSharedPtr<FVCCSimTargetActorItem>>> TargetActorListView;
    TArray<TSharedPtr<FVCCSimTargetActorItem>> TargetActorItems;
    
    // ============================================================================
    // SELECTION STATE
    // ============================================================================
    
    /** Currently selected actors */
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    TWeakObjectPtr<AVCCSimLookAtPath> SelectedLookAtPath;

    /** Selection mode flags */
    bool bSelectingFlashPawn = false;
    bool bSelectingLookAtPath = false;
    
    /** Camera availability flags */
    bool bHasRGBCamera = false;
    bool bHasDepthCamera = false;
    bool bHasSegmentationCamera = false;
    bool bHasNormalCamera = false;
    bool bHasBaseColorCamera = false;
    bool bHasMaterialPropertiesCamera = false;

    /** Camera usage flags */
    bool bUseRGBCamera = true;
    bool bUseDepthCamera = false;
    bool bUseSegmentationCamera = false;
    bool bUseNormalCamera = false;
    bool bUseBaseColorCamera = false;
    bool bUseMaterialPropertiesCamera = false;
    
    /** Flag to use RGBCamera class for capture instead of high-res screenshot */
    bool bUseRGBCameraClass = false;

    /** Flag indicating cameras have been initialized and warmed up */
    bool bIsWarmedUp = false;

    /** UI section expansion state */
    bool bFlashPawnSectionExpanded = false;

    /** Coordinate bounding box (UE world cm) used by bounds-based target selection. */
    FVector BoundsMin = FVector(-100000.0);
    FVector BoundsMax = FVector(100000.0);

    /** Geometric building-detection thresholds (cm): a clustered target counts as a building (and
     *  gets a facade orbit) only if it is at least this tall and this wide. Everything else is still
     *  surveyed. Replaces the old name-based clutter exclusion. */
    float MinBuildingHeight = 300.0f;
    float MinBuildingFootprint = 300.0f;

    /** Connectivity tolerance (cm): two structure pieces merge into one building only if their
     *  oriented boxes come within this of touching. Larger merges gapped pieces; smaller keeps
     *  road-side props separate. Shared by Highlight Targets and Generate Poses. */
    float ConnectGap = 15.0f;

    /** Also export in-box clutter as a separate region_context mesh (geometry only). */
    bool bExportContextMesh = true;

    /** GT mesh export (relocated here from the TexEnhancer panel; geometry + is_glass only). */
    TSharedPtr<class FGTMaterialExporter> GTMaterialExporter;
    bool bGTExportInProgress = false;

    /** Transient actor holding the debug-highlight TextRenderComponents (auto-destroyed). */
    TWeakObjectPtr<AActor> HighlightActor;
};