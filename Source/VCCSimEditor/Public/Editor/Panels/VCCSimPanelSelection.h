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
#include "Engine/World.h"
#include "GameFramework/Actor.h"

class SButton;
class STextBlock;
class SCheckBox;
class AFlashPawn;
class URGBCameraComponent;
class UDepthCameraComponent;
class USegmentationCameraComponent;
class UNormalCameraComponent;

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
    
    /** Update camera availability based on selected FlashPawn */
    void UpdateActiveCameras();
    
    /** Handle actor selection from viewport */
    void HandleActorSelection(AActor* Actor);
    
    /** Auto-select first available FlashPawn in the scene */
    void AutoSelectFlashPawn();

    // ============================================================================
    // GETTERS
    // ============================================================================
    
    /** Get currently selected FlashPawn */
    TWeakObjectPtr<AFlashPawn> GetSelectedFlashPawn() const { return SelectedFlashPawn; }
    
    /** Get currently selected target object */
    TWeakObjectPtr<AActor> GetSelectedTargetObject() const { return SelectedTargetObject; }
    
    /** Check if FlashPawn selection mode is active */
    bool IsSelectingFlashPawn() const { return bSelectingFlashPawn; }
    
    /** Check if Target selection mode is active */
    bool IsSelectingTarget() const { return bSelectingTarget; }
    
    /** Check camera availability */
    bool HasRGBCamera() const { return bHasRGBCamera; }
    bool HasDepthCamera() const { return bHasDepthCamera; }
    bool HasSegmentationCamera() const { return bHasSegmentationCamera; }
    bool HasNormalCamera() const { return bHasNormalCamera; }
    
    /** Check camera usage state */
    bool IsUsingRGBCamera() const { return bUseRGBCamera; }
    bool IsUsingDepthCamera() const { return bUseDepthCamera; }
    bool IsUsingSegmentationCamera() const { return bUseSegmentationCamera; }
    bool IsUsingNormalCamera() const { return bUseNormalCamera; }
    
    /** Get camera parameters from active cameras */
    float GetActiveCameraFOV() const;
    FIntPoint GetActiveCameraResolution() const;
    
    /** Check if any camera is both available and active */
    bool HasAnyActiveCamera() const;

private:
    // ============================================================================
    // UI CREATION
    // ============================================================================
    
    /** Create Flash Pawn selection panel */
    TSharedRef<SWidget> CreatePawnSelectPanel();
    
    /** Create Camera selection panel */
    TSharedRef<SWidget> CreateCameraSelectPanel();
    
    /** Create Target object selection panel */
    TSharedRef<SWidget> CreateTargetSelectPanel();
    
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
    
    /** Handle Target selection toggle */
    void OnSelectTargetToggleChanged(ECheckBoxState NewState);
    
    /** Handle camera checkbox changes */
    void OnRGBCameraCheckboxChanged(ECheckBoxState NewState);
    void OnDepthCameraCheckboxChanged(ECheckBoxState NewState);
    void OnSegmentationCameraCheckboxChanged(ECheckBoxState NewState);
    void OnNormalCameraCheckboxChanged(ECheckBoxState NewState);
    
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
    
    // Target Selection UI
    TSharedPtr<STextBlock> SelectedTargetObjectText;
    TSharedPtr<SCheckBox> SelectTargetToggle;
    
    // ============================================================================
    // SELECTION STATE
    // ============================================================================
    
    /** Currently selected actors */
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    TWeakObjectPtr<AActor> SelectedTargetObject;
    
    /** Selection mode flags */
    bool bSelectingFlashPawn = false;
    bool bSelectingTarget = false;
    
    /** Camera availability flags */
    bool bHasRGBCamera = false;
    bool bHasDepthCamera = false;
    bool bHasSegmentationCamera = false;
    bool bHasNormalCamera = false;
    
    /** Camera usage flags */
    bool bUseRGBCamera = true;  // RGB camera enabled by default
    bool bUseDepthCamera = false;
    bool bUseSegmentationCamera = false;
    bool bUseNormalCamera = false;
    
    /** UI section expansion state */
    bool bFlashPawnSectionExpanded = false;
};