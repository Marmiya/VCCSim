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

DEFINE_LOG_CATEGORY_STATIC(LogSelection, Log, All);

#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Engine/World.h"
#include "Engine/Engine.h"
#include "EngineUtils.h"
#include "Pawns/FlashPawn.h"
#include "Pawns/SimLookAtPath.h"
#include "Sensors/SensorBase.h"
#include "Sensors/RGBCamera.h"
#include "Sensors/DepthCamera.h"
#include "Sensors/SegmentationCamera.h"
#include "Sensors/NormalCamera.h"
#include "Sensors/BaseColorCamera.h"
#include "Sensors/MaterialPropertiesCamera.h"

FVCCSimPanelSelection::FVCCSimPanelSelection()
{
}

FVCCSimPanelSelection::~FVCCSimPanelSelection()
{
    Cleanup();
}

void FVCCSimPanelSelection::Initialize()
{
    UE_LOG(LogSelection, Log, TEXT("VCCSimPanelSelection initialized"));
}

void FVCCSimPanelSelection::Cleanup()
{
    ClearSelections();

    SelectedFlashPawnText.Reset();
    SelectFlashPawnToggle.Reset();
    SelectedLookAtText.Reset();
    SelectLookAtToggle.Reset();
}

void FVCCSimPanelSelection::HandleActorSelection(AActor* Actor)
{
    if (!Actor || !IsValid(Actor))
    {
        return;
    }

    if (bSelectingFlashPawn)
    {
        AFlashPawn* FlashPawn = Cast<AFlashPawn>(Actor);
        if (FlashPawn)
        {
            SelectedFlashPawn = FlashPawn;
            if (SelectedFlashPawnText.IsValid())
            {
                SelectedFlashPawnText->SetText(FText::FromString(FlashPawn->GetActorLabel()));
            }

            bSelectingFlashPawn = false;
            if (SelectFlashPawnToggle.IsValid())
            {
                SelectFlashPawnToggle->SetIsChecked(ECheckBoxState::Unchecked);
            }

            RefreshCameraAvailability();
            bIsWarmedUp = false;

            UE_LOG(LogSelection, Log, TEXT("Selected FlashPawn: %s"), *FlashPawn->GetActorLabel());
        }
    }
    else if (bSelectingLookAtPath)
    {
        AVCCSimLookAtPath* LookAt = Cast<AVCCSimLookAtPath>(Actor);
        if (LookAt)
        {
            SelectedLookAtPath = LookAt;
            if (SelectedLookAtText.IsValid())
            {
                SelectedLookAtText->SetText(FText::FromString(LookAt->GetActorLabel()));
            }

            bSelectingLookAtPath = false;
            if (SelectLookAtToggle.IsValid())
            {
                SelectLookAtToggle->SetIsChecked(ECheckBoxState::Unchecked);
            }

            UE_LOG(LogSelection, Log, TEXT("Selected LookAtPath: %s"), *LookAt->GetActorLabel());
        }
        else
        {
            UE_LOG(LogSelection, Warning, TEXT("Selected actor is not a VCCSimLookAtPath"));
        }
    }
}

void FVCCSimPanelSelection::AutoSelectFlashPawn()
{
    UWorld* World = GEditor->GetEditorWorldContext().World();
    if (!World)
    {
        return;
    }

    SelectedFlashPawn = nullptr;

    AFlashPawn* FirstFoundFlashPawn = nullptr;
    for (TActorIterator<AFlashPawn> ActorIterator(World); ActorIterator; ++ActorIterator)
    {
        AFlashPawn* FlashPawn = *ActorIterator;
        if (FlashPawn && IsValid(FlashPawn))
        {
            FirstFoundFlashPawn = FlashPawn;
            break;
        }
    }

    if (FirstFoundFlashPawn)
    {
        SelectedFlashPawn = FirstFoundFlashPawn;

        if (SelectedFlashPawnText.IsValid())
        {
            SelectedFlashPawnText->SetText(FText::FromString(FirstFoundFlashPawn->GetActorLabel()));
        }

        RefreshCameraAvailability();
        bIsWarmedUp = false;

        UE_LOG(LogSelection, Log, TEXT("Auto-selected FlashPawn: %s"), *FirstFoundFlashPawn->GetActorLabel());
    }
    else
    {
        if (SelectedFlashPawnText.IsValid())
        {
            SelectedFlashPawnText->SetText(FText::FromString("None selected"));
        }
        UE_LOG(LogSelection, Log, TEXT("No FlashPawn found in the scene for auto-selection"));
    }
}

void FVCCSimPanelSelection::AutoSelectLookAtPath()
{
    UWorld* World = GEditor->GetEditorWorldContext().World();
    if (!World)
    {
        return;
    }

    SelectedLookAtPath = nullptr;

    for (TActorIterator<AVCCSimLookAtPath> It(World); It; ++It)
    {
        AVCCSimLookAtPath* LookAt = *It;
        if (LookAt && IsValid(LookAt))
        {
            SelectedLookAtPath = LookAt;
            if (SelectedLookAtText.IsValid())
            {
                SelectedLookAtText->SetText(FText::FromString(LookAt->GetActorLabel()));
            }
            UE_LOG(LogSelection, Log, TEXT("Auto-selected LookAtPath: %s"), *LookAt->GetActorLabel());
            return;
        }
    }

    if (SelectedLookAtText.IsValid())
    {
        SelectedLookAtText->SetText(FText::FromString("None selected"));
    }
    UE_LOG(LogSelection, Log, TEXT("No VCCSimLookAtPath found in the scene for auto-selection"));
}

void FVCCSimPanelSelection::OnSelectFlashPawnToggleChanged(ECheckBoxState NewState)
{
    bSelectingFlashPawn = (NewState == ECheckBoxState::Checked);

    if (bSelectingFlashPawn && bSelectingLookAtPath)
    {
        bSelectingLookAtPath = false;
        if (SelectLookAtToggle.IsValid())
        {
            SelectLookAtToggle->SetIsChecked(ECheckBoxState::Unchecked);
        }
    }
}

void FVCCSimPanelSelection::OnSelectLookAtToggleChanged(ECheckBoxState NewState)
{
    bSelectingLookAtPath = (NewState == ECheckBoxState::Checked);

    if (bSelectingLookAtPath && bSelectingFlashPawn)
    {
        bSelectingFlashPawn = false;
        if (SelectFlashPawnToggle.IsValid())
        {
            SelectFlashPawnToggle->SetIsChecked(ECheckBoxState::Unchecked);
        }
    }
}

void FVCCSimPanelSelection::OnRGBCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseRGBCamera = (NewState == ECheckBoxState::Checked);
}

void FVCCSimPanelSelection::OnDepthCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseDepthCamera = (NewState == ECheckBoxState::Checked);
}

void FVCCSimPanelSelection::OnSegmentationCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseSegmentationCamera = (NewState == ECheckBoxState::Checked);
}

void FVCCSimPanelSelection::OnNormalCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseNormalCamera = (NewState == ECheckBoxState::Checked);
}

void FVCCSimPanelSelection::OnBaseColorCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseBaseColorCamera = (NewState == ECheckBoxState::Checked);
}

void FVCCSimPanelSelection::OnMaterialPropertiesCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseMaterialPropertiesCamera = (NewState == ECheckBoxState::Checked);
}

void FVCCSimPanelSelection::OnUseRGBCameraClassCheckboxChanged(ECheckBoxState NewState)
{
    bUseRGBCameraClass = (NewState == ECheckBoxState::Checked);
}

void FVCCSimPanelSelection::RefreshCameraAvailability()
{
    bHasRGBCamera = false;
    bHasDepthCamera = false;
    bHasSegmentationCamera = false;
    bHasNormalCamera = false;
    bHasBaseColorCamera = false;
    bHasMaterialPropertiesCamera = false;

    if (!SelectedFlashPawn.IsValid())
    {
        return;
    }

    TArray<URGBCameraComponent*> RGBCameras;
    TArray<UDepthCameraComponent*> DepthCameras;
    TArray<USegCameraComponent*> SegmentationCameras;
    TArray<UNormalCameraComponent*> NormalCameras;
    TArray<UBaseColorCameraComponent*> BaseColorCameras;
    TArray<UMaterialPropertiesCameraComponent*> MatPropsCameras;

    SelectedFlashPawn->GetComponents<URGBCameraComponent>(RGBCameras);
    SelectedFlashPawn->GetComponents<UDepthCameraComponent>(DepthCameras);
    SelectedFlashPawn->GetComponents<USegCameraComponent>(SegmentationCameras);
    SelectedFlashPawn->GetComponents<UNormalCameraComponent>(NormalCameras);
    SelectedFlashPawn->GetComponents<UBaseColorCameraComponent>(BaseColorCameras);
    SelectedFlashPawn->GetComponents<UMaterialPropertiesCameraComponent>(MatPropsCameras);

    bHasRGBCamera = (RGBCameras.Num() > 0);
    bHasDepthCamera = (DepthCameras.Num() > 0);
    bHasSegmentationCamera = (SegmentationCameras.Num() > 0);
    bHasNormalCamera = (NormalCameras.Num() > 0);
    bHasBaseColorCamera = (BaseColorCameras.Num() > 0);
    bHasMaterialPropertiesCamera = (MatPropsCameras.Num() > 0);
}

void FVCCSimPanelSelection::WarmupCameras()
{
    if (!SelectedFlashPawn.IsValid())
    {
        return;
    }

    TArray<UCameraBaseComponent*> AllCameras;
    SelectedFlashPawn->GetComponents<UCameraBaseComponent>(AllCameras);

    for (UCameraBaseComponent* Camera : AllCameras)
    {
        if (Camera)
        {
            Camera->WarmupCapture();
        }
    }

    bIsWarmedUp = true;
    UE_LOG(LogSelection, Log, TEXT("Warmed up %d camera(s) on FlashPawn"), AllCameras.Num());
}

void FVCCSimPanelSelection::ClearSelections()
{
    SelectedFlashPawn.Reset();
    SelectedLookAtPath.Reset();
    bSelectingFlashPawn = false;
    bSelectingLookAtPath = false;

    bHasRGBCamera = false;
    bHasDepthCamera = false;
    bHasSegmentationCamera = false;
    bHasNormalCamera = false;
    bHasBaseColorCamera = false;
    bHasMaterialPropertiesCamera = false;
    bUseRGBCamera = true;
    bUseDepthCamera = false;
    bUseSegmentationCamera = false;
    bUseNormalCamera = false;
    bUseBaseColorCamera = false;
    bUseMaterialPropertiesCamera = false;
    bUseRGBCameraClass = false;
}

bool FVCCSimPanelSelection::HasAnyActiveCamera() const
{
    return (bHasRGBCamera && bUseRGBCamera) ||
           (bHasDepthCamera && bUseDepthCamera) ||
           (bHasSegmentationCamera && bUseSegmentationCamera) ||
           (bHasNormalCamera && bUseNormalCamera) ||
           (bHasBaseColorCamera && bUseBaseColorCamera) ||
           (bHasMaterialPropertiesCamera && bUseMaterialPropertiesCamera);
}
