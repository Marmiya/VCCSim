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
#include "Utils/VCCSimUIHelpers.h"
#include "Engine/World.h"
#include "Engine/Engine.h"
#include "EngineUtils.h"
#include "Pawns/FlashPawn.h"
#include "Pawns/SimLookAtPath.h"
#include "Sensors/RGBCamera.h"
#include "Sensors/DepthCamera.h"
#include "Sensors/SegmentationCamera.h"
#include "Sensors/NormalCamera.h"
#include "Sensors/BaseColorCamera.h"
#include "Sensors/MaterialPropertiesCamera.h"

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

FVCCSimPanelSelection::FVCCSimPanelSelection()
{
}

FVCCSimPanelSelection::~FVCCSimPanelSelection()
{
    Cleanup();
}

// ============================================================================
// PUBLIC INTERFACE
// ============================================================================

void FVCCSimPanelSelection::Initialize()
{
    UE_LOG(LogSelection, Log, TEXT("VCCSimPanelSelection initialized"));
}

void FVCCSimPanelSelection::Cleanup()
{
    // Clear selections
    ClearSelections();
    
    // Clear UI references
    SelectedFlashPawnText.Reset();
    SelectFlashPawnToggle.Reset();
    SelectedLookAtText.Reset();
    SelectLookAtToggle.Reset();
}

TSharedRef<SWidget> FVCCSimPanelSelection::CreateSelectionPanel()
{
    return SNew(SExpandableArea)
        .InitiallyCollapsed(!bFlashPawnSectionExpanded)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryTop"))
        .BorderBackgroundColor(FColor(48, 48, 48))
        .OnAreaExpansionChanged_Lambda([this](bool bIsExpanded)
        {
            bFlashPawnSectionExpanded = bIsExpanded;
        })
        .HeaderContent()
        [
            SNew(STextBlock)
            .Text(FText::FromString("Object Selection"))
            .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
            .ColorAndOpacity(FColor(233, 233, 233))
            .TransformPolicy(ETextTransformPolicy::ToUpper)
        ]
        .BodyContent()
        [
            SNew(SBorder)
            .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
            .BorderBackgroundColor(FColor(5, 5, 5, 255))
            .Padding(FMargin(15, 6))
            [
                SNew(SVerticalBox)
                
                // Flash Pawn Selection
                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    CreatePawnSelectPanel()
                ]
                
                // Visual Separator
                +SVerticalBox::Slot()
                .AutoHeight()
                .Padding(FMargin(0, 6, 0, 6))
                [
                    FVCCSimUIHelpers::CreateSeparator()
                ]
                
                // Camera Selection
                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    CreateCameraSelectPanel()
                ]

                // Visual Separator
                +SVerticalBox::Slot()
                .AutoHeight()
                .Padding(FMargin(0, 6, 0, 6))
                [
                    FVCCSimUIHelpers::CreateSeparator()
                ]

                // LookAt Path Selection
                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    CreateLookAtSelectPanel()
                ]
            ]
        ];
}

void FVCCSimPanelSelection::HandleActorSelection(AActor* Actor)
{
    if (!Actor || !IsValid(Actor))
    {
        return;
    }

    // Handle FlashPawn selection
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
            
            // Disable selection mode
            bSelectingFlashPawn = false;
            if (SelectFlashPawnToggle.IsValid())
            {
                SelectFlashPawnToggle->SetIsChecked(ECheckBoxState::Unchecked);
            }
            
            // Update camera availability
            RefreshCameraAvailability();
            
            UE_LOG(LogSelection, Log, TEXT("Selected FlashPawn: %s"), *FlashPawn->GetActorLabel());
        }
    }
    // Handle LookAtPath selection
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

    // Search for FlashPawn actors in the world
    AFlashPawn* FirstFoundFlashPawn = nullptr;
    for (TActorIterator<AFlashPawn> ActorIterator(World); ActorIterator; ++ActorIterator)
    {
        AFlashPawn* FlashPawn = *ActorIterator;
        if (FlashPawn && IsValid(FlashPawn))
        {
            FirstFoundFlashPawn = FlashPawn;
            break; // Select the first valid FlashPawn found
        }
    }

    // If we found a FlashPawn, select it
    if (FirstFoundFlashPawn)
    {
        SelectedFlashPawn = FirstFoundFlashPawn;
        
        // Update the UI text to show the selected FlashPawn
        if (SelectedFlashPawnText.IsValid())
        {
            SelectedFlashPawnText->SetText(FText::FromString(FirstFoundFlashPawn->GetActorLabel()));
        }
        
        // Update camera availability
        RefreshCameraAvailability();
        
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

// ============================================================================
// UI CREATION
// ============================================================================

TSharedRef<SWidget> FVCCSimPanelSelection::CreatePawnSelectPanel()
{
    return SNew(SVerticalBox)
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 4, 0, 4))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        .Padding(FMargin(0, 0, 8, 0))
        [
            SNew(STextBlock)
            .Text(FText::FromString("Current"))
            .MinDesiredWidth(80)
            .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
            .ColorAndOpacity(FColor(233, 233, 233))
        ]
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        [
            SNew(SBorder)
            .Padding(4)
            [
                SAssignNew(SelectedFlashPawnText, STextBlock)
                .Text(FText::FromString("None selected"))
                .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
                .ColorAndOpacity(FColor(233, 233, 233))
            ]
        ]
        +SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        .Padding(FMargin(8, 0, 4, 0))
        [
            SAssignNew(SelectFlashPawnToggle, SCheckBox)
            .IsChecked(bSelectingFlashPawn ? ECheckBoxState::Checked : ECheckBoxState::Unchecked)
            .OnCheckStateChanged(this, &FVCCSimPanelSelection::OnSelectFlashPawnToggleChanged)
        ]
        +SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        [
            SNew(STextBlock)
            .Text(FText::FromString("Click to select"))
            .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
            .ColorAndOpacity(FColor(233, 233, 233))
        ]
    ];
}

TSharedRef<SWidget> FVCCSimPanelSelection::CreateCameraSelectPanel()
{
    return SNew(SVerticalBox)
    
    // Camera Availability Section
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 0, 0, 8))
    [
        SNew(SVerticalBox)
        +SVerticalBox::Slot()
        .AutoHeight()
        .Padding(FMargin(0, 4, 0, 4))
        [
            SNew(STextBlock)
            .Text(FText::FromString("Available & Active Cameras:"))
            .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
            .ColorAndOpacity(FColor(233, 233, 233))
        ]
        +SVerticalBox::Slot()
        .AutoHeight()
        [
            CreateCameraStatusRow()
        ]
    ]
    
    // Camera Parameters Section
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 8, 0, 8))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        .Padding(FMargin(0, 0, 8, 0))
        [
            SNew(STextBlock)
            .Text(FText::FromString("FOV:"))
            .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
            .ColorAndOpacity(FColor(233, 233, 233))
        ]
        +SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        .Padding(FMargin(0, 0, 16, 0))
        [
            SNew(STextBlock)
            .Text_Lambda([this]()
            {
                if (SelectedFlashPawn.IsValid() && HasAnyActiveCamera())
                {
                    float FOV = GetActiveCameraFOV();
                    return FText::FromString(FString::Printf(TEXT("%.1f°"), FOV));
                }
                return FText::FromString(TEXT("N/A"));
            })
            .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
            .ColorAndOpacity(FColor(180, 180, 180))
        ]
        +SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        .Padding(FMargin(0, 0, 8, 0))
        [
            SNew(STextBlock)
            .Text(FText::FromString("Resolution:"))
            .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
            .ColorAndOpacity(FColor(233, 233, 233))
        ]
        +SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        .Padding(FMargin(0, 0, 16, 0))
        [
            SNew(STextBlock)
            .Text_Lambda([this]()
            {
                if (SelectedFlashPawn.IsValid() && HasAnyActiveCamera())
                {
                    FIntPoint Resolution = GetActiveCameraResolution();
                    return FText::FromString(FString::Printf(TEXT("%dx%d"), Resolution.X, Resolution.Y));
                }
                return FText::FromString(TEXT("N/A"));
            })
            .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
            .ColorAndOpacity(FColor(180, 180, 180))
        ]
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        [
            SNew(SSpacer)
        ]
    ];
}

TSharedRef<SWidget> FVCCSimPanelSelection::CreateLookAtSelectPanel()
{
    return SNew(SVerticalBox)
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 0, 0, 4))
    [
        SNew(STextBlock)
        .Text(FText::FromString("LookAt Path:"))
        .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
        .ColorAndOpacity(FColor(233, 233, 233))
    ]
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 4, 0, 4))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        .Padding(FMargin(0, 0, 8, 0))
        [
            SNew(STextBlock)
            .Text(FText::FromString("Current"))
            .MinDesiredWidth(80)
            .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
            .ColorAndOpacity(FColor(233, 233, 233))
        ]
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        [
            SNew(SBorder)
            .Padding(4)
            [
                SAssignNew(SelectedLookAtText, STextBlock)
                .Text(FText::FromString("None selected"))
                .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
                .ColorAndOpacity(FColor(233, 233, 233))
            ]
        ]
        +SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        .Padding(FMargin(8, 0, 4, 0))
        [
            SAssignNew(SelectLookAtToggle, SCheckBox)
            .IsChecked(bSelectingLookAtPath ? ECheckBoxState::Checked : ECheckBoxState::Unchecked)
            .OnCheckStateChanged(this, &FVCCSimPanelSelection::OnSelectLookAtToggleChanged)
        ]
        +SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        [
            SNew(STextBlock)
            .Text(FText::FromString("Click to select"))
            .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
            .ColorAndOpacity(FColor(233, 233, 233))
        ]
    ];
}

TSharedRef<SWidget> FVCCSimPanelSelection::CreateCameraStatusRow()
{
    return SNew(SHorizontalBox)
    
    // RGB Camera
    +SHorizontalBox::Slot()
    .MaxWidth(100)
    .HAlign(HAlign_Center)
    .Padding(FMargin(0, 0, 2, 0))
    [
        CreateCameraStatusBox("RGB", 
            [this]() { return bHasRGBCamera; },
            [this]() { return bUseRGBCamera ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; },
            [this](ECheckBoxState NewState) { OnRGBCameraCheckboxChanged(NewState); })
    ]
    
    // Depth Camera
    +SHorizontalBox::Slot()
    .MaxWidth(100)
    .HAlign(HAlign_Center)
    .Padding(FMargin(2, 0, 2, 0))
    [
        CreateCameraStatusBox("Depth",
            [this]() { return bHasDepthCamera; },
            [this]() { return bUseDepthCamera ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; },
            [this](ECheckBoxState NewState) { OnDepthCameraCheckboxChanged(NewState); })
    ]
    
    // Segmentation Camera
    +SHorizontalBox::Slot()
    .MaxWidth(100)
    .HAlign(HAlign_Center)
    .Padding(FMargin(2, 0, 2, 0))
    [
        CreateCameraStatusBox("Segment",
            [this]() { return bHasSegmentationCamera; },
            [this]() { return bUseSegmentationCamera ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; },
            [this](ECheckBoxState NewState) { OnSegmentationCameraCheckboxChanged(NewState); })
    ]
    
    // Normal Camera
    +SHorizontalBox::Slot()
    .MaxWidth(100)
    .HAlign(HAlign_Center)
    .Padding(FMargin(2, 0, 2, 0))
    [
        CreateCameraStatusBox("Normal",
            [this]() { return bHasNormalCamera; },
            [this]() { return bUseNormalCamera ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; },
            [this](ECheckBoxState NewState) { OnNormalCameraCheckboxChanged(NewState); })
    ]

    // Base Color Camera
    +SHorizontalBox::Slot()
    .MaxWidth(100)
    .HAlign(HAlign_Center)
    .Padding(FMargin(2, 0, 2, 0))
    [
        CreateCameraStatusBox("BaseColor",
            [this]() { return bHasBaseColorCamera; },
            [this]() { return bUseBaseColorCamera ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; },
            [this](ECheckBoxState NewState) { OnBaseColorCameraCheckboxChanged(NewState); })
    ]

    // Material Properties Camera
    +SHorizontalBox::Slot()
    .MaxWidth(100)
    .HAlign(HAlign_Center)
    .Padding(FMargin(2, 0, 0, 0))
    [
        CreateCameraStatusBox("MatProps",
            [this]() { return bHasMaterialPropertiesCamera; },
            [this]() { return bUseMaterialPropertiesCamera ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; },
            [this](ECheckBoxState NewState) { OnMaterialPropertiesCameraCheckboxChanged(NewState); })
    ];
}

TSharedRef<SWidget> FVCCSimPanelSelection::CreateCameraStatusBox(
    const FString& CameraName,
    TFunction<bool()> HasCameraFunc,
    TFunction<ECheckBoxState()> IsCheckedFunc,
    TFunction<void(ECheckBoxState)> OnStateChangedFunc)
{
    return SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(FColor(5, 5, 5, 255))
        .Padding(FMargin(4, 2))
        [
            SNew(SVerticalBox)
            +SVerticalBox::Slot()
            .AutoHeight()
            .HAlign(HAlign_Center)
            [
                SNew(STextBlock)
                .Text(FText::FromString(CameraName))
                .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
                .ColorAndOpacity_Lambda([HasCameraFunc]() {
                    return HasCameraFunc() ? FColor(233, 233, 233) : FColor(120, 120, 120);
                })
            ]
            +SVerticalBox::Slot()
            .AutoHeight()
            .HAlign(HAlign_Center)
            .Padding(FMargin(0, 2, 0, 0))
            [
                SNew(SCheckBox)
                .IsEnabled_Lambda([HasCameraFunc]() { return HasCameraFunc(); })
                .IsChecked_Lambda([IsCheckedFunc]() { return IsCheckedFunc(); })
                .ForegroundColor_Lambda([HasCameraFunc]() {
                    return HasCameraFunc() ? FSlateColor(FColor(0, 200, 0)) : FSlateColor(FColor(200, 0, 0));
                })
                .OnCheckStateChanged_Lambda([OnStateChangedFunc](ECheckBoxState NewState) {
                    OnStateChangedFunc(NewState);
                })
            ]
        ];
}


// ============================================================================
// EVENT HANDLERS
// ============================================================================

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

// ============================================================================
// SELECTION LOGIC
// ============================================================================

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

float FVCCSimPanelSelection::GetActiveCameraFOV() const
{
    if (!SelectedFlashPawn.IsValid())
    {
        return 90.0f; // Default FOV
    }
    
    // Priority order: RGB -> Depth -> Segmentation -> Normal
    if (bHasRGBCamera && bUseRGBCamera)
    {
        TArray<URGBCameraComponent*> RGBCameras;
        SelectedFlashPawn->GetComponents<URGBCameraComponent>(RGBCameras);
        if (RGBCameras.Num() > 0 && RGBCameras[0])
        {
            return RGBCameras[0]->FOV;
        }
    }
    
    if (bHasDepthCamera && bUseDepthCamera)
    {
        TArray<UDepthCameraComponent*> DepthCameras;
        SelectedFlashPawn->GetComponents<UDepthCameraComponent>(DepthCameras);
        if (DepthCameras.Num() > 0 && DepthCameras[0])
        {
            return DepthCameras[0]->FOV;
        }
    }
    
    if (bHasSegmentationCamera && bUseSegmentationCamera)
    {
        TArray<USegCameraComponent*> SegmentationCameras;
        SelectedFlashPawn->GetComponents<USegCameraComponent>(SegmentationCameras);
        if (SegmentationCameras.Num() > 0 && SegmentationCameras[0])
        {
            return SegmentationCameras[0]->FOV;
        }
    }
    
    if (bHasNormalCamera && bUseNormalCamera)
    {
        TArray<UNormalCameraComponent*> NormalCameras;
        SelectedFlashPawn->GetComponents<UNormalCameraComponent>(NormalCameras);
        if (NormalCameras.Num() > 0 && NormalCameras[0])
            return NormalCameras[0]->FOV;
    }

    if (bHasBaseColorCamera && bUseBaseColorCamera)
    {
        TArray<UBaseColorCameraComponent*> BaseColorCameras;
        SelectedFlashPawn->GetComponents<UBaseColorCameraComponent>(BaseColorCameras);
        if (BaseColorCameras.Num() > 0 && BaseColorCameras[0])
            return BaseColorCameras[0]->FOV;
    }

    if (bHasMaterialPropertiesCamera && bUseMaterialPropertiesCamera)
    {
        TArray<UMaterialPropertiesCameraComponent*> MatPropsCameras;
        SelectedFlashPawn->GetComponents<UMaterialPropertiesCameraComponent>(MatPropsCameras);
        if (MatPropsCameras.Num() > 0 && MatPropsCameras[0])
            return MatPropsCameras[0]->FOV;
    }

    return 90.0f;
}

FIntPoint FVCCSimPanelSelection::GetActiveCameraResolution() const
{
    if (!SelectedFlashPawn.IsValid())
    {
        return FIntPoint(1920, 1080); // Default resolution
    }
    
    // Priority order: RGB -> Depth -> Segmentation -> Normal
    if (bHasRGBCamera && bUseRGBCamera)
    {
        TArray<URGBCameraComponent*> RGBCameras;
        SelectedFlashPawn->GetComponents<URGBCameraComponent>(RGBCameras);
        if (RGBCameras.Num() > 0 && RGBCameras[0])
        {
            return FIntPoint(RGBCameras[0]->Width, RGBCameras[0]->Height);
        }
    }
    
    if (bHasDepthCamera && bUseDepthCamera)
    {
        TArray<UDepthCameraComponent*> DepthCameras;
        SelectedFlashPawn->GetComponents<UDepthCameraComponent>(DepthCameras);
        if (DepthCameras.Num() > 0 && DepthCameras[0])
        {
            return FIntPoint(DepthCameras[0]->Width, DepthCameras[0]->Height);
        }
    }
    
    if (bHasSegmentationCamera && bUseSegmentationCamera)
    {
        TArray<USegCameraComponent*> SegmentationCameras;
        SelectedFlashPawn->GetComponents<USegCameraComponent>(SegmentationCameras);
        if (SegmentationCameras.Num() > 0 && SegmentationCameras[0])
        {
            return FIntPoint(SegmentationCameras[0]->Width, SegmentationCameras[0]->Height);
        }
    }
    
    if (bHasNormalCamera && bUseNormalCamera)
    {
        TArray<UNormalCameraComponent*> NormalCameras;
        SelectedFlashPawn->GetComponents<UNormalCameraComponent>(NormalCameras);
        if (NormalCameras.Num() > 0 && NormalCameras[0])
            return FIntPoint(NormalCameras[0]->Width, NormalCameras[0]->Height);
    }

    if (bHasBaseColorCamera && bUseBaseColorCamera)
    {
        TArray<UBaseColorCameraComponent*> BaseColorCameras;
        SelectedFlashPawn->GetComponents<UBaseColorCameraComponent>(BaseColorCameras);
        if (BaseColorCameras.Num() > 0 && BaseColorCameras[0])
            return FIntPoint(BaseColorCameras[0]->Width, BaseColorCameras[0]->Height);
    }

    if (bHasMaterialPropertiesCamera && bUseMaterialPropertiesCamera)
    {
        TArray<UMaterialPropertiesCameraComponent*> MatPropsCameras;
        SelectedFlashPawn->GetComponents<UMaterialPropertiesCameraComponent>(MatPropsCameras);
        if (MatPropsCameras.Num() > 0 && MatPropsCameras[0])
            return FIntPoint(MatPropsCameras[0]->Width, MatPropsCameras[0]->Height);
    }

    return FIntPoint(1920, 1080);
}