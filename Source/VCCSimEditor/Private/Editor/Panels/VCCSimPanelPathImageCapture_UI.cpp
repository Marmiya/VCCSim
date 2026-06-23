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

#include "Editor/Panels/VCCSimPanelPathImageCapture.h"
#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Utils/VCCSimUIHelpers.h"
#include "Pawns/FlashPawn.h"
#include "Styling/AppStyle.h"
#include "Styling/CoreStyle.h"
#include "Widgets/Layout/SBorder.h"

// ============================================================================
// UI CONSTRUCTION
// ============================================================================

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreatePathImageCapturePanel()
{
    return FVCCSimUIHelpers::CreateCollapsibleSection(
        "Path Configuration & Image Capture", 
        SNew(SVerticalBox)
        
        // Path Configuration Section
        +SVerticalBox::Slot()
        .AutoHeight()
        [
            CreatePathConfigSection()
        ]
        
        +SVerticalBox::Slot()
        .MaxHeight(1)
        .Padding(FMargin(0, 8, 0, 8))
        [
           FVCCSimUIHelpers::CreateSeparator()
        ]
        
        // Image Capture Section
        +SVerticalBox::Slot()
        .AutoHeight()
        [
            CreateImageCaptureSection()
        ],
        
        bPathImageCaptureSectionExpanded
    );
}

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreatePathConfigSection()
{
    return SNew(SVerticalBox)

    // Orbit parameters (targets come from the shared list in the Object Selection panel)
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 4, 0, 4))
    [
        FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
            TEXT("Margin (cm)"), OrbitMarginSpinBox, OrbitMarginValue, OrbitMargin, 50.f, 50.f)
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 0, 0, 4))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 4, 0))
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
                TEXT("H-FOV (deg)"), OrbitCameraHFOVSpinBox, OrbitCameraHFOVValue, OrbitCameraHFOV, 5.f, 5.f)
        ]
        +SHorizontalBox::Slot().FillWidth(1.f)
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
                TEXT("Start H (cm)"), OrbitStartHeightSpinBox, OrbitStartHeightValue, OrbitStartHeight, 0.f, 50.f)
        ]
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 0, 0, 4))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 4, 0))
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
                TEXT("H-Overlap"), OrbitHOverlapSpinBox, OrbitHOverlapValue, OrbitHOverlap, 0.f, 0.05f)
        ]
        +SHorizontalBox::Slot().FillWidth(1.f)
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
                TEXT("V-Overlap"), OrbitVOverlapSpinBox, OrbitVOverlapValue, OrbitVOverlap, 0.f, 0.05f)
        ]
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 0, 0, 4))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(0, 0, 4, 0))
        [
            SNew(SCheckBox)
            .IsChecked_Lambda([this]() { return bOrbitIncludeNadir ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
            .OnCheckStateChanged_Lambda([this](ECheckBoxState S) { bOrbitIncludeNadir = (S == ECheckBoxState::Checked); })
        ]
        +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 4, 0))
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
                TEXT("Nadir Alt (cm)"), OrbitNadirAltSpinBox, OrbitNadirAltValue, OrbitNadirAlt, 100.f, 100.f)
        ]
        +SHorizontalBox::Slot().FillWidth(1.f)
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
                TEXT("Tilt (deg)"), OrbitNadirTiltSpinBox, OrbitNadirTiltValue, OrbitNadirTiltAngle, 0.f, 1.f)
        ]
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 0, 0, 4))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 8, 0))
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowInt32(
                TEXT("Oblique Rings"), OrbitObliqueRingsSpinBox, OrbitObliqueRingsValue, OrbitObliqueRings, 0, 1)
        ]
        +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(0, 0, 4, 0))
        [
            SNew(SCheckBox)
            .ToolTipText(FText::FromString(TEXT(
                "5-lens oblique survey: a straight-down nadir pass PLUS four passes tilted by 'Tilt (deg)' "
                "toward N/S/E/W (~5x the survey poses). Off = a single lens pitched by 'Tilt (deg)' (set "
                "Tilt to 0 for straight down). Strip spacing always uses the tilted footprint to hold overlap.")))
            .IsChecked_Lambda([this]() { return bOrbitIncludeOblique ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
            .OnCheckStateChanged_Lambda([this](ECheckBoxState S) { bOrbitIncludeOblique = (S == ECheckBoxState::Checked); })
        ]
        +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(0, 0, 12, 0))
        [
            SNew(STextBlock).Text(FText::FromString(TEXT("Oblique")))
        ]
        +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(0, 0, 4, 0))
        [
            SNew(SCheckBox)
            .ToolTipText(FText::FromString(TEXT(
                "Also generate the per-building facade orbit rings. Off = region oblique/nadir survey only.")))
            .IsChecked_Lambda([this]() { return bOrbitSideOrbit ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
            .OnCheckStateChanged_Lambda([this](ECheckBoxState S) { bOrbitSideOrbit = (S == ECheckBoxState::Checked); })
        ]
        +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center)
        [
            SNew(STextBlock).Text(FText::FromString(TEXT("Side Orbit")))
        ]
    ]

    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        FVCCSimUIHelpers::CreateSeparator()
    ]

    // Load/Save Pose buttons
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 4, 0, 4))
    [
        CreatePoseFileButtons()
    ]

    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        FVCCSimUIHelpers::CreateSeparator()
    ]

    // Action buttons
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 4, 0, 2))
    [
        CreatePoseActionButtons()
    ];
}

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreateImageCaptureSection()
{
    return SNew(SVerticalBox)

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(0, 0, 0, 4)
    [
        FVCCSimUIHelpers::CreatePropertyRow(TEXT("Tick (s)"),
            SNew(SBorder)
            .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
            .BorderBackgroundColor(FColor(5, 5, 5, 255))
            .Padding(4, 0)
            [
                SAssignNew(CaptureTickIntervalSpinBox, SNumericEntryBox<float>)
                .Value_Lambda([this]() { return CaptureTickIntervalValue; })
                .MinValue(0.05f).MaxValue(5.f).Delta(0.05f).AllowSpin(true)
                .ToolTipText(FText::FromString(TEXT(
                    "Interval between capture ticks during auto/dataset capture. "
                    "Takes effect immediately, even while a capture is running.")))
                .OnValueChanged_Lambda([this](float Val)
                {
                    CaptureTickInterval = FMath::Clamp(Val, 0.05f, 5.f);
                    CaptureTickIntervalValue = CaptureTickInterval;
                    if (bAutoCaptureInProgress && GEditor)
                    {
                        GEditor->GetTimerManager()->SetTimer(
                            AutoCaptureTimerHandle,
                            FTimerDelegate::CreateLambda([this]() { TickCaptureSession(); }),
                            CaptureTickInterval,
                            true);
                    }
                })
            ]
        )
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(0, 0, 0, 4)
    [
        CreateMovementButtons()
    ]
    
    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        FVCCSimUIHelpers::CreateSeparator()
    ]
    
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(0, 4, 0, 0)
    [
        CreateCaptureButtons()
    ];
}

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreatePoseFileButtons()
{
    return SNew(SHorizontalBox)
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(4, 2))
        .Text(FText::FromString("Load Predefined Pose"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() { return OnLoadPoseClicked(); })
        .IsEnabled_Lambda([this]() {
            return SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid();
        })
    ]
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(4, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(4, 2))
        .HAlign(HAlign_Center)
        .Text(FText::FromString("Save Generated Pose"))
        .OnClicked_Lambda([this]() { return OnSavePoseClicked(); })
        .IsEnabled_Lambda([this]() {
            return SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid() && 
                   SelectionManager.Pin()->GetSelectedFlashPawn()->GetPoseCount() > 0;
        })
    ];
}
TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreatePoseActionButtons()
{
    return SNew(SHorizontalBox)
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Success")
        .ContentPadding(FMargin(5, 2))
        .Text(FText::FromString("Generate Poses"))
        .HAlign(HAlign_Center)
        .ToolTipText(FText::FromString(TEXT("Orbit targets come from the Target Actors list in the Object Selection panel")))
        .OnClicked_Lambda([this]() { return OnGeneratePosesClicked(); })
        .IsEnabled_Lambda([this]() {
            return SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid() &&
                SelectionManager.Pin()->HasEnabledTargetActors();
        })
    ]
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(4, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SAssignNew(VisualizePathButton, SButton)
        .ButtonStyle(bPathVisualized ? 
           &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") : 
           &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"))
        .ContentPadding(FMargin(5, 2))
        .HAlign(HAlign_Center)
        .Text_Lambda([this]() {
            return FText::FromString(bPathVisualized ? "Hide Path" : "Show Path");
        })
        .OnClicked_Lambda([this]() { return OnTogglePathVisualizationClicked(); })
        .IsEnabled_Lambda([this]() {
            return SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid() && !bPathNeedsUpdate;
        })
    ];
}

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreateMovementButtons()
{
    return SNew(SHorizontalBox)
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(5, 2))
        .Text(FText::FromString("Move Back"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() {
            if (SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid())
            {
                AFlashPawn* Pawn = SelectionManager.Pin()->GetSelectedFlashPawn().Get();
                Pawn->MoveBackward();
                Pawn->PostEditMove(true);
            }
            return FReply::Handled();
        })
        .IsEnabled_Lambda([this]() {
            return SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid();
        })
    ]
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(4, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(5, 2))
        .Text(FText::FromString("Move Next"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() {
            if (SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid())
            {
                AFlashPawn* Pawn = SelectionManager.Pin()->GetSelectedFlashPawn().Get();
                Pawn->MoveForward();
                Pawn->PostEditMove(true);
            }
            return FReply::Handled();
        })
        .IsEnabled_Lambda([this]() {
            return SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid();
        })
    ];
}

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreateCaptureButtons()
{
    return SNew(SHorizontalBox)
    
    // Single Capture button
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Success")
        .ContentPadding(FMargin(5, 2))
        .Text(FText::FromString("Capture This Pose"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() { return OnCaptureImagesClicked(); })
        .IsEnabled_Lambda([this]() {
            if (!SelectionManager.IsValid()) return false;
            auto SelectionManagerPin = SelectionManager.Pin();
            if (!SelectionManagerPin.IsValid() || !SelectionManagerPin->GetSelectedFlashPawn().IsValid()) return false;
            
            return (SelectionManagerPin->IsUsingRGBCamera() && SelectionManagerPin->HasRGBCamera()) ||
                   (SelectionManagerPin->IsUsingDepthCamera() && SelectionManagerPin->HasDepthCamera()) ||
                   (SelectionManagerPin->IsUsingNormalCamera() && SelectionManagerPin->HasNormalCamera()) ||
                   (SelectionManagerPin->IsUsingSegmentationCamera() && SelectionManagerPin->HasSegmentationCamera()) ||
                   (SelectionManagerPin->IsUsingBaseColorCamera() && SelectionManagerPin->HasBaseColorCamera()) ||
                   (SelectionManagerPin->IsUsingMaterialPropertiesCamera() && SelectionManagerPin->HasMaterialPropertiesCamera());
        })
    ]
    
    // Auto Capture button
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(4, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SAssignNew(AutoCaptureButton, SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Primary")
        .ContentPadding(FMargin(5, 2))
        .Text_Lambda([this]() {
            return bAutoCaptureInProgress ? 
                FText::FromString("Stop Capture") :
                FText::FromString("Capture All Poses");
        })
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() {
            if (bAutoCaptureInProgress)
            {
                StopAutoCapture();
                AutoCaptureButton->SetButtonStyle(&FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
            }
            else
            {
                StartAutoCapture();
                AutoCaptureButton->SetButtonStyle(&FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger"));
            }
            return FReply::Handled();
        })
        .IsEnabled_Lambda([this]() {
            if (!SelectionManager.IsValid()) return false;
            auto SelectionManagerPin = SelectionManager.Pin();
            if (!SelectionManagerPin.IsValid() || !SelectionManagerPin->GetSelectedFlashPawn().IsValid()) return false;
            
            return (SelectionManagerPin->IsUsingRGBCamera() && SelectionManagerPin->HasRGBCamera()) || 
                   (SelectionManagerPin->IsUsingDepthCamera() && SelectionManagerPin->HasDepthCamera()) || 
                   (SelectionManagerPin->IsUsingSegmentationCamera() && SelectionManagerPin->HasSegmentationCamera()) ||
                   (SelectionManagerPin->IsUsingNormalCamera() && SelectionManagerPin->HasNormalCamera());
        })
    ];
}
