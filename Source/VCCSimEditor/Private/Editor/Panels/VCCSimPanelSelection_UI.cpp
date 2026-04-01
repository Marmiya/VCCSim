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

#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Utils/VCCSimUIHelpers.h"
#include "Styling/AppStyle.h"

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

                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    CreatePawnSelectPanel()
                ]

                +SVerticalBox::Slot()
                .AutoHeight()
                .Padding(FMargin(0, 6, 0, 6))
                [
                    FVCCSimUIHelpers::CreateSeparator()
                ]

                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    CreateCameraSelectPanel()
                ]

                +SVerticalBox::Slot()
                .AutoHeight()
                .Padding(FMargin(0, 6, 0, 6))
                [
                    FVCCSimUIHelpers::CreateSeparator()
                ]

                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    CreateLookAtSelectPanel()
                ]
            ]
        ];
}

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

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 8, 0, 8))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        .Padding(FMargin(16, 0, 4, 0))
        [
            SNew(STextBlock)
            .Text(FText::FromString("RGB Camera:"))
            .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
            .ColorAndOpacity(FColor(233, 233, 233))
        ]
        +SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        .Padding(FMargin(0, 0, 12, 0))
        [
            SNew(SCheckBox)
            .IsChecked_Lambda([this]() { return bUseRGBCameraClass ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
            .OnCheckStateChanged(this, &FVCCSimPanelSelection::OnUseRGBCameraClassCheckboxChanged)
            .IsEnabled_Lambda([this]() { return bHasRGBCamera && bUseRGBCamera; })
        ]
        +SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        [
            SNew(SButton)
            .Text_Lambda([this]()
            {
                return FText::FromString(bIsWarmedUp ? "Cams Ready" : "Warmup Cams");
            })
            .IsEnabled_Lambda([this]() { return SelectedFlashPawn.IsValid(); })
            .OnClicked_Lambda([this]()
            {
                WarmupCameras();
                return FReply::Handled();
            })
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
