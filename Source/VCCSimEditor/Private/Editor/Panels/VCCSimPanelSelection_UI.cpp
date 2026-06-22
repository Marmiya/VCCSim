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
#include "Styling/CoreStyle.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Views/STableRow.h"
#include "Widgets/Input/SNumericEntryBox.h"

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

                +SVerticalBox::Slot()
                .AutoHeight()
                .Padding(FMargin(0, 6, 0, 6))
                [
                    FVCCSimUIHelpers::CreateSeparator()
                ]

                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    CreateTargetActorListPanel()
                ]
            ]
        ];
}

TSharedRef<SWidget> FVCCSimPanelSelection::CreateTargetActorListPanel()
{
    return SNew(SVerticalBox)

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 0, 0, 4))
    [
        SNew(STextBlock)
        .Text(FText::FromString("Target Actors:"))
        .ToolTipText(FText::FromString(
            "Shared target list used by path generation and GT material export. "
            "Uncheck an entry to keep it in the list but skip it for the next task."))
        .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
        .ColorAndOpacity(FColor(233, 233, 233))
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 0, 0, 2))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot().FillWidth(1.f)
        [
            SNew(SButton)
            .ButtonStyle(FAppStyle::Get(), "FlatButton.Success")
            .ContentPadding(FMargin(5, 2))
            .Text(FText::FromString(TEXT("+ Add Selected Actors")))
            .ToolTipText(FText::FromString(TEXT("Add the actors currently selected in the viewport")))
            .OnClicked_Lambda([this]() { return OnAddTargetActorsClicked(); })
        ]
        +SHorizontalBox::Slot().AutoWidth().Padding(FMargin(4, 0, 0, 0))
        [
            SNew(SButton)
            .ButtonStyle(FAppStyle::Get(), "FlatButton.Danger")
            .ContentPadding(FMargin(5, 2))
            .Text(FText::FromString(TEXT("Clear All")))
            .OnClicked_Lambda([this]() -> FReply
            {
                TargetActorItems.Empty();
                if (TargetActorListView.IsValid())
                    TargetActorListView->RequestListRefresh();
                SaveTargetActorsToConfig();
                return FReply::Handled();
            })
        ]
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 4, 0, 4))
    [
        CreateBoundsSelectPanel()
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    [
        SNew(SBox)
        .HeightOverride(90.f)
        [
            SNew(SBorder)
            .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
            .BorderBackgroundColor(FColor(10, 10, 10, 255))
            .Padding(2)
            [
                SAssignNew(TargetActorListView, SListView<TSharedPtr<FVCCSimTargetActorItem>>)
                .ListItemsSource(&TargetActorItems)
                .SelectionMode(ESelectionMode::None)
                .OnGenerateRow_Lambda([this](TSharedPtr<FVCCSimTargetActorItem> Item,
                    const TSharedRef<STableViewBase>& Owner) -> TSharedRef<ITableRow>
                {
                    return SNew(STableRow<TSharedPtr<FVCCSimTargetActorItem>>, Owner)
                    [
                        SNew(SHorizontalBox)
                        +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(2, 0, 4, 0))
                        [
                            SNew(SCheckBox)
                            .ToolTipText(FText::FromString(TEXT("Process this actor in the next task")))
                            .IsChecked_Lambda([Item]()
                            {
                                return (Item.IsValid() && Item->bEnabled)
                                    ? ECheckBoxState::Checked : ECheckBoxState::Unchecked;
                            })
                            .OnCheckStateChanged_Lambda([this, Item](ECheckBoxState NewState)
                            {
                                if (Item.IsValid())
                                {
                                    Item->bEnabled = (NewState == ECheckBoxState::Checked);
                                    SaveTargetActorsToConfig();
                                }
                            })
                        ]
                        +SHorizontalBox::Slot().FillWidth(1.f).VAlign(VAlign_Center).Padding(FMargin(2, 0))
                        [
                            SNew(STextBlock)
                            .Text(FText::FromString(Item.IsValid() ? Item->Label : FString()))
                            .ColorAndOpacity_Lambda([Item]()
                            {
                                return (Item.IsValid() && Item->bEnabled)
                                    ? FLinearColor(0.8f, 0.9f, 0.8f)
                                    : FLinearColor(0.45f, 0.45f, 0.45f);
                            })
                            .Font(FCoreStyle::GetDefaultFontStyle("Mono", 8))
                        ]
                        +SHorizontalBox::Slot().AutoWidth()
                        [
                            SNew(SButton)
                            .ButtonStyle(FAppStyle::Get(), "FlatButton.Danger")
                            .ContentPadding(FMargin(4, 1))
                            .Text(FText::FromString(TEXT("×")))
                            .OnClicked_Lambda([this, Item]() -> FReply
                            {
                                if (Item.IsValid())
                                {
                                    TargetActorItems.Remove(Item);
                                    if (TargetActorListView.IsValid())
                                        TargetActorListView->RequestListRefresh();
                                    SaveTargetActorsToConfig();
                                }
                                return FReply::Handled();
                            })
                        ]
                    ];
                })
            ]
        ]
    ];
}

TSharedRef<SWidget> FVCCSimPanelSelection::CreateBoundsSelectPanel()
{
    auto CoordBox = [](TFunction<double()> Get, TFunction<void(double)> Set) -> TSharedRef<SWidget>
    {
        return SNew(SNumericEntryBox<double>)
            .AllowSpin(false)
            .MinDesiredValueWidth(46.f)
            .Value_Lambda([Get]() { return TOptional<double>(Get()); })
            .OnValueCommitted_Lambda([Set](double V, ETextCommit::Type) { Set(V); });
    };

    auto LabeledRow = [](const FString& Name,
        TSharedRef<SWidget> X, TSharedRef<SWidget> Y, TSharedRef<SWidget> Z) -> TSharedRef<SWidget>
    {
        return SNew(SHorizontalBox)
        +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(0, 0, 6, 0))
        [
            SNew(STextBlock)
            .Text(FText::FromString(Name))
            .MinDesiredWidth(28)
            .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
            .ColorAndOpacity(FColor(233, 233, 233))
        ]
        +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(1, 0)) [ X ]
        +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(1, 0)) [ Y ]
        +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(1, 0)) [ Z ];
    };

    return SNew(SVerticalBox)

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 0, 0, 2))
    [
        SNew(STextBlock)
        .Text(FText::FromString(TEXT("Add by Bounding Box (UE cm):")))
        .ToolTipText(FText::FromString(TEXT(
            "Add every mesh actor whose bounds center is inside this world-space box. "
            "Use 'Fill From Selection' to set the box from selected actors, then narrow it.")))
        .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
        .ColorAndOpacity(FColor(233, 233, 233))
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 1))
    [
        LabeledRow(TEXT("Min"),
            CoordBox([this]() { return BoundsMin.X; }, [this](double V) { BoundsMin.X = V; }),
            CoordBox([this]() { return BoundsMin.Y; }, [this](double V) { BoundsMin.Y = V; }),
            CoordBox([this]() { return BoundsMin.Z; }, [this](double V) { BoundsMin.Z = V; }))
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 1))
    [
        LabeledRow(TEXT("Max"),
            CoordBox([this]() { return BoundsMax.X; }, [this](double V) { BoundsMax.X = V; }),
            CoordBox([this]() { return BoundsMax.Y; }, [this](double V) { BoundsMax.Y = V; }),
            CoordBox([this]() { return BoundsMax.Z; }, [this](double V) { BoundsMax.Z = V; }))
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 3, 0, 1))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        .Padding(FMargin(0, 0, 6, 0))
        [
            SNew(SCheckBox)
            .IsChecked_Lambda([this]() { return bExcludeClutter ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
            .OnCheckStateChanged_Lambda([this](ECheckBoxState S) { bExcludeClutter = (S == ECheckBoxState::Checked); })
            .ToolTipText(FText::FromString(TEXT("Skip foliage / trees / vehicles / pedestrians by class and name")))
            [
                SNew(STextBlock)
                .Text(FText::FromString(TEXT("Exclude clutter")))
                .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
                .ColorAndOpacity(FColor(233, 233, 233))
            ]
        ]
        +SHorizontalBox::Slot()
        .FillWidth(1.f)
        .Padding(FMargin(4, 0, 0, 0))
        [
            SNew(SButton)
            .ContentPadding(FMargin(5, 2))
            .HAlign(HAlign_Center)
            .Text(FText::FromString(TEXT("Fill From Selection")))
            .ToolTipText(FText::FromString(TEXT("Set Min/Max from the AABB of the actors selected in the viewport")))
            .OnClicked_Lambda([this]() { return OnFillBoundsFromSelectionClicked(); })
        ]
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 1, 0, 0))
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Success")
        .ContentPadding(FMargin(5, 2))
        .HAlign(HAlign_Center)
        .Text(FText::FromString(TEXT("+ Add Actors In Bounds")))
        .ToolTipText(FText::FromString(TEXT("Add all non-clutter mesh actors inside the box to the target list")))
        .OnClicked_Lambda([this]() { return OnAddTargetActorsInBoundsClicked(); })
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
