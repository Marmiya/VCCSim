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

                +SVerticalBox::Slot()
                .AutoHeight()
                .Padding(FMargin(0, 6, 0, 0))
                [
                    CreateGroundActorListPanel()
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
    .Padding(FMargin(0, 2, 0, 0))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot().FillWidth(1.f).VAlign(VAlign_Center)
        [
            SNew(SButton)
            .ContentPadding(FMargin(5, 2))
            .HAlign(HAlign_Center)
            .Text_Lambda([this]() { return FText::FromString(
                HighlightActor.IsValid() ? TEXT("Hide Highlight") : TEXT("Highlight Targets")); })
            .ToolTipText(FText::FromString(TEXT(
                "Toggle boxes around every list actor: green = enabled structure, brown = enabled "
                "ground/terrain, gray = disabled; thick cyan = detected building (gets an orbit); plus "
                "red boxes for in-box mesh actors not in the list. Click again to hide.")))
            .IsEnabled_Lambda([this]() { return TargetActorItems.Num() > 0 || HighlightActor.IsValid(); })
            .OnClicked_Lambda([this]() { return OnHighlightTargetsClicked(); })
        ]
        +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(4, 0, 0, 0))
        [
            SNew(SButton)
            .ContentPadding(FMargin(5, 2))
            .HAlign(HAlign_Center)
            .Text(FText::FromString(TEXT("Recompute")))
            .ToolTipText(FText::FromString(TEXT(
                "Force building detection to recompute, ignoring the cache. Use after a change the cache "
                "signature does not track — scaling an actor, or editing a mesh's collision/geometry "
                "without moving it. Refreshes the highlight if it is currently shown.")))
            .OnClicked_Lambda([this]() { return OnForceRecomputeClicked(); })
        ]
        +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(8, 0, 4, 0))
        [
            SNew(STextBlock)
            .Text(FText::FromString(TEXT("Connect gap (cm):")))
            .ToolTipText(FText::FromString(TEXT(
                "Two structure pieces are merged into one building only if their oriented boxes come "
                "within this distance of touching. Larger = pieces modelled with gaps still merge "
                "(fewer, bigger buildings); smaller = only near-touching pieces merge (road-side props "
                "stay separate). Shared by Highlight Targets and Generate Poses.")))
            .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
            .ColorAndOpacity(FColor(233, 233, 233))
        ]
        +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center)
        [
            SNew(SNumericEntryBox<double>)
            .AllowSpin(false)
            .MinDesiredValueWidth(46.f)
            .Value_Lambda([this]() { return TOptional<double>((double)ConnectGap); })
            .OnValueCommitted_Lambda([this](double V, ETextCommit::Type)
            {
                ConnectGap = FMath::Max(0.f, (float)V);
                SaveTargetActorsToConfig();
            })
        ]
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 2, 0, 0))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot().FillWidth(1.f).VAlign(VAlign_Center)
        [
            SNew(SButton)
            .ButtonStyle(FAppStyle::Get(), "FlatButton.Primary")
            .ContentPadding(FMargin(5, 2))
            .HAlign(HAlign_Center)
            .Text(FText::FromString(TEXT("Export Mesh")))
            .ToolTipText(FText::FromString(TEXT(
                "Export each enabled target actor as its own mesh.gltf + manifest under gt_materials/. "
                "The Python preprocess aggregates them into the combined scene mesh. No material baking — "
                "materials come from the camera captures.")))
            .IsEnabled_Lambda([this]() { return !bGTExportInProgress; })
            .OnClicked_Lambda([this]() { return OnExportGTMeshClicked(); })
        ]
        +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(4, 0, 0, 0))
        [
            SNew(SButton)
            .ContentPadding(FMargin(5, 2))
            .HAlign(HAlign_Center)
            .Text_Lambda([this]() { return FText::FromString(
                HiddenUnmatchedActors.Num() > 0 ? TEXT("Show Unmatched") : TEXT("Hide Unmatched")); })
            .ToolTipText(FText::FromString(TEXT(
                "Debug aid: temporarily hide every enabled actor that ended up GREEN in the highlight — "
                "non-ground actors not merged into any building. What stays visible is the matched "
                "buildings (and ground). Click again to restore.")))
            .IsEnabled_Lambda([this]() { return TargetActorItems.Num() > 0 || HiddenUnmatchedActors.Num() > 0; })
            .OnClicked_Lambda([this]() { return OnHideUnmatchedActorsClicked(); })
        ]
        +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(4, 0, 0, 0))
        [
            SNew(SButton)
            .ContentPadding(FMargin(5, 2))
            .HAlign(HAlign_Center)
            .Text_Lambda([this]() { return FText::FromString(
                HiddenGroundActors.Num() > 0 ? TEXT("Show Ground") : TEXT("Hide Ground")); })
            .ToolTipText(FText::FromString(TEXT(
                "Debug aid for building detection: temporarily hide every enabled list actor classified "
                "as ground (same test as Generate Poses). What stays visible is exactly the structure set "
                "DetectBuildings clusters — a street still visible between two buildings is a piece NOT "
                "recognised as ground that bridges them into one building. Click again to restore.")))
            .IsEnabled_Lambda([this]() { return TargetActorItems.Num() > 0 || HiddenGroundActors.Num() > 0; })
            .OnClicked_Lambda([this]() { return OnHideGroundActorsClicked(); })
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
            "Add every mesh actor whose bounds CENTRE falls inside this world-space box (a corner merely "
            "clipping the box does not count).")))
        .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
        .ColorAndOpacity(FColor(233, 233, 233))
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 1, 0, 0))
    [
        LabeledRow(TEXT(""),
            SNew(STextBlock).Text(FText::FromString(TEXT("X"))).Justification(ETextJustify::Center)
                .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont")).ColorAndOpacity(FColor(160, 160, 160)),
            SNew(STextBlock).Text(FText::FromString(TEXT("Y"))).Justification(ETextJustify::Center)
                .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont")).ColorAndOpacity(FColor(160, 160, 160)),
            SNew(STextBlock).Text(FText::FromString(TEXT("Z"))).Justification(ETextJustify::Center)
                .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont")).ColorAndOpacity(FColor(160, 160, 160)))
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 1))
    [
        LabeledRow(TEXT("Min"),
            CoordBox([this]() { return BoundsMin.X; }, [this](double V) { BoundsMin.X = V; SaveTargetActorsToConfig(); }),
            CoordBox([this]() { return BoundsMin.Y; }, [this](double V) { BoundsMin.Y = V; SaveTargetActorsToConfig(); }),
            CoordBox([this]() { return BoundsMin.Z; }, [this](double V) { BoundsMin.Z = V; SaveTargetActorsToConfig(); }))
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 1))
    [
        LabeledRow(TEXT("Max"),
            CoordBox([this]() { return BoundsMax.X; }, [this](double V) { BoundsMax.X = V; SaveTargetActorsToConfig(); }),
            CoordBox([this]() { return BoundsMax.Y; }, [this](double V) { BoundsMax.Y = V; SaveTargetActorsToConfig(); }),
            CoordBox([this]() { return BoundsMax.Z; }, [this](double V) { BoundsMax.Z = V; SaveTargetActorsToConfig(); }))
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
            SNew(STextBlock)
            .Text(FText::FromString(TEXT("Min building H/W (m):")))
            .ToolTipText(FText::FromString(TEXT(
                "A clustered target gets a facade orbit only if it is at least this tall (H) and this "
                "wide (W). Smaller things and large-flat ground are still surveyed, just not orbited.")))
            .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
            .ColorAndOpacity(FColor(233, 233, 233))
        ]
        +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(1, 0))
        [
            CoordBox([this]() { return MinBuildingHeight / 100.0; },
                     [this](double V) { MinBuildingHeight = FMath::Max(0.f, (float)(V * 100.0)); SaveTargetActorsToConfig(); })
        ]
        +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(1, 0))
        [
            CoordBox([this]() { return MinBuildingFootprint / 100.0; },
                     [this](double V) { MinBuildingFootprint = FMath::Max(0.f, (float)(V * 100.0)); SaveTargetActorsToConfig(); })
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
        .ToolTipText(FText::FromString(TEXT("Add every mesh actor inside the box to the target list (only our own capture pawns are skipped)")))
        .OnClicked_Lambda([this]() { return OnAddTargetActorsInBoundsClicked(); })
    ];
}

TSharedRef<SWidget> FVCCSimPanelSelection::CreateGroundActorListPanel()
{
    return SNew(SExpandableArea)
        .InitiallyCollapsed(!bGroundActorSectionExpanded)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(FColor(40, 40, 40))
        .OnAreaExpansionChanged_Lambda([this](bool bIsExpanded) { bGroundActorSectionExpanded = bIsExpanded; })
        .HeaderContent()
        [
            SNew(STextBlock)
            .Text(FText::FromString(TEXT("Ground Actors (manual override)")))
            .ToolTipText(FText::FromString(TEXT(
                "Actors listed here are forced to count as ground: dropped from building detection (so they "
                "never bridge two buildings) but still surveyed/captured. Use it for ground the automatic "
                "test misses — jagged or small-triangle tiles, or stepped surfaces with large height steps.")))
            .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
            .ColorAndOpacity(FColor(233, 233, 233))
        ]
        .BodyContent()
        [
            SNew(SVerticalBox)

            +SVerticalBox::Slot()
            .AutoHeight()
            .Padding(FMargin(0, 2, 0, 2))
            [
                SNew(SHorizontalBox)
                +SHorizontalBox::Slot().FillWidth(1.f)
                [
                    SNew(SButton)
                    .ButtonStyle(FAppStyle::Get(), "FlatButton.Success")
                    .ContentPadding(FMargin(5, 2))
                    .Text(FText::FromString(TEXT("+ Add Selected Actors")))
                    .ToolTipText(FText::FromString(TEXT("Mark the viewport-selected actors as ground")))
                    .OnClicked_Lambda([this]() { return OnAddGroundActorsClicked(); })
                ]
                +SHorizontalBox::Slot().AutoWidth().Padding(FMargin(4, 0, 0, 0))
                [
                    SNew(SButton)
                    .ButtonStyle(FAppStyle::Get(), "FlatButton.Danger")
                    .ContentPadding(FMargin(5, 2))
                    .Text(FText::FromString(TEXT("Clear All")))
                    .OnClicked_Lambda([this]() -> FReply
                    {
                        GroundActorItems.Empty();
                        if (GroundActorListView.IsValid())
                            GroundActorListView->RequestListRefresh();
                        SaveTargetActorsToConfig();
                        return FReply::Handled();
                    })
                ]
            ]

            +SVerticalBox::Slot()
            .AutoHeight()
            [
                SNew(SBox)
                .HeightOverride(70.f)
                [
                    SNew(SBorder)
                    .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                    .BorderBackgroundColor(FColor(10, 10, 10, 255))
                    .Padding(2)
                    [
                        SAssignNew(GroundActorListView, SListView<TSharedPtr<FString>>)
                        .ListItemsSource(&GroundActorItems)
                        .SelectionMode(ESelectionMode::None)
                        .OnGenerateRow_Lambda([this](TSharedPtr<FString> Item,
                            const TSharedRef<STableViewBase>& Owner) -> TSharedRef<ITableRow>
                        {
                            return SNew(STableRow<TSharedPtr<FString>>, Owner)
                            [
                                SNew(SHorizontalBox)
                                +SHorizontalBox::Slot().FillWidth(1.f).VAlign(VAlign_Center).Padding(FMargin(2, 0))
                                [
                                    SNew(STextBlock)
                                    .Text(FText::FromString(Item.IsValid() ? *Item : FString()))
                                    .ColorAndOpacity(FLinearColor(0.85f, 0.7f, 0.45f))
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
                                            GroundActorItems.Remove(Item);
                                            if (GroundActorListView.IsValid())
                                                GroundActorListView->RequestListRefresh();
                                            SaveTargetActorsToConfig();
                                        }
                                        return FReply::Handled();
                                    })
                                ]
                            ];
                        })
                    ]
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
        +SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        .Padding(FMargin(8, 0, 0, 0))
        [
            SNew(SButton)
            .Text(FText::FromString(TEXT("Fix Duplicate Labels")))
            .ToolTipText(FText::FromString(TEXT(
                "Rename actors that share a label so every label is unique (e.g. ..._System12 / "
                "..._System12_1). The selection, path generation and GT export all resolve actors by "
                "label, so duplicates silently drop all but one. Modifies the scene — save afterwards.")))
            .OnClicked_Lambda([this]() { return OnFixDuplicateLabelsClicked(); })
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
