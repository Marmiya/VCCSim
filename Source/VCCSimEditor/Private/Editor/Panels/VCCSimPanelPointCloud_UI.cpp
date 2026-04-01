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

#include "Editor/Panels/VCCSimPanelPointCloud.h"
#include "Utils/VCCSimUIHelpers.h"
#include "Styling/AppStyle.h"

TSharedRef<SWidget> FVCCSimPanelPointCloud::CreatePointCloudPanel()
{
    return SNew(SExpandableArea)
        .InitiallyCollapsed(!bPointCloudSectionExpanded)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryTop"))
        .BorderBackgroundColor(FColor(48, 48, 48))
        .OnAreaExpansionChanged_Lambda([this](bool bIsExpanded)
        {
            bPointCloudSectionExpanded = bIsExpanded;
        })
        .HeaderContent()
        [
            SNew(STextBlock)
            .Text(FText::FromString("Point Cloud Visualization"))
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

                + SVerticalBox::Slot()
                .AutoHeight()
                .Padding(5)
                [
                    SNew(SHorizontalBox)

                    + SHorizontalBox::Slot()
                    .FillWidth(1.0f)
                    [
                        SAssignNew(PointCloudStatusText, STextBlock)
                        .Text_Lambda([this]()
                        {
                            if (!PointCloudMgr->IsLoaded())
                            {
                                return FText::FromString("No point cloud loaded");
                            }
                            return FText::FromString(FString::Printf(TEXT("Loaded: %d points from %s"),
                                PointCloudMgr->GetPointCount(), *FPaths::GetBaseFilename(PointCloudMgr->GetLoadedPath())));
                        })
                    ]
                ]

                + SVerticalBox::Slot()
                .AutoHeight()
                .Padding(5)
                [
                    SNew(SHorizontalBox)

                    + SHorizontalBox::Slot()
                    .FillWidth(0.5f)
                    [
                        SAssignNew(PointCloudColorStatusText, STextBlock)
                        .Text_Lambda([this]()
                        {
                            return FText::FromString(PointCloudMgr->HasColors() ? "Colors: Available" : "Colors: Not Available");
                        })
                    ]

                    + SHorizontalBox::Slot()
                    .FillWidth(0.5f)
                    [
                        SAssignNew(PointCloudNormalStatusText, STextBlock)
                        .Text_Lambda([this]()
                        {
                            return FText::FromString(PointCloudMgr->HasNormals() ? "Normals: Available" : "Normals: Not Available");
                        })
                    ]
                ]

                + SVerticalBox::Slot()
                .MaxHeight(1)
                [
                    FVCCSimUIHelpers::CreateSeparator()
                ]

                + SVerticalBox::Slot()
                .AutoHeight()
                .Padding(5)
                [
                    CreatePointCloudButtons()
                ]

                + SVerticalBox::Slot()
                .MaxHeight(1)
                [
                    FVCCSimUIHelpers::CreateSeparator()
                ]

                + SVerticalBox::Slot()
                .AutoHeight()
                .Padding(5)
                [
                    CreatePointCloudNormalControls()
                ]
            ]
        ];
}

TSharedRef<SWidget> FVCCSimPanelPointCloud::CreatePointCloudButtons()
{
    return SNew(SHorizontalBox)

        + SHorizontalBox::Slot()
        .FillWidth(0.33f)
        .Padding(2)
        [
            SAssignNew(LoadPointCloudButton, SButton)
            .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
            .ContentPadding(FMargin(4, 2))
            .Text(FText::FromString("Load Point Cloud"))
            .OnClicked_Lambda([this]() { return OnLoadPointCloudClicked(); })
        ]

        + SHorizontalBox::Slot()
        .FillWidth(0.33f)
        .Padding(2)
        [
            SAssignNew(VisualizePointCloudButton, SButton)
            .ButtonStyle(FAppStyle::Get(), "FlatButton.Primary")
            .ContentPadding(FMargin(4, 2))
            .Text_Lambda([this]()
            {
                return FText::FromString(PointCloudMgr->IsVisualized() ? "Hide Point Cloud" : "Show Point Cloud");
            })
            .IsEnabled_Lambda([this]()
            {
                return PointCloudMgr->IsLoaded();
            })
            .OnClicked_Lambda([this]() { return OnTogglePointCloudVisualizationClicked(); })
        ]

        + SHorizontalBox::Slot()
        .FillWidth(0.33f)
        .Padding(2)
        [
            SNew(SButton)
            .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
            .ContentPadding(FMargin(4, 2))
            .Text(FText::FromString("Clear Point Cloud"))
            .IsEnabled_Lambda([this]()
            {
                return PointCloudMgr->IsLoaded();
            })
            .OnClicked_Lambda([this]() { return OnClearPointCloudClicked(); })
        ];
}

TSharedRef<SWidget> FVCCSimPanelPointCloud::CreatePointCloudNormalControls()
{
    return SNew(SHorizontalBox)

        + SHorizontalBox::Slot()
        .FillWidth(0.5f)
        .Padding(2)
        [
            SAssignNew(ShowColorsCheckBox, SCheckBox)
            .Content()
            [
                SNew(STextBlock).Text(FText::FromString("Show Colors"))
            ]
            .IsChecked_Lambda([this]()
            {
                return bShowColors ? ECheckBoxState::Checked : ECheckBoxState::Unchecked;
            })
            .IsEnabled_Lambda([this]()
            {
                return PointCloudMgr->IsLoaded() && PointCloudMgr->HasColors();
            })
            .OnCheckStateChanged_Lambda([this](ECheckBoxState NewState) { OnShowColorsChanged(NewState); })
        ]

        + SHorizontalBox::Slot()
        .FillWidth(0.5f)
        .Padding(2)
        [
            SAssignNew(ShowNormalsCheckBox, SCheckBox)
            .Content()
            [
                SNew(STextBlock).Text(FText::FromString("Show Normals"))
            ]
            .IsChecked_Lambda([this]()
            {
                return bShowNormals ? ECheckBoxState::Checked : ECheckBoxState::Unchecked;
            })
            .IsEnabled_Lambda([this]()
            {
                return PointCloudMgr->IsLoaded() && PointCloudMgr->HasNormals();
            })
            .OnCheckStateChanged_Lambda([this](ECheckBoxState NewState) { OnShowNormalsChanged(NewState); })
        ];
}
