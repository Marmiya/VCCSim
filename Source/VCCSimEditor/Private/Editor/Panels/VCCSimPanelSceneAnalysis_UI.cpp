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

#include "Editor/Panels/VCCSimPanelSceneAnalysis.h"
#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Utils/VCCSimUIHelpers.h"
#include "Simulation/SceneAnalysisManager.h"
#include "Sensors/RGBCamera.h"
#include "Pawns/FlashPawn.h"
#include "Styling/AppStyle.h"

TSharedRef<SWidget> FVCCSimPanelSceneAnalysis::CreateSceneAnalysisPanel()
{
    return FVCCSimUIHelpers::CreateCollapsibleSection(
        "Scene Analysis",
        SNew(SVerticalBox)

    +SVerticalBox::Slot()
    .AutoHeight()
    [
        CreateLimitedRegionControls()
    ]

    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
       FVCCSimUIHelpers::CreateSeparator()
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 4, 0, 4))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        .Padding(FMargin(0, 0, 8, 0))
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat("Safe Distance", SafeDistanceSpinBox, SafeDistanceValue, SafeDistance, 0.0f, 10.0f)
        ]
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat("Safe Height", SafeHeightSpinBox, SafeHeightValue, SafeHeight, 0.0f, 5.0f)
        ]
    ]

    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        FVCCSimUIHelpers::CreateSeparator()
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(0, 4, 0, 4)
    [
        CreateSceneOperationButtons()
    ]

    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        FVCCSimUIHelpers::CreateSeparator()
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(0, 4, 0, 4)
    [
        CreateSafeZoneButtons()
    ]

    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        FVCCSimUIHelpers::CreateSeparator()
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(0, 4, 0, 4)
    [
        CreateCoverageButtons()
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
        CreateComplexityButtons()
    ],
    bSceneAnalysisSectionExpanded);
}

TSharedRef<SWidget> FVCCSimPanelSceneAnalysis::CreateLimitedRegionControls()
{
    return SNew(SVerticalBox)

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 4, 0, 4))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        .Padding(FMargin(0, 0, 8, 0))
        [
            FVCCSimUIHelpers::CreatePropertyRow(
                "Limited MinX",
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5, 5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(LimitedMinXSpinBox, SNumericEntryBox<float>)
                    .Value_Lambda([this]() { return LimitedMinXValue; })
                    .OnValueChanged(SNumericEntryBox<float>::FOnValueChanged::CreateLambda(
                        [this](float NewValue) {
                        LimitedMinX = NewValue;
                        LimitedMinXValue = NewValue;
                    }))
                ]
            )
        ]
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        [
            FVCCSimUIHelpers::CreatePropertyRow(
                "Limited MaxX",
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5, 5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(LimitedMaxXSpinBox, SNumericEntryBox<float>)
                    .Value_Lambda([this]() { return LimitedMaxXValue; })
                    .OnValueChanged(SNumericEntryBox<float>::FOnValueChanged::CreateLambda(
                        [this](float NewValue) {
                        LimitedMaxX = NewValue;
                        LimitedMaxXValue = NewValue;
                    }))
                ]
            )
        ]
    ]

    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        FVCCSimUIHelpers::CreateSeparator()
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 4, 0, 4))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        .Padding(FMargin(0, 0, 8, 0))
        [
            FVCCSimUIHelpers::CreatePropertyRow(
                "Limited MinY",
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5, 5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(LimitedMinYSpinBox, SNumericEntryBox<float>)
                    .Value_Lambda([this]() { return LimitedMinYValue; })
                    .OnValueChanged(SNumericEntryBox<float>::FOnValueChanged::CreateLambda(
                        [this](float NewValue) {
                        LimitedMinY = NewValue;
                        LimitedMinYValue = NewValue;
                    }))
                ]
            )
        ]
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        [
            FVCCSimUIHelpers::CreatePropertyRow(
                "Limited MaxY",
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5, 5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(LimitedMaxYSpinBox, SNumericEntryBox<float>)
                    .Value_Lambda([this]() { return LimitedMaxYValue; })
                    .OnValueChanged(SNumericEntryBox<float>::FOnValueChanged::CreateLambda(
                        [this](float NewValue) {
                        LimitedMaxY = NewValue;
                        LimitedMaxYValue = NewValue;
                    }))
                ]
            )
        ]
    ]

    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        FVCCSimUIHelpers::CreateSeparator()
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 4, 0, 0))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        .Padding(FMargin(0, 0, 8, 0))
        [
            FVCCSimUIHelpers::CreatePropertyRow(
                "Limited MinZ",
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5, 5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(LimitedMinZSpinBox, SNumericEntryBox<float>)
                    .Value_Lambda([this]() { return LimitedMinZValue; })
                    .OnValueChanged(SNumericEntryBox<float>::FOnValueChanged::CreateLambda(
                        [this](float NewValue) {
                        LimitedMinZ = NewValue;
                        LimitedMinZValue = NewValue;
                    }))
                ]
            )
        ]
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        [
            FVCCSimUIHelpers::CreatePropertyRow(
                "Limited MaxZ",
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5, 5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(LimitedMaxZSpinBox, SNumericEntryBox<float>)
                    .Value_Lambda([this]() { return LimitedMaxZValue; })
                    .OnValueChanged(SNumericEntryBox<float>::FOnValueChanged::CreateLambda(
                        [this](float NewValue) {
                        LimitedMaxZ = NewValue;
                        LimitedMaxZValue = NewValue;
                    }))
                ]
            )
        ]
    ];
}

TSharedRef<SWidget> FVCCSimPanelSceneAnalysis::CreateSceneOperationButtons()
{
    return SNew(SHorizontalBox)
    +SHorizontalBox::Slot()
    .MaxWidth(150)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(5, 2))
        .Text(FText::FromString("Scan Scene"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() {
            if (SceneAnalysisManager.IsValid())
            {
                if (bUseLimited)
                {
                    SceneAnalysisManager->ScanSceneRegion3D(
                        LimitedMinX, LimitedMaxX,
                        LimitedMinY, LimitedMaxY,
                        LimitedMinZ, LimitedMaxZ);
                }
                else
                {
                    SceneAnalysisManager->ScanScene();
                }
                bNeedScan = false;
            }
            return FReply::Handled();
        })
        .IsEnabled_Lambda([this]() {
            return SceneAnalysisManager.IsValid();
        })
    ]
    +SHorizontalBox::Slot()
    .MaxWidth(150)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(5, 2))
        .Text(FText::FromString("Register Camera"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() {
            if (SceneAnalysisManager.IsValid())
            {
                URGBCameraComponent* Camera = nullptr;
                if (auto PinnedSelectionManager = SelectionManager.Pin())
                {
                    if (PinnedSelectionManager->GetSelectedFlashPawn().IsValid())
                    {
                        Camera = PinnedSelectionManager->GetSelectedFlashPawn()->GetComponentByClass<URGBCameraComponent>();
                    }
                }
                if (Camera)
                {
                    Camera->CameraName = "CoverageCamera";
                    Camera->ComputeIntrinsics();
                    SceneAnalysisManager->RegisterCamera(Camera);
                }
                bInitCoverage = false;
            }
            return FReply::Handled();
        })
        .IsEnabled_Lambda([this]() {
            auto PinnedSelectionManager = SelectionManager.Pin();
            return SceneAnalysisManager.IsValid() &&
                   PinnedSelectionManager.IsValid() &&
                   PinnedSelectionManager->GetSelectedFlashPawn().IsValid();
        })
    ]
    +SHorizontalBox::Slot()
    .AutoWidth()
    .VAlign(VAlign_Center)
    .Padding(FMargin(8, 0, 4, 0))
    [
        SAssignNew(SelectUseLimitedToggle, SCheckBox)
        .IsChecked(bUseLimited ? ECheckBoxState::Checked : ECheckBoxState::Unchecked)
        .OnCheckStateChanged_Lambda([this](ECheckBoxState NewState) {
            OnUseLimitedToggleChanged(NewState);
        })
    ]
    +SHorizontalBox::Slot()
    .AutoWidth()
    .VAlign(VAlign_Center)
    [
        SNew(STextBlock)
        .Text(FText::FromString("Limited Region"))
    ];
}

TSharedRef<SWidget> FVCCSimPanelSceneAnalysis::CreateSafeZoneButtons()
{
    return SNew(SHorizontalBox)
    +SHorizontalBox::Slot()
    .MaxWidth(150)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(5, 2))
        .Text(FText::FromString("Gen SafeZone"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() {
            if (SceneAnalysisManager.IsValid())
            {
                SceneAnalysisManager->InterfaceGenerateSafeZone(SafeDistance);
                bGenSafeZone = false;
            }
            return FReply::Handled();
        })
        .IsEnabled_Lambda([this]() {
            return SceneAnalysisManager.IsValid() && !bNeedScan;
        })
    ]
    +SHorizontalBox::Slot()
    .MaxWidth(150)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SAssignNew(VisualizeSafeZoneButton, SButton)
        .ButtonStyle(bSafeZoneVisualized ?
           &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") :
           &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"))
        .ContentPadding(FMargin(0, 2))
        .HAlign(HAlign_Center)
        .Text_Lambda([this]() {
            return FText::FromString(bSafeZoneVisualized ? "Hide SafeZone" : "Show SafeZone");
        })
        .OnClicked_Lambda([this]() {
            return OnToggleSafeZoneVisualizationClicked();
        })
        .IsEnabled_Lambda([this]() {
            return SceneAnalysisManager.IsValid() && !bGenSafeZone;
        })
    ]
    +SHorizontalBox::Slot()
    .MaxWidth(150)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(0, 2))
        .Text(FText::FromString("Clear SafeZone"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() {
            if (SceneAnalysisManager.IsValid())
            {
                SceneAnalysisManager->InterfaceClearSafeZoneVisualization();
                bGenSafeZone = true;
                bSafeZoneVisualized = false;
                if (VisualizeSafeZoneButton.IsValid())
                {
                    VisualizeSafeZoneButton->SetButtonStyle(
                        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
                }
            }
            return FReply::Handled();
        })
        .IsEnabled_Lambda([this]() {
            return SceneAnalysisManager.IsValid() && !bGenSafeZone;
        })
    ];
}

TSharedRef<SWidget> FVCCSimPanelSceneAnalysis::CreateCoverageButtons()
{
    return SNew(SHorizontalBox)
    +SHorizontalBox::Slot()
    .MaxWidth(150)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(5, 2))
        .Text(FText::FromString("Gen Coverage"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() {
            if (SceneAnalysisManager.IsValid())
            {
                SceneAnalysisManager->InterfaceUpdateCoverageGrid();
                bGenCoverage = false;
            }
            return FReply::Handled();
        })
        .IsEnabled_Lambda([this]() {
            return SceneAnalysisManager.IsValid() && !bInitCoverage && !bNeedScan;
        })
    ]
    +SHorizontalBox::Slot()
    .MaxWidth(150)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SAssignNew(VisualizeCoverageButton, SButton)
        .ButtonStyle(bCoverageVisualized ?
            &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") :
            &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"))
        .ContentPadding(FMargin(0, 2))
        .HAlign(HAlign_Center)
        .Text_Lambda([this]() {
            return FText::FromString(bCoverageVisualized ? "Hide Coverage" : "Show Coverage");
        })
        .OnClicked_Lambda([this]() {
            return OnToggleCoverageVisualizationClicked();
        })
        .IsEnabled_Lambda([this]() {
            return SceneAnalysisManager.IsValid() && !bGenCoverage && !bInitCoverage;
        })
    ]
    +SHorizontalBox::Slot()
    .MaxWidth(150)
    .Padding(FMargin(0, 0, 0, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(0, 2))
        .Text(FText::FromString("Clear Coverage"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() {
            if (SceneAnalysisManager.IsValid())
            {
                SceneAnalysisManager->InterfaceClearCoverageVisualization();
                bGenCoverage = true;
                bCoverageVisualized = false;
                if (VisualizeCoverageButton.IsValid())
                {
                    VisualizeCoverageButton->SetButtonStyle(
                        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
                }
            }
            return FReply::Handled();
        })
        .IsEnabled_Lambda([this]() {
            return SceneAnalysisManager.IsValid() && !bGenCoverage && !bInitCoverage;
        })
    ];
}

TSharedRef<SWidget> FVCCSimPanelSceneAnalysis::CreateComplexityButtons()
{
    return SNew(SHorizontalBox)
    +SHorizontalBox::Slot()
    .MaxWidth(150)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(5, 2))
        .Text(FText::FromString("Analyze Scene"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() {
            if (SceneAnalysisManager.IsValid())
            {
                SceneAnalysisManager->InterfaceAnalyzeGeometricComplexity();
                bAnalyzeComplexity = false;
            }
            return FReply::Handled();
        })
        .IsEnabled_Lambda([this]() {
            return SceneAnalysisManager.IsValid() && !bNeedScan;
        })
    ]
    +SHorizontalBox::Slot()
    .MaxWidth(150)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SAssignNew(VisualizeComplexityButton, SButton)
        .ButtonStyle(bComplexityVisualized ?
            &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") :
            &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"))
        .ContentPadding(FMargin(0, 2))
        .HAlign(HAlign_Center)
        .Text_Lambda([this]() {
            return FText::FromString(bComplexityVisualized ? "Hide Complexity" : "Show Complexity");
        })
        .OnClicked_Lambda([this]() {
            return OnToggleComplexityVisualizationClicked();
        })
        .IsEnabled_Lambda([this]() {
            return SceneAnalysisManager.IsValid() && !bAnalyzeComplexity;
        })
    ]
    +SHorizontalBox::Slot()
    .MaxWidth(150)
    .Padding(FMargin(0, 0, 0, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(0, 2))
        .Text(FText::FromString("Clear Analysis"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() {
            if (SceneAnalysisManager.IsValid())
            {
                SceneAnalysisManager->InterfaceClearComplexityVisualization();
                bAnalyzeComplexity = true;
                bComplexityVisualized = false;
                if (VisualizeComplexityButton.IsValid())
                {
                    VisualizeComplexityButton->SetButtonStyle(
                        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
                }
            }
            return FReply::Handled();
        })
        .IsEnabled_Lambda([this]() {
            return SceneAnalysisManager.IsValid() && !bAnalyzeComplexity;
        })
    ];
}
