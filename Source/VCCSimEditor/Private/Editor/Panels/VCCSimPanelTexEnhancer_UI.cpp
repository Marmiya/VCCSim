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

#include "Editor/Panels/VCCSimPanelTexEnhancer.h"
#include "Utils/VCCSimUIHelpers.h"

#include "Widgets/Layout/SBorder.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Layout/SSpacer.h"
#include "Widgets/Layout/SScrollBox.h"
#include "Widgets/SBoxPanel.h"
#include "Styling/AppStyle.h"

DEFINE_LOG_CATEGORY_STATIC(LogTexEnhancerUI, Log, All);

// ============================================================================
// MAIN PANEL ENTRY POINT
// ============================================================================

TSharedRef<SWidget> FVCCSimPanelTexEnhancer::CreateTexEnhancerPanel()
{
    return FVCCSimUIHelpers::CreateCollapsibleSection(
        TEXT("TexEnhancer: Data Generation & Evaluation"),
        SNew(SVerticalBox)

        +SVerticalBox::Slot().AutoHeight()
        [ CreateDatasetConfigSection() ]

        +SVerticalBox::Slot().MaxHeight(1).Padding(FMargin(0, 4, 0, 4))
        [ FVCCSimUIHelpers::CreateSeparator() ]

        +SVerticalBox::Slot().AutoHeight()
        [ CreateLightingScheduleSection() ]

        +SVerticalBox::Slot().MaxHeight(1).Padding(FMargin(0, 4, 0, 4))
        [ FVCCSimUIHelpers::CreateSeparator() ]

        +SVerticalBox::Slot().AutoHeight()
        [ CreateGTExportSection() ]

        +SVerticalBox::Slot().MaxHeight(1).Padding(FMargin(0, 4, 0, 4))
        [ FVCCSimUIHelpers::CreateSeparator() ]

        +SVerticalBox::Slot().AutoHeight()
        [ CreateNanobananaSection() ]

        +SVerticalBox::Slot().MaxHeight(1).Padding(FMargin(0, 4, 0, 4))
        [ FVCCSimUIHelpers::CreateSeparator() ]

        +SVerticalBox::Slot().AutoHeight()
        [ CreatePipelineSection() ]

        +SVerticalBox::Slot().MaxHeight(1).Padding(FMargin(0, 4, 0, 4))
        [ FVCCSimUIHelpers::CreateSeparator() ]

        +SVerticalBox::Slot().AutoHeight()
        [ CreateEvaluationSection() ]

        +SVerticalBox::Slot().MaxHeight(1).Padding(FMargin(0, 4, 0, 4))
        [ FVCCSimUIHelpers::CreateSeparator() ],

        bSectionExpanded
    );
}

// ============================================================================
// SECTION 1: DATASET CONFIGURATION
// ============================================================================

TSharedRef<SWidget> FVCCSimPanelTexEnhancer::CreateDatasetConfigSection()
{
    return FVCCSimUIHelpers::CreateSectionContent(
        SNew(SVerticalBox)

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            FVCCSimUIHelpers::CreateSectionHeader(TEXT("Dataset Configuration"))
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            FVCCSimUIHelpers::CreatePropertyRow(TEXT("Output Dir"),
                SNew(SHorizontalBox)
                +SHorizontalBox::Slot().FillWidth(1.f)
                [
                    SAssignNew(OutputDirTextBox, SEditableTextBox)
                    .Text(FText::FromString(OutputDirectory))
                    .OnTextCommitted_Lambda([this](const FText& Text, ETextCommit::Type)
                    {
                        OutputDirectory = Text.ToString();
                        SavePaths();
                    })
                ]
                +SHorizontalBox::Slot().AutoWidth().Padding(FMargin(4, 0, 0, 0))
                [
                    SNew(SButton)
                    .ButtonStyle(FAppStyle::Get(), "FlatButton")
                    .ContentPadding(FMargin(5, 2))
                    .Text(FText::FromString(TEXT("...")))
                    .OnClicked_Lambda([this]() { return OnBrowseOutputDirClicked(); })
                ]
            )
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            FVCCSimUIHelpers::CreatePropertyRow(TEXT("Scene Name"),
                SAssignNew(SceneNameTextBox, SEditableTextBox)
                .Text(FText::FromString(SceneName))
                .OnTextCommitted_Lambda([this](const FText& Text, ETextCommit::Type)
                {
                    SceneName = Text.ToString();
                    SavePaths();
                })
            )
        ]
    );
}

// ============================================================================
// SECTION 2: LIGHTING SCHEDULE
// ============================================================================

TSharedRef<SWidget> FVCCSimPanelTexEnhancer::CreateLightingScheduleSection()
{
    TSharedRef<SVerticalBox> SetAEntries = SNew(SVerticalBox);
    for (int32 i = 0; i < NumLightingSetA; ++i)
    {
        SetAEntries->AddSlot().AutoHeight().Padding(FMargin(0, 1))
        [
            CreateSetALightingEntry(i)
        ];
    }

    TSharedRef<SVerticalBox> SetBEntries = SNew(SVerticalBox);
    for (int32 i = 0; i < NumLightingSetB; ++i)
    {
        SetBEntries->AddSlot().AutoHeight().Padding(FMargin(0, 1))
        [
            CreateSetBLightingEntry(i)
        ];
    }

    return FVCCSimUIHelpers::CreateCollapsibleSection(TEXT("Lighting Schedule"),
        SNew(SVerticalBox)

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            SAssignNew(LightingStatusTextBlock, STextBlock)
            .Text(FText::GetEmpty())
            .ColorAndOpacity(FLinearColor(0.6f, 0.9f, 0.6f))
            .Font(FCoreStyle::GetDefaultFontStyle("Mono", 8))
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            SNew(SBorder)
            .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
            .BorderBackgroundColor(FColor(20, 40, 80, 255))
            .Padding(FMargin(6, 3))
            [
                SNew(SVerticalBox)
                +SVerticalBox::Slot().AutoHeight()
                [
                    SNew(STextBlock)
                    .Text(FText::FromString(TEXT("Set-A  (Estimation)")))
                    .ColorAndOpacity(FLinearColor(0.5f, 0.8f, 1.f))
                    .Font(FCoreStyle::GetDefaultFontStyle("Bold", 9))
                ]
                +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2, 0, 0))
                [
                    SetAEntries
                ]
            ]
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            SNew(SBorder)
            .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
            .BorderBackgroundColor(FColor(80, 40, 10, 255))
            .Padding(FMargin(6, 3))
            [
                SNew(SVerticalBox)
                +SVerticalBox::Slot().AutoHeight()
                [
                    SNew(SHorizontalBox)
                    +SHorizontalBox::Slot().FillWidth(1.f)
                    [
                        SNew(STextBlock)
                        .Text(FText::FromString(TEXT("Set-B  (Evaluation — Held-out)")))
                        .ColorAndOpacity(FLinearColor(1.f, 0.6f, 0.2f))
                        .Font(FCoreStyle::GetDefaultFontStyle("Bold", 9))
                    ]
                ]
                +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2, 0, 0))
                [
                    SetBEntries
                ]
            ]
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [ CreateSunPositionCalculatorWidget() ],
        bLightingScheduleExpanded);
}

TSharedRef<SWidget> FVCCSimPanelTexEnhancer::CreateSunPositionCalculatorWidget()
{
    auto MakeInlineSpinBoxFloat = [this](
        const FString& LabelText,
        TSharedPtr<SNumericEntryBox<float>>& SpinBoxPtr,
        TOptional<float>& ValueOpt,
        float& Var,
        float MinVal, float MaxVal, float Delta) -> TSharedRef<SWidget>
    {
        return SNew(SHorizontalBox)
            +SHorizontalBox::Slot().MaxWidth(76).VAlign(VAlign_Center).Padding(FMargin(0, 0, 4, 0))
            [
                SNew(STextBlock)
                .Text(FText::FromString(LabelText))
                .ColorAndOpacity(FLinearColor(0.85f, 0.85f, 0.85f))
                .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
            ]
            +SHorizontalBox::Slot().FillWidth(1.f)
            [
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5, 5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(SpinBoxPtr, SNumericEntryBox<float>)
                    .Value_Lambda([&ValueOpt]() { return ValueOpt; })
                    .MinValue(MinVal).MaxValue(MaxVal).Delta(Delta).AllowSpin(false)
                    .OnValueChanged_Lambda([&ValueOpt, &Var, MinVal, MaxVal](float Val)
                    {
                        Val = FMath::Clamp(Val, MinVal, MaxVal);
                        Var = Val;
                        ValueOpt = Val;
                    })
                ]
            ];
    };

    auto MakeInlineSpinBoxInt = [this](
        const FString& LabelText,
        TSharedPtr<SNumericEntryBox<int32>>& SpinBoxPtr,
        TOptional<int32>& ValueOpt,
        int32& Var,
        int32 MinVal, int32 MaxVal, int32 Delta) -> TSharedRef<SWidget>
    {
        return SNew(SHorizontalBox)
            +SHorizontalBox::Slot().MaxWidth(40).VAlign(VAlign_Center).Padding(FMargin(0, 0, 2, 0))
            [
                SNew(STextBlock)
                .Text(FText::FromString(LabelText))
                .ColorAndOpacity(FLinearColor(0.85f, 0.85f, 0.85f))
                .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
            ]
            +SHorizontalBox::Slot().FillWidth(1.f)
            [
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5, 5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(SpinBoxPtr, SNumericEntryBox<int32>)
                    .Value_Lambda([&ValueOpt]() { return ValueOpt; })
                    .MinValue(MinVal).MaxValue(MaxVal).Delta(Delta).AllowSpin(false)
                    .OnValueChanged_Lambda([&ValueOpt, &Var, MinVal, MaxVal](int32 Val)
                    {
                        Val = FMath::Clamp(Val, MinVal, MaxVal);
                        Var = Val;
                        ValueOpt = Val;
                    })
                ]
            ];
    };

    return SNew(SBorder)
    .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
    .BorderBackgroundColor(FColor(35, 20, 55, 255))
    .Padding(FMargin(6, 4))
    [
        SNew(SVerticalBox)

        +SVerticalBox::Slot().AutoHeight()
        [
            SNew(STextBlock)
            .Text(FText::FromString(TEXT("Sun Position Calculator")))
            .ColorAndOpacity(FLinearColor(0.75f, 0.5f, 1.f))
            .Font(FCoreStyle::GetDefaultFontStyle("Bold", 9))
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            MakeInlineSpinBoxFloat(TEXT("Latitude"),
                SunCalcLatSpinBox, SunCalcLatValue, SunCalcLatitude, -90.f, 90.f, 0.1f)
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            MakeInlineSpinBoxFloat(TEXT("Longitude"),
                SunCalcLonSpinBox, SunCalcLonValue, SunCalcLongitude, -180.f, 180.f, 0.1f)
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            MakeInlineSpinBoxFloat(TEXT("TZ (UTC\u00B1)"),
                SunCalcTZSpinBox, SunCalcTZValue, SunCalcTimeZone, -12.f, 14.f, 0.5f)
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            SNew(SHorizontalBox)

            +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 4, 0))
            [
                MakeInlineSpinBoxInt(TEXT("Year"),
                    SunCalcYearSpinBox, SunCalcYearValue, SunCalcYear, 1900, 2100, 1)
            ]

            +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 4, 0))
            [
                MakeInlineSpinBoxInt(TEXT("Month"),
                    SunCalcMonthSpinBox, SunCalcMonthValue, SunCalcMonth, 1, 12, 1)
            ]

            +SHorizontalBox::Slot().FillWidth(1.f)
            [
                MakeInlineSpinBoxInt(TEXT("Day"),
                    SunCalcDaySpinBox, SunCalcDayValue, SunCalcDay, 1, 31, 1)
            ]
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            SNew(SHorizontalBox)

            +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 4, 0))
            [
                MakeInlineSpinBoxInt(TEXT("Hour"),
                    SunCalcHourSpinBox, SunCalcHourValue, SunCalcHour, 0, 23, 1)
            ]

            +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 4, 0))
            [
                MakeInlineSpinBoxInt(TEXT("Minute"),
                    SunCalcMinuteSpinBox, SunCalcMinuteValue, SunCalcMinute, 0, 59, 5)
            ]

            +SHorizontalBox::Slot().FillWidth(1.f)
            [ SNew(SSpacer) ]
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            SNew(SHorizontalBox)

            +SHorizontalBox::Slot().AutoWidth().Padding(FMargin(0, 0, 4, 0))
            [
                SNew(SButton)
                .ButtonStyle(FAppStyle::Get(), "FlatButton.Primary")
                .ContentPadding(FMargin(8, 2))
                .Text(FText::FromString(TEXT("Calculate & Apply")))
                .ToolTipText(FText::FromString(TEXT("Compute sun position and apply to Directional Light")))
                .OnClicked_Lambda([this]() { return OnCalculateSunPositionClicked(); })
            ]

            +SHorizontalBox::Slot().MinWidth(50).Padding(FMargin(0, 0, 2, 0))
            [
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5, 5, 5, 255))
                .Padding(2, 0)
                [
                    SAssignNew(DayCycleSpeedSpinBox, SNumericEntryBox<float>)
                    .Value_Lambda([this]() { return DayCycleSpeedValue; })
                    .MinValue(1.f).MaxValue(600.f).Delta(1.f).AllowSpin(false)
                    .ToolTipText(FText::FromString(TEXT("Day cycle duration in real seconds (1=fast, 600=slow)")))
                    .OnValueChanged_Lambda([this](float Val)
                    {
                        DayCycleSpeed      = Val;
                        DayCycleSpeedValue = Val;
                    })
                ]
            ]

            +SHorizontalBox::Slot().AutoWidth().Padding(FMargin(0, 0, 6, 0))
            [
                SNew(SButton)
                .ButtonStyle(FAppStyle::Get(), "FlatButton.Success")
                .ContentPadding(FMargin(6, 2))
                .Text_Lambda([this]()
                {
                    return FText::FromString(bDayCycleActive ? TEXT("Stop") : TEXT("Day Cycle"));
                })
                .ToolTipText(FText::FromString(TEXT("Simulate 24h day/night cycle using current lat/lon/date")))
                .OnClicked_Lambda([this]() { return OnToggleDayCycleClicked(); })
            ]
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            SNew(SHorizontalBox)

            +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(0, 0, 4, 0))
            [
                SNew(STextBlock)
                .Text(FText::FromString(TEXT("Fill slot")))
                .ColorAndOpacity(FLinearColor(0.6f, 0.6f, 0.6f))
                .Font(FCoreStyle::GetDefaultFontStyle("Regular", 8))
            ]

            +SHorizontalBox::Slot().MaxWidth(48).Padding(FMargin(0, 0, 6, 0))
            [
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5, 5, 5, 255))
                .Padding(2, 0)
                [
                    SAssignNew(SunCalcFillSlotSpinBox, SNumericEntryBox<int32>)
                    .Value_Lambda([this]() { return SunCalcFillSlotValue; })
                    .MinValue(1).MaxValue(MaxLightingEntries).Delta(1).AllowSpin(false)
                    .ToolTipText(FText::FromString(TEXT("Target slot index (1-4)")))
                    .OnValueChanged_Lambda([this](int32 Val)
                    {
                        SunCalcFillSlot      = Val;
                        SunCalcFillSlotValue = Val;
                    })
                ]
            ]

            +SHorizontalBox::Slot().AutoWidth().Padding(FMargin(0, 0, 4, 0))
            [
                SNew(SButton)
                .ButtonStyle(FAppStyle::Get(), "FlatButton.Primary")
                .ContentPadding(FMargin(6, 2))
                .Text(FText::FromString(TEXT("-> Set-A")))
                .ToolTipText(FText::FromString(TEXT("Write computed Elevation/Azimuth into the chosen Set-A slot")))
                .OnClicked_Lambda([this]() { return OnFillSetAFromSunPositionClicked(); })
            ]

            +SHorizontalBox::Slot().AutoWidth()
            [
                SNew(SButton)
                .ButtonStyle(FAppStyle::Get(), "FlatButton.Danger")
                .ContentPadding(FMargin(6, 2))
                .Text(FText::FromString(TEXT("-> Set-B")))
                .ToolTipText(FText::FromString(TEXT("Write computed Elevation/Azimuth into the chosen Set-B slot")))
                .OnClicked_Lambda([this]() { return OnFillSetBFromSunPositionClicked(); })
            ]
        ]
    ];
}

TSharedRef<SWidget> FVCCSimPanelTexEnhancer::CreateSetALightingEntry(int32 Index)
{
    return SNew(SHorizontalBox)

    +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(0, 0, 4, 0))
    [
        SNew(STextBlock)
        .Text(FText::FromString(FString::Printf(TEXT("A%d"), Index + 1)))
        .ColorAndOpacity(FLinearColor(0.6f, 0.6f, 0.6f))
        .MinDesiredWidth(18.f)
    ]

    +SHorizontalBox::Slot().MaxWidth(80).Padding(FMargin(0, 0, 2, 0))
    [
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(FColor(5, 5, 5, 255))
        .Padding(2, 0)
        [
            SAssignNew(SetAElevationSpinBox[Index], SNumericEntryBox<float>)
            .Value_Lambda([this, Index]() { return SetAElevationValue[Index]; })
            .MinValue(0.f).MaxValue(90.f).Delta(1.f).AllowSpin(false)
            .ToolTipText(FText::FromString(TEXT("Sun elevation (°)")))
            .OnValueChanged_Lambda([this, Index](float Val)
            {
                SetAElevation[Index] = Val;
                SetAElevationValue[Index] = Val;
            })
        ]
    ]

    +SHorizontalBox::Slot().MaxWidth(80).Padding(FMargin(0, 0, 4, 0))
    [
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(FColor(5, 5, 5, 255))
        .Padding(2, 0)
        [
            SAssignNew(SetAAzimuthSpinBox[Index], SNumericEntryBox<float>)
            .Value_Lambda([this, Index]() { return SetAAzimuthValue[Index]; })
            .MinValue(0.f).MaxValue(360.f).Delta(5.f).AllowSpin(false)
            .ToolTipText(FText::FromString(TEXT("Sun azimuth (°)")))
            .OnValueChanged_Lambda([this, Index](float Val)
            {
                SetAAzimuth[Index] = Val;
                SetAAzimuthValue[Index] = Val;
            })
        ]
    ]

    +SHorizontalBox::Slot().MaxWidth(80).HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Primary")
        .ContentPadding(FMargin(4, 2))
        .Text(FText::FromString(TEXT("Apply")))
        .ToolTipText(FText::FromString(TEXT("Apply this lighting condition to the scene")))
        .OnClicked_Lambda([this, Index]() { return OnApplySetALightingClicked(Index); })
    ];
}

TSharedRef<SWidget> FVCCSimPanelTexEnhancer::CreateSetBLightingEntry(int32 Index)
{
    return SNew(SHorizontalBox)

    +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(0, 0, 4, 0))
    [
        SNew(STextBlock)
        .Text(FText::FromString(FString::Printf(TEXT("B%d"), Index + 1)))
        .ColorAndOpacity(FLinearColor(0.7f, 0.5f, 0.3f))
        .MinDesiredWidth(18.f)
    ]

    +SHorizontalBox::Slot().MaxWidth(80).Padding(FMargin(0, 0, 2, 0))
    [
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(FColor(5, 5, 5, 255))
        .Padding(2, 0)
        [
            SAssignNew(SetBElevationSpinBox[Index], SNumericEntryBox<float>)
            .Value_Lambda([this, Index]() { return SetBElevationValue[Index]; })
            .MinValue(0.f).MaxValue(90.f).Delta(1.f).AllowSpin(false)
            .ToolTipText(FText::FromString(TEXT("Sun elevation (°) — evaluation only")))
            .OnValueChanged_Lambda([this, Index](float Val)
            {
                SetBElevation[Index] = Val;
                SetBElevationValue[Index] = Val;
            })
        ]
    ]

    +SHorizontalBox::Slot().MaxWidth(80).Padding(FMargin(0, 0, 4, 0))
    [
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(FColor(5, 5, 5, 255))
        .Padding(2, 0)
        [
            SAssignNew(SetBAzimuthSpinBox[Index], SNumericEntryBox<float>)
            .Value_Lambda([this, Index]() { return SetBAzimuthValue[Index]; })
            .MinValue(0.f).MaxValue(360.f).Delta(5.f).AllowSpin(false)
            .ToolTipText(FText::FromString(TEXT("Sun azimuth (°) — evaluation only")))
            .OnValueChanged_Lambda([this, Index](float Val)
            {
                SetBAzimuth[Index] = Val;
                SetBAzimuthValue[Index] = Val;
            })
        ]
    ]

    +SHorizontalBox::Slot().MaxWidth(80).HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Warning")
        .ContentPadding(FMargin(4, 2))
        .Text(FText::FromString(TEXT("Apply")))
        .ToolTipText(FText::FromString(TEXT("Apply this evaluation lighting — use only after Set-B is captured")))
        .OnClicked_Lambda([this, Index]() { return OnApplySetBLightingClicked(Index); })
    ];
}

// ============================================================================
// SECTION 4: GT MATERIAL EXPORT
// ============================================================================

TSharedRef<SWidget> FVCCSimPanelTexEnhancer::CreateGTExportSection()
{
    return FVCCSimUIHelpers::CreateSectionContent(
        SNew(SVerticalBox)

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            FVCCSimUIHelpers::CreateSectionHeader(TEXT("GT Material Export"))
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            SNew(SHorizontalBox)
            +SHorizontalBox::Slot().FillWidth(1.f)
            [
                SNew(SButton)
                .ButtonStyle(FAppStyle::Get(), "FlatButton.Success")
                .ContentPadding(FMargin(5, 2))
                .Text(FText::FromString(TEXT("+ Add Selected Actors")))
                .ToolTipText(FText::FromString(TEXT("Add currently selected StaticMeshActors from the viewport")))
                .OnClicked_Lambda([this]() { return OnAddSelectedActorsClicked(); })
            ]
            +SHorizontalBox::Slot().AutoWidth().Padding(FMargin(4, 0, 0, 0))
            [
                SNew(SButton)
                .ButtonStyle(FAppStyle::Get(), "FlatButton.Danger")
                .ContentPadding(FMargin(5, 2))
                .Text(FText::FromString(TEXT("Clear All")))
                .OnClicked_Lambda([this]() -> FReply
                {
                    GTActorListItems.Empty();
                    if (GTActorListView.IsValid())
                        GTActorListView->RequestListRefresh();
                    SavePaths();
                    return FReply::Handled();
                })
            ]
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 0, 0, 2))
        [
            SNew(SBox)
            .HeightOverride(90.f)
            [
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(10, 10, 10, 255))
                .Padding(2)
                [
                    SAssignNew(GTActorListView, SListView<TSharedPtr<FString>>)
                    .ListItemsSource(&GTActorListItems)
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
                                .Text(FText::FromString(*Item))
                                .ColorAndOpacity(FLinearColor(0.8f, 0.9f, 0.8f))
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
                                        const FString S = *Item;
                                        GTActorListItems.RemoveAll([&S](const TSharedPtr<FString>& P)
                                        {
                                            return P.IsValid() && *P == S;
                                        });
                                        if (GTActorListView.IsValid())
                                            GTActorListView->RequestListRefresh();
                                        SavePaths();
                                    }
                                    return FReply::Handled();
                                })
                            ]
                        ];
                    })
                ]
            ]
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 0, 0, 2))
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowInt32(
                TEXT("Tex Res (px)"), GTTexResSpinBox, GTTexResValue,
                GTTextureResolution, 64, 64)
        ]

        +SVerticalBox::Slot().AutoHeight()
        [
            SNew(SButton)
            .ButtonStyle(FAppStyle::Get(), "FlatButton.Primary")
            .ContentPadding(FMargin(5, 2))
            .Text_Lambda([this]() {
                return bGTExportInProgress ? FText::FromString(TEXT("GT export in progress...")) : FText::FromString(TEXT("Export GT Materials"));
            })
            .ToolTipText_Lambda([this]()
            {
                return FText::FromString(FString::Printf(
                    TEXT("Export mesh OBJ + PBR textures for %d actor(s) -> %s/gt_materials/"),
                    GTActorListItems.Num(), *OutputDirectory));
            })
            .IsEnabled_Lambda([this]() { return !bGTExportInProgress && !OutputDirectory.IsEmpty() && !GTActorListItems.IsEmpty(); })
            .OnClicked_Lambda([this]() { return OnExportGTMaterialsClicked(); })
        ]
    );
}

// ============================================================================
// SECTION 5: NANOBANANA PROJECTION
// ============================================================================

TSharedRef<SWidget> FVCCSimPanelTexEnhancer::CreateNanobananaSection()
{
    auto MakeBrowseRow = [this](
        const FString& Label,
        TSharedPtr<SEditableTextBox>& TextBoxPtr,
        FString& Var,
        TFunction<FReply()> BrowseFn) -> TSharedRef<SWidget>
    {
        return FVCCSimUIHelpers::CreatePropertyRow(*Label,
            SNew(SHorizontalBox)
            +SHorizontalBox::Slot().FillWidth(1.f)
            [
                SAssignNew(TextBoxPtr, SEditableTextBox)
                .Text(FText::FromString(Var))
                .OnTextCommitted_Lambda([this, &Var](const FText& T, ETextCommit::Type)
                {
                    Var = T.ToString();
                    SavePaths();
                })
            ]
            +SHorizontalBox::Slot().AutoWidth().Padding(FMargin(4, 0, 0, 0))
            [
                SNew(SButton)
                .ButtonStyle(FAppStyle::Get(), "FlatButton")
                .ContentPadding(FMargin(5, 2))
                .Text(FText::FromString(TEXT("...")))
                .OnClicked_Lambda([BrowseFn]() { return BrowseFn(); })
            ]
        );
    };

    return FVCCSimUIHelpers::CreateCollapsibleSection(TEXT("Nanobanana Projection"),
        SNew(SVerticalBox)

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            MakeBrowseRow(TEXT("Result Dir"), NanobananaResultDirTextBox, NanobananaResultDir,
                [this]() { return OnBrowseNanobananaResultDirClicked(); })
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            MakeBrowseRow(TEXT("Poses File"), NanobananaPosesFileTextBox, NanobananaPosesFile,
                [this]() { return OnBrowseNanobananaPosesFileClicked(); })
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            MakeBrowseRow(TEXT("Manifest JSON"), NanobananaManifestFileTextBox, NanobananaManifestFile,
                [this]() { return OnBrowseNanobananaManifestFileClicked(); })
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            FVCCSimUIHelpers::CreatePropertyRow(TEXT("H-FOV (deg)"),
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5, 5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(NanobananaHFOVSpinBox, SNumericEntryBox<float>)
                    .Value_Lambda([this]() { return NanobananaHFOVValue; })
                    .MinValue(10.f).MaxValue(170.f).Delta(1.f).AllowSpin(false)
                    .OnValueChanged_Lambda([this](float V)
                    {
                        NanobananaHFOV = V; NanobananaHFOVValue = V; SavePaths();
                    })
                ]
            )
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            FVCCSimUIHelpers::CreatePropertyRow(TEXT("Image W / H"),
                SNew(SHorizontalBox)
                +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 2, 0))
                [
                    SNew(SBorder)
                    .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                    .BorderBackgroundColor(FColor(5, 5, 5, 255))
                    .Padding(4, 0)
                    [
                        SAssignNew(NanobananaImageWidthSpinBox, SNumericEntryBox<int32>)
                        .Value_Lambda([this]() { return NanobananaImageWidthValue; })
                        .MinValue(1).MaxValue(8192).Delta(1).AllowSpin(false)
                        .OnValueChanged_Lambda([this](int32 V)
                        {
                            NanobananaImageWidth = V; NanobananaImageWidthValue = V; SavePaths();
                        })
                    ]
                ]
                +SHorizontalBox::Slot().FillWidth(1.f)
                [
                    SNew(SBorder)
                    .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                    .BorderBackgroundColor(FColor(5, 5, 5, 255))
                    .Padding(4, 0)
                    [
                        SAssignNew(NanobananaImageHeightSpinBox, SNumericEntryBox<int32>)
                        .Value_Lambda([this]() { return NanobananaImageHeightValue; })
                        .MinValue(1).MaxValue(8192).Delta(1).AllowSpin(false)
                        .OnValueChanged_Lambda([this](int32 V)
                        {
                            NanobananaImageHeight = V; NanobananaImageHeightValue = V; SavePaths();
                        })
                    ]
                ]
            )
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            FVCCSimUIHelpers::CreatePropertyRow(TEXT("Rays / Class"),
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5, 5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(NanobananaRaysPerClassSpinBox, SNumericEntryBox<int32>)
                    .Value_Lambda([this]() { return NanobananaRaysPerClassValue; })
                    .MinValue(1).MaxValue(1000).Delta(10).AllowSpin(false)
                    .OnValueChanged_Lambda([this](int32 V)
                    {
                        NanobananaRaysPerClass = V; NanobananaRaysPerClassValue = V; SavePaths();
                    })
                ]
            )
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            SNew(SHorizontalBox)
            +SHorizontalBox::Slot().FillWidth(1.f)
            [
                SNew(SButton)
                .ButtonStyle(FAppStyle::Get(), "FlatButton.Success")
                .ContentPadding(FMargin(5, 2))
                .Text_Lambda([this]()
                {
                    return bNanobananaInProgress
                        ? FText::FromString(TEXT("Running projection..."))
                        : FText::FromString(TEXT("Run Projection"));
                })
                .IsEnabled_Lambda([this]()
                {
                    return !bNanobananaInProgress
                        && !NanobananaResultDir.IsEmpty()
                        && !NanobananaPosesFile.IsEmpty()
                        && !NanobananaManifestFile.IsEmpty();
                })
                .OnClicked_Lambda([this]() { return OnRunNanobananaProjectionClicked(); })
            ]
        ]
    , bNanobananaExpanded);
}

// ============================================================================
// SECTION 6: TEXENHANCER PIPELINE
// ============================================================================

TSharedRef<SWidget> FVCCSimPanelTexEnhancer::CreatePipelineSection()
{
    return FVCCSimUIHelpers::CreateSectionContent(
        SNew(SVerticalBox)

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            FVCCSimUIHelpers::CreateSectionHeader(TEXT("TexEnhancer Pipeline"))
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            FVCCSimUIHelpers::CreatePropertyRow(TEXT("Script"),
                SNew(SHorizontalBox)
                +SHorizontalBox::Slot().FillWidth(1.f)
                [
                    SAssignNew(TexEnhancerScriptTextBox, SEditableTextBox)
                    .Text(FText::FromString(TexEnhancerScriptPath))
                    .OnTextCommitted_Lambda([this](const FText& Text, ETextCommit::Type)
                    {
                        TexEnhancerScriptPath = Text.ToString();
                        SavePaths();
                    })
                ]
                +SHorizontalBox::Slot().AutoWidth().Padding(FMargin(4, 0, 0, 0))
                [
                    SNew(SButton)
                    .ButtonStyle(FAppStyle::Get(), "FlatButton")
                    .ContentPadding(FMargin(5, 2))
                    .Text(FText::FromString(TEXT("...")))
                    .OnClicked_Lambda([this]() { return OnBrowseScriptClicked(); })
                ]
            )
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            SNew(SHorizontalBox)

            +SHorizontalBox::Slot().MaxWidth(160).Padding(FMargin(0, 0, 4, 0))
            [
                SNew(SButton)
                .ButtonStyle(FAppStyle::Get(), "FlatButton.Success")
                .ContentPadding(FMargin(5, 2))
                .Text(FText::FromString(TEXT("Run TexEnhancer")))
                .IsEnabled_Lambda([this]() { return !bPipelineInProgress && !TexEnhancerScriptPath.IsEmpty(); })
                .OnClicked_Lambda([this]() { return OnRunTexEnhancerClicked(); })
            ]

            +SHorizontalBox::Slot().MaxWidth(80)
            [
                SNew(SButton)
                .ButtonStyle(FAppStyle::Get(), "FlatButton.Danger")
                .ContentPadding(FMargin(5, 2))
                .Text(FText::FromString(TEXT("Stop")))
                .IsEnabled_Lambda([this]() { return bPipelineInProgress; })
                .OnClicked_Lambda([this]() { return OnStopTexEnhancerClicked(); })
            ]
        ]
    );
}

// ============================================================================
// SECTION 7: EVALUATION
// ============================================================================

TSharedRef<SWidget> FVCCSimPanelTexEnhancer::CreateEvaluationSection()
{
    return FVCCSimUIHelpers::CreateSectionContent(
        SNew(SVerticalBox)

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            FVCCSimUIHelpers::CreateSectionHeader(TEXT("BRDF Evaluation"))
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            FVCCSimUIHelpers::CreatePropertyRow(TEXT("Estimated Dir"),
                SNew(SHorizontalBox)
                +SHorizontalBox::Slot().FillWidth(1.f)
                [
                    SAssignNew(EstimatedMaterialsDirTextBox, SEditableTextBox)
                    .Text(FText::FromString(EstimatedMaterialsDir))
                    .OnTextCommitted_Lambda([this](const FText& Text, ETextCommit::Type)
                    {
                        EstimatedMaterialsDir = Text.ToString();
                        SavePaths();
                    })
                ]
                +SHorizontalBox::Slot().AutoWidth().Padding(FMargin(4, 0, 0, 0))
                [
                    SNew(SButton)
                    .ButtonStyle(FAppStyle::Get(), "FlatButton")
                    .ContentPadding(FMargin(5, 2))
                    .Text(FText::FromString(TEXT("...")))
                    .OnClicked_Lambda([this]() { return OnBrowseEstimatedDirClicked(); })
                ]
            )
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            SNew(SHorizontalBox)
            +SHorizontalBox::Slot().MaxWidth(160)
            [
                SNew(SButton)
                .ButtonStyle(FAppStyle::Get(), "FlatButton.Success")
                .ContentPadding(FMargin(5, 2))
                .Text(FText::FromString(TEXT("Run BRDF Evaluation")))
                .IsEnabled_Lambda([this]()
                {
                    return !bEvalInProgress && !EstimatedMaterialsDir.IsEmpty() && !OutputDirectory.IsEmpty();
                })
                .OnClicked_Lambda([this]() { return OnRunEvaluationClicked(); })
            ]
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            SNew(SBorder)
            .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
            .BorderBackgroundColor(FColor(5, 5, 5, 200))
            .Padding(FMargin(6, 4))
            [
                SAssignNew(EvalResultsTextBlock, STextBlock)
                .Text(FText::FromString(TEXT("No evaluation results yet.")))
                .ColorAndOpacity(FLinearColor(0.7f, 0.7f, 0.7f))
                .AutoWrapText(true)
                .Font(FCoreStyle::GetDefaultFontStyle("Mono", 8))
            ]
        ]
    );
}
