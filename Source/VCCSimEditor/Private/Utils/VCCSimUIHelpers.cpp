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

#include "Utils/VCCSimUIHelpers.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Layout/SExpandableArea.h"
#include "Styling/AppStyle.h"

// ============================================================================
// STYLING CONSTANTS
// ============================================================================

const FColor FVCCSimUIHelpers::PropertyLabelColor = FColor(233, 233, 233);
const FColor FVCCSimUIHelpers::SectionBackgroundColor = FColor(5, 5, 5, 255);
const FColor FVCCSimUIHelpers::SeparatorColor = FColor(2, 2, 2);
const FColor FVCCSimUIHelpers::SectionHeaderBackgroundColor = FColor(48, 48, 48);

// ============================================================================
// LAYOUT HELPER FUNCTIONS
// ============================================================================

TSharedRef<SWidget> FVCCSimUIHelpers::CreatePropertyRow(const FString& Label, TSharedRef<SWidget> Content)
{
    return SNew(SHorizontalBox)
        +SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        .Padding(FMargin(0, 0, 8, 0))
        [
            SNew(STextBlock)
            .Text(FText::FromString(Label))
            .MinDesiredWidth(80)
            .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
            .ColorAndOpacity(PropertyLabelColor)
        ]
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        [
            Content
        ];
}

TSharedRef<SWidget> FVCCSimUIHelpers::CreateSeparator()
{
    return SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(SeparatorColor)
        .Padding(0)
        .Content()
        [
            SNew(SBox)
            .HeightOverride(1.0f)
        ];
}

TSharedRef<SWidget> FVCCSimUIHelpers::CreateCollapsibleSection(const FString& Title, TSharedRef<SWidget> Content, bool& bExpanded)
{
    return SNew(SExpandableArea)
        .InitiallyCollapsed(!bExpanded)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryTop"))
        .BorderBackgroundColor(SectionHeaderBackgroundColor)
        .OnAreaExpansionChanged_Lambda([&bExpanded](bool bIsExpanded) {
            bExpanded = bIsExpanded;
        })
        .HeaderContent()
        [
            SNew(STextBlock)
            .Text(FText::FromString(Title))
            .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
            .ColorAndOpacity(PropertyLabelColor)
            .TransformPolicy(ETextTransformPolicy::ToUpper)
        ]
        .BodyContent()
        [
            SNew(SBorder)
            .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
            .BorderBackgroundColor(SectionBackgroundColor)
            .Padding(FMargin(15, 6))
            [
                Content
            ]
        ];
}

TSharedRef<SWidget> FVCCSimUIHelpers::CreateSectionHeader(const FString& Title)
{
    return SNew(STextBlock)
        .Text(FText::FromString(Title))
        .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
        .ColorAndOpacity(PropertyLabelColor)
        .TransformPolicy(ETextTransformPolicy::ToUpper);
}

TSharedRef<SWidget> FVCCSimUIHelpers::CreateSectionContent(TSharedRef<SWidget> Content)
{
    return SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(SectionBackgroundColor)
        .Padding(FMargin(15, 6))
        [
            Content
        ];
}

// ============================================================================
// NUMERIC ENTRY HELPER FUNCTIONS
// ============================================================================

TSharedRef<SWidget> FVCCSimUIHelpers::CreateNumericPropertyRowInt32(
    const FString& Label,
    TSharedPtr<SNumericEntryBox<int32>>& SpinBox,
    TOptional<int32>& Value,
    int32& ActualVariable,
    int32 MinValue,
    int32 DeltaValue)
{
    return CreatePropertyRow(
        Label,
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(SectionBackgroundColor)
        .Padding(4, 0)
        [
            SAssignNew(SpinBox, SNumericEntryBox<int32>)
            .Value_Lambda([&Value]() { return Value; })
            .MinValue(MinValue)
            .Delta(DeltaValue)
            .AllowSpin(true)
            .OnValueChanged_Lambda([&Value, &ActualVariable](int32 NewValue) {
                Value = NewValue;
                ActualVariable = NewValue;
            })
        ]
    );
}

TSharedRef<SWidget> FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
    const FString& Label,
    TSharedPtr<SNumericEntryBox<float>>& SpinBox,
    TOptional<float>& Value,
    float& ActualVariable,
    float MinValue,
    float DeltaValue)
{
    return CreatePropertyRow(
        Label,
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(SectionBackgroundColor)
        .Padding(4, 0)
        [
            SAssignNew(SpinBox, SNumericEntryBox<float>)
            .Value_Lambda([&Value]() { return Value; })
            .MinValue(MinValue)
            .Delta(DeltaValue)
            .AllowSpin(true)
            .OnValueChanged_Lambda([&Value, &ActualVariable](float NewValue) {
                Value = NewValue;
                ActualVariable = NewValue;
            })
        ]
    );
}