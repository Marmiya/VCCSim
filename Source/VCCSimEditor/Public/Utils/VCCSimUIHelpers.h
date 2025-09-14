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

#pragma once

#include "CoreMinimal.h"
#include "Widgets/SWidget.h"
#include "Widgets/Input/SNumericEntryBox.h"
#include "Widgets/Layout/SExpandableArea.h"
#include "Styling/AppStyle.h"

/**
 * Common UI helper functions for VCCSim editor panels
 * Provides consistent styling and behavior across all panel implementations
 */
class VCCSIMEDITOR_API FVCCSimUIHelpers
{
public:
    // ============================================================================
    // LAYOUT HELPER FUNCTIONS
    // ============================================================================
    
    /**
     * Creates a horizontal property row with label and content
     * @param Label - Text label for the property
     * @param Content - Widget content for the property value
     * @return Configured horizontal box widget
     */
    static TSharedRef<SWidget> CreatePropertyRow(const FString& Label, TSharedRef<SWidget> Content);
    
    /**
     * Creates a visual separator line between sections
     * @return Configured separator widget
     */
    static TSharedRef<SWidget> CreateSeparator();
    
    /**
     * Creates an expandable/collapsible section with header and content
     * @param Title - Section title text
     * @param Content - Section content widget
     * @param bExpanded - Reference to expansion state variable
     * @return Configured expandable area widget
     */
    static TSharedRef<SWidget> CreateCollapsibleSection(const FString& Title, TSharedRef<SWidget> Content, bool& bExpanded);
    
    /**
     * Creates a styled section header
     * @param Title - Header title text
     * @return Configured section header widget
     */
    static TSharedRef<SWidget> CreateSectionHeader(const FString& Title);
    
    /**
     * Creates a styled section content container
     * @param Content - Content to wrap
     * @return Configured section content widget
     */
    static TSharedRef<SWidget> CreateSectionContent(TSharedRef<SWidget> Content);
    
    // ============================================================================
    // NUMERIC ENTRY HELPER FUNCTIONS
    // ============================================================================
    
    /**
     * Creates a numeric property row for int32 values
     * @param Label - Property label text
     * @param SpinBox - Reference to spinbox widget pointer (will be assigned)
     * @param Value - Reference to TOptional value storage
     * @param ActualVariable - Reference to actual variable to update
     * @param MinValue - Minimum allowed value
     * @param DeltaValue - Step increment value
     * @return Configured numeric property row widget
     */
    static TSharedRef<SWidget> CreateNumericPropertyRowInt32(
        const FString& Label,
        TSharedPtr<SNumericEntryBox<int32>>& SpinBox,
        TOptional<int32>& Value,
        int32& ActualVariable,
        int32 MinValue,
        int32 DeltaValue);
    
    /**
     * Creates a numeric property row for float values
     * @param Label - Property label text
     * @param SpinBox - Reference to spinbox widget pointer (will be assigned)
     * @param Value - Reference to TOptional value storage
     * @param ActualVariable - Reference to actual variable to update
     * @param MinValue - Minimum allowed value
     * @param DeltaValue - Step increment value
     * @return Configured numeric property row widget
     */
    static TSharedRef<SWidget> CreateNumericPropertyRowFloat(
        const FString& Label,
        TSharedPtr<SNumericEntryBox<float>>& SpinBox,
        TOptional<float>& Value,
        float& ActualVariable,
        float MinValue,
        float DeltaValue);
    
    /**
     * Template function for creating generic numeric property rows
     * @param Label - Property label text
     * @param SpinBox - Reference to spinbox widget pointer (will be assigned)
     * @param Value - Reference to TOptional value storage
     * @param MinValue - Minimum allowed value
     * @param MaxValue - Maximum allowed value
     * @param DeltaValue - Step increment value
     * @param OnValueChanged - Optional callback for value changes
     * @return Configured numeric property row widget
     */
    template<typename T>
    static TSharedRef<SWidget> CreateNumericPropertyRow(
        const FString& Label,
        TSharedPtr<SNumericEntryBox<T>>& SpinBox,
        TOptional<T>& Value,
        T MinValue,
        T MaxValue,
        T DeltaValue,
        TFunction<void(T)> OnValueChanged = nullptr);
    
    // ============================================================================
    // STYLING CONSTANTS
    // ============================================================================
    
    /** Standard property label color */
    static const FColor PropertyLabelColor;
    
    /** Standard section background color */
    static const FColor SectionBackgroundColor;
    
    /** Standard separator color */
    static const FColor SeparatorColor;
    
    /** Standard section header background color */
    static const FColor SectionHeaderBackgroundColor;

private:
    FVCCSimUIHelpers() = delete; // Static class only
};

// ============================================================================
// TEMPLATE IMPLEMENTATION
// ============================================================================

template<typename T>
TSharedRef<SWidget> FVCCSimUIHelpers::CreateNumericPropertyRow(
    const FString& Label,
    TSharedPtr<SNumericEntryBox<T>>& SpinBox,
    TOptional<T>& Value,
    T MinValue,
    T MaxValue,
    T DeltaValue,
    TFunction<void(T)> OnValueChanged)
{
    return CreatePropertyRow(
        Label,
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(SectionBackgroundColor)
        .Padding(4, 0)
        [
            SAssignNew(SpinBox, SNumericEntryBox<T>)
            .Value_Lambda([&Value]() { return Value; })
            .MinValue(MinValue)
            .MaxValue(MaxValue)
            .Delta(DeltaValue)
            .AllowSpin(true)
            .OnValueChanged_Lambda([&Value, OnValueChanged](T NewValue)
            {
                Value = NewValue;
                if (OnValueChanged)
                {
                    OnValueChanged(NewValue);
                }
            })
        ]
    );
}