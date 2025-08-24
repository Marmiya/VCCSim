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

#pragma once

#include "CoreMinimal.h"
#include "IVCCSimModule.h"
#include "Widgets/Input/SNumericEntryBox.h"
#include "Widgets/Input/SCheckBox.h"

// Forward declarations to avoid PropertyEditor issues in UE 5.6
class IDetailsView;

class SVCCSimPanel;

/**
 * Factory and manager for VCCSim panel modules
 */
class VCCSIMEDITOR_API FVCCSimModuleFactory
{
public:
    /**
     * Create camera module
     */
    static TSharedPtr<IVCCSimCameraModule> CreateCameraModule();

    /**
     * Create pose module
     */
    static TSharedPtr<IVCCSimPoseModule> CreatePoseModule();

    /**
     * Create point cloud module
     */
    static TSharedPtr<IVCCSimPointCloudModule> CreatePointCloudModule();

    /**
     * Create triangle splatting module
     */
    static TSharedPtr<IVCCSimTriangleSplattingModule> CreateTriangleSplattingModule();

    /**
     * Create scene analysis module
     */
    static TSharedPtr<IVCCSimSceneAnalysisModule> CreateSceneAnalysisModule();

    /**
     * Create details view for configuration objects
     */
    static TSharedPtr<IDetailsView> CreateDetailsView();

    /**
     * Create object property entry box for asset selection
     */
    static TSharedRef<SWidget> CreateObjectPropertyEntryBox(
        UClass* AllowedClass,
        TFunction<FString()> GetObjectPathFunc,
        TFunction<void(const FAssetData&)> OnObjectChangedFunc,
        bool bAllowClear = true);

    /**
     * Create file path picker
     */
    static TSharedRef<SWidget> CreateFilePathPicker(
        const FString& DialogTitle,
        const FString& FileTypeFilter,
        TFunction<FString()> GetPathFunc,
        TFunction<void(const FString&)> OnPathChangedFunc);

    /**
     * Create directory path picker
     */
    static TSharedRef<SWidget> CreateDirectoryPathPicker(
        const FString& DialogTitle,
        TFunction<FString()> GetPathFunc,
        TFunction<void(const FString&)> OnPathChangedFunc);

    /**
     * Style helpers
     */
    static const FSlateFontInfo& GetDefaultFont();
    static const FSlateFontInfo& GetBoldFont();
    static const FSlateFontInfo& GetHeaderFont();
    
    /**
     * Color scheme
     */
    static FSlateColor GetPrimaryTextColor();
    static FSlateColor GetSecondaryTextColor();
    static FSlateColor GetSuccessColor();
    static FSlateColor GetWarningColor();
    static FSlateColor GetErrorColor();
};

/**
 * Simple data binding helper for UI components
 * Note: This is a simplified version to avoid template complications
 */
template<typename T>
class TVCCSimDataBinder
{
public:
    TVCCSimDataBinder(T* InDataPtr, TFunction<void()> InOnChangedCallback = nullptr)
        : DataPtr(InDataPtr)
        , OnChangedCallback(InOnChangedCallback)
    {
    }

    /**
     * Create TAttribute for getting value
     */
    TAttribute<T> CreateGetter() const
    {
        return TAttribute<T>::CreateLambda([this]() { return GetValue(); });
    }

    /**
     * Get current value
     */
    T GetValue() const
    {
        return DataPtr ? *DataPtr : T{};
    }

    /**
     * Set value and trigger callback
     */
    void SetValue(const T& NewValue)
    {
        if (DataPtr)
        {
            *DataPtr = NewValue;
            if (OnChangedCallback)
            {
                OnChangedCallback();
            }
        }
    }

private:
    T* DataPtr;
    TFunction<void()> OnChangedCallback;
};

/**
 * Helper macros for creating data binders
 */
#define BIND_DATA(DataMember) \
    TVCCSimDataBinder<decltype(DataMember)>(&DataMember, [this]() { OnConfigurationChanged(); })

#define BIND_DATA_WITH_CALLBACK(DataMember, Callback) \
    TVCCSimDataBinder<decltype(DataMember)>(&DataMember, Callback)

/**
 * UI Layout helpers
 */
class VCCSIMEDITOR_API FVCCSimUIHelpers
{
public:
    /**
     * Create standard property row with label and content
     */
    static TSharedRef<SWidget> CreatePropertyRow(
        const FString& Label, 
        TSharedRef<SWidget> Content,
        bool bFillWidth = true);

    /**
     * Create section header with collapsible functionality
     */
    static TSharedRef<SWidget> CreateCollapsibleSection(
        const FString& Title,
        TSharedRef<SWidget> Content,
        bool& bExpanded);

    /**
     * Create button with standard styling
     */
    static TSharedRef<SWidget> CreateStandardButton(
        const FString& ButtonText,
        FOnClicked OnClicked,
        bool bEnabled = true,
        const FString& ToolTip = FString());

    /**
     * Create status text block with color coding
     */
    static TSharedRef<SWidget> CreateStatusText(
        TAttribute<FText> Text,
        TAttribute<FSlateColor> Color = FSlateColor::UseForeground());

    /**
     * Create progress bar
     */
    static TSharedRef<SWidget> CreateProgressBar(
        TAttribute<TOptional<float>> Percent,
        TAttribute<FText> Text = FText::GetEmpty());

    /**
     * Create separator line
     */
    static TSharedRef<SWidget> CreateSeparator();

    /**
     * Create horizontal spacer
     */
    static TSharedRef<SWidget> CreateHorizontalSpacer(float Width = 10.0f);

    /**
     * Create vertical spacer
     */
    static TSharedRef<SWidget> CreateVerticalSpacer(float Height = 10.0f);
};

/**
 * Notification helper for user feedback
 */
class VCCSIMEDITOR_API FVCCSimNotificationHelper
{
public:
    /**
     * Show success notification
     */
    static void ShowSuccess(const FString& Message, float Duration = 3.0f);

    /**
     * Show warning notification
     */
    static void ShowWarning(const FString& Message, float Duration = 5.0f);

    /**
     * Show error notification
     */
    static void ShowError(const FString& Message, float Duration = 7.0f);

    /**
     * Show progress notification
     */
    static TSharedPtr<SNotificationItem> ShowProgress(const FString& Message);

    /**
     * Update progress notification
     */
    static void UpdateProgress(TSharedPtr<SNotificationItem> Notification, float Progress, const FString& Message);

    /**
     * Complete progress notification
     */
    static void CompleteProgress(TSharedPtr<SNotificationItem> Notification, const FString& Message, bool bSuccess = true);
};