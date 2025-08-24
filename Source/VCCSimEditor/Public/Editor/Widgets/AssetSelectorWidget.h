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
#include "Widgets/SCompoundWidget.h"
#include "PropertyCustomizationHelpers.h"
#include "Widgets/Input/SFilePathPicker.h"
#include "Widgets/Input/SDirectoryPicker.h"
#include "AssetRegistry/AssetData.h"

/**
 * Unified asset selection widget factory for VCCSim Editor
 * Provides consistent UE-standard asset selection across all panels
 */
class VCCSIMEDITOR_API FAssetSelectorFactory
{
public:
    /**
     * Create standard object property entry box for UObject-based assets
     */
    static TSharedRef<SWidget> CreateObjectSelector(
        UClass* AllowedClass,
        TFunction<FString()> GetObjectPathFunc,
        TFunction<void(const FAssetData&)> OnObjectChangedFunc,
        const FString& Label = FString(),
        bool bAllowClear = true,
        bool bShowThumbnail = true);

    /**
     * Create static mesh selector specifically
     */
    static TSharedRef<SWidget> CreateStaticMeshSelector(
        TFunction<FString()> GetMeshPathFunc,
        TFunction<void(const FAssetData&)> OnMeshChangedFunc,
        const FString& Label = TEXT("Static Mesh"),
        bool bAllowClear = true);

    /**
     * Create texture selector
     */
    static TSharedRef<SWidget> CreateTextureSelector(
        TFunction<FString()> GetTexturePathFunc,
        TFunction<void(const FAssetData&)> OnTextureChangedFunc,
        const FString& Label = TEXT("Texture"),
        bool bAllowClear = true);

    /**
     * Create material selector
     */
    static TSharedRef<SWidget> CreateMaterialSelector(
        TFunction<FString()> GetMaterialPathFunc,
        TFunction<void(const FAssetData&)> OnMaterialChangedFunc,
        const FString& Label = TEXT("Material"),
        bool bAllowClear = true);

    /**
     * Create file path picker for external files
     */
    static TSharedRef<SWidget> CreateFilePathSelector(
        const FString& DialogTitle,
        const FString& FileTypeFilter,
        TFunction<FString()> GetPathFunc,
        TFunction<void(const FString&)> OnPathChangedFunc,
        const FString& Label = FString(),
        bool bDirectoryMode = false);

    /**
     * Create directory path picker
     */
    static TSharedRef<SWidget> CreateDirectorySelector(
        const FString& DialogTitle,
        TFunction<FString()> GetPathFunc,
        TFunction<void(const FString&)> OnPathChangedFunc,
        const FString& Label = TEXT("Directory"));

    /**
     * Create property row wrapper with consistent styling
     */
    static TSharedRef<SWidget> CreatePropertyRow(
        const FString& Label,
        TSharedRef<SWidget> Selector,
        bool bRequired = false,
        const FString& ToolTip = FString());
};

/**
 * Enhanced object property entry box with additional features
 */
class VCCSIMEDITOR_API SEnhancedObjectSelector : public SCompoundWidget
{
public:
    SLATE_BEGIN_ARGS(SEnhancedObjectSelector) {}
        SLATE_ARGUMENT(UClass*, AllowedClass)
        SLATE_ARGUMENT(FString, ObjectPath)
        SLATE_ARGUMENT(bool, AllowClear)
        SLATE_ARGUMENT(bool, ShowThumbnail)
        SLATE_ARGUMENT(bool, ShowTypeIcon)
        SLATE_ARGUMENT(FString, PlaceholderText)
        SLATE_EVENT(FOnAssetSelected, OnAssetSelected)
    SLATE_END_ARGS()

    void Construct(const FArguments& InArgs);

    /**
     * Get current object path
     */
    FString GetObjectPath() const;

    /**
     * Set object path
     */
    void SetObjectPath(const FString& InObjectPath);

    /**
     * Clear selection
     */
    void ClearSelection();

private:
    TSharedPtr<SObjectPropertyEntryBox> ObjectSelector;
    UClass* AllowedClass;
    FString CurrentObjectPath;
    bool bAllowClear;
    bool bShowThumbnail;
    bool bShowTypeIcon;
    FString PlaceholderText;

    FOnAssetSelected OnAssetSelected;

    /**
     * Handle object selection changes
     */
    void OnObjectChanged(const FAssetData& AssetData);

    /**
     * Get display text for current selection
     */
    FText GetDisplayText() const;

    /**
     * Get object path for SObjectPropertyEntryBox
     */
    FString GetObjectPathForDisplay() const;
};

/**
 * Enhanced file path selector with validation and preview
 */
class VCCSIMEDITOR_API SEnhancedFilePathSelector : public SCompoundWidget
{
public:
    SLATE_BEGIN_ARGS(SEnhancedFilePathSelector) {}
        SLATE_ARGUMENT(FString, DialogTitle)
        SLATE_ARGUMENT(FString, FileTypeFilter)
        SLATE_ARGUMENT(FString, FilePath)
        SLATE_ARGUMENT(bool, DirectoryMode)
        SLATE_ARGUMENT(bool, ShowValidationIcon)
        SLATE_ARGUMENT(FString, PlaceholderText)
        SLATE_EVENT(FOnPathPicked, OnPathPicked)
    SLATE_END_ARGS()

    void Construct(const FArguments& InArgs);

    /**
     * Get current file path
     */
    FString GetFilePath() const;

    /**
     * Set file path
     */
    void SetFilePath(const FString& InFilePath);

    /**
     * Validate current path
     */
    bool IsPathValid() const;

private:
    TSharedPtr<SFilePathPicker> FilePathPicker;
    FString DialogTitle;
    FString FileTypeFilter;
    FString CurrentFilePath;
    bool bDirectoryMode;
    bool bShowValidationIcon;
    FString PlaceholderText;

    FOnPathPicked OnPathPicked;

    /**
     * Handle path changes
     */
    void OnPathChanged(const FString& NewPath);

    /**
     * Get validation icon based on path status
     */
    const FSlateBrush* GetValidationIcon() const;

    /**
     * Get validation icon color
     */
    FSlateColor GetValidationIconColor() const;

    /**
     * Get validation tooltip text
     */
    FText GetValidationTooltip() const;
};

/**
 * Common asset selector configurations
 */
struct VCCSIMEDITOR_API FAssetSelectorPresets
{
    // File filters
    static const FString ImageFileFilter;
    static const FString MeshFileFilter;
    static const FString TextFileFilter;
    static const FString ConfigFileFilter;
    static const FString AllFilesFilter;

    // Common asset classes
    static UClass* StaticMeshClass;
    static UClass* TextureClass;
    static UClass* MaterialClass;
    static UClass* MaterialInstanceClass;
    static UClass* SoundClass;
    static UClass* AnimationSequenceClass;

    /**
     * Initialize preset values
     */
    static void Initialize();
};