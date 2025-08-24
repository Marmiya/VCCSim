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

/**
 * Integration guide for unified asset selectors in VCCSimEditor
 * 
 * This file demonstrates how to replace existing simple text displays and
 * file path inputs with modern, UE-standard asset selection components.
 */

#include "Editor/Widgets/AssetSelectorWidget.h"
#include "Editor/VCCSimDataStructures.h"

// ============================================================================
// PHASE 1B CONTINUATION: Asset Selector Integration
// ============================================================================

/**
 * Example 1: Replace Triangle Splatting Mesh Selection
 * 
 * Current implementation in VCCSimPanel_gs.cpp uses SObjectPropertyEntryBox directly.
 * The new unified approach provides consistent styling and enhanced features:
 */

/*
// OLD CODE (in VCCSimPanel_gs.cpp):
SNew(SObjectPropertyEntryBox)
.AllowedClass(UStaticMesh::StaticClass())
.ObjectPath_Lambda([this]()
{
    return GSConfig.SelectedMesh.IsValid() ? 
        GSConfig.SelectedMesh->GetPathName() : FString();
})
.OnObjectChanged_Lambda([this](const FAssetData& AssetData)
{
    GSConfig.SelectedMesh = Cast<UStaticMesh>(AssetData.GetAsset());
})

// NEW CODE (using unified factory):
FAssetSelectorFactory::CreateStaticMeshSelector(
    [this]() { return GSConfig.SelectedMesh.IsValid() ? GSConfig.SelectedMesh->GetPathName() : FString(); },
    [this](const FAssetData& AssetData) { 
        GSConfig.SelectedMesh = Cast<UStaticMesh>(AssetData.GetAsset());
        OnGSConfigurationChanged();
    },
    TEXT("Static Mesh"),  // Label
    true                   // Allow clear
)
*/

/**
 * Example 2: Replace File Path Inputs
 * 
 * Current implementation uses SEditableTextBox for file paths.
 * The new approach provides validation, browse buttons, and consistent UX:
 */

/*
// OLD CODE (typical file path input):
SNew(SEditableTextBox)
.Text_Lambda([this]() { return FText::FromString(GSConfig.ImageDirectory); })
.OnTextChanged_Lambda([this](const FText& NewText) { 
    GSConfig.ImageDirectory = NewText.ToString(); 
})

// NEW CODE (using file path selector):
FAssetSelectorFactory::CreateDirectorySelector(
    TEXT("Select Image Directory"),
    [this]() { return GSConfig.ImageDirectory; },
    [this](const FString& NewPath) { 
        GSConfig.ImageDirectory = NewPath;
        OnGSConfigurationChanged();
    },
    TEXT("Image Directory")
)
*/

/**
 * Example 3: Point Cloud File Selection
 * 
 * Replace simple text display with proper file selector:
 */

/*
// OLD CODE (simple text display):
SNew(STextBlock)
.Text_Lambda([this]() { 
    return FText::FromString(LoadedPointCloudPath.IsEmpty() ? 
        TEXT("No point cloud loaded") : 
        FPaths::GetCleanFilename(LoadedPointCloudPath)); 
})

// NEW CODE (enhanced file selector):
FAssetSelectorFactory::CreateFilePathSelector(
    TEXT("Select Point Cloud File"),
    FAssetSelectorPresets::MeshFileFilter + TEXT("|") + 
    TEXT("Point Cloud Files (*.ply;*.pcd;*.pts;*.xyz)|*.ply;*.pcd;*.pts;*.xyz"),
    [this]() { return LoadedPointCloudPath; },
    [this](const FString& NewPath) { 
        LoadedPointCloudPath = NewPath;
        LoadPointCloudFromFile(NewPath);
    },
    TEXT("Point Cloud File")
)
*/

/**
 * Example 4: Configuration File Paths
 * 
 * Enhanced file selection for configuration and pose files:
 */

/*
// For pose file selection:
FAssetSelectorFactory::CreateFilePathSelector(
    TEXT("Select Pose File"),
    TEXT("Pose Files (*.txt;*.csv;*.json)|*.txt;*.csv;*.json|") + FAssetSelectorPresets::AllFilesFilter,
    [this]() { return GSConfig.PoseFilePath; },
    [this](const FString& NewPath) { 
        GSConfig.PoseFilePath = NewPath;
        ValidateGSConfiguration();
    },
    TEXT("Pose File"),
    false  // Not directory mode
)

// For output directory selection:
FAssetSelectorFactory::CreateDirectorySelector(
    TEXT("Select Output Directory"),
    [this]() { return GSConfig.OutputDirectory; },
    [this](const FString& NewPath) { 
        GSConfig.OutputDirectory = NewPath;
        ValidateGSConfiguration();
    },
    TEXT("Output Directory")
)
*/

/**
 * Example 5: Material and Texture Selection
 * 
 * For point cloud visualization materials:
 */

/*
FAssetSelectorFactory::CreateMaterialSelector(
    [this]() { return PointCloudMaterial ? PointCloudMaterial->GetPathName() : FString(); },
    [this](const FAssetData& AssetData) { 
        PointCloudMaterial = Cast<UMaterialInterface>(AssetData.GetAsset());
        UpdatePointCloudVisualization();
    },
    TEXT("Point Cloud Material"),
    true  // Allow clear to use default material
)
*/

// ============================================================================
// INTEGRATION STEPS FOR EXISTING PANELS
// ============================================================================

/**
 * Step 1: Update VCCSimPanel_gs.cpp (Triangle Splatting Panel)
 * 
 * Replace the existing mesh selector and file path inputs:
 */

/*
TSharedRef<SWidget> SVCCSimPanel::CreateGSDataInputSection()
{
    return SNew(SVerticalBox)
        
        // Mesh Selection (enhanced)
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 4)
        [
            FAssetSelectorFactory::CreateStaticMeshSelector(
                [this]() { return GSConfig.SelectedMesh.IsValid() ? GSConfig.SelectedMesh->GetPathName() : FString(); },
                [this](const FAssetData& AssetData) { 
                    GSConfig.SelectedMesh = Cast<UStaticMesh>(AssetData.GetAsset());
                    ShowGSNotification(TEXT("Mesh selection updated"), false);
                },
                TEXT("Initialization Mesh"),
                true
            )
        ]
        
        // Image Directory (enhanced with validation)
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 4)
        [
            FAssetSelectorFactory::CreateDirectorySelector(
                TEXT("Select Image Directory"),
                [this]() { return GSConfig.ImageDirectory; },
                [this](const FString& NewPath) { 
                    GSConfig.ImageDirectory = NewPath;
                    ValidateGSConfiguration();
                },
                TEXT("Image Directory")
            )
        ]
        
        // Pose File (enhanced with validation)
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 4)
        [
            FAssetSelectorFactory::CreateFilePathSelector(
                TEXT("Select Pose File"),
                TEXT("Pose Files (*.txt;*.csv;*.json)|*.txt;*.csv;*.json|All Files (*.*)|*.*"),
                [this]() { return GSConfig.PoseFilePath; },
                [this](const FString& NewPath) { 
                    GSConfig.PoseFilePath = NewPath;
                    ValidateGSConfiguration();
                },
                TEXT("Pose File")
            )
        ]
        
        // Output Directory (enhanced)
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 4)
        [
            FAssetSelectorFactory::CreateDirectorySelector(
                TEXT("Select Output Directory"),
                [this]() { return GSConfig.OutputDirectory; },
                [this](const FString& NewPath) { 
                    GSConfig.OutputDirectory = NewPath;
                    ValidateGSConfiguration();
                },
                TEXT("Output Directory")
            )
        ];
}
*/

/**
 * Step 2: Update Point Cloud Panel
 */

/*
TSharedRef<SWidget> SVCCSimPanel::CreatePointCloudPanel()
{
    return SNew(SVerticalBox)
        
        // Point Cloud File Selection
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 4)
        [
            FAssetSelectorFactory::CreateFilePathSelector(
                TEXT("Select Point Cloud File"),
                TEXT("Point Cloud Files (*.ply;*.pcd;*.pts;*.xyz)|*.ply;*.pcd;*.pts;*.xyz|All Files (*.*)|*.*"),
                [this]() { return LoadedPointCloudPath; },
                [this](const FString& NewPath) { 
                    LoadedPointCloudPath = NewPath;
                    OnLoadPointCloudFromPath(NewPath);
                },
                TEXT("Point Cloud File")
            )
        ]
        
        // Point Cloud Material (optional)
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 4)
        [
            FAssetSelectorFactory::CreateMaterialSelector(
                [this]() { return PointCloudMaterial ? PointCloudMaterial->GetPathName() : FString(); },
                [this](const FAssetData& AssetData) { 
                    PointCloudMaterial = Cast<UMaterialInterface>(AssetData.GetAsset());
                    UpdatePointCloudVisualization();
                },
                TEXT("Visualization Material"),
                true  // Allow clear for default material
            )
        ]
        
        // Point cloud information display
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 8, 0, 0)
        [
            CreatePointCloudInfoDisplay()
        ];
}
*/

/**
 * Benefits of the Unified Asset Selector Approach:
 * 
 * 1. **Consistent UX**: All asset selections use the same styling and behavior
 * 2. **Enhanced Validation**: Real-time validation with visual feedback (icons, colors)
 * 3. **Better Error Handling**: Clear error messages and recovery suggestions
 * 4. **Drag & Drop Support**: Native UE drag-and-drop from Content Browser
 * 5. **Thumbnail Preview**: Visual preview of selected assets where applicable
 * 6. **Type Safety**: Compile-time type checking for asset classes
 * 7. **Extensibility**: Easy to add new asset types and validation rules
 * 8. **Accessibility**: Proper tooltips and keyboard navigation support
 */

// ============================================================================
// MIGRATION CHECKLIST
// ============================================================================

/**
 * □ Replace SObjectPropertyEntryBox direct usage with FAssetSelectorFactory::CreateStaticMeshSelector
 * □ Replace SEditableTextBox file paths with FAssetSelectorFactory::CreateFilePathSelector
 * □ Replace SEditableTextBox directory paths with FAssetSelectorFactory::CreateDirectorySelector
 * □ Add validation icons and status feedback
 * □ Update tooltips with helpful information
 * □ Test drag-and-drop functionality from Content Browser
 * □ Verify file type filtering works correctly
 * □ Ensure proper error handling for invalid paths
 * □ Add loading states for asset validation
 * □ Update notification system for asset changes
 */

// ============================================================================
// FUTURE ENHANCEMENTS
// ============================================================================

/**
 * Phase 2 Enhancements:
 * 
 * 1. **Asset Thumbnails**: Add thumbnail preview for mesh, texture assets
 * 2. **Recently Used**: Quick access to recently selected assets
 * 3. **Favorites**: Bookmark commonly used assets
 * 4. **Asset Validation**: Deep validation of asset compatibility
 * 5. **Batch Selection**: Select multiple assets at once
 * 6. **Asset Import**: Direct import from external files
 * 7. **Preview Window**: In-place preview of selected assets
 * 8. **Asset Details**: Show detailed information about selected assets
 */