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

DEFINE_LOG_CATEGORY_STATIC(LogTriangleSplattingUI, Log, All);

#include "Editor/Panels/VCCSimPanelTriangleSplatting.h"
#include "Utils/TriangleSplattingManager.h"
#include "Utils/VCCSimUIHelpers.h"
#include "Utils/ColmapManager.h"
#include "PropertyCustomizationHelpers.h"
#include "DesktopPlatformModule.h"
#include "ThumbnailRendering/ThumbnailManager.h"

// ============================================================================
// TRIANGLE SPLATTING UI CONSTRUCTION
// ============================================================================

TSharedRef<SWidget> FVCCSimPanelTriangleSplatting::CreateTriangleSplattingPanel()
{
    TSharedRef<SWidget> Panel = FVCCSimUIHelpers::CreateCollapsibleSection(TEXT("Triangle Splatting"), 
        SNew(SVerticalBox)
        
        // Data Input Section
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 4)
        [
            CreateGSDataInputSection()
        ]
        
        + SVerticalBox::Slot()
        .MaxHeight(1)
        [
            FVCCSimUIHelpers::CreateSeparator()
        ]
        
        // Camera Parameters Section
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 4)
        [
            CreateGSCameraParamsSection()
        ]
        
        + SVerticalBox::Slot()
        .MaxHeight(1)
        [
            FVCCSimUIHelpers::CreateSeparator()
        ]
        
        // Training Parameters Section
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 4)
        [
            CreateGSTrainingParamsSection()
        ]
        
        + SVerticalBox::Slot()
        .MaxHeight(1)
        [
            FVCCSimUIHelpers::CreateSeparator()
        ]
        
        // Training Control Section
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 4, 0, 0)
        [
            CreateGSTrainingControlSection()
        ],
        bTriangleSplattingSectionExpanded
    );

    return Panel;
}

TSharedRef<SWidget> FVCCSimPanelTriangleSplatting::CreateGSDataInputSection()
{
    return SNew(SVerticalBox)
        
        // Section header
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2, 0, 8)
        [
            FVCCSimUIHelpers::CreateSectionHeader(TEXT("Data Input"))
        ]
        
        // Image Directory
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            FVCCSimUIHelpers::CreatePropertyRow(TEXT("Image Directory"),
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                [
                    SAssignNew(GSImageDirectoryTextBox, SEditableTextBox)
                    .Text(FText::FromString(GSConfig.ImageDirectory))
                    .OnTextChanged_Lambda([this](const FText& Text)
                    {
                        GSConfig.ImageDirectory = Text.ToString();
                        SavePaths(); // Save when manually typed
                    })
                ]
                + SHorizontalBox::Slot()
                .AutoWidth()
                .Padding(5, 0, 0, 0)
                [
                    SNew(SButton)
                    .Text(FText::FromString(TEXT("Browse...")))
                    .OnClicked_Lambda([this]() {
                        return OnGSBrowseImageDirectoryClicked();
                    })
                ]
            )
        ]
        
        // Camera Intrinsics
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            FVCCSimUIHelpers::CreatePropertyRow(TEXT("Camera Intrinsics"),
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                [
                    SAssignNew(GSCameraIntrinsicsFileTextBox, SEditableTextBox)
                    .Text(FText::FromString(GSConfig.CameraIntrinsicsFilePath))
                    .OnTextChanged_Lambda([this](const FText& Text)
                    {
                        GSConfig.CameraIntrinsicsFilePath = Text.ToString();
                        OnGSCameraIntrinsicsLoaded();
                        SavePaths(); // Save when manually typed
                    })
                    .HintText(FText::FromString(TEXT("Select COLMAP cameras.txt or cameras.bin file")))
                ]
                + SHorizontalBox::Slot()
                .AutoWidth()
                .Padding(5, 0, 0, 0)
                [
                    SNew(SButton)
                    .Text(FText::FromString(TEXT("Browse...")))
                    .OnClicked_Lambda([this]() {
                        return OnGSBrowseCameraIntrinsicsFileClicked();
                    })
                ]
            )
        ]
        
        // Pose File
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            FVCCSimUIHelpers::CreatePropertyRow(TEXT("Pose File"),
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                [
                    SAssignNew(GSPoseFileTextBox, SEditableTextBox)
                    .Text(FText::FromString(GSConfig.PoseFilePath))
                    .OnTextChanged_Lambda([this](const FText& Text)
                    {
                        GSConfig.PoseFilePath = Text.ToString();
                        SavePaths(); // Save when manually typed
                    })
                ]
                + SHorizontalBox::Slot()
                .AutoWidth()
                .Padding(5, 0, 0, 0)
                [
                    SNew(SButton)
                    .Text(FText::FromString(TEXT("Browse...")))
                    .OnClicked_Lambda([this]() {
                        return OnGSBrowsePoseFileClicked();
                    })
                ]
            )
        ]
        
        // Output Directory
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            FVCCSimUIHelpers::CreatePropertyRow(TEXT("Output Directory"),
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                [
                    SAssignNew(GSOutputDirectoryTextBox, SEditableTextBox)
                    .Text(FText::FromString(GSConfig.OutputDirectory))
                    .OnTextChanged_Lambda([this](const FText& Text)
                    {
                        GSConfig.OutputDirectory = Text.ToString();
                        SavePaths(); // Save when manually typed
                    })
                ]
                + SHorizontalBox::Slot()
                .AutoWidth()
                .Padding(5, 0, 0, 0)
                [
                    SNew(SButton)
                    .Text(FText::FromString(TEXT("Browse...")))
                    .OnClicked_Lambda([this]() {
                        return OnGSBrowseOutputDirectoryClicked();
                    })
                ]
            )
        ]
        
        // COLMAP Dataset Path
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            FVCCSimUIHelpers::CreatePropertyRow(TEXT("COLMAP Dataset"),
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                [
                    SAssignNew(GSColmapDatasetTextBox, SEditableTextBox)
                    .Text(FText::FromString(GSConfig.ColmapDatasetPath))
                    .OnTextChanged_Lambda([this](const FText& Text)
                    {
                        GSConfig.ColmapDatasetPath = Text.ToString();
                        SavePaths(); // Save when manually typed
                    })
                    .HintText(FText::FromString(TEXT("Select COLMAP dataset folder (containing sparse/ images/ folders)")))
                ]
                + SHorizontalBox::Slot()
                .AutoWidth()
                .Padding(5, 0, 0, 0)
                [
                    SNew(SButton)
                    .Text(FText::FromString(TEXT("Browse...")))
                    .OnClicked_Lambda([this]() {
                        return OnGSBrowseColmapDatasetClicked();
                    })
                ]
            )
        ]
        
        // Mesh Selection with UE-style asset picker
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            FVCCSimUIHelpers::CreatePropertyRow(TEXT("Mesh Selection"),
                SNew(SObjectPropertyEntryBox)
                .AllowedClass(UStaticMesh::StaticClass())
                .ObjectPath_Lambda([this]()
                {
                    return GSConfig.SelectedMesh.IsValid() ? 
                        GSConfig.SelectedMesh->GetPathName() : FString();
                })
                .OnObjectChanged_Lambda([this](const FAssetData& AssetData)
                {
                    if (AssetData.IsValid())
                    {
                        if (UStaticMesh* NewMesh = Cast<UStaticMesh>(AssetData.GetAsset()))
                        {
                            GSConfig.SelectedMesh = NewMesh;
                            UE_LOG(LogTriangleSplattingUI, Log, TEXT("Selected mesh via asset picker: %s"), *NewMesh->GetName());
                            SavePaths(); // Save immediately after mesh selection
                        }
                    }
                    else
                    {
                        GSConfig.SelectedMesh.Reset();
                        UE_LOG(LogTriangleSplattingUI, Log, TEXT("Cleared mesh selection"));
                        SavePaths(); // Save immediately after mesh cleared
                    }
                })
                .AllowClear(true)
                .DisplayUseSelected(true)
                .DisplayBrowse(true)
                .DisplayThumbnail(true)
                .ThumbnailPool(UThumbnailManager::Get().GetSharedThumbnailPool())
            )
        ];
}

TSharedRef<SWidget> FVCCSimPanelTriangleSplatting::CreateGSCameraParamsSection()
{
    return SNew(SVerticalBox)
        
        // Section header
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2, 0, 4)
        [
            FVCCSimUIHelpers::CreateSectionHeader(TEXT("Camera Parameters"))
        ]
        
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(2, 2)
        [
            SNew(SHorizontalBox)
            
            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            .Padding(2, 0)
            [
                FVCCSimUIHelpers::CreateNumericPropertyRow<int32>(
                    TEXT("Width"),
                    GSImageWidthSpinBox,
                    GSImageWidthValue,
                    64, 7680, 64,
                    [this](int32 NewValue) { OnGSImageWidthChanged(NewValue); }
                )
            ]

            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            .Padding(2, 0)
            [
                FVCCSimUIHelpers::CreateNumericPropertyRow<int32>(
                    TEXT("Height"),
                    GSImageHeightSpinBox,
                    GSImageHeightValue,
                    64, 4320, 64,
                    [this](int32 NewValue) { OnGSImageHeightChanged(NewValue); }
                )
            ]
        ]

        // Focal Length Parameters (fx/fy)
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(2, 2)
        [
            SNew(SHorizontalBox)
            
            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            .Padding(2, 0)
            [
                FVCCSimUIHelpers::CreateNumericPropertyRow<float>(
                    TEXT("Focal Length X"),
                    GSFocalLengthXSpinBox,
                    GSFocalLengthXValue,
                    0.0f, 10000.0f, 10.0f,
                    [this](float NewValue) { OnGSFocalLengthXChanged(NewValue); }
                )
            ]

            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            .Padding(2, 0)
            [
                FVCCSimUIHelpers::CreateNumericPropertyRow<float>(
                    TEXT("Focal Length Y"),
                    GSFocalLengthYSpinBox,
                    GSFocalLengthYValue,
                    0.0f, 10000.0f, 10.0f,
                    [this](float NewValue) { OnGSFocalLengthYChanged(NewValue); }
                )
            ]
        ]

        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(2, 2)
        [
            SNew(SHorizontalBox)
            
            + SHorizontalBox::Slot()
            .FillWidth(0.5f)
            .Padding(2, 0)
            [
                FVCCSimUIHelpers::CreateNumericPropertyRow<float>(
                    TEXT("FOV (°)"),
                    GSFOVSpinBox,
                    GSFOVValue,
                    1.0f, 179.0f, 1.0f,
                    [this](float NewValue) { OnGSFOVChanged(NewValue); }
                )
            ]
            
            + SHorizontalBox::Slot()
            .FillWidth(0.5f)
            .Padding(2, 0)
            [
                SNew(SSpacer)
            ]
        ];
}

TSharedRef<SWidget> FVCCSimPanelTriangleSplatting::CreateGSTrainingParamsSection()
{
    return SNew(SVerticalBox)
        
        // Section header
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2, 0, 4)
        [
            FVCCSimUIHelpers::CreateSectionHeader(TEXT("Training Parameters"))
        ]
        
        // Max Iterations and Init Point Count (same row)
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(2, 2)
        [
            SNew(SHorizontalBox)
            
            // Max Iterations (left half)
            + SHorizontalBox::Slot()
            .FillWidth(0.5f)
            .Padding(0, 0, 5, 0)
            [
                FVCCSimUIHelpers::CreateNumericPropertyRow<int32>(
                    TEXT("Max Iterations"),
                    GSMaxIterationsSpinBox,
                    GSMaxIterationsValue,
                    1, TNumericLimits<int32>::Max(), 100,  // No upper limit
                    [this](int32 NewValue) { OnGSMaxIterationsChanged(NewValue); }
                )
            ]
            
            // Init Point Count (right half)
            + SHorizontalBox::Slot()
            .FillWidth(0.5f)
            .Padding(5, 0, 0, 0)
            [
                FVCCSimUIHelpers::CreateNumericPropertyRow<int32>(
                    TEXT("Init Point Count"),
                    GSInitPointCountSpinBox,
                    GSInitPointCountValue,
                    1, TNumericLimits<int32>::Max(), 1000,  // No upper limit
                    [this](int32 NewValue) { OnGSInitPointCountChanged(NewValue); }
                )
            ]
        ]
        
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 4, 0, 2)
        [
            FVCCSimUIHelpers::CreateSeparator()
        ]
        
        // Mesh Triangle Initialization Options
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(2, 4)
        [
            SNew(SHorizontalBox)
            
            // Use Mesh Triangles checkbox (left half)
            + SHorizontalBox::Slot()
            .FillWidth(0.5f)
            .Padding(0, 0, 5, 0)
            [
                FVCCSimUIHelpers::CreatePropertyRow(TEXT("Use Mesh Triangles"),
                    SNew(SCheckBox)
                    .IsChecked_Lambda([this]() 
                    { 
                        return GSConfig.bUseMeshTriangles ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; 
                    })
                    .OnCheckStateChanged_Lambda([this](ECheckBoxState NewState)
                    {
                        GSConfig.bUseMeshTriangles = (NewState == ECheckBoxState::Checked);
                    })
                    .ToolTipText(FText::FromString(TEXT("Use mesh triangles directly instead of generating from points")))
                )
            ]
            
            // Mesh Opacity (right half)
            + SHorizontalBox::Slot()
            .FillWidth(0.5f)
            .Padding(5, 0, 0, 0)
            [
                FVCCSimUIHelpers::CreateNumericPropertyRow<float>(
                    TEXT("Mesh Opacity"),
                    GSMeshOpacitySpinBox,
                    GSMeshOpacityValue,
                    0.0f, 1.0f, 0.05f,
                    [this](float NewValue) { 
                        GSConfig.MeshOpacity = NewValue; 
                    }
                )
            ]
        ]

        // Triangle Selection Method and Max Count
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(2, 2)
        [
            SNew(SHorizontalBox)
            
            // Triangle Selection Method (left half)
            + SHorizontalBox::Slot()
            .FillWidth(0.5f)
            .Padding(0, 0, 5, 0)
            [
                FVCCSimUIHelpers::CreatePropertyRow(TEXT("Triangle Method"),
                    SNew(STextBlock)
                    .Text(FText::FromString(TEXT("Random")))
                    .ToolTipText(FText::FromString(TEXT("Method for selecting triangles from mesh (Random only for now)")))
                )
            ]
            
            // Max Mesh Triangles (right half)
            + SHorizontalBox::Slot()
            .FillWidth(0.5f)
            .Padding(5, 0, 0, 0)
            [
                FVCCSimUIHelpers::CreateNumericPropertyRow<int32>(
                    TEXT("Max Mesh Triangles"),
                    GSMaxMeshTrianglesSpinBox,
                    GSMaxMeshTrianglesValue,
                    100000, 4000000, 10000,
                    [this](int32 NewValue) { 
                        GSConfig.MaxMeshTriangles = NewValue; 
                    }
                )
            ]
        ];
}

TSharedRef<SWidget> FVCCSimPanelTriangleSplatting::CreateGSTrainingControlSection()
{
    return SNew(SVerticalBox)
        
        // Section header
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2, 0, 4)
        [
            FVCCSimUIHelpers::CreateSectionHeader(TEXT("Training Control"))
        ]
        
        // Control buttons - first row
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            SNew(SHorizontalBox)
            
            + SHorizontalBox::Slot()
            .MaxWidth(150)
            .Padding(0, 0, 5, 0)
            [
                SNew(SButton)
                .Text(FText::FromString(TEXT("Test Transform")))
                .VAlign(VAlign_Center)
                .HAlign(HAlign_Center)
                .IsEnabled_Lambda([this]() { return !bGSTrainingInProgress; })
                .OnClicked_Lambda([this]() {
                    return OnGSTestTransformationClicked();
                })
                .ToolTipText(FText::FromString(TEXT("Export mesh and camera poses "
                                                    "as PLY files for MeshLab validation")))
            ]
            
            + SHorizontalBox::Slot()
            .MaxWidth(150)
            [
                SNew(SButton)
                .Text_Lambda([this]()
                {
                    return bColmapPipelineInProgress ? 
                        FText::FromString(TEXT("Stop COLMAP")) : 
                        FText::FromString(TEXT("COLMAP"));
                })
                .VAlign(VAlign_Center)
                .HAlign(HAlign_Center)
                .IsEnabled_Lambda([this]() { return !bGSTrainingInProgress; })
                .OnClicked_Lambda([this]()
                {
                    if (bColmapPipelineInProgress)
                    {
                        // Stop COLMAP pipeline
                        if (ColmapManager.IsValid())
                        {
                            ColmapManager->StopColmapPipeline();
                            bColmapPipelineInProgress = false;
                            FVCCSimUIHelpers::ShowNotification(TEXT("COLMAP pipeline stopped"));
                        }
                    }
                    else
                    {
                        // Start COLMAP pipeline
                        return OnGSExportColmapClicked();
                    }
                    return FReply::Handled();
                })
                .ToolTipText_Lambda([this]()
                {
                    return bColmapPipelineInProgress ?
                        FText::FromString(TEXT("Stop the running COLMAP pipeline")) :
                        FText::FromString(TEXT("Run complete COLMAP pipeline: "
                                               "Feature extraction → Matching → Sparse reconstruction. "
                                               "Creates timestamped dataset with full COLMAP reconstruction."));
                })
            ]
        ]
        
        // Training buttons - second row
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            SNew(SHorizontalBox)
            
            + SHorizontalBox::Slot()
            .MaxWidth(150)
            .Padding(0, 0, 5, 0)
            [
                SAssignNew(GSTrainingToggleButton, SButton)
                .Text_Lambda([this]()
                {
                    return bGSTrainingInProgress ? 
                        FText::FromString(TEXT("Stop")) : 
                        FText::FromString(TEXT("Train VCCSim"));
                })
                .VAlign(VAlign_Center)
                .HAlign(HAlign_Center)
                .OnClicked_Lambda([this]()
                {
                    if (bGSTrainingInProgress)
                    {
                        return OnGSStopTrainingClicked();
                    }
                    else
                    {
                        return OnGSStartTrainingClicked();
                    }
                })
                .ToolTipText(FText::FromString(TEXT("Train with VCCSim custom algorithm (train_vccsim.py)")))
            ]
            
            + SHorizontalBox::Slot()
            .MaxWidth(150)
            [
                SAssignNew(GSColmapTrainingButton, SButton)
                .Text(FText::FromString(TEXT("Train Original")))
                .VAlign(VAlign_Center)
                .HAlign(HAlign_Center)
                .IsEnabled_Lambda([this]() { return !bGSTrainingInProgress && !bColmapPipelineInProgress; })
                .OnClicked_Lambda([this]() {
                    return OnGSColmapTrainingClicked();
                })
                .ToolTipText(FText::FromString(TEXT("Train with original Triangle Splatting (train.py) for comparison")))
            ]
        ]
        
        // Training Status text
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 4)
        [
            FVCCSimUIHelpers::CreatePropertyRow(TEXT("Training Status"),
                SAssignNew(GSTrainingStatusText, STextBlock)
                .Text_Lambda([this]()
                {
                    if (GSTrainingManager.IsValid())
                    {
                        FString StatusText = GSTrainingManager->GetStatusMessage();
                        if (bGSTrainingInProgress)
                        {
                            float Progress = GSTrainingManager->GetTrainingProgress();
                            StatusText += FString::Printf(TEXT(" (%.1f%%)"), Progress * 100.0f);
                        }
                        return FText::FromString(StatusText);
                    }
                    return FText::FromString(TEXT("Ready"));
                })
                .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
            )
        ]
        
        // COLMAP Status text
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 4)
        [
            FVCCSimUIHelpers::CreatePropertyRow(TEXT("COLMAP Status"),
                SNew(STextBlock)
                .Text_Lambda([this]()
                {
                    if (ColmapManager.IsValid())
                    {
                        FString StatusText = ColmapManager->GetStatusMessage();
                        if (bColmapPipelineInProgress)
                        {
                            float Progress = ColmapManager->GetProgress();
                            StatusText += FString::Printf(TEXT(" (%.1f%%)"), Progress * 100.0f);
                        }
                        return FText::FromString(StatusText);
                    }
                    return FText::FromString(TEXT("Ready"));
                })
                .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
                .ColorAndOpacity_Lambda([this]()
                {
                    if (bColmapPipelineInProgress)
                    {
                        return FLinearColor::Green;
                    }
                    else if (ColmapManager.IsValid() && ColmapManager->GetProgress() >= 1.0f)
                    {
                        return FLinearColor::Blue;
                    }
                    return FLinearColor::White;
                })
            )
        ]
        
        ;
}


// ============================================================================
// TRIANGLE SPLATTING EVENT HANDLERS
// ============================================================================

void* FVCCSimPanelTriangleSplatting::GetParentWindowHandle()
{
    void* ParentWindowHandle = nullptr;
    
    // Use the active top level window for the panel
    TSharedPtr<SWindow> ActiveTopLevelWindow = FSlateApplication::Get().GetActiveTopLevelWindow();
    if (ActiveTopLevelWindow.IsValid())
    {
        ParentWindowHandle = ActiveTopLevelWindow->GetNativeWindow()->GetOSWindowHandle();
    }
    
    return ParentWindowHandle;
}

FReply FVCCSimPanelTriangleSplatting::OnGSBrowseImageDirectoryClicked()
{
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (DesktopPlatform)
    {
        FString SelectedDirectory;
        const bool bFolderSelected = DesktopPlatform->OpenDirectoryDialog(
            GetParentWindowHandle(),
            TEXT("Select Image Directory"),
            GSConfig.ImageDirectory.IsEmpty() ? FPaths::ProjectContentDir() : GSConfig.ImageDirectory,
            SelectedDirectory
        );

        if (bFolderSelected && !SelectedDirectory.IsEmpty())
        {
            GSConfig.ImageDirectory = SelectedDirectory;
            GSImageDirectoryTextBox->SetText(FText::FromString(SelectedDirectory));
            SavePaths(); // Save immediately after path change
        }
    }
    
    return FReply::Handled();
}

FReply FVCCSimPanelTriangleSplatting::OnGSBrowseCameraIntrinsicsFileClicked()
{
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (DesktopPlatform)
    {
        TArray<FString> SelectedFiles;
        const bool bFileSelected = DesktopPlatform->OpenFileDialog(
            GetParentWindowHandle(),
            TEXT("Select Camera Intrinsics File"),
            GSConfig.CameraIntrinsicsFilePath.IsEmpty() ? FPaths::ProjectSavedDir() : FPaths::GetPath(GSConfig.CameraIntrinsicsFilePath),
            TEXT("cameras.txt"),
            TEXT("COLMAP Camera Files|cameras.txt;cameras.bin|Text Files (*.txt)|*.txt|Binary Files (*.bin)|*.bin|All Files (*.*)|*.*"),
            EFileDialogFlags::None,
            SelectedFiles
        );

        if (bFileSelected && SelectedFiles.Num() > 0)
        {
            GSConfig.CameraIntrinsicsFilePath = SelectedFiles[0];
            GSCameraIntrinsicsFileTextBox->SetText(FText::FromString(SelectedFiles[0]));
            OnGSCameraIntrinsicsLoaded();
            SavePaths(); // Save immediately after path change
        }
    }
    
    return FReply::Handled();
}

FReply FVCCSimPanelTriangleSplatting::OnGSBrowsePoseFileClicked()
{
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (DesktopPlatform)
    {
        TArray<FString> SelectedFiles;
        const bool bFileSelected = DesktopPlatform->OpenFileDialog(
            GetParentWindowHandle(),
            TEXT("Select Pose File"),
            GSConfig.PoseFilePath.IsEmpty() ? FPaths::ProjectSavedDir() : FPaths::GetPath(GSConfig.PoseFilePath),
            TEXT("poses.txt"),
            TEXT("Text Files (*.txt)|*.txt|All Files (*.*)|*.*"),
            EFileDialogFlags::None,
            SelectedFiles
        );

        if (bFileSelected && SelectedFiles.Num() > 0)
        {
            GSConfig.PoseFilePath = SelectedFiles[0];
            GSPoseFileTextBox->SetText(FText::FromString(SelectedFiles[0]));
            SavePaths(); // Save immediately after path change
        }
    }
    
    return FReply::Handled();
}

FReply FVCCSimPanelTriangleSplatting::OnGSBrowseOutputDirectoryClicked()
{
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (DesktopPlatform)
    {
        FString SelectedDirectory;
        const bool bFolderSelected = DesktopPlatform->OpenDirectoryDialog(
            GetParentWindowHandle(),
            TEXT("Select Output Directory"),
            GSConfig.OutputDirectory.IsEmpty() ? FPaths::ProjectSavedDir() : GSConfig.OutputDirectory,
            SelectedDirectory
        );

        if (bFolderSelected && !SelectedDirectory.IsEmpty())
        {
            GSConfig.OutputDirectory = SelectedDirectory;
            GSOutputDirectoryTextBox->SetText(FText::FromString(SelectedDirectory));
            SavePaths(); // Save immediately after path change
        }
    }
    
    return FReply::Handled();
}

FReply FVCCSimPanelTriangleSplatting::OnGSBrowseColmapDatasetClicked()
{
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (DesktopPlatform)
    {
        FString SelectedDirectory;
        const bool bFolderSelected = DesktopPlatform->OpenDirectoryDialog(
            GetParentWindowHandle(),
            TEXT("Select COLMAP Dataset Directory"),
            GSConfig.ColmapDatasetPath.IsEmpty() ? GSConfig.OutputDirectory : GSConfig.ColmapDatasetPath,
            SelectedDirectory
        );

        if (bFolderSelected && !SelectedDirectory.IsEmpty())
        {
            // Update path regardless - validation will happen when training starts
            GSConfig.ColmapDatasetPath = SelectedDirectory;
            
            // Optionally validate and show feedback
            FString SparseDir = FPaths::Combine(SelectedDirectory, TEXT("sparse"));
            FString ImagesDir = FPaths::Combine(SelectedDirectory, TEXT("images"));
            
            if (FPaths::DirectoryExists(SparseDir) && FPaths::DirectoryExists(ImagesDir))
            {
                FVCCSimUIHelpers::ShowNotification(TEXT("COLMAP dataset path selected"), false);
            }
            else
            {
                FVCCSimUIHelpers::ShowNotification(TEXT("Warning: Directory doesn't look like "
                                        "a COLMAP dataset (missing sparse/ or images/)"), false);
            }
            
            UE_LOG(LogTriangleSplattingUI, Log, TEXT("Selected COLMAP dataset path: %s"), *SelectedDirectory);
            SavePaths(); // Save immediately after path change
        }
    }
    
    return FReply::Handled();
}

void FVCCSimPanelTriangleSplatting::OnGSFOVChanged(float NewValue)
{
    GSConfig.FOVDegrees = NewValue;
}

void FVCCSimPanelTriangleSplatting::OnGSImageWidthChanged(int32 NewValue)
{
    GSConfig.ImageWidth = NewValue;
}

void FVCCSimPanelTriangleSplatting::OnGSImageHeightChanged(int32 NewValue)
{
    GSConfig.ImageHeight = NewValue;
}

void FVCCSimPanelTriangleSplatting::OnGSFocalLengthXChanged(float NewValue)
{
    GSConfig.FocalLengthX = NewValue;
}

void FVCCSimPanelTriangleSplatting::OnGSFocalLengthYChanged(float NewValue)
{
    GSConfig.FocalLengthY = NewValue;
}

void FVCCSimPanelTriangleSplatting::OnGSMaxIterationsChanged(int32 NewValue)
{
    GSConfig.MaxIterations = NewValue;
}

void FVCCSimPanelTriangleSplatting::OnGSInitPointCountChanged(int32 NewValue)
{
    GSConfig.InitPointCount = NewValue;
}