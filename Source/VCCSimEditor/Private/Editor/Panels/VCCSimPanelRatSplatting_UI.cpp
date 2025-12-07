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


DEFINE_LOG_CATEGORY_STATIC(LogRatSplattingUI, Log, All);

#include "Editor/Panels/VCCSimPanelRatSplatting.h"
#include "Utils/ColmapManager.h"
#include "Utils/SplattingManager.h"
#include "Utils/VCCSimUIHelpers.h"
#include "PropertyCustomizationHelpers.h"
#include "ThumbnailRendering/ThumbnailManager.h"

// ============================================================================
// RATSPLATTING UI CONSTRUCTION
// ============================================================================

TSharedRef<SWidget> FVCCSimPanelRatSplatting::CreateRatSplattingPanel()
{
    TSharedRef<SWidget> Panel = FVCCSimUIHelpers::CreateCollapsibleSection(TEXT("RatSplatting"),
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
        bRatSplattingSectionExpanded
    );

    return Panel;
}

TSharedRef<SWidget> FVCCSimPanelRatSplatting::CreateGSDataInputSection()
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

        // COLMAP Dataset Directory
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
                    .HintText(FText::FromString(TEXT("Path to COLMAP dataset with sparse/ and images/ folders")))
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

        // StaticMesh Selector
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            FVCCSimUIHelpers::CreatePropertyRow(TEXT("Initialization Mesh"),
                SNew(SObjectPropertyEntryBox)
                .ObjectPath_Lambda([this]() -> FString
                {
                    return GSConfig.SelectedMesh.IsValid() ? GSConfig.SelectedMesh.Get()->GetPathName() : FString();
                })
                .AllowedClass(UStaticMesh::StaticClass())
                .OnObjectChanged_Lambda([this](const FAssetData& AssetData)
                {
                    if (UStaticMesh* SelectedMesh = Cast<UStaticMesh>(AssetData.GetAsset()))
                    {
                        GSConfig.SelectedMesh = SelectedMesh;
                        SavePaths(); // Persist mesh selection
                        UE_LOG(LogRatSplattingUI, Log, TEXT("Selected mesh for initialization: %s"), *SelectedMesh->GetName());
                    }
                    else
                    {
                        GSConfig.SelectedMesh.Reset();
                        SavePaths(); // Persist the reset
                    }
                })
                .DisplayThumbnail(true)
                .ThumbnailPool(UThumbnailManager::Get().GetSharedThumbnailPool())
            )
        ];
}

TSharedRef<SWidget> FVCCSimPanelRatSplatting::CreateGSCameraParamsSection()
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

TSharedRef<SWidget> FVCCSimPanelRatSplatting::CreateGSTrainingParamsSection()
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
                        return GSConfig.bUseMeshTriangles ?
                        ECheckBoxState::Checked : ECheckBoxState::Unchecked; 
                    })
                    .OnCheckStateChanged_Lambda([this](ECheckBoxState NewState)
                    {
                        GSConfig.bUseMeshTriangles = (NewState == ECheckBoxState::Checked);
                    })
                    .ToolTipText(FText::FromString(TEXT("Use mesh triangles directly "
                                                        "instead of generating from points")))
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
                    .ToolTipText(FText::FromString(TEXT("Method for selecting triangles "
                                                        "from mesh (Random only for now)")))
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
                    100000, 10000000, 10000,
                    [this](int32 NewValue) { 
                        GSConfig.MaxMeshTriangles = NewValue; 
                    }
                )
            ]
        ];
}

TSharedRef<SWidget> FVCCSimPanelRatSplatting::CreateGSTrainingControlSection()
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
                .Text_Lambda([this]()
                {
                    return bDataPreparationInProgress ?
                        FText::FromString(TEXT("Preparing...")) :
                        FText::FromString(TEXT("Prepare Data"));
                })
                .VAlign(VAlign_Center)
                .HAlign(HAlign_Center)
                .IsEnabled_Lambda([this]() { return !bGSTrainingInProgress && !bDataPreparationInProgress && !bColmapPipelineInProgress; })
                .OnClicked_Lambda([this]() {
                    return OnGSPrepareTrainingDataClicked();
                })
                .ToolTipText(FText::FromString(TEXT("Generate training configuration and data files for RatSplatting/BRDF training (async for mesh triangles)")))
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
                        FText::FromString(TEXT("Train RatSplatting"));
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
            ]
            
            + SHorizontalBox::Slot()
            .MaxWidth(150)
            [
                SAssignNew(GSColmapTrainingButton, SButton)
                .Text(FText::FromString(TEXT("Train Tri-Splatting")))
                .VAlign(VAlign_Center)
                .HAlign(HAlign_Center)
                .IsEnabled_Lambda([this]() { return !bGSTrainingInProgress && !bColmapPipelineInProgress; })
                .OnClicked_Lambda([this]() {
                    return OnGSColmapTrainingClicked();
                })
            ]
        ];
}