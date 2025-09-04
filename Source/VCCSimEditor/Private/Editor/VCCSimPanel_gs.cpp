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

#include "Editor/VCCSimPanel.h"
#include "Utils/TriangleSplattingManager.h"
#include "Utils/ColmapManager.h"
#include "Utils/VCCSimDataConverter.h"
#include "DataStruct_IO/IOUtils.h"
#include "HAL/PlatformFilemanager.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Layout/SScrollBox.h"
#include "Widgets/Layout/SSeparator.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Input/SCheckBox.h"
#include "Widgets/Input/SNumericEntryBox.h"
#include "PropertyCustomizationHelpers.h"
#include "SlateOptMacros.h"
#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"
#include "Framework/Notifications/NotificationManager.h"
#include "Widgets/Notifications/SNotificationList.h"
#include "Engine/StaticMesh.h"
#include "UObject/ConstructorHelpers.h"
#include "Engine/Engine.h"
#include "ThumbnailRendering/ThumbnailManager.h"
#include "Widgets/Images/SImage.h"
#include "Framework/Application/SlateApplication.h"

BEGIN_SLATE_FUNCTION_BUILD_OPTIMIZATION

// ============================================================================
// TRIANGLE SPLATTING INITIALIZATION
// ============================================================================

void SVCCSimPanel::InitializeGSManager()
{
    // Initialize default values
    GSFOVValue = GSConfig.FOVDegrees;
    GSImageWidthValue = GSConfig.ImageWidth;
    GSImageHeightValue = GSConfig.ImageHeight;
    GSFocalLengthXValue = GSConfig.FocalLengthX;
    GSFocalLengthYValue = GSConfig.FocalLengthY;
    GSMaxIterationsValue = GSConfig.MaxIterations;
    GSInitPointCountValue = GSConfig.InitPointCount;
    
    // Create training manager
    GSTrainingManager = MakeShared<FTriangleSplattingManager>();
    
    // Bind delegates
    GSTrainingManager->OnTrainingProgressUpdated.BindLambda(
        [this](float Progress, FString StatusMessage)
    {
        // Progress is automatically tracked by GSTrainingManager
    });
    
    GSTrainingManager->OnTrainingCompleted.BindLambda(
        [this](bool bSuccessful, FString ResultMessage)
    {
        bGSTrainingInProgress = false;
        
        // Stop status update timer
        if (GEditor && GSStatusUpdateTimerHandle.IsValid())
        {
            GEditor->GetTimerManager()->ClearTimer(GSStatusUpdateTimerHandle);
            GSStatusUpdateTimerHandle.Invalidate();
        }
        
        ShowGSNotification(ResultMessage, !bSuccessful);
    });
}

void SVCCSimPanel::InitializeColmapManager()
{
    // Create COLMAP manager
    ColmapManager = MakeShared<FColmapManager>();
    
    // Bind delegates for COLMAP progress updates
    ColmapManager->OnProgressUpdated.BindLambda([this](float Progress, FString StatusMessage)
    {
        // Progress is automatically tracked by ColmapManager
    });
    
    ColmapManager->OnCompleted.BindLambda([this](bool bSuccessful, FString ResultMessage)
    {
        bColmapPipelineInProgress = false;
        
        if (bSuccessful)
        {
            // Auto-fill the COLMAP dataset path with the generated timestamped directory
            FString GeneratedDatasetPath = ColmapManager->GetTimestampedDirectory();
            if (!GeneratedDatasetPath.IsEmpty() && FPaths::DirectoryExists(GeneratedDatasetPath))
            {
                GSConfig.ColmapDatasetPath = GeneratedDatasetPath;
                if (GSColmapDatasetTextBox.IsValid())
                {
                    GSColmapDatasetTextBox->SetText(FText::FromString(GeneratedDatasetPath));
                }
                UE_LOG(LogTemp, Log, TEXT("Auto-filled COLMAP dataset "
                                          "path: %s"), *GeneratedDatasetPath);
                ShowGSNotification(FString::Printf(
                    TEXT("COLMAP completed! Dataset path auto-filled: %s"), 
                    *FPaths::GetCleanFilename(GeneratedDatasetPath)));
            }
            else
            {
                ShowGSNotification(ResultMessage, false);
            }
        }
        else
        {
            ShowGSNotification(ResultMessage, true);
        }
    });
}

// ============================================================================
// TRIANGLE SPLATTING UI CONSTRUCTION
// ============================================================================

TSharedRef<SWidget> SVCCSimPanel::CreateTriangleSplattingPanel()
{
    return CreateCollapsibleSection(TEXT("Triangle Splatting"), 
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
            CreateSeparator()
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
            CreateSeparator()
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
            CreateSeparator()
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
}

TSharedRef<SWidget> SVCCSimPanel::CreateGSDataInputSection()
{
    return SNew(SVerticalBox)
        
        // Section header
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2, 0, 8)
        [
            CreateSectionHeader(TEXT("Data Input"))
        ]
        
        // Image Directory
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            CreatePropertyRow(TEXT("Image Directory"),
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                [
                    SAssignNew(GSImageDirectoryTextBox, SEditableTextBox)
                    .Text(FText::FromString(GSConfig.ImageDirectory))
                    .OnTextChanged_Lambda([this](const FText& Text)
                    {
                        GSConfig.ImageDirectory = Text.ToString();
                    })
                ]
                + SHorizontalBox::Slot()
                .AutoWidth()
                .Padding(5, 0, 0, 0)
                [
                    SNew(SButton)
                    .Text(FText::FromString(TEXT("Browse...")))
                    .OnClicked(this, &SVCCSimPanel::OnGSBrowseImageDirectoryClicked)
                ]
            )
        ]
        
        // Pose File
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            CreatePropertyRow(TEXT("Pose File"),
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                [
                    SAssignNew(GSPoseFileTextBox, SEditableTextBox)
                    .Text(FText::FromString(GSConfig.PoseFilePath))
                    .OnTextChanged_Lambda([this](const FText& Text)
                    {
                        GSConfig.PoseFilePath = Text.ToString();
                    })
                ]
                + SHorizontalBox::Slot()
                .AutoWidth()
                .Padding(5, 0, 0, 0)
                [
                    SNew(SButton)
                    .Text(FText::FromString(TEXT("Browse...")))
                    .OnClicked(this, &SVCCSimPanel::OnGSBrowsePoseFileClicked)
                ]
            )
        ]
        
        // Output Directory
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            CreatePropertyRow(TEXT("Output Directory"),
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                [
                    SAssignNew(GSOutputDirectoryTextBox, SEditableTextBox)
                    .Text(FText::FromString(GSConfig.OutputDirectory))
                    .OnTextChanged_Lambda([this](const FText& Text)
                    {
                        GSConfig.OutputDirectory = Text.ToString();
                    })
                ]
                + SHorizontalBox::Slot()
                .AutoWidth()
                .Padding(5, 0, 0, 0)
                [
                    SNew(SButton)
                    .Text(FText::FromString(TEXT("Browse...")))
                    .OnClicked(this, &SVCCSimPanel::OnGSBrowseOutputDirectoryClicked)
                ]
            )
        ]
        
        // COLMAP Dataset Path
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            CreatePropertyRow(TEXT("COLMAP Dataset"),
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                [
                    SAssignNew(GSColmapDatasetTextBox, SEditableTextBox)
                    .Text_Lambda([this]()
                    {
                        return FText::FromString(GSConfig.ColmapDatasetPath);
                    })
                    .OnTextChanged_Lambda([this](const FText& Text)
                    {
                        GSConfig.ColmapDatasetPath = Text.ToString();
                    })
                    .HintText(FText::FromString(TEXT("Select COLMAP dataset folder (containing sparse/ images/ folders)")))
                ]
                + SHorizontalBox::Slot()
                .AutoWidth()
                .Padding(5, 0, 0, 0)
                [
                    SNew(SButton)
                    .Text(FText::FromString(TEXT("Browse...")))
                    .OnClicked(this, &SVCCSimPanel::OnGSBrowseColmapDatasetClicked)
                ]
            )
        ]
        
        // Mesh Selection with UE-style asset picker
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            CreatePropertyRow(TEXT("Mesh Selection"),
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
                            UE_LOG(LogTemp, Log, TEXT("Selected mesh via asset picker: %s"), *NewMesh->GetName());
                        }
                    }
                    else
                    {
                        GSConfig.SelectedMesh.Reset();
                        UE_LOG(LogTemp, Log, TEXT("Cleared mesh selection"));
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

TSharedRef<SWidget> SVCCSimPanel::CreateGSCameraParamsSection()
{
    return SNew(SVerticalBox)
        
        // Section header
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2, 0, 4)
        [
            CreateSectionHeader(TEXT("Camera Parameters"))
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
                CreateGSNumericPropertyRow<int32>(
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
                CreateGSNumericPropertyRow<int32>(
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
                CreateGSNumericPropertyRow<float>(
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
                CreateGSNumericPropertyRow<float>(
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
                CreateGSNumericPropertyRow<float>(
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

TSharedRef<SWidget> SVCCSimPanel::CreateGSTrainingParamsSection()
{
    return SNew(SVerticalBox)
        
        // Section header
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2, 0, 4)
        [
            CreateSectionHeader(TEXT("Training Parameters"))
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
                CreateGSNumericPropertyRow<int32>(
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
                CreateGSNumericPropertyRow<int32>(
                    TEXT("Init Point Count"),
                    GSInitPointCountSpinBox,
                    GSInitPointCountValue,
                    1, TNumericLimits<int32>::Max(), 1000,  // No upper limit
                    [this](int32 NewValue) { OnGSInitPointCountChanged(NewValue); }
                )
            ]
        ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateGSTrainingControlSection()
{
    return SNew(SVerticalBox)
        
        // Section header
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2, 0, 4)
        [
            CreateSectionHeader(TEXT("Training Control"))
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
                .OnClicked(this, &SVCCSimPanel::OnGSTestTransformationClicked)
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
                            ShowGSNotification(TEXT("COLMAP pipeline stopped"));
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
                .OnClicked(this, &SVCCSimPanel::OnGSColmapTrainingClicked)
                .ToolTipText(FText::FromString(TEXT("Train with original Triangle Splatting (train.py) for comparison")))
            ]
        ]
        
        // Training Status text
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 4)
        [
            CreatePropertyRow(TEXT("Training Status"),
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
            CreatePropertyRow(TEXT("COLMAP Status"),
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
        
        // Current Loss Display
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 4)
        [
            CreatePropertyRow(TEXT("Current Loss"),
                SNew(STextBlock)
                .Text_Lambda([this]()
                {
                    if (bGSTrainingInProgress && !GSCurrentLoss.IsEmpty())
                    {
                        return FText::FromString(GSCurrentLoss);
                    }
                    return FText::FromString(TEXT("N/A"));
                })
                .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
                .ColorAndOpacity_Lambda([this]()
                {
                    return bGSTrainingInProgress ? FLinearColor::Green : FLinearColor::White;
                })
            )
        ];
}

// ============================================================================
// TRIANGLE SPLATTING UI HELPER FUNCTIONS
// ============================================================================

template<typename T>
TSharedRef<SWidget> SVCCSimPanel::CreateGSNumericPropertyRow(
    const FString& Label,
    TSharedPtr<SNumericEntryBox<T>>& SpinBox,
    TOptional<T>& Value,
    T MinValue,
    T MaxValue,
    T DeltaValue,
    TFunction<void(T)> OnValueChanged)
{
    return CreatePropertyRow(Label,
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(FColor(5, 5, 5, 255))
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

// ============================================================================
// TRIANGLE SPLATTING EVENT HANDLERS
// ============================================================================

void* SVCCSimPanel::GetParentWindowHandle()
{
    void* ParentWindowHandle = nullptr;
    
    // Try to find the widget's parent window first
    TSharedPtr<SWindow> ParentWindow = FSlateApplication::Get().FindWidgetWindow(AsShared());
    if (ParentWindow.IsValid())
    {
        ParentWindowHandle = ParentWindow->GetNativeWindow()->GetOSWindowHandle();
    }
    else
    {
        // Fallback to the active top level window
        TSharedPtr<SWindow> ActiveTopLevelWindow = FSlateApplication::Get().GetActiveTopLevelWindow();
        if (ActiveTopLevelWindow.IsValid())
        {
            ParentWindowHandle = ActiveTopLevelWindow->GetNativeWindow()->GetOSWindowHandle();
        }
    }
    
    return ParentWindowHandle;
}

FReply SVCCSimPanel::OnGSBrowseImageDirectoryClicked()
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
        }
    }
    
    return FReply::Handled();
}

FReply SVCCSimPanel::OnGSBrowsePoseFileClicked()
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
        }
    }
    
    return FReply::Handled();
}

FReply SVCCSimPanel::OnGSBrowseOutputDirectoryClicked()
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
        }
    }
    
    return FReply::Handled();
}

FReply SVCCSimPanel::OnGSBrowseColmapDatasetClicked()
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
                ShowGSNotification(TEXT("COLMAP dataset path selected"), false);
            }
            else
            {
                ShowGSNotification(TEXT("Warning: Directory doesn't look like "
                                        "a COLMAP dataset (missing sparse/ or images/)"), false);
            }
            
            UE_LOG(LogTemp, Log, TEXT("Selected COLMAP dataset path: %s"), *SelectedDirectory);
        }
    }
    
    return FReply::Handled();
}

void SVCCSimPanel::OnGSFOVChanged(float NewValue)
{
    GSConfig.FOVDegrees = NewValue;
}

void SVCCSimPanel::OnGSImageWidthChanged(int32 NewValue)
{
    GSConfig.ImageWidth = NewValue;
}

void SVCCSimPanel::OnGSImageHeightChanged(int32 NewValue)
{
    GSConfig.ImageHeight = NewValue;
}

void SVCCSimPanel::OnGSFocalLengthXChanged(float NewValue)
{
    GSConfig.FocalLengthX = NewValue;
}

void SVCCSimPanel::OnGSFocalLengthYChanged(float NewValue)
{
    GSConfig.FocalLengthY = NewValue;
}

void SVCCSimPanel::OnGSMaxIterationsChanged(int32 NewValue)
{
    GSConfig.MaxIterations = NewValue;
}

void SVCCSimPanel::OnGSInitPointCountChanged(int32 NewValue)
{
    GSConfig.InitPointCount = NewValue;
}

// ============================================================================
// TRIANGLE SPLATTING TRAINING CONTROL
// ============================================================================

FReply SVCCSimPanel::OnGSStartTrainingClicked()
{
    if (ValidateGSConfiguration())
    {
        if (GSTrainingManager->StartTraining(GSConfig))
        {
            bGSTrainingInProgress = true;
            
            // Training started
            GSCurrentLoss = TEXT("N/A");
            
            // Start status update timer
            if (GEditor)
            {
                GEditor->GetTimerManager()->SetTimer(
                    GSStatusUpdateTimerHandle,
                    FTimerDelegate::CreateLambda([this]()
                    {
                        if (GSTrainingManager.IsValid())
                        {
                            GSTrainingManager->UpdateTrainingStatus();
                        }
                    }),
                    1.0f, // Update every second
                    true  // Loop
                );
            }
            
            ShowGSNotification(TEXT("Triangle Splatting training started"));
        }
        else
        {
            ShowGSNotification(TEXT("Failed to start training process"), true);
        }
    }
    
    return FReply::Handled();
}

FReply SVCCSimPanel::OnGSStopTrainingClicked()
{
    if (GSTrainingManager.IsValid())
    {
        GSTrainingManager->StopTraining();
    }
    
    bGSTrainingInProgress = false;
    
    // Stop status update timer
    if (GEditor && GSStatusUpdateTimerHandle.IsValid())
    {
        GEditor->GetTimerManager()->ClearTimer(GSStatusUpdateTimerHandle);
        GSStatusUpdateTimerHandle.Invalidate();
    }
    
    ShowGSNotification(TEXT("Training stopped"));
    
    return FReply::Handled();
}

FReply SVCCSimPanel::OnGSColmapTrainingClicked()
{
    // Validate COLMAP dataset path
    if (GSConfig.ColmapDatasetPath.IsEmpty() || !FPaths::DirectoryExists(GSConfig.ColmapDatasetPath))
    {
        ShowGSNotification(TEXT("Please specify a valid COLMAP dataset path"), true);
        return FReply::Handled();
    }
    
    // Validate COLMAP dataset structure
    FString SparseDir = FPaths::Combine(GSConfig.ColmapDatasetPath, TEXT("sparse"));
    FString ImagesDir = FPaths::Combine(GSConfig.ColmapDatasetPath, TEXT("images"));
    
    if (!FPaths::DirectoryExists(SparseDir))
    {
        ShowGSNotification(TEXT("Invalid COLMAP dataset - missing sparse/ folder"), true);
        return FReply::Handled();
    }
    
    if (!FPaths::DirectoryExists(ImagesDir))
    {
        ShowGSNotification(TEXT("Invalid COLMAP dataset - missing images/ folder"), true);
        return FReply::Handled();
    }
    
    // Check for essential COLMAP files in sparse directory (support both txt and bin formats)
    FString SparseSubDir = FPaths::Combine(SparseDir, TEXT("0"));
    
    // Check cameras file (txt or bin)
    FString CamerasTxtFile = FPaths::Combine(SparseSubDir, TEXT("cameras.txt"));
    FString CamerasBinFile = FPaths::Combine(SparseSubDir, TEXT("cameras.bin"));
    bool bHasCameras = FPaths::FileExists(CamerasTxtFile) || FPaths::FileExists(CamerasBinFile);
    
    // Check images file (txt or bin)
    FString ImagesTxtFile = FPaths::Combine(SparseSubDir, TEXT("images.txt"));
    FString ImagesBinFile = FPaths::Combine(SparseSubDir, TEXT("images.bin"));
    bool bHasImages = FPaths::FileExists(ImagesTxtFile) || FPaths::FileExists(ImagesBinFile);
    
    // Check points3D file (txt or bin)
    FString Points3DTxtFile = FPaths::Combine(SparseSubDir, TEXT("points3D.txt"));
    FString Points3DBinFile = FPaths::Combine(SparseSubDir, TEXT("points3D.bin"));
    bool bHasPoints3D = FPaths::FileExists(Points3DTxtFile) || FPaths::FileExists(Points3DBinFile);
    
    if (!bHasCameras || !bHasImages || !bHasPoints3D)
    {
        ShowGSNotification(TEXT("Invalid COLMAP dataset - missing cameras, images or points3D files (txt or bin format) in sparse/0/ folder"), true);
        return FReply::Handled();
    }
    
    if (GSConfig.OutputDirectory.IsEmpty())
    {
        ShowGSNotification(TEXT("Please specify an output directory"), true);
        return FReply::Handled();
    }
    
    // Create output directory if it doesn't exist
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.DirectoryExists(*GSConfig.OutputDirectory))
    {
        if (!PlatformFile.CreateDirectoryTree(*GSConfig.OutputDirectory))
        {
            ShowGSNotification(TEXT("Failed to create output directory"), true);
            return FReply::Handled();
        }
    }
    
    UE_LOG(LogTemp, Log, TEXT("Starting Triangle Splatting training with "
                              "COLMAP dataset: %s"), *GSConfig.ColmapDatasetPath);
    
    // Start Triangle Splatting training directly with COLMAP dataset
    StartTriangleSplattingWithColmapData(GSConfig.ColmapDatasetPath);
    
    return FReply::Handled();
}

void SVCCSimPanel::StartTriangleSplattingWithColmapData(const FString& ColmapDatasetPath)
{
    // Use original train.py for comparison experiments
    FString TriangleSplattingRoot = FPaths::Combine(FPaths::ProjectPluginsDir(), TEXT("VCCSim/Source/triangle-splatting"));
    FString TrainingScript = FPaths::Combine(TriangleSplattingRoot, TEXT("train.py"));
    
    if (!FPaths::FileExists(TrainingScript))
    {
        ShowGSNotification(TEXT("Original Triangle Splatting train.py script not found"), true);
        return;
    }
    
    UE_LOG(LogTemp, Log, TEXT("Using original Triangle Splatting script for comparison: %s"), *TrainingScript);
    
    // Create Triangle Splatting output directory with timestamp
    FString TSOutputParentDir = FPaths::Combine(GSConfig.OutputDirectory, TEXT("triangle_splatting_output"));
    
    // Generate timestamp for this training session
    FDateTime Now = FDateTime::Now();
    FString Timestamp = Now.ToString(TEXT("%Y%m%d_%H%M%S"));
    FString SessionDirName = FString::Printf(TEXT("training_%s"), *Timestamp);
    FString TSOutputDir = FPaths::Combine(TSOutputParentDir, SessionDirName);
    
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.DirectoryExists(*TSOutputDir))
    {
        PlatformFile.CreateDirectoryTree(*TSOutputDir);
    }
    
    UE_LOG(LogTemp, Log, TEXT("Triangle Splatting output directory: %s"), *TSOutputDir);
    
    // Build Triangle Splatting command with micromamba environment Python
    // Try to find micromamba triangle_splatting environment Python executable
    FString PythonCommand;
    FString MicromambaPython = TEXT("C:/micromamba/envs/triangle_splatting/python.exe");
    
    if (FPaths::FileExists(MicromambaPython))
    {
        PythonCommand = MicromambaPython;
        UE_LOG(LogTemp, Log, TEXT("Using micromamba triangle_splatting Python: %s"), *PythonCommand);
    }
    else
    {
        // Fallback to system python
        PythonCommand = TEXT("python");
        UE_LOG(LogTemp, Warning, TEXT("Micromamba triangle_splatting environment not found, using system python"));
    }
    
    FString Arguments = FString::Printf(TEXT("\"%s\" -s \"%s\" -m \"%s\" --eval"), 
        *TrainingScript, *ColmapDatasetPath, *TSOutputDir);
    
    // Add outdoor flag if scene seems outdoor (based on camera coverage)
    // This could be enhanced with better outdoor detection logic
    Arguments += TEXT(" --outdoor");
    
    UE_LOG(LogTemp, Log, TEXT("Starting Triangle Splatting training: %s %s"), *PythonCommand, *Arguments);
    
    // Start Triangle Splatting training process
    if (GSTrainingManager->StartColmapTraining(PythonCommand, Arguments, TSOutputDir))
    {
        bGSTrainingInProgress = true;
        
        // Training with COLMAP data started
        GSCurrentLoss = TEXT("N/A");
        
        // Start status update timer
        if (GEditor)
        {
            GEditor->GetTimerManager()->SetTimer(
                GSStatusUpdateTimerHandle,
                FTimerDelegate::CreateLambda([this]()
                {
                    if (GSTrainingManager.IsValid())
                    {
                        GSTrainingManager->UpdateTrainingStatus();
                        
                        // Update training status
                    }
                }),
                1.0f, // Update every second
                true  // Loop
            );
        }
        
        ShowGSNotification(TEXT("Triangle Splatting training with COLMAP data started"));
    }
    else
    {
        ShowGSNotification(TEXT("Failed to start Triangle Splatting training process"), true);
    }
}

FReply SVCCSimPanel::OnGSTestTransformationClicked()
{
    // Validate basic configuration first
    if (GSConfig.OutputDirectory.IsEmpty())
    {
        ShowGSNotification(TEXT("Please specify an output directory first"), true);
        return FReply::Handled();
    }
    
    // Ensure output directory exists
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.DirectoryExists(*GSConfig.OutputDirectory))
    {
        if (!PlatformFile.CreateDirectoryTree(*GSConfig.OutputDirectory))
        {
            ShowGSNotification(TEXT("Failed to create output directory"), true);
            return FReply::Handled();
        }
    }
    
    bool bExportedAny = false;
    FString StatusMessage;
    
    // Export selected mesh to PLY (both original UE and transformed Triangle Splatting coordinates)
    if (GSConfig.SelectedMesh.IsValid())
    {
        try 
        {
            // Export original UE coordinates
            int32 PointCount = GSInitPointCountValue.Get(10000);
            FPointCloudData OriginalMesh = FVCCSimDataConverter::ConvertMeshToPointCloud(
                GSConfig.SelectedMesh.Get(), PointCount, false); 
            
            FString MeshUEPath = FPaths::Combine(GSConfig.OutputDirectory,
                TEXT("test_mesh_ue_coordinates.ply"));
            if (FVCCSimDataConverter::SavePointCloudToPLY(OriginalMesh, MeshUEPath))
            {
                StatusMessage += FString::Printf(TEXT("Original mesh (UE coords)\n"));
                bExportedAny = true;
            }
            
            // Export transformed Triangle Splatting coordinates
            FPointCloudData TransformedMesh = FVCCSimDataConverter::ConvertMeshToPointCloud(
                GSConfig.SelectedMesh.Get(), PointCount, true);
            
            FString MeshTSPath = FPaths::Combine(GSConfig.OutputDirectory,
                TEXT("test_mesh_ts_coordinates.ply"));
            if (FVCCSimDataConverter::SavePointCloudToPLY(TransformedMesh, MeshTSPath))
            {
                StatusMessage += FString::Printf(TEXT("Transformed mesh (TS coords)\n"));
                bExportedAny = true;
                
                // Log bounding boxes for comparison
                FVector UEMin = FVector(FLT_MAX), UEMax = FVector(-FLT_MAX);
                FVector TSMin = FVector(FLT_MAX), TSMax = FVector(-FLT_MAX);
                
                for (const FRatPoint& Point : OriginalMesh.Points)
                {
                    UEMin = FVector::Min(UEMin, Point.Position);
                    UEMax = FVector::Max(UEMax, Point.Position);
                }
                
                for (const FRatPoint& Point : TransformedMesh.Points)
                {
                    TSMin = FVector::Min(TSMin, Point.Position);
                    TSMax = FVector::Max(TSMax, Point.Position);
                }
                
                StatusMessage += FString::Printf(
                TEXT("UE mesh bbox: Min(%.2f,%.2f,%.2f) Max(%.2f,%.2f,%.2f)\n"), 
                    UEMin.X, UEMin.Y, UEMin.Z, UEMax.X, UEMax.Y, UEMax.Z);
                StatusMessage += FString::Printf(
                TEXT("TS mesh bbox: Min(%.2f,%.2f,%.2f) Max(%.2f,%.2f,%.2f)\n"), 
                        TSMin.X, TSMin.Y, TSMin.Z, TSMax.X, TSMax.Y, TSMax.Z);
            }
            else
            {
                StatusMessage += TEXT("✗ Failed to export transformed mesh\n");
            }
        }
        catch (...)
        {
            StatusMessage += TEXT("✗ Error converting mesh\n");
        }
    }
    else
    {
        StatusMessage += TEXT("⚠ No mesh selected\n");
    }
    
    // Export camera poses as points with normals (camera orientations)
    if (!GSConfig.PoseFilePath.IsEmpty() && FPaths::FileExists(GSConfig.PoseFilePath))
    {
        try
        {
            FCameraIntrinsics Intrinsics = FVCCSimDataConverter::ConvertCameraParamsWithFocalLength(
                GSConfig.FOVDegrees, GSConfig.ImageWidth, GSConfig.ImageHeight,
                GSConfig.FocalLengthX, GSConfig.FocalLengthY);
            
            TArray<FCameraInfo> CameraInfos = FVCCSimDataConverter::ConvertPoseFile(
                GSConfig.PoseFilePath, GSConfig.ImageDirectory, Intrinsics);
            
            if (CameraInfos.Num() > 0)
            {
                FString CameraPLYPath = FPaths::Combine(GSConfig.OutputDirectory,
                    TEXT("test_cameras_transformed.ply"));
                ExportCamerasToPLY(CameraInfos, CameraPLYPath);
                StatusMessage += FString::Printf(
                    TEXT("✓ %d cameras exported\n"), CameraInfos.Num());
                bExportedAny = true;
            }
            else
            {
                StatusMessage += TEXT("✗ No valid cameras found in pose file\n");
            }
        }
        catch (...)
        {
            StatusMessage += TEXT("✗ Error converting camera poses\n");
        }
    }
    else
    {
        StatusMessage += TEXT("⚠ No valid pose file specified\n");
    }
    
    // Show result
    if (bExportedAny)
    {
        StatusMessage += TEXT("\nOpen the PLY files to verify coordinate transformation!");
        ShowGSNotification(StatusMessage);
    }
    else
    {
        ShowGSNotification(TEXT("No data was exported. Please check "
                                "mesh selection and pose file."), true);
    }
    
    return FReply::Handled();
}

FReply SVCCSimPanel::OnGSExportColmapClicked()
{
    // Validate basic configuration first
    if (GSConfig.OutputDirectory.IsEmpty())
    {
        ShowGSNotification(TEXT("Please specify an output directory first"), true);
        return FReply::Handled();
    }
    
    if (GSConfig.PoseFilePath.IsEmpty() || !FPaths::FileExists(GSConfig.PoseFilePath))
    {
        ShowGSNotification(TEXT("Please specify a valid pose file"), true);
        return FReply::Handled();
    }
    
    if (GSConfig.ImageDirectory.IsEmpty() || !FPaths::DirectoryExists(GSConfig.ImageDirectory))
    {
        ShowGSNotification(TEXT("Please specify a valid image directory"), true);
        return FReply::Handled();
    }
    
    try
    {        
        // Start COLMAP pipeline asynchronously
        FString ColmapExecutablePath = TEXT("D:\\colmap-x64-windows-cuda\\bin");
        
        // Create COLMAP-specific output directory
        FString ColmapOutputDir = FPaths::Combine(GSConfig.OutputDirectory, TEXT("colmap_output"));
        
        if (ColmapManager->StartColmapPipeline(GSConfig.ImageDirectory,
            ColmapOutputDir, ColmapExecutablePath))
        {
            bColmapPipelineInProgress = true;
            ShowGSNotification(TEXT("COLMAP pipeline started in background\n\n"));
        }
        else
        {
            ShowGSNotification(TEXT("Failed to start COLMAP pipeline\n\n")
                              TEXT("Pipeline may already be running"), true);
        }
    }
    catch (...)
    {
        ShowGSNotification(TEXT("Unexpected error during COLMAP pipeline execution\n\n")
                          TEXT("Check UE log for detailed error information"), true);
    }
    
    return FReply::Handled();
}

// ============================================================================
// TRIANGLE SPLATTING TRAINING VALIDATION AND UTILITIES
// ============================================================================

bool SVCCSimPanel::ValidateGSConfiguration()
{
    TArray<FString> ErrorMessages;
    
    // Check required paths
    if (GSConfig.ImageDirectory.IsEmpty() || !FPaths::DirectoryExists(GSConfig.ImageDirectory))
    {
        ErrorMessages.Add(TEXT("Valid image directory is required"));
    }
    
    if (GSConfig.PoseFilePath.IsEmpty() || !FPaths::FileExists(GSConfig.PoseFilePath))
    {
        ErrorMessages.Add(TEXT("Valid pose file is required"));
    }
    
    if (GSConfig.OutputDirectory.IsEmpty())
    {
        ErrorMessages.Add(TEXT("Output directory is required"));
    }
    
    // Check mesh selection
    if (!GSConfig.SelectedMesh.IsValid())
    {
        ErrorMessages.Add(TEXT("Please select a mesh for initialization"));
    }
    
    // Check camera parameters
    if (GSConfig.FOVDegrees <= 0 || GSConfig.FOVDegrees >= 180)
    {
        ErrorMessages.Add(TEXT("FOV must be between 1 and 179 degrees"));
    }
    
    if (GSConfig.ImageWidth <= 0 || GSConfig.ImageHeight <= 0)
    {
        ErrorMessages.Add(TEXT("Image dimensions must be positive"));
    }
    
    // Check training parameters
    if (GSConfig.MaxIterations <= 0)
    {
        ErrorMessages.Add(TEXT("Max iterations must be positive"));
    }
    
    if (GSConfig.InitPointCount <= 0)
    {
        ErrorMessages.Add(TEXT("Init point count must be positive"));
    }
    
    
    if (ErrorMessages.Num() > 0)
    {
        FString CombinedError = FString::Join(ErrorMessages, TEXT("\n"));
        ShowGSNotification(CombinedError, true);
        return false;
    }
    
    return true;
}

void SVCCSimPanel::ShowGSNotification(const FString& Message, bool bIsError)
{
    FNotificationInfo NotificationInfo(FText::FromString(Message));
    NotificationInfo.bFireAndForget = true;
    NotificationInfo.FadeOutDuration = 3.0f;
    NotificationInfo.ExpireDuration = 5.0f;
    
    if (bIsError)
    {
        NotificationInfo.Image = FCoreStyle::Get().GetBrush(TEXT("MessageLog.Error"));
    }
    else
    {
        NotificationInfo.Image = FAppStyle::GetBrush(TEXT("Icons.Info"));
    }
    
    FSlateNotificationManager::Get().AddNotification(NotificationInfo);
}

void SVCCSimPanel::ExportCamerasToPLY(
    const TArray<FCameraInfo>& CameraInfos, const FString& OutputPath)
{
    // Use the new unified FPLYWriter class
    FPLYWriter::FPLYWriteConfig Config;
    Config.bIncludeColors = true;
    Config.bIncludeNormals = true;
    Config.bBinaryFormat = false;
    
    bool bSuccess = FPLYWriter::WriteCamerasToPLY(CameraInfos, OutputPath, Config);
    
    if (!bSuccess)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to export cameras to PLY file: %s"), *OutputPath);
    }
}

END_SLATE_FUNCTION_BUILD_OPTIMIZATION