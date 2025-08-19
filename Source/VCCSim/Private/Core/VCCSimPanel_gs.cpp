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

#if WITH_EDITOR

#include "Core/VCCSimPanel.h"
#include "Utils/TriangleSplattingManager.h"
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
    GSMaxIterationsValue = GSConfig.MaxIterations;
    GSLearningRateValue = GSConfig.LearningRate;
    
    // Create training manager
    GSTrainingManager = MakeShared<FTriangleSplattingManager>();
    
    // Bind delegates
    GSTrainingManager->OnTrainingProgressUpdated.BindLambda([this](float Progress, FString StatusMessage)
    {
        GSTrainingProgress = Progress;
        GSTrainingStatusMessage = StatusMessage;
    });
    
    GSTrainingManager->OnTrainingCompleted.BindLambda([this](bool bSuccessful, FString ResultMessage)
    {
        bGSTrainingInProgress = false;
        GSTrainingProgress = bSuccessful ? 1.0f : 0.0f;
        GSTrainingStatusMessage = ResultMessage;
        
        // Stop status update timer
        if (GEditor && GSStatusUpdateTimerHandle.IsValid())
        {
            GEditor->GetTimerManager()->ClearTimer(GSStatusUpdateTimerHandle);
            GSStatusUpdateTimerHandle.Invalidate();
        }
        
        ShowGSNotification(ResultMessage, !bSuccessful);
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

        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(2, 2)
        [
            CreateGSNumericPropertyRow<float>(
                TEXT("FOV (°)"),
                GSFOVSpinBox,
                GSFOVValue,
                1.0f, 179.0f, 1.0f,
                [this](float NewValue) { OnGSFOVChanged(NewValue); }
            )
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
        
        // Max Iterations and Learning Rate in one row
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            SNew(SHorizontalBox)
            
            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            .Padding(0, 0, 5, 0)
            [
                CreateGSNumericPropertyRow<int32>(
                    TEXT("Max Iterations"),
                    GSMaxIterationsSpinBox,
                    GSMaxIterationsValue,
                    100, 50000, 100,
                    [this](int32 NewValue) { OnGSMaxIterationsChanged(NewValue); }
                )
            ]
            
            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            [
                CreateGSNumericPropertyRow<float>(
                    TEXT("Learning Rate"),
                    GSLearningRateSpinBox,
                    GSLearningRateValue,
                    0.0001f, 1.0f, 0.001f,
                    [this](float NewValue) { OnGSLearningRateChanged(NewValue); }
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
        
        // Control buttons
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 2)
        [
            SNew(SHorizontalBox)
            
            + SHorizontalBox::Slot()
            .AutoWidth()
            .Padding(0, 0, 5, 0)
            [
                SAssignNew(GSStartTrainingButton, SButton)
                .Text(FText::FromString(TEXT("Start Training")))
                .IsEnabled_Lambda([this]() { return !bGSTrainingInProgress; })
                .OnClicked(this, &SVCCSimPanel::OnGSStartTrainingClicked)
            ]
            
            + SHorizontalBox::Slot()
            .AutoWidth()
            [
                SAssignNew(GSStopTrainingButton, SButton)
                .Text(FText::FromString(TEXT("Stop Training")))
                .IsEnabled_Lambda([this]() { return bGSTrainingInProgress; })
                .OnClicked(this, &SVCCSimPanel::OnGSStopTrainingClicked)
            ]
        ]
        
        // Status text
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 2)
        [
            CreatePropertyRow(TEXT("Status"),
                SAssignNew(GSTrainingStatusText, STextBlock)
                .Text_Lambda([this]() { return FText::FromString(GSTrainingStatusMessage); })
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

FReply SVCCSimPanel::OnGSBrowseImageDirectoryClicked()
{
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (DesktopPlatform)
    {
        FString SelectedDirectory;
        const bool bFolderSelected = DesktopPlatform->OpenDirectoryDialog(
            nullptr,
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
            nullptr,
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
            nullptr,
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

void SVCCSimPanel::OnGSMaxIterationsChanged(int32 NewValue)
{
    GSConfig.MaxIterations = NewValue;
}

void SVCCSimPanel::OnGSLearningRateChanged(float NewValue)
{
    GSConfig.LearningRate = NewValue;
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
            GSTrainingStatusMessage = TEXT("Training started...");
            
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
    GSTrainingProgress = 0.0f;
    GSTrainingStatusMessage = TEXT("Training stopped by user");
    
    // Stop status update timer
    if (GEditor && GSStatusUpdateTimerHandle.IsValid())
    {
        GEditor->GetTimerManager()->ClearTimer(GSStatusUpdateTimerHandle);
        GSStatusUpdateTimerHandle.Invalidate();
    }
    
    ShowGSNotification(TEXT("Training stopped"));
    
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
    
    if (GSConfig.LearningRate <= 0)
    {
        ErrorMessages.Add(TEXT("Learning rate must be positive"));
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
        NotificationInfo.Image = FCoreStyle::Get().GetBrush(TEXT("MessageLog.Info"));
    }
    
    FSlateNotificationManager::Get().AddNotification(NotificationInfo);
}

END_SLATE_FUNCTION_BUILD_OPTIMIZATION

#endif // WITH_EDITOR