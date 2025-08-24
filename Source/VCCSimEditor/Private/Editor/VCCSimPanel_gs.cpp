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
#include "Widgets/Layout/SBox.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Input/SNumericEntryBox.h"
#include "SlateOptMacros.h"
#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Framework/Notifications/NotificationManager.h"
#include "Widgets/Notifications/SNotificationList.h"

BEGIN_SLATE_FUNCTION_BUILD_OPTIMIZATION

// ============================================================================
// TRIANGLE SPLATTING INITIALIZATION - SIMPLIFIED
// ============================================================================

void SVCCSimPanel::InitializeGSManager()
{
    GSTrainingManager = MakeShared<FTriangleSplattingManager>();
    GSTrainingStatusMessage = TEXT("Ready");
}

// ============================================================================
// TRIANGLE SPLATTING UI CREATION - SIMPLIFIED
// ============================================================================

TSharedRef<SWidget> SVCCSimPanel::CreateTriangleSplattingWidget()
{
    return SNew(SVerticalBox)
        
        // Data input section
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 5)
        [
            CreateGSDataInputSection()
        ]
        
        // Camera parameters section
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 5)
        [
            CreateGSCameraParamsSection()
        ]
        
        // Training parameters section
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 5)
        [
            CreateGSTrainingParamsSection()
        ]
        
        // Training control section
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(0, 5)
        [
            CreateGSTrainingControlSection()
        ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateGSDataInputSection()
{
    return SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("ToolPanel.GroupBorder"))
        [
            SNew(SVerticalBox)
            
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5.0f)
            [
                SNew(STextBlock)
                .Text(FText::FromString(TEXT("Data Input - Structured Configuration")))
                .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
            ]
            
            // Simple status display
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5, 2)
            [
                SNew(STextBlock)
                .Text_Lambda([this]() { 
                    return FText::FromString(FString::Printf(TEXT("Image Dir: %s"), 
                        GSConfig.ImageDirectory.IsEmpty() ? TEXT("Not Set") : *FPaths::GetCleanFilename(GSConfig.ImageDirectory)));
                })
            ]
            
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5, 2)
            [
                SNew(STextBlock)
                .Text_Lambda([this]() { 
                    return FText::FromString(FString::Printf(TEXT("Pose File: %s"), 
                        GSConfig.PoseFilePath.IsEmpty() ? TEXT("Not Set") : *FPaths::GetCleanFilename(GSConfig.PoseFilePath)));
                })
            ]
            
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5, 2)
            [
                SNew(STextBlock)
                .Text_Lambda([this]() { 
                    return FText::FromString(FString::Printf(TEXT("Output Dir: %s"), 
                        GSConfig.OutputDirectory.IsEmpty() ? TEXT("Not Set") : *FPaths::GetCleanFilename(GSConfig.OutputDirectory)));
                })
            ]
            
            // Browse buttons
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5, 2)
            [
                SNew(SHorizontalBox)
                
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                .Padding(0, 0, 2, 0)
                [
                    SNew(SButton)
                    .Text(FText::FromString(TEXT("Browse Images")))
                    .OnClicked(this, &SVCCSimPanel::OnGSBrowseImageDirectoryClicked)
                ]
                
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                .Padding(2, 0, 0, 0)
                [
                    SNew(SButton)
                    .Text(FText::FromString(TEXT("Browse Output")))
                    .OnClicked(this, &SVCCSimPanel::OnGSBrowseOutputDirectoryClicked)
                ]
            ]
        ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateGSCameraParamsSection()
{
    return SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("ToolPanel.GroupBorder"))
        [
            SNew(SVerticalBox)
            
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5.0f)
            [
                SNew(STextBlock)
                .Text(FText::FromString(TEXT("Camera Parameters")))
                .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
            ]
            
            // FOV input
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5, 2)
            [
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot()
                .AutoWidth()
                .VAlign(VAlign_Center)
                [
                    SNew(STextBlock)
                    .Text(FText::FromString(TEXT("FOV:")))
                    .MinDesiredWidth(80.0f)
                ]
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                [
                    SAssignNew(GSFOVSpinBox, SNumericEntryBox<float>)
                    .Value_Lambda([this]() { return GSFOVValue; })
                    .OnValueChanged_Lambda([this](float NewValue) 
                    { 
                        GSConfig.FOVDegrees = NewValue;
                        GSFOVValue = NewValue;
                        OnTriangleSplattingConfigurationChanged();
                    })
                    .MinValue(1.0f)
                    .MaxValue(180.0f)
                    .Delta(1.0f)
                ]
            ]
            
            // Image width input
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5, 2)
            [
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot()
                .AutoWidth()
                .VAlign(VAlign_Center)
                [
                    SNew(STextBlock)
                    .Text(FText::FromString(TEXT("Width:")))
                    .MinDesiredWidth(80.0f)
                ]
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                [
                    SAssignNew(GSImageWidthSpinBox, SNumericEntryBox<int32>)
                    .Value_Lambda([this]() { return GSImageWidthValue; })
                    .OnValueChanged_Lambda([this](int32 NewValue) 
                    { 
                        GSConfig.ImageWidth = NewValue;
                        GSImageWidthValue = NewValue;
                        OnTriangleSplattingConfigurationChanged();
                    })
                    .MinValue(64)
                    .MaxValue(4096)
                    .Delta(64)
                ]
            ]
            
            // Image height input
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5, 2)
            [
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot()
                .AutoWidth()
                .VAlign(VAlign_Center)
                [
                    SNew(STextBlock)
                    .Text(FText::FromString(TEXT("Height:")))
                    .MinDesiredWidth(80.0f)
                ]
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                [
                    SAssignNew(GSImageHeightSpinBox, SNumericEntryBox<int32>)
                    .Value_Lambda([this]() { return GSImageHeightValue; })
                    .OnValueChanged_Lambda([this](int32 NewValue) 
                    { 
                        GSConfig.ImageHeight = NewValue;
                        GSImageHeightValue = NewValue;
                        OnTriangleSplattingConfigurationChanged();
                    })
                    .MinValue(64)
                    .MaxValue(4096)
                    .Delta(64)
                ]
            ]
        ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateGSTrainingParamsSection()
{
    return SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("ToolPanel.GroupBorder"))
        [
            SNew(SVerticalBox)
            
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5.0f)
            [
                SNew(STextBlock)
                .Text(FText::FromString(TEXT("Training Parameters")))
                .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
            ]
            
            // Max iterations input
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5, 2)
            [
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot()
                .AutoWidth()
                .VAlign(VAlign_Center)
                [
                    SNew(STextBlock)
                    .Text(FText::FromString(TEXT("Max Iterations:")))
                    .MinDesiredWidth(80.0f)
                ]
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                [
                    SAssignNew(GSMaxIterationsSpinBox, SNumericEntryBox<int32>)
                    .Value_Lambda([this]() { return GSMaxIterationsValue; })
                    .OnValueChanged_Lambda([this](int32 NewValue) 
                    { 
                        GSConfig.MaxIterations = NewValue;
                        GSMaxIterationsValue = NewValue;
                        OnTriangleSplattingConfigurationChanged();
                    })
                    .MinValue(100)
                    .MaxValue(100000)
                    .Delta(500)
                ]
            ]
            
            // Learning rate input
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5, 2)
            [
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot()
                .AutoWidth()
                .VAlign(VAlign_Center)
                [
                    SNew(STextBlock)
                    .Text(FText::FromString(TEXT("Learning Rate:")))
                    .MinDesiredWidth(80.0f)
                ]
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                [
                    SAssignNew(GSLearningRateSpinBox, SNumericEntryBox<float>)
                    .Value_Lambda([this]() { return GSLearningRateValue; })
                    .OnValueChanged_Lambda([this](float NewValue) 
                    { 
                        GSConfig.LearningRate = NewValue;
                        GSLearningRateValue = NewValue;
                        OnTriangleSplattingConfigurationChanged();
                    })
                    .MinValue(0.0001f)
                    .MaxValue(1.0f)
                    .Delta(0.0001f)
                ]
            ]
        ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateGSTrainingControlSection()
{
    return SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("ToolPanel.GroupBorder"))
        [
            SNew(SVerticalBox)
            
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5.0f)
            [
                SNew(STextBlock)
                .Text(FText::FromString(TEXT("Training Control")))
                .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
            ]
            
            // Training status
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5, 2)
            [
                SAssignNew(GSTrainingStatusText, STextBlock)
                .Text_Lambda([this]() { 
                    if (bGSTrainingInProgress)
                    {
                        return FText::FromString(FString::Printf(TEXT("%s (%.1f%%)"), 
                            *GSTrainingStatusMessage, GSTrainingProgress * 100.0f));
                    }
                    return FText::FromString(GSTrainingStatusMessage);
                })
            ]
            
            // Control buttons
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5, 2)
            [
                SNew(SHorizontalBox)
                
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                .Padding(0, 0, 2, 0)
                [
                    SAssignNew(GSStartTrainingButton, SButton)
                    .Text(FText::FromString(TEXT("Start Training")))
                    .IsEnabled_Lambda([this]() { return !bGSTrainingInProgress; })
                    .OnClicked(this, &SVCCSimPanel::OnGSStartTrainingClicked)
                ]
                
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                .Padding(2, 0, 0, 0)
                [
                    SAssignNew(GSStopTrainingButton, SButton)
                    .Text(FText::FromString(TEXT("Stop Training")))
                    .IsEnabled_Lambda([this]() { return bGSTrainingInProgress; })
                    .OnClicked(this, &SVCCSimPanel::OnGSStopTrainingClicked)
                ]
            ]
        ];
}

// ============================================================================
// EVENT HANDLERS - SIMPLIFIED
// ============================================================================

FReply SVCCSimPanel::OnGSBrowseImageDirectoryClicked()
{
    FString SelectedFolder;
    if (FDesktopPlatformModule::Get()->OpenDirectoryDialog(
        nullptr,
        TEXT("Select Image Directory"),
        GSConfig.ImageDirectory,
        SelectedFolder))
    {
        GSConfig.ImageDirectory = SelectedFolder;
        OnTriangleSplattingConfigurationChanged();
    }
    return FReply::Handled();
}

FReply SVCCSimPanel::OnGSBrowseOutputDirectoryClicked()
{
    FString SelectedFolder;
    if (FDesktopPlatformModule::Get()->OpenDirectoryDialog(
        nullptr,
        TEXT("Select Output Directory"),
        GSConfig.OutputDirectory,
        SelectedFolder))
    {
        GSConfig.OutputDirectory = SelectedFolder;
        OnTriangleSplattingConfigurationChanged();
    }
    return FReply::Handled();
}

// Simplified parameter change handlers
void SVCCSimPanel::OnGSFOVChanged(float NewValue)
{
    GSConfig.FOVDegrees = NewValue;
    OnTriangleSplattingConfigurationChanged();
}

void SVCCSimPanel::OnGSImageWidthChanged(int32 NewValue)
{
    GSConfig.ImageWidth = NewValue;
    OnTriangleSplattingConfigurationChanged();
}

void SVCCSimPanel::OnGSImageHeightChanged(int32 NewValue)
{
    GSConfig.ImageHeight = NewValue;
    OnTriangleSplattingConfigurationChanged();
}

void SVCCSimPanel::OnGSMaxIterationsChanged(int32 NewValue)
{
    GSConfig.MaxIterations = NewValue;
    OnTriangleSplattingConfigurationChanged();
}

void SVCCSimPanel::OnGSLearningRateChanged(float NewValue)
{
    GSConfig.LearningRate = NewValue;
    OnTriangleSplattingConfigurationChanged();
}

// ============================================================================
// TRAINING CONTROL - SIMPLIFIED
// ============================================================================

FReply SVCCSimPanel::OnGSStartTrainingClicked()
{
    if (!ValidateGSConfiguration())
    {
        return FReply::Handled();
    }
    
    bGSTrainingInProgress = true;
    GSTrainingProgress = 0.0f;
    GSTrainingStatusMessage = TEXT("Training Started (Structured Config)");
    
    ShowGSNotification(TEXT("Triangle Splatting training started with structured configuration"));
    
    return FReply::Handled();
}

FReply SVCCSimPanel::OnGSStopTrainingClicked()
{
    bGSTrainingInProgress = false;
    GSTrainingStatusMessage = TEXT("Training Stopped");
    ShowGSNotification(TEXT("Triangle Splatting training stopped"));
    
    return FReply::Handled();
}

FReply SVCCSimPanel::OnGSTestTransformationClicked()
{
    if (!ValidateGSConfiguration())
    {
        return FReply::Handled();
    }
    
    ShowGSNotification(TEXT("Running transformation test with structured configuration"));
    
    return FReply::Handled();
}

FReply SVCCSimPanel::OnGSExportColmapClicked()
{
    if (GSConfig.PoseFilePath.IsEmpty() || GSConfig.OutputDirectory.IsEmpty())
    {
        ShowGSNotification(TEXT("Please set pose file and output directory"), true);
        return FReply::Handled();
    }
    
    // Simplified export
    FString ColmapPath = GSConfig.OutputDirectory / TEXT("colmap_export.txt");
    FString ExportContent = FString::Printf(TEXT("# COLMAP Export\\n# Pose File: %s\\n# Config: %s\\n"), 
        *GSConfig.PoseFilePath, TEXT("Structured"));
    FFileHelper::SaveStringToFile(ExportContent, *ColmapPath);
    ShowGSNotification(FString::Printf(TEXT("Exported to: %s"), *ColmapPath));
    
    return FReply::Handled();
}

// ============================================================================
// VALIDATION AND UTILITY - SIMPLIFIED
// ============================================================================

bool SVCCSimPanel::ValidateGSConfiguration()
{
    if (GSConfig.ImageDirectory.IsEmpty())
    {
        ShowGSNotification(TEXT("Please set image directory"), true);
        return false;
    }
    
    if (GSConfig.OutputDirectory.IsEmpty())
    {
        ShowGSNotification(TEXT("Please set output directory"), true);
        return false;
    }
    
    if (!FPaths::DirectoryExists(GSConfig.ImageDirectory))
    {
        ShowGSNotification(TEXT("Image directory does not exist"), true);
        return false;
    }
    
    return true;
}

void SVCCSimPanel::ShowGSNotification(const FString& Message, bool bIsError)
{
    FNotificationInfo Info(FText::FromString(Message));
    Info.bFireAndForget = true;
    Info.FadeOutDuration = 3.0f;
    Info.ExpireDuration = 5.0f;
    
    if (bIsError)
    {
        Info.Image = FAppStyle::GetBrush(TEXT("MessageLog.Error"));
    }
    else
    {
        Info.Image = FAppStyle::GetBrush(TEXT("MessageLog.Note"));
    }
    
    FSlateNotificationManager::Get().AddNotification(Info);
}

// Simplified placeholder for template function
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
    return SNew(STextBlock).Text(FText::FromString(TEXT("Numeric Property Placeholder")));
}

// Simple placeholder for missing functions
FReply SVCCSimPanel::OnGSBrowsePoseFileClicked()
{
    ShowGSNotification(TEXT("Pose file browser - coming soon"));
    return FReply::Handled();
}

void SVCCSimPanel::ExportCamerasToPLY(const TArray<struct FCameraInfo>& CameraInfos, const FString& OutputPath)
{
    // Simple placeholder
    FFileHelper::SaveStringToFile(TEXT("# PLY Export Placeholder"), *OutputPath);
}

END_SLATE_FUNCTION_BUILD_OPTIMIZATION