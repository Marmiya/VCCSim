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

#include "Core/VCCSimPanel.h"
#include "PropertyEditorModule.h"
#include "Engine/Selection.h"
#include "EngineUtils.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SComboBox.h"
#include "Widgets/Input/SNumericEntryBox.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Text/STextBlock.h"
#include "Pawns/FlashPawn.h"
#include "Misc/DateTime.h"
#include "HAL/FileManager.h"
#include "Misc/Paths.h"
#include "DataType/DataMesh.h"
#include "Sensors/CameraSensor.h"
#include "Sensors/DepthCamera.h"
#include "Sensors/SegmentCamera.h"
#include "Simulation/PathPlanner.h"
#include "Simulation/SceneAnalysisManager.h"
#include "Utils/ImageProcesser.h"
#include "Utils/TrajectoryViewer.h"
#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#include "Misc/FileHelper.h"

// Destructor
SVCCSimPanel::~SVCCSimPanel()
{
    // Unregister from selection events
    if (GEditor && GEditor->GetSelectedActors())
    {
        GEditor->GetSelectedActors()->SelectionChangedEvent.RemoveAll(this);
    }
    
    // Clear timer if active
    if (GEditor && bAutoCaptureInProgress)
    {
        GEditor->GetTimerManager()->ClearTimer(AutoCaptureTimerHandle);
        bAutoCaptureInProgress = false;
    }

    if (PathVisualizationActor.IsValid() && GEditor)
    {
        if (UWorld* World = GEditor->GetEditorWorldContext().World())
        {
            World->DestroyActor(PathVisualizationActor.Get());
        }
        PathVisualizationActor.Reset();
    }

    if (SceneAnalysisManager.IsValid())
    {
        SceneAnalysisManager->ClearSafeZoneVisualization();
        SceneAnalysisManager->ClearCoverageVisualization();
    }
}

// Construct method
void SVCCSimPanel::Construct(const FArguments& InArgs)
{
    NumPosesValue = NumPoses;
    RadiusValue = Radius;
    HeightOffsetValue = HeightOffset;
    VerticalGapValue = VerticalGap;
    SafeDistanceValue = SafeDistance;
    SafeHeightValue = SafeHeight;
    JobNum = MakeShared<std::atomic<int32>>(0);
    
    // Register for selection change events
    USelection* Selection = GEditor->GetSelectedActors();
    if (Selection)
    {
        Selection->SelectionChangedEvent.AddSP(
            SharedThis(this), 
            &SVCCSimPanel::OnSelectionChanged
        );
    }
    
    // Load logo images
    FString PluginDir = FPaths::Combine(FPaths::ProjectPluginsDir(), TEXT("VCCSim"));
    FString VCCLogoPath = FPaths::Combine(PluginDir, TEXT("image/Logo/vcc.png"));
    FString SZULogoPath = FPaths::Combine(PluginDir, TEXT("image/Logo/szu.png"));
    
    // Create dynamic brushes if files exist
    if (FPaths::FileExists(VCCLogoPath))
    {
        VCCLogoBrush = MakeShareable(new FSlateDynamicImageBrush(
            FName(*VCCLogoPath), 
            FVector2D(65, 65),  // Maintain square ratio but smaller for UI
            FColor(255, 255, 255, 255)));
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("VCC logo file not found at: %s"), *VCCLogoPath);
    }

    if (FPaths::FileExists(SZULogoPath))
    {
        // Calculate width to maintain aspect ratio with same height as VCC logo
        float SZUWidth = 80 * (272.0f / 80.0f);  // 85 * 3.4 = ~289
        SZULogoBrush = MakeShareable(new FSlateDynamicImageBrush(
            FName(*SZULogoPath), 
            FVector2D(SZUWidth, 80),  // Same height as VCC logo, width preserves ratio
            FColor(255, 255, 255, 255)));
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("SZU logo file not found at: %s"), *SZULogoPath);
    }

    // Search for the SceneAnalysisManager in the world
    if (UWorld* World = GEditor->GetEditorWorldContext().World())
    {
        for (TActorIterator<ASceneAnalysisManager> It(World); It; ++It)
        {
            SceneAnalysisManager = *It;
            if (SceneAnalysisManager.IsValid())
            {
                SceneAnalysisManager->Initialize(World,
                    FPaths::ProjectSavedDir() / TEXT("VCCSimCaptures"));
                SceneAnalysisManager->InitializeSafeZoneVisualization();
                SceneAnalysisManager->InitializeCoverageVisualization();
                break;
            }
            break;
        }
    }
    
    // Create the widget layout
     ChildSlot
    [
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryTop"))
        .Padding(0)
        [
            SNew(SVerticalBox)
            
            // Logo panel with consistent styling
            +SVerticalBox::Slot()
            .AutoHeight()
            .Padding(0)
            [
                CreateSectionContent(
                    SNew(SHorizontalBox)
                    
                    // VCC Logo (left)
                    +SHorizontalBox::Slot()
                    .AutoWidth()
                    .Padding(8, 0, 0, 0)
                    .HAlign(HAlign_Left)
                    .VAlign(VAlign_Center)
                    [
                        SNew(SImage)
                        .Image_Lambda([this]() {
                            return VCCLogoBrush.IsValid() ? VCCLogoBrush.Get() :
                            FAppStyle::GetBrush("NoBrush");
                        })
                    ]
                    
                    // Spacer to push logos to the edges
                    +SHorizontalBox::Slot()
                    .FillWidth(1.0f)
                    [
                        SNew(SSpacer)
                    ]
                    
                    // SZU Logo (right)
                    +SHorizontalBox::Slot()
                    .AutoWidth()
                    .HAlign(HAlign_Right)
                    .VAlign(VAlign_Center)
                    [
                        SNew(SImage)
                        .Image_Lambda([this]() {
                            return SZULogoBrush.IsValid() ? SZULogoBrush.Get() :
                            FAppStyle::GetBrush("NoBrush");
                        })
                    ]
                )
            ]
            
            // Flash Pawn section
            +SVerticalBox::Slot()
            .AutoHeight()
            [
                CreatePawnSelectPanel()
            ]
            
            // Camera section
            +SVerticalBox::Slot()
            .AutoHeight()
            [
                CreateCameraSelectPanel()
            ]
            
            // Target section
            +SVerticalBox::Slot()
            .AutoHeight()
            [
                CreateTargetSelectPanel()
            ]
            
            // Pose configuration
            +SVerticalBox::Slot()
            .AutoHeight()
            [
                CreatePoseConfigPanel()
            ]
            
            // Capture panel
            +SVerticalBox::Slot()
            .AutoHeight()
            [
                CreateCapturePanel()
            ]

            // Scene analysis panel
            +SVerticalBox::Slot()
            .AutoHeight()
            [
                CreateSceneAnalysisPanel()
            ]
        ]
    ];
}

// Selection panel for FlashPawn
TSharedRef<SWidget> SVCCSimPanel::CreatePawnSelectPanel()
{
    return SNew(SVerticalBox)
    +SVerticalBox::Slot()
    .AutoHeight()
    [
        CreateSectionHeader("Flash Pawn")
    ]
    +SVerticalBox::Slot()
    .AutoHeight()
    [
        CreateSectionContent(
            SNew(SVerticalBox)
            +SVerticalBox::Slot()
            .AutoHeight()
            .Padding(FMargin(0, 4, 0, 4))
            [
                CreatePropertyRow(
                    "Current",
                    SNew(SHorizontalBox)
                    +SHorizontalBox::Slot()
                    .FillWidth(1.0f)
                    [
                        SNew(SBorder)
                        .Padding(4)
                        [
                            SAssignNew(SelectedFlashPawnText, STextBlock)
                            .Text(FText::FromString("None selected"))
                        ]
                    ]
                    +SHorizontalBox::Slot()
                    .AutoWidth()
                    .VAlign(VAlign_Center)
                    .Padding(FMargin(8, 0, 4, 0))
                    [
                        SAssignNew(SelectFlashPawnToggle, SCheckBox)
                        .IsChecked(bSelectingFlashPawn ? ECheckBoxState::Checked : ECheckBoxState::Unchecked)
                        .OnCheckStateChanged(this, &SVCCSimPanel::OnSelectFlashPawnToggleChanged)
                    ]
                    +SHorizontalBox::Slot()
                    .AutoWidth()
                    .VAlign(VAlign_Center)
                    [
                        SNew(STextBlock)
                        .Text(FText::FromString("Click to select"))
                    ]
                )
            ]
        )
    ];
}

// Selection panel for Target Object
TSharedRef<SWidget> SVCCSimPanel::CreateTargetSelectPanel()
{
    return SNew(SVerticalBox)
    +SVerticalBox::Slot()
    .AutoHeight()
    [
        CreateSectionHeader("Select Target Object")
    ]
    +SVerticalBox::Slot()
    .AutoHeight()
    [
        CreateSectionContent(
            SNew(SVerticalBox)
            +SVerticalBox::Slot()
            .AutoHeight()
            .Padding(FMargin(0, 4, 0, 4))
            [
                CreatePropertyRow(
                    "Current",
                    SNew(SHorizontalBox)
                    +SHorizontalBox::Slot()
                    .FillWidth(1.0f)
                    [
                        SNew(SBorder)
                        .Padding(4)
                        [
                            SAssignNew(SelectedTargetObjectText, STextBlock)
                            .Text(FText::FromString("None selected"))
                        ]
                    ]
                    +SHorizontalBox::Slot()
                    .AutoWidth()
                    .VAlign(VAlign_Center)
                    .Padding(FMargin(8, 0, 4, 0))
                    [
                        SAssignNew(SelectTargetToggle, SCheckBox)
                        .IsChecked(bSelectingTarget ? ECheckBoxState::Checked : ECheckBoxState::Unchecked)
                        .OnCheckStateChanged(this, &SVCCSimPanel::OnSelectTargetToggleChanged)
                    ]
                    +SHorizontalBox::Slot()
                    .AutoWidth()
                    .VAlign(VAlign_Center)
                    [
                        SNew(STextBlock)
                        .Text(FText::FromString("Click to select"))
                    ]
                )
            ]
        )
    ];
}

// Camera selection panel
TSharedRef<SWidget> SVCCSimPanel::CreateCameraSelectPanel()
{
    return SNew(SVerticalBox)
    +SVerticalBox::Slot()
    .AutoHeight()
    [
        CreateSectionHeader("Camera Selection")
    ]
    +SVerticalBox::Slot()
    .AutoHeight()
    [
        CreateSectionContent(
            SNew(SVerticalBox)
            // Camera Availability Row
            +SVerticalBox::Slot()
            .AutoHeight()
            .Padding(FMargin(0, 0, 0, 8))
            [
                SNew(SVerticalBox)
                +SVerticalBox::Slot()
                .AutoHeight()
                .Padding(FMargin(0, 4, 0, 4))
                [
                    SNew(STextBlock)
                    .Text(FText::FromString("Available & Active Cameras:"))
                    .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
                ]
                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    SNew(SHorizontalBox)
                    
                    // RGB Camera availability
                    +SHorizontalBox::Slot()
                    .MaxWidth(120)
                    .HAlign(HAlign_Center)
                    .Padding(FMargin(0, 0, 2, 0))
                    [
                        SNew(SBorder)
                        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                        .BorderBackgroundColor(FColor(5,5, 5, 255))
                        .Padding(2, 0)
                        .HAlign(HAlign_Center)
                        [
                            SNew(SHorizontalBox)
                            +SHorizontalBox::Slot()
                            .AutoWidth()
                            .HAlign(HAlign_Left)
                            .VAlign(VAlign_Center)
                            .Padding(FMargin(0, 0, 4, 0))
                            [
                                SNew(SImage)
                                .Image_Lambda([this]() {
                                    return bHasRGBCamera ? 
                                        FAppStyle::GetBrush("Icons.Checkmark") : 
                                        FAppStyle::GetBrush("Icons.X");
                                })
                                .ColorAndOpacity_Lambda([this]() {
                                    return bHasRGBCamera ? 
                                        FColor(10, 200, 10) : 
                                        FColor(200, 10, 10);
                                })
                            ]
                            +SHorizontalBox::Slot()
                            .AutoWidth()
                            .HAlign(HAlign_Center)
                            .VAlign(VAlign_Center)
                            .Padding(FMargin(0, 0, 2, 0))
                            [
                                SAssignNew(RGBCameraCheckBox, SCheckBox)
                                .IsChecked(bUseRGBCamera ? ECheckBoxState::Checked : ECheckBoxState::Unchecked)
                                .OnCheckStateChanged(this, &SVCCSimPanel::OnRGBCameraCheckboxChanged)
                                .IsEnabled_Lambda([this]() {
                                    return bHasRGBCamera;
                                })
                            ]
                            +SHorizontalBox::Slot()
                            .AutoWidth()
                            .HAlign(HAlign_Right)
                            .VAlign(VAlign_Center)
                            .Padding(0, 0, 4, 0)
                            [
                                SNew(STextBlock)
                                .Text(FText::FromString("RGB"))
                            ]
                        ]
                    ]
                    
                    // Depth Camera availability
                    +SHorizontalBox::Slot()
                    .MaxWidth(120)
                    .HAlign(HAlign_Center)
                    .Padding(FMargin(0, 0, 2, 0))
                    [
                        SNew(SBorder)
                        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                        .BorderBackgroundColor(FColor(5,5, 5, 255))
                        .Padding(2, 0)
                        .HAlign(HAlign_Center)
                        [
                            SNew(SHorizontalBox)
                            +SHorizontalBox::Slot()
                            .AutoWidth()
                            .HAlign(HAlign_Left)
                            .VAlign(VAlign_Center)
                            .Padding(FMargin(0, 0, 4, 0))
                            [
                                SNew(SImage)
                                .Image_Lambda([this]() {
                                    return bHasDepthCamera ? 
                                        FAppStyle::GetBrush("Icons.Checkmark") : 
                                        FAppStyle::GetBrush("Icons.X");
                                })
                                .ColorAndOpacity_Lambda([this]() {
                                    return bHasDepthCamera ? 
                                        FColor(20, 200, 20) : 
                                        FColor(200, 20, 20);
                                })
                            ]
                            +SHorizontalBox::Slot()
                            .AutoWidth()
                            .HAlign(HAlign_Center)
                            .VAlign(VAlign_Center)
                            .Padding(FMargin(0, 0, 2, 0))
                            [
                                SAssignNew(DepthCameraCheckBox, SCheckBox)
                                .IsChecked(bUseDepthCamera ? ECheckBoxState::Checked : ECheckBoxState::Unchecked)
                                .OnCheckStateChanged(this, &SVCCSimPanel::OnDepthCameraCheckboxChanged)
                                .IsEnabled_Lambda([this]() {
                                    return bHasDepthCamera;
                                })
                            ]
                            +SHorizontalBox::Slot()
                            .AutoWidth()
                            .Padding(0, 0, 4, 0)
                            .HAlign(HAlign_Right)
                            .VAlign(VAlign_Center)
                            [
                                SNew(STextBlock)
                                .Text(FText::FromString("Depth"))
                            ]
                        ]
                    ]
                    
                    // Segmentation Camera availability
                    +SHorizontalBox::Slot()
                    .MaxWidth(140)
                    .HAlign(HAlign_Center)
                    .Padding(FMargin(0, 0, 2, 0))
                    [
                        SNew(SBorder)
                        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                        .BorderBackgroundColor(FColor(5,5, 5, 255))
                        .Padding(2, 0)
                        .HAlign(HAlign_Center)
                        [
                            SNew(SHorizontalBox)
                            +SHorizontalBox::Slot()
                            .AutoWidth()
                            .HAlign(HAlign_Left)
                            .VAlign(VAlign_Center)
                            .Padding(FMargin(0, 0, 4, 0))
                            [
                                SNew(SImage)
                                .Image_Lambda([this]() {
                                    return bHasSegmentationCamera ? 
                                        FAppStyle::GetBrush("Icons.Checkmark") : 
                                        FAppStyle::GetBrush("Icons.X");
                                })
                                .ColorAndOpacity_Lambda([this]() {
                                    return bHasSegmentationCamera ? 
                                        FColor(20, 200, 20) : 
                                        FColor(200, 20, 20);
                                })
                            ]
                            +SHorizontalBox::Slot()
                            .AutoWidth()
                            .HAlign(HAlign_Center)
                            .VAlign(VAlign_Center)
                            .Padding(FMargin(0, 0, 2, 0))
                            [
                                SAssignNew(SegmentationCameraCheckBox, SCheckBox)
                                .IsChecked(bUseSegmentationCamera ? ECheckBoxState::Checked : ECheckBoxState::Unchecked)
                                .OnCheckStateChanged(this, &SVCCSimPanel::OnSegmentationCameraCheckboxChanged)
                                .IsEnabled_Lambda([this]() {
                                    return bHasSegmentationCamera;
                                })
                            ]
                            +SHorizontalBox::Slot()
                            .AutoWidth()
                            .Padding(0, 0, 4, 0)
                            .HAlign(HAlign_Right)
                            .VAlign(VAlign_Center)
                            [
                                SNew(STextBlock)
                                .Text(FText::FromString("Segmentation"))
                            ]
                        ]
                    ]
                ]
            ]
            // Update button
            +SVerticalBox::Slot()
            .AutoHeight()
            .HAlign(HAlign_Right)
            [
                SNew(SButton)
                .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
                .Text(FText::FromString("Update Cameras"))
                .ContentPadding(FMargin(6, 2))
                .OnClicked_Lambda([this]() {
                    UpdateActiveCameras();
                    return FReply::Handled();
                })
                .IsEnabled_Lambda([this]() {
                    return SelectedFlashPawn.IsValid() && (bHasRGBCamera || bHasDepthCamera || bHasSegmentationCamera);
                })
            ]
        )
    ];
}

// Pose configuration panel 
TSharedRef<SWidget> SVCCSimPanel::CreatePoseConfigPanel()
{
    return SNew(SVerticalBox)
    +SVerticalBox::Slot()
    .AutoHeight()
    [
        CreateSectionHeader("Pose Configuration")
    ]
    +SVerticalBox::Slot()
    .AutoHeight()
    [
        CreateSectionContent(
            SNew(SVerticalBox)
            
            // Number of poses and Vertical Gap row
            +SVerticalBox::Slot()
            .AutoHeight()
            .Padding(FMargin(0, 0, 0, 4))
            [
                SNew(SHorizontalBox)
                // Pose Count
                +SHorizontalBox::Slot()
                .FillWidth(1.0f)
                .Padding(FMargin(0, 0, 8, 0))
                [
                    CreatePropertyRow(
                        "Pose Count",
                        SNew(SBorder)
                        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                        .BorderBackgroundColor(FColor(5,5, 5, 255))
                        .Padding(4, 0)
                        [
                            SAssignNew(NumPosesSpinBox, SNumericEntryBox<int32>)
                            .Value_Lambda([this]() { return NumPosesValue; })
                            .MinValue(1)
                            .Delta(1)
                            .AllowSpin(true)
                            .OnValueChanged(SNumericEntryBox<int32>::FOnValueChanged::CreateLambda([this](int32 NewValue) {
                                NumPoses = NewValue;
                                NumPosesValue = NewValue;
                            }))
                        ]
                    )
                ]
                
                // Vertical Gap
                +SHorizontalBox::Slot()
                .FillWidth(1.0f)
                .Padding(FMargin(0, 0, 0, 0))
                [
                    CreatePropertyRow(
                        "Vertical Gap",
                        SNew(SBorder)
                        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                        .BorderBackgroundColor(FColor(5,5, 5, 255))
                        .Padding(4, 0)
                        [
                            SAssignNew(VerticalGapSpinBox, SNumericEntryBox<float>)
                            .Value_Lambda([this]() { return VerticalGapValue; })
                            .MinValue(0.0f)
                            .Delta(5.0f)
                            .AllowSpin(true)
                            .OnValueChanged(SNumericEntryBox<float>::FOnValueChanged::CreateLambda([this](float NewValue) {
                                VerticalGap = NewValue;
                                VerticalGapValue = NewValue;
                            }))
                        ]
                    )
                ]
            ]
            
            // Separator
            +SVerticalBox::Slot()
            .MaxHeight(1)
            .Padding(FMargin(0, 0, 0, 0))
            [
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(2, 2, 2))
                .Padding(0)
                .Content()
                [
                    SNew(SBox)
                    .HeightOverride(1.0f)
                ]
            ]
            
            // Radius and Height Offset row
            +SVerticalBox::Slot()
            .AutoHeight()
            .Padding(FMargin(0, 4, 0, 4))
            [
                SNew(SHorizontalBox)
                // Radius
                +SHorizontalBox::Slot()
                .FillWidth(1.0f)
                .Padding(FMargin(0, 0, 8, 0))
                [
                    CreatePropertyRow(
                        "Radius",
                        SNew(SBorder)
                        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                        .BorderBackgroundColor(FColor(5,5, 5, 255))
                        .Padding(4, 0)
                        [
                            SAssignNew(RadiusSpinBox, SNumericEntryBox<float>)
                            .Value_Lambda([this]() { return RadiusValue; })
                            .MinValue(100.0f)
                            .Delta(10.0f)
                            .AllowSpin(true)
                            .OnValueChanged(SNumericEntryBox<float>::FOnValueChanged::CreateLambda([this](float NewValue) {
                                Radius = NewValue;
                                RadiusValue = NewValue;
                            }))
                        ]
                    )
                ]
                
                // Height Offset
                +SHorizontalBox::Slot()
                .FillWidth(1.0f)
                .Padding(FMargin(0, 0, 0, 0))
                [
                    CreatePropertyRow(
                        "Height Offset",
                        SNew(SBorder)
                        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                        .BorderBackgroundColor(FColor(5,5, 5, 255))
                        .Padding(4, 0)
                        [
                            SAssignNew(HeightOffsetSpinBox, SNumericEntryBox<float>)
                            .Value_Lambda([this]() { return HeightOffsetValue; })
                            .MinValue(0.0f)
                            .MaxValue(3000.0f)
                            .Delta(10.0f)
                            .AllowSpin(true)
                            .OnValueChanged(SNumericEntryBox<float>::FOnValueChanged::CreateLambda([this](float NewValue) {
                                HeightOffset = NewValue;
                                HeightOffsetValue = NewValue;
                            }))
                        ]
                    )
                ]
            ]
            
            // Separator
            +SVerticalBox::Slot()
            .MaxHeight(1)
            .Padding(FMargin(0, 0, 0, 0))
            [
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(2, 2, 2))
                .Padding(0)
                .Content()
                [
                    SNew(SBox)
                    .HeightOverride(1.0f)
                ]
            ]
            
            // Load/Save Pose buttons
            +SVerticalBox::Slot()
            .AutoHeight()
            .Padding(FMargin(0, 4, 0, 4))
            [
                SNew(SHorizontalBox)
                +SHorizontalBox::Slot()
                .MaxWidth(180)
                .Padding(FMargin(0, 0, 4, 0))
                .HAlign(HAlign_Fill)
                [
                    SNew(SButton)
                    .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
                    .ContentPadding(FMargin(4, 2))
                    .Text(FText::FromString("Load Predefined Pose"))
                    .HAlign(HAlign_Center)
                    .OnClicked(this, &SVCCSimPanel::OnLoadPoseClicked)
                    .IsEnabled_Lambda([this]() {
                        return SelectedFlashPawn.IsValid();
                    })
                ]
                +SHorizontalBox::Slot()
                .MaxWidth(180)
                .Padding(FMargin(4, 0, 4, 0))
                .HAlign(HAlign_Fill)
                [
                    SNew(SButton)
                    .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
                    .ContentPadding(FMargin(4, 2))
                    .HAlign(HAlign_Center)
                    .Text(FText::FromString("Save Generated Pose"))
                    .OnClicked(this, &SVCCSimPanel::OnSavePoseClicked)
                    .IsEnabled_Lambda([this]() {
                        return SelectedFlashPawn.IsValid() && SelectedFlashPawn->GetPoseCount() > 0;
                    })
                ]
            ]
            
            // Separator
            +SVerticalBox::Slot()
            .MaxHeight(1)
            .Padding(FMargin(0, 0, 0, 0))
            [
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(2, 2, 2))
                .Padding(0)
                .Content()
                [
                    SNew(SBox)
                    .HeightOverride(1.0f)
                ]
            ]
            
            // Action buttons
            +SVerticalBox::Slot()
            .AutoHeight()
            .Padding(FMargin(0, 4, 0, 2))
            [
                SNew(SHorizontalBox)
                +SHorizontalBox::Slot()
                .MaxWidth(120)
                .Padding(FMargin(0, 0, 4, 0))
                .HAlign(HAlign_Fill)
                [
                    SNew(SButton)
                    .ButtonStyle(FAppStyle::Get(), "FlatButton.Success")
                    .ContentPadding(FMargin(5, 2))
                    .Text(FText::FromString("Generate Poses"))
                    .HAlign(HAlign_Center)
                    .OnClicked(this, &SVCCSimPanel::OnGeneratePosesClicked)
                    .IsEnabled_Lambda([this]() {
                        return SelectedFlashPawn.IsValid() && SelectedTargetObject.IsValid();
                    })
                ]
                +SHorizontalBox::Slot()
                .MaxWidth(120)
                .Padding(FMargin(4, 0, 4, 0))
                .HAlign(HAlign_Fill)
                [
                    SAssignNew(VisualizePathButton, SButton)
                    .ButtonStyle(bPathVisualized ? 
                       &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") : 
                       &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"))
                    .ContentPadding(FMargin(5, 2))
                    .HAlign(HAlign_Center)
                    .Text_Lambda([this]() {
                        return FText::FromString(bPathVisualized ? "Hide Path" : "Show Path");
                    })
                    .OnClicked(this, &SVCCSimPanel::OnTogglePathVisualizationClicked)
                    .IsEnabled_Lambda([this]() {
                        return SelectedFlashPawn.IsValid() && !bPathNeedsUpdate;
                    })
                ]
            ]
        )
    ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateCapturePanel()
{
    return SNew(SVerticalBox)
    +SVerticalBox::Slot()
    .AutoHeight()
    [
        CreateSectionHeader("Image Capture")
    ]
        
    +SVerticalBox::Slot()
    .AutoHeight()
    [
        CreateSectionContent(
            SNew(SVerticalBox)
            +SVerticalBox::Slot()
            .AutoHeight()
            .Padding(0, 0, 0, 4)
            [
                SNew(SHorizontalBox)
                +SHorizontalBox::Slot()
                .MaxWidth(100)
                .Padding(FMargin(0, 0, 4, 0))
                .HAlign(HAlign_Fill)
                [
                    SNew(SButton)
                    .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
                    .ContentPadding(FMargin(5, 2))
                    .Text(FText::FromString("Move Back"))
                    .HAlign(HAlign_Center)
                    .OnClicked_Lambda([this]() {
                        if (SelectedFlashPawn.IsValid())
                        {
                            SelectedFlashPawn->MoveBackward();
                        }
                        return FReply::Handled();
                    })
                    .IsEnabled_Lambda([this]() {
                        return SelectedFlashPawn.IsValid();
                    })
                ]
                +SHorizontalBox::Slot()
                .MaxWidth(100)
                .Padding(FMargin(4, 0, 4, 0))
                .HAlign(HAlign_Fill)
                [
                    SNew(SButton)
                    .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
                    .ContentPadding(FMargin(5, 2))
                    .Text(FText::FromString("Move Next"))
                    .HAlign(HAlign_Center)
                    .OnClicked_Lambda([this]() {
                        if (SelectedFlashPawn.IsValid())
                        {
                            SelectedFlashPawn->MoveForward();
                        }
                        return FReply::Handled();
                    })
                    .IsEnabled_Lambda([this]() {
                        return SelectedFlashPawn.IsValid();
                    })
                ]
            ]
            // Separator
           +SVerticalBox::Slot()
           .MaxHeight(1)
           .Padding(FMargin(0, 0, 0, 0))
           [
               SNew(SBorder)
               .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
               .BorderBackgroundColor(FColor(2, 2, 2))
               .Padding(0)
               .Content()
               [
                   SNew(SBox)
                   .HeightOverride(1.0f)
               ]
           ]
            +SVerticalBox::Slot()
            .AutoHeight()
            .Padding(0, 4, 0, 0)
            [
                SNew(SHorizontalBox)
                
                // Single Capture button
                +SHorizontalBox::Slot()
                .MaxWidth(180)
                .Padding(FMargin(0, 0, 4, 0))
                .HAlign(HAlign_Fill)
                [
                    SNew(SButton)
                    .ButtonStyle(FAppStyle::Get(), "FlatButton.Success")
                    .ContentPadding(FMargin(5, 2))
                    .Text(FText::FromString("Capture Current View"))
                    .HAlign(HAlign_Center)
                    .OnClicked(this, &SVCCSimPanel::OnCaptureImagesClicked)
                    .IsEnabled_Lambda([this]() {
                        return SelectedFlashPawn.IsValid() && 
                               (bUseRGBCamera && bHasRGBCamera) || 
                               (bUseDepthCamera && bHasDepthCamera) || 
                               (bUseSegmentationCamera && bHasSegmentationCamera);
                    })
                ]
                
                // Auto Capture button
                +SHorizontalBox::Slot()
                .MaxWidth(180)
                .Padding(FMargin(4, 0, 4, 0))
                .HAlign(HAlign_Fill)
                [
                    SNew(SButton)
                    .ButtonStyle(FAppStyle::Get(), "FlatButton.Primary")
                    .ContentPadding(FMargin(5, 2))
                    .Text(FText::FromString("Auto-Capture All Poses"))
                    .HAlign(HAlign_Center)
                    .OnClicked_Lambda([this]() {
                        StartAutoCapture();
                        return FReply::Handled();
                    })
                    .IsEnabled_Lambda([this]() {
                        return SelectedFlashPawn.IsValid() && 
                               ((bUseRGBCamera && bHasRGBCamera) || 
                                (bUseDepthCamera && bHasDepthCamera) || 
                                (bUseSegmentationCamera && bHasSegmentationCamera));
                    })
                ]
            ]
        )
    ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateSceneAnalysisPanel()
{
    return SNew(SVerticalBox)
    +SVerticalBox::Slot()
    .AutoHeight()
    [
        CreateSectionHeader("Scene Analysis")
    ]
    +SVerticalBox::Slot()
    .AutoHeight()
    [
    CreateSectionContent(
        SNew(SVerticalBox)
        +SVerticalBox::Slot()
        .AutoHeight()
        .Padding(FMargin(0, 0, 0, 4))
        [
            SNew(SHorizontalBox)
            // Pose Count
            +SHorizontalBox::Slot()
            .FillWidth(1.0f)
            .Padding(FMargin(0, 0, 8, 0))
            [
                CreatePropertyRow(
                    "Safe Distance",
                    SNew(SBorder)
                    .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                    .BorderBackgroundColor(FColor(5,5, 5, 255))
                    .Padding(4, 0)
                    [
                        SAssignNew(SafeDistanceSpinBox, SNumericEntryBox<float>)
                        .Value_Lambda([this]() { return SafeDistanceValue; })
                        .MinValue(0.f)
                        .Delta(10.f)
                        .AllowSpin(true)
                        .OnValueChanged(SNumericEntryBox<float>::FOnValueChanged::CreateLambda(
                            [this](float NewValue) {
                            SafeDistance = NewValue;
                            SafeDistanceValue = NewValue;
                        }))
                    ]
                )
            ]
            
            // Vertical Gap
            +SHorizontalBox::Slot()
            .FillWidth(1.0f)
            .Padding(FMargin(0, 0, 0, 0))
            [
                CreatePropertyRow(
                    "Safe Height",
                    SNew(SBorder)
                    .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                    .BorderBackgroundColor(FColor(5,5, 5, 255))
                    .Padding(4, 0)
                    [
                        SAssignNew(SafeHeightSpinBox, SNumericEntryBox<float>)
                        .Value_Lambda([this]() { return SafeHeightValue; })
                        .MinValue(0.0f)
                        .Delta(5.0f)
                        .AllowSpin(true)
                        .OnValueChanged(SNumericEntryBox<float>::FOnValueChanged::CreateLambda(
                            [this](float NewValue) {
                            SafeHeight = NewValue;
                            SafeHeightValue = NewValue;
                        }))
                    ]
                )
            ]
        ]
    
    // Separator
    +SVerticalBox::Slot()
    .MaxHeight(1)
    .Padding(FMargin(0, 0, 0, 0))
    [
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(FColor(2, 2, 2))
        .Padding(0)
        .Content()
        [
            SNew(SBox)
            .HeightOverride(1.0f)
        ]
    ]
        

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(0, 4, 0, 4)
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot()
        .MaxWidth(120)
        .Padding(FMargin(0, 0, 4, 0))
        .HAlign(HAlign_Fill)
        [
            SNew(SButton)
            .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
            .ContentPadding(FMargin(5, 2))
            .Text(FText::FromString("Scan Scene"))
            .HAlign(HAlign_Center)
            .OnClicked_Lambda([this]() {
                if (SceneAnalysisManager.IsValid())
                {
                    SceneAnalysisManager->ScanScene();
                    bNeedScan = false;
                }
                return FReply::Handled();
            })
            .IsEnabled_Lambda([this]() {
                return SceneAnalysisManager.IsValid();
            })
        ]
        +SHorizontalBox::Slot()
        .MaxWidth(120)
        .Padding(FMargin(0, 0, 4, 0))
        .HAlign(HAlign_Fill)
        [
            SNew(SButton)
            .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
            .ContentPadding(FMargin(5, 2))
            .Text(FText::FromString("Gen SafeZone"))
            .HAlign(HAlign_Center)
            .OnClicked_Lambda([this]() {
                if (SceneAnalysisManager.IsValid())
                {
                    SceneAnalysisManager->GenerateSafeZone(SafeDistance, SafeHeight);
                    bGenSafeZone = false;
                }
                return FReply::Handled();
            })
            .IsEnabled_Lambda([this]() {
                return SceneAnalysisManager.IsValid() && !bNeedScan;
            })
        ]
        +SHorizontalBox::Slot()
        .MaxWidth(120)
        .Padding(FMargin(0, 0, 4, 0))
        .HAlign(HAlign_Fill)
        [
            SAssignNew(VisualizeSafeZoneButton, SButton)
            .ButtonStyle(bSafeZoneVisualized ? 
               &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") : 
               &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"))
            .ContentPadding(FMargin(0, 2))
            .HAlign(HAlign_Center)
            .Text_Lambda([this]() {
                return FText::FromString(bSafeZoneVisualized ? "Hide SafeZone" : "Show SafeZone");
            })
            .OnClicked(this, &SVCCSimPanel::OnToggleSafeZoneVisualizationClicked)
            .IsEnabled_Lambda([this]() {
                return SceneAnalysisManager.IsValid() && !bGenSafeZone;
            })
        ]
        +SHorizontalBox::Slot()
        .MaxWidth(120)
        .Padding(FMargin(0, 0, 4, 0))
        .HAlign(HAlign_Fill)
        [
            SNew(SButton)
            .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
            .ContentPadding(FMargin(0, 2))
            .Text(FText::FromString("Clear SafeZone"))
            .HAlign(HAlign_Center)
            .OnClicked_Lambda([this]() {
                SceneAnalysisManager->ClearSafeZoneVisualization();
                bGenSafeZone = true;
                return FReply::Handled();
            })
            .IsEnabled_Lambda([this]() {
                return SceneAnalysisManager.IsValid() && !bGenSafeZone;
            })
        ]
    ]

    // Separator
    +SVerticalBox::Slot()
    .MaxHeight(1)
    .Padding(FMargin(0, 0, 0, 0))
    [
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(FColor(2, 2, 2))
        .Padding(0)
        .Content()
        [
            SNew(SBox)
            .HeightOverride(1.0f)
        ]
    ]
    
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(0, 4, 0, 4)
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot()
        .MaxWidth(120)
        .Padding(FMargin(0, 0, 4, 0))
        .HAlign(HAlign_Fill)
        [
            SNew(SButton)
            .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
            .ContentPadding(FMargin(5, 2))
            .Text(FText::FromString("Init Coverage"))
            .HAlign(HAlign_Center)
            .OnClicked_Lambda([this]() {
                if (SceneAnalysisManager.IsValid())
                {
                    URGBCameraComponent* Camera =
                        SelectedFlashPawn->GetComponentByClass<URGBCameraComponent>();
                    Camera->CameraName = "CoverageCamera";
                    if (Camera)
                    {
                        Camera->ComputeIntrinsics();
                        SceneAnalysisManager->RegisterCamera(Camera);
                    }
                    bInitCoverage = false;
                }
                return FReply::Handled();
            })
            .IsEnabled_Lambda([this]() {
                return SceneAnalysisManager.IsValid() && SelectedFlashPawn.IsValid();
            })
        ]
        +SHorizontalBox::Slot()
        .MaxWidth(120)
        .Padding(FMargin(0, 0, 4, 0))
        .HAlign(HAlign_Fill)
        [
            SNew(SButton)
            .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
            .ContentPadding(FMargin(5, 2))
            .Text(FText::FromString("Gen Coverage"))
            .HAlign(HAlign_Center)
            .OnClicked_Lambda([this]() {
                if (SceneAnalysisManager.IsValid())
                {
                    TArray<FTransform> CoverageTransforms;
                    const auto Positions = SelectedFlashPawn->PendingPositions;
                    const auto Rotations = SelectedFlashPawn->PendingRotations;
                    for (int32 i = 0; i < Positions.Num(); ++i)
                    {
                        FTransform Transform;
                        Transform.SetLocation(Positions[i]);
                        Transform.SetRotation(FQuat(Rotations[i]));
                        CoverageTransforms.Add(Transform);
                    }
                    SceneAnalysisManager->ComputeCoverage(CoverageTransforms, "CoverageCamera");
                    bGenCoverage = false;
                }
                return FReply::Handled();
            })
            .IsEnabled_Lambda([this]() {
                return SceneAnalysisManager.IsValid() && !bInitCoverage;
            })
        ]
        +SHorizontalBox::Slot()
        .MaxWidth(120)
        .Padding(FMargin(0, 0, 4, 0))
        .HAlign(HAlign_Fill)
        [
            SAssignNew(VisualizeCoverageButton, SButton)
            .ButtonStyle(bCoverageVisualized ? 
               &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") : 
               &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"))
            .ContentPadding(FMargin(0, 2))
            .HAlign(HAlign_Center)
            .Text_Lambda([this]() {
                return FText::FromString(bCoverageVisualized ? "Hide Coverage" : "Show Coverage");
            })
            .OnClicked(this, &SVCCSimPanel::OnToggleCoverageVisualizationClicked)
            .IsEnabled_Lambda([this]() {
                return SceneAnalysisManager.IsValid() && !bGenCoverage;
            })
        ]
        +SHorizontalBox::Slot()
        .MaxWidth(120)
        .Padding(FMargin(0, 0, 4, 0))
        .HAlign(HAlign_Fill)
        [
            SNew(SButton)
            .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
            .ContentPadding(FMargin(0, 2))
            .Text(FText::FromString("Clear Coverage"))
            .HAlign(HAlign_Center)
            .OnClicked_Lambda([this]() {
                SceneAnalysisManager->ClearCoverageVisualization();
                bGenCoverage = true;
                return FReply::Handled();
            })
            .IsEnabled_Lambda([this]() {
                return SceneAnalysisManager.IsValid() && !bGenCoverage;
            })
        ]
    ]

    // Separator
    +SVerticalBox::Slot()
    .MaxHeight(1)
    .Padding(FMargin(0, 0, 0, 0))
    [
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(FColor(2, 2, 2))
        .Padding(0)
        .Content()
        [
            SNew(SBox)
            .HeightOverride(1.0f)
        ]
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(0, 4, 0, 0)
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot()
        .MaxWidth(150)
        .Padding(FMargin(0, 0, 4, 0))
        .HAlign(HAlign_Fill)
        [
            SNew(SButton)
            .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
            .ContentPadding(FMargin(0, 2))
            .Text(FText::FromString("Analyze Complexity"))
            .HAlign(HAlign_Center)
            .OnClicked_Lambda([this]() {
                if (SceneAnalysisManager.IsValid())
                {
                    SceneAnalysisManager->AnalyzeGeometricComplexity();
                    bAnalyzeComplexity = false;
                }
                return FReply::Handled();
            })
            .IsEnabled_Lambda([this]() {
                return SceneAnalysisManager.IsValid();
            })
        ]
        +SHorizontalBox::Slot()
        .MaxWidth(150)
        .Padding(FMargin(0, 0, 4, 0))
        .HAlign(HAlign_Fill)
        [
            SAssignNew(VisualizeComplexityButton, SButton)
            .ButtonStyle(bComplexityVisualized ? 
               &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") : 
               &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"))
            .ContentPadding(FMargin(0, 2))
            .HAlign(HAlign_Center)
            .Text_Lambda([this]() {
                return FText::FromString(bComplexityVisualized ? "Hide Complexity" : "Show Complexity");
            })
            .OnClicked(this, &SVCCSimPanel::OnToggleComplexityVisualizationClicked)
            .IsEnabled_Lambda([this]() {
                return SceneAnalysisManager.IsValid() && !bAnalyzeComplexity;
            })
        ]
        +SHorizontalBox::Slot()
        .MaxWidth(150)
        .Padding(FMargin(0, 0, 4, 0))
        .HAlign(HAlign_Fill)
        [
            SNew(SButton)
            .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
            .ContentPadding(FMargin(0, 2))
            .Text(FText::FromString("Clear Complexity"))
            .HAlign(HAlign_Center)
            .OnClicked_Lambda([this]() {
                SceneAnalysisManager->ClearComplexityVisualization();
                bAnalyzeComplexity = true;
                return FReply::Handled();
            })
            .IsEnabled_Lambda([this]() {
                return SceneAnalysisManager.IsValid() && !bAnalyzeComplexity;
            })
        ]
    ]
    )
    ];
}

// Selection state toggle callbacks
void SVCCSimPanel::OnSelectFlashPawnToggleChanged(ECheckBoxState NewState)
{
    bSelectingFlashPawn = (NewState == ECheckBoxState::Checked);
    
    // If turning on FlashPawn selection, disable target selection
    if (bSelectingFlashPawn && bSelectingTarget)
    {
        bSelectingTarget = false;
        SelectTargetToggle->SetIsChecked(ECheckBoxState::Unchecked);
    }
}

void SVCCSimPanel::OnSelectTargetToggleChanged(ECheckBoxState NewState)
{
    bSelectingTarget = (NewState == ECheckBoxState::Checked);
    
    // If turning on Target selection, disable FlashPawn selection
    if (bSelectingTarget && bSelectingFlashPawn)
    {
        bSelectingFlashPawn = false;
        SelectFlashPawnToggle->SetIsChecked(ECheckBoxState::Unchecked);
    }
}

// Camera checkbox callbacks
void SVCCSimPanel::OnRGBCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseRGBCamera = (NewState == ECheckBoxState::Checked);
    if (bUseRGBCamera)
    {
        TArray<URGBCameraComponent*> RGBCameras;
        SelectedFlashPawn->GetComponents<URGBCameraComponent>(RGBCameras);
        for (URGBCameraComponent* Camera : RGBCameras)
        {
            if (Camera)
            {
                Camera->SetActive(bUseRGBCamera);
                Camera->InitializeRenderTargets();
                Camera->SetCaptureComponent();
            }
        }
    }
}

void SVCCSimPanel::OnDepthCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseDepthCamera = (NewState == ECheckBoxState::Checked);
    if (bUseDepthCamera)
    {
        TArray<UDepthCameraComponent*> DepthCameras;
        SelectedFlashPawn->GetComponents<UDepthCameraComponent>(DepthCameras);
        for (UDepthCameraComponent* Camera : DepthCameras)
        {
            if (Camera)
            {
                Camera->SetActive(bUseDepthCamera);
                Camera->InitializeRenderTargets();
                Camera->SetCaptureComponent();
            }
        }
    }
}

void SVCCSimPanel::OnSegmentationCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseSegmentationCamera = (NewState == ECheckBoxState::Checked);
    if (bUseSegmentationCamera)
    {
        TArray<USegmentationCameraComponent*> SegmentationCameras;
        SelectedFlashPawn->GetComponents<USegmentationCameraComponent>(SegmentationCameras);
        for (USegmentationCameraComponent* Camera : SegmentationCameras)
        {
            if (Camera)
            {
                Camera->SetActive(bUseSegmentationCamera);
                Camera->InitializeRenderTargets();
                Camera->SetCaptureComponent();
            }
        }
    }
}

// Selection changed callback
void SVCCSimPanel::OnSelectionChanged(UObject* Object)
{
    // Skip if we're not in selection mode
    if (!bSelectingFlashPawn && !bSelectingTarget)
    {
        return;
    }
    
    USelection* Selection = GEditor->GetSelectedActors();
    if (!Selection || Selection->Num() == 0)
    {
        return;
    }
    
    // Process only the first selected actor
    AActor* Actor = Cast<AActor>(Selection->GetSelectedObject(0));
    if (!Actor)
    {
        return;
    }
    
    // If we're selecting a FlashPawn
    if (bSelectingFlashPawn)
    {
        AFlashPawn* FlashPawn = Cast<AFlashPawn>(Actor);
        if (FlashPawn)
        {
            SelectedFlashPawn = FlashPawn;
            SelectedFlashPawnText->SetText(FText::FromString(FlashPawn->GetActorLabel()));
            
            // Turn off selection mode
            bSelectingFlashPawn = false;
            SelectFlashPawnToggle->SetIsChecked(ECheckBoxState::Unchecked);
            
            // Check what camera components are available
            CheckCameraComponents();
            
            UE_LOG(LogTemp, Display, TEXT("Selected FlashPawn: %s"),
                *FlashPawn->GetActorLabel());
        }
    }
    // If we're selecting a target
    else if (bSelectingTarget)
    {
        // Skip if it's a FlashPawn (can't target itself)
        if (!Actor->IsA<AFlashPawn>())
        {
            SelectedTargetObject = Actor;
            SelectedTargetObjectText->SetText(FText::FromString(Actor->GetActorLabel()));
            
            // Turn off selection mode
            bSelectingTarget = false;
            SelectTargetToggle->SetIsChecked(ECheckBoxState::Unchecked);
            
            UE_LOG(LogTemp, Display, TEXT("Selected Target: %s"), *Actor->GetActorLabel());
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("Cannot select a FlashPawn as a target"));
        }
    }
}

// Check what camera components are available on the selected FlashPawn
void SVCCSimPanel::CheckCameraComponents()
{
    bHasRGBCamera = false;
    bHasDepthCamera = false;
    bHasSegmentationCamera = false;
    
    if (!SelectedFlashPawn.IsValid())
    {
        return;
    }
    
    // Check for RGB cameras
    TArray<URGBCameraComponent*> RGBCameras;
    SelectedFlashPawn->GetComponents<URGBCameraComponent>(RGBCameras);
    bHasRGBCamera = (RGBCameras.Num() > 0);
    
    // Check for Depth cameras
    TArray<UDepthCameraComponent*> DepthCameras;
    SelectedFlashPawn->GetComponents<UDepthCameraComponent>(DepthCameras);
    bHasDepthCamera = (DepthCameras.Num() > 0);
    
    // Check for Segmentation cameras
    TArray<USegmentationCameraComponent*> SegmentationCameras;
    SelectedFlashPawn->GetComponents<USegmentationCameraComponent>(SegmentationCameras);
    bHasSegmentationCamera = (SegmentationCameras.Num() > 0);
    
    UE_LOG(LogTemp, Display, TEXT("FlashPawn camera components: "
                                  "RGB=%d, Depth=%d, Segmentation=%d"),
        bHasRGBCamera ? 1 : 0, bHasDepthCamera ? 1 : 0, bHasSegmentationCamera ? 1 : 0);
    
    // Reset checkboxes if corresponding cameras aren't available
    if (!bHasRGBCamera)
    {
        bUseRGBCamera = false;
        RGBCameraCheckBox->SetIsChecked(ECheckBoxState::Unchecked);
    }
    
    if (!bHasDepthCamera)
    {
        bUseDepthCamera = false;
        DepthCameraCheckBox->SetIsChecked(ECheckBoxState::Unchecked);
    }
    
    if (!bHasSegmentationCamera)
    {
        bUseSegmentationCamera = false;
        SegmentationCameraCheckBox->SetIsChecked(ECheckBoxState::Unchecked);
    }

    UpdateActiveCameras();
}

// Update active camera components
void SVCCSimPanel::UpdateActiveCameras()
{
    if (!SelectedFlashPawn.IsValid())
    {
        return;
    }
    
    // Update RGB cameras
    TArray<URGBCameraComponent*> RGBCameras;
    SelectedFlashPawn->GetComponents<URGBCameraComponent>(RGBCameras);
    for (URGBCameraComponent* Camera : RGBCameras)
    {
        if (Camera)
        {
            Camera->SetActive(bUseRGBCamera);
            Camera->InitializeRenderTargets();
            Camera->SetCaptureComponent();
        }
    }
    
    // Update Depth cameras
    TArray<UDepthCameraComponent*> DepthCameras;
    SelectedFlashPawn->GetComponents<UDepthCameraComponent>(DepthCameras);
    for (UDepthCameraComponent* Camera : DepthCameras)
    {
        if (Camera)
        {
            Camera->SetActive(bUseDepthCamera);
            Camera->InitializeRenderTargets();
            Camera->SetCaptureComponent();
        }
    }
    
    // Update Segmentation cameras
    TArray<USegmentationCameraComponent*> SegmentationCameras;
    SelectedFlashPawn->GetComponents<USegmentationCameraComponent>(SegmentationCameras);
    for (USegmentationCameraComponent* Camera : SegmentationCameras)
    {
        if (Camera)
        {
            Camera->SetActive(bUseSegmentationCamera);
            Camera->InitializeRenderTargets();
            Camera->SetCaptureComponent();
        }
    }
    
    UE_LOG(LogTemp, Display, TEXT("Camera activation updated: "
                                  "RGB=%s, Depth=%s, Segmentation=%s"),
        bUseRGBCamera ? TEXT("On") : TEXT("Off"),
        bUseDepthCamera ? TEXT("On") : TEXT("Off"),
        bUseSegmentationCamera ? TEXT("On") : TEXT("Off"));
}

// Generate poses around target
FReply SVCCSimPanel::OnGeneratePosesClicked()
{
    GeneratePosesAroundTarget();
    bPathVisualized = false;
    bPathNeedsUpdate = true;

    if (PathVisualizationActor.IsValid() && GEditor)
    {
        if (UWorld* World = GEditor->GetEditorWorldContext().World())
        {
            World->DestroyActor(PathVisualizationActor.Get());
        }
        PathVisualizationActor.Reset();
    }
    
    UpdatePathVisualization();
    return FReply::Handled();
}

void SVCCSimPanel::GeneratePosesAroundTarget()
{
    if (!SelectedFlashPawn.IsValid() || !SelectedTargetObject.IsValid())
    {
        UE_LOG(LogTemp, Warning, 
            TEXT("Must select both a FlashPawn and a target object"));
        return;
    }

    TArray<FVector> Positions;
    TArray<FRotator> Rotations;

    TArray<FMeshInfo> MeshInfos;

    TArray<UStaticMeshComponent*> MeshComponents;
    SelectedTargetObject->GetComponents<UStaticMeshComponent>(MeshComponents);
    for (UStaticMeshComponent* MeshComponent : MeshComponents)
    {
        if (MeshComponent)
        {
            FMeshInfo MeshInfo;
            ASceneAnalysisManager::ExtractMeshData(
                MeshComponent, 
                MeshInfo
            );
            MeshInfos.Add(MeshInfo);
        }
    }

    UPathPlanner::SemiSphericalPath(
        MeshInfos, Radius, NumPoses,
        VerticalGap, Positions, Rotations);
    
    // Set the path on the FlashPawn
    SelectedFlashPawn->SetPathPanel(Positions, Rotations);
    SelectedFlashPawn->MoveTo(0);
    
    // Update NumPoses to match actual number of generated poses
    NumPoses = Positions.Num();
    NumPosesValue = NumPoses;
    
    // Reset any ongoing auto-capture
    bAutoCaptureInProgress = false;
    GEditor->GetTimerManager()->ClearTimer(AutoCaptureTimerHandle);
}

// Capture image from current pose
FReply SVCCSimPanel::OnCaptureImagesClicked()
{
    CaptureImageFromCurrentPose();
    return FReply::Handled();
}

void SVCCSimPanel::SaveRGB(int32 PoseIndex, bool& bAnyCaptured)
{
    TArray<URGBCameraComponent*> RGBCameras;
    SelectedFlashPawn->GetComponents<URGBCameraComponent>(RGBCameras);
    *JobNum += RGBCameras.Num();
            
    for (int32 i = 0; i < RGBCameras.Num(); ++i)
    {
        URGBCameraComponent* Camera = RGBCameras[i];
        if (Camera && Camera->IsActive())
        {
            // Get camera index or use iterator index
            int32 CameraIndex = Camera->GetCameraIndex();
            if (CameraIndex < 0) CameraIndex = i;
                    
            // Filename for this camera
            FString Filename = SaveDirectory / FString::Printf(
                TEXT("RGB_Cam%02d_Pose%03d.png"), 
                CameraIndex, 
                PoseIndex
            );
                    
            // Capture the image
            FIntPoint Size = {Camera->GetImageSize().first, Camera->GetImageSize().second};

            // Get image data and save asynchronously
            Camera->AsyncGetRGBImageData(
                [Filename, Size, JobNum = this->JobNum](const TArray<FColor>& ImageData)
                {
                    (new FAutoDeleteAsyncTask<FAsyncImageSaveTask>(ImageData, Size, Filename))
                    ->StartBackgroundTask();
                    *JobNum -= 1;
                });
                    
            bAnyCaptured = true;
        }
    }
}

void SVCCSimPanel::SaveDepth(int32 PoseIndex, bool& bAnyCaptured)
{
    TArray<UDepthCameraComponent*> DepthCameras;
    SelectedFlashPawn->GetComponents<UDepthCameraComponent>(DepthCameras);
    *JobNum += DepthCameras.Num();

    for (int32 i = 0; i < DepthCameras.Num(); ++i)
    {
        UDepthCameraComponent* Camera = DepthCameras[i];
        if (Camera && Camera->IsActive())
        {
            // Get camera index or use iterator index
            int32 CameraIndex = Camera->GetCameraIndex();
            if (CameraIndex < 0) CameraIndex = i;
                    
            // Filename for this camera
            FString Filename = SaveDirectory / FString::Printf(
                TEXT("Depth_Cam%02d_Pose%03d.png"), 
                CameraIndex, 
                PoseIndex
            );
                    
            // Capture the image
            FIntPoint Size = {Camera->GetImageSize().first, Camera->GetImageSize().second};
                    
            // Get image data and save asynchronously
            Camera->AsyncGetDepthImageData(
                [Filename, Size, JobNum = this->JobNum](const TArray<FFloat16Color>& ImageData)
                {
                    TArray<FColor> ConvertedImageData;
                    ConvertedImageData.Reserve(ImageData.Num());
                    for (const FFloat16Color& Color : ImageData)
                    {
                        uint8 DepthValue = FMath::Clamp(
                            FMath::RoundToInt(Color.R / 100.f), 0, 255);
                        ConvertedImageData.Add(FColor(DepthValue,
                            DepthValue, DepthValue, 255));
                    }
                    (new FAutoDeleteAsyncTask<FAsyncImageSaveTask>(ConvertedImageData, Size, Filename))
                    ->StartBackgroundTask();
                    *JobNum -= 1;
                });
                    
            bAnyCaptured = true;
        }
    }
}

void SVCCSimPanel::SaveSeg(int32 PoseIndex, bool& bAnyCaptured)
{
    TArray<USegmentationCameraComponent*> SegmentationCameras;
    SelectedFlashPawn->GetComponents<USegmentationCameraComponent>(SegmentationCameras);
    *JobNum += SegmentationCameras.Num();

    for (int32 i = 0; i < SegmentationCameras.Num(); ++i)
    {
        USegmentationCameraComponent* Camera = SegmentationCameras[i];
        if (Camera && Camera->IsActive())
        {
            // Get camera index or use iterator index
            int32 CameraIndex = Camera->GetCameraIndex();
            if (CameraIndex < 0) CameraIndex = i;
                    
            // Filename for this camera
            FString Filename = SaveDirectory / FString::Printf(
                TEXT("Seg_Cam%02d_Pose%03d.png"), 
                CameraIndex, 
                PoseIndex
            );
                    
            // Capture the image
            FIntPoint Size = {Camera->GetImageSize().first, Camera->GetImageSize().second};
                    
            // Get image data and save asynchronously
            Camera->AsyncGetSegmentationImageData(
                [Filename, Size, JobNum = this->JobNum](TArray<FColor> ImageData)
                {
                    for (FColor& Color : ImageData)
                    {
                        Color.A = 255; // Ensure alpha is set to 255
                    }
                    (new FAutoDeleteAsyncTask<FAsyncImageSaveTask>(ImageData, Size, Filename))
                    ->StartBackgroundTask();
                    *JobNum -= 1;
                });
                    
            bAnyCaptured = true;
        }
    }
}

void SVCCSimPanel::CaptureImageFromCurrentPose()
{
    if (!SelectedFlashPawn.IsValid())
    {
        UE_LOG(LogTemp, Warning, TEXT("No FlashPawn selected"));
        return;
    }
    
    // Create a directory for saving images if it doesn't exist yet
    if (SaveDirectory.IsEmpty())
    {
        SaveDirectory = FPaths::ProjectSavedDir() / TEXT("VCCSimCaptures") / GetTimestampedFilename();
        IFileManager::Get().MakeDirectory(*SaveDirectory, true);
        UE_LOG(LogTemp, Display, TEXT("Saving images to: %s"), *SaveDirectory);
    }
    
    // Check if the FlashPawn is ready to capture
    if (SelectedFlashPawn->IsReady())
    {
        // Pose index for filename
        int32 PoseIndex = SelectedFlashPawn->GetCurrentIndex();
        
        // Track if any cameras were captured
        bool bAnyCaptured = false;
        
        // Capture with RGB cameras if enabled
        if (bUseRGBCamera && bHasRGBCamera)
        {
            SaveRGB(PoseIndex, bAnyCaptured);
        }
        
        // Capture with Depth cameras if enabled
        if (bUseDepthCamera && bHasDepthCamera)
        {
            SaveDepth(PoseIndex, bAnyCaptured);
        }
        
        // Capture with Segmentation cameras if enabled
        if (bUseSegmentationCamera && bHasSegmentationCamera)
        {
            SaveSeg(PoseIndex, bAnyCaptured);
        }
        
        // Log if no images were captured
        if (!bAnyCaptured)
        {
            UE_LOG(LogTemp, Warning, TEXT("No images captured. "
                                          "Ensure cameras are enabled."));
        }
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("FlashPawn not ready for capture. "
                                      "Wait for it to reach position."));
    }
}

// Start auto-capture
void SVCCSimPanel::StartAutoCapture()
{
    if (!SelectedFlashPawn.IsValid())
    {
        UE_LOG(LogTemp, Warning, TEXT("No FlashPawn selected"));
        return;
    }
    
    // Create a directory for saving images
    SaveDirectory = FPaths::ProjectSavedDir() / TEXT("VCCSimCaptures") / GetTimestampedFilename();
    IFileManager::Get().MakeDirectory(*SaveDirectory, true);
    UE_LOG(LogTemp, Display, TEXT("Auto-capture started. Saving images to: %s"), *SaveDirectory);
    
    // Start the capture process
    bAutoCaptureInProgress = true;
    *JobNum = 0;

    SelectedFlashPawn->MoveTo(0);
    
    // Set up a timer to check if the FlashPawn is ready for capture
    GEditor->GetTimerManager()->SetTimer(
        AutoCaptureTimerHandle,
        [this]()
        {
            if (!bAutoCaptureInProgress || !SelectedFlashPawn.IsValid())
            {
                // Stop the timer if auto-capture is cancelled or FlashPawn is invalid
                GEditor->GetTimerManager()->ClearTimer(AutoCaptureTimerHandle);
                bAutoCaptureInProgress = false;
                return;
            }
            
            // Check if the FlashPawn is ready to capture
            if (SelectedFlashPawn->IsReady())
            {
                CaptureImageFromCurrentPose();
                SelectedFlashPawn->MoveToNext();
                
                // If we've finished capturing all poses, stop the auto-capture
                if (SelectedFlashPawn->GetCurrentIndex() == NumPoses - 1)
                {
                    SaveDirectory.Empty(); // Reset for next capture session
                    UE_LOG(LogTemp, Display, TEXT("Auto-capture completed"));
                    bAutoCaptureInProgress = false;
                    GEditor->GetTimerManager()->ClearTimer(AutoCaptureTimerHandle);
                }
            }
            else if (*JobNum == 0)
            {
                SelectedFlashPawn->MoveForward();
            }
        },
        0.01f,
        true
    );
}

// Helper to get a timestamped filename
FString SVCCSimPanel::GetTimestampedFilename()
{
    FDateTime Now = FDateTime::Now();
    return FString::Printf(TEXT("%04d-%02d-%02d_%02d-%02d-%02d"),
        Now.GetYear(), Now.GetMonth(), Now.GetDay(),
        Now.GetHour(), Now.GetMinute(), Now.GetSecond());
}

// Load predefined pose from file
void SVCCSimPanel::LoadPredefinedPose()
{
    if (!SelectedFlashPawn.IsValid())
    {
        UE_LOG(LogTemp, Warning, TEXT("No FlashPawn selected"));
        return;
    }
    
    // Open file dialog to select pose file
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (DesktopPlatform)
    {
        TArray<FString> OpenFilenames;
        FString ExtensionStr = TEXT("Pose Files (*.txt)|*.txt");
        
        bool bOpened = DesktopPlatform->OpenFileDialog(
            FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr),
            TEXT("Load Pose File"),
            FPaths::ProjectSavedDir(),
            TEXT(""),
            *ExtensionStr,
            EFileDialogFlags::None,
            OpenFilenames
        );
        
        if (bOpened && OpenFilenames.Num() > 0)
        {
            FString SelectedFile = OpenFilenames[0];
            UE_LOG(LogTemp, Display, TEXT("Loading pose file: %s"), *SelectedFile);
            
            // Read file content
            TArray<FString> FileLines;
            if (FFileHelper::LoadFileToStringArray(FileLines, *SelectedFile))
            {
                TArray<FVector> Positions;
                TArray<FRotator> Rotations;
                
                for (const FString& Line : FileLines)
                {
                    // Parse line
                    // Expected format: X Y Z Pitch Yaw Roll
                    TArray<FString> Values;
                    Line.ParseIntoArray(Values, TEXT(" "), true);
                    
                    if (Values.Num() >= 6)
                    {
                        float X = FCString::Atof(*Values[0]);
                        float Y = FCString::Atof(*Values[1]);
                        float Z = FCString::Atof(*Values[2]);
                        float Pitch = FCString::Atof(*Values[3]);
                        float Yaw = FCString::Atof(*Values[4]);
                        float Roll = FCString::Atof(*Values[5]);
                        
                        Positions.Add(FVector(X, Y, Z));
                        Rotations.Add(FRotator(Pitch, Yaw, Roll));
                    }
                }
                
                if (Positions.Num() > 0 && Positions.Num() == Rotations.Num())
                {
                    // Set the path on the FlashPawn
                    SelectedFlashPawn->SetPathPanel(Positions, Rotations);
                    
                    // Update NumPoses
                    NumPoses = Positions.Num();
                    NumPosesValue = NumPoses;
                    
                    UE_LOG(LogTemp, Display, TEXT("Loaded %d poses from file"),
                        Positions.Num());
                    bPathVisualized = false;
                    bPathNeedsUpdate = true;
                    
                    if (PathVisualizationActor.IsValid() && GEditor)
                    {
                        if (UWorld* World = GEditor->GetEditorWorldContext().World())
                        {
                            World->DestroyActor(PathVisualizationActor.Get());
                        }
                        PathVisualizationActor.Reset();
                    }
                    
                    UpdatePathVisualization();
                }
                else
                {
                    UE_LOG(LogTemp, Warning, TEXT("Failed to parse pose file: "
                                                  "Invalid format or empty file"));
                }
            }
            else
            {
                UE_LOG(LogTemp, Warning, TEXT("Failed to load file"));
            }
        }
    }
}

// Save generated pose to file
void SVCCSimPanel::SaveGeneratedPose()
{
    if (!SelectedFlashPawn.IsValid())
    {
        UE_LOG(LogTemp, Warning, TEXT("No FlashPawn selected"));
        return;
    }
    
    // Check if there are poses to save
    int32 PoseCount = SelectedFlashPawn->GetPoseCount();
    if (PoseCount <= 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("No poses to save"));
        return;
    }
    
    // Open file dialog to select save location
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (DesktopPlatform)
    {
        TArray<FString> SaveFilenames;
        FString ExtensionStr = TEXT("Pose Files (*.txt)|*.txt");
        
        bool bSaved = DesktopPlatform->SaveFileDialog(
            FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr),
            TEXT("Save Pose File"),
            FPaths::ProjectSavedDir(),
            TEXT("poses.txt"),
            *ExtensionStr,
            EFileDialogFlags::None,
            SaveFilenames
        );
        
        if (bSaved && SaveFilenames.Num() > 0)
        {
            FString SelectedFile = SaveFilenames[0];
            
            // Ensure the file has .txt extension
            if (!SelectedFile.EndsWith(TEXT(".txt")))
            {
                SelectedFile += TEXT(".txt");
            }
            
            UE_LOG(LogTemp, Display, TEXT("Saving pose file: %s"), *SelectedFile);
            
            // Get positions and rotations from FlashPawn
            TArray<FVector> Positions;
            TArray<FRotator> Rotations;
            SelectedFlashPawn->GetCurrentPath(Positions, Rotations);
            
            // Build file content
            FString FileContent;
            for (int32 i = 0; i < Positions.Num(); ++i)
            {
                const FVector& Pos = Positions[i];
                const FRotator& Rot = Rotations[i];
                
                // Format: X Y Z Pitch Yaw Roll
                FileContent += FString::Printf(
                    TEXT("%.6f %.6f %.6f %.6f %.6f %.6f\n"),
                    Pos.X, Pos.Y, Pos.Z,
                    Rot.Pitch, Rot.Yaw, Rot.Roll
                );
            }
            
            // Save to file
            if (FFileHelper::SaveStringToFile(FileContent, *SelectedFile))
            {
                UE_LOG(LogTemp, Display, TEXT("Successfully saved %d poses to file"),
                    Positions.Num());
            }
            else
            {
                UE_LOG(LogTemp, Warning, TEXT("Failed to save file"));
            }
        }
    }
}

// Button callbacks for pose file functions
FReply SVCCSimPanel::OnLoadPoseClicked()
{
    LoadPredefinedPose();
    return FReply::Handled();
}

FReply SVCCSimPanel::OnSavePoseClicked()
{
    SaveGeneratedPose();
    return FReply::Handled();
}

// Create a standardized section header with proper styling
TSharedRef<SWidget> SVCCSimPanel::CreateSectionHeader(const FString& Title)
{
    return SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryTop"))
        .Padding(FMargin(10, 7))
        [
            SNew(STextBlock)
            .Text(FText::FromString(Title))
            .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
            .ColorAndOpacity(FColor(233, 233, 233)) // Light text for dark background
            .TransformPolicy(ETextTransformPolicy::ToUpper)
        ];
}

// Create a standardized content container with proper styling
TSharedRef<SWidget> SVCCSimPanel::CreateSectionContent(TSharedRef<SWidget> Content)
{
    return SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(FColor(5,5, 5, 255))
        .Padding(FMargin(15, 6))
        [
            Content
        ];
}

// Create a standardized property row with label and content
TSharedRef<SWidget> SVCCSimPanel::CreatePropertyRow(
    const FString& Label, TSharedRef<SWidget> Content)
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
        .ColorAndOpacity(FColor(233, 233, 233)) 
    ]
    +SHorizontalBox::Slot()
    .FillWidth(1.0f)
    [
        Content
    ];
}

FReply SVCCSimPanel::OnTogglePathVisualizationClicked()
{
    if (!SelectedFlashPawn.IsValid())
    {
        return FReply::Handled();
    }
    
    // Toggle the visualization state
    bPathVisualized = !bPathVisualized;

    if (bPathVisualized)
    {
        // Show the path visualization
        ShowPathVisualization();
    }
    else
    {
        // Hide the path visualization
        HidePathVisualization();
    }

    VisualizePathButton->SetButtonStyle(bPathVisualized ? 
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") : 
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
    
    return FReply::Handled();
}

void SVCCSimPanel::UpdatePathVisualization()
{
    TArray<FVector> Positions = SelectedFlashPawn->PendingPositions;
    TArray<FRotator> Rotations = SelectedFlashPawn->PendingRotations;

    if (Positions.Num() == 0 || Positions.Num() != Rotations.Num())
    {
        UE_LOG(LogTemp, Warning, TEXT("No valid path available for visualization"));
        bPathVisualized = false;
    }

    UMaterialInterface* PathMaterial = LoadObject<UMaterialInterface>(nullptr, 
       TEXT("/VCCSim/Materials/M_Rat_Path_ice.M_Rat_Path_ice"));
    UMaterialInterface* CameraMaterial = LoadObject<UMaterialInterface>(nullptr, 
        TEXT("/VCCSim/Materials/M_Rat_Path_Blue.M_Rat_Path_Blue"));
        
    if (!PathMaterial || !CameraMaterial)
    {
        UE_LOG(LogTemp, Warning, TEXT("Failed to load path visualization materials"));
        bPathVisualized = false;
    }
        
    // Generate new visualization actor
    PathVisualizationActor = UTrajectoryViewer::GenerateVisibleElements(
        GEditor->GetEditorWorldContext().World(),
        Positions,
        Rotations,
        PathMaterial,
        CameraMaterial,
        2.0f,     // Path width
        30.0f,    // Cone size
        45.0f     // Cone length
    );
        
    if (!PathVisualizationActor.IsValid())
    {
        UE_LOG(LogTemp, Warning, TEXT("Failed to create path visualization"));
        bPathVisualized = false;
    }

    HidePathVisualization();
    bPathNeedsUpdate = false;
    VisualizePathButton->SetButtonStyle(bPathVisualized ? 
    &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") : 
    &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
}

void SVCCSimPanel::ShowPathVisualization()
{
    if (PathVisualizationActor.IsValid())
    {
        // Show the actor in game
        PathVisualizationActor->SetActorHiddenInGame(false);
        
        // Show the actor in editor
        PathVisualizationActor->SetIsTemporarilyHiddenInEditor(false);
        
        // Show all components
        TArray<UActorComponent*> Components;
        PathVisualizationActor->GetComponents(UStaticMeshComponent::StaticClass(), Components);
        for (UActorComponent* Component : Components)
        {
            UStaticMeshComponent* MeshComp = Cast<UStaticMeshComponent>(Component);
            if (MeshComp)
            {
                MeshComp->SetVisibility(true);
                MeshComp->SetHiddenInGame(false);
            }
        }
    }
}

void SVCCSimPanel::HidePathVisualization()
{
    // First check if the weak pointer is valid
    if (PathVisualizationActor.IsValid())
    {
        // Hide the actor in game
        PathVisualizationActor->SetActorHiddenInGame(true);
        
        // Hide the actor in editor
        PathVisualizationActor->SetIsTemporarilyHiddenInEditor(true);
        
        // Hide all components (for thoroughness)
        TArray<UActorComponent*> Components;
        PathVisualizationActor->GetComponents(UStaticMeshComponent::StaticClass(), Components);
        for (UActorComponent* Component : Components)
        {
            UStaticMeshComponent* MeshComp = Cast<UStaticMeshComponent>(Component);
            if (MeshComp)
            {
                MeshComp->SetVisibility(false);
                MeshComp->SetHiddenInGame(true);
            }
        }
    }
}

FReply SVCCSimPanel::OnToggleSafeZoneVisualizationClicked()
{
    if (!SceneAnalysisManager.IsValid())
    {
        return FReply::Handled();
    }
    
    // Toggle the visualization state
    bSafeZoneVisualized = !bSafeZoneVisualized;

    if (bSafeZoneVisualized)
    {
        SceneAnalysisManager->VisualizeSafeZone(true);
    }
    else
    {
        SceneAnalysisManager->VisualizeSafeZone(false);
    }

    VisualizeSafeZoneButton->SetButtonStyle(bSafeZoneVisualized ? 
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") : 
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
    
    return FReply::Handled();
}

FReply SVCCSimPanel::OnToggleCoverageVisualizationClicked()
{
    if (!SceneAnalysisManager.IsValid())
    {
        return FReply::Handled();
    }
    
    // Toggle the visualization state
    bCoverageVisualized = !bCoverageVisualized;
    
    SceneAnalysisManager->VisualizeCoverage(bCoverageVisualized);
    
    VisualizeCoverageButton->SetButtonStyle(bCoverageVisualized ? 
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") : 
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
    
    return FReply::Handled();
}

FReply SVCCSimPanel::OnToggleComplexityVisualizationClicked()
{
    if (!SceneAnalysisManager.IsValid())
    {
        return FReply::Handled();
    }
    
    // Toggle the visualization state
    bComplexityVisualized = !bComplexityVisualized;
    
    SceneAnalysisManager->VisualizeComplexity(bComplexityVisualized);
    
    VisualizeComplexityButton->SetButtonStyle(bComplexityVisualized ? 
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") : 
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
    
    return FReply::Handled();
}

namespace FVCCSimPanelFactory
{
    const FName TabId = FName("VCCSimPanel");
}

void FVCCSimPanelFactory::RegisterTabSpawner(FTabManager& TabManager)
{
    TabManager.RegisterTabSpawner(
        TabId, 
        FOnSpawnTab::CreateLambda([](const FSpawnTabArgs& InArgs) -> TSharedRef<SDockTab> {
            return SNew(SDockTab)
                .TabRole(ETabRole::NomadTab)
                .Label(FText::FromString("Image Acquisition"))
                [
                    SNew(SVCCSimPanel)
                ];
        })
    )
    .SetDisplayName(FText::FromString("Image Acquisition"));
}