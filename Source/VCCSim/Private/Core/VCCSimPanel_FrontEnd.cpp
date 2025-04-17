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
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SNumericEntryBox.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Text/STextBlock.h"
#include "Pawns/FlashPawn.h"
#include "Misc/DateTime.h"
#include "HAL/FileManager.h"
#include "Misc/Paths.h"
#include "Sensors/CameraSensor.h"
#include "Simulation/SceneAnalysisManager.h"
#include "WorkspaceMenuStructure.h"
#include "WorkspaceMenuStructureModule.h"
#include "Misc/FileHelper.h"
#include "Styling/SlateStyleRegistry.h"
#include "Editor/UnrealEd/Public/Editor.h"
#include "Editor/UnrealEd/Public/Selection.h"
#include "EngineUtils.h"

void SVCCSimPanel::Construct(const FArguments& InArgs)
{
    NumPosesValue = NumPoses;
    RadiusValue = Radius;
    HeightOffsetValue = HeightOffset;
    VerticalGapValue = VerticalGap;
    SafeDistanceValue = SafeDistance;
    SafeHeightValue = SafeHeight;
    LimitedMinXValue = LimitedMinX;
    LimitedMaxXValue = LimitedMaxX;
    LimitedMinYValue = LimitedMinY;
    LimitedMaxYValue = LimitedMaxY;
    LimitedMinZValue = LimitedMinZ;
    LimitedMaxZValue = LimitedMaxZ;
    JobNum = MakeShared<std::atomic<int32>>(0);
    
    // Register for selection change events
    if (USelection* Selection = GEditor->GetSelectedActors())
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
        CreateSectionHeader("Path Configuration")
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
                .MaxWidth(180)
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
                .MaxWidth(180)
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
                .MaxWidth(180)
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
                .MaxWidth(180)
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
        
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        .Padding(FMargin(0, 0, 8, 0))
        [
            CreatePropertyRow(
                "Limited MinX",
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5,5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(LimitedMinXSpinBox, SNumericEntryBox<float>)
                    .Value_Lambda([this]() { return LimitedMinXValue; })
                    .OnValueChanged(SNumericEntryBox<float>::FOnValueChanged::CreateLambda(
                        [this](float NewValue) {
                        LimitedMinX = NewValue;
                        LimitedMinXValue = NewValue;
                    }))
                ]
            )
        ]
        
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        .Padding(FMargin(0, 0, 0, 0))
        [
            CreatePropertyRow(
                "Limited MaxX",
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5,5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(LimitedMaxXSpinBox, SNumericEntryBox<float>)
                    .Value_Lambda([this]() { return LimitedMaxXValue; })
                    .OnValueChanged(SNumericEntryBox<float>::FOnValueChanged::CreateLambda(
                        [this](float NewValue) {
                        LimitedMaxX = NewValue;
                        LimitedMaxXValue = NewValue;
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
    .Padding(FMargin(0, 0, 0, 4))
    [
        SNew(SHorizontalBox)
        
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        .Padding(FMargin(0, 0, 8, 0))
        [
            CreatePropertyRow(
                "Limited MinY",
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5,5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(LimitedMinYSpinBox, SNumericEntryBox<float>)
                    .Value_Lambda([this]() { return LimitedMinYValue; })
                    .OnValueChanged(SNumericEntryBox<float>::FOnValueChanged::CreateLambda(
                        [this](float NewValue) {
                        LimitedMinY = NewValue;
                        LimitedMinYValue = NewValue;
                    }))
                ]
            )
        ]
        
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        .Padding(FMargin(0, 0, 0, 0))
        [
            CreatePropertyRow(
                "Limited MaxY",
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5,5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(LimitedMaxYSpinBox, SNumericEntryBox<float>)
                    .Value_Lambda([this]() { return LimitedMaxYValue; })
                    .OnValueChanged(SNumericEntryBox<float>::FOnValueChanged::CreateLambda(
                        [this](float NewValue) {
                        LimitedMaxY = NewValue;
                        LimitedMaxYValue = NewValue;
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
    .Padding(FMargin(0, 0, 0, 4))
    [
        SNew(SHorizontalBox)
        
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        .Padding(FMargin(0, 0, 8, 0))
        [
            CreatePropertyRow(
                "Limited MinZ",
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5,5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(LimitedMinZSpinBox, SNumericEntryBox<float>)
                    .Value_Lambda([this]() { return LimitedMinZValue; })
                    .OnValueChanged(SNumericEntryBox<float>::FOnValueChanged::CreateLambda(
                        [this](float NewValue) {
                        LimitedMinZ = NewValue;
                        LimitedMinZValue = NewValue;
                    }))
                ]
            )
        ]
        
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        .Padding(FMargin(0, 0, 0, 0))
        [
            CreatePropertyRow(
                "Limited MaxZ",
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5,5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(LimitedMaxZSpinBox, SNumericEntryBox<float>)
                    .Value_Lambda([this]() { return LimitedMaxZValue; })
                    .OnValueChanged(SNumericEntryBox<float>::FOnValueChanged::CreateLambda(
                        [this](float NewValue) {
                        LimitedMaxZ = NewValue;
                        LimitedMaxZValue = NewValue;
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
    .Padding(FMargin(0, 0, 0, 4))
    [
        SNew(SHorizontalBox)
        
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
        .MaxWidth(150)
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
                    if (bUseLimited)
                    {
                        SceneAnalysisManager->ScanSceneRegion3D(
                            LimitedMinX, LimitedMaxX,
                            LimitedMinY, LimitedMaxY,
                            LimitedMinZ, LimitedMaxZ);
                    }
                    else
                    {
                        SceneAnalysisManager->ScanScene();
                    }
                    bNeedScan = false;
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
            SNew(SButton)
            .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
            .ContentPadding(FMargin(5, 2))
            .Text(FText::FromString("Register Camera"))
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
        .AutoWidth()
        .VAlign(VAlign_Center)
        .Padding(FMargin(8, 0, 4, 0))
        [
            SAssignNew(SelectUseLimitedToggle, SCheckBox)
            .IsChecked(bUseLimited ? ECheckBoxState::Checked : ECheckBoxState::Unchecked)
            .OnCheckStateChanged(this, &SVCCSimPanel::OnUseLimitedToggleChanged)
        ]
        +SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        [
            SNew(STextBlock)
            .Text(FText::FromString("Limited Region"))
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
        .MaxWidth(150)
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
                    SceneAnalysisManager->GenerateSafeZone(SafeDistance);
                    bGenSafeZone = false;
                }
                return FReply::Handled();
            })
            .IsEnabled_Lambda([this]() {
                return SceneAnalysisManager.IsValid() && !bNeedScan;
            })
        ]
        +SHorizontalBox::Slot()
        .MaxWidth(150)
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
        .MaxWidth(150)
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
                bSafeZoneVisualized = false;
                VisualizeSafeZoneButton->SetButtonStyle(
                    &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
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
        .MaxWidth(150)
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
        .MaxWidth(150)
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
        .MaxWidth(150)
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
                bCoverageVisualized = false;
                VisualizeCoverageButton ->SetButtonStyle(
                    &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
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
                return SceneAnalysisManager.IsValid() && !bNeedScan;
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
                bComplexityVisualized = false;
                VisualizeComplexityButton ->SetButtonStyle(
                    &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
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

namespace FVCCSimPanelFactory
{
    const FName TabId = FName("VCCSimPanel");
}

void FVCCSimPanelFactory::RegisterTabSpawner(FTabManager& TabManager)
{
    // Create the style set if it doesn't exist
    static FName VCCSimStyleName("VCCSimStyle");
    static TSharedPtr<FSlateStyleSet> StyleSet;
    
    if (!StyleSet.IsValid())
    {
        StyleSet = MakeShareable(new FSlateStyleSet(VCCSimStyleName));
        
        // Set the content root
        FString PluginDir = FPaths::Combine(FPaths::ProjectPluginsDir(), TEXT("VCCSim"));
        StyleSet->SetContentRoot(PluginDir);
        
        // Register the icon
        FString VCCLogoPath = FPaths::Combine(PluginDir, TEXT("image/Logo/vcc.png"));
        StyleSet->Set("VCCSimStyle.TabIcon",
            new FSlateImageBrush(VCCLogoPath, FVector2D(16, 16)));
        
        // Register the style
        FSlateStyleRegistry::RegisterSlateStyle(*StyleSet.Get());
    }
    
    // Now use the registered icon in the tab spawner
    TabManager.RegisterTabSpawner(
        TabId, 
        FOnSpawnTab::CreateLambda([](const FSpawnTabArgs& InArgs) -> TSharedRef<SDockTab> {
            return SNew(SDockTab)
                .TabRole(ETabRole::NomadTab)
                .Label(FText::FromString("VCCSIM Panel"))
                [
                    SNew(SVCCSimPanel)
                ];
        })
    )
    .SetDisplayName(FText::FromString("VCCSIM"))
    .SetIcon(FSlateIcon(VCCSimStyleName, "VCCSimStyle.TabIcon"))
    .SetGroup(WorkspaceMenu::GetMenuStructure().GetLevelEditorCategory());
}