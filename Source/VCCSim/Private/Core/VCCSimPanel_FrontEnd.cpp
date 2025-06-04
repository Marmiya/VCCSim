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

#include "DesktopPlatformModule.h"
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
#include "MaterialDomain.h"

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
            FVector2D(65, 65),
            FColor(255, 255, 255, 255)));
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("VCC logo file not found at: %s"), *VCCLogoPath);
    }

    if (FPaths::FileExists(SZULogoPath))
    {
        float SZUWidth = 80 * (272.0f / 80.0f);
        SZULogoBrush = MakeShareable(new FSlateDynamicImageBrush(
            FName(*SZULogoPath), 
            FVector2D(SZUWidth, 80),
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
                SceneAnalysisManager->InterfaceInitializeSafeZoneVisualization();
                SceneAnalysisManager->InterfaceInitializeCoverageVisualization();
                SceneAnalysisManager->InterfaceInitializeComplexityVisualization();
                break;
            }
            break;
        }
    }
    
    // Create the widget layout with scrollbar
    ChildSlot
    [
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryTop"))
        .Padding(0)
        [
            // Add SScrollBox here to enable scrolling
            SNew(SScrollBox)
            .ScrollBarAlwaysVisible(false)  // Changed to false - only show when needed
            .AllowOverscroll(EAllowOverscroll::No)
            .ScrollBarThickness(FVector2D(8.0f, 8.0f))
            + SScrollBox::Slot()
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

                // Point Cloud panel
                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    CreatePointCloudPanel()
                ]
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
                    SceneAnalysisManager->InterfaceGenerateSafeZone(SafeDistance);
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
                SceneAnalysisManager->InterfaceClearSafeZoneVisualization();
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
                    SceneAnalysisManager->InterfaceComputeCoverage(
                        CoverageTransforms, "CoverageCamera");
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
                SceneAnalysisManager->InterfaceClearCoverageVisualization();
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
                    SceneAnalysisManager->InterfaceAnalyzeGeometricComplexity();
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
                SceneAnalysisManager->InterfaceClearComplexityVisualization();
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

TSharedRef<SWidget> SVCCSimPanel::CreatePointCloudPanel()
{
    return SNew(SVerticalBox)
    +SVerticalBox::Slot()
    .AutoHeight()
    [
        CreateSectionHeader("Point Cloud")
    ]
    +SVerticalBox::Slot()
    .AutoHeight()
    [
        CreateSectionContent(
            SNew(SVerticalBox)
            
            // Point cloud status
            +SVerticalBox::Slot()
            .AutoHeight()
            .Padding(FMargin(0, 4, 0, 4))
            [
                CreatePropertyRow(
                    "Status",
                    SNew(SBorder)
                    .Padding(4)
                    [
                        SAssignNew(PointCloudStatusText, STextBlock)
                        .Text_Lambda([this]() {
                            if (bPointCloudLoaded)
                            {
                                return FText::FromString(FString::Printf(TEXT("Loaded: %d points (Original scale)"), PointCloudCount));
                            }
                            return FText::FromString("No point cloud loaded");
                        })
                        .ColorAndOpacity_Lambda([this]() {
                            return bPointCloudLoaded ? FSlateColor(FLinearColor::Green) : FSlateColor(FLinearColor::Gray);
                        })
                    ]
                )
            ]
            
            // Color status
            +SVerticalBox::Slot()
            .AutoHeight()
            .Padding(FMargin(0, 0, 0, 4))
            [
                CreatePropertyRow(
                    "Colors",
                    SNew(SBorder)
                    .Padding(4)
                    [
                        SAssignNew(PointCloudColorStatusText, STextBlock)
                        .Text_Lambda([this]() {
                            if (!bPointCloudLoaded)
                            {
                                return FText::FromString("No data");
                            }
                            return FText::FromString(bPointCloudHasColors ? "RGB colors detected" : "Using default orange");
                        })
                        .ColorAndOpacity_Lambda([this]() {
                            if (!bPointCloudLoaded) return FSlateColor(FLinearColor::Gray);
                            return bPointCloudHasColors ? FSlateColor(FLinearColor::Blue) : FSlateColor(FLinearColor::Yellow);
                        })
                    ]
                )
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
            
            // Control buttons
            +SVerticalBox::Slot()
            .AutoHeight()
            .Padding(0, 4, 0, 4)
            [
                SNew(SHorizontalBox)
                
                // Load Point Cloud button
                +SHorizontalBox::Slot()
                .MaxWidth(150)
                .Padding(FMargin(0, 0, 4, 0))
                .HAlign(HAlign_Fill)
                [
                    SAssignNew(LoadPointCloudButton, SButton)
                    .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
                    .ContentPadding(FMargin(5, 2))
                    .Text(FText::FromString("Load PLY File"))
                    .HAlign(HAlign_Center)
                    .OnClicked(this, &SVCCSimPanel::OnLoadPointCloudClicked)
                ]
                
                // Visualize Point Cloud button
                +SHorizontalBox::Slot()
                .MaxWidth(150)
                .Padding(FMargin(0, 0, 4, 0))
                .HAlign(HAlign_Fill)
                [
                    SAssignNew(VisualizePointCloudButton, SButton)
                    .ButtonStyle(bPointCloudVisualized ? 
                       &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") : 
                       &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"))
                    .ContentPadding(FMargin(5, 2))
                    .HAlign(HAlign_Center)
                    .Text_Lambda([this]() {
                        return FText::FromString(bPointCloudVisualized ? "Hide Point Cloud" : "Show Point Cloud");
                    })
                    .OnClicked(this, &SVCCSimPanel::OnTogglePointCloudVisualizationClicked)
                    .IsEnabled_Lambda([this]() {
                        return bPointCloudLoaded;
                    })
                ]
                
                // Clear Point Cloud button
                +SHorizontalBox::Slot()
                .MaxWidth(150)
                .Padding(FMargin(0, 0, 0, 0))
                .HAlign(HAlign_Fill)
                [
                    SNew(SButton)
                    .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
                    .ContentPadding(FMargin(5, 2))
                    .Text(FText::FromString("Clear Point Cloud"))
                    .HAlign(HAlign_Center)
                    .OnClicked_Lambda([this]() {
                        ClearPointCloudVisualization();
                        PointCloudData.Empty();
                        bPointCloudLoaded = false;
                        bPointCloudVisualized = false;
                        bPointCloudHasColors = false;
                        PointCloudCount = 0;
                        LoadedPointCloudPath.Empty();
                        
                        // Update button style
                        VisualizePointCloudButton->SetButtonStyle(
                            &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
                        
                        return FReply::Handled();
                    })
                    .IsEnabled_Lambda([this]() {
                        return bPointCloudLoaded;
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

FReply SVCCSimPanel::OnLoadPointCloudClicked()
{
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (DesktopPlatform)
    {
        TArray<FString> OpenFilenames;
        FString ExtensionStr = TEXT("PLY Files (*.ply)|*.ply");
        
        const bool bOpened = DesktopPlatform->OpenFileDialog(
            FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr),
            TEXT("Load Point Cloud PLY File"),
            FPaths::ProjectDir(),
            TEXT(""),
            *ExtensionStr,
            EFileDialogFlags::None,
            OpenFilenames
        );
        
        if (bOpened && OpenFilenames.Num() > 0)
        {
            const FString& SelectedFile = OpenFilenames[0];
            
            // Use the new PLY loader without coordinate scaling
            FPLYLoader::FPLYLoadResult LoadResult = FPLYLoader::LoadPLYFile(
                SelectedFile, 
                DefaultPointColor
            );
            
            if (LoadResult.bSuccess)
            {
                // Update our data
                PointCloudData = MoveTemp(LoadResult.Points);
                bPointCloudLoaded = true;
                bPointCloudHasColors = LoadResult.bHasColors;
                PointCloudCount = LoadResult.PointCount;
                LoadedPointCloudPath = SelectedFile;
                
                // Clear any existing visualization
                if (bPointCloudVisualized)
                {
                    ClearPointCloudVisualization();
                    bPointCloudVisualized = false;
                    VisualizePointCloudButton->SetButtonStyle(
                        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
                }
                
                UE_LOG(LogTemp, Warning, TEXT("Successfully loaded point cloud: %s (Points: %d, Colors: %s, No coordinate transform)"), 
                    *SelectedFile, PointCloudCount, bPointCloudHasColors ? TEXT("Yes") : TEXT("No"));
            }
            else
            {
                UE_LOG(LogTemp, Error, TEXT("Failed to load point cloud: %s - %s"), 
                    *SelectedFile, *LoadResult.ErrorMessage);
                
                // Clear data on failure
                PointCloudData.Empty();
                bPointCloudLoaded = false;
                bPointCloudHasColors = false;
                PointCloudCount = 0;
                LoadedPointCloudPath.Empty();
            }
        }
    }
    
    return FReply::Handled();
}

// Toggle Point Cloud Visualization
FReply SVCCSimPanel::OnTogglePointCloudVisualizationClicked()
{
    if (!bPointCloudLoaded)
    {
        return FReply::Handled();
    }
    
    if (bPointCloudVisualized)
    {
        // Hide point cloud
        if (PointCloudActor.IsValid())
        {
            PointCloudActor->SetActorHiddenInGame(true);
            PointCloudActor->SetActorEnableCollision(false);
        }
        bPointCloudVisualized = false;
        
        // Update button style
        VisualizePointCloudButton->SetButtonStyle(
            &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
    }
    else
    {
        // Show point cloud
        if (!PointCloudActor.IsValid())
        {
            CreateProceduralPointCloudVisualization();
        }
        else
        {
            PointCloudActor->SetActorHiddenInGame(false);
            PointCloudActor->SetActorEnableCollision(true);
        }
        bPointCloudVisualized = true;
        
        // Update button style
        VisualizePointCloudButton->SetButtonStyle(
            &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger"));
    }
    
    return FReply::Handled();
}

void SVCCSimPanel::CreateProceduralPointCloudVisualization()
{
    if (PointCloudData.Num() == 0)
    {
        return;
    }
    
    UWorld* World = GEditor->GetEditorWorldContext().World();
    if (!World)
    {
        return;
    }
    
    // Clear existing visualization
    ClearPointCloudVisualization();
    
    // Create new actor for point cloud
    AActor* NewActor = World->SpawnActor<AActor>();
    NewActor->SetActorLabel(TEXT("ProceduralPointCloud_Visualization"));
    PointCloudActor = NewActor;
    
    // Create procedural mesh component
    UProceduralMeshComponent* ProcMeshComp = NewObject<UProceduralMeshComponent>(NewActor);
    NewActor->SetRootComponent(ProcMeshComp);
    PointCloudComponent = ProcMeshComp;
    
    // Generate mesh data
    TArray<FVector> Vertices;
    TArray<int32> Triangles;
    TArray<FVector> Normals;
    TArray<FVector2D> UVs;
    TArray<FColor> VertexColors;
    
    GeneratePointCloudMesh(Vertices, Triangles, Normals, UVs, VertexColors);
    
    // Create the procedural mesh
    ProcMeshComp->CreateMeshSection(
        0,                          // Section index
        Vertices,                   // Vertices
        Triangles,                  // Triangles
        Normals,                    // Normals
        UVs,                        // UVs
        VertexColors,               // Vertex colors
        TArray<FProcMeshTangent>(), // Tangents (empty)
        true                        // Create collision
    );
    
    // Create and set a material that supports vertex colors
    CreateAndSetVertexColorMaterial(ProcMeshComp);
    
    // Configure component settings
    ProcMeshComp->bUseComplexAsSimpleCollision = false;
    ProcMeshComp->RegisterComponent();
    
    UE_LOG(LogTemp, Warning, TEXT("Created procedural point cloud mesh with %d vertices, %d triangles"), 
        Vertices.Num(), Triangles.Num() / 3);
}

void SVCCSimPanel::GeneratePointCloudMesh(TArray<FVector>& Vertices, 
                                         TArray<int32>& Triangles, 
                                         TArray<FVector>& Normals,
                                         TArray<FVector2D>& UVs,
                                         TArray<FColor>& VertexColors)
{
    // Reserve space for 24 vertices per point (4 vertices per face, 6 faces) and 36 indices per point (12 triangles)
    int32 PointCount = PointCloudData.Num();
    Vertices.Reserve(PointCount * 24);
    Triangles.Reserve(PointCount * 36);
    Normals.Reserve(PointCount * 24);
    UVs.Reserve(PointCount * 24);
    VertexColors.Reserve(PointCount * 24);
    
    int32 VertexIndex = 0;
    
    for (const FRatPoint& Point : PointCloudData)
    {
        FVector Center = Point.Position;
        FColor PointColor = Point.Color.ToFColor(false);
        
        // Create a small cube at each point position
        float HalfSize = PointSize * 0.5f;
        
        // Define 8 cube corner positions
        TArray<FVector> CubeCorners = {
            FVector(-HalfSize, -HalfSize, -HalfSize), // 0: Bottom-back-left
            FVector(HalfSize, -HalfSize, -HalfSize),  // 1: Bottom-back-right
            FVector(HalfSize, HalfSize, -HalfSize),   // 2: Bottom-front-right
            FVector(-HalfSize, HalfSize, -HalfSize),  // 3: Bottom-front-left
            FVector(-HalfSize, -HalfSize, HalfSize),  // 4: Top-back-left
            FVector(HalfSize, -HalfSize, HalfSize),   // 5: Top-back-right
            FVector(HalfSize, HalfSize, HalfSize),    // 6: Top-front-right
            FVector(-HalfSize, HalfSize, HalfSize)    // 7: Top-front-left
        };
        
        // Define each face with correct vertices, normals, and UVs
        struct FCubeFace
        {
            TArray<int32> VertexIndices;
            FVector Normal;
            TArray<FVector2D> FaceUVs;
        };
        
        TArray<FCubeFace> CubeFaces = {
            // Bottom face (Z-)
            {{0, 1, 2, 3}, FVector(0, 0, -1), {{0,0}, {1,0}, {1,1}, {0,1}}},
            // Top face (Z+)
            {{4, 7, 6, 5}, FVector(0, 0, 1), {{0,0}, {1,0}, {1,1}, {0,1}}},
            // Front face (Y+)
            {{3, 2, 6, 7}, FVector(0, 1, 0), {{0,0}, {1,0}, {1,1}, {0,1}}},
            // Back face (Y-)
            {{1, 0, 4, 5}, FVector(0, -1, 0), {{0,0}, {1,0}, {1,1}, {0,1}}},
            // Right face (X+)
            {{2, 1, 5, 6}, FVector(1, 0, 0), {{0,0}, {1,0}, {1,1}, {0,1}}},
            // Left face (X-)
            {{0, 3, 7, 4}, FVector(-1, 0, 0), {{0,0}, {1,0}, {1,1}, {0,1}}}
        };
        
        // Add vertices for each face (4 vertices per face)
        for (const FCubeFace& Face : CubeFaces)
        {
            for (int32 i = 0; i < 4; ++i)
            {
                int32 CornerIndex = Face.VertexIndices[i];
                Vertices.Add(Center + CubeCorners[CornerIndex]);
                Normals.Add(Face.Normal);
                UVs.Add(Face.FaceUVs[i]);
                VertexColors.Add(PointColor);
            }
        }
        
        // Add triangles for each face (2 triangles per face)
        for (int32 FaceIndex = 0; FaceIndex < 6; ++FaceIndex)
        {
            int32 FaceVertexStart = VertexIndex + (FaceIndex * 4);
            
            // First triangle: 0-1-2 (counter-clockwise)
            Triangles.Add(FaceVertexStart + 0);
            Triangles.Add(FaceVertexStart + 1);
            Triangles.Add(FaceVertexStart + 2);
            
            // Second triangle: 0-2-3 (counter-clockwise)
            Triangles.Add(FaceVertexStart + 0);
            Triangles.Add(FaceVertexStart + 2);
            Triangles.Add(FaceVertexStart + 3);
        }
        
        VertexIndex += 24; // 6 faces × 4 vertices per face
    }
    
    UE_LOG(LogTemp, Log, TEXT("Generated fixed cube mesh data: %d points "
                              "-> %d vertices, %d triangles"), 
        PointCount, Vertices.Num(), Triangles.Num() / 3);
}

void SVCCSimPanel::CreateAndSetVertexColorMaterial(UProceduralMeshComponent* ProcMeshComp)
{
    // Try to load a vertex color material first
    UMaterial* VertexColorMaterial = LoadObject<UMaterial>(nullptr, 
        TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
    
    if (!VertexColorMaterial)
    {
        // Fallback to default material
        VertexColorMaterial = UMaterial::GetDefaultMaterial(MD_Surface);
    }
    
    if (VertexColorMaterial)
    {
        // Create dynamic material instance
        UMaterialInstanceDynamic* DynamicMaterial =
            UMaterialInstanceDynamic::Create(VertexColorMaterial, nullptr);
        
        if (DynamicMaterial)
        {
            // Set parameters to use vertex colors
            DynamicMaterial->SetScalarParameterValue(TEXT("UseVertexColor"), 1.0f);
            DynamicMaterial->SetVectorParameterValue(TEXT("BaseColor"), FLinearColor::White);
            
            // Apply the material
            ProcMeshComp->SetMaterial(0, DynamicMaterial);
            
            UE_LOG(LogTemp, Log, TEXT("Applied vertex color material to procedural mesh"));
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("Failed to create dynamic material instance"));
        }
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("Could not load material for procedural mesh"));
    }
}

// Clear Point Cloud Visualization
void SVCCSimPanel::ClearPointCloudVisualization()
{
    if (PointCloudActor.IsValid())
    {
        PointCloudActor->Destroy();
        PointCloudActor.Reset();
    }
    PointCloudComponent.Reset();
}