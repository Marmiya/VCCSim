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
#include "Editor/Panels/VCCSimPanelPointCloud.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "DesktopPlatformModule.h"
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
#include "DataStruct_IO/IOUtils.h"
#include "DataStruct_IO/PointCloudRenderer.h"

// ============================================================================
// MAIN CONSTRUCTION
// ============================================================================

void SVCCSimPanel::Construct(const FArguments& InArgs)
{
    // Initialize default values
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
    LoadLogoImages();
    
    // Initialize scene analysis manager
    InitializeSceneAnalysisManager();
    
    // Initialize Point Cloud manager
    PointCloudManager = MakeShared<FVCCSimPanelPointCloud>();
    PointCloudManager->Initialize();
    
    // Initialize Triangle Splatting manager
    InitializeGSManager();
    
    // Initialize COLMAP manager
    InitializeColmapManager();
    
    // Create the main widget layout
    CreateMainLayout();
    
    // Auto-select FlashPawn if available in the scene (after UI is created)
    AutoSelectFlashPawn();
}

void SVCCSimPanel::LoadLogoImages()
{
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

    if (FPaths::FileExists(SZULogoPath))
    {
        float SZUWidth = 80 * (272.0f / 80.0f);
        SZULogoBrush = MakeShareable(new FSlateDynamicImageBrush(
            FName(*SZULogoPath), 
            FVector2D(SZUWidth, 80),
            FColor(255, 255, 255, 255)));
    }
}

void SVCCSimPanel::InitializeSceneAnalysisManager()
{
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
}

void SVCCSimPanel::CreateMainLayout()
{
    ChildSlot
    [
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryTop"))
        .Padding(0)
        [
            SNew(SScrollBox)
            .ScrollBarAlwaysVisible(false)
            .AllowOverscroll(EAllowOverscroll::No)
            .ScrollBarThickness(FVector2D(8.0f, 8.0f))
            + SScrollBox::Slot()
            [
                SNew(SVerticalBox)
                
                // Logo panel
                +SVerticalBox::Slot()
                .AutoHeight()
                .Padding(0)
                [
                    CreateLogoPanel()
                ]
                
                // Flash Pawn section
                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    CreateCollapsibleSection("Flash Pawn", CreatePawnSelectPanel(), bFlashPawnSectionExpanded)
                ]
                
                // Camera section
                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    CreateCollapsibleSection("Camera Selection", CreateCameraSelectPanel(), bCameraSectionExpanded)
                ]
                
                // Target section
                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    CreateCollapsibleSection("Target Object", CreateTargetSelectPanel(), bTargetSectionExpanded)
                ]
                
                // Pose configuration
                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    CreateCollapsibleSection("Path Configuration", CreatePoseConfigPanel(), bPoseConfigSectionExpanded)
                ]
                
                // Capture panel
                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    CreateCollapsibleSection("Image Capture", CreateCapturePanel(), bCaptureSectionExpanded)
                ]

                // Scene analysis panel
                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    CreateCollapsibleSection("Scene Analysis", CreateSceneAnalysisPanel(), bSceneAnalysisSectionExpanded)
                ]

                // Point Cloud panel
                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    CreatePointCloudPanel()
                ]

                // Triangle Splatting panel
                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    CreateTriangleSplattingPanel()
                ]
            ]
        ]
    ];
}

// ============================================================================
// UI STYLING HELPERS
// ============================================================================

TSharedRef<SWidget> SVCCSimPanel::CreateSectionHeader(const FString& Title)
{
    return SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryTop"))
        .Padding(FMargin(10, 7))
        [
            SNew(STextBlock)
            .Text(FText::FromString(Title))
            .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
            .ColorAndOpacity(FColor(233, 233, 233))
            .TransformPolicy(ETextTransformPolicy::ToUpper)
        ];
}

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

TSharedRef<SWidget> SVCCSimPanel::CreateCollapsibleSection(
    const FString& Title, TSharedRef<SWidget> Content, bool& bExpanded)
{
    return SNew(SExpandableArea)
        .InitiallyCollapsed(!bExpanded)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryTop"))
        .BorderBackgroundColor(FColor(48, 48, 48))
        .OnAreaExpansionChanged_Lambda([&bExpanded](bool bIsExpanded) {
            bExpanded = bIsExpanded;
        })
        .HeaderContent()
        [
            SNew(STextBlock)
            .Text(FText::FromString(Title))
            .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
            .ColorAndOpacity(FColor(233, 233, 233))
            .TransformPolicy(ETextTransformPolicy::ToUpper)
        ]
        .BodyContent()
        [
            SNew(SBorder)
            .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
            .BorderBackgroundColor(FColor(5, 5, 5, 255))
            .Padding(FMargin(15, 6))
            [
                Content
            ]
        ];
}

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

TSharedRef<SWidget> SVCCSimPanel::CreateLogoPanel()
{
    return CreateSectionContent(
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
    );
}

// ============================================================================
// UI PANEL CREATION
// ============================================================================

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
                    CreateCameraStatusRow()
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

TSharedRef<SWidget> SVCCSimPanel::CreateCameraStatusRow()
{
    return SNew(SHorizontalBox)
    
    // RGB Camera
    +SHorizontalBox::Slot()
    .MaxWidth(100)
    .HAlign(HAlign_Center)
    .Padding(FMargin(0, 0, 2, 0))
    [
        CreateCameraStatusBox("RGB", 
            [this]() { return bHasRGBCamera; },
            [this]() { return bUseRGBCamera ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; },
            [this](ECheckBoxState NewState) { OnRGBCameraCheckboxChanged(NewState); })
    ]
    
    // Depth Camera
    +SHorizontalBox::Slot()
    .MaxWidth(100)
    .HAlign(HAlign_Center)
    .Padding(FMargin(0, 0, 2, 0))
    [
        CreateCameraStatusBox("Depth",
            [this]() { return bHasDepthCamera; },
            [this]() { return bUseDepthCamera ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; },
            [this](ECheckBoxState NewState) { OnDepthCameraCheckboxChanged(NewState); })
    ]
    
    // Normal Camera
    +SHorizontalBox::Slot()
    .MaxWidth(100)
    .HAlign(HAlign_Center)
    .Padding(FMargin(0, 0, 2, 0))
    [
        CreateCameraStatusBox("Normal",
            [this]() { return bHasNormalCamera; },
            [this]() { return bUseNormalCamera ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; },
            [this](ECheckBoxState NewState) { OnNormalCameraCheckboxChanged(NewState); })
    ]
    
    // Segmentation Camera
    +SHorizontalBox::Slot()
    .MaxWidth(120)
    .HAlign(HAlign_Center)
    .Padding(FMargin(0, 0, 2, 0))
    [
        CreateCameraStatusBox("Segmentation",
            [this]() { return bHasSegmentationCamera; },
            [this]() { return bUseSegmentationCamera ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; },
            [this](ECheckBoxState NewState) { OnSegmentationCameraCheckboxChanged(NewState); })
    ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateCameraStatusBox(
    const FString& CameraName,
    TFunction<bool()> HasCameraFunc,
    TFunction<ECheckBoxState()> CheckBoxStateFunc,
    TFunction<void(ECheckBoxState)> OnCheckBoxChangedFunc)
{
    return SNew(SBorder)
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
                .Image_Lambda([HasCameraFunc]() {
                    return HasCameraFunc() ? 
                        FAppStyle::GetBrush("Icons.Checkmark") : 
                        FAppStyle::GetBrush("Icons.X");
                })
                .ColorAndOpacity_Lambda([HasCameraFunc]() {
                    return HasCameraFunc() ? 
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
                SNew(SCheckBox)
                .IsChecked_Lambda([CheckBoxStateFunc]() { return CheckBoxStateFunc(); })
                .OnCheckStateChanged_Lambda([OnCheckBoxChangedFunc](ECheckBoxState NewState) { OnCheckBoxChangedFunc(NewState); })
                .IsEnabled_Lambda([HasCameraFunc]() { return HasCameraFunc(); })
            ]
            +SHorizontalBox::Slot()
            .AutoWidth()
            .HAlign(HAlign_Right)
            .VAlign(VAlign_Center)
            .Padding(0, 0, 4, 0)
            [
                SNew(STextBlock)
                .Text(FText::FromString(CameraName))
            ]
        ];
}

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
                +SHorizontalBox::Slot()
                .FillWidth(1.0f)
                .Padding(FMargin(0, 0, 8, 0))
                [
                    CreateNumericPropertyRowInt32("Pose Count", NumPosesSpinBox, NumPosesValue, NumPoses, 1, 1)
                ]
                +SHorizontalBox::Slot()
                .FillWidth(1.0f)
                [
                    CreateNumericPropertyRowFloat("Vertical Gap", VerticalGapSpinBox, VerticalGapValue, VerticalGap, 0.0f, 5.0f)
                ]
            ]
            
            +SVerticalBox::Slot()
            .MaxHeight(1)
            .Padding(FMargin(0, 0, 0, 0))
            [
                CreateSeparator()
            ]
            
            // Radius and Height Offset row
            +SVerticalBox::Slot()
            .AutoHeight()
            .Padding(FMargin(0, 4, 0, 4))
            [
                SNew(SHorizontalBox)
                +SHorizontalBox::Slot()
                .FillWidth(1.0f)
                .Padding(FMargin(0, 0, 8, 0))
                [
                    CreateNumericPropertyRowFloat("Radius", RadiusSpinBox, RadiusValue, Radius, 100.0f, 10.0f)
                ]
                +SHorizontalBox::Slot()
                .FillWidth(1.0f)
                [
                    CreateNumericPropertyRowFloat("Height Offset", HeightOffsetSpinBox, HeightOffsetValue, HeightOffset, 0.0f, 10.0f)
                ]
            ]
            
            +SVerticalBox::Slot()
            .MaxHeight(1)
            [
                CreateSeparator()
            ]
            
            // Load/Save Pose buttons
            +SVerticalBox::Slot()
            .AutoHeight()
            .Padding(FMargin(0, 4, 0, 4))
            [
                CreatePoseFileButtons()
            ]
            
            +SVerticalBox::Slot()
            .MaxHeight(1)
            [
                CreateSeparator()
            ]
            
            // Action buttons
            +SVerticalBox::Slot()
            .AutoHeight()
            .Padding(FMargin(0, 4, 0, 2))
            [
                CreatePoseActionButtons()
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
                CreateMovementButtons()
            ]
            
            +SVerticalBox::Slot()
            .MaxHeight(1)
            [
                CreateSeparator()
            ]
            
            +SVerticalBox::Slot()
            .AutoHeight()
            .Padding(0, 4, 0, 0)
            [
                CreateCaptureButtons()
            ]
        )
    ];
}

// ============================================================================
// SCENE ANALYSIS OPERATIONS
// ============================================================================

TSharedRef<SWidget> SVCCSimPanel::CreateSceneAnalysisPanel()
{
    return SNew(SVerticalBox)
    
    // Limited region controls
    +SVerticalBox::Slot()
    .AutoHeight()
    [
        CreateLimitedRegionControls()
    ]
    
    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        CreateSeparator()
    ]
    
    // Safe distance controls
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 4, 0, 4))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        .Padding(FMargin(0, 0, 8, 0))
        [
            CreateNumericPropertyRowFloat("Safe Distance", SafeDistanceSpinBox, SafeDistanceValue, SafeDistance, 0.0f, 10.0f)
        ]
        +SHorizontalBox::Slot()
        .FillWidth(1.0f)
        [
            CreateNumericPropertyRowFloat("Safe Height", SafeHeightSpinBox, SafeHeightValue, SafeHeight, 0.0f, 5.0f)
        ]
    ]
    
    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        CreateSeparator()
    ]

    // Scene operations
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(0, 4, 0, 4)
    [
        CreateSceneOperationButtons()
    ]

    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        CreateSeparator()
    ]
        
    // Safe zone operations
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(0, 4, 0, 4)
    [
        CreateSafeZoneButtons()
    ]

    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        CreateSeparator()
    ]
    
    // Coverage operations
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(0, 4, 0, 4)
    [
        CreateCoverageButtons()
    ]

    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        CreateSeparator()
    ]

    // Complexity operations
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(0, 4, 0, 0)
    [
        CreateComplexityButtons()
    ];
}

FReply SVCCSimPanel::OnToggleSafeZoneVisualizationClicked()
{
    if (!SceneAnalysisManager.IsValid())
    {
        return FReply::Handled();
    }
    
    bSafeZoneVisualized = !bSafeZoneVisualized;
    SceneAnalysisManager->InterfaceVisualizeSafeZone(bSafeZoneVisualized);

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
    
    bCoverageVisualized = !bCoverageVisualized;
    SceneAnalysisManager->InterfaceVisualizeCoverage(bCoverageVisualized);
    
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
    
    bComplexityVisualized = !bComplexityVisualized;
    SceneAnalysisManager->InterfaceVisualizeComplexity(bComplexityVisualized);
    
    VisualizeComplexityButton->SetButtonStyle(bComplexityVisualized ? 
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") : 
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
    
    return FReply::Handled();
}

// ============================================================================
// POINT CLOUD OPERATIONS
// ============================================================================

TSharedRef<SWidget> SVCCSimPanel::CreatePointCloudPanel()
{
    if (PointCloudManager.IsValid())
    {
        return PointCloudManager->CreatePointCloudPanel();
    }
    else
    {
        return SNew(STextBlock)
            .Text(FText::FromString("Point Cloud Manager not initialized"));
    }
}

// ============================================================================
// UI HELPER WIDGETS
// ============================================================================

template<typename T>
TSharedRef<SWidget> SVCCSimPanel::CreateNumericPropertyRow(
    const FString& Label,
    TSharedPtr<SNumericEntryBox<T>>& SpinBox,
    TOptional<T>& Value,
    T MinValue,
    T DeltaValue)
{
    return CreatePropertyRow(
        Label,
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(FColor(5,5, 5, 255))
        .Padding(4, 0)
        [
            SAssignNew(SpinBox, SNumericEntryBox<T>)
            .Value_Lambda([&Value]() { return Value; })
            .MinValue(MinValue)
            .Delta(DeltaValue)
            .AllowSpin(true)
            .OnValueChanged_Lambda([&Value](T NewValue) {
                Value = NewValue;
            })
        ]
    );
}

TSharedRef<SWidget> SVCCSimPanel::CreateSeparator()
{
    return SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(FColor(2, 2, 2))
        .Padding(0)
        .Content()
        [
            SNew(SBox)
            .HeightOverride(1.0f)
        ];
}

// ============================================================================
// UI BUTTON GROUPS
// ============================================================================

TSharedRef<SWidget> SVCCSimPanel::CreatePoseFileButtons()
{
    return SNew(SHorizontalBox)
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
    ];
}

TSharedRef<SWidget> SVCCSimPanel::CreatePoseActionButtons()
{
    return SNew(SHorizontalBox)
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
    ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateMovementButtons()
{
    return SNew(SHorizontalBox)
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
    ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateCaptureButtons()
{
    return SNew(SHorizontalBox)
    
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
                   ((bUseRGBCamera && bHasRGBCamera) || 
                    (bUseDepthCamera && bHasDepthCamera) || 
                    (bUseNormalCamera && bHasNormalCamera) ||
                    (bUseSegmentationCamera && bHasSegmentationCamera));
        })
    ]
    
    // Auto Capture button
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(4, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SAssignNew(AutoCaptureButton, SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Primary")
        .ContentPadding(FMargin(5, 2))
        .Text_Lambda([this]() {
            return bAutoCaptureInProgress ? 
                FText::FromString("Stop Capture") : 
                FText::FromString("Auto-Capture All Poses");
        })
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() {
            if (bAutoCaptureInProgress)
            {
                StopAutoCapture();
                AutoCaptureButton->SetButtonStyle(&FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
            }
            else
            {
                StartAutoCapture();
                AutoCaptureButton->SetButtonStyle(&FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger"));
            }
            return FReply::Handled();
        })
        .IsEnabled_Lambda([this]() {
            return SelectedFlashPawn.IsValid() && 
                   ((bUseRGBCamera && bHasRGBCamera) || 
                    (bUseDepthCamera && bHasDepthCamera) || 
                    (bUseSegmentationCamera && bHasSegmentationCamera) ||
                    (bUseNormalCamera && bHasNormalCamera));
        })
    ];
}



// ============================================================================
// SCENE ANALYSIS UI COMPONENTS
// ============================================================================

TSharedRef<SWidget> SVCCSimPanel::CreateLimitedRegionControls()
{
    return SNew(SVerticalBox)
    // X Range Controls
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
    
    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        CreateSeparator()
    ]
    
    // Y Range Controls  
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 4, 0, 4))
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
    
    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        CreateSeparator()
    ]

    // Z Range Controls
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 4, 0, 0))
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
    ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateSceneOperationButtons()
{
    return SNew(SHorizontalBox)
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
    ];
}


TSharedRef<SWidget> SVCCSimPanel::CreateSafeZoneButtons()
{
    return SNew(SHorizontalBox)
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
    ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateCoverageButtons()
{
    return SNew(SHorizontalBox)
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
    ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateComplexityButtons()
{
    return SNew(SHorizontalBox)
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
    .Padding(FMargin(0, 0, 0, 0))
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
    ];
}

// ============================================================================
// TEMPLATE SPECIALIZATIONS
// ============================================================================

// Simplified numeric property row creator that doesn't need specializations
TSharedRef<SWidget> SVCCSimPanel::CreateNumericPropertyRowInt32(
    const FString& Label,
    TSharedPtr<SNumericEntryBox<int32>>& SpinBox,
    TOptional<int32>& Value,
    int32& ActualVariable,
    int32 MinValue,
    int32 DeltaValue)
{
    return CreatePropertyRow(
        Label,
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(FColor(5,5, 5, 255))
        .Padding(4, 0)
        [
            SAssignNew(SpinBox, SNumericEntryBox<int32>)
            .Value_Lambda([&Value]() { return Value; })
            .MinValue(MinValue)
            .Delta(DeltaValue)
            .AllowSpin(true)
            .OnValueChanged_Lambda([&Value, &ActualVariable](int32 NewValue) {
                Value = NewValue;
                ActualVariable = NewValue;
            })
        ]
    );
}

TSharedRef<SWidget> SVCCSimPanel::CreateNumericPropertyRowFloat(
    const FString& Label,
    TSharedPtr<SNumericEntryBox<float>>& SpinBox,
    TOptional<float>& Value,
    float& ActualVariable,
    float MinValue,
    float DeltaValue)
{
    return CreatePropertyRow(
        Label,
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(FColor(5,5, 5, 255))
        .Padding(4, 0)
        [
            SAssignNew(SpinBox, SNumericEntryBox<float>)
            .Value_Lambda([&Value]() { return Value; })
            .MinValue(MinValue)
            .Delta(DeltaValue)
            .AllowSpin(true)
            .OnValueChanged_Lambda([&Value, &ActualVariable](float NewValue) {
                Value = NewValue;
                ActualVariable = NewValue;
            })
        ]
    );
}

// ============================================================================
// FACTORY REGISTRATION
// ============================================================================

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
    
    // Register the tab spawner
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








// ============================================================================
// AUTO SELECTION FUNCTIONALITY
// ============================================================================

void SVCCSimPanel::AutoSelectFlashPawn()
{
    // Clear any existing selection
    SelectedFlashPawn = nullptr;
    
    // Get the current editor world
    if (!GEditor)
    {
        return;
    }
    
    UWorld* World = GEditor->GetEditorWorldContext().World();
    if (!World)
    {
        return;
    }
    
    // Search for FlashPawn actors in the world
    AFlashPawn* FirstFoundFlashPawn = nullptr;
    for (TActorIterator<AFlashPawn> ActorIterator(World); ActorIterator; ++ActorIterator)
    {
        AFlashPawn* FlashPawn = *ActorIterator;
        if (FlashPawn && IsValid(FlashPawn))
        {
            FirstFoundFlashPawn = FlashPawn;
            break; // Select the first valid FlashPawn found
        }
    }
    
    // If we found a FlashPawn, select it
    if (FirstFoundFlashPawn)
    {
        SelectedFlashPawn = FirstFoundFlashPawn;
        
        // Update the UI text to show the selected FlashPawn
        if (SelectedFlashPawnText.IsValid())
        {
            SelectedFlashPawnText->SetText(FText::FromString(FirstFoundFlashPawn->GetActorLabel()));
        }
        
        // Check what camera components are available on this FlashPawn
        CheckCameraComponents();
        
        UE_LOG(LogTemp, Log, TEXT("Auto-selected FlashPawn: %s"), *FirstFoundFlashPawn->GetActorLabel());
    }
    else
    {
        // No FlashPawn found, ensure UI shows "None selected"
        if (SelectedFlashPawnText.IsValid())
        {
            SelectedFlashPawnText->SetText(FText::FromString("None selected"));
        }
        
        UE_LOG(LogTemp, Log, TEXT("No FlashPawn found in the scene for auto-selection"));
    }
}

