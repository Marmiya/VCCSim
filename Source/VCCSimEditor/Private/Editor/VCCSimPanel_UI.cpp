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
#include "Components/InstancedStaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "DesktopPlatformModule.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SNumericEntryBox.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Layout/SSplitter.h"
#include "Widgets/Layout/SScrollBox.h"
#include "Widgets/Views/SListView.h"
#include "Widgets/Views/SHeaderRow.h"
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
// #include "Editor/VCCSimModuleFactory.h"  // Temporarily disabled due to UE 5.6 compatibility issues

// ============================================================================
// MAIN CONSTRUCTION - MODERN UI SYSTEM
// ============================================================================

void SVCCSimPanel::Construct(const FArguments& InArgs)
{
    InitializeStructuredConfigurations();
    LoadLogoImages();
    InitializeSceneAnalysisManager();
    CreateModernUIWidgets();
    CreateModernUILayout();

    if (GEditor && GEditor->GetSelectedActors())
    {
        GEditor->GetSelectedActors()->SelectionChangedEvent.AddRaw(this, &SVCCSimPanel::OnSelectionChanged);
    }
}

// ============================================================================
// MODERN UI LAYOUT - HORIZONTAL SPLIT WITH ORGANIZED PANELS
// ============================================================================

void SVCCSimPanel::CreateModernUILayout()
{
    ChildSlot
    [
        SNew(SVerticalBox)
        
        // Logo header
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5.0f)
        [
            CreateLogoPanel()
        ]
        
        // Main content with horizontal splitter
        + SVerticalBox::Slot()
        .FillHeight(1.0f)
        [
            SNew(SSplitter)
            .Orientation(Orient_Horizontal)
            
            // Left panel: Control and configuration
            + SSplitter::Slot()
            .Value(0.4f)
            [
                CreateModernControlPanel()
            ]
            
            // Right panel: Status display and details
            + SSplitter::Slot()
            .Value(0.6f)
            [
                CreateModernStatusPanel()
            ]
        ]
    ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateModernControlPanel()
{
    return SNew(SScrollBox)
        .Orientation(Orient_Vertical)
        
        + SScrollBox::Slot()
        .Padding(5.0f)
        [
            SNew(SVerticalBox)
            
            // Selection controls
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(0, 2)
            [
                CreateCollapsibleSection(TEXT("Selection"), CreatePawnTargetSelector(), bFlashPawnSectionExpanded)
            ]
            
            // Camera configuration using modern widget
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(0, 2)
            [
                CreateCollapsibleSection(TEXT("Camera Setup"), 
                    CreateCameraConfigurationWidget(), // Temporary placeholder
                    bCameraSectionExpanded)
            ]
            
            // Pose configuration using modern widget  
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(0, 2)
            [
                CreateCollapsibleSection(TEXT("Pose Generation"), CreatePoseConfigurationWidget(), bPoseConfigSectionExpanded)
            ]
            
            // Scene analysis controls
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(0, 2)
            [
                CreateCollapsibleSection(TEXT("Scene Analysis"), CreateSceneAnalysisWidget(), bSceneAnalysisSectionExpanded)
            ]
            
            // Point cloud controls
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(0, 2)
            [
                CreateCollapsibleSection(TEXT("Point Cloud"), CreatePointCloudPanel(), bPointCloudSectionExpanded)
            ]
            
            // Triangle Splatting using modern widget
            + SVerticalBox::Slot()
            .AutoHeight() 
            .Padding(0, 2)
            [
                CreateCollapsibleSection(TEXT("Triangle Splatting"), 
                    CreateTriangleSplattingWidget(),
                    bTriangleSplattingSectionExpanded)
            ]
        ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateModernStatusPanel()
{
    return SNew(SVerticalBox)
        
        // Real-time camera status display
        + SVerticalBox::Slot()
        .FillHeight(0.5f)
        .Padding(5.0f)
        [
            SNew(SBorder)
            .BorderImage(FAppStyle::GetBrush("ToolPanel.GroupBorder"))
            [
                SNew(SVerticalBox)
                
                + SVerticalBox::Slot()
                .AutoHeight()
                .Padding(5.0f)
                [
                    SNew(STextBlock)
                    .Text(FText::FromString(TEXT("Camera Status Monitor")))
                    .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
                ]
                
                + SVerticalBox::Slot()
                .FillHeight(1.0f)
                .Padding(5.0f)
                [
                    CreateCameraStatusListView()
                ]
            ]
        ]
        
        // Configuration details view
        + SVerticalBox::Slot()
        .FillHeight(0.5f)
        .Padding(5.0f)
        [
            SNew(SBorder)
            .BorderImage(FAppStyle::GetBrush("ToolPanel.GroupBorder"))
            [
                SNew(SVerticalBox)
                
                + SVerticalBox::Slot()
                .AutoHeight()
                .Padding(5.0f)
                [
                    SNew(STextBlock)
                    .Text(FText::FromString(TEXT("Configuration Details")))
                    .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
                ]
                
                + SVerticalBox::Slot()
                .FillHeight(1.0f)
                .Padding(5.0f)
                [
                    CreateModernDetailsView()
                ]
            ]
        ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateModernDetailsView()
{
    return SNew(SScrollBox)
        + SScrollBox::Slot()
        [
            SNew(SVerticalBox)
            
            // Current pose configuration display
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(0, 5)
            [
                SNew(SHorizontalBox)
                
                + SHorizontalBox::Slot()
                .AutoWidth()
                .VAlign(VAlign_Center)
                [
                    SNew(STextBlock)
                    .Text(FText::FromString(TEXT("Poses:")))
                    .MinDesiredWidth(80.0f)
                ]
                
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                .VAlign(VAlign_Center)
                [
                    SNew(STextBlock)
                    .Text_Lambda([this]() { 
                        return FText::FromString(FString::Printf(TEXT("%d (R: %.1f, Gap: %.1f)"), 
                            PoseConfig.NumPoses, PoseConfig.Radius, PoseConfig.VerticalGap));
                    })
                ]
            ]
            
            // Camera configuration display
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(0, 5)
            [
                SNew(SHorizontalBox)
                
                + SHorizontalBox::Slot()
                .AutoWidth()
                .VAlign(VAlign_Center)
                [
                    SNew(STextBlock)
                    .Text(FText::FromString(TEXT("Cameras:")))
                    .MinDesiredWidth(80.0f)
                ]
                
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                .VAlign(VAlign_Center)
                [
                    SNew(STextBlock)
                    .Text_Lambda([this]() { 
                        FString CameraStatus;
                        if (CameraConfig.bUseRGB) CameraStatus += TEXT("RGB ");
                        if (CameraConfig.bUseDepth) CameraStatus += TEXT("Depth ");
                        if (CameraConfig.bUseSegmentation) CameraStatus += TEXT("Seg ");
                        if (CameraConfig.bUseNormal) CameraStatus += TEXT("Normal ");
                        return FText::FromString(CameraStatus.IsEmpty() ? TEXT("None") : CameraStatus);
                    })
                ]
            ]
            
            // Triangle Splatting status
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(0, 5)
            [
                SNew(SHorizontalBox)
                
                + SHorizontalBox::Slot()
                .AutoWidth()
                .VAlign(VAlign_Center)
                [
                    SNew(STextBlock)
                    .Text(FText::FromString(TEXT("GS Status:")))
                    .MinDesiredWidth(80.0f)
                ]
                
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                .VAlign(VAlign_Center)
                [
                    SNew(STextBlock)
                    .Text_Lambda([this]() { 
                        return FText::FromString(bGSTrainingInProgress ? 
                            FString::Printf(TEXT("Training %.1f%%"), GSTrainingProgress * 100.0f) : 
                            GSTrainingStatusMessage);
                    })
                ]
            ]
            
            // Point cloud status
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(0, 5)
            [
                SNew(SHorizontalBox)
                
                + SHorizontalBox::Slot()
                .AutoWidth()
                .VAlign(VAlign_Center)
                [
                    SNew(STextBlock)
                    .Text(FText::FromString(TEXT("Point Cloud:")))
                    .MinDesiredWidth(80.0f)
                ]
                
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                .VAlign(VAlign_Center)
                [
                    SNew(STextBlock)
                    .Text_Lambda([this]() { 
                        if (!bPointCloudLoaded) return FText::FromString(TEXT("Not loaded"));
                        return FText::FromString(FString::Printf(TEXT("%d points (%s%s%s)"), 
                            PointCloudCount,
                            bPointCloudHasColors ? TEXT("Color ") : TEXT(""),
                            bPointCloudHasNormals ? TEXT("Normal ") : TEXT(""),
                            bShowNormals ? TEXT("Visible") : TEXT("Hidden")));
                    })
                ]
            ]
        ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateCameraStatusListView()
{
    TSharedPtr<SListView<TSharedPtr<FCameraStatusRow>>> CameraListView;
    
    // Initialize the camera status rows member variable
    if (!CameraStatusRows.IsValid())
    {
        CameraStatusRows = MakeShared<TArray<TSharedPtr<FCameraStatusRow>>>();
    }
    
    // Clear and populate with current data
    CameraStatusRows->Empty();
    CameraStatusRows->Add(MakeShareable(new FCameraStatusRow(TEXT("RGB Camera"), CameraConfig.bUseRGB, true)));
    CameraStatusRows->Add(MakeShareable(new FCameraStatusRow(TEXT("Depth Camera"), CameraConfig.bUseDepth, true)));
    CameraStatusRows->Add(MakeShareable(new FCameraStatusRow(TEXT("Segmentation"), CameraConfig.bUseSegmentation, true)));
    CameraStatusRows->Add(MakeShareable(new FCameraStatusRow(TEXT("Normal Camera"), CameraConfig.bUseNormal, true)));

    return SAssignNew(CameraListView, SListView<TSharedPtr<FCameraStatusRow>>)
        .ListItemsSource(CameraStatusRows.Get())
        .OnGenerateRow_Lambda([](TSharedPtr<FCameraStatusRow> Row, const TSharedRef<STableViewBase>& Table)
        {
            return SNew(STableRow<TSharedPtr<FCameraStatusRow>>, Table)
            [
                SNew(SHorizontalBox)
                
                + SHorizontalBox::Slot()
                .AutoWidth()
                .VAlign(VAlign_Center)
                .Padding(5, 2)
                [
                    SNew(SBox)
                    .WidthOverride(12)
                    .HeightOverride(12)
                    [
                        SNew(SColorBlock)
                        .Color(Row->StatusColor)
                    ]
                ]
                
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                .VAlign(VAlign_Center)
                .Padding(5, 2)
                [
                    SNew(STextBlock)
                    .Text(FText::FromString(Row->Name))
                ]
                
                + SHorizontalBox::Slot()
                .AutoWidth()
                .VAlign(VAlign_Center)
                .Padding(5, 2)
                [
                    SNew(STextBlock)
                    .Text(FText::FromString(Row->bIsActive ? TEXT("Active") : TEXT("Inactive")))
                    .ColorAndOpacity(Row->StatusColor)
                ]
            ];
        })
        .HeaderRow
        (
            SNew(SHeaderRow)
            
            + SHeaderRow::Column("Status")
            .DefaultLabel(FText::FromString(TEXT("Camera Systems")))
            .FillWidth(1.0f)
        );
}

// Modern UI widgets are initialized in VCCSimPanel.cpp
// Configuration change handlers are implemented in VCCSimPanel.cpp