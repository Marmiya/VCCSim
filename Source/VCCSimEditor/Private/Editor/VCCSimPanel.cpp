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
#include "Utils/VCCSimUIHelpers.h"
#include "Editor/Panels/VCCSimPanelPointCloud.h"
#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Editor/Panels/VCCSimPanelPathImageCapture.h"
#include "Editor/Panels/VCCSimPanelSceneAnalysis.h"
#include "Editor/Panels/VCCSimPanelRatSplatting.h"
#include "Editor/UnrealEd/Public/Selection.h"

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

SVCCSimPanel::~SVCCSimPanel()
{
    // Unregister from selection events
    if (GEditor && GEditor->GetSelectedActors())
    {
        GEditor->GetSelectedActors()->SelectionChangedEvent.RemoveAll(this);
    }
    
    // Clean up PathImageCapture manager
    if (PathImageCaptureManager.IsValid())
    {
        PathImageCaptureManager->Cleanup();
        PathImageCaptureManager.Reset();
    }

    // Clean up Scene Analysis manager
    if (SceneAnalysisManager.IsValid())
    {
        SceneAnalysisManager->Cleanup();
        SceneAnalysisManager.Reset();
    }

    // Clean up RatSplatting manager
    if (RatSplattingManager.IsValid())
    {
        RatSplattingManager->Cleanup();
        RatSplattingManager.Reset();
    }

    // Clean up Point Cloud manager
    if (PointCloudManager.IsValid())
    {
        PointCloudManager->Cleanup();
        PointCloudManager.Reset();
    }
    
    // Clean up Selection manager
    if (SelectionManager.IsValid())
    {
        SelectionManager->Cleanup();
        SelectionManager.Reset();
    }
}

// ============================================================================
// MAIN CONSTRUCTION
// ============================================================================

void SVCCSimPanel::Construct(const FArguments& InArgs)
{
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
    
    // Initialize Point Cloud manager
    PointCloudManager = MakeShared<FVCCSimPanelPointCloud>();
    PointCloudManager->Initialize();
    
    // Initialize Selection manager
    SelectionManager = MakeShared<FVCCSimPanelSelection>();
    SelectionManager->Initialize();
    
    // Initialize PathImageCapture manager
    PathImageCaptureManager = MakeShared<FVCCSimPanelPathImageCapture>();
    PathImageCaptureManager->Initialize();
    PathImageCaptureManager->SetSelectionManager(SelectionManager);
    
    // Initialize Scene Analysis manager
    SceneAnalysisManager = MakeShared<FVCCSimPanelSceneAnalysis>();
    SceneAnalysisManager->Initialize(SelectionManager);
    
    // Initialize RatSplatting manager
    RatSplattingManager = MakeShared<FVCCSimPanelRatSplatting>();
    RatSplattingManager->Initialize();
    
    // Create the main widget layout
    CreateMainLayout();
    
    // Auto-select FlashPawn if available in the scene (after UI is created)
    if (SelectionManager.IsValid())
    {
        SelectionManager->AutoSelectFlashPawn();
    }
}


void SVCCSimPanel::OnSelectionChanged(UObject* Object)
{
    // Delegate to Selection Manager
    if (SelectionManager.IsValid())
    {
        USelection* Selection = GEditor->GetSelectedActors();
        if (Selection && Selection->Num() > 0)
        {
            AActor* Actor = Cast<AActor>(Selection->GetSelectedObject(0));
            if (Actor)
            {
                SelectionManager->HandleActorSelection(Actor);
            }
        }
        return;
    }
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
                
                // Selection section (Flash Pawn, Camera, Target Object)
                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    SelectionManager.IsValid() ? SelectionManager->CreateSelectionPanel() : 
                    SNew(STextBlock).Text(FText::FromString("Selection Manager not initialized"))
                ]
                
                // Path Configuration & Image Capture panel
                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    PathImageCaptureManager.IsValid() ? PathImageCaptureManager->CreatePathImageCapturePanel() : 
                    SNew(STextBlock).Text(FText::FromString("PathImageCapture Manager not initialized"))
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

                // RatSplatting panel
                +SVerticalBox::Slot()
                .AutoHeight()
                [
                    CreateRatSplattingPanel()
                ]
            ]
        ]
    ];
}

// ============================================================================
// UI STYLING HELPERS
// ============================================================================


TSharedRef<SWidget> SVCCSimPanel::CreateLogoPanel()
{
    return FVCCSimUIHelpers::CreateSectionContent(
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

TSharedRef<SWidget> SVCCSimPanel::CreateSceneAnalysisPanel()
{
    if (SceneAnalysisManager.IsValid())
    {
        return SceneAnalysisManager->CreateSceneAnalysisPanel();
    }
    else
    {
        return SNew(STextBlock)
            .Text(FText::FromString("Scene Analysis Manager not initialized"));
    }
}

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

TSharedRef<SWidget> SVCCSimPanel::CreateRatSplattingPanel()
{
    if (RatSplattingManager.IsValid())
    {
        return RatSplattingManager->CreateRatSplattingPanel();
    }
    else
    {
        return SNew(STextBlock)
            .Text(FText::FromString("RatSplatting Manager not initialized"));
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
    return FVCCSimUIHelpers::CreatePropertyRow(
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

FString SVCCSimPanel::GetTimestampedFilename()
{
    FDateTime Now = FDateTime::Now();
    return FString::Printf(TEXT("%04d%02d%02d_%02d%02d%02d"),
        Now.GetYear(), Now.GetMonth(), Now.GetDay(),
        Now.GetHour(), Now.GetMinute(), Now.GetSecond());
}

// ============================================================================
// PANEL FACTORY IMPLEMENTATION
// ============================================================================

namespace FVCCSimPanelFactory
{
    const FName TabId = FName("VCCSimPanel");
    
    void RegisterTabSpawner(FTabManager& TabManager)
    {
        TabManager.RegisterTabSpawner(TabId, FOnSpawnTab::CreateLambda([](const FSpawnTabArgs& Args)
        {
            return SNew(SDockTab)
                .TabRole(ETabRole::PanelTab)
                [
                    SNew(SVCCSimPanel)
                ];
        }))
        .SetDisplayName(NSLOCTEXT("VCCSimEditor", "VCCSimPanelTabTitle", "VCCSim"))
        .SetMenuType(ETabSpawnerMenuType::Hidden)
        .SetIcon(FSlateIcon(FAppStyle::GetAppStyleSetName(), "LevelEditor.Tabs.Viewports"));
    }
}