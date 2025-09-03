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
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program. If not, see <https://www.gnu.org/licenses/>.
*/

#include "Editor/Panels/VCCSimPanelPointCloud.h"

// UE Core
#include "Engine/World.h"
#include "Engine/Engine.h"
#include "UObject/UObjectGlobals.h"
#include "Framework/Application/SlateApplication.h"
#include "IDesktopPlatform.h"
#include "DesktopPlatformModule.h"
#include "Framework/Notifications/NotificationManager.h"
#include "Widgets/Notifications/SNotificationList.h"

// Slate UI
#include "Widgets/Input/SButton.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Input/SCheckBox.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/Layout/SBorder.h"
#include "Styling/AppStyle.h"
#include "SlateOptMacros.h"

// Engine Components
#include "Components/InstancedStaticMeshComponent.h"
#include "ProceduralMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInterface.h"
#include "Materials/MaterialInstanceDynamic.h"

#include "DataStruct_IO/PointCloudRenderer.h"
#include "DataStruct_IO/IOUtils.h"

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

FVCCSimPanelPointCloud::FVCCSimPanelPointCloud()
{
    // Initialize default values
    bShowNormals = true;
    bShowColors = true;
}

FVCCSimPanelPointCloud::~FVCCSimPanelPointCloud()
{
    Cleanup();
}

// ============================================================================
// PUBLIC INTERFACE
// ============================================================================

void FVCCSimPanelPointCloud::Initialize()
{
    // Initialize any required resources
    UE_LOG(LogTemp, Log, TEXT("VCCSimPanelPointCloud initialized"));
}

void FVCCSimPanelPointCloud::Cleanup()
{
    // Clean up point cloud visualization
    ClearPointCloudVisualization();
    
    // Clear UI references
    LoadPointCloudButton.Reset();
    VisualizePointCloudButton.Reset();
    ShowNormalsCheckBox.Reset();
    ShowColorsCheckBox.Reset();
    PointCloudStatusText.Reset();
    PointCloudColorStatusText.Reset();
    PointCloudNormalStatusText.Reset();
}

TSharedRef<SWidget> FVCCSimPanelPointCloud::CreatePointCloudPanel()
{
    return SNew(SExpandableArea)
        .InitiallyCollapsed(!bPointCloudSectionExpanded)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryTop"))
        .BorderBackgroundColor(FColor(48, 48, 48))
        .OnAreaExpansionChanged_Lambda([this](bool bIsExpanded)
        {
            bPointCloudSectionExpanded = bIsExpanded;
        })
        .HeaderContent()
        [
            SNew(STextBlock)
            .Text(FText::FromString("Point Cloud Visualization"))
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
            SNew(SVerticalBox)

            // Status information
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5)
            [
                SNew(SHorizontalBox)

                // Point cloud status
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                [
                    SAssignNew(PointCloudStatusText, STextBlock)
                    .Text_Lambda([this]()
                    {
                        if (!bPointCloudLoaded)
                        {
                            return FText::FromString("No point cloud loaded");
                        }
                        return FText::FromString(FString::Printf(TEXT("Loaded: %d points from %s"), 
                            PointCloudCount, *FPaths::GetBaseFilename(LoadedPointCloudPath)));
                    })
                ]
            ]

            // Color and normal status
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5)
            [
                SNew(SHorizontalBox)

                + SHorizontalBox::Slot()
                .FillWidth(0.5f)
                [
                    SAssignNew(PointCloudColorStatusText, STextBlock)
                    .Text_Lambda([this]()
                    {
                        return FText::FromString(bPointCloudHasColors ? "Colors: Available" : "Colors: Not Available");
                    })
                ]

                + SHorizontalBox::Slot()
                .FillWidth(0.5f)
                [
                    SAssignNew(PointCloudNormalStatusText, STextBlock)
                    .Text_Lambda([this]()
                    {
                        return FText::FromString(bPointCloudHasNormals ? "Normals: Available" : "Normals: Not Available");
                    })
                ]
            ]

            // Control buttons
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5)
            [
                CreatePointCloudButtons()
            ]

            // Normal and color controls
            + SVerticalBox::Slot()
            .AutoHeight()
            .Padding(5)
            [
                CreatePointCloudNormalControls()
            ]
            ]
        ];
}

void FVCCSimPanelPointCloud::UpdateVisualization()
{
    if (bPointCloudLoaded && bPointCloudVisualized)
    {
        // Update existing visualization if needed
        if (UWorld* World = GEditor->GetEditorWorldContext().World())
        {
            CreateColoredPointCloudVisualization(World);
        }
    }
}

// ============================================================================
// UI CREATION
// ============================================================================

TSharedRef<SWidget> FVCCSimPanelPointCloud::CreatePointCloudButtons()
{
    return SNew(SHorizontalBox)

        // Load Point Cloud button
        + SHorizontalBox::Slot()
        .FillWidth(0.5f)
        .Padding(2)
        [
            SAssignNew(LoadPointCloudButton, SButton)
            .Text(FText::FromString("Load Point Cloud"))
            .OnClicked_Lambda([this]() { return OnLoadPointCloudClicked(); })
        ]

        // Visualize Point Cloud button
        + SHorizontalBox::Slot()
        .FillWidth(0.5f)
        .Padding(2)
        [
            SAssignNew(VisualizePointCloudButton, SButton)
            .Text_Lambda([this]()
            {
                return FText::FromString(bPointCloudVisualized ? "Hide Point Cloud" : "Show Point Cloud");
            })
            .IsEnabled_Lambda([this]()
            {
                return bPointCloudLoaded;
            })
            .OnClicked_Lambda([this]() { return OnTogglePointCloudVisualizationClicked(); })
        ];
}

TSharedRef<SWidget> FVCCSimPanelPointCloud::CreatePointCloudNormalControls()
{
    return SNew(SHorizontalBox)

        // Show Colors checkbox
        + SHorizontalBox::Slot()
        .FillWidth(0.5f)
        .Padding(2)
        [
            SAssignNew(ShowColorsCheckBox, SCheckBox)
            .Content()
            [
                SNew(STextBlock).Text(FText::FromString("Show Colors"))
            ]
            .IsChecked_Lambda([this]()
            {
                return bShowColors ? ECheckBoxState::Checked : ECheckBoxState::Unchecked;
            })
            .IsEnabled_Lambda([this]()
            {
                return bPointCloudLoaded && bPointCloudHasColors;
            })
            .OnCheckStateChanged_Lambda([this](ECheckBoxState NewState) { OnShowColorsChanged(NewState); })
        ]

        // Show Normals checkbox
        + SHorizontalBox::Slot()
        .FillWidth(0.5f)
        .Padding(2)
        [
            SAssignNew(ShowNormalsCheckBox, SCheckBox)
            .Content()
            [
                SNew(STextBlock).Text(FText::FromString("Show Normals"))
            ]
            .IsChecked_Lambda([this]()
            {
                return bShowNormals ? ECheckBoxState::Checked : ECheckBoxState::Unchecked;
            })
            .IsEnabled_Lambda([this]()
            {
                return bPointCloudLoaded && bPointCloudHasNormals;
            })
            .OnCheckStateChanged_Lambda([this](ECheckBoxState NewState) { OnShowNormalsChanged(NewState); })
        ];
}

// ============================================================================
// EVENT HANDLERS
// ============================================================================

FReply FVCCSimPanelPointCloud::OnLoadPointCloudClicked()
{
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (!DesktopPlatform)
    {
        return FReply::Handled();
    }

    TArray<FString> OpenedFiles;
    const bool bOpened = DesktopPlatform->OpenFileDialog(
        FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr),
        TEXT("Select PLY Point Cloud File"),
        TEXT(""),
        TEXT(""),
        TEXT("PLY Files (*.ply)|*.ply"),
        EFileDialogFlags::None,
        OpenedFiles
    );

    if (bOpened && OpenedFiles.Num() > 0)
    {
        const FString& FilePath = OpenedFiles[0];
        if (LoadPointCloudFromFile(FilePath))
        {
            LoadedPointCloudPath = FilePath;
            UE_LOG(LogTemp, Log, TEXT("Successfully loaded point cloud from: %s"), *FilePath);
        }
    }

    return FReply::Handled();
}

FReply FVCCSimPanelPointCloud::OnTogglePointCloudVisualizationClicked()
{
    if (!bPointCloudLoaded)
    {
        return FReply::Handled();
    }

    if (bPointCloudVisualized)
    {
        // Hide visualization
        ClearPointCloudVisualization();
        bPointCloudVisualized = false;
    }
    else
    {
        // Show visualization
        if (UWorld* World = GEditor->GetEditorWorldContext().World())
        {
            CreateColoredPointCloudVisualization(World);
            bPointCloudVisualized = true;
        }
    }

    return FReply::Handled();
}

void FVCCSimPanelPointCloud::OnShowNormalsChanged(ECheckBoxState NewState)
{
    bShowNormals = (NewState == ECheckBoxState::Checked);
    if (bPointCloudVisualized)
    {
        UpdateVisualization();
    }
}

void FVCCSimPanelPointCloud::OnShowColorsChanged(ECheckBoxState NewState)
{
    bShowColors = (NewState == ECheckBoxState::Checked);
    if (bPointCloudVisualized)
    {
        UpdateVisualization();
    }
}

// ============================================================================
// POINT CLOUD OPERATIONS
// ============================================================================

bool FVCCSimPanelPointCloud::LoadPointCloudFromFile(const FString& FilePath)
{
    // Clear existing data
    PointCloudData.Empty();
    bPointCloudLoaded = false;
    bPointCloudHasColors = false;
    bPointCloudHasNormals = false;
    PointCloudCount = 0;

    // Use VCCSim's PLY loader
    FPLYLoader::FPLYLoadResult LoadResult = FPLYLoader::LoadPLYFile(FilePath, FLinearColor::White);
    
    if (LoadResult.bSuccess && LoadResult.PointCount > 0)
    {
        // Copy data to our local storage
        PointCloudData = MoveTemp(LoadResult.Points);
        PointCloudCount = LoadResult.PointCount;
        bPointCloudHasColors = LoadResult.bHasColors;
        bPointCloudHasNormals = LoadResult.bHasNormals;
        bPointCloudLoaded = true;

        UE_LOG(LogTemp, Warning, TEXT("Loaded point cloud: %d points, Colors: %s, Normals: %s"), 
            PointCloudCount,
            bPointCloudHasColors ? TEXT("Yes") : TEXT("No"),
            bPointCloudHasNormals ? TEXT("Yes") : TEXT("No"));

        return true;
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to load point cloud from: %s"), *FilePath);
        return false;
    }
}

void FVCCSimPanelPointCloud::CreateSpherePointCloudVisualization()
{
    // Implementation will be moved from VCCSimPanel_UI.cpp
    UE_LOG(LogTemp, Log, TEXT("CreateSpherePointCloudVisualization - Implementation to be moved"));
}

void FVCCSimPanelPointCloud::CreateColoredPointCloudVisualization(UWorld* World)
{
    if (!World || PointCloudData.Num() == 0)
    {
        return;
    }

    UE_LOG(LogTemp, Log, TEXT("Creating colored point cloud visualization with %d points"), PointCloudData.Num());

    // Clear existing visualization
    ClearPointCloudVisualization();

    // Create point cloud data structure
    FPointCloudData PointCloudDataStruct;
    PointCloudDataStruct.Points = PointCloudData;

    // Spawn actor for point cloud rendering
    AActor* NewActor = World->SpawnActor<AActor>();
    if (NewActor)
    {
        NewActor->SetActorLabel(TEXT("VCCSim_PointCloud"));
        PointCloudActor = NewActor;

        // Add point cloud renderer component
        UPointCloudRenderer* PointCloudRenderer = NewObject<UPointCloudRenderer>(NewActor);
        if (PointCloudRenderer)
        {
            NewActor->AddInstanceComponent(PointCloudRenderer);
            PointCloudRenderer->AttachToComponent(NewActor->GetRootComponent(), 
                FAttachmentTransformRules::KeepWorldTransform);
            PointCloudRenderer->RegisterComponent();

            // Render the point cloud
            PointCloudRenderer->RenderPointCloud(PointCloudDataStruct, bShowColors, 1.0f);
            ParticlePointCloudRenderer = PointCloudRenderer;

            UE_LOG(LogTemp, Warning, TEXT("Created colored point cloud visualization using particle system: %d points"), 
                PointCloudData.Num());
        }
    }
}

void FVCCSimPanelPointCloud::ClearPointCloudVisualization()
{
    // Clear particle renderer
    if (ParticlePointCloudRenderer.IsValid())
    {
        ParticlePointCloudRenderer->ClearPointCloud();
        ParticlePointCloudRenderer.Reset();
    }

    // Clear instanced components
    if (PointCloudInstancedComponent.IsValid())
    {
        PointCloudInstancedComponent->ClearInstances();
        PointCloudInstancedComponent.Reset();
    }

    if (NormalLinesInstancedComponent.IsValid())
    {
        NormalLinesInstancedComponent->ClearInstances();
        NormalLinesInstancedComponent.Reset();
    }

    // Destroy actor
    if (PointCloudActor.IsValid())
    {
        PointCloudActor->Destroy();
        PointCloudActor.Reset();
    }

    UE_LOG(LogTemp, Log, TEXT("Cleared point cloud visualization"));
}

// ============================================================================
// MATERIAL AND RENDERING
// ============================================================================

void FVCCSimPanelPointCloud::SetupPointCloudMaterial(UInstancedStaticMeshComponent* MeshComponent)
{
    if (!MeshComponent)
    {
        return;
    }

    UMaterialInterface* Material = LoadPointCloudMaterial();
    if (Material)
    {
        MeshComponent->SetMaterial(0, Material);
        UE_LOG(LogTemp, Log, TEXT("Applied point cloud material"));
    }
    else
    {
        CreateSimplePointCloudMaterial(MeshComponent);
    }
}

UMaterialInterface* FVCCSimPanelPointCloud::LoadPointCloudMaterial()
{
    return LoadObject<UMaterialInterface>(nullptr, TEXT("/VCCSim/Materials/M_Point_Color"));
}

void FVCCSimPanelPointCloud::CreateSimplePointCloudMaterial(UInstancedStaticMeshComponent* MeshComponent)
{
    // Create a simple dynamic material for point cloud visualization
    UMaterialInterface* BaseMaterial = LoadObject<UMaterialInterface>(
        nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial"));
    
    if (BaseMaterial && MeshComponent)
    {
        UMaterialInstanceDynamic* DynamicMaterial = UMaterialInstanceDynamic::Create(BaseMaterial, MeshComponent);
        if (DynamicMaterial)
        {
            // Set default orange color for points
            DynamicMaterial->SetVectorParameterValue(TEXT("Color"), FLinearColor(1.0f, 0.5f, 0.0f, 1.0f));
            MeshComponent->SetMaterial(0, DynamicMaterial);
            UE_LOG(LogTemp, Log, TEXT("Created simple point cloud material"));
        }
    }
}

void FVCCSimPanelPointCloud::CreateBasicPointCloudMaterial(UProceduralMeshComponent* MeshComponent)
{
    if (!MeshComponent)
    {
        return;
    }

    UMaterialInterface* BaseMaterial = LoadObject<UMaterialInterface>(
        nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial"));
    
    if (BaseMaterial)
    {
        UMaterialInstanceDynamic* DynamicMaterial = UMaterialInstanceDynamic::Create(BaseMaterial, MeshComponent);
        if (DynamicMaterial)
        {
            DynamicMaterial->SetVectorParameterValue(TEXT("Color"), FLinearColor(1.0f, 0.5f, 0.0f, 1.0f));
            MeshComponent->SetMaterial(0, DynamicMaterial);
            UE_LOG(LogTemp, Log, TEXT("Applied basic point cloud material to procedural mesh"));
        }
    }
}