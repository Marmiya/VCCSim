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

#pragma once

#include "CoreMinimal.h"
#include "Widgets/SCompoundWidget.h"
#include "Widgets/SWidget.h"
#include "DataStruct_IO/PointCloud.h"
#include "Engine/World.h"

class SButton;
class STextBlock;
class SCheckBox;
class UInstancedStaticMeshComponent;
class UProceduralMeshComponent;
class UPointCloudRenderer;
class UMaterialInterface;

/**
 * Point Cloud panel functionality manager
 * Handles all point cloud loading, visualization, and UI management
 */
class VCCSIMEDITOR_API FVCCSimPanelPointCloud
{
public:
    FVCCSimPanelPointCloud();
    ~FVCCSimPanelPointCloud();

    // ============================================================================
    // PUBLIC INTERFACE
    // ============================================================================
    
    /** Initialize the point cloud manager */
    void Initialize();
    
    /** Cleanup resources */
    void Cleanup();
    
    /** Create the point cloud UI panel */
    TSharedRef<SWidget> CreatePointCloudPanel();
    
    /** Update point cloud visualization state */
    void UpdateVisualization();
    
    /** Get current point cloud count */
    int32 GetPointCloudCount() const { return PointCloudCount; }
    
    /** Check if point cloud is loaded */
    bool IsPointCloudLoaded() const { return bPointCloudLoaded; }
    
    /** Check if point cloud is visualized */
    bool IsPointCloudVisualized() const { return bPointCloudVisualized; }

private:
    // ============================================================================
    // UI CREATION
    // ============================================================================
    
    /** Create point cloud control buttons */
    TSharedRef<SWidget> CreatePointCloudButtons();
    
    /** Create point cloud normal/color controls */
    TSharedRef<SWidget> CreatePointCloudNormalControls();
    
    /** Create a visual separator for UI sections */
    TSharedRef<SWidget> CreateSeparator();
    
    // ============================================================================
    // EVENT HANDLERS
    // ============================================================================
    
    /** Handle load point cloud button click */
    FReply OnLoadPointCloudClicked();
    
    /** Handle toggle point cloud visualization button click */
    FReply OnTogglePointCloudVisualizationClicked();
    
    /** Handle clear point cloud button click */
    FReply OnClearPointCloudClicked();
    
    /** Handle show normals checkbox change */
    void OnShowNormalsChanged(ECheckBoxState NewState);
    
    /** Handle show colors checkbox change */
    void OnShowColorsChanged(ECheckBoxState NewState);
    
    // ============================================================================
    // POINT CLOUD OPERATIONS
    // ============================================================================
    
    /** Load point cloud from PLY file */
    bool LoadPointCloudFromFile(const FString& FilePath);
    
    /** Create colored point cloud visualization using particle system */
    void CreateColoredPointCloudVisualization(UWorld* World);
    
    /** Create normals visualization using instanced cylinders */
    void CreateNormalVisualization(AActor* Owner);
    
    /** Clear all point cloud visualizations */
    void ClearPointCloudVisualization();
    
    // ============================================================================
    // MATERIAL AND RENDERING
    // ============================================================================
    
    /** Setup point cloud material for instanced mesh component */
    void SetupPointCloudMaterial(UInstancedStaticMeshComponent* MeshComponent);
    
    /** Load point cloud material */
    UMaterialInterface* LoadPointCloudMaterial();
    
    /** Create simple point cloud material */
    void CreateSimplePointCloudMaterial(UInstancedStaticMeshComponent* MeshComponent);
    
    /** Create basic point cloud material for procedural mesh */
    void CreateBasicPointCloudMaterial(UProceduralMeshComponent* MeshComponent);

private:
    // ============================================================================
    // UI ELEMENTS
    // ============================================================================
    
    TSharedPtr<SButton> LoadPointCloudButton;
    TSharedPtr<SButton> VisualizePointCloudButton;
    TSharedPtr<SCheckBox> ShowNormalsCheckBox;
    TSharedPtr<SCheckBox> ShowColorsCheckBox;
    TSharedPtr<STextBlock> PointCloudStatusText;
    TSharedPtr<STextBlock> PointCloudColorStatusText;
    TSharedPtr<STextBlock> PointCloudNormalStatusText;
    
    // ============================================================================
    // POINT CLOUD STATE
    // ============================================================================
    
    /** Point cloud data */
    TArray<FRatPoint> PointCloudData;
    
    /** Point cloud actor reference */
    TWeakObjectPtr<AActor> PointCloudActor;
    
    /** Instanced static mesh component for point visualization */
    TWeakObjectPtr<UInstancedStaticMeshComponent> PointCloudInstancedComponent;
    
    /** Normal lines mesh component */
    TWeakObjectPtr<UInstancedStaticMeshComponent> NormalLinesInstancedComponent;
    
    /** Particle-based point cloud renderer */
    TWeakObjectPtr<UPointCloudRenderer> ParticlePointCloudRenderer;
    
    /** Point cloud state flags */
    bool bPointCloudVisualized = false;
    bool bPointCloudLoaded = false;
    bool bPointCloudHasColors = false;
    bool bPointCloudHasNormals = false;
    bool bShowNormals = false;
    bool bShowColors = false;
    
    /** Point cloud file information */
    FString LoadedPointCloudPath;
    int32 PointCloudCount = 0;
    
    /** UI section expansion state */
    bool bPointCloudSectionExpanded = false;
};