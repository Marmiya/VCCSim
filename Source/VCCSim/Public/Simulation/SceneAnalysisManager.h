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

#pragma once

#include "CoreMinimal.h"
#include "DataType/DataMesh.h"
#include "SceneAnalysisManager.generated.h"

class URGBCameraComponent;
class USemanticAnalyzer;
class USafeZoneAnalyzer;
class UComplexityAnalyzer;
class UCoverageAnalyzer;

UCLASS(BlueprintType, Blueprintable)
class VCCSIM_API ASceneAnalysisManager : public AActor
{
    GENERATED_BODY()

public:
    ASceneAnalysisManager();
    bool Initialize(UWorld* InWorld, FString&& Path);
    void ScanScene();
    void ScanSceneRegion3D(float MinX, float MaxX, float MinY,
        float MaxY, float MinZ, float MaxZ);
    void RegisterCamera(URGBCameraComponent* CameraComponent);
    
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis")
    float GridResolution = 50.0f;

    /* -------------------------- Helper functions ------------------------- */
    FMeshInfo GetMeshInfo(int32 MeshID) const;
    TArray<FMeshInfo> GetAllMeshInfo() const;
    static void ExtractMeshData(UStaticMeshComponent* MeshComponent,
    FMeshInfo& OutMeshInfo);
    FIntVector WorldToGridCoordinates(const FVector& WorldPos) const;
    void InitializeUnifiedGrid();
    
    /* ----------------------------- SubModules ---------------------------- */
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|SubModules")
    USemanticAnalyzer* SemanticAnalyzer;
    bool InterfaceVisualizeSemanticAnalysis();
    
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|SubModules")
    USafeZoneAnalyzer* SafeZoneAnalyzer;
    void InterfaceVisualizeSafeZone(bool bShow);
    void InterfaceClearSafeZoneVisualization();
    void InterfaceInitializeSafeZoneVisualization();
    void InterfaceGenerateSafeZone(const float& SafeDistance);

    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|SubModules")
    UComplexityAnalyzer* ComplexityAnalyzer;
    void InterfaceVisualizeComplexity(bool bShow);
    void InterfaceClearComplexityVisualization();
    void InterfaceInitializeComplexityVisualization();
    void InterfaceAnalyzeGeometricComplexity();

    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|SubModules")
    UCoverageAnalyzer* CoverageAnalyzer;
    void InterfaceVisualizeCoverage(bool bShow);
    void InterfaceClearCoverageVisualization();
    void InterfaceInitializeCoverageVisualization();
    void InterfaceUpdateCoverageGrid();
    void InterfaceComputeCoverage(const TArray<FTransform>& CameraTransforms,
        const FString& CameraName);
        
    /* ------------------------------ Test --------------------------------- */
    FString LogPath;
    void ExportMeshesToPly();
    FString GeneratePlyContent(const FMeshInfo& MeshInfo);
    void VisualizeSceneMeshes(float Duration, bool bShowWireframe,
        bool bShowVertices, float VertexSize);

private:
    void ScanSceneImpl(const TOptional<FBox>& RegionBounds);

private:
    UPROPERTY()
    UWorld* World;
    
    TArray<FMeshInfo> SceneMeshes;
    int32 TotalPointsInScene;
    int32 TotalTrianglesInScene;
    
    TMap<FIntVector, FUnifiedGridCell> UnifiedGrid;
    FVector GridOrigin;
    FVector GridSize;
    bool bGridInitialized = false;

    TMap<FString, FMatrix44f> CameraIntrinsics;
};