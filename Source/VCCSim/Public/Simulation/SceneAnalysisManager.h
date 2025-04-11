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
#include "ProceduralMeshComponent.h"
#include "SceneAnalysisManager.generated.h"

class URGBCameraComponent;

struct FCoverageData
{
    float CoveragePercentage;
    TSet<int32> VisibleMeshIDs;
    TArray<FVector> VisiblePoints;
    int32 TotalVisibleTriangles;
};

struct FUnifiedGridCell
{
    // Coverage data
    int32 TotalPoints;
    int32 VisiblePoints;
    float Coverage;
    
    // Complexity data
    float CurvatureScore;
    float EdgeDensityScore;
    float AngleVariationScore;
    float ComplexityScore; 
    
    FUnifiedGridCell() 
            : TotalPoints(0)
            , VisiblePoints(0)
            , Coverage(0.0f)
            , CurvatureScore(0.0f)
            , EdgeDensityScore(0.0f)
            , AngleVariationScore(0.0f)
            , ComplexityScore(0.0f)
    {}
};

UENUM(BlueprintType)
enum class ESceneComplexityPreset : uint8
{
    Generic         UMETA(DisplayName = "Generic Scene"),
    UrbanOutdoor    UMETA(DisplayName = "Urban Outdoor"),
    IndoorCluttered UMETA(DisplayName = "Indoor Cluttered"),
    NaturalTerrain  UMETA(DisplayName = "Natural Terrain"),
    MechanicalParts UMETA(DisplayName = "Mechanical Parts"),
    Custom          UMETA(DisplayName = "Custom Parameters")
};

UCLASS(BlueprintType, Blueprintable)
class VCCSIM_API ASceneAnalysisManager : public AActor
{
    GENERATED_BODY()

public:
    // Base functions
    ASceneAnalysisManager();
    bool Initialize(UWorld* InWorld, FString&& Path);
    void ScanScene();
    void ScanSceneRegion3D(float MinX, float MaxX, float MinY, float MaxY, float MinZ, float MaxZ);
    void RegisterCamera(URGBCameraComponent* CameraComponent);
    
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis")
    float GridResolution = 50.0f;
    
    /* ------------------------------- Coverage ----------------------------- */
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Coverage")
    UProceduralMeshComponent* CoverageVisualizationMesh;
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Coverage")
    UMaterialInterface* CoverageMaterial;
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Coverage")
    int SamplingDensity = 1;
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Coverage")
    bool bUseVertexSampling = true;
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Coverage", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float VisualizationThreshold = 0.f;
    
    float GetTotalCoveragePercentage() const { return CurrentCoveragePercentage;}
    void ResetCoverage();
    FCoverageData ComputeCoverage(const TArray<FTransform>& CameraTransforms, const FString& CameraName);
    FCoverageData ComputeCoverage(const FTransform& CameraTransform, const FString& CameraName);
    UFUNCTION(BlueprintCallable, Category = "SceneAnalysis|Coverage")
    void InitializeCoverageVisualization();
    UFUNCTION(BlueprintCallable, Category = "SceneAnalysis|Coverage")
    void VisualizeCoverage(bool bShow);
    UFUNCTION(BlueprintCallable, Category = "SceneAnalysis|Coverage")
    void ClearCoverageVisualization();
    void UpdateCoverageGrid();
    void CreateCoverageMesh();

    /* -------------------------- Safe zone -------------------------- */
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|SafeZone")
    UProceduralMeshComponent* SafeZoneVisualizationMesh;
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|SafeZone")
    UMaterialInterface* SafeZoneMaterial;
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|SafeZone")
    FLinearColor SafeZoneColor = FLinearColor(1.0f, 0.47f, 0.47f);

    TArray<FBox> MeshSafeZones;
    
    UFUNCTION(BlueprintCallable, Category = "SceneAnalysis|SafeZone")
    void InitializeSafeZoneVisualization();
    UFUNCTION(BlueprintCallable, Category = "SceneAnalysis|SafeZone")
    void VisualizeSafeZone(bool Vis);
    UFUNCTION(BlueprintCallable, Category = "SceneAnalysis|SafeZone")
    void ClearSafeZoneVisualization();
    void GenerateSafeZone(const float& SafeDistance);
    void CreateSafeZoneMesh();
    
    /* --------------------- Complexity --------------------------- */
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Complexity", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float CurvatureWeight = 0.4f;
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Complexity", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float EdgeDensityWeight = 0.3f;
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Complexity", meta = (ClampMin = "0.0", ClampMax = "1.0"))
    float AngleVariationWeight = 0.3f;
    // Curvature-specific parameters
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Complexity", meta = (ClampMin = "0.0"))
    float CurvatureSensitivity = 0.8f;  // Higher values make the system more sensitive to curvature changes
    // Edge density parameters
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Complexity", meta = (ClampMin = "0.0"))
    float EdgeDensityNormalizationFactor = 10.0f;  // Scales edge count relative to cell volume
    // Angle variation parameters
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Complexity", meta = (ClampMin = "0.0", ClampMax = "90.0"))
    float AngleVariationThreshold = 45.0f;  // Angle difference (in degrees) considered significant
    // Adaptive normalization toggle
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Complexity")
    bool bUseAdaptiveNormalization = true;  // Auto-adjust for scene characteristics
    // Scene-specific calibration
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Complexity")
    ESceneComplexityPreset SceneComplexityPreset = ESceneComplexityPreset::Generic;
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Complexity")
    UProceduralMeshComponent* ComplexityVisualizationMesh;
    UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Complexity")
    UMaterialInterface* ComplexityMaterial;
    
    void AnalyzeGeometricComplexity();
    TArray<FIntVector> GetHighComplexityRegions(float ComplexityThreshold = 0.7f);
    void ApplyComplexityPreset(ESceneComplexityPreset Preset);
    UFUNCTION(BlueprintCallable, Category = "SceneAnalysis|Complexity")
    void InitializeComplexityVisualization();
    UFUNCTION(BlueprintCallable, Category = "SceneAnalysis|Complexity")
    void VisualizeComplexity(bool bShow);
    UFUNCTION(BlueprintCallable, Category = "SceneAnalysis|Complexity")
    void ClearComplexityVisualization();

    /* -------------------------- Helper functions ------------------------- */
    FMeshInfo GetMeshInfo(int32 MeshID) const;
    TArray<FMeshInfo> GetAllMeshInfo() const;
    static void ExtractMeshData(UStaticMeshComponent* MeshComponent,
    FMeshInfo& OutMeshInfo);
    FIntVector WorldToGridCoordinates(const FVector& WorldPos) const;
    void InitializeUnifiedGrid();

    float CalculateCurvatureScore(const TArray<FVector>& Normals);
    float CalculateEdgeDensityScore(int32 EdgeCount, float CellVolume);
    float CalculateAngleVariationScore(const TArray<float>& Angles);
    FVector CalculateTriangleNormal(const FVector& V0, const FVector& V1, const FVector& V2);
    
    /* ----------------------------- Test ----------------------------- */
    FString LogPath;
    void ExportMeshesToPly();
    FString GeneratePlyContent(const FMeshInfo& MeshInfo);
    void VisualizeSceneMeshes(float Duration, bool bShowWireframe,
        bool bShowVertices, float VertexSize);
    void VisualizeSampledPoints(float Duration, float VertexSize);

private:
    void ScanSceneImpl(const TOptional<FBox>& RegionBounds);
    void ConstructFrustum(FConvexVolume& OutFrustum,
        const FTransform& CameraPose, const FMatrix44f& CameraIntrinsic);
    bool IsPointVisibleFromCamera(const FVector& Point,
        const FTransform& CameraPose) const;
    TArray<FVector> SamplePointsOnMesh(const FMeshInfo& MeshInfo);

private:
    UPROPERTY()
    UWorld* World;
    
    TMap<FString, FMatrix44f> CameraIntrinsics;
    
    TArray<FMeshInfo> SceneMeshes;
    int32 TotalPointsInScene;
    int32 TotalTrianglesInScene;
    
    TSet<int32> CurrentlyVisibleMeshIDs;

    TMap<FIntVector, FUnifiedGridCell> UnifiedGrid;
    FVector GridOrigin;
    FVector GridSize;
    bool bGridInitialized = false;
    
    // Coverage
    TMap<FVector, bool> CoverageMap;
    float CurrentCoveragePercentage;
    TArray<FVector> VisiblePoints;
    TArray<FVector> InvisiblePoints;
    bool bCoverageVisualizationDirty = false;

    // Safe zone
    UPROPERTY()
    FBox ExpandedSceneBounds;
    bool bSafeZoneDirty = false;
    TArray<FMeshInfo> SafetyEnvelopes;

    // Complexity Analysis
    void CreateComplexityMesh();
    bool bComplexityVisualizationDirty = false;
};