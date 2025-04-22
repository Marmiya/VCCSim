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
#include "ComplexityAnalyzer.generated.h"

class UProceduralMeshComponent;

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

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class UComplexityAnalyzer : public UPrimitiveComponent
{
	GENERATED_BODY()
public:
	UComplexityAnalyzer();

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

	// Parent Data
	float* GridResolutionPtr = nullptr;
	TArray<FMeshInfo>* MeshInfos = nullptr;
	int32* TotalPointsInScenePtr = nullptr;
	int32* TotalTrianglesInScenePtr = nullptr;
	TMap<FIntVector, FUnifiedGridCell>* UnifiedGridPtr = nullptr;
	FVector* GridOriginPtr = nullptr;
	FVector* GridSizePtr = nullptr;
	bool* GridInitializedPtr = nullptr;

	FIntVector WorldToGridCoordinates(const FVector& WorldPos) const;

private:
	float CalculateCurvatureScore(const TArray<FVector>& Normals);
	float CalculateEdgeDensityScore(int32 EdgeCount, float CellVolume);
	float CalculateAngleVariationScore(const TArray<float>& Angles);
	FVector CalculateTriangleNormal(const FVector& V0, const FVector& V1, const FVector& V2);
	void CreateComplexityMesh();
	bool bComplexityVisualizationDirty = false;
};