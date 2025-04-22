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
#include "CoverageAnalyzer.generated.h"

class UProceduralMeshComponent;

struct FCoverageData
{
	float CoveragePercentage;
	TSet<int32> VisibleMeshIDs;
	TArray<FVector> VisiblePoints;
	int32 TotalVisibleTriangles;
};

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class UCoverageAnalyzer : public UPrimitiveComponent
{
	GENERATED_BODY()
	
public:
	UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Coverage")
	UProceduralMeshComponent* CoverageVisualizationMesh = nullptr;
	UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Coverage")
	UMaterialInterface* CoverageMaterial = nullptr;
	UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Coverage")
	int SamplingDensity = 1;
	UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Coverage")
	bool bUseVertexSampling = true;
	UPROPERTY(EditAnywhere, Category = "SceneAnalysis|Coverage",
		meta = (ClampMin = "0.0", ClampMax = "1.0"))
	float VisualizationThreshold = 0.f;
    
	float GetTotalCoveragePercentage() const { return CurrentCoveragePercentage;}
	void ResetCoverage();
	void PrepareCoverage();
	FCoverageData ComputeCoverage(const TArray<FTransform>& CameraTransforms,
		const FString& CameraName);
	FCoverageData ComputeCoverage(const FTransform& CameraTransform,
		const FString& CameraName);
	UFUNCTION(BlueprintCallable, Category = "SceneAnalysis|Coverage")
	void InitializeCoverageVisualization();
	UFUNCTION(BlueprintCallable, Category = "SceneAnalysis|Coverage")
	void VisualizeCoverage(bool bShow);
	UFUNCTION(BlueprintCallable, Category = "SceneAnalysis|Coverage")
	void ClearCoverageVisualization();
	void UpdateCoverageGrid();
	void CreateCoverageMesh();

	// Parent Data
	float* GridResolutionPtr = nullptr;
	TMap<FString, FMatrix44f>* CameraIntrinsicsPtr;
	TArray<FMeshInfo>* MeshInfos = nullptr;
	bool* GridInitializedPtr = nullptr;
	TMap<FIntVector, FUnifiedGridCell>* UnifiedGridPtr = nullptr;
	FVector* GridOriginPtr = nullptr;
	FVector* GridSizePtr = nullptr;
	UPROPERTY()
	UWorld* World;

	/* ----------------------------- Debug Helper ---------------------------- */
	void VisualizeSampledPoints(float Duration, float VertexSize);
private:
	void ConstructFrustum(FConvexVolume& OutFrustum,
		const FTransform& CameraPose, const FMatrix44f& CameraIntrinsic);
	TArray<FVector> SamplePointsOnMesh(const FMeshInfo& MeshInfo);
	FIntVector WorldToGridCoordinates(const FVector& WorldPos) const;

	TSet<int32> CurrentlyVisibleMeshIDs;
	TMap<FVector, bool> CoverageMap;
	float CurrentCoveragePercentage = 0.f;
	TArray<FVector> VisiblePoints;
	TArray<FVector> InvisiblePoints;
	bool bCoverageVisualizationDirty = false;
};