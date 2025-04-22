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
#include "SemanticAnalyzer.generated.h"

class UProceduralMeshComponent;
struct FProcMeshTangent;

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class USemanticAnalyzer : public UPrimitiveComponent
{
	GENERATED_BODY()
public:
	USemanticAnalyzer();
	virtual void TickComponent(float DeltaTime, enum ELevelTick TickType,
		FActorComponentTickFunction* ThisTickFunction) override;
	void ShowSemanticAnalysis();

	UPROPERTY(VisibleAnywhere)
	TObjectPtr<APawn> CenterCharacter = nullptr;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float DisplayDistance = 1000.0f;
	UPROPERTY(VisibleAnywhere)
	FVector LastPosition = {100000.0f, 100000.0f, 100000.0f};
	UPROPERTY(EditAnywhere)
	bool bShowSemanticAnalysis = false;

private:
	UPROPERTY()
	TArray<USceneComponent*> VisualizationComponents;
    
	void ClearVisualization();
	void CreateVisualizationForActor(AActor* Actor);
	void CreateBoundingBoxMesh(UProceduralMeshComponent* MeshComponent, const FVector& Origin, const FVector& Extent);
	void CreateBoxEdges(TArray<FVector>& Vertices, TArray<int32>& Triangles, 
						TArray<FVector>& Normals, TArray<FVector2D>& UV0, 
						TArray<FColor>& VertexColors, TArray<FProcMeshTangent>& Tangents);

	UPROPERTY()
	float TickInterval = 2.f;
	UPROPERTY()
	float TimeSinceLastUpdate = 0.f;
};
