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
#include "SafeZoneAnalyzer.generated.h"

class UProceduralMeshComponent;

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class USafeZoneAnalyzer : public UPrimitiveComponent
{
	GENERATED_BODY()
public:
	USafeZoneAnalyzer();
    
	UPROPERTY(EditAnywhere, Category = "SafeZoneAnalyzer")
	UProceduralMeshComponent* SafeZoneVisualizationMesh;
	UPROPERTY(EditAnywhere, Category = "SafeZoneAnalyzer")
	UMaterialInterface* SafeZoneMaterial;
	UPROPERTY(EditAnywhere, Category = "SafeZoneAnalyzer")
	FLinearColor SafeZoneColor = FLinearColor(1.0f, 0.47f, 0.47f);

	TArray<FBox> MeshSafeZones;
	TArray<FMeshInfo>* MeshInfos = nullptr;
    
	UFUNCTION(BlueprintCallable, Category = "SafeZoneAnalyzer")
	void InitializeSafeZoneVisualization();
	UFUNCTION(BlueprintCallable, Category = "SafeZoneAnalyzer")
	void VisualizeSafeZone(bool Vis);
	UFUNCTION(BlueprintCallable, Category = "SafeZoneAnalyzer")
	void ClearSafeZoneVisualization();
	void GenerateSafeZone(const float& SafeDistance);
	void CreateSafeZoneMesh();

private:
	bool bSafeZoneDirty = false;
};