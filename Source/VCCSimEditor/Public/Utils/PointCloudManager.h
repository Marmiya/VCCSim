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
#include "DataStructures/PointCloud.h"

class UInstancedStaticMeshComponent;
class UProceduralMeshComponent;
class UPointCloudRenderer;
class UMaterialInterface;

class VCCSIMEDITOR_API FPointCloudManager
{
public:
    FPointCloudManager();
    ~FPointCloudManager();

    bool LoadFromFile(const FString& FilePath);

    void ShowVisualization(UWorld* World, bool bWithColors, bool bWithNormals);
    void UpdateNormalVisibility(bool bShow);
    void ClearVisualization();
    void ClearData();

    bool IsLoaded() const { return bPointCloudLoaded; }
    bool IsVisualized() const { return bPointCloudVisualized; }
    bool HasColors() const { return bPointCloudHasColors; }
    bool HasNormals() const { return bPointCloudHasNormals; }
    int32 GetPointCount() const { return PointCloudCount; }
    const FString& GetLoadedPath() const { return LoadedPointCloudPath; }
    AActor* GetPointCloudActor() const { return PointCloudActor.Get(); }

private:
    void CreateColoredPointCloudVisualization(UWorld* World, bool bWithColors, bool bWithNormals);
    void CreateNormalVisualization(AActor* Owner);
    void SetupPointCloudMaterial(UInstancedStaticMeshComponent* MeshComponent);
    UMaterialInterface* LoadPointCloudMaterial();
    void CreateSimplePointCloudMaterial(UInstancedStaticMeshComponent* MeshComponent);
    void CreateBasicPointCloudMaterial(UProceduralMeshComponent* MeshComponent);

    TArray<FRatPoint> PointCloudData;
    TWeakObjectPtr<AActor> PointCloudActor;
    TWeakObjectPtr<UInstancedStaticMeshComponent> PointCloudInstancedComponent;
    TWeakObjectPtr<UInstancedStaticMeshComponent> NormalLinesInstancedComponent;
    TWeakObjectPtr<UPointCloudRenderer> ParticlePointCloudRenderer;

    bool bPointCloudVisualized = false;
    bool bPointCloudLoaded = false;
    bool bPointCloudHasColors = false;
    bool bPointCloudHasNormals = false;
    FString LoadedPointCloudPath;
    int32 PointCloudCount = 0;
};
