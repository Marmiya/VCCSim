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
#include "Components/SplineComponent.h"
#include "Components/SplineMeshComponent.h"
#include "Containers/Deque.h"
#include "GameFramework/Actor.h"
#include "TrajectoryViewer.generated.h"

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class VCCSIM_API UTrajectoryViewer : public USceneComponent
{
    GENERATED_BODY()
    
public:
    UTrajectoryViewer();

    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    virtual void TickComponent(float DeltaTime, enum ELevelTick TickType,
       FActorComponentTickFunction* ThisTickFunction) override;

    static AActor* GenerateVisibleElements(
        UWorld* World,
        const TArray<FVector>& InPositions,
        const TArray<FRotator>& InRotations,
        float PathWidth = 5.0f,
        float ConeSize = 20.0f,
        float ConeLength = 30.0f);
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Trajectory")
    USplineComponent* SplineComponent;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Trajectory")
    float TraveledDistance = -0.1f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Trajectory")
    float DisplayDistance = -0.1f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Trajectory")
    float LastTraveledDistance = -250.0f;

    // Path visualization properties
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Path Visualization")
    UStaticMesh* PathMesh;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Path Visualization")
    UMaterialInterface* PathMaterial;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Path Visualization")
    float PathWidth = 0.05f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Path Visualization")
    float StepSize = 4.0f;
    
private:
    float TotalLength = -0.1f;
    
    TDeque<USplineMeshComponent*> PathMeshes;

    // Helper functions
    void UpdatePartialPath();
    void ClearSplineMeshes();
    FVector GetPointAtDistance(float Distance) const;
    
    // Material instance for dynamic color changes
    UPROPERTY()
    UMaterialInstanceDynamic* ObstacleMaterialInstance;
};