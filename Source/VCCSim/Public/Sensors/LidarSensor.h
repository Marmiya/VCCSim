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
#include "GameFramework/Actor.h"
#include "SensorBase.h"
#include "DataStructures/PointCloud.h"
#include "LidarSensor.generated.h"

class UInsMeshHolder;

DECLARE_MULTICAST_DELEGATE_OneParam(FOnLidarPointCloudUpdated, const TArray<FVector>&);

namespace LidarDefaults
{
    constexpr int32 NumRays = 32;
    constexpr int32 NumPoints = 3200;
    constexpr double ScannerRangeInner = 300.0;
    constexpr double ScannerRangeOuter = 8000.0;
    constexpr double ScannerAngleUp = 30.0;
    constexpr double ScannerAngleDown = 30.0;
    constexpr bool bVisualizePoints = true;
}

class FLiDARConfig : public FSensorConfig
{
public:
    int32 NumRays = LidarDefaults::NumRays;
    int32 NumPoints = LidarDefaults::NumPoints;
    double ScannerRangeInner = LidarDefaults::ScannerRangeInner;
    double ScannerRangeOuter = LidarDefaults::ScannerRangeOuter;
    double ScannerAngleUp = LidarDefaults::ScannerAngleUp;
    double ScannerAngleDown = LidarDefaults::ScannerAngleDown;
    bool bVisualizePoints = LidarDefaults::bVisualizePoints;
};

UCLASS(ClassGroup = (VCCSIM), meta = (BlueprintSpawnableComponent))
class VCCSIM_API ULidarComponent : public USensorBaseComponent
{
    GENERATED_BODY()

public:
    ULidarComponent();
    virtual void Configure(const FSensorConfig& Config) override final;
    virtual ESensorType GetSensorType() const override { return ESensorType::Lidar; }

    UFUNCTION(BlueprintCallable, Category = "Lidar")
    void InitSensor();

    UFUNCTION(BlueprintCallable, Category = "Lidar")
    void VisualizePointCloud();

    TArray<FVector3f> GetPointCloudData();
    TPair<TArray<FVector3f>, FVCCSimOdom> GetPointCloudDataAndOdom();

    FOnLidarPointCloudUpdated OnPointCloudUpdated;

protected:
    virtual void BeginPlay() override;
    virtual void OnComponentCreated() override;
    TArray<FVector3f> PerformLineTraces(FVCCSimOdom* Odom = nullptr);

public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Lidar|Config")
    int32 NumRays = LidarDefaults::NumRays;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Lidar|Config")
    int32 NumPoints = LidarDefaults::NumPoints;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Lidar|Config")
    double ScannerRangeInner = LidarDefaults::ScannerRangeInner;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Lidar|Config")
    double ScannerRangeOuter = LidarDefaults::ScannerRangeOuter;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Lidar|Config")
    double ScannerAngleUp = LidarDefaults::ScannerAngleUp;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Lidar|Config")
    double ScannerAngleDown = LidarDefaults::ScannerAngleDown;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Lidar|Config")
    bool bVisualizePoints = LidarDefaults::bVisualizePoints;

    // Constructor properties
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Lidar|Debug")
    int ActualNumPoints = -1;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR|Performance")
    int32 ChunkSize = 256;

    UPROPERTY()
    UInsMeshHolder* MeshHolder;

private:
    TArray<FVector> LocalStartPoints;
    TArray<FVector> LocalEndPoints;
    TArray<FLiDARPoint> PointPool;

    FCollisionQueryParams QueryParams;

    int32 NumChunks = 0;
    TArray<int32> ChunkStartIndices;
    TArray<int32> ChunkEndIndices;

    void ProcessChunk(int32 ChunkIndex, const FTransform& WorldTransform);
    TArray<FTransform> GetHitTransforms() const;
};