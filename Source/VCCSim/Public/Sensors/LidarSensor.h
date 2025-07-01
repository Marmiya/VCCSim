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
#include "Utils/InsMeshHolder.h"

#include "LidarSensor.generated.h"

class ARecorder;
struct FLiDARPoint;

class FLiDarConfig : public FSensorConfig
{
public:
    int32 NumRays = 32;
    int32 NumPoints = 3200;
    double ScannerRangeInner = 300;
    double ScannerRangeOuter = 8000;
    double ScannerAngleUp = 30;
    double ScannerAngleDown = 30;
    bool bVisualizePoints = true;
};

UCLASS(ClassGroup = (VCCSIM), meta = (BlueprintSpawnableComponent))
class VCCSIM_API ULidarComponent : public UPrimitiveComponent
{
    GENERATED_BODY()

public:
    
    ULidarComponent();
    void RConfigure(const FLiDarConfig& Config, ARecorder* Recorder);
    UFUNCTION()
    void SetRecordState(bool RState){ RecordState = RState; }

    UFUNCTION(BlueprintCallable, Category = "Lidar")
    void FirstCall();
    void InitSensor();

    UFUNCTION(BlueprintCallable, Category = "Lidar")
    void VisualizePointCloud();

    // For grpc server
    TArray<FVector3f> GetPointCloudData();
    TPair<TArray<FVector3f>, FVCCSimOdom> GetPointCloudDataAndOdom();

protected:
    
    virtual void BeginPlay() override;
    virtual void OnComponentCreated() override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType,
        FActorComponentTickFunction* ThisTickFunction) override;
    TArray<FVector3f> PerformLineTraces(FVCCSimOdom* Odom = nullptr);

public:

    // Properties can be set in the editor and config file.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Lidar|Config")
    int32 NumRays = 32;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Lidar|Config")
    int32 NumPoints = 3200;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Lidar|Config")
    double ScannerRangeInner = 300;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Lidar|Config")
    double ScannerRangeOuter = 8000;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Lidar|Config")
    double ScannerAngleUp = 30;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Lidar|Config")
    double ScannerAngleDown = 30;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Lidar|Config")
    bool bVisualizePoints = true;

    // Constructor properties
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Lidar|Debug")
    int ActualNumPoints = -1;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR|Performance")
    float UpdateThresholdDistance = 1.f;  // In centimeters
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR|Performance")
    float UpdateThresholdAngle = 0.5; // In degrees
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR|Performance")
    int32 ChunkSize = 256;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Lidar|Debug")
    bool bRecorded = false;

    UPROPERTY()
    UInsMeshHolder* MeshHolder;

private:
    static constexpr int32 CACHE_LINE_SIZE = 64;

    TArray<FVector> LocalStartPoints;
    TArray<FVector> LocalEndPoints;
    
    FVector LastLocation; 
    FRotator LastRotation;
    TArray<FVector> CachedStartPoints;
    TArray<FVector> CachedEndPoints;
    TArray<FLiDARPoint> PointPool;
    
    FCollisionQueryParams QueryParams;
    
    int32 NumChunks = 0;
    TArray<int32> ChunkStartIndices;
    TArray<int32> ChunkEndIndices;
    
    void ProcessChunk(int32 ChunkIndex);
    void UpdateCachedPoints(const FVector& NewLocation,
        const FRotator& NewRotation);
    bool ShouldUpdateCache(const FVector& NewLocation,
        const FRotator& NewRotation) const;
    TArray<FTransform> GetHitTransforms() const;

    UPROPERTY()
    AActor* ParentActor = nullptr;
    UPROPERTY()
    ARecorder* RecorderPtr = nullptr;
    float RecordInterval = -1.f;
    bool RecordState = false;
    float TimeSinceLastCapture = 0.f;
};