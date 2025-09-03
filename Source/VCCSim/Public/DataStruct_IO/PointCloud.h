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
#include "Engine/Engine.h"
#include "HAL/PlatformFilemanager.h"
#include "GenericPlatform/GenericPlatformFile.h"
#include "Misc/FileHelper.h"
#include "Math/Vector.h"
#include "PointCloud.generated.h"

/**
 * Enhanced point data structure for point cloud loading with normal support
 */
USTRUCT(BlueprintType)
struct VCCSIM_API FRatPoint
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Point")
    FVector Position;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Point")
    FLinearColor Color;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Point")
    FVector Normal;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Point")
    bool bHasNormal;

    FRatPoint()
        : Position(FVector::ZeroVector)
        , Color(FLinearColor::White)
        , Normal(FVector::UpVector)
        , bHasNormal(false)
    {
    }

    FRatPoint(const FVector& InPosition, const FLinearColor& InColor)
        : Position(InPosition)
        , Color(InColor)
        , Normal(FVector::UpVector)
        , bHasNormal(false)
    {
    }

    FRatPoint(const FVector& InPosition, const FLinearColor& InColor, const FVector& InNormal)
        : Position(InPosition)
        , Color(InColor)
        , Normal(InNormal)
        , bHasNormal(true)
    {
    }
};

/**
 * Point cloud data structure using FRatPoint
 * Provides unified point cloud representation for various modules
 */
USTRUCT(BlueprintType)
struct VCCSIM_API FPointCloudData
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "PointCloud")
    TArray<FRatPoint> Points;

    FPointCloudData()
    {
        Points.Empty();
    }
    
    /**
     * Add a point to the cloud
     */
    void AddPoint(const FVector& Position, const FLinearColor& Color = FLinearColor::White, 
                  const FVector& Normal = FVector::UpVector, bool bHasNormal = false)
    {
        FRatPoint NewPoint;
        NewPoint.Position = Position;
        NewPoint.Color = Color;
        NewPoint.Normal = Normal;
        NewPoint.bHasNormal = bHasNormal;
        Points.Add(NewPoint);
    }
    
    /**
     * Add a pre-constructed point
     */
    void AddPoint(const FRatPoint& Point)
    {
        Points.Add(Point);
    }
    
    /**
     * Get the number of points
     */
    int32 GetPointCount() const
    {
        return Points.Num();
    }
    
    /**
     * Clear all points
     */
    void Clear()
    {
        Points.Empty();
    }
    
    /**
     * Reserve memory for expected number of points
     */
    void Reserve(int32 Count)
    {
        Points.Reserve(Count);
    }
    
    /**
     * Check if the point cloud has normal data
     */
    bool HasNormals() const
    {
        for (const FRatPoint& Point : Points)
        {
            if (Point.bHasNormal)
            {
                return true;
            }
        }
        return false;
    }
    
    /**
     * Check if the point cloud contains non-default vertex colors
     * Treats pure white as the default (no color) value.
     */
    bool HasColors() const
    {
        const FLinearColor DefaultColor = FLinearColor::White;
        for (const FRatPoint& Point : Points)
        {
            if (!Point.Color.Equals(DefaultColor))
            {
                return true;
            }
        }
        return false;
    }
    
    /**
     * Get arrays of positions, colors and normals (for compatibility)
     */
    void GetArrays(TArray<FVector>& OutPositions, TArray<FLinearColor>& OutColors, 
                   TArray<FVector>& OutNormals) const
    {
        const int32 Count = Points.Num();
        OutPositions.Reserve(Count);
        OutColors.Reserve(Count);
        OutNormals.Reserve(Count);
        
        for (const FRatPoint& Point : Points)
        {
            OutPositions.Add(Point.Position);
            OutColors.Add(Point.Color);
            OutNormals.Add(Point.Normal);
        }
    }
    
    /**
     * Simplify point cloud by uniform sampling to avoid rendering performance issues
     * @param MaxPoints Maximum number of points to keep
     * @return New FPointCloudData with simplified points
     */
    FPointCloudData SimplifyPointCloud(int32 MaxPoints = 50000) const
    {
        if (Points.Num() <= MaxPoints)
        {
            // No simplification needed
            return *this;
        }
        
        FPointCloudData SimplifiedData;
        SimplifiedData.Reserve(MaxPoints);
        
        // Use uniform sampling to select points
        const int32 Step = FMath::Max(1, Points.Num() / MaxPoints);
        
        // Deterministic sampling with fixed step
        for (int32 i = 0; i < Points.Num() && SimplifiedData.GetPointCount() < MaxPoints; i += Step)
        {
            SimplifiedData.AddPoint(Points[i]);
        }
        
        UE_LOG(LogTemp, Warning, TEXT("Simplified point cloud from %d to %d points (%.1f%% reduction)"), 
               Points.Num(), SimplifiedData.GetPointCount(), 
               (1.0f - (float)SimplifiedData.GetPointCount() / (float)Points.Num()) * 100.0f);
        
        return SimplifiedData;
    }
};

USTRUCT(BlueprintType)
struct VCCSIM_API FLiDARPoint
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR")
    FVector Position;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR")
    float Intensity;

    bool bHit = false;
    
    FLiDARPoint()
        : Position(FVector::ZeroVector), Intensity(0.0f) {}

    FLiDARPoint(const FVector& InPosition, float InIntensity = 0.0f)
        : Position(InPosition), Intensity(InIntensity) {}
};

