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
#include "Math/Matrix.h"
#include "CameraData.generated.h"

/**
 * Camera intrinsics structure
 */
USTRUCT(BlueprintType)
struct VCCSIM_API FCameraIntrinsics
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
    int32 Width;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
    int32 Height;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
    float FocalX;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
    float FocalY;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
    float CenterX;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
    float CenterY;

    FCameraIntrinsics()
        : Width(1920)
        , Height(1080)
        , FocalX(0.0f)
        , FocalY(0.0f)
        , CenterX(0.0f)
        , CenterY(0.0f)
    {
    }
    
    FCameraIntrinsics(int32 InWidth, int32 InHeight, float InFocalX,
        float InFocalY, float InCenterX, float InCenterY)
        : Width(InWidth)
        , Height(InHeight)
        , FocalX(InFocalX)
        , FocalY(InFocalY)
        , CenterX(InCenterX)
        , CenterY(InCenterY)
    {
    }
};

/**
 * Camera information structure for right-handed coordinate system models
 * Used primarily for Triangle Splatting and 3D reconstruction
 */
USTRUCT(BlueprintType)
struct VCCSIM_API FCameraInfo
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
    int32 UID;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
    FMatrix RotationMatrix;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
    FVector Translation;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
    float FOVDegrees;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
    FString ImagePath;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
    FString ImageName;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
    int32 Width;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
    int32 Height;
    
    // Camera intrinsics
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
    float FocalX;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
    float FocalY;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
    float CenterX;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
    float CenterY;

    FCameraInfo()
        : UID(0)
        , RotationMatrix(FMatrix::Identity)
        , Translation(FVector::ZeroVector)
        , FOVDegrees(90.0f)
        , ImagePath(TEXT(""))
        , ImageName(TEXT(""))
        , Width(1920)
        , Height(1080)
        , FocalX(0.0f)
        , FocalY(0.0f)
        , CenterX(0.0f)
        , CenterY(0.0f)
    {
    }
};

/**
 * Pose data structure matching VCCSim format
 * Supports both Panel format (6 values) and Recorder format (7 values)
 */
USTRUCT(BlueprintType)
struct VCCSIM_API FVCCSimPoseData
{
    GENERATED_BODY()

public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pose")
    double Timestamp;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pose")
    FVector Location;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pose")
    FRotator Rotation;

    FVCCSimPoseData()
        : Timestamp(0.0)
        , Location(FVector::ZeroVector)
        , Rotation(FRotator::ZeroRotator)
    {
    }
    
    FVCCSimPoseData(double InTimestamp, const FVector& InLocation, const FRotator& InRotation)
        : Timestamp(InTimestamp)
        , Location(InLocation)
        , Rotation(InRotation)
    {
    }
};