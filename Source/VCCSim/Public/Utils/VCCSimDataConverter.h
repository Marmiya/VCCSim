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
#include "Engine/StaticMesh.h"
#include "Math/Matrix.h"
#include "VCCSimDataConverter.generated.h"

/**
 * Camera information structure for right-handed coordinate system models
 */
USTRUCT()
struct VCCSIM_API FCameraInfo
{
    GENERATED_BODY()

public:
    int32 UID;
    FMatrix RotationMatrix;
    FVector Translation;
    float FOVDegrees;
    FString ImagePath;
    FString ImageName;
    int32 Width;
    int32 Height;
    
    // Camera intrinsics
    float FocalX;
    float FocalY;
    float CenterX;
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
 * Point cloud data structure
 */
USTRUCT()
struct VCCSIM_API FPointCloudData
{
    GENERATED_BODY()

public:
    TArray<FVector> Points;
    TArray<FLinearColor> Colors;
    TArray<FVector> Normals;

    FPointCloudData()
    {
        Points.Empty();
        Colors.Empty();
        Normals.Empty();
    }
    
    void AddPoint(const FVector& Point, const FLinearColor& Color =
        FLinearColor::White, const FVector& Normal = FVector::UpVector)
    {
        Points.Add(Point);
        Colors.Add(Color);
        Normals.Add(Normal);
    }
    
    int32 GetPointCount() const
    {
        return Points.Num();
    }
};

/**
 * Camera intrinsics structure
 */
USTRUCT()
struct VCCSIM_API FCameraIntrinsics
{
    GENERATED_BODY()

public:
    int32 Width;
    int32 Height;
    float FocalX;
    float FocalY;
    float CenterX;
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
 * Pose data structure matching VCCSim format
 */
USTRUCT()
struct VCCSIM_API FVCCSimPoseData
{
    GENERATED_BODY()

public:
    double Timestamp;
    FVector Location;
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


class VCCSIM_API FVCCSimDataConverter
{
public:
    // ============================================================================
    // POSE FILE CONVERSION
    // ============================================================================
    
    /**
     * Parse VCCSim pose file and convert to camera information array
     * @param PoseFilePath Path to the pose.txt file
     * @param ImageDirectory Directory containing the images
     * @param Intrinsics Camera intrinsic parameters
     * @return Array of camera information structures
     */
    static TArray<FCameraInfo> ConvertPoseFile(
        const FString& PoseFilePath, 
        const FString& ImageDirectory,
        const FCameraIntrinsics& Intrinsics);
    
    /**
     * Parse individual pose line from VCCSim format
     * @param Line Text line from pose file
     * @param bIsRecorderFormat True if using Recorder format (7 values), false for Panel format (6 values)
     * @return Parsed pose data, or invalid data if parsing failed
     */
    static FVCCSimPoseData ParsePoseLine(const FString& Line, bool& bIsRecorderFormat);
    
    /**
     * Determine pose file format by analyzing content
     * @param PoseFilePath Path to pose file
     * @return True if Recorder format (7 values), false if Panel format (6 values)
     */
    static bool DeterminePoseFileFormat(const FString& PoseFilePath);

    // ============================================================================
    // COORDINATE SYSTEM CONVERSION
    // ============================================================================
    
    /**
     * Convert UE location to right-handed coordinates
     * @param UELocation Location in UE coordinates (cm, left-handed, Z-up)
     * @return Location in right-handed coordinates (m, right-handed, Z-up)
     */
    static FVector ConvertLocation(const FVector& UELocation);
    
    /**
     * Convert UE rotation to right-handed rotation
     * @param UERotation Rotation in UE coordinates
     * @return Rotation matrix for the right-handed coordinate system
     */
    static FMatrix ConvertRotation(const FRotator& UERotation);

    // ============================================================================
    // CAMERA PARAMETER CONVERSION
    // ============================================================================
    
    /**
     * Convert UE camera parameters to right-handed coordinate system format
     * @param FOVDegrees Field of view in degrees
     * @param Width Image width in pixels
     * @param Height Image height in pixels
     * @return Camera intrinsics structure
     */
    static FCameraIntrinsics ConvertCameraParams(float FOVDegrees, int32 Width, int32 Height);
    
    /**
     * Calculate focal length from FOV and image dimensions
     * @param FOVDegrees Field of view in degrees
     * @param ImageDimension Width or height in pixels
     * @return Focal length in pixels
     */
    static float CalculateFocalLength(float FOVDegrees, int32 ImageDimension);

    // ============================================================================
    // MESH PROCESSING
    // ============================================================================
    
    /**
     * Convert UE Static Mesh to point cloud for initialization
     * @param Mesh Static mesh to convert
     * @param SampleCount Number of points to sample from the mesh
     * @param bApplyCoordinateTransformation If true, apply UE to right-handed coordinate transformation
     * @return Point cloud data structure
     */
    static FPointCloudData ConvertMeshToPointCloud(
        UStaticMesh* Mesh, 
        int32 SampleCount = 10000, 
        bool bApplyCoordinateTransformation = true);
    
    /**
     * Generate random point cloud for initialization when no mesh is provided
     * @param CameraInfos Camera information array to determine scene bounds
     * @param PointCount Number of points to generate
     * @return Point cloud data structure
     */
    static FPointCloudData GenerateRandomPointCloud(
        const TArray<FCameraInfo>& CameraInfos, int32 PointCount = 10000);
    
    // ============================================================================
    // FILE I/O OPERATIONS
    // ============================================================================
    
    /**
     * Save camera information to right-handed coordinate system compatible format
     * @param CameraInfos Array of camera information
     * @param OutputPath Output directory path
     * @return True if successful
     */
    static bool SaveCameraInfo(const TArray<FCameraInfo>& CameraInfos,
        const FString& OutputPath);
    
    /**
     * Save point cloud data to PLY format
     * @param PointCloudData Point cloud data to save
     * @param OutputFilePath Output PLY file path
     * @return True if successful
     */
    static bool SavePointCloudToPLY(const FPointCloudData& PointCloudData,
        const FString& OutputFilePath);
    
    /**
     * Create directory structure for right-handed coordinate system models
     * @param OutputPath Base output directory
     * @return True if successful
     */
    static bool CreateModelDirectoryStructure(const FString& OutputPath);

    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================
    
    /**
     * Validate image directory and return list of valid images
     * @param ImageDirectory Directory to validate
     * @param ValidExtensions Array of valid image extensions (e.g., "jpg", "png")
     * @return Array of valid image file paths
     */
    static TArray<FString> ValidateImageDirectory(const FString& ImageDirectory,
        const TArray<FString>& ValidExtensions = {"jpg", "png", "jpeg"});
    
    /**
     * Generate image file name for given pose index
     * @param PoseIndex Index of the pose
     * @param Extension File extension (without dot)
     * @return Generated file name
     */
    static FString GenerateImageFileName(int32 PoseIndex,
        const FString& Extension = TEXT("jpg"));
    
    /**
     * Calculate scene bounding box from camera positions
     * @param CameraInfos Array of camera information
     * @return Bounding box of the scene
     */
    static FBox CalculateSceneBounds(const TArray<FCameraInfo>& CameraInfos);

    static bool SaveColmapFormat(
        const TArray<FCameraInfo>& CameraInfos, 
        const FString& OutputPath);

    static FVector GetCameraForwardDirection(const FMatrix& ConvertedRotationMatrix);
    static FVector GetCameraRightDirection(const FMatrix& ConvertedRotationMatrix);
    static FVector GetCameraUpDirection(const FMatrix& ConvertedRotationMatrix);
    
private:
    /**
     * Sample points from mesh vertices
     * @param Mesh Static mesh to sample from
     * @param SampleCount Number of points to sample
     * @param bRandomSampling Use random or uniform sampling
     * @return Array of sampled points
     */
    static TArray<FVector> SampleMeshVertices(UStaticMesh* Mesh,
        int32 SampleCount, bool bRandomSampling);
    
    
    /**
     * Generate colors for point cloud based on position
     * @param Points Array of 3D points
     * @return Array of colors corresponding to points
     */
    static TArray<FLinearColor> GeneratePointColors(const TArray<FVector>& Points);
    
    /**
     * Calculate normals for point cloud (simplified approach)
     * @param Points Array of 3D points
     * @return Array of normals corresponding to points
     */
    static TArray<FVector> CalculatePointNormals(const TArray<FVector>& Points);
    
    // TODO: Remove them and use true colmap
    static bool CreateColmapDirectoryStructure(const FString& OutputPath);
    static bool SaveColmapCameras(
        const TArray<FCameraInfo>& CameraInfos, const FString& OutputPath);
    static bool SaveColmapImages(
        const TArray<FCameraInfo>& CameraInfos, const FString& OutputPath);
    static bool SaveColmapPoints3D(const FString& OutputPath);
};