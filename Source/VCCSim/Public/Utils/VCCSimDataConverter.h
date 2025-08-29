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
#include "DataStruct_IO/CameraData.h"
#include "DataStruct_IO/PointCloud.h"


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
     * Convert normal vector from UE coordinate system to Triangle Splatting coordinate system
     * @param UENormal Normal vector in UE coordinate system
     * @return Normal vector in Triangle Splatting coordinate system
     */
    static FVector ConvertNormal(const FVector& UENormal);
    
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
     * Convert camera parameters with optional direct focal length specification
     * If FocalX and FocalY are > 0, they are used directly; otherwise FOV is used to calculate them
     * @param FOVDegrees Field of view in degrees (fallback if fx/fy not provided)
     * @param Width Image width in pixels
     * @param Height Image height in pixels
     * @param FocalX Focal length X in pixels (fx) - if > 0, used directly
     * @param FocalY Focal length Y in pixels (fy) - if > 0, used directly
     * @return Camera intrinsics structure
     */
    static FCameraIntrinsics ConvertCameraParamsWithFocalLength(
        float FOVDegrees, int32 Width, int32 Height, float FocalX = 0.0f, float FocalY = 0.0f);
    
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

    // ============================================================================
    // COLMAP INTEGRATION
    // ============================================================================
    
    /**
     * Run complete COLMAP pipeline using system executable
     * @param ImageDirectory Source directory containing images
     * @param OutputPath Base output directory (timestamped folder will be created)
     * @param ColmapExecutablePath Path to COLMAP executable
     * @return True if successful
     */
    static bool RunColmapPipeline(
        const FString& ImageDirectory,
        const FString& OutputPath,
        const FString& ColmapExecutablePath = TEXT("D:\\colmap-x64-windows-cuda"));
    
    /**
     * Create timestamped COLMAP dataset directory
     * @param BaseOutputPath Base output directory
     * @return Full path to created timestamped directory
     */
    static FString CreateTimestampedColmapDirectory(const FString& BaseOutputPath);
    
    /**
     * Copy images to COLMAP dataset structure for processing
     * @param ImageDirectory Source directory containing images
     * @param ColmapDatasetPath Path to COLMAP dataset directory
     * @return True if successful
     */
    static bool PrepareColmapDataset(
        const FString& ImageDirectory,
        const FString& ColmapDatasetPath);
    
    /**
     * Run COLMAP feature extraction using external command executor
     * @param ColmapExecutablePath Path to COLMAP executable
     * @param DatasetPath Path to dataset directory
     * @param DatabasePath Path to database file
     * @param CommandExecutor Function to execute COLMAP commands (for process management)
     * @return True if successful
     */
    static bool RunColmapFeatureExtraction(
        const FString& ColmapExecutablePath,
        const FString& DatasetPath,
        const FString& DatabasePath,
        TFunction<bool(const FString&, const FString&, const FString&)> CommandExecutor = nullptr);
    
    /**
     * Run COLMAP feature matching using external command executor
     * @param ColmapExecutablePath Path to COLMAP executable
     * @param DatabasePath Path to database file
     * @param CommandExecutor Function to execute COLMAP commands (for process management)
     * @return True if successful
     */
    static bool RunColmapFeatureMatching(
        const FString& ColmapExecutablePath,
        const FString& DatabasePath,
        TFunction<bool(const FString&, const FString&, const FString&)> CommandExecutor = nullptr);
    
    /**
     * Run COLMAP sparse reconstruction using external command executor
     * @param ColmapExecutablePath Path to COLMAP executable
     * @param DatabasePath Path to database file
     * @param ImagePath Path to image directory
     * @param OutputPath Path to output sparse reconstruction
     * @param CommandExecutor Function to execute COLMAP commands (for process management)
     * @return True if successful
     */
    static bool RunColmapSparseReconstruction(
        const FString& ColmapExecutablePath,
        const FString& DatabasePath,
        const FString& ImagePath,
        const FString& OutputPath,
        TFunction<bool(const FString&, const FString&, const FString&)> CommandExecutor = nullptr);
    
    /**
     * Convert COLMAP binary model to text format for debugging
     * @param ColmapExecutablePath Path to COLMAP executable directory
     * @param BinaryModelPath Path to binary model directory (sparse/0)
     * @param TextModelPath Path to output text model directory
     * @param CommandExecutor Optional command executor for process management
     * @return True if conversion succeeded
     */
    static bool RunColmapModelConverter(
        const FString& ColmapExecutablePath,
        const FString& BinaryModelPath,
        const FString& TextModelPath,
        TFunction<bool(const FString&, const FString&, const FString&)> CommandExecutor = nullptr);

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
     * Sample mesh vertices with colors for point cloud generation
     * @param Mesh Static mesh to sample from
     * @param SampleCount Number of vertices to sample
     * @param bRandomSampling If true, use random sampling; if false, use uniform sampling
     * @param OutPositions Array of sampled vertex positions
     * @param OutColors Array of sampled vertex colors
     * @return True if sampling was successful
     */
    static bool SampleMeshVerticesWithColors(UStaticMesh* Mesh,
        int32 SampleCount, bool bRandomSampling,
        TArray<FVector>& OutPositions, TArray<FLinearColor>& OutColors);
    
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
};