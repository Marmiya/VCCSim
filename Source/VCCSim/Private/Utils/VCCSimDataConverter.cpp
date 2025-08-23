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

#include "Utils/VCCSimDataConverter.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/FileManager.h"
#include "Engine/StaticMesh.h"
#include "Engine/StaticMeshSourceData.h"
#include "MeshDescription.h"
#include "DataStruct_IO/IOUtils.h"
#include "Math/UnrealMathUtility.h"
#include "Math/RandomStream.h"

// ============================================================================
// POSE FILE CONVERSION
// ============================================================================

TArray<FCameraInfo> FVCCSimDataConverter::ConvertPoseFile(
    const FString& PoseFilePath, 
    const FString& ImageDirectory,
    const FCameraIntrinsics& Intrinsics)
{
    TArray<FCameraInfo> CameraInfos;
    
    // Read pose file
    TArray<FString> FileLines;
    if (!FFileHelper::LoadFileToStringArray(FileLines, *PoseFilePath))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to read pose file: %s"), *PoseFilePath);
        return CameraInfos;
    }
    
    // Determine file format
    bool bIsRecorderFormat = DeterminePoseFileFormat(PoseFilePath);
    
    // Get valid image files
    TArray<FString> ImageFiles = ValidateImageDirectory(ImageDirectory);
    if (ImageFiles.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("No valid images found in directory: %s"), *ImageDirectory);
    }
    
    int32 ValidPoseIndex = 0;
    
    // Process each line
    for (int32 LineIndex = 0; LineIndex < FileLines.Num(); ++LineIndex)
    {
        const FString& Line = FileLines[LineIndex];
        
        // Skip comments and empty lines
        if (Line.IsEmpty() || Line.StartsWith(TEXT("#")))
        {
            continue;
        }
        
        // Parse pose line
        bool bIsRecorderFormatLocal = bIsRecorderFormat;
        FVCCSimPoseData PoseData = ParsePoseLine(Line, bIsRecorderFormatLocal);
        
        if (PoseData.Location.IsZero() && PoseData.Rotation.IsZero())
        {
            // Invalid pose data, skip
            continue;
        }
        
        // Create camera info
        FCameraInfo CameraInfo;
        CameraInfo.UID = ValidPoseIndex;
        
        // Convert coordinate system
        CameraInfo.RotationMatrix = ConvertRotation(PoseData.Rotation);
        CameraInfo.Translation = ConvertLocation(PoseData.Location);
        
        // Set camera parameters
        CameraInfo.FOVDegrees = FMath::RadiansToDegrees(2.0f * FMath::Atan(Intrinsics.Width * 0.5f / Intrinsics.FocalX));
        CameraInfo.Width = Intrinsics.Width;
        CameraInfo.Height = Intrinsics.Height;
        CameraInfo.FocalX = Intrinsics.FocalX;
        CameraInfo.FocalY = Intrinsics.FocalY;
        CameraInfo.CenterX = Intrinsics.CenterX;
        CameraInfo.CenterY = Intrinsics.CenterY;
        
        // Set image paths
        FString ImageFileName = GenerateImageFileName(ValidPoseIndex);
        CameraInfo.ImageName = ImageFileName;
        CameraInfo.ImagePath = FPaths::Combine(ImageDirectory, ImageFileName);
        
        // Verify image exists
        if (ValidPoseIndex < ImageFiles.Num())
        {
            CameraInfo.ImagePath = ImageFiles[ValidPoseIndex];
            CameraInfo.ImageName = FPaths::GetCleanFilename(CameraInfo.ImagePath);
        }
        
        CameraInfos.Add(CameraInfo);
        ValidPoseIndex++;
    }
    
    UE_LOG(LogTemp, Log, TEXT("Converted %d poses to camera info"), CameraInfos.Num());
    
    return CameraInfos;
}

FVCCSimPoseData FVCCSimDataConverter::ParsePoseLine(const FString& Line, bool& bIsRecorderFormat)
{
    FVCCSimPoseData PoseData;
    
    TArray<FString> Values;
    Line.ParseIntoArray(Values, TEXT(" "), true);
    
    if (Values.Num() == 7)
    {
        // Recorder format: Timestamp X Y Z Roll Pitch Yaw
        bIsRecorderFormat = true;
        PoseData.Timestamp = FCString::Atod(*Values[0]);
        PoseData.Location.X = FCString::Atof(*Values[1]);
        PoseData.Location.Y = FCString::Atof(*Values[2]);
        PoseData.Location.Z = FCString::Atof(*Values[3]);
        PoseData.Rotation.Roll = FCString::Atof(*Values[4]);
        PoseData.Rotation.Pitch = FCString::Atof(*Values[5]);
        PoseData.Rotation.Yaw = FCString::Atof(*Values[6]);
    }
    else if (Values.Num() == 6)
    {
        // Panel format: X Y Z Pitch Yaw Roll
        bIsRecorderFormat = false;
        PoseData.Timestamp = 0.0; // No timestamp in panel format
        PoseData.Location.X = FCString::Atof(*Values[0]);
        PoseData.Location.Y = FCString::Atof(*Values[1]);
        PoseData.Location.Z = FCString::Atof(*Values[2]);
        PoseData.Rotation.Pitch = FCString::Atof(*Values[3]);
        PoseData.Rotation.Yaw = FCString::Atof(*Values[4]);
        PoseData.Rotation.Roll = FCString::Atof(*Values[5]);
    }
    else
    {
        // Invalid format
        UE_LOG(LogTemp, Warning, TEXT("Invalid pose line format: %s"), *Line);
        return FVCCSimPoseData(); // Return zero data
    }
    
    
    return PoseData;
}

bool FVCCSimDataConverter::DeterminePoseFileFormat(const FString& PoseFilePath)
{
    TArray<FString> FileLines;
    if (!FFileHelper::LoadFileToStringArray(FileLines, *PoseFilePath))
    {
        return false;
    }
    
    // Check first non-comment line
    for (const FString& Line : FileLines)
    {
        if (!Line.IsEmpty() && !Line.StartsWith(TEXT("#")))
        {
            TArray<FString> Values;
            Line.ParseIntoArray(Values, TEXT(" "), true);
            return Values.Num() == 7; // True for Recorder format, false for Panel format
        }
    }
    
    return false; // Default to Panel format
}


// ============================================================================
// COORDINATE SYSTEM CONVERSION
// ============================================================================

FVector FVCCSimDataConverter::ConvertLocation(const FVector& UELocation)
{
    // Convert from UE left-handed system (cm) to right-handed system (m)
    // X and Z axes maintain direction, Y axis flipped, units converted to meters
    return FVector(
        UELocation.X * 0.01f,  // X: forward direction unchanged, cm to m
        -UELocation.Y * 0.01f, // Y: right to left (flipped for right-handed), cm to m  
        UELocation.Z * 0.01f   // Z: up direction unchanged, cm to m
    );
}

FMatrix FVCCSimDataConverter::ConvertRotation(const FRotator& UERotation)
{    
    // Get UE rotation matrix
    FMatrix UEMatrix = FRotationMatrix::Make(UERotation);
    
    FMatrix CoordTransform = FMatrix::Identity;
    CoordTransform.M[0][0] = 1.0f;   // X -> X
    CoordTransform.M[1][1] = -1.0f;  // Y -> -Y  
    CoordTransform.M[2][2] = 1.0f;   // Z -> Z
    
    // Apply transformation: ConvertedMatrix = CoordTransform * UEMatrix * CoordTransform^T
    // Since CoordTransform is diagonal, CoordTransform^T = CoordTransform
    return CoordTransform * UEMatrix * CoordTransform;
}

FVector FVCCSimDataConverter::GetCameraForwardDirection(const FMatrix& ConvertedRotationMatrix)
{
    // Camera forward direction is the first column of the rotation matrix
    return ConvertedRotationMatrix.GetColumn(0).GetSafeNormal();
}

FVector FVCCSimDataConverter::GetCameraRightDirection(const FMatrix& ConvertedRotationMatrix)
{
    // Camera right direction is the second column of the rotation matrix
    return ConvertedRotationMatrix.GetColumn(1).GetSafeNormal();
}

FVector FVCCSimDataConverter::GetCameraUpDirection(const FMatrix& ConvertedRotationMatrix)
{
    // Camera up direction is the third column of the rotation matrix
    return ConvertedRotationMatrix.GetColumn(2).GetSafeNormal();
}

// ============================================================================
// CAMERA PARAMETER CONVERSION
// ============================================================================

FCameraIntrinsics FVCCSimDataConverter::ConvertCameraParams(
    float FOVDegrees, int32 Width, int32 Height)
{
    // Calculate focal lengths from FOV
    float FocalX = CalculateFocalLength(FOVDegrees, Width);
    float FocalY = CalculateFocalLength(FOVDegrees, Height);
    
    // Principal point at image center
    float CenterX = Width * 0.5f;
    float CenterY = Height * 0.5f;
    
    return FCameraIntrinsics(Width, Height, FocalX, FocalY, CenterX, CenterY);
}

float FVCCSimDataConverter::CalculateFocalLength(float FOVDegrees, int32 ImageDimension)
{
    float FOVRadians = FMath::DegreesToRadians(FOVDegrees);
    return (ImageDimension * 0.5f) / FMath::Tan(FOVRadians * 0.5f);
}

// ============================================================================
// MESH PROCESSING
// ============================================================================

FPointCloudData FVCCSimDataConverter::ConvertMeshToPointCloud(
    UStaticMesh* Mesh, 
    int32 SampleCount, 
    bool bApplyCoordinateTransformation)
{
    FPointCloudData PointCloudData;
    
    if (!Mesh || !Mesh->GetRenderData() || !Mesh->GetRenderData()->LODResources.IsValidIndex(0))
    {
        UE_LOG(LogTemp, Error, TEXT("Invalid mesh for point cloud conversion"));
        return PointCloudData;
    }
    
    // Sample points from mesh (in UE coordinates)
    TArray<FVector> SampledPoints = SampleMeshVertices(Mesh, SampleCount, true);
    
    // Apply coordinate transformation if requested
    if (bApplyCoordinateTransformation)
    {
        UE_LOG(LogTemp, Log, TEXT("Applying coordinate transformation to mesh points"));
        
        // Transform each point from UE to right-handed coordinates
        for (int32 i = 0; i < SampledPoints.Num(); ++i)
        {
            FVector UEPoint = SampledPoints[i];
            FVector TSPoint = ConvertLocation(UEPoint);
            SampledPoints[i] = TSPoint;
        }
        
        UE_LOG(LogTemp, Log, TEXT("Transformed %d mesh points from UE to "
                                  "right-handed coordinates"), SampledPoints.Num());
    }
    
    // Generate colors and normals
    TArray<FLinearColor> Colors = GeneratePointColors(SampledPoints);
    TArray<FVector> Normals = CalculatePointNormals(SampledPoints);
    
    // Fill point cloud data using FRatPoint
    PointCloudData.Reserve(SampledPoints.Num());
    for (int32 i = 0; i < SampledPoints.Num(); ++i)
    {
        PointCloudData.AddPoint(SampledPoints[i], Colors[i], Normals[i], true);
    }
    
    UE_LOG(LogTemp, Log, TEXT("Converted mesh to point cloud with %d points"),
        PointCloudData.GetPointCount());
    
    return PointCloudData;
}

FPointCloudData FVCCSimDataConverter::GenerateRandomPointCloud(
    const TArray<FCameraInfo>& CameraInfos, int32 PointCount)
{
    FPointCloudData PointCloudData;
    
    if (CameraInfos.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("No camera info provided for random point cloud generation"));
        return PointCloudData;
    }
    
    // Calculate scene bounds from camera positions
    FBox SceneBounds = CalculateSceneBounds(CameraInfos);
    
    // Expand bounds slightly
    SceneBounds = SceneBounds.ExpandBy(SceneBounds.GetSize().GetMax() * 0.1f);
    
    // Generate random points within scene bounds
    FRandomStream RandomStream(FMath::Rand());
    
    PointCloudData.Reserve(PointCount);
    for (int32 i = 0; i < PointCount; ++i)
    {
        FVector RandomPoint = FVector(
            RandomStream.FRandRange(SceneBounds.Min.X, SceneBounds.Max.X),
            RandomStream.FRandRange(SceneBounds.Min.Y, SceneBounds.Max.Y),
            RandomStream.FRandRange(SceneBounds.Min.Z, SceneBounds.Max.Z)
        );
        
        // Generate a random color
        FLinearColor RandomColor = FLinearColor(
            RandomStream.FRand(),
            RandomStream.FRand(),
            RandomStream.FRand(),
            1.0f
        );
        
        // Generate a random normal (normalized)
        FVector RandomNormal = FVector(
            RandomStream.FRandRange(-1.0f, 1.0f),
            RandomStream.FRandRange(-1.0f, 1.0f),
            RandomStream.FRandRange(-1.0f, 1.0f)
        ).GetSafeNormal();
        
        PointCloudData.AddPoint(RandomPoint, RandomColor, RandomNormal, true);
    }
    
    UE_LOG(LogTemp, Log, TEXT("Generated random point cloud with %d points"),
        PointCloudData.GetPointCount());
    
    return PointCloudData;
}


// ============================================================================
// FILE I/O OPERATIONS
// ============================================================================

bool FVCCSimDataConverter::SaveCameraInfo(
    const TArray<FCameraInfo>& CameraInfos, const FString& OutputPath)
{
    // Create output directory
    if (!CreateModelDirectoryStructure(OutputPath))
    {
        return false;
    }
    
    // Save camera info in JSON format for right-handed coordinate system models
    FString CameraInfoPath = FPaths::Combine(OutputPath, TEXT("camera_info.json"));
    FString JsonContent = TEXT("[\n");
    
    for (int32 i = 0; i < CameraInfos.Num(); ++i)
    {
        const FCameraInfo& Info = CameraInfos[i];
        
        JsonContent += FString::Printf(TEXT(
            "  {\n"
            "    \"uid\": %d,\n"
            "    \"image_path\": \"%s\",\n"
            "    \"image_name\": \"%s\",\n"
            "    \"width\": %d,\n"
            "    \"height\": %d,\n"
            "    \"focal_x\": %.6f,\n"
            "    \"focal_y\": %.6f,\n"
            "    \"center_x\": %.6f,\n"
            "    \"center_y\": %.6f,\n"
            "    \"rotation\": [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f],\n"
            "    \"translation\": [%.6f, %.6f, %.6f]\n"
            "  }%s\n"
        ),
            Info.UID,
            *Info.ImagePath,
            *Info.ImageName,
            Info.Width,
            Info.Height,
            Info.FocalX,
            Info.FocalY,
            Info.CenterX,
            Info.CenterY,
            Info.RotationMatrix.M[0][0], Info.RotationMatrix.M[0][1], Info.RotationMatrix.M[0][2],
            Info.RotationMatrix.M[1][0], Info.RotationMatrix.M[1][1], Info.RotationMatrix.M[1][2],
            Info.RotationMatrix.M[2][0], Info.RotationMatrix.M[2][1], Info.RotationMatrix.M[2][2],
            Info.Translation.X, Info.Translation.Y, Info.Translation.Z,
            (i < CameraInfos.Num() - 1) ? TEXT(",") : TEXT("")
        );
    }
    
    JsonContent += TEXT("]\n");
    
    return FFileHelper::SaveStringToFile(JsonContent, *CameraInfoPath);
}

bool FVCCSimDataConverter::SavePointCloudToPLY(const FPointCloudData& PointCloudData,
    const FString& OutputFilePath)
{
    // Use the new unified FPLYWriter class
    FPLYWriter::FPLYWriteConfig Config;
    Config.bIncludeColors = true;
    Config.bIncludeNormals = true;
    Config.bBinaryFormat = false;
    
    return FPLYWriter::WritePointCloudToPLY(PointCloudData, OutputFilePath, Config);
}

bool FVCCSimDataConverter::CreateModelDirectoryStructure(const FString& OutputPath)
{
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    
    // Create main output directory
    if (!PlatformFile.CreateDirectoryTree(*OutputPath))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create output directory: %s"), *OutputPath);
        return false;
    }
    
    // Create subdirectories if needed
    TArray<FString> Subdirectories = {
        TEXT("images"),
        TEXT("output"),
        TEXT("logs")
    };
    
    for (const FString& SubDir : Subdirectories)
    {
        FString SubDirPath = FPaths::Combine(OutputPath, SubDir);
        if (!PlatformFile.CreateDirectoryTree(*SubDirPath))
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to create subdirectory: %s"), *SubDirPath);
            return false;
        }
    }
    
    return true;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

TArray<FString> FVCCSimDataConverter::ValidateImageDirectory(
    const FString& ImageDirectory, const TArray<FString>& ValidExtensions)
{
    TArray<FString> ValidImages;
    
    if (!FPaths::DirectoryExists(ImageDirectory))
    {
        UE_LOG(LogTemp, Error, TEXT("Image directory does not exist: %s"), *ImageDirectory);
        return ValidImages;
    }
    
    TArray<FString> FoundFiles;
    IFileManager::Get().FindFiles(FoundFiles, *ImageDirectory, nullptr);
    
    for (const FString& FileName : FoundFiles)
    {
        FString Extension = FPaths::GetExtension(FileName).ToLower();
        
        if (ValidExtensions.Contains(Extension))
        {
            ValidImages.Add(FPaths::Combine(ImageDirectory, FileName));
        }
    }
    
    ValidImages.Sort();
    
    UE_LOG(LogTemp, Log, TEXT("Found %d valid images in directory: %s"),
        ValidImages.Num(), *ImageDirectory);
    
    return ValidImages;
}

FString FVCCSimDataConverter::GenerateImageFileName(int32 PoseIndex, const FString& Extension)
{
    return FString::Printf(TEXT("%06d.%s"), PoseIndex, *Extension);
}

FBox FVCCSimDataConverter::CalculateSceneBounds(const TArray<FCameraInfo>& CameraInfos)
{
    FBox Bounds;
    Bounds.Init();
    
    for (const FCameraInfo& CameraInfo : CameraInfos)
    {
        Bounds += CameraInfo.Translation;
    }
    
    if (!Bounds.IsValid)
    {
        // Fallback to default bounds
        Bounds = FBox(FVector(-10.0f), FVector(10.0f));
    }
    
    return Bounds;
}

// ============================================================================
// PRIVATE HELPER FUNCTIONS
// ============================================================================


TArray<FVector> FVCCSimDataConverter::SampleMeshVertices(
    UStaticMesh* Mesh, int32 SampleCount, bool bRandomSampling)
{
    TArray<FVector> SampledPoints;
    
    if (!Mesh || !Mesh->GetRenderData() || Mesh->GetRenderData()->LODResources.Num() == 0)
    {
        return SampledPoints;
    }
    
    const FStaticMeshLODResources& LODResource = Mesh->GetRenderData()->LODResources[0];
    const FPositionVertexBuffer& PositionBuffer = LODResource.VertexBuffers.PositionVertexBuffer;
    
    int32 VertexCount = PositionBuffer.GetNumVertices();
    if (VertexCount == 0)
    {
        return SampledPoints;
    }
    
    if (bRandomSampling)
    {
        // Random sampling
        FRandomStream RandomStream(FMath::Rand());
        for (int32 i = 0; i < SampleCount; ++i)
        {
            int32 RandomIndex = RandomStream.RandRange(0, VertexCount - 1);
            FVector Vertex = (FVector)PositionBuffer.VertexPosition(RandomIndex);
            SampledPoints.Add(Vertex);
        }
    }
    else
    {
        // Uniform sampling
        int32 Step = FMath::Max(1, VertexCount / SampleCount);
        for (int32 i = 0; i < VertexCount && SampledPoints.Num() < SampleCount; i += Step)
        {
            FVector Vertex = (FVector)PositionBuffer.VertexPosition(i);
            SampledPoints.Add(Vertex);
        }
    }
    
    return SampledPoints;
}

TArray<FLinearColor> FVCCSimDataConverter::GeneratePointColors(const TArray<FVector>& Points)
{
    TArray<FLinearColor> Colors;
    Colors.Reserve(Points.Num());
    
    // Generate colors based on position or use a default scheme
    for (const FVector& Point : Points)
    {
        // Simple color generation based on Z coordinate
        float NormalizedZ = (Point.Z + 1000.0f) / 2000.0f; // Assume Z range [-1000, 1000]
        NormalizedZ = FMath::Clamp(NormalizedZ, 0.0f, 1.0f);
        
        FLinearColor Color = FLinearColor(NormalizedZ, 1.0f - NormalizedZ, 0.5f, 1.0f);
        Colors.Add(Color);
    }
    
    return Colors;
}

TArray<FVector> FVCCSimDataConverter::CalculatePointNormals(const TArray<FVector>& Points)
{
    TArray<FVector> Normals;
    Normals.Reserve(Points.Num());
    
    // Simple normal calculation - for a more sophisticated approach,
    // you would need mesh connectivity information
    for (int32 i = 0; i < Points.Num(); ++i)
    {
        FVector Normal = FVector::UpVector; // Default to up vector
        
        // Try to estimate normal from nearby points
        if (i > 0 && i < Points.Num() - 1)
        {
            FVector V1 = Points[i] - Points[i - 1];
            FVector V2 = Points[i + 1] - Points[i];
            Normal = FVector::CrossProduct(V1, V2).GetSafeNormal();
        }
        
        Normals.Add(Normal);
    }
    
    return Normals;
}

// ============================================================================
// COLMAP FORMAT OUTPUT
// ============================================================================

bool FVCCSimDataConverter::SaveColmapFormat(
    const TArray<FCameraInfo>& CameraInfos, 
    const FString& OutputPath)
{
    // Create COLMAP directory structure
    if (!CreateColmapDirectoryStructure(OutputPath))
    {
        return false;
    }
    
    // Save cameras.txt
    if (!SaveColmapCameras(CameraInfos, OutputPath))
    {
        return false;
    }
    
    // Save images.txt  
    if (!SaveColmapImages(CameraInfos, OutputPath))
    {
        return false;
    }
    
    // Save empty points3D.txt
    if (!SaveColmapPoints3D(OutputPath))
    {
        return false;
    }
    
    UE_LOG(LogTemp, Log, TEXT("Successfully saved COLMAP format to: %s"), *OutputPath);
    return true;
}

bool FVCCSimDataConverter::CreateColmapDirectoryStructure(const FString& OutputPath)
{
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    
    // Create main output directory
    if (!PlatformFile.CreateDirectoryTree(*OutputPath))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create output directory: %s"), *OutputPath);
        return false;
    }
    
    // Create COLMAP sparse reconstruction directory
    FString SparseDir = FPaths::Combine(OutputPath, TEXT("sparse"), TEXT("0"));
    if (!PlatformFile.CreateDirectoryTree(*SparseDir))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create sparse directory: %s"), *SparseDir);
        return false;
    }
    
    return true;
}

bool FVCCSimDataConverter::SaveColmapCameras(
    const TArray<FCameraInfo>& CameraInfos, const FString& OutputPath)
{
    if (CameraInfos.Num() == 0)
    {
        return false;
    }
    
    FString CamerasPath = FPaths::Combine(OutputPath, TEXT("sparse"), TEXT("0"), TEXT("cameras.txt"));
    FString Content;
    
    // Header
    Content += TEXT("# Camera list with one line of data per camera:\n");
    Content += TEXT("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n");
    Content += TEXT("# Number of cameras: 1\n");
    
    // Assume all cameras have same parameters (use first camera)
    const FCameraInfo& FirstCamera = CameraInfos[0];
    Content += FString::Printf(TEXT("1 PINHOLE %d %d %.6f %.6f %.6f %.6f\n"),
        FirstCamera.Width,
        FirstCamera.Height,
        FirstCamera.FocalX,
        FirstCamera.FocalY,
        FirstCamera.CenterX,
        FirstCamera.CenterY
    );
    
    return FFileHelper::SaveStringToFile(Content, *CamerasPath);
}

bool FVCCSimDataConverter::SaveColmapImages(
    const TArray<FCameraInfo>& CameraInfos, const FString& OutputPath)
{
    FString ImagesPath = FPaths::Combine(OutputPath, TEXT("sparse"), TEXT("0"), TEXT("images.txt"));
    FString Content;
    
    // Header
    Content += TEXT("# Image list with two lines of data per image:\n");
    Content += TEXT("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n");
    Content += TEXT("# POINTS2D[] as (X, Y, POINT3D_ID)\n");
    Content += FString::Printf(TEXT("# Number of images: %d, mean observations per image: 0\n"),
        CameraInfos.Num());
    
    // Convert each camera pose
    for (int32 i = 0; i < CameraInfos.Num(); ++i)
    {
        const FCameraInfo& CameraInfo = CameraInfos[i];
        
        // Convert rotation matrix to quaternion
        FQuat Quaternion = FQuat(CameraInfo.RotationMatrix);
        
        // COLMAP uses camera-to-world transformation, but we have world-to-camera
        // So we need to invert the transformation
        FQuat InvQuaternion = Quaternion.Inverse();
        FVector InvTranslation = -(InvQuaternion * CameraInfo.Translation);
        
        // Write image line
        Content += FString::Printf(TEXT("%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f 1 %s\n"),
            i + 1,  // IMAGE_ID (1-based)
            InvQuaternion.W, InvQuaternion.X, InvQuaternion.Y, InvQuaternion.Z,  // Quaternion
            InvTranslation.X, InvTranslation.Y, InvTranslation.Z,  // Translation
            *CameraInfo.ImageName  // Image name
        );
        
        // Empty points2D line
        Content += TEXT("\n");
    }
    
    return FFileHelper::SaveStringToFile(Content, *ImagesPath);
}

bool FVCCSimDataConverter::SaveColmapPoints3D(const FString& OutputPath)
{
    FString Points3DPath = FPaths::Combine(OutputPath, TEXT("sparse"), TEXT("0"), TEXT("points3D.txt"));
    FString Content;
    
    // Header for empty points3D file
    Content += TEXT("# 3D point list with one line of data per point:\n");
    Content += TEXT("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n");
    Content += TEXT("# Number of points: 0, mean track length: 0\n");
    
    return FFileHelper::SaveStringToFile(Content, *Points3DPath);
}