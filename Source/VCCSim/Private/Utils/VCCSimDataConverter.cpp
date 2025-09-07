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
#include "HAL/PlatformProcess.h"
#include "Misc/DateTime.h"

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
        UE_LOG(LogTemp, Warning, TEXT("No valid images found in directory: %s"),
            *ImageDirectory);
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
        
        if (PoseData.Location.IsZero() && PoseData.Quaternion.IsIdentity())
        {
            // Invalid pose data, skip
            continue;
        }
        
        // Create camera info
        FCameraInfo CameraInfo;
        CameraInfo.UID = ValidPoseIndex;
        
        FMatrix UEMatrix = FQuatRotationMatrix::Make(PoseData.Quaternion);
        FMatrix CoordTransform = FMatrix::Identity;
        CoordTransform.M[0][0] = 0.0f;   // X_ue = Y_colmap
        CoordTransform.M[0][1] = 1.0f;   
        CoordTransform.M[1][0] = 1.0f;   // Y_ue = X_colmap
        CoordTransform.M[1][1] = 0.0f;   
        CoordTransform.M[2][2] = 1.0f;   // Z_ue = Z_colmap
        FMatrix ConvertedMatrix = CoordTransform * UEMatrix * CoordTransform.GetTransposed();
        CameraInfo.Rotation = ConvertedMatrix.ToQuat();
        CameraInfo.Position = ConvertLocation(PoseData.Location);
        
        // Set camera parameters
        CameraInfo.FOVDegrees = FMath::RadiansToDegrees(2.0f *
            FMath::Atan(Intrinsics.Width * 0.5f / Intrinsics.FocalX));
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
    
    UE_LOG(LogTemp, Log, TEXT("Processed %d poses"), CameraInfos.Num());
    
    return CameraInfos;
}

FVCCSimPoseData FVCCSimDataConverter::ParsePoseLine(
    const FString& Line, bool& bIsRecorderFormat)
{
    FVCCSimPoseData PoseData;
    
    TArray<FString> Values;
    Line.ParseIntoArray(Values, TEXT(" "), true);
    
    if (Values.Num() == 8)
    {
        // Recorder format: Timestamp X Y Z Qx Qy Qz Qw
        bIsRecorderFormat = true;
        PoseData.Timestamp = FCString::Atod(*Values[0]);
        PoseData.Location.X = FCString::Atof(*Values[1]);
        PoseData.Location.Y = FCString::Atof(*Values[2]);
        PoseData.Location.Z = FCString::Atof(*Values[3]);
        PoseData.Quaternion.X = FCString::Atof(*Values[4]);
        PoseData.Quaternion.Y = FCString::Atof(*Values[5]);
        PoseData.Quaternion.Z = FCString::Atof(*Values[6]);
        PoseData.Quaternion.W = FCString::Atof(*Values[7]);
    }
    else if (Values.Num() == 7)
    {
        // Panel format: X Y Z Qx Qy Qz Qw
        bIsRecorderFormat = false;
        PoseData.Timestamp = 0.0; // No timestamp in panel format
        PoseData.Location.X = FCString::Atof(*Values[0]);
        PoseData.Location.Y = FCString::Atof(*Values[1]);
        PoseData.Location.Z = FCString::Atof(*Values[2]);
        PoseData.Quaternion.X = FCString::Atof(*Values[3]);
        PoseData.Quaternion.Y = FCString::Atof(*Values[4]);
        PoseData.Quaternion.Z = FCString::Atof(*Values[5]);
        PoseData.Quaternion.W = FCString::Atof(*Values[6]);
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
            return Values.Num() == 8;
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
    // Swap X and Y axes: UE(X_forward, Y_right, Z_up) -> RightHanded(X_right, Y_forward, Z_up)
    // This matches the inverse transformation in Colmap_2_ply.py
    return FVector(
        UELocation.Y * 0.01f,  // X_rh = Y_ue (right direction), cm to m
        UELocation.X * 0.01f,  // Y_rh = X_ue (forward direction), cm to m
        UELocation.Z * 0.01f   // Z_rh = Z_ue (up direction unchanged), cm to m
    );
}

FVector FVCCSimDataConverter::ConvertNormal(const FVector& UENormal)
{
    // Convert normal from UE left-handed system to right-handed system
    // Swap X and Y axes, consistent with position transformation
    return FVector(
        UENormal.Y,   // X_rh = Y_ue (right direction)
        UENormal.X,   // Y_rh = X_ue (forward direction)  
        UENormal.Z    // Z_rh = Z_ue (up direction unchanged)
    );
}

FMatrix FVCCSimDataConverter::ConvertRotation(const FRotator& UERotation)
{    
    // Get UE rotation matrix
    FMatrix UEMatrix = FRotationMatrix::Make(UERotation);
    
    // Swap X and Y axes transformation matrix
    FMatrix CoordTransform = FMatrix::Identity;
    CoordTransform.M[0][0] = 0.0f;   // X_rh = Y_ue
    CoordTransform.M[0][1] = 1.0f;   
    CoordTransform.M[1][0] = 1.0f;   // Y_rh = X_ue
    CoordTransform.M[1][1] = 0.0f;   
    CoordTransform.M[2][2] = 1.0f;   // Z_rh = Z_ue
    
    // Apply transformation: ConvertedMatrix = CoordTransform * UEMatrix * CoordTransform^T
    return CoordTransform * UEMatrix * CoordTransform.GetTransposed();
}

FMatrix FVCCSimDataConverter::ConvertRotation(const FQuat& UEQuaternion)
{    
    // Convert quaternion to rotation matrix
    FMatrix UEMatrix = FQuatRotationMatrix::Make(UEQuaternion);
    
    // Swap X and Y axes transformation matrix
    FMatrix CoordTransform = FMatrix::Identity;
    CoordTransform.M[0][0] = 0.0f;   // X_rh = Y_ue
    CoordTransform.M[0][1] = 1.0f;   
    CoordTransform.M[1][0] = 1.0f;   // Y_rh = X_ue
    CoordTransform.M[1][1] = 0.0f;   
    CoordTransform.M[2][2] = 1.0f;   // Z_rh = Z_ue
    
    // Apply transformation: ConvertedMatrix = CoordTransform * UEMatrix * CoordTransform^T
    return CoordTransform * UEMatrix * CoordTransform.GetTransposed();
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
    
    return FCameraIntrinsics(Width, Height, FocalX, FocalY,
        CenterX, CenterY);
}

FCameraIntrinsics FVCCSimDataConverter::ConvertCameraParamsWithFocalLength(
    float FOVDegrees, int32 Width, int32 Height, float FocalX, float FocalY)
{
    // Use provided focal lengths if they are valid (> 0), otherwise calculate from FOV
    float EffectiveFocalX, EffectiveFocalY;
    
    if (FocalX > 0.0f)
    {
        EffectiveFocalX = FocalX;
    }
    else
    {
        EffectiveFocalX = CalculateFocalLength(FOVDegrees, Width);
    }
    
    if (FocalY > 0.0f)
    {
        EffectiveFocalY = FocalY;
    }
    else
    {
        EffectiveFocalY = CalculateFocalLength(FOVDegrees, Height);
    }
    
    // Principal point at image center
    float CenterX = Width * 0.5f;
    float CenterY = Height * 0.5f;
    
    return FCameraIntrinsics(Width, Height, EffectiveFocalX,
        EffectiveFocalY, CenterX, CenterY);
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
    
    // Sample points and colors from mesh (in UE coordinates)
    TArray<FVector> SampledPoints;
    TArray<FLinearColor> Colors;
    
    if (!SampleMeshVerticesWithColors(Mesh, SampleCount, true,
        SampledPoints, Colors))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to sample mesh vertices with colors"));
        return PointCloudData;
    }
    
    // Apply coordinate transformation if requested
    if (bApplyCoordinateTransformation)
    {
        // Transform each point from UE to right-handed coordinates
        for (int32 i = 0; i < SampledPoints.Num(); ++i)
        {
            SampledPoints[i] = ConvertLocation(SampledPoints[i]);
        }
        
        UE_LOG(LogTemp, Log, TEXT("Transformed %d points to"
                                  " right-handed coordinates"), SampledPoints.Num());
    }
    
    // Fill point cloud data with random normals for Triangle Splatting compatibility
    PointCloudData.Reserve(SampledPoints.Num());
    for (int32 i = 0; i < SampledPoints.Num(); ++i)
    {
        // Generate random normalized normal vector for Triangle Splatting compatibility
        FVector RandomNormal = FVector(
            FMath::RandRange(-1.0f, 1.0f),
            FMath::RandRange(-1.0f, 1.0f), 
            FMath::RandRange(-1.0f, 1.0f)
        ).GetSafeNormal();
        
        PointCloudData.AddPoint(SampledPoints[i], Colors[i],
            RandomNormal, true);
    }
    
    UE_LOG(LogTemp, Log, TEXT("Created point cloud with %d points"),
        PointCloudData.GetPointCount());
    
    return PointCloudData;
}

FPointCloudData FVCCSimDataConverter::GenerateRandomPointCloud(
    const TArray<FCameraInfo>& CameraInfos, int32 PointCount)
{
    FPointCloudData PointCloudData;
    
    if (CameraInfos.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("No camera info provided for "
                                    "random point cloud generation"));
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
        
        PointCloudData.AddPoint(RandomPoint, RandomColor,
            FVector::ZeroVector, false);
    }
    
    UE_LOG(LogTemp, Log, TEXT("Generated %d random points"),
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
            "    \"rotation\": [%.6f, %.6f, %.6f, %.6f],\n"
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
            (float)Info.Rotation.X, (float)Info.Rotation.Y, (float)Info.Rotation.Z, (float)Info.Rotation.W,
            (float)Info.Position.X, (float)Info.Position.Y, (float)Info.Position.Z,
            (i < CameraInfos.Num() - 1) ? TEXT(",") : TEXT("")
        );
    }
    
    JsonContent += TEXT("]\n");
    
    return FFileHelper::SaveStringToFile(JsonContent, *CameraInfoPath);
}

bool FVCCSimDataConverter::SavePointCloudToPLY(const FPointCloudData& PointCloudData,
    const FString& OutputFilePath)
{
    FPLYWriter::FPLYWriteConfig Config;
    Config.bIncludeColors = true;
    Config.bIncludeNormals = true;   // Include normals for Triangle Splatting compatibility
    Config.bBinaryFormat = false;
    
    return FPLYWriter::WritePointCloudToPLY(PointCloudData, OutputFilePath, Config);
}

bool FVCCSimDataConverter::CreateModelDirectoryStructure(const FString& OutputPath)
{
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    
    // Create main output directory
    if (!PlatformFile.CreateDirectoryTree(*OutputPath))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create output directory: %s"),
            *OutputPath);
        return false;
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
        UE_LOG(LogTemp, Error, TEXT("Image directory does "
                                    "not exist: %s"), *ImageDirectory);
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
    
    UE_LOG(LogTemp, Log, TEXT("Found %d images in %s"),
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
        Bounds += CameraInfo.Position;
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

bool FVCCSimDataConverter::SampleMeshVerticesWithColors(UStaticMesh* Mesh,
    int32 SampleCount, bool bRandomSampling,
    TArray<FVector>& OutPositions, TArray<FLinearColor>& OutColors)
{
    OutPositions.Empty();
    OutColors.Empty();
    
    if (!Mesh || !Mesh->GetRenderData() || Mesh->GetRenderData()->LODResources.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("Invalid mesh for vertex sampling with colors"));
        return false;
    }
    
    const FStaticMeshLODResources& LODResource = Mesh->GetRenderData()->LODResources[0];
    const FPositionVertexBuffer& PositionBuffer = LODResource.VertexBuffers.PositionVertexBuffer;
    const FColorVertexBuffer& ColorBuffer = LODResource.VertexBuffers.ColorVertexBuffer;
    
    int32 VertexCount = PositionBuffer.GetNumVertices();
    bool bHasVertexColors = ColorBuffer.GetNumVertices() > 0;
    
    if (VertexCount == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("Mesh has no vertices"));
        return false;
    }
    
    SampleCount = FMath::Min(SampleCount, VertexCount);
    OutPositions.Reserve(SampleCount);
    OutColors.Reserve(SampleCount);
    
    if (bRandomSampling)
    {
        // Random sampling
        FRandomStream RandomStream(FDateTime::Now().GetTicks());
        TSet<int32> UsedIndices;
        
        while (OutPositions.Num() < SampleCount)
        {
            int32 RandomIndex = RandomStream.RandRange(0, VertexCount - 1);
            if (!UsedIndices.Contains(RandomIndex))
            {
                UsedIndices.Add(RandomIndex);
                
                FVector Vertex = (FVector)PositionBuffer.VertexPosition(RandomIndex);
                OutPositions.Add(Vertex);
                
                // Get vertex color if available, otherwise use white
                FLinearColor VertexColor = FLinearColor::White;
                if (bHasVertexColors && (uint32)RandomIndex < ColorBuffer.GetNumVertices())
                {
                    FColor Color = ColorBuffer.VertexColor(RandomIndex);
                    // Convert without gamma correction to preserve original brightness
                    VertexColor = FLinearColor(
                        Color.R / 255.0f,
                        Color.G / 255.0f, 
                        Color.B / 255.0f,
                        Color.A / 255.0f
                    );
                }
                OutColors.Add(VertexColor);
            }
        }
    }
    else
    {
        // Uniform sampling
        int32 Step = FMath::Max(1, VertexCount / SampleCount);
        for (int32 i = 0; i < VertexCount && OutPositions.Num() < SampleCount; i += Step)
        {
            FVector Vertex = (FVector)PositionBuffer.VertexPosition(i);
            OutPositions.Add(Vertex);
            
            // Get vertex color if available, otherwise use white
            FLinearColor VertexColor = FLinearColor::White;
            if (bHasVertexColors && (uint32)i < ColorBuffer.GetNumVertices())
            {
                FColor Color = ColorBuffer.VertexColor(i);
                // Convert without gamma correction to preserve original brightness
                VertexColor = FLinearColor(
                    Color.R / 255.0f,
                    Color.G / 255.0f,
                    Color.B / 255.0f,
                    Color.A / 255.0f
                );
            }
            OutColors.Add(VertexColor);
        }
    }
    
    UE_LOG(LogTemp, Log, TEXT("Sampled %d vertices %s"), OutPositions.Num(), 
        bHasVertexColors ? TEXT("with vertex colors") : TEXT("with default colors"));
    
    return true;
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
        
        FLinearColor Color = FLinearColor(NormalizedZ, 1.0f - NormalizedZ,
            0.5f, 1.0f);
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
// COLMAP INTEGRATION
// ============================================================================

FString FVCCSimDataConverter::CreateTimestampedColmapDirectory(const FString& BaseOutputPath)
{
    // Create organized COLMAP directory structure
    FString ColmapParentDir = FPaths::Combine(BaseOutputPath, TEXT("TriangleSplatting"));
    FString ColmapOutputDir = FPaths::Combine(ColmapParentDir, TEXT("colmap_output"));
    
    // Generate timestamp string
    FDateTime Now = FDateTime::Now();
    FString Timestamp = Now.ToString(TEXT("%Y%m%d_%H%M%S"));
    FString TimestampedDirName = FString::Printf(TEXT("dataset_%s"), *Timestamp);
    
    FString TimestampedPath = FPaths::Combine(ColmapOutputDir, TimestampedDirName);
    
    // Create the directory
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.CreateDirectoryTree(*TimestampedPath))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create timestamped "
                                    "COLMAP directory: %s"), *TimestampedPath);
        return FString();
    }
    
    UE_LOG(LogTemp, Log, TEXT("Created COLMAP dataset: %s"),
        *FPaths::GetCleanFilename(TimestampedPath));
    return TimestampedPath;
}

bool FVCCSimDataConverter::PrepareColmapDataset(
    const FString& ImageDirectory,
    const FString& ColmapDatasetPath)
{
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    
    // Validate source image directory
    if (!PlatformFile.DirectoryExists(*ImageDirectory))
    {
        UE_LOG(LogTemp, Error, TEXT("Source image directory does not exist: %s"),
            *ImageDirectory);
        return false;
    }
    
    // Create COLMAP images directory
    FString ImagesDir = FPaths::Combine(ColmapDatasetPath, TEXT("images"));
    if (!PlatformFile.CreateDirectoryTree(*ImagesDir))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create images directory: %s"), *ImagesDir);
        return false;
    }
    
    // Get all valid image files from source directory
    TArray<FString> ValidImageFiles = ValidateImageDirectory(ImageDirectory);
    
    if (ValidImageFiles.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("No valid images found in directory: %s"),
            *ImageDirectory);
        return false;
    }
    
    // Copy all images to COLMAP images directory
    int32 CopiedImages = 0;
    
    for (const FString& SourceImagePath : ValidImageFiles)
    {
        FString DestImagePath = FPaths::Combine(ImagesDir,
            FPaths::GetCleanFilename(SourceImagePath));
        
        if (PlatformFile.CopyFile(*DestImagePath, *SourceImagePath))
        {
            CopiedImages++;
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("Failed to copy: %s"),
                *FPaths::GetCleanFilename(SourceImagePath));
        }
    }
    
    UE_LOG(LogTemp, Log, TEXT("Prepared %d images for COLMAP"), CopiedImages);
    return CopiedImages > 0;
}

bool FVCCSimDataConverter::RunColmapFeatureExtraction(
    const FString& ColmapExecutablePath,
    const FString& DatasetPath,
    const FString& DatabasePath,
    TFunction<bool(const FString&, const FString&, const FString&)> CommandExecutor)
{
    // Construct COLMAP executable path
    FString ColmapExePath = FPaths::Combine(ColmapExecutablePath, TEXT("COLMAP.exe"));
    
    // Verify executable exists
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.FileExists(*ColmapExePath))
    {
        UE_LOG(LogTemp, Error, TEXT("COLMAP executable not found at: %s"),
            *ColmapExePath);
        return false;
    }
    
    // Prepare command arguments
    FString ImagePath = FPaths::Combine(DatasetPath, TEXT("images"));
    FString Arguments = FString::Printf(
        TEXT("feature_extractor --database_path \"%s\" --image_path \"%s\" "
             "--ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE"),
        *DatabasePath, *ImagePath);
    
    // Use custom command executor if provided (for proper process management)
    if (CommandExecutor)
    {
        return CommandExecutor(ColmapExePath, Arguments, TEXT("Feature Extraction"));
    }
    
    // Fallback to basic execution (legacy mode)
    UE_LOG(LogTemp, Log, TEXT("Running COLMAP feature extraction: %s %s"),
        *ColmapExePath, *Arguments);
    
    int32 ReturnCode = -1;
    FString StdOut, StdErr;
    
    bool bSuccess = FPlatformProcess::ExecProcess(
        *ColmapExePath, 
        *Arguments, 
        &ReturnCode, 
        &StdOut, 
        &StdErr
    );
    
    if (!bSuccess || ReturnCode != 0)
    {
        UE_LOG(LogTemp, Error, TEXT("COLMAP feature extraction failed. "
                                    "Return code: %d"), ReturnCode);
        if (!StdErr.IsEmpty())
        {
            UE_LOG(LogTemp, Error, TEXT("COLMAP error output: %s"), *StdErr);
        }
        return false;
    }
    
    UE_LOG(LogTemp, Log, TEXT("Feature extraction complete"));
    return true;
}

bool FVCCSimDataConverter::RunColmapFeatureMatching(
    const FString& ColmapExecutablePath,
    const FString& DatabasePath,
    TFunction<bool(const FString&, const FString&, const FString&)> CommandExecutor)
{
    // Construct COLMAP executable path
    FString ColmapExePath = FPaths::Combine(ColmapExecutablePath, TEXT("COLMAP.exe"));
    
    // Prepare command arguments
    FString Arguments = FString::Printf(
        TEXT("exhaustive_matcher --database_path \"%s\""),
        *DatabasePath);
    
    // Use custom command executor if provided (for proper process management)
    if (CommandExecutor)
    {
        return CommandExecutor(ColmapExePath, Arguments, TEXT("Feature Matching"));
    }
    
    // Fallback to basic execution (legacy mode)
    UE_LOG(LogTemp, Log, TEXT("Running COLMAP feature matching: %s %s"),
        *ColmapExePath, *Arguments);
    
    int32 ReturnCode = -1;
    FString StdOut, StdErr;
    
    bool bSuccess = FPlatformProcess::ExecProcess(
        *ColmapExePath, 
        *Arguments, 
        &ReturnCode, 
        &StdOut, 
        &StdErr
    );
    
    if (!bSuccess || ReturnCode != 0)
    {
        UE_LOG(LogTemp, Error, TEXT("COLMAP feature matching failed. "
                                    "Return code: %d"), ReturnCode);
        if (!StdErr.IsEmpty())
        {
            UE_LOG(LogTemp, Error, TEXT("COLMAP error output: %s"), *StdErr);
        }
        return false;
    }
    
    UE_LOG(LogTemp, Log, TEXT("Feature matching complete"));
    return true;
}

bool FVCCSimDataConverter::RunColmapSparseReconstruction(
    const FString& ColmapExecutablePath,
    const FString& DatabasePath,
    const FString& ImagePath,
    const FString& OutputPath,
    TFunction<bool(const FString&, const FString&, const FString&)> CommandExecutor)
{
    // Construct COLMAP executable path
    FString ColmapExePath = FPaths::Combine(ColmapExecutablePath, TEXT("COLMAP.exe"));
    
    // Create sparse output directory
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    FString SparseOutputPath = FPaths::Combine(OutputPath, TEXT("sparse"));
    if (!PlatformFile.CreateDirectoryTree(*SparseOutputPath))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create sparse "
                                    "output directory: %s"), *SparseOutputPath);
        return false;
    }
    
    // Prepare command arguments
    FString Arguments = FString::Printf(
        TEXT("mapper --database_path \"%s\" --image_path \"%s\" --output_path \"%s\""),
        *DatabasePath, *ImagePath, *SparseOutputPath);
    
    // Use custom command executor if provided (for proper process management)
    if (CommandExecutor)
    {
        return CommandExecutor(ColmapExePath, Arguments, TEXT("Sparse Reconstruction"));
    }
    
    // Fallback to basic execution (legacy mode)
    UE_LOG(LogTemp, Log, TEXT("Running COLMAP sparse reconstruction: %s %s"),
        *ColmapExePath, *Arguments);
    
    int32 ReturnCode = -1;
    FString StdOut, StdErr;
    
    bool bSuccess = FPlatformProcess::ExecProcess(
        *ColmapExePath, 
        *Arguments, 
        &ReturnCode, 
        &StdOut, 
        &StdErr
    );
    
    if (!bSuccess || ReturnCode != 0)
    {
        UE_LOG(LogTemp, Error, TEXT("COLMAP sparse reconstruction failed. "
                                    "Return code: %d"), ReturnCode);
        if (!StdErr.IsEmpty())
        {
            UE_LOG(LogTemp, Error, TEXT("COLMAP error output: %s"), *StdErr);
        }
        return false;
    }
    
    UE_LOG(LogTemp, Log, TEXT("Sparse reconstruction complete"));
    return true;
}

bool FVCCSimDataConverter::RunColmapModelConverter(
    const FString& ColmapExecutablePath,
    const FString& BinaryModelPath,
    const FString& TextModelPath,
    TFunction<bool(const FString&, const FString&, const FString&)> CommandExecutor)
{
    // Construct COLMAP executable path
    FString ColmapExePath = FPaths::Combine(ColmapExecutablePath, TEXT("COLMAP.exe"));
    
    // Create text model output directory
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.CreateDirectoryTree(*TextModelPath))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create text model"
                                    " output directory: %s"), *TextModelPath);
        return false;
    }
    
    // Prepare command arguments for binary to text conversion
    FString Arguments = FString::Printf(
        TEXT("model_converter --input_path \"%s\" --output_path \"%s\""
             " --output_type TXT"),
        *BinaryModelPath, *TextModelPath);
    
    // Use custom command executor if provided (for proper process management)
    if (CommandExecutor)
    {
        return CommandExecutor(ColmapExePath, Arguments,
            TEXT("Model Converter (Binary to Text)"));
    }
    
    // Fallback to basic execution (legacy mode)
    UE_LOG(LogTemp, Log, TEXT("Running COLMAP model converter: %s %s"),
        *ColmapExePath, *Arguments);
    
    int32 ReturnCode = -1;
    FString StdOut, StdErr;
    
    bool bSuccess = FPlatformProcess::ExecProcess(
        *ColmapExePath, 
        *Arguments, 
        &ReturnCode, 
        &StdOut, 
        &StdErr
    );
    
    if (!bSuccess || ReturnCode != 0)
    {
        UE_LOG(LogTemp, Error, TEXT("COLMAP model converter failed. "
                                    "Return code: %d"), ReturnCode);
        if (!StdErr.IsEmpty())
        {
            UE_LOG(LogTemp, Error, TEXT("COLMAP model converter error "
                                        "output: %s"), *StdErr);
        }
        return false;
    }
    
    UE_LOG(LogTemp, Log, TEXT("Model conversion complete"));
    return true;
}

bool FVCCSimDataConverter::RunColmapPipeline(
    const FString& ImageDirectory,
    const FString& OutputPath,
    const FString& ColmapExecutablePath)
{
    if (ImageDirectory.IsEmpty() ||
        !FPlatformFileManager::Get().GetPlatformFile().DirectoryExists(*ImageDirectory))
    {
        UE_LOG(LogTemp, Error, TEXT("Invalid image directory provided "
                                    "for COLMAP pipeline: %s"), *ImageDirectory);
        return false;
    }
    
    // Create timestamped directory
    FString TimestampedDir = CreateTimestampedColmapDirectory(OutputPath);
    if (TimestampedDir.IsEmpty())
    {
        return false;
    }
    
    // Prepare dataset (images directory)
    if (!PrepareColmapDataset(ImageDirectory, TimestampedDir))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to prepare COLMAP dataset"));
        return false;
    }
    
    // Define paths
    FString DatabasePath = FPaths::Combine(TimestampedDir, TEXT("database.db"));
    FString ImagePath = FPaths::Combine(TimestampedDir, TEXT("images"));
    
    UE_LOG(LogTemp, Log, TEXT("Starting COLMAP pipeline from %s"),
        *FPaths::GetCleanFilename(ImageDirectory));
    
    UE_LOG(LogTemp, Log, TEXT("Step 1/3: Feature extraction"));
    if (!RunColmapFeatureExtraction(ColmapExecutablePath,
        TimestampedDir, DatabasePath))
    {
        UE_LOG(LogTemp, Error, TEXT("COLMAP feature extraction failed"));
        return false;
    }
    
    UE_LOG(LogTemp, Log, TEXT("Step 2/3: Feature matching"));
    if (!RunColmapFeatureMatching(ColmapExecutablePath, DatabasePath))
    {
        UE_LOG(LogTemp, Error, TEXT("COLMAP feature matching failed"));
        return false;
    }
    
    UE_LOG(LogTemp, Log, TEXT("Step 3/3: Sparse reconstruction"));
    if (!RunColmapSparseReconstruction(ColmapExecutablePath, DatabasePath,
        ImagePath, TimestampedDir))
    {
        UE_LOG(LogTemp, Error, TEXT("COLMAP sparse reconstruction failed"));
        return false;
    }
    
    UE_LOG(LogTemp, Log, TEXT("COLMAP pipeline completed: %s"),
        *FPaths::GetCleanFilename(TimestampedDir));
    
    return true;
}