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
#include "UObject/NoExportTypes.h"
#include "Engine/StaticMesh.h"
#include "VCCSimDataStructures.generated.h"

/**
 * Camera configuration structure for VCCSim panel
 */
USTRUCT(BlueprintType)
struct VCCSIMEDITOR_API FCameraConfiguration
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, Category = "RGB Camera")
    bool bUseRGB = true;
    
    UPROPERTY(EditAnywhere, Category = "Depth Camera", meta = (EditCondition = "bHasDepthCamera"))
    bool bUseDepth = false;
    
    UPROPERTY(EditAnywhere, Category = "Segmentation Camera", meta = (EditCondition = "bHasSegmentationCamera"))
    bool bUseSegmentation = false;
    
    UPROPERTY(EditAnywhere, Category = "Normal Camera", meta = (EditCondition = "bHasNormalCamera"))
    bool bUseNormal = false;
    
    // Camera capability flags (not editable, set by system)
    bool bHasRGBCamera = false;
    bool bHasDepthCamera = false;
    bool bHasSegmentationCamera = false;
    bool bHasNormalCamera = false;
    
    FCameraConfiguration()
    {
        bUseRGB = true;
        bUseDepth = false;
        bUseSegmentation = false;
        bUseNormal = false;
        bHasRGBCamera = false;
        bHasDepthCamera = false;
        bHasSegmentationCamera = false;
        bHasNormalCamera = false;
    }
    
    // Helper methods
    bool HasAnyActiveCamera() const
    {
        return bUseRGB || bUseDepth || bUseSegmentation || bUseNormal;
    }
    
    int32 GetActiveCameraCount() const
    {
        int32 Count = 0;
        if (bUseRGB) Count++;
        if (bUseDepth) Count++;
        if (bUseSegmentation) Count++;
        if (bUseNormal) Count++;
        return Count;
    }
};

/**
 * Pose generation configuration structure
 */
USTRUCT(BlueprintType)
struct VCCSIMEDITOR_API FPoseConfiguration
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, Category = "Basic Settings", meta = (ClampMin = "1", ClampMax = "1000", ToolTip = "Number of poses to generate"))
    int32 NumPoses = 50;
    
    UPROPERTY(EditAnywhere, Category = "Basic Settings", meta = (ClampMin = "0.1", ClampMax = "10000.0", ToolTip = "Radius for pose generation"))
    float Radius = 500.0f;
    
    UPROPERTY(EditAnywhere, Category = "Basic Settings", meta = (ToolTip = "Height offset from target"))
    float HeightOffset = 0.0f;
    
    UPROPERTY(EditAnywhere, Category = "Advanced Settings", meta = (ClampMin = "0.1", ToolTip = "Vertical gap between pose layers"))
    float VerticalGap = 50.0f;
    
    UPROPERTY(EditAnywhere, Category = "Safety Settings", meta = (ClampMin = "0.1", ToolTip = "Safe distance from obstacles"))
    float SafeDistance = 200.0f;
    
    UPROPERTY(EditAnywhere, Category = "Safety Settings", meta = (ClampMin = "0.1", ToolTip = "Safe height above ground"))
    float SafeHeight = 200.0f;
    
    FPoseConfiguration()
    {
        NumPoses = 50;
        Radius = 500.0f;
        HeightOffset = 0.0f;
        VerticalGap = 50.0f;
        SafeDistance = 200.0f;
        SafeHeight = 200.0f;
    }
    
    // Validation methods
    bool IsValid() const
    {
        return NumPoses > 0 && Radius > 0.1f && VerticalGap > 0.1f && SafeDistance > 0.1f && SafeHeight > 0.1f;
    }
};

/**
 * Limited region configuration for pose generation
 */
USTRUCT(BlueprintType)
struct VCCSIMEDITOR_API FLimitedRegionConfiguration
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, Category = "Region Limits", meta = (ToolTip = "Use limited region for pose generation"))
    bool bUseLimited = false;
    
    UPROPERTY(EditAnywhere, Category = "X Bounds", meta = (EditCondition = "bUseLimited"))
    float MinX = 0.0f;
    
    UPROPERTY(EditAnywhere, Category = "X Bounds", meta = (EditCondition = "bUseLimited"))
    float MaxX = 5000.0f;
    
    UPROPERTY(EditAnywhere, Category = "Y Bounds", meta = (EditCondition = "bUseLimited"))
    float MinY = -9500.0f;
    
    UPROPERTY(EditAnywhere, Category = "Y Bounds", meta = (EditCondition = "bUseLimited"))
    float MaxY = -7000.0f;
    
    UPROPERTY(EditAnywhere, Category = "Z Bounds", meta = (EditCondition = "bUseLimited"))
    float MinZ = -20.0f;
    
    UPROPERTY(EditAnywhere, Category = "Z Bounds", meta = (EditCondition = "bUseLimited"))
    float MaxZ = 2000.0f;
    
    FLimitedRegionConfiguration()
    {
        bUseLimited = false;
        MinX = 0.0f;
        MaxX = 5000.0f;
        MinY = -9500.0f;
        MaxY = -7000.0f;
        MinZ = -20.0f;
        MaxZ = 2000.0f;
    }
    
    // Helper methods
    FBox GetBoundingBox() const
    {
        return FBox(FVector(MinX, MinY, MinZ), FVector(MaxX, MaxY, MaxZ));
    }
    
    bool IsPointInside(const FVector& Point) const
    {
        if (!bUseLimited) return true;
        return Point.X >= MinX && Point.X <= MaxX &&
               Point.Y >= MinY && Point.Y <= MaxY &&
               Point.Z >= MinZ && Point.Z <= MaxZ;
    }
};

/**
 * Point Cloud information structure for display
 */
USTRUCT(BlueprintType)
struct VCCSIMEDITOR_API FPointCloudInfo
{
    GENERATED_BODY()

    UPROPERTY(VisibleAnywhere, Category = "File Info")
    FString FilePath;
    
    UPROPERTY(VisibleAnywhere, Category = "Point Data")
    int32 PointCount = 0;
    
    UPROPERTY(VisibleAnywhere, Category = "Point Data")
    bool bHasColors = false;
    
    UPROPERTY(VisibleAnywhere, Category = "Point Data")
    bool bHasNormals = false;
    
    UPROPERTY(VisibleAnywhere, Category = "Bounds")
    FVector BoundingBoxMin = FVector::ZeroVector;
    
    UPROPERTY(VisibleAnywhere, Category = "Bounds")
    FVector BoundingBoxMax = FVector::ZeroVector;
    
    FPointCloudInfo()
    {
        PointCount = 0;
        bHasColors = false;
        bHasNormals = false;
        BoundingBoxMin = FVector::ZeroVector;
        BoundingBoxMax = FVector::ZeroVector;
    }
    
    // UI helper methods
    FText GetDisplayName() const
    {
        return FText::FromString(FPaths::GetCleanFilename(FilePath));
    }
    
    FText GetStatusText() const
    {
        FString StatusText = FString::Printf(TEXT("%d points"), PointCount);
        if (bHasColors) StatusText += TEXT(" | Colors");
        if (bHasNormals) StatusText += TEXT(" | Normals");
        return FText::FromString(StatusText);
    }
    
    FSlateColor GetStatusColor() const
    {
        if (PointCount == 0) return FSlateColor(FLinearColor::Red);
        return FSlateColor(FLinearColor::Green);
    }
    
    FVector GetBoundingBoxSize() const
    {
        return BoundingBoxMax - BoundingBoxMin;
    }
    
    bool IsValid() const
    {
        return PointCount > 0 && !FilePath.IsEmpty();
    }
};

/**
 * Scene Analysis configuration
 */
USTRUCT(BlueprintType)
struct VCCSIMEDITOR_API FSceneAnalysisConfiguration
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, Category = "Analysis Options")
    bool bNeedScan = true;
    
    UPROPERTY(EditAnywhere, Category = "Safe Zone")
    bool bGenSafeZone = true;
    
    UPROPERTY(EditAnywhere, Category = "Coverage")
    bool bInitCoverage = true;
    
    UPROPERTY(EditAnywhere, Category = "Coverage")
    bool bGenCoverage = true;
    
    UPROPERTY(EditAnywhere, Category = "Complexity")
    bool bAnalyzeComplexity = true;
    
    FSceneAnalysisConfiguration()
    {
        bNeedScan = true;
        bGenSafeZone = true;
        bInitCoverage = true;
        bGenCoverage = true;
        bAnalyzeComplexity = true;
    }
};

/**
 * Triangle Splatting configuration structure (enhanced version)
 */
USTRUCT(BlueprintType)
struct VCCSIMEDITOR_API FTriangleSplattingConfiguration
{
    GENERATED_BODY()

    // Input paths
    UPROPERTY(EditAnywhere, Category = "Input Data", meta = (ToolTip = "Directory containing input images"))
    FString ImageDirectory;
    
    UPROPERTY(EditAnywhere, Category = "Input Data", meta = (ToolTip = "Path to pose file"))
    FString PoseFilePath;
    
    UPROPERTY(EditAnywhere, Category = "Input Data", meta = (ToolTip = "Output directory for results"))
    FString OutputDirectory;
    
    // Mesh configuration
    UPROPERTY(EditAnywhere, Category = "Mesh", meta = (ToolTip = "Static mesh for initialization"))
    TSoftObjectPtr<UStaticMesh> SelectedMesh;
    
    UPROPERTY(EditAnywhere, Category = "Mesh", meta = (ToolTip = "Use mesh for initialization"))
    bool bUseMeshInitialization = true;
    
    // Camera parameters
    UPROPERTY(EditAnywhere, Category = "Camera", meta = (ClampMin = "10.0", ClampMax = "180.0", ToolTip = "Field of view in degrees"))
    float FOVDegrees = 90.0f;
    
    UPROPERTY(EditAnywhere, Category = "Camera", meta = (ClampMin = "1", ClampMax = "8192", ToolTip = "Image width"))
    int32 ImageWidth = 1920;
    
    UPROPERTY(EditAnywhere, Category = "Camera", meta = (ClampMin = "1", ClampMax = "8192", ToolTip = "Image height"))
    int32 ImageHeight = 1080;
    
    // Training parameters
    UPROPERTY(EditAnywhere, Category = "Training", meta = (ClampMin = "100", ClampMax = "100000", ToolTip = "Maximum training iterations"))
    int32 MaxIterations = 30000;
    
    UPROPERTY(EditAnywhere, Category = "Training", meta = (ClampMin = "0.0001", ClampMax = "1.0", ToolTip = "Learning rate"))
    float LearningRate = 0.01f;
    
    FTriangleSplattingConfiguration()
    {
        ImageDirectory = TEXT("");
        PoseFilePath = TEXT("");
        OutputDirectory = FPaths::ProjectSavedDir() / TEXT("TriangleSplatting");
        bUseMeshInitialization = true;
        FOVDegrees = 90.0f;
        ImageWidth = 1920;
        ImageHeight = 1080;
        MaxIterations = 30000;
        LearningRate = 0.01f;
    }
    
    // Validation methods
    bool IsValid() const
    {
        return !ImageDirectory.IsEmpty() && 
               !PoseFilePath.IsEmpty() && 
               !OutputDirectory.IsEmpty() &&
               FOVDegrees > 10.0f && FOVDegrees < 180.0f &&
               ImageWidth > 0 && ImageHeight > 0 &&
               MaxIterations > 0 && LearningRate > 0.0f;
    }
    
    FString GetValidationError() const
    {
        if (ImageDirectory.IsEmpty()) return TEXT("Image directory is required");
        if (PoseFilePath.IsEmpty()) return TEXT("Pose file path is required");
        if (OutputDirectory.IsEmpty()) return TEXT("Output directory is required");
        if (FOVDegrees <= 10.0f || FOVDegrees >= 180.0f) return TEXT("FOV must be between 10 and 180 degrees");
        if (ImageWidth <= 0 || ImageHeight <= 0) return TEXT("Image dimensions must be positive");
        if (MaxIterations <= 0) return TEXT("Max iterations must be positive");
        if (LearningRate <= 0.0f) return TEXT("Learning rate must be positive");
        return TEXT("");
    }
};

/**
 * Camera status information for UI display
 */
USTRUCT()
struct VCCSIMEDITOR_API FCameraStatusInfo
{
    GENERATED_BODY()

    FString CameraName;
    bool bIsAvailable = false;
    bool bIsEnabled = false;
    FString StatusMessage;
    
    FCameraStatusInfo()
    {
        bIsAvailable = false;
        bIsEnabled = false;
    }
    
    FCameraStatusInfo(const FString& InName, bool bInAvailable, bool bInEnabled, const FString& InStatusMessage = TEXT(""))
        : CameraName(InName)
        , bIsAvailable(bInAvailable)
        , bIsEnabled(bInEnabled)
        , StatusMessage(InStatusMessage)
    {
    }
    
    FText GetDisplayName() const
    {
        return FText::FromString(CameraName);
    }
    
    FSlateColor GetStatusColor() const
    {
        if (!bIsAvailable) return FSlateColor(FLinearColor::Gray);
        return bIsEnabled ? FSlateColor(FLinearColor::Green) : FSlateColor(FLinearColor::Yellow);
    }
    
    FText GetStatusText() const
    {
        if (!bIsAvailable) return FText::FromString(TEXT("Not Available"));
        return bIsEnabled ? FText::FromString(TEXT("Enabled")) : FText::FromString(TEXT("Disabled"));
    }
};