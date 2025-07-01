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

/*
 * Enhanced point data structure for point cloud loading with normal support
 */
struct FRatPoint
{
    FVector Position;
    FLinearColor Color;
    FVector Normal;
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
 * PLY file loader utility class with enhanced normal support
 */
class FPLYLoader
{
public:
    /**
     * PLY file format enumeration
     */
    enum class EPLYFormat
    {
        ASCII,
        BinaryLittleEndian,
        BinaryBigEndian,
        Unknown
    };

    /**
     * PLY property information structure
     */
    struct FPLYProperty
    {
        FString Type;       // "float", "double", "uchar", etc.
        FString Name;       // "x", "y", "z", "red", "nx", "ny", "nz", etc.
        int32 Size;         // Size in bytes

        FPLYProperty(const FString& InType, const FString& InName);
    };

    /**
     * Enhanced PLY loading result structure with normal support
     */
    struct FPLYLoadResult
    {
        bool bSuccess = false;
        bool bHasColors = false;
        bool bHasNormals = false;
        int32 PointCount = 0;
        TArray<FRatPoint> Points;
        FString ErrorMessage;

        FPLYLoadResult() = default;
    };

public:
    /**
     * Load a PLY file and return the result with normal support
     * @param FilePath Path to the PLY file
     * @param DefaultColor Default color to use if no colors are present in the file
     * @return PLY loading result with normal information
     */
    static FPLYLoadResult LoadPLYFile(const FString& FilePath, 
                                      const FLinearColor& DefaultColor = FLinearColor(1.0f, 0.5f, 0.0f, 1.0f));

private:
    /**
     * Load ASCII format PLY file with normal support
     */
    static bool LoadASCIIPLY(const TArray<FString>& Lines, 
                            int32 HeaderEndIndex, 
                            int32 VertexCount, 
                            const TArray<FPLYProperty>& Properties,
                            const FLinearColor& DefaultColor,
                            TArray<FRatPoint>& OutPoints,
                            bool bHasColors,
                            bool bHasNormals);

    /**
     * Load binary format PLY file with normal support
     */
    static bool LoadBinaryPLY(const FString& FilePath, 
                             int32 HeaderSize, 
                             int32 VertexCount, 
                             const TArray<FPLYProperty>& Properties, 
                             EPLYFormat Format,
                             const FLinearColor& DefaultColor,
                             TArray<FRatPoint>& OutPoints,
                             bool bHasColors,
                             bool bHasNormals);

    /**
     * Detect PLY file format from header lines
     */
    static EPLYFormat DetectPLYFormat(const TArray<FString>& HeaderLines);

    /**
     * Parse PLY properties from header lines
     */
    static TArray<FPLYProperty> ParsePLYProperties(const TArray<FString>& HeaderLines);

    /**
     * Find the end of the header in the file lines
     */
    static int32 FindHeaderEnd(const TArray<FString>& Lines);

    /**
     * Get vertex count from header lines
     */
    static int32 GetVertexCount(const TArray<FString>& HeaderLines);

    /**
     * Check if the PLY file has color properties
     */
    static bool HasColorProperties(const TArray<FPLYProperty>& Properties);

    /**
     * Check if the PLY file has normal properties
     */
    static bool HasNormalProperties(const TArray<FPLYProperty>& Properties);

    // Binary reading helper functions
    static float ReadFloat(const uint8* Data, bool bLittleEndian);
    static double ReadDouble(const uint8* Data, bool bLittleEndian);
    static uint8 ReadUChar(const uint8* Data);
    static uint16 ReadUShort(const uint8* Data, bool bLittleEndian);
    static uint32 ReadUInt(const uint8* Data, bool bLittleEndian);
    static int32 ReadInt(const uint8* Data, bool bLittleEndian);
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

