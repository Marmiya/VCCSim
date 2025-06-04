// PointCloud.h - Updated for no coordinate transform
#pragma once

#include "CoreMinimal.h"
#include "Engine/Engine.h"
#include "HAL/PlatformFilemanager.h"
#include "GenericPlatform/GenericPlatformFile.h"
#include "Misc/FileHelper.h"

/**
 * Point data structure for point cloud loading
 */
struct FRatPoint
{
    FVector Position;
    FLinearColor Color;

    FRatPoint()
        : Position(FVector::ZeroVector)
        , Color(FLinearColor::White)
    {
    }

    FRatPoint(const FVector& InPosition, const FLinearColor& InColor)
        : Position(InPosition)
        , Color(InColor)
    {
    }
};

/**
 * PLY file loader utility class
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
        FString Name;       // "x", "y", "z", "red", etc.
        int32 Size;         // Size in bytes

        FPLYProperty(const FString& InType, const FString& InName);
    };

    /**
     * PLY loading result structure
     */
    struct FPLYLoadResult
    {
        bool bSuccess = false;
        bool bHasColors = false;
        int32 PointCount = 0;
        TArray<FRatPoint> Points;
        FString ErrorMessage;

        FPLYLoadResult() = default;
    };

public:
    /**
     * Load a PLY file and return the result
     * @param FilePath Path to the PLY file
     * @param DefaultColor Default color to use if no colors are present in the file
     * @return PLY loading result (NO coordinate scaling applied)
     */
    static FPLYLoadResult LoadPLYFile(const FString& FilePath, 
                                      const FLinearColor& DefaultColor = FLinearColor(1.0f, 0.5f, 0.0f, 1.0f));

private:
    /**
     * Load ASCII format PLY file
     */
    static bool LoadASCIIPLY(const TArray<FString>& Lines, 
                            int32 HeaderEndIndex, 
                            int32 VertexCount, 
                            const TArray<FPLYProperty>& Properties,
                            const FLinearColor& DefaultColor,
                            TArray<FRatPoint>& OutPoints,
                            bool bHasColors);

    /**
     * Load binary format PLY file
     */
    static bool LoadBinaryPLY(const FString& FilePath, 
                             int32 HeaderSize, 
                             int32 VertexCount, 
                             const TArray<FPLYProperty>& Properties, 
                             EPLYFormat Format,
                             const FLinearColor& DefaultColor,
                             TArray<FRatPoint>& OutPoints,
                             bool bHasColors);

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

    // Binary reading helper functions
    static float ReadFloat(const uint8* Data, bool bLittleEndian);
    static double ReadDouble(const uint8* Data, bool bLittleEndian);
    static uint8 ReadUChar(const uint8* Data);
    static uint16 ReadUShort(const uint8* Data, bool bLittleEndian);
    static uint32 ReadUInt(const uint8* Data, bool bLittleEndian);
    static int32 ReadInt(const uint8* Data, bool bLittleEndian);
};