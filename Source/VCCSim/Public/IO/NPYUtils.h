/*
* Copyright (C) 2025 Visual Computing Research Center, Shenzhen University
*
* NPY file utilities for mesh data serialization
* Optimized for Triangle Splatting mesh triangle data I/O
*/

#pragma once

#include "CoreMinimal.h"

/**
 * NPY file utilities for float32 arrays
 * Supports 2D float arrays (N, 3) format for positions, colors, normals
 * Provides both read and write capabilities
 */
class VCCSIM_API FNPYUtils
{
public:
    /**
     * Write a 2D float array to NPY file
     * @param FilePath Output file path (.npy)
     * @param Data Pointer to float data
     * @param NumRows Number of rows (N)
     * @param NumCols Number of columns (typically 3 for XYZ/RGB data)
     * @return True if successful
     */
    static bool WriteFloat32Array2D(
        const FString& FilePath,
        const float* Data,
        int32 NumRows,
        int32 NumCols = 3
    );
    
    /**
     * Write vertex positions to NPY file
     * @param FilePath Output file path
     * @param Positions Array of FVector positions
     * @return True if successful
     */
    static bool WritePositions(
        const FString& FilePath,
        const TArray<FVector>& Positions
    );
    
    /**
     * Write vertex colors to NPY file (converts to 0-1 range)
     * @param FilePath Output file path
     * @param Colors Array of FLinearColor colors
     * @return True if successful
     */
    static bool WriteColors(
        const FString& FilePath,
        const TArray<FLinearColor>& Colors
    );
    
    /**
     * Write vertex colors from FVector to NPY file (assumes already in 0-1 range)
     * @param FilePath Output file path
     * @param Colors Array of FVector colors (RGB as XYZ)
     * @return True if successful
     */
    static bool WriteColorsFromVector(
        const FString& FilePath,
        const TArray<FVector>& Colors
    );
    
    /**
     * Write vertex normals to NPY file
     * @param FilePath Output file path
     * @param Normals Array of FVector normals
     * @return True if successful
     */
    static bool WriteNormals(
        const FString& FilePath,
        const TArray<FVector>& Normals
    );

private:
    /**
     * Create NPY header for float32 2D array
     * @param NumRows Number of rows
     * @param NumCols Number of columns  
     * @return NPY header string
     */
    static FString CreateNPYHeader(int32 NumRows, int32 NumCols);
    
    /**
     * Convert UE coordinate to Triangle Splatting coordinate (if needed)
     * @param UEVector Input UE vector
     * @return Converted vector
     */
    static FVector ConvertCoordinate(const FVector& UEVector);
};