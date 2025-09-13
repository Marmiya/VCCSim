/*
* Copyright (C) 2025 Visual Computing Research Center, Shenzhen University
*/

#include "IO/NPYUtils.h"
#include "HAL/PlatformFilemanager.h"
#include "Misc/FileHelper.h"
#include "Math/Vector.h"
#include "Math/Color.h"

bool FNPYUtils::WriteFloat32Array2D(const FString& FilePath, const float* Data, int32 NumRows, int32 NumCols)
{
    if (!Data || NumRows <= 0 || NumCols <= 0)
    {
        UE_LOG(LogTemp, Error, TEXT("NPYWriter: Invalid data parameters"));
        return false;
    }
    
    // Create proper NPY header as raw bytes
    TArray<uint8> FileData;
    
    // Magic number: "\x93NUMPY"
    FileData.Append({0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59});
    
    // Version: 1.0
    FileData.Append({0x01, 0x00});
    
    // Create dictionary string
    FString DictStr = FString::Printf(TEXT("{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }"), NumRows, NumCols);
    
    // Convert to UTF-8 and calculate padding
    FTCHARToUTF8 UTF8Dict(*DictStr);
    int32 DictLen = UTF8Dict.Length();
    
    // Header length must be divisible by 64 bytes (after magic + version + header_len)
    int32 HeaderLen = 10 + DictLen + 1;  // magic(6) + version(2) + header_len(2) + dict + newline
    int32 Padding = 64 - (HeaderLen % 64);
    if (Padding == 64) Padding = 0;
    
    int32 TotalDictLen = DictLen + Padding + 1; // dict + padding + newline
    
    // Header length (little endian, 2 bytes for version 1.0)
    FileData.Add(static_cast<uint8>(TotalDictLen & 0xFF));
    FileData.Add(static_cast<uint8>((TotalDictLen >> 8) & 0xFF));
    
    // Dictionary
    for (int32 i = 0; i < DictLen; i++)
    {
        FileData.Add(static_cast<uint8>(UTF8Dict.Get()[i]));
    }
    
    // Padding spaces
    for (int32 i = 0; i < Padding; i++)
    {
        FileData.Add(0x20); // Space character
    }
    
    // Newline
    FileData.Add(0x0A);
    
    // Add float data (assuming little-endian system)
    int32 DataSize = NumRows * NumCols * sizeof(float);
    const uint8* DataBytes = reinterpret_cast<const uint8*>(Data);
    FileData.Append(DataBytes, DataSize);
    
    // Write to file
    if (!FFileHelper::SaveArrayToFile(FileData, *FilePath))
    {
        UE_LOG(LogTemp, Error, TEXT("NPYWriter: Failed to write file: %s"), *FilePath);
        return false;
    }
    
    return true;
}

bool FNPYUtils::WritePositions(const FString& FilePath, const TArray<FVector>& Positions)
{
    if (Positions.Num() == 0)
    {
        return false;
    }
    
    // NO coordinate transformation - keep same as PLY writer for consistency
    TArray<float> FloatData;
    FloatData.Reserve(Positions.Num() * 3);
    
    for (const FVector& Pos : Positions)
    {
        FloatData.Add(static_cast<float>(Pos.X));
        FloatData.Add(static_cast<float>(Pos.Y));
        FloatData.Add(static_cast<float>(Pos.Z));
    }
    
    return WriteFloat32Array2D(FilePath, FloatData.GetData(), Positions.Num(), 3);
}

bool FNPYUtils::WriteColors(const FString& FilePath, const TArray<FLinearColor>& Colors)
{
    if (Colors.Num() == 0)
    {
        return false;
    }
    
    // Convert to float array (already in 0-1 range)
    TArray<float> FloatData;
    FloatData.Reserve(Colors.Num() * 3);
    
    for (const FLinearColor& Color : Colors)
    {
        FloatData.Add(Color.R);
        FloatData.Add(Color.G);
        FloatData.Add(Color.B);
    }
    
    return WriteFloat32Array2D(FilePath, FloatData.GetData(), Colors.Num(), 3);
}

bool FNPYUtils::WriteColorsFromVector(const FString& FilePath, const TArray<FVector>& Colors)
{
    if (Colors.Num() == 0)
    {
        return false;
    }
    
    // Convert FVector colors to float array
    // FVector colors are assumed to be in 0-1 range (R=X, G=Y, B=Z)
    TArray<float> FloatData;
    FloatData.Reserve(Colors.Num() * 3);
    
    for (const FVector& Color : Colors)
    {
        FloatData.Add(static_cast<float>(Color.X)); // R
        FloatData.Add(static_cast<float>(Color.Y)); // G
        FloatData.Add(static_cast<float>(Color.Z)); // B
    }
    
    return WriteFloat32Array2D(FilePath, FloatData.GetData(), Colors.Num(), 3);
}

bool FNPYUtils::WriteNormals(const FString& FilePath, const TArray<FVector>& Normals)
{
    if (Normals.Num() == 0)
    {
        return false;
    }
    
    // NO coordinate transformation - keep same as PLY writer for consistency
    TArray<float> FloatData;
    FloatData.Reserve(Normals.Num() * 3);
    
    for (const FVector& Normal : Normals)
    {
        FloatData.Add(static_cast<float>(Normal.X));
        FloatData.Add(static_cast<float>(Normal.Y));
        FloatData.Add(static_cast<float>(Normal.Z));
    }
    
    return WriteFloat32Array2D(FilePath, FloatData.GetData(), Normals.Num(), 3);
}

FString FNPYUtils::CreateNPYHeader(int32 NumRows, int32 NumCols)
{
    // This was too complex - let me use a simpler approach
    // We'll write the raw bytes directly instead of using FString
    return FString(); // Placeholder - we'll write bytes directly in WriteFloat32Array2D
}

FVector FNPYUtils::ConvertCoordinate(const FVector& UEVector)
{
    // Convert UE coordinate (Left-handed, Z-up, centimeters) to Triangle Splatting (Right-handed, Z-up, meters)
    return FVector(
        UEVector.X * 0.01f,    // X: cm to m
        -UEVector.Y * 0.01f,   // Y: flip and cm to m (handedness change)
        UEVector.Z * 0.01f     // Z: cm to m
    );
}