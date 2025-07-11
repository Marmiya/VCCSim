﻿/*
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

#include "Utils/ImageProcesser.h"
#include "ImageUtils.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"

void FAsyncImageSaveTask::DoWork()
{
    TArray64<uint8> CompressedBitmap;
    FImageUtils::PNGCompressImageArray(Size.X, Size.Y, Pixels,
       CompressedBitmap);

    if (!FFileHelper::SaveArrayToFile(CompressedBitmap, *FilePath))
    {
       UE_LOG(LogTemp, Error, TEXT("Failed to save render target to file."));
    }
}

void FAsyncDepth16SaveTask::DoWork()
{
    // Convert FFloat16Color depth data to 16-bit grayscale
    TArray<uint16> DepthData16;
    DepthData16.Reserve(DepthPixels.Num());
    
    for (const FFloat16Color& DepthPixel : DepthPixels)
    {
        // Extract depth value from R channel and scale to 16-bit range
        float DepthValue = DepthPixel.R.GetFloat() * DepthScale;
        uint16 Depth16 = FMath::Clamp(
            FMath::RoundToInt(DepthValue), 
            0, 
            65535
        );
        DepthData16.Add(Depth16);
    }
    
    // Get image wrapper module
    IImageWrapperModule& ImageWrapperModule =
        FModuleManager::LoadModuleChecked<IImageWrapperModule>(FName("ImageWrapper"));
    TSharedPtr<IImageWrapper> ImageWrapper =
        ImageWrapperModule.CreateImageWrapper(EImageFormat::PNG);
    
    if (ImageWrapper.IsValid())
    {
        // Convert uint16 array to uint8 array for image wrapper (little endian)
        TArray<uint8> RawData;
        RawData.Reserve(DepthData16.Num() * 2);
        
        for (uint16 Value : DepthData16)
        {
            RawData.Add(Value & 0xFF);        // Low byte
            RawData.Add((Value >> 8) & 0xFF); // High byte
        }
        
        // Set the image data (16-bit grayscale)
        if (ImageWrapper->SetRaw(
            RawData.GetData(), 
            RawData.Num(), 
            Size.X, 
            Size.Y, 
            ERGBFormat::Gray, 
            16
        ))
        {
            // Get compressed PNG data
            TArray64<uint8> CompressedData = ImageWrapper->GetCompressed();
            
            // Save to file
            if (!FFileHelper::SaveArrayToFile(CompressedData, *FilePath))
            {
                UE_LOG(LogTemp, Error, TEXT("Failed to save 16-bit depth "
                                            "image to file: %s"), *FilePath);
            }
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to set raw data for 16-bit "
                                        "depth image: %s"), *FilePath);
        }
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create PNG image wrapper "
                                    "for 16-bit depth saving"));
    }
}

void FAsyncPLYSaveTask::DoWork()
{
    if (PointCloud.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("FAsyncPLYSaveTask: "
                                      "No points to save to %s"), *FilePath);
        return;
    }

    // Create PLY file content
    FString PLYContent;
        
    // PLY header
    PLYContent += TEXT("ply\n");
    PLYContent += TEXT("format ascii 1.0\n");
    PLYContent += FString::Printf(TEXT("comment Point Cloud generated by UE - %s\n"), 
        *FDateTime::Now().ToString());
    PLYContent += TEXT("comment UE Coordinate System: X=Forward, Y=Right, Z=Up (left-handed)\n");
    PLYContent += TEXT("comment Units: centimeters\n");
    PLYContent += FString::Printf(TEXT("element vertex %d\n"), PointCloud.Num());
    PLYContent += TEXT("property float x\n");
    PLYContent += TEXT("property float y\n");
    PLYContent += TEXT("property float z\n");
    PLYContent += TEXT("property uchar red\n");
    PLYContent += TEXT("property uchar green\n");
    PLYContent += TEXT("property uchar blue\n");
    PLYContent += TEXT("end_header\n");
        
    // Write vertex data
    for (const FDCPoint& Point : PointCloud)
    {
        const FVector& Loc = Point.Location;
            
        // Add point with default white color (you can modify this for depth-based coloring)
        PLYContent += FString::Printf(TEXT("%.6f %.6f %.6f 255 255 255\n"), 
            Loc.X, Loc.Y, Loc.Z);
    }
        
    // Save to file
    if (!FFileHelper::SaveStringToFile(PLYContent, *FilePath))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to save PLY point cloud to file: %s"), *FilePath);
    }
    else
    {
        UE_LOG(LogTemp, Log, TEXT("Successfully saved %d points to PLY file: %s"), 
            PointCloud.Num(), *FilePath);
    }
}

void FAsyncNormalEXRSaveTask::DoWork()
{
    // Use modern UE image API with FImage
    FImage Image;
    Image.Init(Size.X, Size.Y, ERawImageFormat::RGBA32F);
    
    // Convert FLinearColor to FImage data
    TArrayView64<FLinearColor> ImageData = Image.AsRGBA32F();
    
    if (ImageData.Num() != NormalPixels.Num())
    {
        UE_LOG(LogTemp, Error, TEXT("Image data size mismatch: Expected %lld, got %d"), 
            ImageData.Num(), NormalPixels.Num());
        return;
    }
    
    // Copy normal data to image
    for (int32 i = 0; i < NormalPixels.Num(); ++i)
    {
        ImageData[i] = NormalPixels[i];
    }
    
    // Get image wrapper module
    IImageWrapperModule& ImageWrapperModule =
        FModuleManager::LoadModuleChecked<IImageWrapperModule>(FName("ImageWrapper"));
    
    // Compress image using modern API
    TArray64<uint8> CompressedData;
    bool bSuccess = ImageWrapperModule.CompressImage(
        CompressedData,
        EImageFormat::EXR,
        Image,
        100
    );
    
    if (bSuccess && CompressedData.Num() > 0)
    {
        // Save to file
        if (!FFileHelper::SaveArrayToFile(CompressedData, *FilePath))
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to save EXR normal image to file: %s"), *FilePath);
        }
        else
        {
            UE_LOG(LogTemp, Log, TEXT("Successfully saved EXR normal image to: %s"), *FilePath);
        }
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to compress normal image to EXR format"));
    }
}