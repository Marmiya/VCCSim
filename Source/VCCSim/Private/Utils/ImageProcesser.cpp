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
            FMath::RoundToInt(DepthValue * 65535.0f), 
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
            16 // 16 bits per pixel
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