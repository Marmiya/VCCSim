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

#include "Utils/AsyncFileWriter.h"
#include "DataStructures/RecordData.h"
#include "HAL/PlatformFilemanager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Engine/Engine.h"
#include "ImageUtils.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"

DEFINE_LOG_CATEGORY(LogAsyncFileWriter);

FAsyncFileWriter::FAsyncFileWriter(const FString& InBasePath)
    : BasePath(InBasePath)
{
    // Ensure BasePath is absolute
    if (FPaths::IsRelative(BasePath))
    {
        BasePath = FPaths::ConvertRelativePathToFull(BasePath);
    }

    WriterThread.Reset(FRunnableThread::Create(this, TEXT("AsyncFileWriter"), 0, TPri_BelowNormal));
    UE_LOG(LogAsyncFileWriter, Log, TEXT("AsyncFileWriter initialized with base path: %s"), *BasePath);
}

FAsyncFileWriter::~FAsyncFileWriter()
{
    Stop();
    if (WriterThread.IsValid())
    {
        WriterThread->WaitForCompletion();
        WriterThread.Reset();
    }
    UE_LOG(LogAsyncFileWriter, Log, TEXT("AsyncFileWriter destroyed"));
}

void FAsyncFileWriter::WriteDataAsync(const FSensorDataPacket& DataPacket)
{
    UE_LOG(LogAsyncFileWriter, Log, TEXT("WriteDataAsync called for sensor type %d"),
           static_cast<int32>(DataPacket.Type));

    if (bShouldStop.load())
    {
        UE_LOG(LogAsyncFileWriter, Warning, TEXT("WriteDataAsync: Writer is stopped, ignoring packet"));
        return;
    }

    if (!DataPacket.bValid || !DataPacket.Data.IsValid())
    {
        UE_LOG(LogAsyncFileWriter, Warning, TEXT("Invalid data packet received for writing"));
        return;
    }

    WriteQueue.Enqueue(DataPacket);
}


void FAsyncFileWriter::Flush()
{
    while (!WriteQueue.IsEmpty())
    {
        FPlatformProcess::Sleep(0.001f);
    }
}

bool FAsyncFileWriter::Init()
{
    UE_LOG(LogAsyncFileWriter, Log, TEXT("AsyncFileWriter thread initialized"));
    return true;
}

uint32 FAsyncFileWriter::Run()
{
    while (!bShouldStop.load())
    {
        ProcessWriteQueue();
        FPlatformProcess::Sleep(0.001f);
    }

    ProcessWriteQueue();
    UE_LOG(LogAsyncFileWriter, Log, TEXT("AsyncFileWriter thread finished"));
    return 0;
}

void FAsyncFileWriter::Stop()
{
    bShouldStop.store(true);
    UE_LOG(LogAsyncFileWriter, Log, TEXT("AsyncFileWriter stop requested"));
}

void FAsyncFileWriter::Exit()
{
    UE_LOG(LogAsyncFileWriter, Log, TEXT("AsyncFileWriter thread exit"));
}

void FAsyncFileWriter::ProcessWriteQueue()
{
    FSensorDataPacket DataPacket;
    while (WriteQueue.Dequeue(DataPacket))
    {
        WriteDataToFile(DataPacket);
    }
}

void FAsyncFileWriter::WriteDataToFile(const FSensorDataPacket& DataPacket)
{
    if (!DataPacket.bValid || !DataPacket.Data.IsValid() || !DataPacket.OwnerActor)
    {
        UE_LOG(LogAsyncFileWriter, Warning,
            TEXT("WriteDataToFile: Invalid packet - Valid: %s, Data: %s, Actor: %s"),
               DataPacket.bValid ? TEXT("true") : TEXT("false"),
               DataPacket.Data.IsValid() ? TEXT("valid") : TEXT("invalid"),
               DataPacket.OwnerActor ? TEXT("valid") : TEXT("null"));
        return;
    }

    FString FilePath = GetFilePathForSensor(DataPacket);

    switch (DataPacket.Type)
    {
        case ESensorType::RGBCamera:
        {
            if (const FRGBCameraData* RGBData = static_cast<const FRGBCameraData*>(DataPacket.Data.Get()))
            {
                // Use proper PNG compression like ImageProcesser
                TArray64<uint8> CompressedBitmap;
                FImageUtils::PNGCompressImageArray(RGBData->Width, RGBData->Height, RGBData->Data, CompressedBitmap);

                if (!FFileHelper::SaveArrayToFile(CompressedBitmap, *FilePath))
                {
                    UE_LOG(LogAsyncFileWriter, Error, TEXT("Failed to save RGB image to file: %s"), *FilePath);
                }
            }
            break;
        }

        case ESensorType::DepthCamera:
        {
            if (const FDepthCameraData* DepthData = static_cast<const FDepthCameraData*>(DataPacket.Data.Get()))
            {
                // Convert float depth data to visual grayscale PNG
                TArray<FColor> VisualPixels;
                VisualPixels.Reserve(DepthData->Data.Num());

                // Find depth range for normalization
                float MinDepth = 0.0f, MaxDepth = 100.0f; // Default reasonable range
                if (DepthData->Data.Num() > 0)
                {
                    TArray<float> ValidDepths = DepthData->Data;
                    ValidDepths.RemoveAll([](float Depth) { return !FMath::IsFinite(Depth) || Depth < 0.0f; });

                    if (ValidDepths.Num() > 0)
                    {
                        ValidDepths.Sort();
                        MinDepth = ValidDepths[FMath::FloorToInt(ValidDepths.Num() * 0.02f)];
                        MaxDepth = ValidDepths[FMath::FloorToInt(ValidDepths.Num() * 0.98f)];
                        if (MaxDepth <= MinDepth) MaxDepth = MinDepth + 100.0f;
                    }
                }

                // Convert to grayscale (closer = darker, farther = lighter)
                for (float Depth : DepthData->Data)
                {
                    float ClampedDepth = FMath::Clamp(Depth, MinDepth, MaxDepth);
                    float NormalizedDepth = (ClampedDepth - MinDepth) / (MaxDepth - MinDepth);
                    float InvertedDepth = 1.0f - NormalizedDepth;
                    uint8 GrayValue = FMath::RoundToInt(InvertedDepth * 255.0f);
                    VisualPixels.Add(FColor(GrayValue, GrayValue, GrayValue, 255));
                }

                // Save as PNG
                TArray64<uint8> CompressedBitmap;
                FImageUtils::PNGCompressImageArray(DepthData->Width, DepthData->Height, VisualPixels, CompressedBitmap);

                if (!FFileHelper::SaveArrayToFile(CompressedBitmap, *FilePath))
                {
                    UE_LOG(LogAsyncFileWriter, Error, TEXT("Failed to save depth image to file: %s"), *FilePath);
                }
            }
            break;
        }

        case ESensorType::NormalCamera:
        {
            if (const FNormalCameraData* NormalData = static_cast<const FNormalCameraData*>(DataPacket.Data.Get()))
            {
                // Use modern UE image API with FImage for EXR
                FImage Image;
                Image.Init(NormalData->Width, NormalData->Height, ERawImageFormat::RGBA32F);

                // Convert FLinearColor to FImage data
                TArrayView64<FLinearColor> ImageData = Image.AsRGBA32F();

                if (ImageData.Num() == NormalData->Data.Num())
                {
                    // Copy normal data to image
                    for (int32 i = 0; i < NormalData->Data.Num(); ++i)
                    {
                        ImageData[i] = NormalData->Data[i];
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
                        if (!FFileHelper::SaveArrayToFile(CompressedData, *FilePath))
                        {
                            UE_LOG(LogAsyncFileWriter, Error, TEXT("Failed to save normal EXR image to file: %s"), *FilePath);
                        }
                    }
                    else
                    {
                        UE_LOG(LogAsyncFileWriter, Error, TEXT("Failed to compress normal image to EXR format"));
                    }
                }
            }
            break;
        }

        case ESensorType::SegmentationCamera:
        {
            if (const FSegmentationCameraData* SegData = static_cast<const FSegmentationCameraData*>(DataPacket.Data.Get()))
            {
                // Use proper PNG compression for segmentation data
                TArray64<uint8> CompressedBitmap;
                FImageUtils::PNGCompressImageArray(SegData->Width, SegData->Height, SegData->Data, CompressedBitmap);

                if (!FFileHelper::SaveArrayToFile(CompressedBitmap, *FilePath))
                {
                    UE_LOG(LogAsyncFileWriter, Error, TEXT("Failed to save segmentation image to file: %s"), *FilePath);
                }
            }
            break;
        }

        case ESensorType::Lidar:
        {
            if (const FLiDARData* LidarData = static_cast<const FLiDARData*>(DataPacket.Data.Get()))
            {
                // Save as PLY point cloud format
                if (LidarData->Data.Num() > 0)
                {
                    FString PLYContent;

                    // PLY header
                    PLYContent += TEXT("ply\n");
                    PLYContent += TEXT("format ascii 1.0\n");
                    PLYContent += FString::Printf(TEXT("comment LiDAR Point Cloud generated by VCCSim - %s\n"),
                        *FDateTime::Now().ToString());
                    PLYContent += TEXT("comment UE Coordinate System: X=Forward, Y=Right, Z=Up (left-handed)\n");
                    PLYContent += TEXT("comment Units: centimeters\n");
                    PLYContent += FString::Printf(TEXT("element vertex %d\n"), LidarData->Data.Num());
                    PLYContent += TEXT("property float x\n");
                    PLYContent += TEXT("property float y\n");
                    PLYContent += TEXT("property float z\n");
                    PLYContent += TEXT("property uchar red\n");
                    PLYContent += TEXT("property uchar green\n");
                    PLYContent += TEXT("property uchar blue\n");
                    PLYContent += TEXT("end_header\n");

                    // Write vertex data with default white color
                    for (const FVector3f& Point : LidarData->Data)
                    {
                        PLYContent += FString::Printf(TEXT("%.6f %.6f %.6f 255 255 255\n"),
                            Point.X, Point.Y, Point.Z);
                    }

                    // Save to file
                    if (!FFileHelper::SaveStringToFile(PLYContent, *FilePath))
                    {
                        UE_LOG(LogAsyncFileWriter, Error, TEXT("Failed to save LiDAR PLY file: %s"), *FilePath);
                    }
                    else
                    {
                        UE_LOG(LogAsyncFileWriter, Log, TEXT("Successfully saved %d LiDAR points to PLY: %s"),
                            LidarData->Data.Num(), *FilePath);
                    }
                }
                else
                {
                    UE_LOG(LogAsyncFileWriter, Warning, TEXT("No LiDAR points to save"));
                }
            }
            break;
        }
    }
}

FString FAsyncFileWriter::GetFilePathForSensor(const FSensorDataPacket& DataPacket)
{
    FString ActorName = DataPacket.OwnerActor->GetName();
    FString SensorTypeStr;
    FString FileExtension;

    switch (DataPacket.Type)
    {
        case ESensorType::RGBCamera:
            SensorTypeStr = TEXT("RGB");
            FileExtension = TEXT(".png");
            break;
        case ESensorType::DepthCamera:
            SensorTypeStr = TEXT("Depth");
            FileExtension = TEXT(".png");
            break;
        case ESensorType::NormalCamera:
            SensorTypeStr = TEXT("Normal");
            FileExtension = TEXT(".exr");
            break;
        case ESensorType::SegmentationCamera:
            SensorTypeStr = TEXT("Segmentation");
            FileExtension = TEXT(".png");
            break;
        case ESensorType::Lidar:
            SensorTypeStr = TEXT("Lidar");
            FileExtension = TEXT(".ply");
            break;
        default:
            SensorTypeStr = TEXT("Unknown");
            FileExtension = TEXT(".data");
            break;
    }

    // Format filename with 9 decimal places for timestamp (matching original)
    FString FileName = FString::Printf(TEXT("%.9f%s"), DataPacket.Timestamp, *FileExtension);

    return FPaths::Combine(BasePath, ActorName, SensorTypeStr, FileName);
}