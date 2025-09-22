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
    if (bShouldStop.load())
    {
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
    UE_LOG(LogAsyncFileWriter, Log, TEXT("AsyncFileWriter flush completed"));
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
        return;
    }

    CreateDirectoryStructure(DataPacket);
    FString FilePath = GetFilePathForSensor(DataPacket);

    switch (DataPacket.Type)
    {
        case ESensorType::RGBCamera:
        {
            if (const FRGBCameraData* RGBData = static_cast<const FRGBCameraData*>(DataPacket.Data.Get()))
            {
                TArray<uint8> ImageData;
                ImageData.SetNumUninitialized(RGBData->Width * RGBData->Height * 3);

                for (int32 i = 0; i < RGBData->Data.Num(); ++i)
                {
                    const FColor& Pixel = RGBData->Data[i];
                    ImageData[i * 3] = Pixel.R;
                    ImageData[i * 3 + 1] = Pixel.G;
                    ImageData[i * 3 + 2] = Pixel.B;
                }

                FFileHelper::SaveArrayToFile(ImageData, *FilePath);
            }
            break;
        }

        case ESensorType::DepthCamera:
        {
            if (const FDepthCameraData* DepthData = static_cast<const FDepthCameraData*>(DataPacket.Data.Get()))
            {
                TArray<uint8> RawData;
                RawData.SetNumUninitialized(DepthData->Data.Num() * sizeof(float));
                FMemory::Memcpy(RawData.GetData(), DepthData->Data.GetData(), RawData.Num());
                FFileHelper::SaveArrayToFile(RawData, *FilePath);
            }
            break;
        }

        case ESensorType::NormalCamera:
        {
            if (const FNormalCameraData* NormalData = static_cast<const FNormalCameraData*>(DataPacket.Data.Get()))
            {
                TArray<uint8> RawData;
                RawData.SetNumUninitialized(NormalData->Data.Num() * sizeof(FLinearColor));
                FMemory::Memcpy(RawData.GetData(), NormalData->Data.GetData(), RawData.Num());
                FFileHelper::SaveArrayToFile(RawData, *FilePath);
            }
            break;
        }

        case ESensorType::SegmentationCamera:
        {
            if (const FSegmentationCameraData* SegData = static_cast<const FSegmentationCameraData*>(DataPacket.Data.Get()))
            {
                TArray<uint8> ImageData;
                ImageData.SetNumUninitialized(SegData->Width * SegData->Height * 3);

                for (int32 i = 0; i < SegData->Data.Num(); ++i)
                {
                    const FColor& Pixel = SegData->Data[i];
                    ImageData[i * 3] = Pixel.R;
                    ImageData[i * 3 + 1] = Pixel.G;
                    ImageData[i * 3 + 2] = Pixel.B;
                }

                FFileHelper::SaveArrayToFile(ImageData, *FilePath);
            }
            break;
        }

        case ESensorType::Lidar:
        {
            if (const FLiDARData* LidarData = static_cast<const FLiDARData*>(DataPacket.Data.Get()))
            {
                TArray<uint8> RawData;
                RawData.SetNumUninitialized(LidarData->Data.Num() * sizeof(FVector3f));
                FMemory::Memcpy(RawData.GetData(), LidarData->Data.GetData(), RawData.Num());
                FFileHelper::SaveArrayToFile(RawData, *FilePath);
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

void FAsyncFileWriter::CreateDirectoryStructure(const FSensorDataPacket& DataPacket)
{
    // Directory creation is now handled by Recorder - AsyncFileWriter just writes to existing directories
    // This method is kept for interface compatibility but does nothing
}