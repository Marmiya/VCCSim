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
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"

DEFINE_LOG_CATEGORY(LogAsyncFileWriter);

FAsyncFileWriter::FAsyncFileWriter(const FString& InBasePath, bool InBetterVisuals)
    : BasePath(InBasePath), bBetterVisuals(InBetterVisuals)
{
    if (FPaths::IsRelative(BasePath))
    {
        BasePath = FPaths::ConvertRelativePathToFull(BasePath);
    }

    IOWorker = MakeUnique<FIOWorker>(this);
    IOThread.Reset(FRunnableThread::Create(IOWorker.Get(),
        TEXT("AsyncFileWriter_IO"), 0, TPri_Normal));
    UE_LOG(LogAsyncFileWriter, Log, TEXT("AsyncFileWriter initialized "
                                         "with parallel compression"));
}

FAsyncFileWriter::~FAsyncFileWriter()
{
    bShouldStop.store(true);
    if (IOWorker.IsValid())
    {
        IOWorker->Stop();
    }
    if (IOThread.IsValid())
    {
        IOThread->WaitForCompletion();
        IOThread.Reset();
    }
    IOWorker.Reset();
    UE_LOG(LogAsyncFileWriter, Log, TEXT("AsyncFileWriter destroyed"));
}

void FAsyncFileWriter::WriteData(FSensorDataPacket&& DataPacket)
{
    if (bShouldStop.load())
    {
        UE_LOG(LogAsyncFileWriter, Warning, TEXT("WriteDataBatch: Writer "
                                                 "is stopped, ignoring packets"));
        return;
    }
    
    if (!DataPacket.Data.IsValid())
    {
        UE_LOG(LogAsyncFileWriter, Warning, TEXT("Invalid data packet received"));
        return;
    }

    PendingCompressionTasks.fetch_add(1);

    AsyncTask(ENamedThreads::AnyThread, [this, DataPacket]()
    {
        MakeUnique<FAsyncTask<FImageCompressionTask>>(DataPacket, BasePath,
            CompressedDataQueue, bBetterVisuals)->StartSynchronousTask();
        PendingCompressionTasks.fetch_sub(1);
    });
}

void FAsyncFileWriter::Flush()
{
    while (PendingCompressionTasks.load() > 0 || !CompressedDataQueue.IsEmpty())
    {
        FPlatformProcess::YieldThread();
    }
    UE_LOG(LogAsyncFileWriter, Log, TEXT("Flush completed - all tasks finished"));
}


uint32 FAsyncFileWriter::FIOWorker::Run()
{
    while (!bShouldStop.load())
    {
        TSharedPtr<FCompressedImageData> CompressedData;
        while (Owner->CompressedDataQueue.Dequeue(CompressedData))
        {
            if (CompressedData.IsValid())
            {
                Owner->WriteCompressedDataToFile(*CompressedData);
            }
        }
        FPlatformProcess::YieldThread();
    }

    TSharedPtr<FCompressedImageData> CompressedData;
    while (Owner->CompressedDataQueue.Dequeue(CompressedData))
    {
        if (CompressedData.IsValid())
        {
            Owner->WriteCompressedDataToFile(*CompressedData);
        }
    }

    UE_LOG(LogAsyncFileWriter, Log, TEXT("I/O worker finished"));
    return 0;
}

void FAsyncFileWriter::WriteCompressedDataToFile(const FCompressedImageData& CompressedData)
{
    // Ensure directory exists
    FString Directory = FPaths::GetPath(CompressedData.FilePath);
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.DirectoryExists(*Directory))
    {
        PlatformFile.CreateDirectoryTree(*Directory);
    }

    if (!FFileHelper::SaveArrayToFile(CompressedData.CompressedData, *CompressedData.FilePath))
    {
        UE_LOG(LogAsyncFileWriter, Error, TEXT("Failed to save file: %s"),
            *CompressedData.FilePath);
    }
}

// ================== FImageCompressionTask Implementation ==================

void FImageCompressionTask::DoWork()
{
    if (!DataPacket.Data.IsValid() || !DataPacket.OwnerActor)
    {
        UE_LOG(LogAsyncFileWriter, Warning,
            TEXT("Invalid data packet in compression task"));
        return;
    }

    switch (DataPacket.Type)
    {
        case ESensorType::RGBCamera:
            CompressRGBData();
            break;
        case ESensorType::DepthCamera:
            CompressDepthData();
            break;
        case ESensorType::NormalCamera:
            CompressNormalData();
            break;
        case ESensorType::SegmentationCamera:
            CompressSegmentationData();
            break;
        case ESensorType::BaseColorCamera:
            CompressBaseColorData();
            break;
        case ESensorType::MaterialPropertiesCamera:
            CompressMaterialPropertiesData();
            break;
        case ESensorType::Lidar:
            CompressLidarData();
            break;
        default:
            UE_LOG(LogAsyncFileWriter, Warning,
                TEXT("Unknown sensor type for compression"));
            break;
    }
}

TSharedPtr<FCompressedImageData> FImageCompressionTask::CreateCompressedResult(
    const FString& SubPath, const FString& Extension, double Timestamp)
{
    auto Result = MakeShared<FCompressedImageData>();
    Result->SensorType = DataPacket.Type;

    FString FileName = FString::Printf(TEXT("%.6f.%s"), Timestamp, *Extension);
    Result->FilePath = FPaths::Combine(BasePath, DataPacket.OwnerActor->GetName(), SubPath, FileName);

    return Result;
}

bool FImageCompressionTask::CompressImageToPNG(
    const TArray<FColor>& ImageData, int32 Width, int32 Height, TArray64<uint8>& OutCompressedData)
{
    IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>(FName("ImageWrapper"));

    FImage Image;
    Image.Init(Width, Height, ERawImageFormat::BGRA8);
    TArrayView64<FColor> ImageView = Image.AsBGRA8();

    if (ImageView.Num() == ImageData.Num())
    {
        for (int32 i = 0; i < ImageData.Num(); ++i)
        {
            ImageView[i] = ImageData[i];
        }

        return ImageWrapperModule.CompressImage(OutCompressedData, EImageFormat::PNG, Image, 100);
    }

    return false;
}

bool FImageCompressionTask::CompressImageToEXR(
    const TArray<FFloat16Color>& ImageData, int32 Width, int32 Height, TArray64<uint8>& OutCompressedData)
{
    IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>(FName("ImageWrapper"));

    FImage Image;
    Image.Init(Width, Height, ERawImageFormat::RGBA16F);
    TArrayView64<FFloat16Color> ImageView = Image.AsRGBA16F();

    if (ImageView.Num() == ImageData.Num())
    {
        for (int32 i = 0; i < ImageData.Num(); ++i)
        {
            ImageView[i] = ImageData[i];
        }

        return ImageWrapperModule.CompressImage(OutCompressedData, EImageFormat::EXR, Image, 100);
    }

    return false;
}

void FImageCompressionTask::CompressRGBData()
{
    const FRGBCameraData* RGBData = static_cast<const FRGBCameraData*>(DataPacket.Data.Get());

    if (RGBData && RGBData->RGBData.Num() > 0)
    {
        auto RGBResult =
            CreateCompressedResult(TEXT("RGB"), TEXT("png"), DataPacket.Data->Timestamp);
        CompressImageToPNG(RGBData->RGBData, RGBData->Width, RGBData->Height, RGBResult->CompressedData);
        CompressedDataQueue.Enqueue(RGBResult);
    }
    else
    {
        UE_LOG(LogAsyncFileWriter, Error, TEXT("No RGB data to compress for actor: %s"),
            *DataPacket.OwnerActor->GetName());
    }
}

void FImageCompressionTask::CompressBaseColorData()
{
    const FBaseColorCameraData* BCData = static_cast<const FBaseColorCameraData*>(DataPacket.Data.Get());

    if (BCData && BCData->Data.Num() > 0)
    {
        auto Result = CreateCompressedResult(TEXT("BaseColor"), TEXT("png"), DataPacket.Data->Timestamp);
        CompressImageToPNG(BCData->Data, BCData->Width, BCData->Height, Result->CompressedData);
        CompressedDataQueue.Enqueue(Result);
    }
    else
    {
        UE_LOG(LogAsyncFileWriter, Error, TEXT("No BaseColor data to compress for actor: %s"),
            *DataPacket.OwnerActor->GetName());
    }
}

void FImageCompressionTask::CompressMaterialPropertiesData()
{
    const FMaterialPropertiesCameraData* MPData =
        static_cast<const FMaterialPropertiesCameraData*>(DataPacket.Data.Get());

    if (MPData && MPData->Data.Num() > 0)
    {
        auto Result = CreateCompressedResult(TEXT("MaterialProperties"), TEXT("png"), DataPacket.Data->Timestamp);
        CompressImageToPNG(MPData->Data, MPData->Width, MPData->Height, Result->CompressedData);
        CompressedDataQueue.Enqueue(Result);
    }
    else
    {
        UE_LOG(LogAsyncFileWriter, Error, TEXT("No MaterialProperties data to compress for actor: %s"),
            *DataPacket.OwnerActor->GetName());
    }
}

// Google's Turbo colormap, polynomial approximation by Anton Mikhailov.
// Input in [0,1], output is an sRGB display color (low=blue, mid=green/yellow, high=red).
static FColor TurboColormap(float X)
{
    X = FMath::Clamp(X, 0.0f, 1.0f);
    const float X2 = X * X;
    const float X3 = X2 * X;
    const float X4 = X2 * X2;
    const float X5 = X2 * X3;

    const float R = 0.13572138f + 4.61539260f * X - 42.66032258f * X2
        + 132.13108234f * X3 - 152.94239396f * X4 + 59.28637943f * X5;
    const float G = 0.09140261f + 2.19418839f * X + 4.84296658f * X2
        - 14.18503333f * X3 + 4.27729857f * X4 + 2.82956604f * X5;
    const float B = 0.10667330f + 12.64194608f * X - 60.58204836f * X2
        + 110.36276771f * X3 - 89.90310912f * X4 + 27.34824973f * X5;

    return FColor(
        (uint8)FMath::Clamp(FMath::RoundToInt(R * 255.0f), 0, 255),
        (uint8)FMath::Clamp(FMath::RoundToInt(G * 255.0f), 0, 255),
        (uint8)FMath::Clamp(FMath::RoundToInt(B * 255.0f), 0, 255),
        255);
}

void FImageCompressionTask::CompressDepthData()
{
    const FDepthCameraData* DepthData = static_cast<const FDepthCameraData*>(DataPacket.Data.Get());

    if (!DepthData || DepthData->DepthData.Num() == 0)
    {
        return;
    }

    auto DepthResult =
        CreateCompressedResult(TEXT("Depth"), TEXT("png"), DataPacket.Data->Timestamp);

    if (bBetterVisuals)
    {
        // Turbo colormap over a dynamic range: near = blue, far = red.
        // Range comes from 2nd/98th percentiles to ignore outliers.
        TArray<float> SortedDepths = DepthData->DepthData;
        SortedDepths.RemoveAll([](float Depth)
            { return Depth < 0.0f || !FMath::IsFinite(Depth); });
        SortedDepths.Sort();

        float MinRange = 0.0f;
        float MaxRange = 1.0f;
        if (SortedDepths.Num() > 0)
        {
            int32 LowIndex = FMath::Clamp(
                FMath::FloorToInt(SortedDepths.Num() * 0.02f), 0, SortedDepths.Num() - 1);
            int32 HighIndex = FMath::Clamp(
                FMath::FloorToInt(SortedDepths.Num() * 0.98f), LowIndex, SortedDepths.Num() - 1);
            MinRange = SortedDepths[LowIndex];
            MaxRange = SortedDepths[HighIndex];
            if (MaxRange <= MinRange)
            {
                MaxRange = MinRange + 1.0f;
            }
        }

        TArray<FColor> VisualPixels;
        VisualPixels.Reserve(DepthData->DepthData.Num());
        for (float Depth : DepthData->DepthData)
        {
            float Clamped = FMath::Clamp(Depth, MinRange, MaxRange);
            float Normalized = (Clamped - MinRange) / (MaxRange - MinRange);
            VisualPixels.Add(TurboColormap(Normalized));
        }

        CompressImageToPNG(VisualPixels, DepthData->Width, DepthData->Height,
            DepthResult->CompressedData);
        CompressedDataQueue.Enqueue(DepthResult);
        return;
    }

    // Raw 16-bit depth (centimeters) as G16 PNG.
    TArray<uint16> DepthData16;
    DepthData16.Reserve(DepthData->DepthData.Num());

    for (const float& DepthPixel : DepthData->DepthData)
    {
        uint16 Depth16 = FMath::Clamp(FMath::RoundToInt(DepthPixel), 0, 65535);
        DepthData16.Add(Depth16);
    }

    IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>(FName("ImageWrapper"));

    FImage DepthImage;
    DepthImage.Init(DepthData->Width, DepthData->Height, ERawImageFormat::G16);
    TArrayView64<uint16> DepthImageView = DepthImage.AsG16();

    if (DepthImageView.Num() == DepthData16.Num())
    {
        for (int32 i = 0; i < DepthData16.Num(); ++i)
        {
            DepthImageView[i] = DepthData16[i];
        }
        ImageWrapperModule.CompressImage(DepthResult->CompressedData,
            EImageFormat::PNG, DepthImage, 100);
        CompressedDataQueue.Enqueue(DepthResult);
    }
}

void FImageCompressionTask::CompressNormalData()
{
    const FNormalCameraData* NormalData = static_cast<const FNormalCameraData*>(DataPacket.Data.Get());
    if (NormalData && NormalData->Data.Num() > 0)
    {
        if (bBetterVisuals)
        {
            // Map normal vectors from [-1,1] to [0,255] RGB and save as PNG.
            auto Result =
                CreateCompressedResult(TEXT("Normal"), TEXT("png"), DataPacket.Data->Timestamp);

            constexpr float SaturationBoost = 1.5f;
            TArray<FColor> VisualPixels;
            VisualPixels.Reserve(NormalData->Data.Num());
            for (const FFloat16Color& Pixel : NormalData->Data)
            {
                FVector NormalVec(Pixel.R.GetFloat(), Pixel.G.GetFloat(), Pixel.B.GetFloat());
                NormalVec.Normalize();

                // Map [-1,1] -> [0,1], then boost saturation in HSV for vividness.
                FLinearColor Color(
                    (NormalVec.X + 1.0f) * 0.5f,
                    (NormalVec.Y + 1.0f) * 0.5f,
                    (NormalVec.Z + 1.0f) * 0.5f);
                FLinearColor HSV = Color.LinearRGBToHSV();
                HSV.G = FMath::Min(HSV.G * SaturationBoost, 1.0f); // G = Saturation
                FLinearColor Boosted = HSV.HSVToLinearRGB();

                uint8 R = (uint8)FMath::Clamp(FMath::RoundToInt(Boosted.R * 255.0f), 0, 255);
                uint8 G = (uint8)FMath::Clamp(FMath::RoundToInt(Boosted.G * 255.0f), 0, 255);
                uint8 B = (uint8)FMath::Clamp(FMath::RoundToInt(Boosted.B * 255.0f), 0, 255);
                VisualPixels.Add(FColor(R, G, B, 255));
            }

            CompressImageToPNG(VisualPixels, NormalData->Width, NormalData->Height,
                Result->CompressedData);
            CompressedDataQueue.Enqueue(Result);
            return;
        }

        auto Result =
            CreateCompressedResult(TEXT("Normal"), TEXT("exr"), DataPacket.Data->Timestamp);
        CompressImageToEXR(NormalData->Data, NormalData->Width,
            NormalData->Height, Result->CompressedData);
        CompressedDataQueue.Enqueue(Result);
    }
    else
    {
        UE_LOG(LogAsyncFileWriter, Error, TEXT("No Normal data to compress for actor: %s"),
            *DataPacket.OwnerActor->GetName());
    }
}

void FImageCompressionTask::CompressSegmentationData()
{
    const FSegmentationCameraData* SegData =
        static_cast<const FSegmentationCameraData*>(DataPacket.Data.Get());
    if (SegData && SegData->Data.Num() > 0)
    {
        auto Result =
            CreateCompressedResult(TEXT("Segmentation"), TEXT("png"), DataPacket.Data->Timestamp);
        CompressImageToPNG(SegData->Data, SegData->Width, SegData->Height, Result->CompressedData);
        CompressedDataQueue.Enqueue(Result);
    }
    else
    {
        UE_LOG(LogAsyncFileWriter, Error, TEXT("No Segmentation data to compress for actor: %s"),
            *DataPacket.OwnerActor->GetName());
    }
}

void FImageCompressionTask::CompressLidarData()
{
    const FLiDARData* LidarData = static_cast<const FLiDARData*>(DataPacket.Data.Get());
    if (!LidarData || LidarData->Data.Num() == 0)
    {
        return;
    }

    auto Result = CreateCompressedResult(TEXT("Lidar"), TEXT("ply"), DataPacket.Data->Timestamp);

    FString PLYContent;
    PLYContent += TEXT("ply\n");
    PLYContent += TEXT("format ascii 1.0\n");
    PLYContent += FString::Printf(TEXT("comment LiDAR Point Cloud generated by VCCSim - %s\n"), *FDateTime::Now().ToString());
    PLYContent += FString::Printf(TEXT("element vertex %d\n"), LidarData->Data.Num());
    PLYContent += TEXT("property float x\n");
    PLYContent += TEXT("property float y\n");
    PLYContent += TEXT("property float z\n");
    PLYContent += TEXT("property uchar red\n");
    PLYContent += TEXT("property uchar green\n");
    PLYContent += TEXT("property uchar blue\n");
    PLYContent += TEXT("end_header\n");

    for (const FVector3f& Point : LidarData->Data)
    {
        PLYContent += FString::Printf(TEXT("%.6f %.6f %.6f 255 255 255\n"),
            Point.X, Point.Y, Point.Z);
    }

    FTCHARToUTF8 UTF8String(*PLYContent);
    TArray64<uint8> UTF8Data;
    UTF8Data.Append(reinterpret_cast<const uint8*>(UTF8String.Get()), UTF8String.Length());

    Result->CompressedData = MoveTemp(UTF8Data);
    CompressedDataQueue.Enqueue(Result);
}