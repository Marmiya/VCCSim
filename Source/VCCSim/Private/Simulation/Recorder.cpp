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

#include "Simulation/Recorder.h"
#include "HAL/FileManager.h"
#include "Misc/Paths.h"
#include "Async/AsyncWork.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Modules/ModuleManager.h"
#include "Containers/StringConv.h"
#include "HAL/PlatformFileManager.h"


IImageWrapper* FImageWrapperCache::GetPNGWrapper()
{
    FScopeLock Lock(&CacheLock);
    if (!PNGWrapper)
    {
        if (!ImageWrapperModule)
        {
            ImageWrapperModule = &FModuleManager::LoadModuleChecked<
                IImageWrapperModule>(FName("ImageWrapper"));
        }
        PNGWrapper = ImageWrapperModule->CreateImageWrapper(EImageFormat::PNG);
    }
    return PNGWrapper.Get();
}

// BufferPool implementation
TArray<uint8>* FBufferPool::AcquireBuffer(int32 Size)
{
    FScopeLock Lock(&PoolLock);
    if (Buffers.Num() > 0)
    {
        auto* Buffer = Buffers.Pop();
        Buffer->SetNum(Size, false);
        return Buffer;
    }
    return new TArray<uint8>();
}

void FBufferPool::ReleaseBuffer(TArray<uint8>* Buffer)
{
    if (!Buffer) return;
    FScopeLock Lock(&PoolLock);
    if (Buffers.Num() < FRecorderConfig::MaxPoolSize)
    {
        Buffer->Empty();
        Buffers.Push(Buffer);
    }
    else
    {
        delete Buffer;
    }
}

// AdaptiveSleeper implementation
void FAdaptiveSleeper::Sleep()
{
    if (EmptyCount++ > 10)
    {
        CurrentInterval = FMath::Min(
            CurrentInterval * FRecorderConfig::SleepMultiplier,
            FRecorderConfig::MaxSleepInterval
        );
    }
    else
    {
        CurrentInterval = FRecorderConfig::MinSleepInterval;
    }
    FPlatformProcess::Sleep(CurrentInterval);
}

void FAdaptiveSleeper::Reset()
{
    CurrentInterval = FRecorderConfig::MinSleepInterval;
    EmptyCount = 0;
}

// RecorderWorker implementation
FRecorderWorker::FRecorderWorker(const FString& InBasePath, int32 InBufferSize)
    : BasePath(InBasePath)
    , BufferSize(InBufferSize)
    , Thread(nullptr)
    , bStopRequested(false)
{
    Thread = FRunnableThread::Create(this,
        TEXT("RecorderWorker"), 0, TPri_Normal);
}

FRecorderWorker::~FRecorderWorker()
{
    Stop();
    if (Thread)
    {
        Thread->WaitForCompletion();
        delete Thread;
        Thread = nullptr;
    }
}

bool FRecorderWorker::Init()
{
    return true;
}

void FRecorderWorker::Stop()
{
    bStopRequested = true;
}

void FRecorderWorker::Exit()
{
    FScopeLock Lock(&QueueLock);
    DataQueue.Empty();
}

uint32 FRecorderWorker::Run()
{    
    TArray<FPawnBuffers> BatchBuffers;
    BatchBuffers.Reserve(FRecorderConfig::BatchSize);

    while (!bStopRequested)
    {
        const int32 ProcessedCount = ProcessBatch(BatchBuffers);
    
        if (ProcessedCount == 0)
        {
            Sleeper.Sleep();
        }
        else
        {
            Sleeper.Reset();
        }
    }

    return 0;
}

int32 FRecorderWorker::ProcessBatch(TArray<FPawnBuffers>& BatchBuffers)
{
    BatchBuffers.Reset();
    int32 ProcessedCount = 0;

    {
        FScopeLock Lock(&QueueLock);
        while (BatchBuffers.Num() < FRecorderConfig::BatchSize && !DataQueue.IsEmpty())
        {
            FPawnBuffers Buffer(BufferSize);
            if (DataQueue.Dequeue(Buffer))
            {
                BatchBuffers.Add(MoveTemp(Buffer));
                ProcessedCount++;
            }
        }
    }

    for (auto& Buffer : BatchBuffers)
    {
        ProcessBuffer(Buffer);
    }

    return ProcessedCount;
}

void FRecorderWorker::ProcessBuffer(FPawnBuffers& Buffer)
{
    // Process Pose data
    {
        FPoseData PoseData;
        while (Buffer.Pose.Dequeue(PoseData))
        {
            const FString PoseContent = FString::Printf(
                TEXT("%.9f %.6f %.6f %.6f %.6f %.6f %.6f\n"),
                PoseData.Timestamp,
                PoseData.Location.X, PoseData.Location.Y, PoseData.Location.Z,
                PoseData.Rotation.Roll, PoseData.Rotation.Pitch, PoseData.Rotation.Yaw
            );
        
            FFileHelper::SaveStringToFile(
                PoseContent,
                *FPaths::Combine(Buffer.PawnDirectory, TEXT("pose.txt")),
                FFileHelper::EEncodingOptions::AutoDetect,
                &IFileManager::Get(),
                FILEWRITE_Append
            );
        }
    }

    // Process LiDAR data
    {
        FLidarData LidarData;
        while (Buffer.Lidar.Dequeue(LidarData))
        {
            SaveLidarData(LidarData, Buffer.PawnDirectory);
        }
    }

    // Process Depth data
    {
        FDepthCameraData DepthData;
        while (Buffer.DepthC.Dequeue(DepthData))
        {
            SaveDepthData(DepthData, Buffer.PawnDirectory);
        }
    }

    // Process RGB data
    {
        FRGBCameraData RGBData;
        while (Buffer.RGBC.Dequeue(RGBData))
        {
            SaveRGBData(RGBData, Buffer.PawnDirectory);
        }
    }
}

bool FRecorderWorker::SaveLidarData(const FLidarData& LidarData, const FString& Directory)
{
    const FString Filename = FString::Printf(
        TEXT("%s.ply"),
        *FString::Printf(TEXT("%.9f"), LidarData.Timestamp)
    );

    const FString FilePath = FPaths::Combine(
        FPaths::ConvertRelativePathToFull(Directory), TEXT("Lidar"), Filename);

    TStringBuilder<FRecorderConfig::StringReserveSize> PlyContent;

    // Write PLY header
    PlyContent.Append(
        TEXT("ply\nformat ascii 1.0\n")
        TEXT("comment Timestamp: ")
    ).Appendf(TEXT("%.9f\n"), LidarData.Timestamp);

    PlyContent.Appendf(
        TEXT("element vertex %d\n")
        TEXT("property float x\nproperty float y\nproperty float z\n")
        TEXT("property int hit\nend_header\n"),
        LidarData.Data.Num()
    );

    // Write points
    for (const auto& Point : LidarData.Data)
    {
        PlyContent.Appendf(
            TEXT("%f %f %f %d\n"),
            Point.X,
            Point.Y,
            Point.Z,
            1
        );
    }

    return FFileHelper::SaveStringToFile(PlyContent.ToString(), *FilePath);
}

bool FRecorderWorker::SaveDepthData(const FDepthCameraData& DepthData, const FString& Directory)
{
    if (DepthData.Data.Num() > 0)
    {
        const FString Filename = FString::Printf(
        TEXT("%s_%d.png"),
        *FString::Printf(TEXT("%.9f"), DepthData.Timestamp),
        DepthData.SensorIndex
        );

        const FString FilePath = FPaths::Combine(Directory, TEXT("Depth"), Filename);

        auto* ImageBuffer =
            BufferPool.AcquireBuffer(DepthData.Width * DepthData.Height);
        if (!ImageBuffer) return false;

        // Find depth range and convert to grayscale in single pass
        float MinDepth = FLT_MAX;
        float MaxDepth = -FLT_MAX;

        for (float Depth : DepthData.Data)
        {
            MinDepth = FMath::Min(MinDepth, Depth);
            MaxDepth = FMath::Max(MaxDepth, Depth);
        }

        const float Range = MaxDepth - MinDepth;
        const float Scale = Range > 0.0f ? 255.0f / Range : 0.0f;

        ImageBuffer->SetNum(DepthData.Data.Num());

        // Vectorizable loop
        for (int32 i = 0; i < DepthData.Data.Num(); ++i)
        {
            (*ImageBuffer)[i] = FMath::RoundToInt((DepthData.Data[i] - MinDepth) * Scale);
        }

        // Use cached wrapper
        bool bSuccess = false;
        auto* PNGWrapper = FImageWrapperCache::Get().GetPNGWrapper();

        if (PNGWrapper && PNGWrapper->SetRaw(
            ImageBuffer->GetData(),
            ImageBuffer->Num(),
            DepthData.Width,
            DepthData.Height,
            ERGBFormat::Gray,
            8))
        {
            const TArray64<uint8>& PNGData = PNGWrapper->GetCompressed();
            bSuccess = FFileHelper::SaveArrayToFile(PNGData, *FilePath);
        }
        BufferPool.ReleaseBuffer(ImageBuffer);
        return bSuccess;
    }
    return false;
}

bool FRecorderWorker::SaveRGBData(
    const FRGBCameraData& RGBData, const FString& Directory)
{
    if (RGBData.Data.Num() == 0 || RGBData.Width <= 0 || RGBData.Height <= 0 ||
        RGBData.Data.Num() != RGBData.Width * RGBData.Height)
    {
        return false;
    }

    const FString Filename = FString::Printf(
        TEXT("%s_%d.png"),
        *FString::Printf(TEXT("%.9f"), RGBData.Timestamp),
        RGBData.SensorIndex
    );

    const FString FilePath = FPaths::Combine(Directory, TEXT("RGB"), Filename);

    const int32 ExpectedSize = RGBData.Width * RGBData.Height * 4;
    auto* ImageBuffer = BufferPool.AcquireBuffer(ExpectedSize);
    if (!ImageBuffer) return false;

    ImageBuffer->SetNum(ExpectedSize, false);

    // Optimized memory copy using raw pointers
    uint8* Dest = ImageBuffer->GetData();
    const int32 NumPixels = RGBData.Data.Num();

    // Vectorizable loop
    for (int32 i = 0; i < NumPixels; ++i)
    {
        const auto& Color = RGBData.Data[i];
        const int32 Base = i * 4;
        Dest[Base] = Color.R;
        Dest[Base + 1] = Color.G;
        Dest[Base + 2] = Color.B;
        Dest[Base + 3] = Color.A;
    }

    // Use cached wrapper
    bool bSuccess = false;
    auto* PNGWrapper = FImageWrapperCache::Get().GetPNGWrapper();

    if (PNGWrapper && PNGWrapper->SetRaw(
        ImageBuffer->GetData(),
        ImageBuffer->Num(),
        RGBData.Width,
        RGBData.Height,
        ERGBFormat::RGBA,
        8))
    {
        const TArray64<uint8>& PNGData = PNGWrapper->GetCompressed();
        bSuccess = FFileHelper::SaveArrayToFile(PNGData, *FilePath);
    }

    BufferPool.ReleaseBuffer(ImageBuffer);
    return bSuccess;
}

void FRecorderWorker::EnqueueBuffer(FPawnBuffers&& Buffer)
{
    if (bStopRequested) return;

    FScopeLock Lock(&QueueLock);
    DataQueue.Enqueue(MoveTemp(Buffer));
}

bool FRecorderWorker::IsQueueEmpty() const
{
    FScopeLock Lock(&QueueLock);
    return DataQueue.IsEmpty();
}

// AsyncSubmitTask implementation
FAsyncSubmitTask::FAsyncSubmitTask(ARecorder* InRecorder, FSubmissionData&& InData)
    : Recorder(InRecorder)
    , SubmissionData(MoveTemp(InData))
{
}

void FAsyncSubmitTask::DoWork()
{
    if (!ensureMsgf(Recorder && SubmissionData.Pawn,
        TEXT("Invalid recorder or pawn in AsyncSubmitTask")))
    {
        return;
    }

    FScopeLock Lock(&Recorder->BufferLock);

    auto& PawnBuffers = Recorder->ActiveBuffers.FindOrAdd(
        SubmissionData.Pawn,
        FPawnBuffers(Recorder->BufferSize)
    );

    auto EnqueueDataWithRetry = [this, &PawnBuffers](auto& Buffer, auto&& Data)
    {
        if (!Buffer.Enqueue(MoveTemp(Data)))
        {
            Recorder->SwapAndProcessBuffers();
            Buffer.Enqueue(MoveTemp(Data));
        }
    };

    switch (SubmissionData.Type)
    {
    case EDataType::Pose:
        EnqueueDataWithRetry(PawnBuffers.Pose,
            *static_cast<FPoseData*>(SubmissionData.Data.Get()));
        break;
    case EDataType::Lidar:
        EnqueueDataWithRetry(PawnBuffers.Lidar,
            *static_cast<FLidarData*>(SubmissionData.Data.Get()));
        break;
    case EDataType::DepthC:
        EnqueueDataWithRetry(PawnBuffers.DepthC,
            *static_cast<FDepthCameraData*>(SubmissionData.Data.Get()));
        break;
    case EDataType::RGBC:
        EnqueueDataWithRetry(PawnBuffers.RGBC,
            *static_cast<FRGBCameraData*>(SubmissionData.Data.Get()));
        break;
    }

    --Recorder->PendingTasks;
}

// ARecorder implementation
ARecorder::ARecorder()
    : PendingTasks(0)
{
    PrimaryActorTick.bCanEverTick = false;
}

void ARecorder::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    StopRecording();

    Super::EndPlay(EndPlayReason);
}

void ARecorder::StartRecording()
{
    if (bRecording) return;

    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.CreateDirectoryTree(*RecordingPath))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create recording directory: %s"),
            *RecordingPath);
        return;
    }

    // Create directories for each registered pawn
    for (const auto& PawnEntry : PawnDirectories)
    {
        if (!CreatePawnDirectories(PawnEntry.Key, PawnEntry.Value))
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to create pawn directories"));
            return;
        }
    }

    // Initialize worker
    RecorderWorker = MakeUnique<FRecorderWorker>(RecordingPath, BufferSize);
    bRecording = true;
}

void ARecorder::StopRecording()
{
    if (!bRecording) return;

    bRecording = false;

    // Wait for pending tasks with timeout
    const double StartTime = FPlatformTime::Seconds();
    constexpr double TimeoutSeconds = 5.0;

    while (PendingTasks > 0)
    {
        FPlatformProcess::Sleep(0.01f);
        if (FPlatformTime::Seconds() - StartTime > TimeoutSeconds)
        {
            UE_LOG(LogTemp, Warning, TEXT("Timeout waiting for pending tasks. Remaining: %d"),
                PendingTasks.Load());
            break;
        }
    }

    // Process remaining buffers
    SwapAndProcessBuffers();

    // Wait for worker to finish
    if (RecorderWorker)
    {
        const double WorkerStartTime = FPlatformTime::Seconds();
        while (!RecorderWorker->IsQueueEmpty())
        {
            FPlatformProcess::Sleep(0.01f);
            if (FPlatformTime::Seconds() - WorkerStartTime > TimeoutSeconds)
            {
                UE_LOG(LogTemp, Warning, TEXT("Timeout waiting for worker queue to empty"));
                break;
            }
        }
        RecorderWorker.Reset();
    }

    // Clear buffers
    {
        FScopeLock Lock(&BufferLock);
        ActiveBuffers.Empty();
        ProcessingBuffers.Empty();
    }
}

void ARecorder::ToggleRecording()
{
    RecordState = !RecordState;
    
    UE_LOG(LogTemp, Warning, TEXT("Recording state: "
                                  "%s"), RecordState ? TEXT("ON") : TEXT("OFF"));
    
    OnRecordStateChanged.Broadcast(RecordState);
}

void ARecorder::RegisterPawn(AActor* Pawn, bool bHasLidar,
    bool bHasDepth, bool bHasRGB, bool bHasNormal, bool bHasSegmentation)
{
    if (!Pawn) return;

    FPawnDirectoryInfo DirInfo;
    DirInfo.bHasLidar = bHasLidar;
    DirInfo.bHasDepth = bHasDepth;
    DirInfo.bHasRGB = bHasRGB;
    DirInfo.bHasNormal = bHasNormal;
    DirInfo.bHasSegmentation = bHasSegmentation;

    // Store the full directory path
    DirInfo.PawnDirectory = FPaths::Combine(RecordingPath, Pawn->GetName());

    if (bRecording)
    {
        if (!CreatePawnDirectories(Pawn, DirInfo))
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to create directories for pawn:"
                                        " %s"), *Pawn->GetName());
            return;
        }
    }

    PawnDirectories.Add(Pawn, DirInfo);

    // Initialize buffers with the correct directory
    FScopeLock Lock(&BufferLock);
    if (!ActiveBuffers.Contains(Pawn))
    {
        ActiveBuffers.Add(Pawn, FPawnBuffers(BufferSize, DirInfo.PawnDirectory));
    }
}

bool ARecorder::CreatePawnDirectories(
    AActor* Pawn, const FPawnDirectoryInfo& DirInfo)
{
    if (!Pawn) return false;

    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();

    // Create pawn base directory
    const FString PawnName = Pawn->GetName();
    const FString PawnDirectory = FPaths::Combine(RecordingPath, PawnName);

    if (!PlatformFile.CreateDirectoryTree(*PawnDirectory))
    {
        return false;
    }

    // Create sensor-specific directories
    TArray<TPair<bool, FString>> DirectoriesToCreate = {
        {DirInfo.bHasLidar, TEXT("Lidar")},
        {DirInfo.bHasDepth, TEXT("Depth")},
        {DirInfo.bHasRGB, TEXT("RGB")},
        {DirInfo.bHasNormal, TEXT("Normal")},
        {DirInfo.bHasSegmentation, TEXT("Segmentation")}
    };

    for (const auto& DirPair : DirectoriesToCreate)
    {
        if (DirPair.Key)
        {
            const FString SensorPath = FPaths::Combine(PawnDirectory, DirPair.Value);
            if (!PlatformFile.CreateDirectoryTree(*SensorPath))
            {
                return false;
            }
        }
    }

    // Create and initialize pose file
    const FString PoseFilePath = FPaths::Combine(PawnDirectory, TEXT("pose.txt"));
    if (!FFileHelper::SaveStringToFile(
        TEXT("# Timestamp X Y Z Roll Pitch Yaw\n"),
        *PoseFilePath,
        FFileHelper::EEncodingOptions::AutoDetect))
    {
        return false;
    }

    return true;
}

void ARecorder::SwapAndProcessBuffers()
{
    FScopeLock Lock(&BufferLock);

    // Swap active and processing buffers
    ProcessingBuffers.Empty();
    for (auto& PawnBuffer : ActiveBuffers)
    {
        const FPawnDirectoryInfo* DirInfo = PawnDirectories.Find(PawnBuffer.Key);
        if (!DirInfo)
        {
            continue;
        }
    
        ProcessingBuffers.Add(PawnBuffer.Key, MoveTemp(PawnBuffer.Value));
        PawnBuffer.Value = FPawnBuffers(BufferSize, DirInfo->PawnDirectory);
    }

    // Submit processing buffers to worker
    if (RecorderWorker && !ProcessingBuffers.IsEmpty())
    {
        for (auto& PawnBuffer : ProcessingBuffers)
        {
            RecorderWorker->EnqueueBuffer(MoveTemp(PawnBuffer.Value));
        }
    }
}

template<typename T>
void ARecorder::SubmitData(AActor* Pawn, T&& Data, EDataType Type)
{
    if (!bRecording)
    {
        UE_LOG(LogTemp, Warning, TEXT("Data submitted while not recording"));
        return;
    }

    if (!PawnDirectories.Contains(Pawn))
    {
        UE_LOG(LogTemp, Warning, TEXT("Pawn not registered: %s"), 
            *Pawn->GetName());
        return;
    }

    if (PendingTasks >= FRecorderConfig::MaxPendingTasks)
    {
        UE_LOG(LogTemp, Warning, TEXT("Max pending tasks reached"));
        return;
    }

    const auto* DirInfo = PawnDirectories.Find(Pawn);
    if (!DirInfo)
    {
        return;
    }

    // Type-specific validation
    switch (Type)
    {
    case EDataType::Lidar:
        if (!DirInfo->bHasLidar) return;
        break;
    case EDataType::DepthC:
        if (!DirInfo->bHasDepth) return;
        break;
    case EDataType::RGBC:
        if (!DirInfo->bHasRGB) return;
        break;
    case EDataType::NormalC:
        if (!DirInfo->bHasNormal) return;
        break;
    case EDataType::SegmentationC:
        if (!DirInfo->bHasSegmentation) return;
        break;
    default:
        break;
    }

    ++PendingTasks;

    FSubmissionData SubmissionData;
    SubmissionData.Type = Type;
    SubmissionData.Pawn = Pawn;
    SubmissionData.Data = MakeShared<T>(MoveTemp(Data));

    (new FAsyncTask<FAsyncSubmitTask>(this, MoveTemp(SubmissionData)))->StartBackgroundTask();
}

void ARecorder::SubmitPoseData(AActor* Pawn, FPoseData&& Data)
{
    SubmitData<FPoseData>(Pawn, MoveTemp(Data), EDataType::Pose);
}

void ARecorder::SubmitLidarData(AActor* Pawn, FLidarData&& Data)
{
    SubmitData<FLidarData>(Pawn, MoveTemp(Data), EDataType::Lidar);
}

void ARecorder::SubmitDepthData(AActor* Pawn, FDepthCameraData&& Data)
{
    SubmitData<FDepthCameraData>(Pawn, MoveTemp(Data), EDataType::DepthC);
}

void ARecorder::SubmitNormalData(AActor* Pawn, FNormalCameraData&& Data)
{
    SubmitData<FNormalCameraData>(Pawn, MoveTemp(Data), EDataType::NormalC);
}

void ARecorder::SubmitRGBData(AActor* Pawn, FRGBCameraData&& Data)
{
    // Validate data before submission
    if (Data.Width <= 0 || Data.Height <= 0 || Data.Data.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("Invalid RGB data submitted: W=%d H=%d DataSize=%d"), 
            Data.Width, Data.Height, Data.Data.Num());
        return;
    }

    if (Data.Data.Num() != Data.Width * Data.Height)
    {
        UE_LOG(LogTemp, Warning, TEXT("RGB data size mismatch: Expected %d, Got %d"), 
            Data.Width * Data.Height, Data.Data.Num());
        return;
    }

    SubmitData<FRGBCameraData>(Pawn, MoveTemp(Data), EDataType::RGBC);
}

void ARecorder::SubmitSegmentationData(AActor* Pawn, FSegmentationCameraData&& Data)
{
    SubmitData<FSegmentationCameraData>(Pawn, MoveTemp(Data), EDataType::SegmentationC);
}
