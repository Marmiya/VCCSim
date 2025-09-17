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

DEFINE_LOG_CATEGORY_STATIC(LogRecorder, Log, All);

#include "Simulation/Recorder.h"
#include "HAL/FileManager.h"
#include "Misc/Paths.h"
#include "Async/AsyncWork.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Modules/ModuleManager.h"
#include "Containers/StringConv.h"
#include "HAL/PlatformFileManager.h"
#include "Utils/ImageProcesser.h"


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
FBufferPool::~FBufferPool()
{
    Cleanup();
}

TArray<uint8>* FBufferPool::AcquireBuffer(int32 Size)
{
    FScopeLock Lock(&PoolLock);
    if (Buffers.Num() > 0)
    {
        auto* Buffer = Buffers.Pop();
        Buffer->SetNum(Size, EAllowShrinking::No);
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

void FBufferPool::Cleanup()
{
    FScopeLock Lock(&PoolLock);
    for (auto* Buffer : Buffers)
    {
        delete Buffer;
    }
    Buffers.Empty();
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
FRecorderWorker::FRecorderWorker(const FString& InBasePath, int32 InBufferSize, bool bInBetterVisualsRecording, ARecorder* InRecorder)
    : BasePath(InBasePath)
    , BufferSize(InBufferSize)
    , bBetterVisualsRecording(bInBetterVisualsRecording)
    , RecorderRef(InRecorder)
    , Thread(nullptr)
    , bStopRequested(false)
{
    // Debug logging
    UE_LOG(LogRecorder, Warning, TEXT("FRecorderWorker created with BufferSize: %d"), BufferSize);
    
    // Validate immediately
    check(BufferSize > 0 && BufferSize < 100000);
    
    Thread = FRunnableThread::Create(this, TEXT("RecorderWorker"), 0, TPri_Normal);
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

    // Cleanup buffer pool
    BufferPool.Cleanup();
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

    // Validate BufferSize before using it
    if (BufferSize <= 0 || BufferSize > 100000)
    {
        UE_LOG(LogRecorder, Error, TEXT("Corrupted BufferSize detected: %d."
                                    " Stopping worker."), BufferSize);
        return 0;  // Stop processing to prevent crashes
    }

    {
        FScopeLock Lock(&QueueLock);
        while (BatchBuffers.Num() < FRecorderConfig::BatchSize && !DataQueue.IsEmpty())
        {
            FPawnBuffers Buffer(BufferSize);  // Now safe to use
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
                TEXT("%.9f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n"),
                PoseData.Timestamp,
                PoseData.Location.X, PoseData.Location.Y, PoseData.Location.Z,
                PoseData.Quaternion.X, PoseData.Quaternion.Y, PoseData.Quaternion.Z, PoseData.Quaternion.W
            );
        
            FFileHelper::SaveStringToFile(
                PoseContent,
                *FPaths::Combine(Buffer.PawnDirectory, TEXT("poses.txt")),
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

    // Process Normal data
    {
        FNormalCameraData NormalData;
        while (Buffer.NormalC.Dequeue(NormalData))
        {
            SaveNormalData(NormalData, Buffer.PawnDirectory);
        }
    }

    // Process Segmentation data
    {
        FSegmentationCameraData SegmentationData;
        while (Buffer.SegmentationC.Dequeue(SegmentationData))
        {
            SaveSegmentationData(SegmentationData, Buffer.PawnDirectory);
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
    if (DepthData.Data.Num() == 0 || DepthData.Width <= 0 || DepthData.Height <= 0)
    {
        UE_LOG(LogRecorder, Warning, TEXT("Invalid depth data: W=%d H=%d DataSize=%d"),
            DepthData.Width, DepthData.Height, DepthData.Data.Num());
        return false;
    }

    const FString Filename = FString::Printf(
        TEXT("%s_%d.png"),
        *FString::Printf(TEXT("%.9f"), DepthData.Timestamp),
        DepthData.SensorIndex
    );

    const FString FilePath = FPaths::Combine(Directory, TEXT("Depth"), Filename);

    if (bBetterVisualsRecording)
    {
        // Use visual-friendly depth format (inverted, gamma corrected)
        if (RecorderRef)
        {
            RecorderRef->IncrementAsyncTasks();
        }
        AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [this, DepthData, FilePath]() {
            FAsyncDepthVisualSaveTask Task(
                DepthData.Data,
                FIntPoint(DepthData.Width, DepthData.Height),
                FilePath,
                0.0f,       // MinRange
                5000.0f    // MaxRange
            );
            Task.DoWork();
            if (RecorderRef)
            {
                RecorderRef->DecrementAsyncTasks();
            }
        });
    }
    else
    {
        // Use accurate 16-bit format
        TArray<FFloat16Color> DepthPixels;
        DepthPixels.Reserve(DepthData.Data.Num());

        for (float Depth : DepthData.Data)
        {
            FFloat16Color DepthPixel;
            DepthPixel.R = FFloat16(Depth);
            DepthPixel.G = FFloat16(0.0f);
            DepthPixel.B = FFloat16(0.0f);
            DepthPixel.A = FFloat16(1.0f);
            DepthPixels.Add(DepthPixel);
        }

        if (RecorderRef)
        {
            RecorderRef->IncrementAsyncTasks();
        }
        AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [this, DepthPixels = MoveTemp(DepthPixels), FilePath, DepthData]() {
            FAsyncDepth16SaveTask Task(
                DepthPixels,
                FIntPoint(DepthData.Width, DepthData.Height),
                FilePath,
                1.0f  // Depth scale
            );
            Task.DoWork();
            if (RecorderRef)
            {
                RecorderRef->DecrementAsyncTasks();
            }
        });
    }

    return true;
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

    ImageBuffer->SetNum(ExpectedSize, EAllowShrinking::No);

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
    case EDataType::NormalC:
        EnqueueDataWithRetry(PawnBuffers.NormalC,
            *static_cast<FNormalCameraData*>(SubmissionData.Data.Get()));
        break;
    case EDataType::SegmentationC:
        EnqueueDataWithRetry(PawnBuffers.SegmentationC,
            *static_cast<FSegmentationCameraData*>(SubmissionData.Data.Get()));
        break;
    }

    --Recorder->PendingTasks;
}

// ARecorder implementation
ARecorder::ARecorder()
    : PendingTasks(0)
    , PendingAsyncTasks(0)
{
    PrimaryActorTick.bCanEverTick = false;
    
    // Ensure BufferSize is always valid
    if (BufferSize <= 0 || BufferSize > 100000)
    {
        UE_LOG(LogRecorder, Warning, TEXT("Invalid BufferSize in constructor: %d. Setting to 100."), BufferSize);
        BufferSize = 100;
    }
}

void ARecorder::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    StopRecording();

    // Force cleanup all remaining resources
    if (RecorderWorker)
    {
        RecorderWorker.Reset();
    }

    // Force cleanup buffer pools
    BufferPool.Cleanup();

    // Clear all maps to ensure memory is released
    PawnDirectories.Empty();
    {
        FScopeLock Lock(&BufferLock);
        ActiveBuffers.Empty();
        ProcessingBuffers.Empty();
    }

    Super::EndPlay(EndPlayReason);
}

void ARecorder::StartRecording()
{
    if (bRecording) return;

    // Validate BufferSize before creating worker
    if (BufferSize <= 0 || BufferSize > 100000)
    {
        UE_LOG(LogRecorder, Error, TEXT("Invalid BufferSize: %d. Resetting to default 100."), BufferSize);
        BufferSize = 100;  // Reset to safe default
    }

    UE_LOG(LogRecorder, Warning, TEXT("Starting recording with BufferSize: %d"), BufferSize);

    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.CreateDirectoryTree(*RecordingPath))
    {
        UE_LOG(LogRecorder, Error, TEXT("Failed to create recording directory: %s"),
            *RecordingPath);
        return;
    }

    // Create directories for each registered pawn
    for (const auto& PawnEntry : PawnDirectories)
    {
        if (!CreatePawnDirectories(PawnEntry.Key, PawnEntry.Value))
        {
            UE_LOG(LogRecorder, Error, TEXT("Failed to create pawn directories"));
            return;
        }
    }

    // Initialize worker with validated BufferSize and visual settings
    RecorderWorker = MakeUnique<FRecorderWorker>(RecordingPath, BufferSize, bBetterVisualsRecording, this);
    bRecording = true;
}

void ARecorder::StopRecording()
{
    if (!bRecording) return;

    bRecording = false;

    // Wait for pending tasks with timeout
    const double StartTime = FPlatformTime::Seconds();
    constexpr double TimeoutSeconds = 10.0; // Increased timeout for async image tasks

    while (PendingTasks > 0 || PendingAsyncTasks > 0)
    {
        FPlatformProcess::Sleep(0.01f);
        if (FPlatformTime::Seconds() - StartTime > TimeoutSeconds)
        {
            UE_LOG(LogRecorder, Warning, TEXT("Timeout waiting for tasks. Pending: %d, AsyncTasks: %d"),
                PendingTasks.Load(), PendingAsyncTasks.Load());
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
                UE_LOG(LogRecorder, Warning, TEXT("Timeout waiting for worker queue to empty"));
                break;
            }
        }
        RecorderWorker.Reset();
    }

    // Clear buffers and cleanup buffer pool
    {
        FScopeLock Lock(&BufferLock);
        ActiveBuffers.Empty();
        ProcessingBuffers.Empty();
    }

    // Cleanup buffer pool to release all allocated memory
    BufferPool.Cleanup();
}

void ARecorder::ToggleRecording()
{
    RecordState = !RecordState;
    
    UE_LOG(LogRecorder, Warning, TEXT("Recording state: "
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
            UE_LOG(LogRecorder, Error, TEXT("Failed to create directories for pawn:"
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
    const FString PoseFilePath = FPaths::Combine(PawnDirectory, TEXT("poses.txt"));
    if (!FFileHelper::SaveStringToFile(
        TEXT("# Format: Timestamp X Y Z Qx Qy Qz Qw\n"),
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
        UE_LOG(LogRecorder, Warning, TEXT("Data submitted while not recording"));
        return;
    }

    if (!PawnDirectories.Contains(Pawn))
    {
        UE_LOG(LogRecorder, Warning, TEXT("Pawn not registered: %s"), 
            *Pawn->GetName());
        return;
    }

    if (PendingTasks >= FRecorderConfig::MaxPendingTasks)
    {
        UE_LOG(LogRecorder, Warning, TEXT("Max pending tasks reached"));
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

    // Use AsyncTask to avoid memory leaks
    AsyncTask(ENamedThreads::GameThread, [this, SubmissionData = MoveTemp(SubmissionData)]() mutable {
        FAsyncSubmitTask Task(this, MoveTemp(SubmissionData));
        Task.DoWork();
    });
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
        UE_LOG(LogRecorder, Warning, TEXT("Invalid RGB data submitted: W=%d H=%d DataSize=%d"), 
            Data.Width, Data.Height, Data.Data.Num());
        return;
    }

    if (Data.Data.Num() != Data.Width * Data.Height)
    {
        UE_LOG(LogRecorder, Warning, TEXT("RGB data size mismatch: Expected %d, Got %d"), 
            Data.Width * Data.Height, Data.Data.Num());
        return;
    }

    SubmitData<FRGBCameraData>(Pawn, MoveTemp(Data), EDataType::RGBC);
}

void ARecorder::SubmitSegmentationData(AActor* Pawn, FSegmentationCameraData&& Data)
{
    SubmitData<FSegmentationCameraData>(Pawn, MoveTemp(Data), EDataType::SegmentationC);
}

bool FRecorderWorker::SaveNormalData(const FNormalCameraData& NormalData, const FString& Directory)
{
    if (NormalData.Data.Num() == 0 || NormalData.Width <= 0 || NormalData.Height <= 0)
    {
        UE_LOG(LogRecorder, Warning, TEXT("Invalid normal data: W=%d H=%d DataSize=%d"),
            NormalData.Width, NormalData.Height, NormalData.Data.Num());
        return false;
    }

    if (bBetterVisualsRecording)
    {
        // Use visual-friendly PNG format
        const FString Filename = FString::Printf(
            TEXT("%s_%d.png"),
            *FString::Printf(TEXT("%.9f"), NormalData.Timestamp),
            NormalData.SensorIndex
        );

        const FString FilePath = FPaths::Combine(Directory, TEXT("Normal"), Filename);

        if (RecorderRef)
        {
            RecorderRef->IncrementAsyncTasks();
        }
        AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [this, NormalData, FilePath]() {
            FAsyncNormalVisualSaveTask Task(
                NormalData.Data,
                FIntPoint(NormalData.Width, NormalData.Height),
                FilePath
            );
            Task.DoWork();
            if (RecorderRef)
            {
                RecorderRef->DecrementAsyncTasks();
            }
        });
    }
    else
    {
        // Use accurate EXR format
        const FString Filename = FString::Printf(
            TEXT("%s_%d.exr"),
            *FString::Printf(TEXT("%.9f"), NormalData.Timestamp),
            NormalData.SensorIndex
        );

        const FString FilePath = FPaths::Combine(Directory, TEXT("Normal"), Filename);

        if (RecorderRef)
        {
            RecorderRef->IncrementAsyncTasks();
        }
        AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [this, NormalData, FilePath]() {
            FAsyncNormalEXRSaveTask Task(
                NormalData.Data,
                FIntPoint(NormalData.Width, NormalData.Height),
                FilePath
            );
            Task.DoWork();
            if (RecorderRef)
            {
                RecorderRef->DecrementAsyncTasks();
            }
        });
    }

    return true;
}

bool FRecorderWorker::SaveSegmentationData(const FSegmentationCameraData& SegmentationData, const FString& Directory)
{
    if (SegmentationData.Data.Num() == 0 || SegmentationData.Width <= 0 || SegmentationData.Height <= 0)
    {
        UE_LOG(LogRecorder, Warning, TEXT("Invalid segmentation data: W=%d H=%d DataSize=%d"),
            SegmentationData.Width, SegmentationData.Height, SegmentationData.Data.Num());
        return false;
    }

    const FString Filename = FString::Printf(
        TEXT("%s.png"),
        *FString::Printf(TEXT("%.9f"), SegmentationData.Timestamp)
    );

    const FString FilePath = FPaths::Combine(Directory, TEXT("Segmentation"), Filename);

    // Use AsyncTask to avoid memory leaks
    if (RecorderRef)
    {
        RecorderRef->IncrementAsyncTasks();
    }
    AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [this, SegmentationData, FilePath]() {
        FAsyncImageSaveTask Task(
            SegmentationData.Data,
            FIntPoint(SegmentationData.Width, SegmentationData.Height),
            FilePath
        );
        Task.DoWork();
        if (RecorderRef)
        {
            RecorderRef->DecrementAsyncTasks();
        }
    });

    return true;
}
