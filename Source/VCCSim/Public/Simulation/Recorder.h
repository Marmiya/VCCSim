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

#pragma once

#include "CoreMinimal.h"
#include "HAL/Runnable.h"
#include "HAL/RunnableThread.h"
#include "Containers/Queue.h"
#include "Recorder.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(
    FRecordStateChanged, bool, RecordState);

// Configuration struct for tunable parameters
struct FRecorderConfig
{
    // Buffer and memory settings
    static constexpr int32 StringReserveSize = 16 * 1024;   // 16KB for string builders
    static constexpr int32 InitialPoolSize = 8;             // Pre-allocated buffers
    static constexpr int32 MaxPoolSize = 32;                // Maximum buffer pool size

    // Threading and processing settings
    static constexpr float MinSleepInterval = 0.0001f;      // 0.1ms minimum sleep
    static constexpr float MaxSleepInterval = 0.016f;       // Never sleep longer than one frame
    static constexpr float SleepMultiplier = 1.5f;          // Exponential backoff
    static constexpr int32 BatchSize = 32;                  // Process multiple items per batch
    static constexpr int32 MaxQueueSize = 10000;           // Maximum items in queue
    static constexpr int32 MaxPendingTasks = 1000;         // Maximum pending async tasks
};

class IImageWrapper;

class FImageWrapperCache
{
public:
    static FImageWrapperCache& Get()
    {
        static FImageWrapperCache Instance;
        return Instance;
    }

    ::IImageWrapper* GetPNGWrapper();

private:
    FImageWrapperCache() : ImageWrapperModule(nullptr) {}
    ~FImageWrapperCache() = default;

    FCriticalSection CacheLock;
    IImageWrapperModule* ImageWrapperModule;
    TSharedPtr<IImageWrapper> PNGWrapper;
};

// Sensor data structures
struct FSensorData
{
    double Timestamp;  // High-precision timestamp
    virtual ~FSensorData() = default;
};

struct FPoseData final : public FSensorData
{
    FVector Location;
    FRotator Rotation;
};

struct FLidarData final : public FSensorData
{
    TArray<FVector3f> Data;
};

struct FDepthCameraData final : public FSensorData
{
    int32 Width;
    int32 Height;
    int32 SensorIndex;
    TArray<float> Data;
};

struct FRGBCameraData final : public FSensorData
{
    int32 Width;
    int32 Height;
    int32 SensorIndex;
    TArray<FColor> Data;
};

struct FSegmentationCameraData final : public FSensorData
{
    int32 Width;
    int32 Height;
    TArray<FColor> Data;
};

struct FNormalCameraData final : public FSensorData
{
    int32 Width;
    int32 Height;
    int32 SensorIndex;
    TArray<FLinearColor> Data;
};

// Buffer pool for efficient memory management
class FBufferPool
{
public:
    TArray<uint8>* AcquireBuffer(int32 Size);
    void ReleaseBuffer(TArray<uint8>* Buffer);

private:
    FCriticalSection PoolLock;
    TArray<TArray<uint8>*> Buffers;
};

// Adaptive sleep strategy
class FAdaptiveSleeper
{
public:
    void Sleep();
    void Reset();

private:
    float CurrentInterval = FRecorderConfig::MinSleepInterval;
    int32 EmptyCount = 0;
};

// Ring buffer with improved memory management
template<typename T>
class TRingBuffer
{
public:

    explicit TRingBuffer(int32 Size) 
        : MaxSize(Size)
        , Head(0)
        , Tail(0)
        , ItemCount(0)
    {
        Buffer.SetNum(Size);
    }

    TRingBuffer(TRingBuffer&& Other) noexcept
        : Buffer(MoveTemp(Other.Buffer))
        , MaxSize(Other.MaxSize)
        , Head(Other.Head)
        , Tail(Other.Tail)
        , ItemCount(Other.ItemCount)
    {
        Other.MaxSize = 0;
        Other.Head = 0;
        Other.Tail = 0;
        Other.ItemCount = 0;
    }

    TRingBuffer& operator=(TRingBuffer&& Other) noexcept
    {
        if (this != &Other)
        {
            Buffer = MoveTemp(Other.Buffer);
            MaxSize = Other.MaxSize;
            Head = Other.Head;
            Tail = Other.Tail;
            ItemCount = Other.ItemCount;

            Other.MaxSize = 0;
            Other.Head = 0;
            Other.Tail = 0;
            Other.ItemCount = 0;
        }
        return *this;
    }

    // Delete copy operations explicitly
    TRingBuffer(const TRingBuffer&) = delete;
    TRingBuffer& operator=(const TRingBuffer&) = delete;

    bool Enqueue(T&& Item)
    {
        FScopeLock Lock(&BufferLock);
        if (ItemCount >= MaxSize) return false;
    
        Buffer[Tail] = MoveTemp(Item);
        Tail = (Tail + 1) % MaxSize;
        ++ItemCount;
        return true;
    }

    bool Dequeue(T& OutItem)
    {
        FScopeLock Lock(&BufferLock);
        if (ItemCount == 0) return false;
    
        OutItem = MoveTemp(Buffer[Head]);
        Head = (Head + 1) % MaxSize;
        --ItemCount;
        return true;
    }

    void Reset() 
    { 
        FScopeLock Lock(&BufferLock);
        Head = Tail = ItemCount = 0; 
    }

    bool IsEmpty() const 
    { 
        return ItemCount == 0; 
    }

    int32 Count() const 
    { 
        return ItemCount; 
    }

private:
    TArray<T> Buffer;
    int32 MaxSize;
    int32 Head;
    int32 Tail;
    int32 ItemCount;
    FCriticalSection BufferLock;
};

// Storage for pawn-specific data
struct FPawnBuffers
{
    static constexpr int32 DefaultSize = 10;

    TRingBuffer<FPoseData> Pose;
    TRingBuffer<FLidarData> Lidar;
    TRingBuffer<FDepthCameraData> DepthC;
    TRingBuffer<FRGBCameraData> RGBC;
    FString PawnDirectory;

    FPawnBuffers()
        : Pose(DefaultSize)
        , Lidar(DefaultSize)
        , DepthC(DefaultSize)
        , RGBC(DefaultSize)
    {}

    explicit FPawnBuffers(int32 Size, const FString& InPawnDirectory = TEXT("")) 
        : Pose(Size)
        , Lidar(Size)
        , DepthC(Size)
        , RGBC(Size)
        , PawnDirectory(InPawnDirectory)
    {}

    // Delete copy constructor and assignment
    FPawnBuffers(const FPawnBuffers&) = delete;
    FPawnBuffers& operator=(const FPawnBuffers&) = delete;

    // Add move constructor and assignment
    FPawnBuffers(FPawnBuffers&& Other) noexcept
        : Pose(MoveTemp(Other.Pose))
        , Lidar(MoveTemp(Other.Lidar))
        , DepthC(MoveTemp(Other.DepthC))
        , RGBC(MoveTemp(Other.RGBC))
        , PawnDirectory(MoveTemp(Other.PawnDirectory))  // Add this line
    {}

    FPawnBuffers& operator=(FPawnBuffers&& Other) noexcept
    {
        if (this != &Other)
        {
            Pose = MoveTemp(Other.Pose);
            Lidar = MoveTemp(Other.Lidar);
            DepthC = MoveTemp(Other.DepthC);
            RGBC = MoveTemp(Other.RGBC);
            PawnDirectory = MoveTemp(Other.PawnDirectory);  // Add this line
        }
        return *this;
    }
};

// Directory information for each pawn
struct FPawnDirectoryInfo
{
    bool bHasLidar = false;
    bool bHasDepth = false;
    bool bHasRGB = false;
    bool bHasNormal = false;
    bool bHasSegmentation = false;
    FString PawnDirectory;
    FString PoseFilePath;
};

// Task types for async submission
enum class EDataType : uint8
{
    Pose,
    Lidar,
    DepthC,
    RGBC,
    NormalC,
    SegmentationC,
};

// Unified data structure for async tasks
struct FSubmissionData
{
    EDataType Type;
    AActor* Pawn;
    TSharedPtr<FSensorData> Data;
};

// Worker thread for data processing
class FRecorderWorker final : public FRunnable 
{
public:
    FRecorderWorker(const FString& InBasePath, int32 InBufferSize);
    virtual ~FRecorderWorker() override;

    // FRunnable interface
    virtual bool Init() override;
    virtual uint32 Run() override;
    virtual void Stop() override;
    virtual void Exit() override;

    void EnqueueBuffer(FPawnBuffers&& Buffer);
    bool IsQueueEmpty() const;

private:
    FString BasePath;
    int32 BufferSize;

    FRunnableThread* Thread;

    mutable FCriticalSection QueueLock;
    TQueue<FPawnBuffers> DataQueue;

    TAtomic<bool> bStopRequested;

    FBufferPool BufferPool;
    FAdaptiveSleeper Sleeper;

    // Helper functions
    int32 ProcessBatch(TArray<FPawnBuffers>& BatchBuffers);
    void ProcessBuffer(FPawnBuffers& Buffer);
    bool SaveLidarData(const FLidarData& LidarData, const FString& Directory);
    bool SaveDepthData(const FDepthCameraData& DepthData, const FString& Directory);
    bool SaveRGBData(const FRGBCameraData& RGBData, const FString& Directory);
};

// Async task for data submission
class FAsyncSubmitTask : public FNonAbandonableTask
{
public:
    FAsyncSubmitTask(class ARecorder* InRecorder, FSubmissionData&& InData);
    void DoWork();

    FORCEINLINE TStatId GetStatId() const
    {
        RETURN_QUICK_DECLARE_CYCLE_STAT(FAsyncSubmitTask,
            STATGROUP_ThreadPoolAsyncTasks);
    }

private:
    class ARecorder* Recorder;
    FSubmissionData SubmissionData;
};

// Main recorder actor
UCLASS()
class VCCSIM_API ARecorder : public AActor
{
    GENERATED_BODY()
    
    friend class FAsyncSubmitTask;

public:
    ARecorder();
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

    // Recording control
    void StartRecording();
    void StopRecording();
    void ToggleRecording();

    // Pawn registration
    void RegisterPawn(AActor* Pawn,  bool bHasLidar, bool bHasDepth,
        bool bHasRGB, bool bHasNormal,  bool bHasSegmentation);

    // Data submission methods
    void SubmitPoseData(AActor* Pawn, FPoseData&& Data);
    void SubmitLidarData(AActor* Pawn, FLidarData&& Data);
    void SubmitDepthData(AActor* Pawn, FDepthCameraData&& Data);
    void SubmitRGBData(AActor* Pawn, FRGBCameraData&& Data);
    void SubmitNormalData(AActor* Pawn, FNormalCameraData&& Data);
    void SubmitSegmentationData(AActor* Pawn, FSegmentationCameraData&& Data);

    // Configuration properties
    UPROPERTY(EditAnywhere, Category = "Recording")
    int32 BufferSize = 100;

    FString RecordingPath;

    FRecordStateChanged OnRecordStateChanged;
    bool RecordState = false; // If the sensors should submit data

private:
    bool bRecording = false; // Recording state of the recorder
    
    TAtomic<int32> PendingTasks;

    TMap<AActor*, FPawnDirectoryInfo> PawnDirectories;
    TMap<AActor*, FPawnBuffers> ActiveBuffers;
    TMap<AActor*, FPawnBuffers> ProcessingBuffers;

    FCriticalSection BufferLock;
    TUniquePtr<FRecorderWorker> RecorderWorker;
    FBufferPool BufferPool;

    // Helper functions
    void SwapAndProcessBuffers();
    bool CreatePawnDirectories(AActor* Pawn, const FPawnDirectoryInfo& DirInfo);

    template<typename T>
    void SubmitData(AActor* Pawn, T&& Data, EDataType Type);
};

// Template specializations
template void ARecorder::SubmitData<FPoseData>(AActor*, FPoseData&&, EDataType);
template void ARecorder::SubmitData<FLidarData>(AActor*, FLidarData&&, EDataType);
template void ARecorder::SubmitData<FDepthCameraData>(AActor*, FDepthCameraData&&, EDataType);
template void ARecorder::SubmitData<FRGBCameraData>(AActor*, FRGBCameraData&&, EDataType);
template void ARecorder::SubmitData<FNormalCameraData>(AActor*, FNormalCameraData&&, EDataType);