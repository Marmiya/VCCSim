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
#include "Async/Async.h"
#include "Async/AsyncWork.h"
#include "Sensors/SensorBase.h"
#include <atomic>

enum class ESensorType : uint8;

DECLARE_LOG_CATEGORY_EXTERN(LogAsyncFileWriter, Log, All);


struct FCompressedImageData
{
    TArray64<uint8> CompressedData;
    FString FilePath;
    ESensorType SensorType;
};

class FImageCompressionTask : public FNonAbandonableTask
{
public:

    FImageCompressionTask(
        const FSensorDataPacket& InDataPacket,
        const FString& InBasePath,
        TQueue<TSharedPtr<FCompressedImageData>, EQueueMode::Mpsc>& InCompressedDataQueue)
        : DataPacket(InDataPacket), BasePath(InBasePath),
            CompressedDataQueue(InCompressedDataQueue){}

    void DoWork();

    FORCEINLINE TStatId GetStatId() const
    {
        RETURN_QUICK_DECLARE_CYCLE_STAT(FImageCompressionTask, STATGROUP_ThreadPoolAsyncTasks);
    }

private:
    FSensorDataPacket DataPacket;
    FString BasePath;
    TQueue<TSharedPtr<FCompressedImageData>, EQueueMode::Mpsc>& CompressedDataQueue;

    void CompressRGBData();
    void CompressDepthData();
    void CompressNormalData();
    void CompressSegmentationData();
    void CompressLidarData();

    TSharedPtr<FCompressedImageData> CreateCompressedResult(const FString& SubPath, const FString& Extension, double Timestamp);
    bool CompressImageToPNG(const TArray<FColor>& ImageData, int32 Width, int32 Height, TArray64<uint8>& OutCompressedData);
    bool CompressImageToEXR(const TArray<FFloat16Color>& ImageData, int32 Width, int32 Height, TArray64<uint8>& OutCompressedData);
};

class VCCSIM_API FAsyncFileWriter
{
public:
    explicit FAsyncFileWriter(const FString& InBasePath);
    ~FAsyncFileWriter();

    void WriteData(FSensorDataPacket&& DataPacket);
    void Flush();
    int32 GetPendingTaskCount() const { return PendingCompressionTasks.load(); }

private:
    void WriteCompressedDataToFile(const FCompressedImageData& CompressedData);

    FString BasePath;
    std::atomic<int32> PendingCompressionTasks{0};
    TQueue<TSharedPtr<FCompressedImageData>, EQueueMode::Mpsc> CompressedDataQueue;
    TUniquePtr<FRunnableThread> IOThread;
    std::atomic<bool> bShouldStop{false};

    mutable FCriticalSection DirectoryCreationLock;
    TSet<FString> CreatedDirectories;

    class FIOWorker : public FRunnable
    {
    public:
        FIOWorker(FAsyncFileWriter* InOwner) : Owner(InOwner) {}
        virtual uint32 Run() override;
        virtual void Stop() override { bShouldStop = true; }
    private:
        FAsyncFileWriter* Owner;
        std::atomic<bool> bShouldStop{false};
    };
    TUniquePtr<FIOWorker> IOWorker;
};