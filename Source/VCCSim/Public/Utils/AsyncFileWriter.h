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
#include "Sensors/ISensorDataProvider.h"
#include <atomic>

DECLARE_LOG_CATEGORY_EXTERN(LogAsyncFileWriter, Log, All);

class VCCSIM_API FAsyncFileWriter : public FRunnable
{
public:
    explicit FAsyncFileWriter(const FString& InBasePath);
    virtual ~FAsyncFileWriter() override;

    void WriteDataAsync(const FSensorDataPacket& DataPacket);
    void Flush();

    virtual bool Init() override;
    virtual uint32 Run() override;
    virtual void Stop() override;
    virtual void Exit() override;

private:
    void ProcessWriteQueue();
    void WriteDataToFile(const FSensorDataPacket& DataPacket);
    FString GetFilePathForSensor(const FSensorDataPacket& DataPacket);
    void CreateDirectoryStructure(const FSensorDataPacket& DataPacket);

    FString BasePath;
    TQueue<FSensorDataPacket, EQueueMode::Mpsc> WriteQueue;
    TUniquePtr<FRunnableThread> WriterThread;
    std::atomic<bool> bShouldStop{false};

    mutable FCriticalSection DirectoryCreationLock;
    TSet<FString> CreatedDirectories;
};