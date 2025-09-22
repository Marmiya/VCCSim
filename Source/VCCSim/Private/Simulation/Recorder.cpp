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
#include "Engine/World.h"
#include "Misc/Paths.h"
#include "TimerManager.h"
#include "HAL/PlatformFilemanager.h"

DEFINE_LOG_CATEGORY_STATIC(LogRecorder, Log, All);

ARecorder::ARecorder()
{
    PrimaryActorTick.bCanEverTick = false;

    // Create timestamp-based directory like RuntimeLogs
    FDateTime Now = FDateTime::Now();
    FString Timestamp = FString::Printf(TEXT("%04d%02d%02d_%02d%02d%02d"),
        Now.GetYear(), Now.GetMonth(), Now.GetDay(),
        Now.GetHour(), Now.GetMinute(), Now.GetSecond());

    RecordingPath = FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("RuntimeLogs"), Timestamp);

    // Create complete recording directory structure immediately
    CreateRecordingDirectoryStructure();
}

void ARecorder::CreateRecordingDirectoryStructure()
{
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();

    // Create base recording directory
    if (PlatformFile.CreateDirectoryTree(*RecordingPath))
    {
        UE_LOG(LogRecorder, Log, TEXT("Created base recording directory: %s"), *RecordingPath);
    }
    else
    {
        UE_LOG(LogRecorder, Error, TEXT("Failed to create base recording directory: %s"), *RecordingPath);
        return;
    }

    // Pre-create standard sensor directories for common sensor types
    TArray<FString> SensorDirectories = {
        TEXT("RGB"),
        TEXT("Depth"),
        TEXT("Normal"),
        TEXT("Segmentation"),
        TEXT("Lidar")
    };

    // Create directories for potential drone actors (we don't know actor names yet)
    // These will be created dynamically when sensors are registered
    UE_LOG(LogRecorder, Log, TEXT("Recording directory structure ready at: %s"), *RecordingPath);
}

ARecorder::~ARecorder()
{
    StopRecording();
    ShutdownAsyncWriter();
}

void ARecorder::StartRecording()
{
    if (bIsRecording)
    {
        UE_LOG(LogRecorder, Warning, TEXT("Recording already active"));
        return;
    }

    if (!GetWorld())
    {
        UE_LOG(LogRecorder, Error, TEXT("Cannot start recording - no valid world"));
        return;
    }

    // Initialize file writer
    InitializeAsyncWriter();

    // Start the recording timer
    float ClampedInterval = FMath::Max(RecordingInterval, 0.001f);
    GetWorld()->GetTimerManager().SetTimer(
        RecordingTimerHandle,
        [this]()
        {
            TickRecording();
        },
        ClampedInterval,
        true
    );

    bIsRecording = true;
    RecordState = true;

    OnRecordStateChanged.Broadcast(true);
    UE_LOG(LogRecorder, Log, TEXT("Recording started at path: %s with interval: %.4f seconds"),
           *RecordingPath, ClampedInterval);
}

void ARecorder::StopRecording()
{
    if (!bIsRecording)
    {
        return;
    }

    // Clear the timer
    if (GetWorld() && RecordingTimerHandle.IsValid())
    {
        GetWorld()->GetTimerManager().ClearTimer(RecordingTimerHandle);
        RecordingTimerHandle.Invalidate();
    }

    // Shutdown file writer
    ShutdownAsyncWriter();

    bIsRecording = false;
    RecordState = false;

    OnRecordStateChanged.Broadcast(false);
    UE_LOG(LogRecorder, Log, TEXT("Recording stopped"));
}

void ARecorder::ToggleRecording()
{
    if (bIsRecording)
    {
        StopRecording();
    }
    else
    {
        StartRecording();
    }
}

void ARecorder::CreateActorDirectories(const FString& ActorName, TSet<ESensorType>&& SensorTypes)
{
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();

    // Create actor directory
    FString ActorDir = FPaths::Combine(RecordingPath, ActorName);
    if (PlatformFile.CreateDirectoryTree(*ActorDir))
    {
        UE_LOG(LogRecorder, Log, TEXT("Created actor directory: %s"), *ActorDir);
    }

    // Create only the sensor directories that will be recorded
    if (SensorTypes.Find(ESensorType::RGBCamera))
    {
        FString RGBDir = FPaths::Combine(ActorDir, TEXT("RGB"));
        PlatformFile.CreateDirectoryTree(*RGBDir);
    }
    if (SensorTypes.Find(ESensorType::DepthCamera))
    {
        FString DepthDir = FPaths::Combine(ActorDir, TEXT("Depth"));
        PlatformFile.CreateDirectoryTree(*DepthDir);
    }
    if (SensorTypes.Find(ESensorType::NormalCamera))
    {
        FString NormalDir = FPaths::Combine(ActorDir, TEXT("Normal"));
        PlatformFile.CreateDirectoryTree(*NormalDir);
    }
    if (SensorTypes.Find(ESensorType::SegmentationCamera))
    {
        FString SegDir = FPaths::Combine(ActorDir, TEXT("Segmentation"));
        PlatformFile.CreateDirectoryTree(*SegDir);
    }
    if (SensorTypes.Find(ESensorType::Lidar))
    {
        FString LidarDir = FPaths::Combine(ActorDir, TEXT("Lidar"));
        PlatformFile.CreateDirectoryTree(*LidarDir);
    }
}

void ARecorder::TickRecording()
{
    if (!bIsRecording)
    {
        UE_LOG(LogRecorder, Warning, TEXT("TickRecording: Recording not active"));
        return;
    }

    SensorRegistry.CleanupInvalidEntries();
    CollectSensorDataConcurrently();
}

void ARecorder::CollectSensorDataConcurrently()
{
    TArray<FSensorRegistryEntry> AllSensors = SensorRegistry.GetAllSensors();
    if (AllSensors.Num() == 0)
    {
        return;
    }

    TArray<TFuture<FSensorDataPacket>> ConcurrentTasks;
    ConcurrentTasks.Reserve(AllSensors.Num());

    for (const FSensorRegistryEntry& Entry : AllSensors)
    {
        if (Entry.IsValid())
        {
            if (ISensorDataProvider* Provider = Entry.GetProvider())
            {
                TFuture<FSensorDataPacket> Future = Provider->CaptureDataAsync();
                ConcurrentTasks.Add(MoveTemp(Future));
            }
        }
    }

    if (ConcurrentTasks.Num() == 0)
    {
        return;
    }

    // Stream processing: handle each sensor result as soon as it's ready
    for (auto& Future : ConcurrentTasks)
    {
        Future.Then([this](TFuture<FSensorDataPacket> CompletedFuture)
        {
            FSensorDataPacket Result = CompletedFuture.Get();
            if (Result.bValid && Result.Data.IsValid())
            {
                // Process immediately on background thread - no GameThread needed
                Async(EAsyncExecution::TaskGraph, [this, Result = MoveTemp(Result)]()
                {
                    if (FileWriter.IsValid())
                    {
                        FileWriter->WriteDataAsync(Result);
                    }
                });
            }
        });
    }
}


void ARecorder::InitializeAsyncWriter()
{
    if (FileWriter.IsValid())
    {
        return;
    }

    FileWriter = MakeUnique<FAsyncFileWriter>(RecordingPath);
    UE_LOG(LogRecorder, Log, TEXT("Async file writer initialized"));
}

void ARecorder::ShutdownAsyncWriter()
{
    if (FileWriter.IsValid())
    {
        FileWriter->Flush();
        FileWriter.Reset();
        UE_LOG(LogRecorder, Log, TEXT("Async file writer shutdown"));
    }
}

void ARecorder::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    StopRecording();
    Super::EndPlay(EndPlayReason);
}