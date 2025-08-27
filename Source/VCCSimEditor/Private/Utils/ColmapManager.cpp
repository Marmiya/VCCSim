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

#include "Utils/ColmapManager.h"
#include "Utils/VCCSimDataConverter.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformFilemanager.h"
#include "Misc/Paths.h"
#include "Misc/DateTime.h"
#include "Engine/Engine.h"
#include "Framework/Notifications/NotificationManager.h"
#include "Widgets/Notifications/SNotificationList.h"
#include "Async/AsyncWork.h"

FColmapManager::FColmapManager()
    : ColmapThread(nullptr)
    , bIsRunning(false)
    , CurrentProcessHandle()
    , CurrentProgress(0.0f)
    , CurrentStatusMessage(TEXT("Ready"))
{
}

FColmapManager::~FColmapManager()
{
    StopColmapPipeline();
}

bool FColmapManager::StartColmapPipeline(
    const TArray<FCameraInfo>& InCameraInfos,
    const FString& InOutputPath,
    const FString& InColmapExecutablePath)
{
    if (bIsRunning)
    {
        UE_LOG(LogTemp, Warning, TEXT("COLMAP pipeline is already running"));
        return false;
    }
    
    // Store parameters
    CameraInfos = InCameraInfos;
    OutputPath = InOutputPath;
    ColmapExecutablePath = InColmapExecutablePath;
    
    // Reset state
    StopTaskCounter.Reset();
    CurrentProgress = 0.0f;
    CurrentStatusMessage = TEXT("Initializing COLMAP pipeline...");
    
    // Start background thread
    ColmapThread = FRunnableThread::Create(this, TEXT("ColmapPipeline"), 0, TPri_Normal);
    if (ColmapThread)
    {
        bIsRunning = true;
        UE_LOG(LogTemp, Log, TEXT("COLMAP pipeline started in background thread"));
        return true;
    }
    
    UE_LOG(LogTemp, Error, TEXT("Failed to create COLMAP pipeline thread"));
    return false;
}

void FColmapManager::StopColmapPipeline()
{
    if (bIsRunning)
    {
        UE_LOG(LogTemp, Log, TEXT("Stopping COLMAP pipeline..."));
        
        // Set stop flag
        StopTaskCounter.Increment();
        
        // Terminate any running COLMAP process
        TerminateCurrentProcess();
        
        // Don't wait for thread completion to avoid blocking
        // The thread will check StopTaskCounter and exit naturally
        if (ColmapThread)
        {
            // Kill the thread more aggressively if needed
            ColmapThread->Kill(true);
            delete ColmapThread;
            ColmapThread = nullptr;
        }
        
        bIsRunning = false;
        UpdateProgress(0.0f, TEXT("Stopped by user"));
        
        UE_LOG(LogTemp, Log, TEXT("COLMAP pipeline stopped"));
    }
}

bool FColmapManager::Init()
{
    return true;
}

uint32 FColmapManager::Run()
{
    UE_LOG(LogTemp, Log, TEXT("COLMAP pipeline thread started"));
    
    bool bSuccess = false;
    try
    {
        bSuccess = RunColmapPipelineInternal();
    }
    catch (...)
    {
        UE_LOG(LogTemp, Error, TEXT("Exception occurred in COLMAP pipeline thread"));
        UpdateProgress(0.0f, TEXT("Error: Unexpected exception occurred"));
    }
    
    // Notify completion on game thread
    AsyncTask(ENamedThreads::GameThread, [this, bSuccess]()
    {
        FString CompletionMessage = bSuccess ? 
            TEXT("COLMAP pipeline completed successfully!") :
            TEXT("COLMAP pipeline failed. Check logs for details.");
        
        if (OnCompleted.IsBound())
        {
            OnCompleted.ExecuteIfBound(bSuccess, CompletionMessage);
        }
        
        bIsRunning = false;
    });
    
    return 0;
}

void FColmapManager::Stop()
{
    StopTaskCounter.Increment();
}

void FColmapManager::Exit()
{
    // Cleanup if needed
}

void FColmapManager::UpdateProgress(float Progress, const FString& StatusMessage)
{
    {
        FScopeLock Lock(&StatusCriticalSection);
        CurrentProgress = Progress;
        CurrentStatusMessage = StatusMessage;
    }
    
    // Update UI on game thread
    AsyncTask(ENamedThreads::GameThread, [this, Progress, StatusMessage]()
    {
        if (OnProgressUpdated.IsBound())
        {
            OnProgressUpdated.ExecuteIfBound(Progress, StatusMessage);
        }
    });
    
    UE_LOG(LogTemp, Log, TEXT("COLMAP Progress: %.1f%% - %s"), Progress * 100.0f, *StatusMessage);
}

bool FColmapManager::RunColmapPipelineInternal()
{
    if (StopTaskCounter.GetValue() != 0)
        return false;
    
    UpdateProgress(0.05f, TEXT("Creating timestamped directory..."));
    
    // Create timestamped directory
    TimestampedDirectory = FVCCSimDataConverter::CreateTimestampedColmapDirectory(OutputPath);
    if (TimestampedDirectory.IsEmpty())
    {
        UpdateProgress(0.0f, TEXT("Failed to create output directory"));
        return false;
    }
    
    if (StopTaskCounter.GetValue() != 0)
        return false;
    
    UpdateProgress(0.1f, TEXT("Preparing dataset..."));
    
    // Step 1: Prepare dataset
    if (!PrepareDataset())
    {
        UpdateProgress(0.0f, TEXT("Failed to prepare dataset"));
        return false;
    }
    
    if (StopTaskCounter.GetValue() != 0)
        return false;
    
    UpdateProgress(0.25f, TEXT("Extracting features..."));
    
    // Step 2: Feature extraction
    if (!RunFeatureExtraction())
    {
        UpdateProgress(0.0f, TEXT("Feature extraction failed"));
        return false;
    }
    
    if (StopTaskCounter.GetValue() != 0)
        return false;
    
    UpdateProgress(0.6f, TEXT("Matching features..."));
    
    // Step 3: Feature matching
    if (!RunFeatureMatching())
    {
        UpdateProgress(0.0f, TEXT("Feature matching failed"));
        return false;
    }
    
    if (StopTaskCounter.GetValue() != 0)
        return false;
    
    UpdateProgress(0.8f, TEXT("Running sparse reconstruction..."));
    
    // Step 4: Sparse reconstruction
    if (!RunSparseReconstruction())
    {
        UpdateProgress(0.0f, TEXT("Sparse reconstruction failed"));
        return false;
    }
    
    UpdateProgress(1.0f, FString::Printf(TEXT("COLMAP pipeline completed! Results: %s"), *TimestampedDirectory));
    return true;
}

bool FColmapManager::PrepareDataset()
{
    return FVCCSimDataConverter::PrepareColmapDataset(CameraInfos, TimestampedDirectory);
}

bool FColmapManager::RunFeatureExtraction()
{
    FString DatabasePath = FPaths::Combine(TimestampedDirectory, TEXT("database.db"));
    
    // Create command executor lambda that uses our process management
    auto CommandExecutor = [this](const FString& Command, const FString& Arguments, const FString& StepName) -> bool
    {
        return ExecuteColmapCommand(Command, Arguments, StepName);
    };
    
    return FVCCSimDataConverter::RunColmapFeatureExtraction(ColmapExecutablePath, TimestampedDirectory, DatabasePath, CommandExecutor);
}

bool FColmapManager::RunFeatureMatching()
{
    FString DatabasePath = FPaths::Combine(TimestampedDirectory, TEXT("database.db"));
    
    // Create command executor lambda that uses our process management
    auto CommandExecutor = [this](const FString& Command, const FString& Arguments, const FString& StepName) -> bool
    {
        return ExecuteColmapCommand(Command, Arguments, StepName);
    };
    
    return FVCCSimDataConverter::RunColmapFeatureMatching(ColmapExecutablePath, DatabasePath, CommandExecutor);
}

bool FColmapManager::RunSparseReconstruction()
{
    FString DatabasePath = FPaths::Combine(TimestampedDirectory, TEXT("database.db"));
    FString ImagePath = FPaths::Combine(TimestampedDirectory, TEXT("images"));
    
    // Create command executor lambda that uses our process management
    auto CommandExecutor = [this](const FString& Command, const FString& Arguments, const FString& StepName) -> bool
    {
        return ExecuteColmapCommand(Command, Arguments, StepName);
    };
    
    return FVCCSimDataConverter::RunColmapSparseReconstruction(ColmapExecutablePath, DatabasePath, ImagePath, TimestampedDirectory, CommandExecutor);
}

void FColmapManager::TerminateCurrentProcess()
{
    FScopeLock Lock(&ProcessCriticalSection);
    
    if (CurrentProcessHandle.IsValid())
    {
        UE_LOG(LogTemp, Log, TEXT("Terminating COLMAP process..."));
        
        // Check if process is still running
        if (FPlatformProcess::IsProcRunning(CurrentProcessHandle))
        {
            // Try to terminate gracefully first
            FPlatformProcess::TerminateProc(CurrentProcessHandle, false);
            
            // Give it a moment to terminate gracefully
            FPlatformProcess::Sleep(0.5f);
            
            // If still running, force kill
            if (FPlatformProcess::IsProcRunning(CurrentProcessHandle))
            {
                UE_LOG(LogTemp, Warning, TEXT("Graceful termination failed, force killing COLMAP process"));
                FPlatformProcess::TerminateProc(CurrentProcessHandle, true);
            }
        }
        
        // Close the process handle
        FPlatformProcess::CloseProc(CurrentProcessHandle);
        CurrentProcessHandle.Reset();
        
        UE_LOG(LogTemp, Log, TEXT("COLMAP process terminated"));
    }
}

bool FColmapManager::ExecuteColmapCommand(const FString& Command, const FString& Arguments, const FString& StepName)
{
    if (StopTaskCounter.GetValue() != 0)
        return false;
    
    UE_LOG(LogTemp, Log, TEXT("Executing COLMAP step: %s"), *StepName);
    UE_LOG(LogTemp, Log, TEXT("Command: %s %s"), *Command, *Arguments);
    
    // Create the process
    {
        FScopeLock Lock(&ProcessCriticalSection);
        CurrentProcessHandle = FPlatformProcess::CreateProc(
            *Command,
            *Arguments,
            false, // bLaunchDetached
            true,  // bLaunchHidden
            true,  // bLaunchReallyHidden
            nullptr, // OutProcessID
            0,     // PriorityModifier
            nullptr, // OptionalWorkingDirectory
            nullptr  // PipeWriteChild
        );
    }
    
    if (!CurrentProcessHandle.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create COLMAP process for %s"), *StepName);
        return false;
    }
    
    // Wait for process completion while checking for stop requests
    bool bProcessCompleted = false;
    int32 ReturnCode = -1;
    
    while (!bProcessCompleted)
    {
        // Check if we should stop
        if (StopTaskCounter.GetValue() != 0)
        {
            UE_LOG(LogTemp, Log, TEXT("Stop requested during %s"), *StepName);
            TerminateCurrentProcess();
            return false;
        }
        
        // Check if process is still running
        if (FPlatformProcess::IsProcRunning(CurrentProcessHandle))
        {
            // Sleep briefly to avoid busy waiting
            FPlatformProcess::Sleep(0.1f);
        }
        else
        {
            // Process completed
            bProcessCompleted = true;
            FPlatformProcess::GetProcReturnCode(CurrentProcessHandle, &ReturnCode);
        }
    }
    
    // Clean up process handle
    {
        FScopeLock Lock(&ProcessCriticalSection);
        if (CurrentProcessHandle.IsValid())
        {
            FPlatformProcess::CloseProc(CurrentProcessHandle);
            CurrentProcessHandle.Reset();
        }
    }
    
    if (ReturnCode != 0)
    {
        UE_LOG(LogTemp, Error, TEXT("COLMAP %s failed. Return code: %d"), *StepName, ReturnCode);
        return false;
    }
    
    UE_LOG(LogTemp, Log, TEXT("COLMAP %s completed successfully"), *StepName);
    return true;
}