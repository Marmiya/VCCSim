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
#include "DataStruct_IO/CameraData.h"

DECLARE_DELEGATE_TwoParams(FOnColmapProgressUpdated, float, FString);
DECLARE_DELEGATE_TwoParams(FOnColmapCompleted, bool, FString);

/**
 * Asynchronous COLMAP pipeline manager
 * Runs COLMAP operations in background thread to prevent editor blocking
 */
class VCCSIMEDITOR_API FColmapManager : public FRunnable
{
public:
    FColmapManager();
    virtual ~FColmapManager();
    
    /**
     * Start COLMAP pipeline asynchronously
     * @param ImageDirectory Source directory containing images
     * @param OutputPath Base output directory
     * @param ColmapExecutablePath Path to COLMAP executable
     * @return True if successfully started
     */
    bool StartColmapPipeline(
        const FString& ImageDirectory,
        const FString& OutputPath,
        const FString& ColmapExecutablePath = TEXT("D:\\colmap-x64-windows-cuda"));
    
    /**
     * Stop COLMAP pipeline
     */
    void StopColmapPipeline();
    
    /**
     * Check if COLMAP pipeline is running
     */
    bool IsRunning() const { return bIsRunning; }
    
    /**
     * Get current progress (0.0 to 1.0)
     */
    float GetProgress() const { return CurrentProgress; }
    
    /**
     * Get current status message
     */
    FString GetStatusMessage() const { return CurrentStatusMessage; }
    
    /**
     * Get the timestamped directory path created for the current COLMAP run
     */
    FString GetTimestampedDirectory() const { return TimestampedDirectory; }
    
    // Delegates for progress updates
    FOnColmapProgressUpdated OnProgressUpdated;
    FOnColmapCompleted OnCompleted;

    // FRunnable interface
    virtual bool Init() override;
    virtual uint32 Run() override;
    virtual void Stop() override;
    virtual void Exit() override;

private:
    /** Thread for running COLMAP */
    FRunnableThread* ColmapThread;
    
    /** Flag to control thread execution */
    FThreadSafeCounter StopTaskCounter;
    
    /** Current execution state */
    bool bIsRunning;
    
    /** Current COLMAP process handle */
    FProcHandle CurrentProcessHandle;
    
    /** Process management mutex */
    mutable FCriticalSection ProcessCriticalSection;
    
    /** Current progress and status */
    float CurrentProgress;
    FString CurrentStatusMessage;
    
    /** Pipeline parameters */
    FString ImageDirectory;
    FString OutputPath;
    FString ColmapExecutablePath;
    FString TimestampedDirectory;
    
    /** Critical section for thread-safe access */
    mutable FCriticalSection StatusCriticalSection;
    
    /**
     * Update progress and status (thread-safe)
     */
    void UpdateProgress(float Progress, const FString& StatusMessage);
    
    /**
     * Execute COLMAP pipeline steps
     */
    bool RunColmapPipelineInternal();
    
    /**
     * Individual COLMAP steps
     */
    bool PrepareDataset();
    bool RunFeatureExtraction();
    bool RunFeatureMatching();
    bool RunSparseReconstruction();
    bool RunModelConverter();
    
    /**
     * Execute COLMAP command with progress monitoring and process management
     */
    bool ExecuteColmapCommand(const FString& Command, const FString& Arguments, const FString& StepName);
    
    /**
     * Terminate current COLMAP process if running
     */
    void TerminateCurrentProcess();
};