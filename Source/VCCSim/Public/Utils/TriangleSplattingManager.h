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
#include "HAL/PlatformProcess.h"
#include "Misc/DateTime.h"

// Forward declarations
struct FTriangleSplattingConfig;

DECLARE_DELEGATE_TwoParams(FOnTrainingProgressUpdated, float /* Progress */, FString /* StatusMessage */);
DECLARE_DELEGATE_TwoParams(FOnTrainingCompleted, bool /* bSuccessful */, FString /* ResultMessage */);

/**
 * Training status enumeration
 */
UENUM()
enum class ETrainingStatus : uint8
{
    Idle,
    Preparing,
    Running,
    Completed,
    Failed,
    Cancelled
};

/**
 * Triangle Splatting Training Manager
 * Handles Python process execution and monitoring for Triangle Splatting training
 */
class VCCSIM_API FTriangleSplattingManager
{
public:
    // ============================================================================
    // CONSTRUCTION / DESTRUCTION
    // ============================================================================
    
    FTriangleSplattingManager();
    ~FTriangleSplattingManager();

    // ============================================================================
    // TRAINING CONTROL
    // ============================================================================
    
    /**
     * Start training process with given configuration
     * @param Config Training configuration
     * @return True if training started successfully
     */
    bool StartTraining(const FTriangleSplattingConfig& Config);
    
    /**
     * Stop currently running training process
     */
    void StopTraining();
    
    /**
     * Check if training is currently in progress
     * @return True if training is running
     */
    bool IsTrainingInProgress() const { return CurrentStatus == ETrainingStatus::Running; }
    
    /**
     * Get current training status
     * @return Current training status
     */
    ETrainingStatus GetTrainingStatus() const { return CurrentStatus; }
    
    /**
     * Get current training progress (0.0 to 1.0)
     * @return Training progress as float
     */
    float GetTrainingProgress() const { return TrainingProgress; }
    
    /**
     * Get last status message
     * @return Status message string
     */
    FString GetStatusMessage() const { return StatusMessage; }

    // ============================================================================
    // DELEGATES
    // ============================================================================
    
    /** Delegate fired when training progress is updated */
    FOnTrainingProgressUpdated OnTrainingProgressUpdated;
    
    /** Delegate fired when training completes (success or failure) */
    FOnTrainingCompleted OnTrainingCompleted;

    // ============================================================================
    // MONITORING
    // ============================================================================
    
    /**
     * Update training status by checking process and log files
     * Should be called regularly during training
     */
    void UpdateTrainingStatus();
    
    /**
     * Force refresh of training status
     */
    void RefreshStatus();

private:
    // ============================================================================
    // INTERNAL STATE
    // ============================================================================
    
    // Current training configuration (using pointer to avoid circular dependency)
    TSharedPtr<FTriangleSplattingConfig> CurrentConfig;
    
    // Process management
    FProcHandle TrainingProcessHandle;
    TSharedPtr<class FMonitoredProcess> ProcessMonitor;
    
    // Status tracking
    ETrainingStatus CurrentStatus;
    float TrainingProgress;
    FString StatusMessage;
    FString LastError;
    
    // File paths
    FString ConfigFilePath;
    FString LogFilePath;
    FString OutputDirectory;
    
    // Timing
    FDateTime TrainingStartTime;
    FDateTime LastUpdateTime;
    
    // Update frequency (in seconds)
    static constexpr float StatusUpdateInterval = 1.0f;

    // ============================================================================
    // INTERNAL METHODS
    // ============================================================================
    
    /**
     * Validate training configuration before starting
     * @param Config Configuration to validate
     * @return True if configuration is valid
     */
    bool ValidateConfiguration(const FTriangleSplattingConfig& Config);
    
    /**
     * Prepare data and create configuration files for training
     * @param Config Training configuration
     * @return True if preparation succeeded
     */
    bool PrepareTrainingData(const FTriangleSplattingConfig& Config);
    
    /**
     * Create JSON configuration file for Python training script
     * @param Config Training configuration
     * @return Path to created config file, or empty string if failed
     */
    FString CreateConfigurationFile(const FTriangleSplattingConfig& Config);
    
    /**
     * Launch Python training process
     * @return True if process launched successfully
     */
    bool LaunchTrainingProcess();
    
    /**
     * Check if required Python environment is available
     * @return True if Python environment is valid
     */
    bool CheckPythonEnvironment();
    
    /**
     * Get path to Python executable
     * @return Path to Python executable
     */
    FString GetPythonExecutablePath();
    
    /**
     * Get path to Triangle Splatting training script
     * @return Path to training script
     */
    FString GetTrainingScriptPath();
    
    /**
     * Parse training log file to extract progress information
     * @return Parsed progress (0.0 to 1.0)
     */
    float ParseTrainingProgress();
    
    /**
     * Parse training log file to extract current loss value
     * @return Current loss value as string
     */
    FString ParseCurrentLoss();
    
    /**
     * Read latest entries from training log file
     * @param MaxLines Maximum number of lines to read from end of file
     * @return Array of log lines
     */
    TArray<FString> ReadRecentLogLines(int32 MaxLines = 50);
    
    /**
     * Check if training process is still running
     * @return True if process is active
     */
    bool IsProcessRunning();
    
    /**
     * Get process exit code if process has terminated
     * @param OutExitCode Output parameter for exit code
     * @return True if exit code was retrieved
     */
    bool GetProcessExitCode(int32& OutExitCode);
    
    /**
     * Handle training completion (success or failure)
     * @param bSuccessful Whether training completed successfully
     * @param ResultMessage Result message
     */
    void HandleTrainingCompletion(bool bSuccessful, const FString& ResultMessage);
    
    /**
     * Clean up resources after training completion
     */
    void CleanupTraining();
    
    /**
     * Update status message and notify delegates
     * @param NewMessage New status message
     */
    void UpdateStatusMessage(const FString& NewMessage);
    
    /**
     * Set training progress and notify delegates
     * @param NewProgress New progress value (0.0 to 1.0)
     */
    void UpdateProgress(float NewProgress);

    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================
    
    /**
     * Ensure output directory exists and is writable
     * @param DirectoryPath Path to directory
     * @return True if directory is ready
     */
    bool EnsureDirectoryExists(const FString& DirectoryPath);
    
    /**
     * Generate unique timestamp-based filename
     * @param BaseName Base filename without extension
     * @param Extension File extension
     * @return Generated filename with timestamp
     */
    FString GenerateTimestampedFilename(const FString& BaseName, const FString& Extension);
    
    /**
     * Convert configuration to JSON string
     * @param Config Configuration to convert
     * @return JSON string representation
     */
    FString ConfigToJsonString(const FTriangleSplattingConfig& Config);
    
    /**
     * Log message to both UE log and training log file
     * @param Message Message to log
     * @param bIsError Whether this is an error message
     */
    void LogMessage(const FString& Message, bool bIsError = false);
};