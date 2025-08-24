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

#include "Utils/TriangleSplattingManager.h"
#include "Editor/VCCSimPanel.h"
#include "Editor/VCCSimDataStructures.h"
#include "Utils/VCCSimDataConverter.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/PlatformFilemanager.h"
#include "Misc/DateTime.h"

// ============================================================================
// CONSTRUCTION / DESTRUCTION
// ============================================================================

FTriangleSplattingManager::FTriangleSplattingManager()
    : CurrentStatus(ETrainingStatus::Idle)
    , TrainingProgress(0.0f)
    , StatusMessage(TEXT("Ready"))
    , LastError(TEXT(""))
    , ConfigFilePath(TEXT(""))
    , LogFilePath(TEXT(""))
    , OutputDirectory(TEXT(""))
    , TrainingStartTime(FDateTime::MinValue())
    , LastUpdateTime(FDateTime::MinValue())
{
}

FTriangleSplattingManager::~FTriangleSplattingManager()
{
    StopTraining();
}

// ============================================================================
// TRAINING CONTROL
// ============================================================================

bool FTriangleSplattingManager::StartTraining(const FTriangleSplattingConfiguration& Config)
{
    if (IsTrainingInProgress())
    {
        LogMessage(TEXT("Training is already in progress"), true);
        return false;
    }
    
    CurrentConfig = MakeShared<FTriangleSplattingConfiguration>(Config);
    CurrentStatus = ETrainingStatus::Preparing;
    TrainingProgress = 0.0f;
    StatusMessage = TEXT("Preparing training...");
    TrainingStartTime = FDateTime::Now();
    
    // Validate configuration
    if (!ValidateConfiguration(Config))
    {
        CurrentStatus = ETrainingStatus::Failed;
        return false;
    }
    
    // Check Python environment
    if (!CheckPythonEnvironment())
    {
        CurrentStatus = ETrainingStatus::Failed;
        StatusMessage = TEXT("Python environment not found or invalid");
        LogMessage(StatusMessage, true);
        return false;
    }
    
    // Prepare training data
    if (!PrepareTrainingData(Config))
    {
        CurrentStatus = ETrainingStatus::Failed;
        StatusMessage = TEXT("Failed to prepare training data");
        LogMessage(StatusMessage, true);
        return false;
    }
    
    // Launch training process
    if (!LaunchTrainingProcess())
    {
        CurrentStatus = ETrainingStatus::Failed;
        StatusMessage = TEXT("Failed to launch training process");
        LogMessage(StatusMessage, true);
        return false;
    }
    
    CurrentStatus = ETrainingStatus::Running;
    StatusMessage = TEXT("Training started successfully");
    LogMessage(StatusMessage);
    
    return true;
}

void FTriangleSplattingManager::StopTraining()
{
    if (IsTrainingInProgress())
    {
        LogMessage(TEXT("Stopping training process..."));
        
        if (TrainingProcessHandle.IsValid())
        {
            FPlatformProcess::TerminateProc(TrainingProcessHandle);
            FPlatformProcess::CloseProc(TrainingProcessHandle);
            TrainingProcessHandle = FProcHandle();
        }
        
        CurrentStatus = ETrainingStatus::Cancelled;
        StatusMessage = TEXT("Training cancelled by user");
        TrainingProgress = 0.0f;
        
        CleanupTraining();
        
        OnTrainingCompleted.ExecuteIfBound(false, TEXT("Training was cancelled"));
    }
}

// ============================================================================
// MONITORING
// ============================================================================

void FTriangleSplattingManager::UpdateTrainingStatus()
{
    if (!IsTrainingInProgress())
    {
        return;
    }
    
    FDateTime CurrentTime = FDateTime::Now();
    
    // Limit update frequency
    if ((CurrentTime - LastUpdateTime).GetTotalSeconds() < StatusUpdateInterval)
    {
        return;
    }
    
    LastUpdateTime = CurrentTime;
    
    // Check if process is still running
    if (!IsProcessRunning())
    {
        int32 ExitCode;
        bool bGotExitCode = GetProcessExitCode(ExitCode);
        
        if (bGotExitCode && ExitCode == 0)
        {
            // Training completed successfully
            TrainingProgress = 1.0f;
            HandleTrainingCompletion(true, TEXT("Training completed successfully"));
        }
        else
        {
            // Training failed
            FString ErrorMessage = FString::Printf(TEXT("Training failed with exit code: %d"), ExitCode);
            HandleTrainingCompletion(false, ErrorMessage);
        }
        
        return;
    }
    
    // Update progress from log file
    float NewProgress = ParseTrainingProgress();
    if (NewProgress > TrainingProgress)
    {
        UpdateProgress(NewProgress);
    }
    
    // Update status message with current loss
    FString CurrentLoss = ParseCurrentLoss();
    if (!CurrentLoss.IsEmpty())
    {
        FString NewStatusMessage = FString::Printf(TEXT("Training... Loss: %s"), *CurrentLoss);
        UpdateStatusMessage(NewStatusMessage);
    }
}

void FTriangleSplattingManager::RefreshStatus()
{
    LastUpdateTime = FDateTime::MinValue(); // Force update on next call
    UpdateTrainingStatus();
}

// ============================================================================
// INTERNAL VALIDATION AND PREPARATION
// ============================================================================

bool FTriangleSplattingManager::ValidateConfiguration(const FTriangleSplattingConfiguration& Config)
{
    TArray<FString> ErrorMessages;
    
    // Check required paths
    if (Config.ImageDirectory.IsEmpty() || !FPaths::DirectoryExists(Config.ImageDirectory))
    {
        ErrorMessages.Add(TEXT("Image directory is invalid or does not exist"));
    }
    
    if (Config.PoseFilePath.IsEmpty() || !FPaths::FileExists(Config.PoseFilePath))
    {
        ErrorMessages.Add(TEXT("Pose file is invalid or does not exist"));
    }
    
    if (Config.OutputDirectory.IsEmpty())
    {
        ErrorMessages.Add(TEXT("Output directory must be specified"));
    }
    
    // Check camera parameters
    if (Config.FOVDegrees <= 0 || Config.FOVDegrees >= 180)
    {
        ErrorMessages.Add(TEXT("FOV must be between 0 and 180 degrees"));
    }
    
    if (Config.ImageWidth <= 0 || Config.ImageHeight <= 0)
    {
        ErrorMessages.Add(TEXT("Image dimensions must be positive"));
    }
    
    // Check training parameters
    if (Config.MaxIterations <= 0)
    {
        ErrorMessages.Add(TEXT("Max iterations must be positive"));
    }
    
    if (Config.LearningRate <= 0)
    {
        ErrorMessages.Add(TEXT("Learning rate must be positive"));
    }
    
    if (ErrorMessages.Num() > 0)
    {
        LastError = FString::Join(ErrorMessages, TEXT("; "));
        LogMessage(FString::Printf(TEXT("Configuration validation failed: %s"), *LastError), true);
        return false;
    }
    
    return true;
}

bool FTriangleSplattingManager::PrepareTrainingData(const FTriangleSplattingConfiguration& Config)
{
    OutputDirectory = Config.OutputDirectory;
    
    // Ensure output directory exists
    if (!EnsureDirectoryExists(OutputDirectory))
    {
        return false;
    }
    
    // Create subdirectories
    TArray<FString> SubDirectories = { TEXT("logs"), TEXT("output"), TEXT("config") };
    for (const FString& SubDir : SubDirectories)
    {
        FString SubDirPath = FPaths::Combine(OutputDirectory, SubDir);
        if (!EnsureDirectoryExists(SubDirPath))
        {
            return false;
        }
    }
    
    // Create configuration file
    ConfigFilePath = CreateConfigurationFile(Config);
    if (ConfigFilePath.IsEmpty())
    {
        LogMessage(TEXT("Failed to create configuration file"), true);
        return false;
    }
    
    // Set up log file path
    LogFilePath = FPaths::Combine(OutputDirectory, TEXT("logs"), 
        GenerateTimestampedFilename(TEXT("training"), TEXT("log")));
    
    // Convert pose data and camera information
    FCameraIntrinsics Intrinsics = FVCCSimDataConverter::ConvertCameraParams(
        Config.FOVDegrees, Config.ImageWidth, Config.ImageHeight);
    
    TArray<FCameraInfo> CameraInfos = FVCCSimDataConverter::ConvertPoseFile(
        Config.PoseFilePath, Config.ImageDirectory, Intrinsics);
    
    if (CameraInfos.Num() == 0)
    {
        LogMessage(TEXT("No valid camera poses found in pose file"), true);
        return false;
    }
    
    // Save camera information
    FString CameraInfoPath = FPaths::Combine(OutputDirectory, TEXT("config"));
    if (!FVCCSimDataConverter::SaveCameraInfo(CameraInfos, CameraInfoPath))
    {
        LogMessage(TEXT("Failed to save camera information"), true);
        return false;
    }
    
    // If mesh is provided, convert to point cloud for initialization
    if (Config.SelectedMesh.IsValid())
    {
        FPointCloudData PointCloud = FVCCSimDataConverter::ConvertMeshToPointCloud(
            Config.SelectedMesh.Get(), 10000, true);
        
        if (PointCloud.GetPointCount() > 0)
        {
            FString PointCloudPath = FPaths::Combine(OutputDirectory, TEXT("config"), TEXT("init_points.ply"));
            FVCCSimDataConverter::SavePointCloudToPLY(PointCloud, PointCloudPath);
        }
    }
    
    LogMessage(FString::Printf(TEXT("Training data prepared with %d camera poses"), CameraInfos.Num()));
    
    return true;
}

FString FTriangleSplattingManager::CreateConfigurationFile(const FTriangleSplattingConfiguration& Config)
{
    FString ConfigPath = FPaths::Combine(OutputDirectory, TEXT("config"), TEXT("vccsim_training_config.json"));
    
    FString ConfigContent = ConfigToJsonString(Config);
    
    if (FFileHelper::SaveStringToFile(ConfigContent, *ConfigPath))
    {
        return ConfigPath;
    }
    
    return FString();
}

// ============================================================================
// PYTHON PROCESS MANAGEMENT
// ============================================================================

bool FTriangleSplattingManager::LaunchTrainingProcess()
{
    FString PythonPath = GetPythonExecutablePath();
    FString ScriptPath = GetTrainingScriptPath();
    
    if (PythonPath.IsEmpty() || ScriptPath.IsEmpty())
    {
        return false;
    }
    
    // Build command line arguments
    FString Arguments = FString::Printf(TEXT("\"%s\" --config \"%s\" --output \"%s\" --log \"%s\""),
        *ScriptPath, *ConfigFilePath, *OutputDirectory, *LogFilePath);
    
    LogMessage(FString::Printf(TEXT("Launching Python process: %s %s"), *PythonPath, *Arguments));
    
    // Launch process
    TrainingProcessHandle = FPlatformProcess::CreateProc(
        *PythonPath,
        *Arguments,
        true,  // bLaunchDetached
        true,  // bLaunchHidden
        true,  // bLaunchReallyHidden
        nullptr, // OutProcessID
        0,     // PriorityModifier
        nullptr, // OptionalWorkingDirectory
        nullptr  // PipeWriteChild
    );
    
    if (!TrainingProcessHandle.IsValid())
    {
        LogMessage(TEXT("Failed to launch training process"), true);
        return false;
    }
    
    LogMessage(TEXT("Training process launched successfully"));
    return true;
}

bool FTriangleSplattingManager::CheckPythonEnvironment()
{
    FString PythonPath = GetPythonExecutablePath();
    
    if (PythonPath.IsEmpty())
    {
        return false;
    }
    
    // Test Python by running a simple command
    FString TestCommand = TEXT("--version");
    
    int32 ReturnCode;
    FString StdOut;
    FString StdErr;
    
    bool bSuccess = FPlatformProcess::ExecProcess(
        *PythonPath,
        *TestCommand,
        &ReturnCode,
        &StdOut,
        &StdErr
    );
    
    if (!bSuccess || ReturnCode != 0)
    {
        LogMessage(FString::Printf(TEXT("Python test failed: %s"), *StdErr), true);
        return false;
    }
    
    LogMessage(FString::Printf(TEXT("Python environment validated: %s"), *StdOut.TrimStartAndEnd()));
    return true;
}

FString FTriangleSplattingManager::GetPythonExecutablePath()
{
    // Try common Python executable names and locations
    TArray<FString> PossiblePaths = {
        TEXT("python"),
        TEXT("python3"),
        TEXT("C:/Python39/python.exe"),
        TEXT("C:/Python310/python.exe"),
        TEXT("C:/Python311/python.exe"),
        TEXT("C:/Users/%USERNAME%/AppData/Local/Programs/Python/Python39/python.exe"),
        TEXT("C:/Users/%USERNAME%/AppData/Local/Programs/Python/Python310/python.exe"),
        TEXT("C:/Users/%USERNAME%/AppData/Local/Programs/Python/Python311/python.exe")
    };
    
    for (const FString& Path : PossiblePaths)
    {
        FString ExpandedPath = Path;
        
        // Expand environment variables if needed
        if (ExpandedPath.Contains(TEXT("%USERNAME%")))
        {
            FString Username = FPlatformMisc::GetEnvironmentVariable(TEXT("USERNAME"));
            ExpandedPath = ExpandedPath.Replace(TEXT("%USERNAME%"), *Username);
        }
        
        // Check if executable exists
        if (FPaths::FileExists(ExpandedPath))
        {
            return ExpandedPath;
        }
        
        // For system PATH executables, try to locate them
        FString FoundPath = FPlatformProcess::GetApplicationName(0); // This doesn't work for finding python
        // Alternative approach would be to execute "where python" on Windows
    }
    
    LogMessage(TEXT("Python executable not found. Please ensure Python is installed and in PATH."), true);
    return FString();
}

FString FTriangleSplattingManager::GetTrainingScriptPath()
{
    // Path to the VCCSim-specific Triangle Splatting training script
    FString PluginDir = FPaths::ProjectPluginsDir() / TEXT("VCCSim");
    
    // Try improved script first
    FString ImprovedScriptPath = PluginDir / TEXT("Source/VCCSim/triangle-splatting/train_vccsim_improved.py");
    if (FPaths::FileExists(ImprovedScriptPath))
    {
        return ImprovedScriptPath;
    }
    
    // Try original VCCSim script
    FString ScriptPath = PluginDir / TEXT("Source/VCCSim/triangle-splatting/train_vccsim.py");
    if (FPaths::FileExists(ScriptPath))
    {
        return ScriptPath;
    }
    
    // Try game engine script
    FString AlternativeScriptPath = PluginDir / TEXT("Source/VCCSim/triangle-splatting/train_game_engine.py");
    if (FPaths::FileExists(AlternativeScriptPath))
    {
        return AlternativeScriptPath;
    }
    
    LogMessage(TEXT("Training script not found. Please ensure Triangle Splatting scripts are available."), true);
    return FString();
}

// ============================================================================
// LOG PARSING AND MONITORING
// ============================================================================

float FTriangleSplattingManager::ParseTrainingProgress()
{
    if (LogFilePath.IsEmpty() || !FPaths::FileExists(LogFilePath))
    {
        return 0.0f;
    }
    
    TArray<FString> RecentLines = ReadRecentLogLines(10);
    
    // Look for progress indicators in log lines
    for (const FString& Line : RecentLines)
    {
        // Look for patterns like "Iteration 1000/5000" or "Progress: 20%"
        if (Line.Contains(TEXT("Iteration")))
        {
            // Extract current iteration and max iterations
            FString IterationPattern = TEXT("Iteration ");
            int32 IterationIndex = Line.Find(IterationPattern);
            if (IterationIndex != INDEX_NONE)
            {
                FString AfterIteration = Line.Mid(IterationIndex + IterationPattern.Len());
                TArray<FString> Parts;
                AfterIteration.ParseIntoArray(Parts, TEXT("/"));
                
                if (Parts.Num() >= 2)
                {
                    int32 CurrentIter = FCString::Atoi(*Parts[0]);
                    int32 MaxIter = FCString::Atoi(*Parts[1]);
                    
                    if (MaxIter > 0)
                    {
                        return FMath::Clamp(static_cast<float>(CurrentIter) / static_cast<float>(MaxIter), 0.0f, 1.0f);
                    }
                }
            }
        }
        else if (Line.Contains(TEXT("Progress:")))
        {
            // Extract percentage
            FString ProgressPattern = TEXT("Progress: ");
            int32 ProgressIndex = Line.Find(ProgressPattern);
            if (ProgressIndex != INDEX_NONE)
            {
                FString AfterProgress = Line.Mid(ProgressIndex + ProgressPattern.Len());
                int32 PercentIndex = AfterProgress.Find(TEXT("%"));
                if (PercentIndex != INDEX_NONE)
                {
                    FString PercentStr = AfterProgress.Left(PercentIndex);
                    float Percent = FCString::Atof(*PercentStr);
                    return FMath::Clamp(Percent / 100.0f, 0.0f, 1.0f);
                }
            }
        }
    }
    
    return TrainingProgress; // Return current progress if no new progress found
}

FString FTriangleSplattingManager::ParseCurrentLoss()
{
    if (LogFilePath.IsEmpty() || !FPaths::FileExists(LogFilePath))
    {
        return FString();
    }
    
    TArray<FString> RecentLines = ReadRecentLogLines(5);
    
    // Look for loss values in recent log lines
    for (const FString& Line : RecentLines)
    {
        if (Line.Contains(TEXT("Loss:")) || Line.Contains(TEXT("loss:")))
        {
            // Extract loss value
            TArray<FString> LossPatterns = { TEXT("Loss: "), TEXT("loss: "), TEXT("Loss="), TEXT("loss=") };
            
            for (const FString& Pattern : LossPatterns)
            {
                int32 LossIndex = Line.Find(Pattern, ESearchCase::IgnoreCase);
                if (LossIndex != INDEX_NONE)
                {
                    FString AfterLoss = Line.Mid(LossIndex + Pattern.Len());
                    
                    // Extract numeric value (could be scientific notation)
                    int32 SpaceIndex = AfterLoss.Find(TEXT(" "));
                    if (SpaceIndex != INDEX_NONE)
                    {
                        AfterLoss = AfterLoss.Left(SpaceIndex);
                    }
                    
                    // Validate that it's a number
                    if (AfterLoss.IsNumeric() || AfterLoss.Contains(TEXT("e")) || AfterLoss.Contains(TEXT("E")))
                    {
                        return AfterLoss.TrimStartAndEnd();
                    }
                }
            }
        }
    }
    
    return FString();
}

TArray<FString> FTriangleSplattingManager::ReadRecentLogLines(int32 MaxLines)
{
    TArray<FString> Lines;
    
    if (LogFilePath.IsEmpty() || !FPaths::FileExists(LogFilePath))
    {
        return Lines;
    }
    
    TArray<FString> AllLines;
    if (FFileHelper::LoadFileToStringArray(AllLines, *LogFilePath))
    {
        // Get the last MaxLines lines
        int32 StartIndex = FMath::Max(0, AllLines.Num() - MaxLines);
        for (int32 i = StartIndex; i < AllLines.Num(); ++i)
        {
            Lines.Add(AllLines[i]);
        }
    }
    
    return Lines;
}

// ============================================================================
// PROCESS STATUS CHECKING
// ============================================================================

bool FTriangleSplattingManager::IsProcessRunning()
{
    if (!TrainingProcessHandle.IsValid())
    {
        return false;
    }
    
    return FPlatformProcess::IsProcRunning(TrainingProcessHandle);
}

bool FTriangleSplattingManager::GetProcessExitCode(int32& OutExitCode)
{
    if (!TrainingProcessHandle.IsValid())
    {
        return false;
    }
    
    return FPlatformProcess::GetProcReturnCode(TrainingProcessHandle, &OutExitCode);
}

void FTriangleSplattingManager::HandleTrainingCompletion(bool bSuccessful, const FString& ResultMessage)
{
    if (bSuccessful)
    {
        CurrentStatus = ETrainingStatus::Completed;
    }
    else
    {
        CurrentStatus = ETrainingStatus::Failed;
    }
    
    UpdateStatusMessage(ResultMessage);
    LogMessage(ResultMessage, !bSuccessful);
    
    CleanupTraining();
    
    OnTrainingCompleted.ExecuteIfBound(bSuccessful, ResultMessage);
}

void FTriangleSplattingManager::CleanupTraining()
{
    if (TrainingProcessHandle.IsValid())
    {
        FPlatformProcess::CloseProc(TrainingProcessHandle);
        TrainingProcessHandle = FProcHandle();
    }
    
    ProcessMonitor.Reset();
}

// ============================================================================
// STATUS UPDATES
// ============================================================================

void FTriangleSplattingManager::UpdateStatusMessage(const FString& NewMessage)
{
    if (StatusMessage != NewMessage)
    {
        StatusMessage = NewMessage;
        OnTrainingProgressUpdated.ExecuteIfBound(TrainingProgress, StatusMessage);
    }
}

void FTriangleSplattingManager::UpdateProgress(float NewProgress)
{
    if (FMath::Abs(TrainingProgress - NewProgress) > 0.01f) // Only update if significant change
    {
        TrainingProgress = FMath::Clamp(NewProgress, 0.0f, 1.0f);
        OnTrainingProgressUpdated.ExecuteIfBound(TrainingProgress, StatusMessage);
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

bool FTriangleSplattingManager::EnsureDirectoryExists(const FString& DirectoryPath)
{
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    
    if (!PlatformFile.DirectoryExists(*DirectoryPath))
    {
        if (!PlatformFile.CreateDirectoryTree(*DirectoryPath))
        {
            LogMessage(FString::Printf(TEXT("Failed to create directory: %s"), *DirectoryPath), true);
            return false;
        }
    }
    
    return true;
}

FString FTriangleSplattingManager::GenerateTimestampedFilename(const FString& BaseName, const FString& Extension)
{
    FString Timestamp = FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S"));
    return FString::Printf(TEXT("%s_%s.%s"), *BaseName, *Timestamp, *Extension);
}

FString FTriangleSplattingManager::ConfigToJsonString(const FTriangleSplattingConfiguration& Config)
{
    return FString::Printf(TEXT(
        "{\n"
        "  \"image_directory\": \"%s\",\n"
        "  \"pose_file\": \"%s\",\n"
        "  \"output_directory\": \"%s\",\n"
        "  \"camera\": {\n"
        "    \"fov_degrees\": %.2f,\n"
        "    \"width\": %d,\n"
        "    \"height\": %d\n"
        "  },\n"
        "  \"training\": {\n"
        "    \"max_iterations\": %d,\n"
        "    \"learning_rate\": %.6f\n"
        "  },\n"
        "  \"mesh\": {\n"
        "    \"use_mesh_initialization\": %s,\n"
        "    \"mesh_path\": \"%s\"\n"
        "  }\n"
        "}\n"
    ),
        *Config.ImageDirectory.Replace(TEXT("\\"), TEXT("/")),
        *Config.PoseFilePath.Replace(TEXT("\\"), TEXT("/")),
        *Config.OutputDirectory.Replace(TEXT("\\"), TEXT("/")),
        Config.FOVDegrees,
        Config.ImageWidth,
        Config.ImageHeight,
        Config.MaxIterations,
        Config.LearningRate,
        Config.bUseMeshInitialization ? TEXT("true") : TEXT("false"),
        Config.SelectedMesh.IsValid() ? *Config.SelectedMesh->GetPathName() : TEXT("")
    );
}

void FTriangleSplattingManager::LogMessage(const FString& Message, bool bIsError)
{
    // Log to UE
    if (bIsError)
    {
        UE_LOG(LogTemp, Error, TEXT("TriangleSplattingManager: %s"), *Message);
    }
    else
    {
        UE_LOG(LogTemp, Log, TEXT("TriangleSplattingManager: %s"), *Message);
    }
    
    // Also log to training log file if available
    if (!LogFilePath.IsEmpty())
    {
        FString TimestampedMessage = FString::Printf(TEXT("[%s] %s\n"), 
            *FDateTime::Now().ToString(TEXT("%Y-%m-%d %H:%M:%S")), 
            *Message);
        
        FFileHelper::SaveStringToFile(TimestampedMessage, *LogFilePath, 
            FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), FILEWRITE_Append);
    }
}