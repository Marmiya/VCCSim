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
#include "Utils/VCCSimDataConverter.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/PlatformFilemanager.h"
#include "Misc/DateTime.h"
#include "Engine/Engine.h"

// ============================================================================
// ASYNC TRIANGLE EXTRACTION TASK
// ============================================================================

/**
 * Async task for extracting mesh triangles without blocking the UI thread
 */
class FTriangleExtractionTask : public FNonAbandonableTask
{
public:
    FTriangleExtractionTask(const FTriangleSplattingConfig& InConfig, const FString& InOutputPath, FTriangleSplattingManager* InManager)
        : Config(InConfig)
        , OutputPath(InOutputPath)
        , Manager(InManager)
        , bSucceeded(false)
    {
    }

    void DoWork()
    {
        if (!Config.SelectedMesh.IsValid())
        {
            ErrorMessage = TEXT("Selected mesh is not valid");
            bSucceeded = false;
            return;
        }

        try
        {
            // Extract mesh triangles
            FVCCSimDataConverter::FMeshTriangleData TriangleData =
                FVCCSimDataConverter::ExtractMeshTriangles(
                Config.SelectedMesh.Get(),
                Config.MaxMeshTriangles,
                Config.MeshTriangleMethod,
                true // Apply coordinate transformation
            );
            
            // Save to PLY file
            if (FVCCSimDataConverter::SaveMeshTrianglesToPLY(TriangleData, OutputPath))
            {
                bSucceeded = true;
                ResultMessage = FString::Printf(TEXT("Successfully exported %d triangles to %s"), TriangleData.TriangleCount, *OutputPath);
            }
            else
            {
                bSucceeded = false;
                ErrorMessage = TEXT("Failed to save triangles to PLY file");
            }
        }
        catch (const std::exception& e)
        {
            bSucceeded = false;
            ErrorMessage = FString::Printf(TEXT("Triangle extraction failed: %s"), UTF8_TO_TCHAR(e.what()));
        }
        catch (...)
        {
            bSucceeded = false;
            ErrorMessage = TEXT("Unknown error during triangle extraction");
        }
    }

    FORCEINLINE TStatId GetStatId() const
    {
        RETURN_QUICK_DECLARE_CYCLE_STAT(FTriangleExtractionTask, STATGROUP_ThreadPoolAsyncTasks);
    }

    bool WasSuccessful() const { return bSucceeded; }
    FString GetErrorMessage() const { return ErrorMessage; }
    FString GetResultMessage() const { return ResultMessage; }

private:
    FTriangleSplattingConfig Config;
    FString OutputPath;
    FTriangleSplattingManager* Manager;
    
    bool bSucceeded;
    FString ErrorMessage;
    FString ResultMessage;
};

// Type alias for the async task
typedef TSharedPtr<FAsyncTask<FTriangleExtractionTask>> FTriangleExtractionTaskPtr;

// ============================================================================
// CONSTRUCTION / DESTRUCTION
// ============================================================================

FTriangleSplattingManager::FTriangleSplattingManager()
    : ReadPipe(nullptr)
    , WritePipe(nullptr)
    , CurrentStatus(ETrainingStatus::Idle)
    , TrainingProgress(0.0f)
    , StatusMessage(TEXT("Ready"))
    , LastError(TEXT(""))
    , ConfigFilePath(TEXT(""))
    , LogFilePath(TEXT(""))
    , PythonLogFilePath(TEXT(""))
    , OutputDirectory(TEXT(""))
    , TrainingStartTime(FDateTime::MinValue())
    , LastUpdateTime(FDateTime::MinValue())
    , TriangleExtractionTask(nullptr)
    , bTriangleExtractionInProgress(false)
{
}

FTriangleSplattingManager::~FTriangleSplattingManager()
{
    StopTraining();
    
    // Clean up async task if still running
    if (TriangleExtractionTask)
    {
        FTriangleExtractionTaskPtr* TaskPtr = static_cast<FTriangleExtractionTaskPtr*>(TriangleExtractionTask);
        delete TaskPtr;
        TriangleExtractionTask = nullptr;
    }
}

// ============================================================================
// TRAINING CONTROL
// ============================================================================

bool FTriangleSplattingManager::StartTraining(const FTriangleSplattingConfig& Config)
{
    if (IsTrainingInProgress())
    {
        LogMessage(TEXT("Training is already in progress"), true);
        return false;
    }
    
    CurrentConfig = MakeShared<FTriangleSplattingConfig>(Config);
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
    
    // Prepare training data (directories, config files, pose conversion)
    if (!PrepareTrainingData(Config))
    {
        CurrentStatus = ETrainingStatus::Failed;
        StatusMessage = TEXT("Failed to prepare training data");
        LogMessage(StatusMessage, true);
        return false;
    }
    
    // If using mesh triangles, start async extraction, otherwise launch training directly
    if (Config.bUseMeshTriangles && Config.SelectedMesh.IsValid())
    {
        // Start async triangle extraction - training will launch when extraction completes
        StartAsyncTriangleExtraction(Config);
        return true; // Return success, training will start asynchronously
    }
    
    // Launch training process directly (non-mesh case)
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

bool FTriangleSplattingManager::StartColmapTraining(const FString& PythonCommand, const FString& Arguments, const FString& OutputDir)
{
    if (IsTrainingInProgress())
    {
        LogMessage(TEXT("Training is already in progress"), true);
        return false;
    }
    
    CurrentStatus = ETrainingStatus::Preparing;
    TrainingProgress = 0.0f;
    StatusMessage = TEXT("Starting Triangle Splatting training with COLMAP data...");
    TrainingStartTime = FDateTime::Now();
    
    // Create organized session directory for COLMAP training
    // OutputDir is already: ProjectSavedDir()/TriangleSplatting
    FString TSTrainingDir = FPaths::Combine(OutputDir, TEXT("RatSplatting"));
    
    FDateTime Now = FDateTime::Now();
    FString Timestamp = Now.ToString(TEXT("%Y%m%d_%H%M%S"));
    FString SessionDirName = FString::Printf(TEXT("colmap_session_%s"), *Timestamp);
    OutputDirectory = FPaths::Combine(TSTrainingDir, SessionDirName);
    
    // Ensure session directory structure exists (simplified)
    if (!EnsureDirectoryExists(OutputDirectory))
    {
        CurrentStatus = ETrainingStatus::Failed;
        StatusMessage = TEXT("Failed to create training session directory");
        LogMessage(StatusMessage, true);
        return false;
    }
    
    // Set up log file path directly in session directory
    LogFilePath = FPaths::Combine(OutputDirectory, TEXT("training.log"));
    PythonLogFilePath = FPaths::Combine(OutputDirectory, TEXT("python_training.log"));
    
    UE_LOG(LogTemp, Log, TEXT("Starting COLMAP training: %s %s"), *PythonCommand, *Arguments);
    
    // Clear previous output buffer
    PythonOutputBuffer.Empty();
    
    // Create pipes for capturing Python output
    FPlatformProcess::CreatePipe(ReadPipe, WritePipe);
    
    // Create the training process with output redirection
    TrainingProcessHandle = FPlatformProcess::CreateProc(
        *PythonCommand,
        *Arguments,
        false, // bLaunchDetached
        true,  // bLaunchHidden
        true,  // bLaunchReallyHidden
        nullptr, // OutProcessID
        0,     // PriorityModifier
        nullptr, // OptionalWorkingDirectory
        WritePipe  // PipeWriteChild - redirect stdout to our pipe
    );
    
    if (!TrainingProcessHandle.IsValid())
    {
        CurrentStatus = ETrainingStatus::Failed;
        StatusMessage = TEXT("Failed to launch Triangle Splatting training process");
        LogMessage(StatusMessage, true);
        return false;
    }
    
    CurrentStatus = ETrainingStatus::Running;
    StatusMessage = TEXT("Triangle Splatting training with COLMAP data started successfully");
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
    // Check for triangle extraction completion first
    if (bTriangleExtractionInProgress)
    {
        OnTriangleExtractionComplete();
        return;
    }
    
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
    
    // Read Python process output from pipe
    if (ReadPipe)
    {
        FString PipeOutput = FPlatformProcess::ReadPipe(ReadPipe);
        if (!PipeOutput.IsEmpty())
        {
            // Append to Python output buffer
            PythonOutputBuffer += PipeOutput;
            
            // Also write to log file
            if (!LogFilePath.IsEmpty())
            {
                FFileHelper::SaveStringToFile(PipeOutput, *LogFilePath, 
                    FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), FILEWRITE_Append);
            }
            
            // Log all messages to UE console
            TArray<FString> Lines;
            PipeOutput.ParseIntoArrayLines(Lines);
            for (const FString& Line : Lines)
            {
                if (!Line.TrimStartAndEnd().IsEmpty())
                {
                    UE_LOG(LogTemp, Log, TEXT("Training: %s"), *Line.TrimStartAndEnd());
                }
            }
        }
    }
    
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
    
}

void FTriangleSplattingManager::RefreshStatus()
{
    LastUpdateTime = FDateTime::MinValue(); // Force update on next call
    UpdateTrainingStatus();
}

// ============================================================================
// INTERNAL VALIDATION AND PREPARATION
// ============================================================================

bool FTriangleSplattingManager::ValidateConfiguration(const FTriangleSplattingConfig& Config)
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
    
    
    if (ErrorMessages.Num() > 0)
    {
        LastError = FString::Join(ErrorMessages, TEXT("; "));
        LogMessage(FString::Printf(TEXT("Configuration validation failed: %s"), *LastError), true);
        return false;
    }
    
    return true;
}

bool FTriangleSplattingManager::PrepareTrainingData(const FTriangleSplattingConfig& Config)
{
    // Create organized directory structure for Triangle Splatting training sessions
    // Config.OutputDirectory is already: ProjectSavedDir()/TriangleSplatting
    FString TSTrainingDir = FPaths::Combine(Config.OutputDirectory, TEXT("RatSplatting"));
    
    // Generate timestamp for this training session
    FDateTime Now = FDateTime::Now();
    FString Timestamp = Now.ToString(TEXT("%Y%m%d_%H%M%S"));
    FString SessionDirName = FString::Printf(TEXT("session_%s"), *Timestamp);
    OutputDirectory = FPaths::Combine(TSTrainingDir, SessionDirName);
    
    UE_LOG(LogTemp, Log, TEXT("Creating Triangle Splatting training session: %s"), *OutputDirectory);
    
    // Create main session directory
    if (!EnsureDirectoryExists(OutputDirectory))
    {
        return false;
    }
    
    // Create organized subdirectories without redundancy
    // Create simplified directory structure (reduced redundancy) 
    TArray<FString> MainSubDirectories = { 
        TEXT("config")    // Training configuration files
    };
    
    for (const FString& SubDir : MainSubDirectories)
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
    
    // Set up log file paths - separate files to avoid write conflicts
    LogFilePath = FPaths::Combine(OutputDirectory, TEXT("training.log"));  // C++ manager log
    PythonLogFilePath = FPaths::Combine(OutputDirectory, TEXT("python_training.log"));  // Python script log
    
    // Convert pose data and camera information with fx/fy priority over FOV
    FCameraIntrinsics Intrinsics = FVCCSimDataConverter::ConvertCameraParamsWithFocalLength(
        Config.FOVDegrees, Config.ImageWidth, Config.ImageHeight, 
        Config.FocalLengthX, Config.FocalLengthY);
    
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
    
    // Handle initialization data based on configuration
    if (Config.SelectedMesh.IsValid())
    {
        if (Config.bUseMeshTriangles)
        {
            // Skip point cloud generation when using direct mesh triangles
            UE_LOG(LogTemp, Log, TEXT("Using mesh triangles for initialization - skipping point cloud generation"));
        }
        else
        {
            // Traditional point cloud initialization from mesh
            FPointCloudData PointCloud = FVCCSimDataConverter::ConvertMeshToPointCloud(
                Config.SelectedMesh.Get(), Config.InitPointCount, true);
            
            if (PointCloud.GetPointCount() > 0)
            {
                FString PointCloudPath = FPaths::Combine(OutputDirectory, TEXT("config"), TEXT("init_points.ply"));
                FVCCSimDataConverter::SavePointCloudToPLY(PointCloud, PointCloudPath);
            }
        }
    }
    
    LogMessage(FString::Printf(TEXT("Training data prepared with %d camera poses"), CameraInfos.Num()));
    
    return true;
}

FString FTriangleSplattingManager::CreateConfigurationFile(const FTriangleSplattingConfig& Config)
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
    
    // Build command line arguments with unbuffered output (simplified with single workspace parameter)
    FString Arguments = FString::Printf(TEXT("-u \"%s\" --workspace \"%s\""),
        *ScriptPath, *OutputDirectory);
    
    LogMessage(FString::Printf(TEXT("Launching Python process: %s %s"), *PythonPath, *Arguments));
    
    // Create pipes for stdout and stderr capture
    void* PipeReadChild = nullptr;
    void* PipeWriteChild = nullptr;
    if (!FPlatformProcess::CreatePipe(PipeReadChild, PipeWriteChild))
    {
        LogMessage(TEXT("Failed to create pipe for process output"), true);
        return false;
    }
    
    // Store pipe handles for later cleanup
    ReadPipe = PipeReadChild;
    WritePipe = PipeWriteChild;
    
    // Launch process with stdout/stderr redirection
    TrainingProcessHandle = FPlatformProcess::CreateProc(
        *PythonPath,
        *Arguments,
        true,  // bLaunchDetached
        false, // bLaunchHidden - set to false for better debugging
        false, // bLaunchReallyHidden - set to false for better debugging  
        nullptr, // OutProcessID
        0,     // PriorityModifier
        nullptr, // OptionalWorkingDirectory
        PipeWriteChild,  // PipeWriteChild - redirect stdout/stderr to pipe
        PipeReadChild    // PipeReadChild
    );
    
    if (!TrainingProcessHandle.IsValid())
    {
        LogMessage(TEXT("Failed to launch training process"), true);
        return false;
    }
    
    // Update status to running
    CurrentStatus = ETrainingStatus::Running;
    
    LogMessage(TEXT("Training process launched successfully"));
    
    // Give the process a moment to start, then check if it's actually running
    FPlatformProcess::Sleep(0.5f);  // Wait 500ms
    if (!IsProcessRunning())
    {
        int32 ExitCode;
        if (GetProcessExitCode(ExitCode))
        {
            LogMessage(FString::Printf(TEXT("Python process exited immediately with code: %d"), ExitCode), true);
        }
        else
        {
            LogMessage(TEXT("Python process failed to start or exited immediately"), true);
        }
        CurrentStatus = ETrainingStatus::Failed;
        return false;
    }
    
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
        TEXT("C:/micromamba/envs/triangle_splatting/python.exe"), // Micromamba triangle_splatting environment (priority)
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
    
    // Try VCCSim custom script first (for new algorithm development)
    FString ScriptPath = PluginDir / TEXT("Source/triangle-splatting/train_vccsim.py");
    if (FPaths::FileExists(ScriptPath))
    {
        return ScriptPath;
    }
    
    // Try game engine script as fallback
    FString AlternativeScriptPath = PluginDir / TEXT("Source/triangle-splatting/train_game_engine.py");
    if (FPaths::FileExists(AlternativeScriptPath))
    {
        return AlternativeScriptPath;
    }
    
    // Try original script as last resort
    FString OriginalScriptPath = PluginDir / TEXT("Source/triangle-splatting/train.py");
    if (FPaths::FileExists(OriginalScriptPath))
    {
        return OriginalScriptPath;
    }
    
    LogMessage(TEXT("No Triangle Splatting training scripts found. Please ensure scripts are available."), true);
    return FString();
}

// ============================================================================
// LOG PARSING AND MONITORING
// ============================================================================

float FTriangleSplattingManager::ParseTrainingProgress()
{
    // Check Python log file first, then fallback to manager log file
    FString LogFileToCheck = PythonLogFilePath;
    if (LogFileToCheck.IsEmpty() || !FPaths::FileExists(LogFileToCheck))
    {
        LogFileToCheck = LogFilePath;
    }
    
    if (LogFileToCheck.IsEmpty() || !FPaths::FileExists(LogFileToCheck))
    {
        return 0.0f;
    }
    
    TArray<FString> RecentLines = ReadRecentLogLines(10, LogFileToCheck);
    
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

TArray<FString> FTriangleSplattingManager::ReadRecentLogLines(int32 MaxLines, const FString& SpecificLogFilePath)
{
    TArray<FString> Lines;
    
    FString FileToRead = SpecificLogFilePath.IsEmpty() ? LogFilePath : SpecificLogFilePath;
    if (FileToRead.IsEmpty() || !FPaths::FileExists(FileToRead))
    {
        return Lines;
    }
    
    TArray<FString> AllLines;
    if (FFileHelper::LoadFileToStringArray(AllLines, *FileToRead))
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
    
    // Clean up pipes
    if (ReadPipe != nullptr)
    {
        FPlatformProcess::ClosePipe(ReadPipe, WritePipe);
        ReadPipe = nullptr;
        WritePipe = nullptr;
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

FString FTriangleSplattingManager::ConfigToJsonString(const FTriangleSplattingConfig& Config)
{
    // Handle mesh triangle data export path - actual extraction will be done asynchronously
    FString MeshTrianglesPath;
    if (Config.bUseMeshTriangles && Config.SelectedMesh.IsValid())
    {
        // Set the expected path for mesh triangles (will be created by async extraction)
        MeshTrianglesPath = FPaths::Combine(OutputDirectory, TEXT("config"), TEXT("mesh_triangles.ply"));
        MeshTrianglesPath = MeshTrianglesPath.Replace(TEXT("\\"), TEXT("/"));
        
        // Note: Actual triangle extraction is handled asynchronously in StartTraining
        UE_LOG(LogTemp, Log, TEXT("Mesh triangles will be exported to: %s"), *MeshTrianglesPath);
    }

    return FString::Printf(TEXT(
        "{\n"
        "  \"image_directory\": \"%s\",\n"
        "  \"pose_file\": \"%s\",\n"
        "  \"output_directory\": \"%s\",\n"
        "  \"camera\": {\n"
        "    \"fov_degrees\": %.2f,\n"
        "    \"width\": %d,\n"
        "    \"height\": %d,\n"
        "    \"focal_length_x\": %.2f,\n"
        "    \"focal_length_y\": %.2f\n"
        "  },\n"
        "  \"training\": {\n"
        "    \"max_iterations\": %d\n"
        "  },\n"
        "  \"mesh\": {\n"
        "    \"use_mesh_initialization\": %s,\n"
        "    \"mesh_path\": \"%s\",\n"
        "    \"use_mesh_triangles\": %s,\n"
        "    \"max_mesh_triangles\": %d,\n"
        "    \"mesh_triangle_method\": \"%s\",\n"
        "    \"mesh_triangles_file\": \"%s\"\n"
        "  }\n"
        "}\n"
    ),
        *Config.ImageDirectory.Replace(TEXT("\\"), TEXT("/")),
        *Config.PoseFilePath.Replace(TEXT("\\"), TEXT("/")),
        *OutputDirectory.Replace(TEXT("\\"), TEXT("/")),
        Config.FOVDegrees,
        Config.ImageWidth,
        Config.ImageHeight,
        Config.FocalLengthX,
        Config.FocalLengthY,
        Config.MaxIterations,
        Config.bUseMeshInitialization ? TEXT("true") : TEXT("false"),
        Config.SelectedMesh.IsValid() ? *Config.SelectedMesh->GetPathName() : TEXT(""),
        Config.bUseMeshTriangles ? TEXT("true") : TEXT("false"),
        Config.MaxMeshTriangles,
        *Config.MeshTriangleMethod,
        *MeshTrianglesPath
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

FString FTriangleSplattingManager::GetTrainingOutput()
{
    // Read from Python process pipe if available
    if (ReadPipe != nullptr)
    {
        FString NewOutput = FPlatformProcess::ReadPipe(ReadPipe);
        if (!NewOutput.IsEmpty())
        {
            PythonOutputBuffer += NewOutput;
        }
    }
    
    return PythonOutputBuffer;
}

// ============================================================================
// ASYNC TRIANGLE EXTRACTION
// ============================================================================

void FTriangleSplattingManager::StartAsyncTriangleExtraction(const FTriangleSplattingConfig& Config)
{
    if (bTriangleExtractionInProgress)
    {
        UE_LOG(LogTemp, Warning, TEXT("Triangle extraction already in progress"));
        return;
    }
    
    if (!Config.SelectedMesh.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("Cannot extract triangles: no mesh selected"));
        return;
    }
    
    bTriangleExtractionInProgress = true;
    StatusMessage = TEXT("Extracting mesh triangles...");
    
    // Determine output path for mesh triangles
    FString MeshTrianglesPath = FPaths::Combine(OutputDirectory, TEXT("config"), TEXT("mesh_triangles.ply"));
    
    // Create the async task
    FTriangleExtractionTaskPtr* TaskPtr = new FTriangleExtractionTaskPtr(
        MakeShared<FAsyncTask<FTriangleExtractionTask>>(Config, MeshTrianglesPath, this)
    );
    TriangleExtractionTask = TaskPtr;
    
    // Start the async task
    (*TaskPtr)->StartBackgroundTask();
    
    UE_LOG(LogTemp, Log, TEXT("Started async triangle extraction for %d triangles"), Config.MaxMeshTriangles);
}

void FTriangleSplattingManager::OnTriangleExtractionComplete()
{
    if (!TriangleExtractionTask)
    {
        return;
    }
    
    FTriangleExtractionTaskPtr* TaskPtr = static_cast<FTriangleExtractionTaskPtr*>(TriangleExtractionTask);
    if (!TaskPtr->IsValid() || !(*TaskPtr)->IsDone())
    {
        return;
    }
    
    const FTriangleExtractionTask& Task = (*TaskPtr)->GetTask();
    
    bTriangleExtractionInProgress = false;
    
    if (Task.WasSuccessful())
    {
        UE_LOG(LogTemp, Log, TEXT("Triangle extraction completed successfully: %s"), *Task.GetResultMessage());
        StatusMessage = TEXT("Triangle extraction completed, starting training...");
        
        // Continue with training process after successful extraction
        if (CurrentConfig.IsValid())
        {
            // Launch training process now that triangle extraction is complete
            if (!LaunchTrainingProcess())
            {
                CurrentStatus = ETrainingStatus::Failed;
                StatusMessage = TEXT("Failed to launch training process");
                LogMessage(StatusMessage, true);
                OnTrainingCompleted.ExecuteIfBound(false, StatusMessage);
            }
            else
            {
                // Training launched successfully
                StatusMessage = TEXT("Training started successfully");
                LogMessage(StatusMessage);
            }
        }
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Triangle extraction failed: %s"), *Task.GetErrorMessage());
        CurrentStatus = ETrainingStatus::Failed;
        StatusMessage = FString::Printf(TEXT("Triangle extraction failed: %s"), *Task.GetErrorMessage());
        OnTrainingCompleted.ExecuteIfBound(false, StatusMessage);
    }
    
    // Clean up the task
    delete TaskPtr;
    TriangleExtractionTask = nullptr;
}