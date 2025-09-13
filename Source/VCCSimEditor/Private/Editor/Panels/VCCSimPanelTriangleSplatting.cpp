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

DEFINE_LOG_CATEGORY_STATIC(LogTriangleSplatting, Log, All);

#include "Editor/Panels/VCCSimPanelTriangleSplatting.h"
#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Utils/TriangleSplattingManager.h"
#include "Utils/ColmapManager.h"
#include "Utils/VCCSimDataConverter.h"
#include "IO/PLYUtils.h"
#include "HAL/PlatformFileManager.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Layout/SScrollBox.h"
#include "Widgets/Layout/SSeparator.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Layout/SExpandableArea.h"
#include "Widgets/Layout/SSpacer.h"
#include "Widgets/Input/SCheckBox.h"
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/SBoxPanel.h"
#include "SlateOptMacros.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"
#include "Framework/Notifications/NotificationManager.h"
#include "Widgets/Notifications/SNotificationList.h"
#include "Engine/StaticMesh.h"
#include "Editor/UnrealEd/Public/Editor.h"
#include "Engine/Engine.h"
#include "UObject/ConstructorHelpers.h"
#include "Widgets/Images/SImage.h"
#include "Framework/Application/SlateApplication.h"
#include "Misc/ConfigCacheIni.h"

BEGIN_SLATE_FUNCTION_BUILD_OPTIMIZATION

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

FVCCSimPanelTriangleSplatting::FVCCSimPanelTriangleSplatting()
{
    // Set default output directory
    GSConfig.OutputDirectory = TEXT("C:/UEProjects/VCCSimDev/Saved/TriangleSplatting");
    
    // Initialize default values
    GSFOVValue = GSConfig.FOVDegrees;
    GSImageWidthValue = GSConfig.ImageWidth;
    GSImageHeightValue = GSConfig.ImageHeight;
    GSFocalLengthXValue = GSConfig.FocalLengthX;
    GSFocalLengthYValue = GSConfig.FocalLengthY;
    GSMaxIterationsValue = GSConfig.MaxIterations;
    GSInitPointCountValue = GSConfig.InitPointCount;
    
    // Initialize mesh triangle values
    GSMaxMeshTrianglesValue = GSConfig.MaxMeshTriangles;
    GSMeshOpacityValue = GSConfig.MeshOpacity;
    
    // Initialize triangle selection methods
    TriangleSelectionMethods.Add(MakeShared<FString>(TEXT("Random")));
    // Future: Add more methods like "Uniform", "ImportanceBased", etc.
    
    // Defer loading of saved paths to Initialize() to avoid duplicate loads
}

FVCCSimPanelTriangleSplatting::~FVCCSimPanelTriangleSplatting()
{
    // Clean up training resources
    if (GSTrainingManager.IsValid() && GSTrainingManager->IsTrainingInProgress())
    {
        GSTrainingManager->StopTraining();
    }
    
    // Clear training timer
    if (GEditor && GSStatusUpdateTimerHandle.IsValid())
    {
        GEditor->GetTimerManager()->ClearTimer(GSStatusUpdateTimerHandle);
        GSStatusUpdateTimerHandle.Invalidate();
    }
}

void FVCCSimPanelTriangleSplatting::Initialize(
    TSharedPtr<FVCCSimPanelSelection> InSelectionManager)
{
    SelectionManager = InSelectionManager;
    
    InitializeGSManager();
    InitializeColmapManager();
    LoadPaths();
    UE_LOG(LogTriangleSplatting, Log, TEXT("VCCSimPanelTriangleSplatting initialized"));
}

void FVCCSimPanelTriangleSplatting::Cleanup()
{
    // Clean up training resources
    if (GSTrainingManager.IsValid())
    {
        if (GSTrainingManager->IsTrainingInProgress())
        {
            GSTrainingManager->StopTraining();
        }
        GSTrainingManager.Reset();
    }
    
    // Clean up COLMAP resources
    if (ColmapManager.IsValid())
    {
        ColmapManager.Reset();
    }
    
    // Clear timer
    if (GEditor && GSStatusUpdateTimerHandle.IsValid())
    {
        GEditor->GetTimerManager()->ClearTimer(GSStatusUpdateTimerHandle);
        GSStatusUpdateTimerHandle.Invalidate();
    }
    
    SelectionManager.Reset();
}

// ============================================================================
// TRIANGLE SPLATTING INITIALIZATION
// ============================================================================

void FVCCSimPanelTriangleSplatting::InitializeGSManager()
{
    // Create training manager
    GSTrainingManager = MakeShared<FTriangleSplattingManager>();
    
    GSTrainingManager->OnTrainingCompleted.BindLambda(
        [this](bool bSuccessful, FString ResultMessage)
    {
        bGSTrainingInProgress = false;
        
        // Stop status update timer
        if (GEditor && GSStatusUpdateTimerHandle.IsValid())
        {
            GEditor->GetTimerManager()->ClearTimer(GSStatusUpdateTimerHandle);
            GSStatusUpdateTimerHandle.Invalidate();
        }
        
        ShowGSNotification(ResultMessage, !bSuccessful);
    });
}

void FVCCSimPanelTriangleSplatting::InitializeColmapManager()
{
    // Create COLMAP manager
    ColmapManager = MakeShared<FColmapManager>();
    
    // Bind delegates for COLMAP progress updates
    ColmapManager->OnProgressUpdated.BindLambda([this](float Progress, FString StatusMessage)
    {
        // Progress is automatically tracked by ColmapManager
    });
    
    ColmapManager->OnCompleted.BindLambda([this](bool bSuccessful, FString ResultMessage)
    {
        bColmapPipelineInProgress = false;
        
        if (bSuccessful)
        {
            // Auto-fill the COLMAP dataset path with the generated timestamped directory
            FString GeneratedDatasetPath = ColmapManager->GetTimestampedDirectory();
            if (!GeneratedDatasetPath.IsEmpty() && FPaths::DirectoryExists(GeneratedDatasetPath))
            {
                GSConfig.ColmapDatasetPath = GeneratedDatasetPath;
                if (GSColmapDatasetTextBox.IsValid())
                {
                    GSColmapDatasetTextBox->SetText(FText::FromString(GeneratedDatasetPath));
                }
                UE_LOG(LogTriangleSplatting, Log, TEXT("Auto-filled COLMAP dataset "
                                          "path: %s"), *GeneratedDatasetPath);
                // Persist the auto-filled dataset path
                SavePaths();
                ShowGSNotification(FString::Printf(
                    TEXT("COLMAP completed! Dataset path auto-filled: %s"), 
                    *FPaths::GetCleanFilename(GeneratedDatasetPath)));
            }
            else
            {
                ShowGSNotification(ResultMessage, false);
            }
        }
        else
        {
            ShowGSNotification(ResultMessage, true);
        }
    });
}

// ============================================================================
// TRIANGLE SPLATTING TRAINING CONTROL
// ============================================================================

FReply FVCCSimPanelTriangleSplatting::OnGSStartTrainingClicked()
{
    if (ValidateGSConfiguration())
    {
        if (GSTrainingManager->StartTraining(GSConfig))
        {
            bGSTrainingInProgress = true;
            
            // Training started
            
            // Start status update timer
            if (GEditor)
            {
                GEditor->GetTimerManager()->SetTimer(
                    GSStatusUpdateTimerHandle,
                    FTimerDelegate::CreateLambda([this]()
                    {
                        if (GSTrainingManager.IsValid())
                        {
                            GSTrainingManager->UpdateTrainingStatus();
                        }
                    }),
                    1.0f, // Update every second
                    true  // Loop
                );
            }
            
            ShowGSNotification(TEXT("Triangle Splatting training started"));
        }
        else
        {
            ShowGSNotification(TEXT("Failed to start training process"), true);
        }
    }
    
    return FReply::Handled();
}

FReply FVCCSimPanelTriangleSplatting::OnGSStopTrainingClicked()
{
    if (GSTrainingManager.IsValid())
    {
        GSTrainingManager->StopTraining();
    }
    
    bGSTrainingInProgress = false;
    
    // Stop status update timer
    if (GEditor && GSStatusUpdateTimerHandle.IsValid())
    {
        GEditor->GetTimerManager()->ClearTimer(GSStatusUpdateTimerHandle);
        GSStatusUpdateTimerHandle.Invalidate();
    }
    
    ShowGSNotification(TEXT("Training stopped"));
    
    return FReply::Handled();
}

FReply FVCCSimPanelTriangleSplatting::OnGSColmapTrainingClicked()
{
    // Validate COLMAP dataset path
    if (GSConfig.ColmapDatasetPath.IsEmpty() || !FPaths::DirectoryExists(GSConfig.ColmapDatasetPath))
    {
        ShowGSNotification(TEXT("Please specify a valid COLMAP dataset path"), true);
        return FReply::Handled();
    }
    
    // Validate COLMAP dataset structure
    FString SparseDir = FPaths::Combine(GSConfig.ColmapDatasetPath, TEXT("sparse"));
    FString ImagesDir = FPaths::Combine(GSConfig.ColmapDatasetPath, TEXT("images"));
    
    if (!FPaths::DirectoryExists(SparseDir))
    {
        ShowGSNotification(TEXT("Invalid COLMAP dataset - missing sparse/ folder"), true);
        return FReply::Handled();
    }
    
    if (!FPaths::DirectoryExists(ImagesDir))
    {
        ShowGSNotification(TEXT("Invalid COLMAP dataset - missing images/ folder"), true);
        return FReply::Handled();
    }
    
    // Check for essential COLMAP files in sparse directory (support both txt and bin formats)
    FString SparseSubDir = FPaths::Combine(SparseDir, TEXT("0"));
    
    // Check cameras file (txt or bin)
    FString CamerasTxtFile = FPaths::Combine(SparseSubDir, TEXT("cameras.txt"));
    FString CamerasBinFile = FPaths::Combine(SparseSubDir, TEXT("cameras.bin"));
    bool bHasCameras = FPaths::FileExists(CamerasTxtFile) || FPaths::FileExists(CamerasBinFile);
    
    // Check images file (txt or bin)
    FString ImagesTxtFile = FPaths::Combine(SparseSubDir, TEXT("images.txt"));
    FString ImagesBinFile = FPaths::Combine(SparseSubDir, TEXT("images.bin"));
    bool bHasImages = FPaths::FileExists(ImagesTxtFile) || FPaths::FileExists(ImagesBinFile);
    
    // Check points3D file (txt or bin)
    FString Points3DTxtFile = FPaths::Combine(SparseSubDir, TEXT("points3D.txt"));
    FString Points3DBinFile = FPaths::Combine(SparseSubDir, TEXT("points3D.bin"));
    bool bHasPoints3D = FPaths::FileExists(Points3DTxtFile) || FPaths::FileExists(Points3DBinFile);
    
    if (!bHasCameras || !bHasImages || !bHasPoints3D)
    {
        ShowGSNotification(TEXT("Invalid COLMAP dataset - missing cameras, images or points3D files (txt or bin format) in sparse/0/ folder"), true);
        return FReply::Handled();
    }
    
    if (GSConfig.OutputDirectory.IsEmpty())
    {
        ShowGSNotification(TEXT("Please specify an output directory"), true);
        return FReply::Handled();
    }
    
    // Create output directory if it doesn't exist
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.DirectoryExists(*GSConfig.OutputDirectory))
    {
        if (!PlatformFile.CreateDirectoryTree(*GSConfig.OutputDirectory))
        {
            ShowGSNotification(TEXT("Failed to create output directory"), true);
            return FReply::Handled();
        }
    }
    
    UE_LOG(LogTriangleSplatting, Log, TEXT("Starting Triangle Splatting training with "
                              "COLMAP dataset: %s"), *GSConfig.ColmapDatasetPath);
    
    // Start Triangle Splatting training directly with COLMAP dataset
    StartTriangleSplattingWithColmapData(GSConfig.ColmapDatasetPath);
    
    return FReply::Handled();
}

void FVCCSimPanelTriangleSplatting::StartTriangleSplattingWithColmapData(const FString& ColmapDatasetPath)
{
    // Use original train.py for comparison experiments
    FString TriangleSplattingRoot = FPaths::Combine(FPaths::ProjectPluginsDir(), TEXT("VCCSim/Source/triangle-splatting"));
    FString TrainingScript = FPaths::Combine(TriangleSplattingRoot, TEXT("train.py"));
    
    if (!FPaths::FileExists(TrainingScript))
    {
        ShowGSNotification(TEXT("Original Triangle Splatting train.py script not found"), true);
        return;
    }
    
    UE_LOG(LogTriangleSplatting, Log, TEXT("Using original Triangle Splatting script for comparison: %s"), *TrainingScript);
    
    // Create Triangle Splatting output directory with timestamp
    FString TSOutputParentDir = FPaths::Combine(GSConfig.OutputDirectory, TEXT("triangle_splatting_output"));
    
    // Generate timestamp for this training session
    FDateTime Now = FDateTime::Now();
    FString Timestamp = Now.ToString(TEXT("%Y%m%d_%H%M%S"));
    FString SessionDirName = FString::Printf(TEXT("training_%s"), *Timestamp);
    FString TSOutputDir = FPaths::Combine(TSOutputParentDir, SessionDirName);
    
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.DirectoryExists(*TSOutputDir))
    {
        PlatformFile.CreateDirectoryTree(*TSOutputDir);
    }
    
    UE_LOG(LogTriangleSplatting, Log, TEXT("Triangle Splatting output directory: %s"), *TSOutputDir);
    
    // Build Triangle Splatting command with micromamba environment Python
    // Try to find micromamba triangle_splatting environment Python executable
    FString PythonCommand;
    FString MicromambaPython = TEXT("C:/micromamba/envs/triangle_splatting/python.exe");
    
    if (FPaths::FileExists(MicromambaPython))
    {
        PythonCommand = MicromambaPython;
        UE_LOG(LogTriangleSplatting, Log, TEXT("Using micromamba triangle_splatting Python: %s"), *PythonCommand);
    }
    else
    {
        // Fallback to system python
        PythonCommand = TEXT("python");
        UE_LOG(LogTriangleSplatting, Warning, TEXT("Micromamba triangle_splatting environment not found, using system python"));
    }
    
    FString Arguments = FString::Printf(TEXT("-u \"%s\" -s \"%s\" -m \"%s\" --eval"), 
        *TrainingScript, *ColmapDatasetPath, *TSOutputDir);
    
    // Add outdoor flag if scene seems outdoor (based on camera coverage)
    // This could be enhanced with better outdoor detection logic
    Arguments += TEXT(" --outdoor");
    
    UE_LOG(LogTriangleSplatting, Log, TEXT("Starting Triangle Splatting training: %s %s"), *PythonCommand, *Arguments);
    
    // Start Triangle Splatting training process
    if (GSTrainingManager->StartColmapTraining(PythonCommand, Arguments, TSOutputDir))
    {
        bGSTrainingInProgress = true;
        
        // Training with COLMAP data started
        
        // Start status update timer
        if (GEditor)
        {
            GEditor->GetTimerManager()->SetTimer(
                GSStatusUpdateTimerHandle,
                FTimerDelegate::CreateLambda([this]()
                {
                    if (GSTrainingManager.IsValid())
                    {
                        GSTrainingManager->UpdateTrainingStatus();
                        
                        // Update training status
                    }
                }),
                1.0f, // Update every second
                true  // Loop
            );
        }
        
        ShowGSNotification(TEXT("Triangle Splatting training with COLMAP data started"));
    }
    else
    {
        ShowGSNotification(TEXT("Failed to start Triangle Splatting training process"), true);
    }
}

FReply FVCCSimPanelTriangleSplatting::OnGSTestTransformationClicked()
{
    // Validate basic configuration first
    if (GSConfig.OutputDirectory.IsEmpty())
    {
        ShowGSNotification(TEXT("Please specify an output directory first"), true);
        return FReply::Handled();
    }
    
    // Create Test Transform subdirectory
    FString TestTransformOutputDir = FPaths::Combine(GSConfig.OutputDirectory, TEXT("Test Transform"));
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.DirectoryExists(*TestTransformOutputDir))
    {
        if (!PlatformFile.CreateDirectoryTree(*TestTransformOutputDir))
        {
            ShowGSNotification(TEXT("Failed to create Test Transform directory"), true);
            return FReply::Handled();
        }
    }
    
    bool bExportedAny = false;
    FString StatusMessage;
    
    // Export selected mesh to PLY (both original UE and transformed Triangle Splatting coordinates)
    if (GSConfig.SelectedMesh.IsValid())
    {
        try 
        {
            // Export original UE coordinates
            int32 PointCount = GSInitPointCountValue.Get(10000);
            FPointCloudData OriginalMesh = FVCCSimDataConverter::ConvertMeshToPointCloud(
                GSConfig.SelectedMesh.Get(), PointCount, false); 
            
            FString MeshUEPath = FPaths::Combine(TestTransformOutputDir,
                TEXT("test_mesh_ue_coordinates.ply"));
            if (FVCCSimDataConverter::SavePointCloudToPLY(OriginalMesh, MeshUEPath))
            {
                StatusMessage += FString::Printf(TEXT("Original mesh (UE coords)\n"));
                bExportedAny = true;
            }
            
            // Export transformed Triangle Splatting coordinates
            FPointCloudData TransformedMesh = FVCCSimDataConverter::ConvertMeshToPointCloud(
                GSConfig.SelectedMesh.Get(), PointCount, true);
            
            FString MeshTSPath = FPaths::Combine(TestTransformOutputDir,
                TEXT("test_mesh_ts_coordinates.ply"));
            if (FVCCSimDataConverter::SavePointCloudToPLY(TransformedMesh, MeshTSPath))
            {
                StatusMessage += FString::Printf(TEXT("Transformed mesh (TS coords)\n"));
                bExportedAny = true;
                
                // Log bounding boxes for comparison
                FVector UEMin = FVector(FLT_MAX), UEMax = FVector(-FLT_MAX);
                FVector TSMin = FVector(FLT_MAX), TSMax = FVector(-FLT_MAX);
                
                for (const FRatPoint& Point : OriginalMesh.Points)
                {
                    UEMin = FVector::Min(UEMin, Point.Position);
                    UEMax = FVector::Max(UEMax, Point.Position);
                }
                
                for (const FRatPoint& Point : TransformedMesh.Points)
                {
                    TSMin = FVector::Min(TSMin, Point.Position);
                    TSMax = FVector::Max(TSMax, Point.Position);
                }
                
                StatusMessage += FString::Printf(
                TEXT("UE mesh bbox: Min(%.2f,%.2f,%.2f) Max(%.2f,%.2f,%.2f)\n"), 
                    UEMin.X, UEMin.Y, UEMin.Z, UEMax.X, UEMax.Y, UEMax.Z);
                StatusMessage += FString::Printf(
                TEXT("TS mesh bbox: Min(%.2f,%.2f,%.2f) Max(%.2f,%.2f,%.2f)\n"), 
                        TSMin.X, TSMin.Y, TSMin.Z, TSMax.X, TSMax.Y, TSMax.Z);
            }
            else
            {
                StatusMessage += TEXT("✗ Failed to export transformed mesh\n");
            }
        }
        catch (...)
        {
            StatusMessage += TEXT("✗ Error converting mesh\n");
        }
    }
    else
    {
        StatusMessage += TEXT("⚠ No mesh selected\n");
    }
    
    // Export camera poses as points with normals (camera orientations)
    if (!GSConfig.PoseFilePath.IsEmpty() && FPaths::FileExists(GSConfig.PoseFilePath))
    {
        try
        {
            FCameraIntrinsics Intrinsics = FVCCSimDataConverter::ConvertCameraParamsWithFocalLength(
                GSConfig.FOVDegrees, GSConfig.ImageWidth, GSConfig.ImageHeight,
                GSConfig.FocalLengthX, GSConfig.FocalLengthY);
            
            TArray<FCameraInfo> CameraInfos = FVCCSimDataConverter::ConvertPoseFile(
                GSConfig.PoseFilePath, GSConfig.ImageDirectory, Intrinsics);
            
            if (CameraInfos.Num() > 0)
            {
                FString CameraPLYPath = FPaths::Combine(TestTransformOutputDir,
                    TEXT("test_cameras_transformed.ply"));
                ExportCamerasToPLY(CameraInfos, CameraPLYPath);
                
                // Save CameraInfo data for comparison
                FString CameraInfoPath = FPaths::Combine(TestTransformOutputDir,
                    TEXT("camera_info_data.txt"));
                SaveCameraInfoData(CameraInfos, CameraInfoPath);
                
                StatusMessage += FString::Printf(
                    TEXT("✓ %d cameras exported\n✓ CameraInfo data saved\n"), CameraInfos.Num());
                bExportedAny = true;
            }
            else
            {
                StatusMessage += TEXT("✗ No valid cameras found in pose file\n");
            }
        }
        catch (...)
        {
            StatusMessage += TEXT("✗ Error converting camera poses\n");
        }
    }
    else
    {
        StatusMessage += TEXT("⚠ No valid pose file specified\n");
    }
    
    // Show result
    if (bExportedAny)
    {
        StatusMessage += TEXT("\nOpen the PLY files to verify coordinate transformation!");
        ShowGSNotification(StatusMessage);
    }
    else
    {
        ShowGSNotification(TEXT("No data was exported. Please check "
                                "mesh selection and pose file."), true);
    }
    
    return FReply::Handled();
}

FReply FVCCSimPanelTriangleSplatting::OnGSExportColmapClicked()
{
    // Validate basic configuration first
    if (GSConfig.OutputDirectory.IsEmpty())
    {
        ShowGSNotification(TEXT("Please specify an output directory first"), true);
        return FReply::Handled();
    }
    
    if (GSConfig.PoseFilePath.IsEmpty() || !FPaths::FileExists(GSConfig.PoseFilePath))
    {
        ShowGSNotification(TEXT("Please specify a valid pose file"), true);
        return FReply::Handled();
    }
    
    if (GSConfig.ImageDirectory.IsEmpty() || !FPaths::DirectoryExists(GSConfig.ImageDirectory))
    {
        ShowGSNotification(TEXT("Please specify a valid image directory"), true);
        return FReply::Handled();
    }
    
    try
    {        
        // Start COLMAP pipeline asynchronously
        FString ColmapExecutablePath = TEXT("D:\\colmap-x64-windows-cuda\\bin");
        
        // Create COLMAP-specific output directory
        FString ColmapOutputDir = FPaths::Combine(GSConfig.OutputDirectory, TEXT("colmap_output"));
        
        if (ColmapManager->StartColmapPipeline(GSConfig.ImageDirectory,
            ColmapOutputDir, ColmapExecutablePath))
        {
            bColmapPipelineInProgress = true;
            ShowGSNotification(TEXT("COLMAP pipeline started in background\n\n"));
        }
        else
        {
            ShowGSNotification(TEXT("Failed to start COLMAP pipeline\n\n")
                              TEXT("Pipeline may already be running"), true);
        }
    }
    catch (...)
    {
        ShowGSNotification(TEXT("Unexpected error during COLMAP pipeline execution\n\n")
                          TEXT("Check UE log for detailed error information"), true);
    }
    
    return FReply::Handled();
}

// ============================================================================
// TRIANGLE SPLATTING TRAINING VALIDATION AND UTILITIES
// ============================================================================

bool FVCCSimPanelTriangleSplatting::ValidateGSConfiguration()
{
    TArray<FString> ErrorMessages;
    
    // Check required paths
    if (GSConfig.ImageDirectory.IsEmpty() || !FPaths::DirectoryExists(GSConfig.ImageDirectory))
    {
        ErrorMessages.Add(TEXT("Valid image directory is required"));
    }
    
    if (GSConfig.PoseFilePath.IsEmpty() || !FPaths::FileExists(GSConfig.PoseFilePath))
    {
        ErrorMessages.Add(TEXT("Valid pose file is required"));
    }
    
    if (GSConfig.OutputDirectory.IsEmpty())
    {
        ErrorMessages.Add(TEXT("Output directory is required"));
    }
    
    // Check mesh selection
    if (!GSConfig.SelectedMesh.IsValid())
    {
        ErrorMessages.Add(TEXT("Please select a mesh for initialization"));
    }
    
    // Check camera parameters
    if (GSConfig.FOVDegrees <= 0 || GSConfig.FOVDegrees >= 180)
    {
        ErrorMessages.Add(TEXT("FOV must be between 1 and 179 degrees"));
    }
    
    if (GSConfig.ImageWidth <= 0 || GSConfig.ImageHeight <= 0)
    {
        ErrorMessages.Add(TEXT("Image dimensions must be positive"));
    }
    
    // Check training parameters
    if (GSConfig.MaxIterations <= 0)
    {
        ErrorMessages.Add(TEXT("Max iterations must be positive"));
    }
    
    if (GSConfig.InitPointCount <= 0)
    {
        ErrorMessages.Add(TEXT("Init point count must be positive"));
    }
    
    
    if (ErrorMessages.Num() > 0)
    {
        FString CombinedError = FString::Join(ErrorMessages, TEXT("\n"));
        ShowGSNotification(CombinedError, true);
        return false;
    }
    
    return true;
}

void FVCCSimPanelTriangleSplatting::ShowGSNotification(const FString& Message, bool bIsError)
{
    FNotificationInfo NotificationInfo(FText::FromString(Message));
    NotificationInfo.bFireAndForget = true;
    NotificationInfo.FadeOutDuration = 3.0f;
    NotificationInfo.ExpireDuration = 5.0f;
    
    if (bIsError)
    {
        NotificationInfo.Image = FCoreStyle::Get().GetBrush(TEXT("MessageLog.Error"));
    }
    else
    {
        NotificationInfo.Image = FAppStyle::GetBrush(TEXT("Icons.Info"));
    }
    
    FSlateNotificationManager::Get().AddNotification(NotificationInfo);
}

void FVCCSimPanelTriangleSplatting::ExportCamerasToPLY(
    const TArray<FCameraInfo>& CameraInfos, const FString& OutputPath)
{
    // Use the new unified FPLYWriter class
    FPLYWriter::FPLYWriteConfig Config;
    Config.bIncludeColors = true;
    Config.bIncludeNormals = true;
    Config.bBinaryFormat = false;
    
    bool bSuccess = FPLYWriter::WriteCamerasToPLY(CameraInfos, OutputPath, Config);
    
    if (!bSuccess)
    {
        UE_LOG(LogTriangleSplatting, Error, TEXT("Failed to export cameras to PLY file: %s"), *OutputPath);
    }
}

void FVCCSimPanelTriangleSplatting::SaveCameraInfoData(
    const TArray<FCameraInfo>& CameraInfos, const FString& OutputPath)
{
    // Use the new utility function from FCameraInfoUtils
    bool bSuccess = FCameraInfoUtils::SaveCameraInfoToFile(
        CameraInfos, 
        OutputPath, 
        TEXT("Triangle Splatting (Right-handed, Z-up, meters)")
    );
    
    if (!bSuccess)
    {
        UE_LOG(LogTriangleSplatting, Error, TEXT("Failed to save CameraInfo data using utility function"));
    }
}

void FVCCSimPanelTriangleSplatting::OnGSCameraIntrinsicsLoaded()
{
    if (!GSConfig.CameraIntrinsicsFilePath.IsEmpty() && FPaths::FileExists(GSConfig.CameraIntrinsicsFilePath))
    {
        if (LoadCameraIntrinsicsFromColmap(GSConfig.CameraIntrinsicsFilePath))
        {
            ShowGSNotification(TEXT("Camera intrinsics loaded successfully"), false);
        }
        else
        {
            ShowGSNotification(TEXT("Failed to load camera intrinsics"), true);
        }
    }
}

bool FVCCSimPanelTriangleSplatting::LoadCameraIntrinsicsFromColmap(const FString& FilePath)
{
    if (!FPaths::FileExists(FilePath))
    {
        UE_LOG(LogTriangleSplatting, Error, TEXT("Camera intrinsics file does not exist: %s"), *FilePath);
        return false;
    }
    
    FString FileName = FPaths::GetCleanFilename(FilePath);
    
    if (FileName.Contains(TEXT("cameras.txt")))
    {
        return LoadCameraIntrinsicsFromColmapText(FilePath);
    }
    else if (FileName.Contains(TEXT("cameras.bin")))
    {
        return LoadCameraIntrinsicsFromColmapBinary(FilePath);
    }
    else
    {
        UE_LOG(LogTriangleSplatting, Error, TEXT("Unsupported camera file format: %s"), *FileName);
        return false;
    }
}

bool FVCCSimPanelTriangleSplatting::LoadCameraIntrinsicsFromColmapText(const FString& FilePath)
{
    TArray<FString> Lines;
    if (!FFileHelper::LoadFileToStringArray(Lines, *FilePath))
    {
        UE_LOG(LogTriangleSplatting, Error, TEXT("Failed to read camera file: %s"), *FilePath);
        return false;
    }
    
    for (const FString& Line : Lines)
    {
        FString TrimmedLine = Line.TrimStartAndEnd();
        if (TrimmedLine.IsEmpty() || TrimmedLine.StartsWith(TEXT("#")))
        {
            continue;
        }
        
        TArray<FString> Parts;
        TrimmedLine.ParseIntoArray(Parts, TEXT(" "), true);
        
        if (Parts.Num() < 4)
        {
            continue;
        }
        
        // Parse camera parameters
        // COLMAP format: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
        FString Model = Parts[1];
        int32 Width = FCString::Atoi(*Parts[2]);
        int32 Height = FCString::Atoi(*Parts[3]);
        
        // Update UI values
        GSImageWidthValue = Width;
        GSImageHeightValue = Height;
        GSConfig.ImageWidth = Width;
        GSConfig.ImageHeight = Height;
        
        if (Model == TEXT("PINHOLE") && Parts.Num() >= 8)
        {
            // PINHOLE: fx, fy, cx, cy
            float fx = FCString::Atof(*Parts[4]);
            float fy = FCString::Atof(*Parts[5]);
            float cx = FCString::Atof(*Parts[6]);
            float cy = FCString::Atof(*Parts[7]);
            
            // Update focal lengths
            GSFocalLengthXValue = fx;
            GSFocalLengthYValue = fy;
            GSConfig.FocalLengthX = fx;
            GSConfig.FocalLengthY = fy;
            
            // Calculate FOV from focal length
            float FovRadians = 2.0f * FMath::Atan(Width / (2.0f * fx));
            float FovDegrees = FMath::RadiansToDegrees(FovRadians);
            GSFOVValue = FovDegrees;
            GSConfig.FOVDegrees = FovDegrees;
            
            UE_LOG(LogTriangleSplatting, Log, TEXT("Loaded PINHOLE camera: %dx%d, fx=%.2f, fy=%.2f, FOV=%.2f°"), 
                Width, Height, fx, fy, FovDegrees);
            return true;
        }
        else if (Model == TEXT("SIMPLE_PINHOLE") && Parts.Num() >= 7)
        {
            // SIMPLE_PINHOLE: f, cx, cy
            float f = FCString::Atof(*Parts[4]);
            float cx = FCString::Atof(*Parts[5]);
            float cy = FCString::Atof(*Parts[6]);
            
            // Use same focal length for both directions
            GSFocalLengthXValue = f;
            GSFocalLengthYValue = f;
            GSConfig.FocalLengthX = f;
            GSConfig.FocalLengthY = f;
            
            // Calculate FOV from focal length
            float FovRadians = 2.0f * FMath::Atan(Width / (2.0f * f));
            float FovDegrees = FMath::RadiansToDegrees(FovRadians);
            GSFOVValue = FovDegrees;
            GSConfig.FOVDegrees = FovDegrees;
            
            UE_LOG(LogTriangleSplatting, Log, TEXT("Loaded SIMPLE_PINHOLE camera: %dx%d, f=%.2f, FOV=%.2f°"), 
                Width, Height, f, FovDegrees);
            return true;
        }
        else
        {
            UE_LOG(LogTriangleSplatting, Warning, TEXT("Unsupported camera model: %s"), *Model);
        }
    }
    
    return false;
}

bool FVCCSimPanelTriangleSplatting::LoadCameraIntrinsicsFromColmapBinary(const FString& FilePath)
{
    TArray<uint8> FileData;
    if (!FFileHelper::LoadFileToArray(FileData, *FilePath))
    {
        UE_LOG(LogTriangleSplatting, Error, TEXT("Failed to read binary camera file: %s"), *FilePath);
        return false;
    }
    
    if (FileData.Num() < 8)
    {
        UE_LOG(LogTriangleSplatting, Error, TEXT("Camera file too small: %s"), *FilePath);
        return false;
    }
    
    // Read number of cameras (8 bytes)
    uint64 NumCameras = *reinterpret_cast<const uint64*>(FileData.GetData());
    
    if (NumCameras == 0)
    {
        UE_LOG(LogTriangleSplatting, Warning, TEXT("No cameras found in file: %s"), *FilePath);
        return false;
    }
    
    int32 Offset = 8;
    
    // We only process the first valid camera we find
    for (uint64 i = 0; i < NumCameras && Offset < FileData.Num(); ++i)
    {
        if (Offset + 24 > FileData.Num()) // Minimum required bytes for camera header
        {
            break;
        }
        
        // Read camera ID (4 bytes)
        uint32 CameraId = *reinterpret_cast<const uint32*>(FileData.GetData() + Offset);
        Offset += 4;
        
        // Read model ID (4 bytes)
        uint32 ModelId = *reinterpret_cast<const uint32*>(FileData.GetData() + Offset);
        Offset += 4;
        
        // Read width and height (8 bytes)
        uint64 Width = *reinterpret_cast<const uint64*>(FileData.GetData() + Offset);
        Offset += 8;
        uint64 Height = *reinterpret_cast<const uint64*>(FileData.GetData() + Offset);
        Offset += 8;
        
        // Update UI values
        GSImageWidthValue = static_cast<int32>(Width);
        GSImageHeightValue = static_cast<int32>(Height);
        GSConfig.ImageWidth = static_cast<int32>(Width);
        GSConfig.ImageHeight = static_cast<int32>(Height);
        
        // Read parameters based on model
        if (ModelId == 0) // SIMPLE_PINHOLE
        {
            if (Offset + 24 > FileData.Num()) 
            {
                break; // Not enough data for parameters
            }
            
            double f = *reinterpret_cast<const double*>(FileData.GetData() + Offset);
            double cx = *reinterpret_cast<const double*>(FileData.GetData() + Offset + 8);
            double cy = *reinterpret_cast<const double*>(FileData.GetData() + Offset + 16);
            
            GSFocalLengthXValue = static_cast<float>(f);
            GSFocalLengthYValue = static_cast<float>(f);
            GSConfig.FocalLengthX = static_cast<float>(f);
            GSConfig.FocalLengthY = static_cast<float>(f);
            
            float FovRadians = 2.0f * FMath::Atan(Width / (2.0f * f));
            float FovDegrees = FMath::RadiansToDegrees(FovRadians);
            GSFOVValue = FovDegrees;
            GSConfig.FOVDegrees = FovDegrees;
            
            UE_LOG(LogTriangleSplatting, Log, TEXT("Loaded SIMPLE_PINHOLE camera from binary: %dx%d, f=%.2f, FOV=%.2f°"), 
                (int32)Width, (int32)Height, (float)f, FovDegrees);
            return true;
        }
        else if (ModelId == 1) // PINHOLE
        {
            if (Offset + 32 > FileData.Num()) 
            {
                break; // Not enough data for parameters
            }
            
            double fx = *reinterpret_cast<const double*>(FileData.GetData() + Offset);
            double fy = *reinterpret_cast<const double*>(FileData.GetData() + Offset + 8);
            double cx = *reinterpret_cast<const double*>(FileData.GetData() + Offset + 16);
            double cy = *reinterpret_cast<const double*>(FileData.GetData() + Offset + 24);
            
            GSFocalLengthXValue = static_cast<float>(fx);
            GSFocalLengthYValue = static_cast<float>(fy);
            GSConfig.FocalLengthX = static_cast<float>(fx);
            GSConfig.FocalLengthY = static_cast<float>(fy);
            
            float FovRadians = 2.0f * FMath::Atan(Width / (2.0f * fx));
            float FovDegrees = FMath::RadiansToDegrees(FovRadians);
            GSFOVValue = FovDegrees;
            GSConfig.FOVDegrees = FovDegrees;
            
            UE_LOG(LogTriangleSplatting, Log, TEXT("Loaded PINHOLE camera from binary: %dx%d, fx=%.2f, fy=%.2f, FOV=%.2f°"), 
                (int32)Width, (int32)Height, (float)fx, (float)fy, FovDegrees);
            return true;
        }
        else
        {
            UE_LOG(LogTriangleSplatting, Warning, TEXT("Unsupported camera model ID: %d"), ModelId);
            // Skip this camera and try the next one
            continue;
        }
    }
    
    return false;
}

// ============================================================================
// PATH PERSISTENCE
// ============================================================================

void FVCCSimPanelTriangleSplatting::SavePaths()
{
    // Save to both project's Saved folder and editor config for better persistence
    SavePathsToProjectFile();
}

void FVCCSimPanelTriangleSplatting::LoadPaths()
{
    if (LoadPathsFromProjectFile())
    {
        // Update UI values to reflect loaded configuration
        UpdateUIFromConfig();

        // If intrinsics path is available, auto-load camera parameters from file
        if (!GSConfig.CameraIntrinsicsFilePath.IsEmpty() &&
            FPaths::FileExists(GSConfig.CameraIntrinsicsFilePath))
        {
            OnGSCameraIntrinsicsLoaded();
        }
    }
}

// ============================================================================
// PATH PERSISTENCE IMPLEMENTATION
// ============================================================================

void FVCCSimPanelTriangleSplatting::SavePathsToProjectFile()
{
    const FString ConfigFilePath = GetPathConfigFilePath();

    TArray<FString> Lines;
    Lines.Add(FString::Printf(TEXT("ImageDirectory=%s"), *GSConfig.ImageDirectory));
    Lines.Add(FString::Printf(TEXT("CameraIntrinsicsFilePath=%s"), *GSConfig.CameraIntrinsicsFilePath));
    Lines.Add(FString::Printf(TEXT("PoseFilePath=%s"), *GSConfig.PoseFilePath));
    Lines.Add(FString::Printf(TEXT("OutputDirectory=%s"), *GSConfig.OutputDirectory));
    Lines.Add(FString::Printf(TEXT("ColmapDatasetPath=%s"), *GSConfig.ColmapDatasetPath));
    const FString MeshPath = GSConfig.SelectedMesh.IsValid() ? GSConfig.SelectedMesh->GetPathName() : TEXT("");
    Lines.Add(FString::Printf(TEXT("SelectedMeshPath=%s"), *MeshPath));

    // Ensure directory exists
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    const FString ConfigDir = FPaths::GetPath(ConfigFilePath);
    if (!PlatformFile.DirectoryExists(*ConfigDir))
    {
        PlatformFile.CreateDirectoryTree(*ConfigDir);
    }

    const FString FileContent = FString::Join(Lines, TEXT("\n"));
    if (FFileHelper::SaveStringToFile(FileContent, *ConfigFilePath))
    {
        UE_LOG(LogTriangleSplatting, Log, TEXT("Triangle Splatting paths saved to: %s"), *ConfigFilePath);
    }
    else
    {
        UE_LOG(LogTriangleSplatting, Warning, TEXT("Failed to save Triangle Splatting "
                                      "paths to: %s"), *ConfigFilePath);
    }
}

bool FVCCSimPanelTriangleSplatting::LoadPathsFromProjectFile()
{
    // Load key=value config file for Triangle Splatting paths
    const FString PathsFile = GetPathConfigFilePath();

    auto LoadKeyValueFile = [this](const FString& FilePath) -> bool
    {
        FString FileContent;
        if (!FFileHelper::LoadFileToString(FileContent, *FilePath))
        {
            return false;
        }

        bool bAnySetLocal = false;
        TArray<FString> Lines;
        FileContent.ParseIntoArrayLines(Lines, true);

        auto ExtractKV = [&Lines](const FString& Key) -> FString
        {
            const FString Prefix = Key + TEXT("=");
            for (const FString& L : Lines)
            {
                if (L.StartsWith(Prefix))
                {
                    FString Value = L.Mid(Prefix.Len());
                    return Value.TrimStartAndEnd();
                }
            }
            return FString();
        };

        auto TrySet = [&bAnySetLocal](FString& Target, const FString& Value)
        {
            if (!Value.IsEmpty()) { Target = Value; bAnySetLocal = true; }
        };

        TrySet(GSConfig.ImageDirectory, ExtractKV(TEXT("ImageDirectory")));
        TrySet(GSConfig.CameraIntrinsicsFilePath, ExtractKV(TEXT("CameraIntrinsicsFilePath")));
        TrySet(GSConfig.PoseFilePath, ExtractKV(TEXT("PoseFilePath")));
        TrySet(GSConfig.OutputDirectory, ExtractKV(TEXT("OutputDirectory")));
        TrySet(GSConfig.ColmapDatasetPath, ExtractKV(TEXT("ColmapDatasetPath")));

        const FString MeshPathKV = ExtractKV(TEXT("SelectedMeshPath"));
        if (!MeshPathKV.IsEmpty())
        {
            if (UStaticMesh* LoadedMesh = LoadObject<UStaticMesh>(nullptr, *MeshPathKV))
            {
                GSConfig.SelectedMesh = LoadedMesh;
                bAnySetLocal = true;
            }
        }

        if (bAnySetLocal)
        {
            UE_LOG(LogTriangleSplatting, Log, TEXT("Triangle Splatting paths loaded from: %s"), *FilePath);
        }
        return bAnySetLocal;
    };

    // Load key=value file if present
    if (FPaths::FileExists(PathsFile))
    {
        return LoadKeyValueFile(PathsFile);
    }

    return false;
}

FString FVCCSimPanelTriangleSplatting::GetPathConfigFilePath() const
{
    // Save to project's Saved folder: ProjectDir/Saved/Config/VCCSimTriangleSplatting.txt
    const FString ProjectSavedDir = FPaths::ProjectSavedDir();
    const FString ConfigDir = FPaths::Combine(ProjectSavedDir, TEXT("Config"));
    return FPaths::Combine(ConfigDir, TEXT("VCCSimTriangleSplatting.txt"));
}

void FVCCSimPanelTriangleSplatting::UpdateUIFromConfig()
{
    // Update TOptional values from config
    GSFOVValue = GSConfig.FOVDegrees;
    GSImageWidthValue = GSConfig.ImageWidth;
    GSImageHeightValue = GSConfig.ImageHeight;
    GSFocalLengthXValue = GSConfig.FocalLengthX;
    GSFocalLengthYValue = GSConfig.FocalLengthY;
    GSMaxIterationsValue = GSConfig.MaxIterations;
    GSInitPointCountValue = GSConfig.InitPointCount;
    GSMaxMeshTrianglesValue = GSConfig.MaxMeshTriangles;
    GSMeshOpacityValue = GSConfig.MeshOpacity;
    
    // Update text box contents if widgets exist
    if (GSImageDirectoryTextBox.IsValid())
    {
        GSImageDirectoryTextBox->SetText(FText::FromString(GSConfig.ImageDirectory));
    }
    
    if (GSCameraIntrinsicsFileTextBox.IsValid())
    {
        GSCameraIntrinsicsFileTextBox->SetText(FText::FromString(GSConfig.CameraIntrinsicsFilePath));
    }
    
    if (GSPoseFileTextBox.IsValid())
    {
        GSPoseFileTextBox->SetText(FText::FromString(GSConfig.PoseFilePath));
    }
    
    if (GSOutputDirectoryTextBox.IsValid())
    {
        GSOutputDirectoryTextBox->SetText(FText::FromString(GSConfig.OutputDirectory));
    }
    
    if (GSColmapDatasetTextBox.IsValid())
    {
        GSColmapDatasetTextBox->SetText(FText::FromString(GSConfig.ColmapDatasetPath));
    }
    
    // Silent update; avoid noisy logs during panel creation
    UE_LOG(LogTriangleSplatting, Log, TEXT("  ImageDirectory: %s"), *GSConfig.ImageDirectory);
    UE_LOG(LogTriangleSplatting, Log, TEXT("  CameraIntrinsicsFilePath1: %s"), *GSConfig.CameraIntrinsicsFilePath);
    UE_LOG(LogTriangleSplatting, Log, TEXT("  PoseFilePath: %s"), *GSConfig.PoseFilePath);
    UE_LOG(LogTriangleSplatting, Log, TEXT("  OutputDirectory: %s"), *GSConfig.OutputDirectory);
    UE_LOG(LogTriangleSplatting, Log, TEXT("  ColmapDatasetPath: %s"), *GSConfig.ColmapDatasetPath);
}

END_SLATE_FUNCTION_BUILD_OPTIMIZATION
