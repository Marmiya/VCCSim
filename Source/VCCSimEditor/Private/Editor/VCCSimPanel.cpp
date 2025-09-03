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

#include "Editor/VCCSimPanel.h"
#include "Engine/Selection.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Text/STextBlock.h"
#include "Pawns/FlashPawn.h"
#include "Misc/DateTime.h"
#include "HAL/FileManager.h"
#include "Misc/Paths.h"
#include "DataStruct_IO/Mesh.h"
#include "Sensors/CameraSensor.h"
#include "Sensors/DepthCamera.h"
#include "Sensors/SegmentCamera.h"
#include "Sensors/NormalCamera.h"
#include "Simulation/PathPlanner.h"
#include "Simulation/SceneAnalysisManager.h"
#include "Utils/ImageProcesser.h"
#include "Utils/TrajectoryViewer.h"
#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#include "Misc/FileHelper.h"
#include "HighResScreenshot.h"
#include "LevelEditorViewport.h"
#include "Utils/TriangleSplattingManager.h"
#include "DrawDebugHelpers.h"
#include "EngineUtils.h"

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

SVCCSimPanel::~SVCCSimPanel()
{
    // Unregister from selection events
    if (GEditor && GEditor->GetSelectedActors())
    {
        GEditor->GetSelectedActors()->SelectionChangedEvent.RemoveAll(this);
    }
    
    // Clear timer if active
    if (GEditor && bAutoCaptureInProgress)
    {
        GEditor->GetTimerManager()->ClearTimer(AutoCaptureTimerHandle);
        bAutoCaptureInProgress = false;
    }

    // Clean up path visualization
    if (GEditor && GEditor->GetEditorWorldContext().World())
    {
        UWorld* World = GEditor->GetEditorWorldContext().World();
        
        // Clean up any PathVisualization actors
        for (TActorIterator<AActor> ActorIterator(World); ActorIterator; ++ActorIterator)
        {
            AActor* Actor = *ActorIterator;
            if (Actor && (Actor->GetActorLabel().Contains(TEXT("PathVisualization")) || 
                         Actor->Tags.Contains(FName("VCCSimPathViz"))))
            {
                World->DestroyActor(Actor);
            }
        }
    }
    
    if (PathVisualizationActor.IsValid() && GEditor)
    {
        if (UWorld* World = GEditor->GetEditorWorldContext().World())
        {
            World->DestroyActor(PathVisualizationActor.Get());
        }
        PathVisualizationActor.Reset();
    }

    // Clean up scene analysis visualizations
    if (SceneAnalysisManager.IsValid())
    {
        SceneAnalysisManager->InterfaceClearSafeZoneVisualization();
        SceneAnalysisManager->InterfaceClearCoverageVisualization();
        SceneAnalysisManager->InterfaceClearComplexityVisualization();
    }

    // Clean up Triangle Splatting resources
    if (GSTrainingManager.IsValid() && GSTrainingManager->IsTrainingInProgress())
    {
        GSTrainingManager->StopTraining();
    }
    
    // Clear Triangle Splatting timer
    if (GEditor && GSStatusUpdateTimerHandle.IsValid())
    {
        GEditor->GetTimerManager()->ClearTimer(GSStatusUpdateTimerHandle);
        GSStatusUpdateTimerHandle.Invalidate();
    }
}

// ============================================================================
// SELECTION MANAGEMENT
// ============================================================================

void SVCCSimPanel::OnSelectFlashPawnToggleChanged(ECheckBoxState NewState)
{
    bSelectingFlashPawn = (NewState == ECheckBoxState::Checked);
    
    // If turning on FlashPawn selection, disable target selection
    if (bSelectingFlashPawn && bSelectingTarget)
    {
        bSelectingTarget = false;
        SelectTargetToggle->SetIsChecked(ECheckBoxState::Unchecked);
    }
}

void SVCCSimPanel::OnSelectTargetToggleChanged(ECheckBoxState NewState)
{
    bSelectingTarget = (NewState == ECheckBoxState::Checked);
    
    // If turning on Target selection, disable FlashPawn selection
    if (bSelectingTarget && bSelectingFlashPawn)
    {
        bSelectingFlashPawn = false;
        SelectFlashPawnToggle->SetIsChecked(ECheckBoxState::Unchecked);
    }
}

void SVCCSimPanel::OnUseLimitedToggleChanged(ECheckBoxState NewState)
{
    bUseLimited = (NewState == ECheckBoxState::Checked);
}

void SVCCSimPanel::OnSelectionChanged(UObject* Object)
{
    // Skip if we're not in selection mode
    if (!bSelectingFlashPawn && !bSelectingTarget)
    {
        return;
    }
    
    USelection* Selection = GEditor->GetSelectedActors();
    if (!Selection || Selection->Num() == 0)
    {
        return;
    }
    
    // Process only the first selected actor
    AActor* Actor = Cast<AActor>(Selection->GetSelectedObject(0));
    if (!Actor)
    {
        return;
    }
    
    // Handle FlashPawn selection
    if (bSelectingFlashPawn)
    {
        AFlashPawn* FlashPawn = Cast<AFlashPawn>(Actor);
        if (FlashPawn)
        {
            SelectedFlashPawn = FlashPawn;
            SelectedFlashPawnText->SetText(FText::FromString(FlashPawn->GetActorLabel()));
            
            // Turn off selection mode
            bSelectingFlashPawn = false;
            SelectFlashPawnToggle->SetIsChecked(ECheckBoxState::Unchecked);
            
            // Check what camera components are available
            CheckCameraComponents();
        }
    }
    // Handle target selection
    else if (bSelectingTarget)
    {
        // Skip if it's a FlashPawn (can't target itself)
        if (!Actor->IsA<AFlashPawn>())
        {
            SelectedTargetObject = Actor;
            SelectedTargetObjectText->SetText(FText::FromString(Actor->GetActorLabel()));
            
            // Turn off selection mode
            bSelectingTarget = false;
            SelectTargetToggle->SetIsChecked(ECheckBoxState::Unchecked);
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("Cannot select a FlashPawn as a target"));
        }
    }
}

// ============================================================================
// CAMERA MANAGEMENT
// ============================================================================

void SVCCSimPanel::OnRGBCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseRGBCamera = (NewState == ECheckBoxState::Checked);
    if (bUseRGBCamera)
    {
        TArray<URGBCameraComponent*> RGBCameras;
        SelectedFlashPawn->GetComponents<URGBCameraComponent>(RGBCameras);
        for (URGBCameraComponent* Camera : RGBCameras)
        {
            if (Camera)
            {
                Camera->SetActive(bUseRGBCamera);
                Camera->InitializeRenderTargets();
                Camera->SetCaptureComponent();
            }
        }
    }
}

void SVCCSimPanel::OnDepthCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseDepthCamera = (NewState == ECheckBoxState::Checked);
    if (bUseDepthCamera)
    {
        TArray<UDepthCameraComponent*> DepthCameras;
        SelectedFlashPawn->GetComponents<UDepthCameraComponent>(DepthCameras);
        for (UDepthCameraComponent* Camera : DepthCameras)
        {
            if (Camera)
            {
                Camera->SetActive(bUseDepthCamera);
                Camera->InitializeRenderTargets();
                Camera->SetCaptureComponent();
            }
        }
    }
}

void SVCCSimPanel::OnSegmentationCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseSegmentationCamera = (NewState == ECheckBoxState::Checked);
    if (bUseSegmentationCamera)
    {
        TArray<USegmentationCameraComponent*> SegmentationCameras;
        SelectedFlashPawn->GetComponents<USegmentationCameraComponent>(SegmentationCameras);
        for (USegmentationCameraComponent* Camera : SegmentationCameras)
        {
            if (Camera)
            {
                Camera->SetActive(bUseSegmentationCamera);
                Camera->InitializeRenderTargets();
                Camera->SetCaptureComponent();
            }
        }
    }
}

void SVCCSimPanel::OnNormalCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseNormalCamera = (NewState == ECheckBoxState::Checked);
    if (bUseNormalCamera)
    {
        TArray<UNormalCameraComponent*> NormalCameras;
        SelectedFlashPawn->GetComponents<UNormalCameraComponent>(NormalCameras);
        for (UNormalCameraComponent* Camera : NormalCameras)
        {
            if (Camera)
            {
                Camera->SetActive(bUseNormalCamera);
                Camera->InitializeRenderTargets();
                Camera->SetCaptureComponent();
            }
        }
    }
}

void SVCCSimPanel::CheckCameraComponents()
{
    bHasRGBCamera = false;
    bHasDepthCamera = false;
    bHasNormalCamera = false;
    bHasSegmentationCamera = false;
    
    if (!SelectedFlashPawn.IsValid())
    {
        return;
    }
    
    // Check for RGB cameras
    TArray<URGBCameraComponent*> RGBCameras;
    SelectedFlashPawn->GetComponents<URGBCameraComponent>(RGBCameras);
    bHasRGBCamera = (RGBCameras.Num() > 0);
    
    // Check for Depth cameras
    TArray<UDepthCameraComponent*> DepthCameras;
    SelectedFlashPawn->GetComponents<UDepthCameraComponent>(DepthCameras);
    bHasDepthCamera = (DepthCameras.Num() > 0);
    
    // Check for Normal cameras
    TArray<UNormalCameraComponent*> NormalCameras;
    SelectedFlashPawn->GetComponents<UNormalCameraComponent>(NormalCameras);
    bHasNormalCamera = (NormalCameras.Num() > 0);
    
    // Check for Segmentation cameras
    TArray<USegmentationCameraComponent*> SegmentationCameras;
    SelectedFlashPawn->GetComponents<USegmentationCameraComponent>(SegmentationCameras);
    bHasSegmentationCamera = (SegmentationCameras.Num() > 0);
    
    // Reset checkboxes if corresponding cameras aren't available
    if (!bHasRGBCamera)
    {
        bUseRGBCamera = false;
        RGBCameraCheckBox->SetIsChecked(ECheckBoxState::Unchecked);
    }
    
    if (!bHasDepthCamera)
    {
        bUseDepthCamera = false;
        DepthCameraCheckBox->SetIsChecked(ECheckBoxState::Unchecked);
    }
    
    if (!bHasNormalCamera)
    {
        bUseNormalCamera = false;
        NormalCameraCheckBox->SetIsChecked(ECheckBoxState::Unchecked);
    }
    
    if (!bHasSegmentationCamera)
    {
        bUseSegmentationCamera = false;
        SegmentationCameraCheckBox->SetIsChecked(ECheckBoxState::Unchecked);
    }

    UpdateActiveCameras();
}

void SVCCSimPanel::UpdateActiveCameras()
{
    if (!SelectedFlashPawn.IsValid())
    {
        return;
    }
    
    // Update RGB cameras
    TArray<URGBCameraComponent*> RGBCameras;
    SelectedFlashPawn->GetComponents<URGBCameraComponent>(RGBCameras);
    for (URGBCameraComponent* Camera : RGBCameras)
    {
        if (Camera)
        {
            Camera->SetActive(bUseRGBCamera);
            Camera->InitializeRenderTargets();
            Camera->SetCaptureComponent();
        }
    }
    
    // Update Depth cameras
    TArray<UDepthCameraComponent*> DepthCameras;
    SelectedFlashPawn->GetComponents<UDepthCameraComponent>(DepthCameras);
    for (UDepthCameraComponent* Camera : DepthCameras)
    {
        if (Camera)
        {
            Camera->SetActive(bUseDepthCamera);
            Camera->InitializeRenderTargets();
            Camera->SetCaptureComponent();
        }
    }
    
    // Update Normal cameras
    TArray<UNormalCameraComponent*> NormalCameras;
    SelectedFlashPawn->GetComponents<UNormalCameraComponent>(NormalCameras);
    for (UNormalCameraComponent* Camera : NormalCameras)
    {
        if (Camera)
        {
            Camera->SetActive(bUseNormalCamera);
            Camera->InitializeRenderTargets();
            Camera->SetCaptureComponent();
        }
    }
    
    // Update Segmentation cameras
    TArray<USegmentationCameraComponent*> SegmentationCameras;
    SelectedFlashPawn->GetComponents<USegmentationCameraComponent>(SegmentationCameras);
    for (USegmentationCameraComponent* Camera : SegmentationCameras)
    {
        if (Camera)
        {
            Camera->SetActive(bUseSegmentationCamera);
            Camera->InitializeRenderTargets();
            Camera->SetCaptureComponent();
        }
    }
}

// ============================================================================
// POSE GENERATION AND MANAGEMENT
// ============================================================================

FReply SVCCSimPanel::OnGeneratePosesClicked()
{
    GeneratePosesAroundTarget();
    
    // Clean up any existing visualization
    HidePathVisualization();
    
    // Allow path visualization after generating poses
    bPathVisualized = false;
    bPathNeedsUpdate = false;
    
    return FReply::Handled();
}

void SVCCSimPanel::GeneratePosesAroundTarget()
{
    if (!SelectedFlashPawn.IsValid() || !SelectedTargetObject.IsValid())
    {
        UE_LOG(LogTemp, Warning, TEXT("Must select both "
                                      "a FlashPawn and a target object"));
        return;
    }

    TArray<FVector> Positions;
    TArray<FRotator> Rotations;
    TArray<FMeshInfo> MeshInfos;

    TArray<UStaticMeshComponent*> MeshComponents;
    SelectedTargetObject->GetComponents<UStaticMeshComponent>(MeshComponents);
    for (UStaticMeshComponent* MeshComponent : MeshComponents)
    {
        if (MeshComponent)
        {
            FMeshInfo MeshInfo;
            ASceneAnalysisManager::ExtractMeshData(MeshComponent, MeshInfo);
            MeshInfos.Add(MeshInfo);
        }
    }

    UPathPlanner::SemiSphericalPath(
        MeshInfos, Radius, NumPoses,
        VerticalGap, Positions, Rotations);
    
    // Set the path on the FlashPawn
    SelectedFlashPawn->SetPathPanel(Positions, Rotations);
    SelectedFlashPawn->MoveTo(0);
    
    // Update NumPoses to match actual number of generated poses
    NumPoses = Positions.Num();
    NumPosesValue = NumPoses;
    
    // Reset any ongoing auto-capture
    bAutoCaptureInProgress = false;
    GEditor->GetTimerManager()->ClearTimer(AutoCaptureTimerHandle);
}

void SVCCSimPanel::LoadPredefinedPose()
{
    if (!SelectedFlashPawn.IsValid())
    {
        UE_LOG(LogTemp, Warning, TEXT("No FlashPawn selected"));
        return;
    }
    
    // Open file dialog to select pose file
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (DesktopPlatform)
    {
        TArray<FString> OpenFilenames;
        FString ExtensionStr = TEXT("Pose Files (*.txt)|*.txt");
        
        bool bOpened = DesktopPlatform->OpenFileDialog(
            FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr),
            TEXT("Load Pose File"),
            FPaths::ProjectSavedDir(),
            TEXT(""),
            *ExtensionStr,
            EFileDialogFlags::None,
            OpenFilenames
        );
        
        if (bOpened && OpenFilenames.Num() > 0)
        {
            FString SelectedFile = OpenFilenames[0];
            
            // Read file content
            TArray<FString> FileLines;
            if (FFileHelper::LoadFileToStringArray(FileLines, *SelectedFile))
            {
                TArray<FVector> Positions;
                TArray<FRotator> Rotations;
                
                for (const FString& Line : FileLines)
                {
                    if (Line.IsEmpty() || Line.StartsWith(TEXT("#")))
                    {
                        continue; // Skip comments and empty lines
                    }
                    
                    // Parse line - Expected format: X Y Z Qx Qy Qz Qw
                    TArray<FString> Values;
                    Line.ParseIntoArray(Values, TEXT(" "), true);
                    
                    if (Values.Num() >= 7)
                    {
                        float X = FCString::Atof(*Values[0]);
                        float Y = FCString::Atof(*Values[1]);
                        float Z = FCString::Atof(*Values[2]);
                        float Qx = FCString::Atof(*Values[3]);
                        float Qy = FCString::Atof(*Values[4]);
                        float Qz = FCString::Atof(*Values[5]);
                        float Qw = FCString::Atof(*Values[6]);
                        
                        Positions.Add(FVector(X, Y, Z));
    
                        FQuat Quaternion(Qx, Qy, Qz, Qw);
                        Quaternion.Normalize();
                        FRotator Rotation = Quaternion.Rotator();
                        Rotations.Add(Rotation);
                    }
                }
                
                if (Positions.Num() > 0 && Positions.Num() == Rotations.Num())
                {
                    // Set the path on the FlashPawn
                    SelectedFlashPawn->SetPathPanel(Positions, Rotations);
                    
                    // Update NumPoses
                    NumPoses = Positions.Num();
                    NumPosesValue = NumPoses;
                    
                    // Clean up any existing visualization
                    HidePathVisualization();
                    
                    // Allow path visualization after loading
                    bPathVisualized = false;
                    bPathNeedsUpdate = false;
                    
                    UE_LOG(LogTemp, Log, TEXT("Successfully loaded %d poses from file"), Positions.Num());
                }
                else
                {
                    UE_LOG(LogTemp, Warning, TEXT("Failed to parse pose file:"
                                                  " Invalid format or empty file"));
                }
            }
            else
            {
                UE_LOG(LogTemp, Warning, TEXT("Failed to load file"));
            }
        }
    }
}

void SVCCSimPanel::SaveGeneratedPose()
{
    if (!SelectedFlashPawn.IsValid())
    {
        UE_LOG(LogTemp, Warning, TEXT("No FlashPawn selected"));
        return;
    }
    
    // Check if there are poses to save
    int32 PoseCount = SelectedFlashPawn->GetPoseCount();
    if (PoseCount <= 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("No poses to save"));
        return;
    }
    
    // Open file dialog to select save location
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (DesktopPlatform)
    {
        TArray<FString> SaveFilenames;
        FString ExtensionStr = TEXT("Pose Files (*.txt)|*.txt");
        
        bool bSaved = DesktopPlatform->SaveFileDialog(
            FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr),
            TEXT("Save Pose File"),
            FPaths::ProjectSavedDir(),
            TEXT("poses.txt"),
            *ExtensionStr,
            EFileDialogFlags::None,
            SaveFilenames
        );
        
        if (bSaved && SaveFilenames.Num() > 0)
        {
            FString SelectedFile = SaveFilenames[0];
            
            // Ensure the file has .txt extension
            if (!SelectedFile.EndsWith(TEXT(".txt")))
            {
                SelectedFile += TEXT(".txt");
            }
            
            // Get positions and rotations from FlashPawn
            TArray<FVector> Positions;
            TArray<FRotator> Rotations;
            SelectedFlashPawn->GetCurrentPath(Positions, Rotations);
            
            // Build file content with header comments
            FString FileContent;
            FileContent += TEXT("# UE coordinate system poses (left-handed, cm)\n");
            FileContent += TEXT("# Coordinate axes: +X forward, +Y right, +Z up\n");
            FileContent += TEXT("# Format: X Y Z Qx Qy Qz Qw\n");
            FileContent += TEXT("# Quaternion order: [x, y, z, w] (UE format, scalar last)\n");
            
            for (int32 i = 0; i < Positions.Num(); ++i)
            {
                const FVector& Pos = Positions[i];
                const FRotator& Rot = Rotations[i];
                
                // Convert rotator to quaternion
                FQuat Quat = Rot.Quaternion();
                
                // Format: X Y Z Qx Qy Qz Qw
                FileContent += FString::Printf(
                    TEXT("%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n"),
                    Pos.X, Pos.Y, Pos.Z,
                    Quat.X, Quat.Y, Quat.Z, Quat.W
                );
            }
            
            // Save to file
            if (!FFileHelper::SaveStringToFile(FileContent, *SelectedFile))
            {
                UE_LOG(LogTemp, Warning, TEXT("Failed to save file"));
            }
        }
    }
}

FReply SVCCSimPanel::OnLoadPoseClicked()
{
    LoadPredefinedPose();
    return FReply::Handled();
}

FReply SVCCSimPanel::OnSavePoseClicked()
{
    SaveGeneratedPose();
    return FReply::Handled();
}

// ============================================================================
// PATH VISUALIZATION
// ============================================================================

FReply SVCCSimPanel::OnTogglePathVisualizationClicked()
{
    if (!SelectedFlashPawn.IsValid())
    {
        return FReply::Handled();
    }
    
    // Toggle the visualization state
    bPathVisualized = !bPathVisualized;

    if (bPathVisualized)
    {
        UE_LOG(LogTemp, Warning, TEXT("Showing path visualization..."));
        ShowPathVisualization();
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("Hiding path visualization..."));
        HidePathVisualization();
    }

    VisualizePathButton->SetButtonStyle(bPathVisualized ? 
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") : 
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
    
    return FReply::Handled();
}

void SVCCSimPanel::UpdatePathVisualization()
{
    const TArray<FVector> Positions = SelectedFlashPawn->PendingPositions;
    const TArray<FRotator> Rotations = SelectedFlashPawn->PendingRotations;

    if (Positions.Num() == 0 || Positions.Num() != Rotations.Num())
    {
        bPathVisualized = false;
        return;
    }

    PathVisualizationActor = UTrajectoryViewer::GenerateVisibleElements(
        GEditor->GetEditorWorldContext().World(),
        Positions,
        Rotations,
        nullptr, // PathMaterial
        nullptr, // CameraMaterial  
        5.f,     // Path width
        15.0f,   // Cone size
        75.0f    // Cone length
    );
        
    if (!PathVisualizationActor.IsValid())
    {
        bPathVisualized = false;
        return;
    }

    PathVisualizationActor->Tags.Add(FName("NotSMActor"));
    bPathNeedsUpdate = false;
}

void SVCCSimPanel::ShowPathVisualization()
{
    if (SelectedFlashPawn.IsValid())
    {
        UpdatePathVisualization();
    }
}

void SVCCSimPanel::HidePathVisualization()
{
    if (GEditor && GEditor->GetEditorWorldContext().World())
    {
        UWorld* World = GEditor->GetEditorWorldContext().World();
        FlushPersistentDebugLines(World);
        
        // Clean up any PathVisualization actors in the world
        for (TActorIterator<AActor> ActorIterator(World); ActorIterator; ++ActorIterator)
        {
            AActor* Actor = *ActorIterator;
            if (Actor && (Actor->GetActorLabel().Contains(TEXT("PathVisualization")) || 
                         Actor->Tags.Contains(FName("VCCSimPathViz"))))
            {
                World->DestroyActor(Actor);
            }
        }
    }
    
    if (PathVisualizationActor.IsValid() && GEditor)
    {
        if (UWorld* World = GEditor->GetEditorWorldContext().World())
        {
            World->DestroyActor(PathVisualizationActor.Get());
        }
        PathVisualizationActor.Reset();
    }
}

// ============================================================================
// IMAGE CAPTURE OPERATIONS
// ============================================================================

FReply SVCCSimPanel::OnCaptureImagesClicked()
{
    CaptureImageFromCurrentPose();
    return FReply::Handled();
}

void SVCCSimPanel::CaptureImageFromCurrentPose()
{
    if (!SelectedFlashPawn.IsValid())
    {
        UE_LOG(LogTemp, Warning, TEXT("No FlashPawn selected"));
        return;
    }
    
    // Create a directory for saving images if it doesn't exist yet
    if (SaveDirectory.IsEmpty())
    {
        SaveDirectory = FPaths::ProjectSavedDir() / TEXT("VCCSimCaptures")
        / GetTimestampedFilename();
        IFileManager::Get().MakeDirectory(*SaveDirectory, true);
    }
    
    // Check if the FlashPawn is ready to capture
    if (SelectedFlashPawn->IsReady())
    {
        // Pose index for filename
        int32 PoseIndex = SelectedFlashPawn->GetCurrentIndex();
        
        // Track if any cameras were captured
        bool bAnyCaptured = false;
        
        // Capture with RGB cameras if enabled
        if (bUseRGBCamera && bHasRGBCamera)
        {
            SaveRGB(PoseIndex, bAnyCaptured);
        }
        
        // Capture with Depth cameras if enabled
        if (bUseDepthCamera && bHasDepthCamera)
        {
            SaveDepth(PoseIndex, bAnyCaptured);
        }
        
        // Capture with Normal cameras if enabled
        if (bUseNormalCamera && bHasNormalCamera)
        {
            SaveNormal(PoseIndex, bAnyCaptured);
        }
        
        // Capture with Segmentation cameras if enabled
        if (bUseSegmentationCamera && bHasSegmentationCamera)
        {
            SaveSeg(PoseIndex, bAnyCaptured);
        }
        
        // Log if no images were captured
        if (!bAnyCaptured)
        {
            UE_LOG(LogTemp, Warning, TEXT("No images captured. "
                                          "Ensure cameras are enabled."));
        }
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("FlashPawn not ready for capture. "
                                      "Wait for it to reach position."));
    }
}

void SVCCSimPanel::SaveRGB(int32 PoseIndex, bool& bAnyCaptured)
{
    if (!SelectedFlashPawn.IsValid())
    {
        return;
    }
    
    // Get the RGB cameras
    TArray<URGBCameraComponent*> RGBCameras;
    SelectedFlashPawn->GetComponents<URGBCameraComponent>(RGBCameras);
    *JobNum += RGBCameras.Num();
    
    // Get the editor viewport
    FEditorViewportClient* ViewportClient = nullptr;
    for (FLevelEditorViewportClient* LevelVC : GEditor->GetLevelViewportClients())
    {
        if (LevelVC && LevelVC->Viewport && !LevelVC->IsOrtho())
        {
            ViewportClient = LevelVC;
            break;
        }
    }
    
    if (!ViewportClient)
    {
        UE_LOG(LogTemp, Error, TEXT("No valid editor viewport found"));
        *JobNum -= RGBCameras.Num();
        return;
    }
    
    for (int32 i = 0; i < RGBCameras.Num(); ++i)
    {
        URGBCameraComponent* Camera = RGBCameras[i];
        if (Camera && Camera->IsActive())
        {
            // Get camera index or use iterator index
            int32 CameraIndex = Camera->GetCameraIndex();
            if (CameraIndex < 0) CameraIndex = i;
            
            // Filename for this camera
            FString Filename = SaveDirectory / FString::Printf(
                TEXT("RGB_Cam%02d_Pose%03d.png"), 
                CameraIndex, 
                PoseIndex
            );
            
            FIntPoint CameraSize = {Camera->GetImageSize().first,
                Camera->GetImageSize().second};
            FTransform CameraTransform = Camera->GetComponentTransform();
            
            ViewportClient->SetViewLocation(CameraTransform.GetLocation());
            ViewportClient->SetViewRotation(CameraTransform.GetRotation().Rotator());
            ViewportClient->ViewFOV = 67.38f;
            ViewportClient->Invalidate();
            ViewportClient->Viewport->Draw();
            
            // Setup high resolution screenshot
            FHighResScreenshotConfig& HighResScreenshotConfig = GetHighResScreenshotConfig();
            HighResScreenshotConfig.SetResolution(CameraSize.X, CameraSize.Y);
            HighResScreenshotConfig.SetFilename(Filename);
            HighResScreenshotConfig.bMaskEnabled = false;
            HighResScreenshotConfig.bCaptureHDR = false;
            
            FScreenshotRequest::RequestScreenshot(Filename, false, false);
            *JobNum -= 1;
            
            bAnyCaptured = true;
        }
        else
        {
            *JobNum -= 1;
        }
    }
}

void SVCCSimPanel::SaveDepth(int32 PoseIndex, bool& bAnyCaptured)
{
    if (!SelectedFlashPawn.IsValid())
    {
        return;
    }
    
    TArray<UDepthCameraComponent*> DepthCameras;
    SelectedFlashPawn->GetComponents<UDepthCameraComponent>(DepthCameras);
    *JobNum += DepthCameras.Num();

    for (int32 i = 0; i < DepthCameras.Num(); ++i)
    {
        UDepthCameraComponent* Camera = DepthCameras[i];
        if (Camera && Camera->IsActive())
        {
            // Get camera index or use iterator index
            int32 CameraIndex = Camera->GetCameraIndex();
            if (CameraIndex < 0) CameraIndex = i;
            
            FString DepthFilename = SaveDirectory / FString::Printf(
                TEXT("Depth16_Cam%02d_Pose%03d.png"), 
                CameraIndex, 
                PoseIndex
            );
            
            // Filename for this camera's point cloud
            FString PLYFilename = SaveDirectory / FString::Printf(
                TEXT("PointCloud_Cam%02d_Pose%03d.ply"), 
                CameraIndex, 
                PoseIndex
            );
            
            // Capture point cloud data
            // Camera->AsyncGetPointCloudData(
            //     [this, Camera, PLYFilename, DepthFilename, PoseIndex, CameraIndex]()
            //     {
            //         // Process the points in a background thread
            //         AsyncTask(ENamedThreads::AnyBackgroundHiPriTask,
            //             [this, Camera, PLYFilename, DepthFilename, PoseIndex, CameraIndex]()
            //             {
            //                 try
            //                 {
            //                     // Generate point cloud
            //                     TArray<FDCPoint> PointCloud = Camera->GeneratePointCloud();
            //                     
            //                     if (PointCloud.Num() > 0)
            //                     {
            //                         // Save point cloud to PLY file asynchronously
            //                         (new FAutoDeleteAsyncTask<FAsyncPLYSaveTask>(
            //                             PointCloud, 
            //                             PLYFilename))
            //                         ->StartBackgroundTask();
            //                     }
            //                     
            //                     Camera->AsyncGetDepthImageData(
            //                         [DepthFilename, Camera](const TArray<FFloat16Color>& ImageData)
            //                         {
            //                             FIntPoint Size = {Camera->GetImageSize().first, Camera->GetImageSize().second};
            //                             float DepthScale = 1.0f;
            //                             
            //                             (new FAutoDeleteAsyncTask<FAsyncDepth16SaveTask>(
            //                                 ImageData, 
            //                                 Size, 
            //                                 DepthFilename,
            //                                 DepthScale))
            //                             ->StartBackgroundTask();
            //                         });
            //                 }
            //                 catch (...)
            //                 {
            //                     // Handle any exceptions silently
            //                 }
            //                 
            //                 // Decrement job counter on game thread
            //                 AsyncTask(ENamedThreads::GameThread, [this]()
            //                 {
            //                     *JobNum -= 1;
            //                 });
            //             });
            //     });
            
            FIntPoint Size = {Camera->GetImageSize().first,
                Camera->GetImageSize().second};
            
            Camera->AsyncGetDepthImageData(
           [DepthFilename, Size, JobNum = this->JobNum]
           (const TArray<FFloat16Color>& ImageData)
           {
               float DepthScale = 1.0f;
    
               (new FAutoDeleteAsyncTask<FAsyncDepth16SaveTask>(
                   ImageData, 
                   Size, 
                   DepthFilename, 
                   DepthScale))
               ->StartBackgroundTask();
    
               *JobNum -= 1;
           });
            
            bAnyCaptured = true;
        }
        else
        {
            *JobNum -= 1;
        }
    }
}

void SVCCSimPanel::SaveSeg(int32 PoseIndex, bool& bAnyCaptured)
{
    TArray<USegmentationCameraComponent*> SegmentationCameras;
    SelectedFlashPawn->GetComponents<USegmentationCameraComponent>(SegmentationCameras);
    *JobNum += SegmentationCameras.Num();

    for (int32 i = 0; i < SegmentationCameras.Num(); ++i)
    {
        USegmentationCameraComponent* Camera = SegmentationCameras[i];
        if (Camera && Camera->IsActive())
        {
            // Get camera index or use iterator index
            int32 CameraIndex = Camera->GetCameraIndex();
            if (CameraIndex < 0) CameraIndex = i;
                    
            // Filename for this camera
            FString Filename = SaveDirectory / FString::Printf(
                TEXT("Seg_Cam%02d_Pose%03d.png"), 
                CameraIndex, 
                PoseIndex
            );
                    
            // Capture the image
            FIntPoint Size = {Camera->GetImageSize().first, Camera->GetImageSize().second};
                    
            // Get image data and save asynchronously
            Camera->AsyncGetSegmentationImageData(
                [Filename, Size, JobNum = this->JobNum](TArray<FColor> ImageData)
                {
                    for (FColor& Color : ImageData)
                    {
                        Color.A = 255; // Ensure alpha is set to 255
                    }
                    (new FAutoDeleteAsyncTask<FAsyncImageSaveTask>(ImageData, Size, Filename))
                    ->StartBackgroundTask();
                    *JobNum -= 1;
                });
                    
            bAnyCaptured = true;
        }
    }
}

void SVCCSimPanel::SaveNormal(int32 PoseIndex, bool& bAnyCaptured)
{
    if (!SelectedFlashPawn.IsValid())
    {
        return;
    }
    
    TArray<UNormalCameraComponent*> NormalCameras;
    SelectedFlashPawn->GetComponents<UNormalCameraComponent>(NormalCameras);
    *JobNum += NormalCameras.Num();

    for (int32 i = 0; i < NormalCameras.Num(); ++i)
    {
        UNormalCameraComponent* Camera = NormalCameras[i];
        if (Camera && Camera->IsActive())
        {
            // Get camera index or use iterator index
            int32 CameraIndex = Camera->GetCameraIndex();
            if (CameraIndex < 0) CameraIndex = i;
            
            // Generate filename for EXR format
            FString NormalEXRFilename = SaveDirectory / FString::Printf(
                TEXT("Normal_Cam%02d_Pose%03d.exr"), 
                CameraIndex, 
                PoseIndex
            );
            
            FIntPoint Size = {Camera->GetImageSize().first, Camera->GetImageSize().second};
            
            // Save high precision normals as EXR
            Camera->AsyncGetNormalImageData(
                [NormalEXRFilename, Size, JobNum = this->JobNum]
                (const TArray<FLinearColor>& NormalData)
                {
                    (new FAutoDeleteAsyncTask<FAsyncNormalEXRSaveTask>(
                        NormalData, 
                        Size, 
                        NormalEXRFilename))
                    ->StartBackgroundTask();
                    
                    *JobNum -= 1;
                });
            
            bAnyCaptured = true;
        }
        else
        {
            *JobNum -= 1;
        }
    }
}

void SVCCSimPanel::StartAutoCapture()
{
    if (!SelectedFlashPawn.IsValid())
    {
        UE_LOG(LogTemp, Warning, TEXT("No FlashPawn selected"));
        return;
    }
    
    // Create a directory for saving images
    SaveDirectory = FPaths::ProjectSavedDir() / TEXT("VCCSimCaptures") /
        GetTimestampedFilename();
    IFileManager::Get().MakeDirectory(*SaveDirectory, true);
    
    // Start the capture process
    bAutoCaptureInProgress = true;
    *JobNum = 0;

    SelectedFlashPawn->MoveTo(0);
    
    // Set up a timer to check if the FlashPawn is ready for capture
    GEditor->GetTimerManager()->SetTimer(
        AutoCaptureTimerHandle,
        [this]()
        {
            if (!bAutoCaptureInProgress || !SelectedFlashPawn.IsValid())
            {
                // Stop the timer if auto-capture is cancelled or FlashPawn is invalid
                GEditor->GetTimerManager()->ClearTimer(AutoCaptureTimerHandle);
                bAutoCaptureInProgress = false;
                
                // Reset button style to original color
                if (AutoCaptureButton.IsValid())
                {
                    AutoCaptureButton->SetButtonStyle(&FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
                }
                return;
            }
            
            // Check if the FlashPawn is ready to capture
            if (SelectedFlashPawn->IsReady())
            {
                CaptureImageFromCurrentPose();
                SelectedFlashPawn->MoveToNext();
                
                // If we've finished capturing all poses, stop the auto-capture
                if (SelectedFlashPawn->GetCurrentIndex() == NumPoses - 1)
                {
                    SaveDirectory.Empty(); // Reset for next capture session
                    bAutoCaptureInProgress = false;
                    GEditor->GetTimerManager()->ClearTimer(AutoCaptureTimerHandle);
                    
                    // Reset button style to original color
                    if (AutoCaptureButton.IsValid())
                    {
                        AutoCaptureButton->SetButtonStyle(&FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
                    }
                }
            }
            else if (*JobNum == 0)
            {
                SelectedFlashPawn->MoveForward();
            }
        },
        0.2f,
        true
    );
}

void SVCCSimPanel::StopAutoCapture()
{
    if (bAutoCaptureInProgress)
    {
        bAutoCaptureInProgress = false;
        
        if (GEditor)
        {
            GEditor->GetTimerManager()->ClearTimer(AutoCaptureTimerHandle);
        }
        
        SaveDirectory.Empty(); // Reset for next capture session
        UE_LOG(LogTemp, Log, TEXT("Auto-capture stopped by user"));
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

FString SVCCSimPanel::GetTimestampedFilename()
{
    FDateTime Now = FDateTime::Now();
    return FString::Printf(TEXT("%04d-%02d-%02d_%02d-%02d-%02d"),
        Now.GetYear(), Now.GetMonth(), Now.GetDay(),
        Now.GetHour(), Now.GetMinute(), Now.GetSecond());
}

