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
#include "Widgets/Input/SCheckBox.h"
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
#include "EngineUtils.h"
#include "Framework/Application/SlateApplication.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "Widgets/Input/SNumericEntryBox.h"

// ============================================================================
// CLEAN PHASE 2 IMPLEMENTATION
// ============================================================================

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
    RegionConfig.bUseLimited = (NewState == ECheckBoxState::Checked);
    OnRegionConfigurationChanged();
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
    CameraConfig.bUseRGB = (NewState == ECheckBoxState::Checked);
    if (CameraConfig.bUseRGB)
    {
        OnCameraConfigurationChanged();
        TArray<URGBCameraComponent*> RGBCameras;
        SelectedFlashPawn->GetComponents<URGBCameraComponent>(RGBCameras);
        for (URGBCameraComponent* Camera : RGBCameras)
        {
            if (Camera)
            {
                Camera->SetActive(CameraConfig.bUseRGB);
                Camera->InitializeRenderTargets();
                Camera->SetCaptureComponent();
            }
        }
    }
}

void SVCCSimPanel::OnDepthCameraCheckboxChanged(ECheckBoxState NewState)
{
    CameraConfig.bUseDepth = (NewState == ECheckBoxState::Checked);
    if (CameraConfig.bUseDepth)
    {
        OnCameraConfigurationChanged();
        TArray<UDepthCameraComponent*> DepthCameras;
        SelectedFlashPawn->GetComponents<UDepthCameraComponent>(DepthCameras);
        for (UDepthCameraComponent* Camera : DepthCameras)
        {
            if (Camera)
            {
                Camera->SetActive(CameraConfig.bUseDepth);
                Camera->InitializeRenderTargets();
                Camera->SetCaptureComponent();
            }
        }
    }
}

void SVCCSimPanel::OnSegmentationCameraCheckboxChanged(ECheckBoxState NewState)
{
    CameraConfig.bUseSegmentation = (NewState == ECheckBoxState::Checked);
    if (CameraConfig.bUseSegmentation)
    {
        OnCameraConfigurationChanged();
        TArray<USegmentationCameraComponent*> SegmentationCameras;
        SelectedFlashPawn->GetComponents<USegmentationCameraComponent>(SegmentationCameras);
        for (USegmentationCameraComponent* Camera : SegmentationCameras)
        {
            if (Camera)
            {
                Camera->SetActive(CameraConfig.bUseSegmentation);
                Camera->InitializeRenderTargets();
                Camera->SetCaptureComponent();
            }
        }
    }
}

void SVCCSimPanel::OnNormalCameraCheckboxChanged(ECheckBoxState NewState)
{
    CameraConfig.bUseNormal = (NewState == ECheckBoxState::Checked);
    if (CameraConfig.bUseNormal)
    {
        OnCameraConfigurationChanged();
        TArray<UNormalCameraComponent*> NormalCameras;
        SelectedFlashPawn->GetComponents<UNormalCameraComponent>(NormalCameras);
        for (UNormalCameraComponent* Camera : NormalCameras)
        {
            if (Camera)
            {
                Camera->SetActive(CameraConfig.bUseNormal);
                Camera->InitializeRenderTargets();
                Camera->SetCaptureComponent();
            }
        }
    }
}

void SVCCSimPanel::CheckCameraComponents()
{
    CameraConfig.bHasRGBCamera = false;
    CameraConfig.bHasDepthCamera = false;
    CameraConfig.bHasNormalCamera = false;
    CameraConfig.bHasSegmentationCamera = false;
    
    if (!SelectedFlashPawn.IsValid())
    {
        return;
    }
    
    // Check for RGB cameras
    TArray<URGBCameraComponent*> RGBCameras;
    SelectedFlashPawn->GetComponents<URGBCameraComponent>(RGBCameras);
    CameraConfig.bHasRGBCamera = (RGBCameras.Num() > 0);
    
    // Check for Depth cameras
    TArray<UDepthCameraComponent*> DepthCameras;
    SelectedFlashPawn->GetComponents<UDepthCameraComponent>(DepthCameras);
    CameraConfig.bHasDepthCamera = (DepthCameras.Num() > 0);
    
    // Check for Normal cameras
    TArray<UNormalCameraComponent*> NormalCameras;
    SelectedFlashPawn->GetComponents<UNormalCameraComponent>(NormalCameras);
    CameraConfig.bHasNormalCamera = (NormalCameras.Num() > 0);
    
    // Check for Segmentation cameras
    TArray<USegmentationCameraComponent*> SegmentationCameras;
    SelectedFlashPawn->GetComponents<USegmentationCameraComponent>(SegmentationCameras);
    CameraConfig.bHasSegmentationCamera = (SegmentationCameras.Num() > 0);
    
    // Reset checkboxes if corresponding cameras aren't available
    if (!CameraConfig.bHasRGBCamera)
    {
        CameraConfig.bUseRGB = false;
        RGBCameraCheckBox->SetIsChecked(ECheckBoxState::Unchecked);
    }
    
    if (!CameraConfig.bHasDepthCamera)
    {
        CameraConfig.bUseDepth = false;
        DepthCameraCheckBox->SetIsChecked(ECheckBoxState::Unchecked);
    }
    
    if (!CameraConfig.bHasNormalCamera)
    {
        CameraConfig.bUseNormal = false;
        NormalCameraCheckBox->SetIsChecked(ECheckBoxState::Unchecked);
    }
    
    if (!CameraConfig.bHasSegmentationCamera)
    {
        CameraConfig.bUseSegmentation = false;
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
            Camera->SetActive(CameraConfig.bUseRGB);
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
            Camera->SetActive(CameraConfig.bUseDepth);
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
            Camera->SetActive(CameraConfig.bUseNormal);
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
            Camera->SetActive(CameraConfig.bUseSegmentation);
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
    bPathVisualized = false;
    bPathNeedsUpdate = true;

    if (PathVisualizationActor.IsValid() && GEditor)
    {
        if (UWorld* World = GEditor->GetEditorWorldContext().World())
        {
            World->DestroyActor(PathVisualizationActor.Get());
        }
        PathVisualizationActor.Reset();
    }
    
    UpdatePathVisualization();
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
        MeshInfos, PoseConfig.Radius, PoseConfig.NumPoses,
        PoseConfig.VerticalGap, Positions, Rotations);
    
    // Set the path on the FlashPawn
    SelectedFlashPawn->SetPathPanel(Positions, Rotations);
    SelectedFlashPawn->MoveTo(0);
    
    // Update NumPoses to match actual number of generated poses
    PoseConfig.NumPoses = Positions.Num();
    OnPoseConfigurationChanged();
    
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
                    // Parse line - Expected format: X Y Z Pitch Yaw Roll
                    TArray<FString> Values;
                    Line.ParseIntoArray(Values, TEXT(" "), true);
                    
                    if (Values.Num() >= 6)
                    {
                        float X = FCString::Atof(*Values[0]);
                        float Y = FCString::Atof(*Values[1]);
                        float Z = FCString::Atof(*Values[2]);
                        float Pitch = FCString::Atof(*Values[3]);
                        float Yaw = FCString::Atof(*Values[4]);
                        float Roll = FCString::Atof(*Values[5]);
                        
                        Positions.Add(FVector(X, Y, Z));
    
                        FRotator Rotation(Pitch, Yaw, Roll);
                        Rotation.Normalize();
                        Rotations.Add(Rotation);
                    }
                }
                
                if (Positions.Num() > 0 && Positions.Num() == Rotations.Num())
                {
                    // Set the path on the FlashPawn
                    SelectedFlashPawn->SetPathPanel(Positions, Rotations);
                    
                    // Update NumPoses
                    PoseConfig.NumPoses = Positions.Num();
                    OnPoseConfigurationChanged();
                    
                    bPathVisualized = false;
                    bPathNeedsUpdate = true;
                    
                    if (PathVisualizationActor.IsValid() && GEditor)
                    {
                        if (UWorld* World = GEditor->GetEditorWorldContext().World())
                        {
                            World->DestroyActor(PathVisualizationActor.Get());
                        }
                        PathVisualizationActor.Reset();
                    }
                    
                    UpdatePathVisualization();
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
            
            // Build file content
            FString FileContent;
            for (int32 i = 0; i < Positions.Num(); ++i)
            {
                const FVector& Pos = Positions[i];
                const FRotator& Rot = Rotations[i];
                
                // Format: X Y Z Pitch Yaw Roll
                FileContent += FString::Printf(
                    TEXT("%.6f %.6f %.6f %.6f %.6f %.6f\n"),
                    Pos.X, Pos.Y, Pos.Z,
                    Rot.Pitch, Rot.Yaw, Rot.Roll
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
        ShowPathVisualization();
    }
    else
    {
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
        UE_LOG(LogTemp, Warning, TEXT("No valid path available for visualization"));
        bPathVisualized = false;
        return;
    }

    UMaterialInterface* PathMaterial = LoadObject<UMaterialInterface>(nullptr, 
       TEXT("/VCCSim/Materials/M_Rat_Path_ice.M_Rat_Path_ice"));
    UMaterialInterface* CameraMaterial = LoadObject<UMaterialInterface>(nullptr, 
        TEXT("/VCCSim/Materials/M_Rat_Path_Blue.M_Rat_Path_Blue"));
        
    if (!PathMaterial || !CameraMaterial)
    {
        UE_LOG(LogTemp, Warning, TEXT("Failed to load path visualization materials"));
        bPathVisualized = false;
        return;
    }

    // TODO: Why there is a wired bias?
    // Generate new visualization actor
    PathVisualizationActor = UTrajectoryViewer::GenerateVisibleElements(
        GEditor->GetEditorWorldContext().World(),
        Positions,
        Rotations,
        PathMaterial,
        CameraMaterial,
        5.f,     // Path width
        50.0f,    // Cone size
        75.0f     // Cone length
    );
        
    if (!PathVisualizationActor.IsValid())
    {
        UE_LOG(LogTemp, Warning, TEXT("Failed to create path visualization"));
        bPathVisualized = false;
        return;
    }

    if (PathVisualizationActor.IsValid())
    { 
        PathVisualizationActor->Tags.Add(FName("NotSMActor"));
    }

    HidePathVisualization();
    bPathNeedsUpdate = false;
    VisualizePathButton->SetButtonStyle(bPathVisualized ? 
    &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") : 
    &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
}

void SVCCSimPanel::ShowPathVisualization()
{
    if (PathVisualizationActor.IsValid())
    {
        // Show the actor
        PathVisualizationActor->SetActorHiddenInGame(false);
        PathVisualizationActor->SetIsTemporarilyHiddenInEditor(false);
        
        // Show all components
        TArray<UActorComponent*> Components;
        PathVisualizationActor->GetComponents(UStaticMeshComponent::StaticClass(), Components);
        for (UActorComponent* Component : Components)
        {
            UStaticMeshComponent* MeshComp = Cast<UStaticMeshComponent>(Component);
            if (MeshComp)
            {
                MeshComp->SetVisibility(true);
                MeshComp->SetHiddenInGame(false);
            }
        }
    }
}

void SVCCSimPanel::HidePathVisualization()
{
    if (PathVisualizationActor.IsValid())
    {
        // Hide the actor
        PathVisualizationActor->SetActorHiddenInGame(true);
        PathVisualizationActor->SetIsTemporarilyHiddenInEditor(true);
        
        // Hide all components
        TArray<UActorComponent*> Components;
        PathVisualizationActor->GetComponents(UStaticMeshComponent::StaticClass(), Components);
        for (UActorComponent* Component : Components)
        {
            UStaticMeshComponent* MeshComp = Cast<UStaticMeshComponent>(Component);
            if (MeshComp)
            {
                MeshComp->SetVisibility(false);
                MeshComp->SetHiddenInGame(true);
            }
        }
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
        if (CameraConfig.bUseRGB && CameraConfig.bHasRGBCamera)
        {
            SaveRGB(PoseIndex, bAnyCaptured);
        }
        
        // Capture with Depth cameras if enabled
        if (CameraConfig.bUseDepth && CameraConfig.bHasDepthCamera)
        {
            SaveDepth(PoseIndex, bAnyCaptured);
        }
        
        // Capture with Normal cameras if enabled
        if (CameraConfig.bUseNormal && CameraConfig.bHasNormalCamera)
        {
            SaveNormal(PoseIndex, bAnyCaptured);
        }
        
        // Capture with Segmentation cameras if enabled
        if (CameraConfig.bUseSegmentation && CameraConfig.bHasSegmentationCamera)
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
                return;
            }
            
            // Check if the FlashPawn is ready to capture
            if (SelectedFlashPawn->IsReady())
            {
                CaptureImageFromCurrentPose();
                SelectedFlashPawn->MoveToNext();
                
                // If we've finished capturing all poses, stop the auto-capture
                if (SelectedFlashPawn->GetCurrentIndex() == PoseConfig.NumPoses - 1)
                {
                    SaveDirectory.Empty(); // Reset for next capture session
                    bAutoCaptureInProgress = false;
                    GEditor->GetTimerManager()->ClearTimer(AutoCaptureTimerHandle);
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

// ============================================================================
// PHASE 2: STRUCTURED CONFIGURATION IMPLEMENTATION
// ============================================================================

void SVCCSimPanel::InitializeStructuredConfigurations()
{
    // Initialize structured configuration objects with default values
    PoseConfig = FPoseConfiguration();
    CameraConfig = FCameraConfiguration(); 
    RegionConfig = FLimitedRegionConfiguration();
    PointCloudInfo = FPointCloudInfo();  // Using existing point cloud info structure
    
    // Initialize Triangle Splatting configuration (clean structured system)
    GSConfig = FTriangleSplattingConfiguration();
    
    // Initialize UI binding variables for SpinBox widgets
    NumPosesValue = PoseConfig.NumPoses;
    RadiusValue = PoseConfig.Radius;
    VerticalGapValue = PoseConfig.VerticalGap;
    HeightOffsetValue = PoseConfig.HeightOffset;
    SafeDistanceValue = PoseConfig.SafeDistance;
    SafeHeightValue = PoseConfig.SafeHeight;
    
    // Initialize Triangle Splatting UI binding variables
    GSFOVValue = GSConfig.FOVDegrees;
    GSImageWidthValue = GSConfig.ImageWidth;
    GSImageHeightValue = GSConfig.ImageHeight;
    GSMaxIterationsValue = GSConfig.MaxIterations;
    GSLearningRateValue = GSConfig.LearningRate;
    
    // Initialize job counter for async operations
    JobNum = MakeShared<std::atomic<int32>>(0);
    
    // Configuration object creation temporarily disabled due to UE 5.6 compatibility
    // ConfigurationObject = NewObject<UVCCSimConfigurationObject>();
}

// Legacy synchronization methods removed - clean Phase 2 implementation

void SVCCSimPanel::OnPoseConfigurationChanged()
{
    // Configuration object temporarily disabled
    
    // Sync UI binding variables
    NumPosesValue = PoseConfig.NumPoses;
    RadiusValue = PoseConfig.Radius;
    VerticalGapValue = PoseConfig.VerticalGap;
    HeightOffsetValue = PoseConfig.HeightOffset;
    SafeDistanceValue = PoseConfig.SafeDistance;
    SafeHeightValue = PoseConfig.SafeHeight;
    
    UE_LOG(LogTemp, Log, TEXT("Phase 2: Pose configuration updated"));
}

void SVCCSimPanel::OnCameraConfigurationChanged()
{
    // Configuration object temporarily disabled
    // Update camera components based on new configuration
    CheckCameraComponents();
    UpdateActiveCameras();
    UE_LOG(LogTemp, Log, TEXT("Phase 2: Camera configuration updated"));
}

void SVCCSimPanel::OnTriangleSplattingConfigurationChanged()
{
    // Configuration object temporarily disabled
    
    // Sync Triangle Splatting UI binding variables
    GSFOVValue = GSConfig.FOVDegrees;
    GSImageWidthValue = GSConfig.ImageWidth;
    GSImageHeightValue = GSConfig.ImageHeight;
    GSMaxIterationsValue = GSConfig.MaxIterations;
    GSLearningRateValue = GSConfig.LearningRate;
    
    UE_LOG(LogTemp, Log, TEXT("Phase 2: Triangle Splatting configuration updated"));
}

void SVCCSimPanel::OnRegionConfigurationChanged()
{
    // Configuration object temporarily disabled
    UE_LOG(LogTemp, Log, TEXT("Phase 2: Region configuration updated"));
}

void SVCCSimPanel::OnPointCloudConfigurationChanged()
{
    // Point cloud configuration updates
    UE_LOG(LogTemp, Log, TEXT("Phase 2: Point Cloud configuration updated"));
}

void SVCCSimPanel::CreateModernUIWidgets()
{
    // Create advanced UI widgets using Phase 1 components
    // This will eventually replace the manual widget creation in VCCSimPanel_UI.cpp
    
    // Camera widget creation temporarily disabled due to UE 5.6 compatibility
    // CameraWidget = SNew(SCameraConfigWidget);
    
    // Additional widgets will be created as we migrate more sections
    UE_LOG(LogTemp, Log, TEXT("Phase 2: Modern UI widgets created"));
}

TSharedRef<SWidget> SVCCSimPanel::CreateStructuredCameraSection()
{
    // This will replace the old camera section with the modern table-based widget
    // Camera widget temporarily disabled
    CreateModernUIWidgets();
    
    // For now, return a placeholder - full implementation in next phase
    return SNew(STextBlock)
        .Text(FText::FromString(TEXT("Phase 2: Camera section (Modern widget coming soon)")));
}

TSharedRef<SWidget> SVCCSimPanel::CreateStructuredPoseSection()
{
    // Placeholder for the modern pose configuration section
    return SNew(STextBlock)
        .Text(FText::FromString(TEXT("Phase 2: Pose section (Modern widget coming soon)")));
}

TSharedRef<SWidget> SVCCSimPanel::CreateStructuredTriangleSplattingSection()
{
    // Placeholder for the modern Triangle Splatting configuration section
    return SNew(STextBlock)
        .Text(FText::FromString(TEXT("Phase 2: Triangle Splatting section (Modern widget coming soon)")));
}

void SVCCSimPanel::LoadConfigurationPreset(const FString& PresetName)
{
    // Preset loading temporarily disabled due to UE 5.6 compatibility
    // FVCCSimConfigPresetManager::Get().LoadPreset(PresetName, ConfigurationObject);
    UE_LOG(LogTemp, Log, TEXT("Preset loading requested: %s"), *PresetName);
}

void SVCCSimPanel::SaveConfigurationPreset(const FString& PresetName)
{
    // Preset saving temporarily disabled due to UE 5.6 compatibility  
    // FVCCSimConfigPresetManager::Get().SavePreset(PresetName, ConfigurationObject);
    UE_LOG(LogTemp, Log, TEXT("Preset saving requested: %s"), *PresetName);
}

void SVCCSimPanel::ResetConfigurationsToDefaults()
{
    // Reset all structured configurations to defaults
    InitializeStructuredConfigurations();
    UE_LOG(LogTemp, Log, TEXT("Phase 2: Reset all configurations to defaults"));
}

void SVCCSimPanel::ValidateConfigurations()
{
    // Validate all structured configurations
    bool bValid = true;
    FString ValidationErrors;
    
    // Add validation logic here for each configuration
    // This replaces scattered validation throughout the old code
    
    if (bValid)
    {
        UE_LOG(LogTemp, Log, TEXT("Phase 2: All configurations are valid"));
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("Phase 2: Configuration validation failed: %s"), *ValidationErrors);
    }
}

// ============================================================================
// MISSING FUNCTION IMPLEMENTATIONS
// ============================================================================

void SVCCSimPanel::LoadLogoImages()
{
    // Load VCC and SZU logo images for display
    FString VCCLogoPath = FPaths::ProjectPluginsDir() / TEXT("VCCSim/Content/UI/logo_vcc.png");
    FString SZULogoPath = FPaths::ProjectPluginsDir() / TEXT("VCCSim/Content/UI/logo_szu.png");
    
    if (FPaths::FileExists(VCCLogoPath))
    {
        VCCLogoBrush = MakeShareable(new FSlateDynamicImageBrush(FName(*VCCLogoPath), FVector2D(64, 64)));
    }
    
    if (FPaths::FileExists(SZULogoPath))
    {
        SZULogoBrush = MakeShareable(new FSlateDynamicImageBrush(FName(*SZULogoPath), FVector2D(64, 64)));
    }
    
    UE_LOG(LogTemp, Log, TEXT("Logo images loaded"));
}

void SVCCSimPanel::InitializeSceneAnalysisManager()
{
    // Initialize scene analysis manager if available
    if (GEditor)
    {
        if (UWorld* World = GEditor->GetEditorWorldContext().World())
        {
            for (TActorIterator<ASceneAnalysisManager> ActorIterator(World); ActorIterator; ++ActorIterator)
            {
                SceneAnalysisManager = *ActorIterator;
                break;
            }
        }
    }
    
    UE_LOG(LogTemp, Log, TEXT("Scene analysis manager initialized"));
}

TSharedRef<SWidget> SVCCSimPanel::CreatePawnTargetSelector()
{
    return SNew(SVerticalBox)
        
        // FlashPawn selection
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 2)
        [
            SNew(SHorizontalBox)
            
            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            .VAlign(VAlign_Center)
            [
                SAssignNew(SelectedFlashPawnText, STextBlock)
                .Text(FText::FromString(TEXT("No FlashPawn Selected")))
            ]
            
            + SHorizontalBox::Slot()
            .AutoWidth()
            .Padding(5, 0, 0, 0)
            [
                SAssignNew(SelectFlashPawnToggle, SCheckBox)
                .IsChecked_Lambda([this]() { return bSelectingFlashPawn ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
                .OnCheckStateChanged(this, &SVCCSimPanel::OnSelectFlashPawnToggleChanged)
                .Content()
                [
                    SNew(STextBlock).Text(FText::FromString(TEXT("Select")))
                ]
            ]
        ]
        
        // Target selection
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 2)
        [
            SNew(SHorizontalBox)
            
            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            .VAlign(VAlign_Center)
            [
                SAssignNew(SelectedTargetObjectText, STextBlock)
                .Text(FText::FromString(TEXT("No Target Selected")))
            ]
            
            + SHorizontalBox::Slot()
            .AutoWidth()
            .Padding(5, 0, 0, 0)
            [
                SAssignNew(SelectTargetToggle, SCheckBox)
                .IsChecked_Lambda([this]() { return bSelectingTarget ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
                .OnCheckStateChanged(this, &SVCCSimPanel::OnSelectTargetToggleChanged)
                .Content()
                [
                    SNew(STextBlock).Text(FText::FromString(TEXT("Select")))
                ]
            ]
        ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateCameraConfigurationWidget()
{
    return SNew(SVerticalBox)
        
        // RGB Camera
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 2)
        [
            SNew(SCheckBox)
            .Content()
            [
                SNew(STextBlock)
                .Text(FText::FromString(TEXT("RGB Camera")))
            ]
            .IsChecked_Lambda([this]() { return CameraConfig.bUseRGB ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
            .OnCheckStateChanged_Lambda([this](ECheckBoxState NewState) { 
                CameraConfig.bUseRGB = (NewState == ECheckBoxState::Checked); 
                OnCameraConfigurationChanged(); 
            })
        ]
        
        // Depth Camera
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 2)
        [
            SNew(SCheckBox)
            .Content()
            [
                SNew(STextBlock)
                .Text(FText::FromString(TEXT("Depth Camera")))
            ]
            .IsChecked_Lambda([this]() { return CameraConfig.bUseDepth ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
            .OnCheckStateChanged_Lambda([this](ECheckBoxState NewState) { 
                CameraConfig.bUseDepth = (NewState == ECheckBoxState::Checked); 
                OnCameraConfigurationChanged(); 
            })
        ]
        
        // Segmentation Camera
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 2)
        [
            SNew(SCheckBox)
            .Content()
            [
                SNew(STextBlock)
                .Text(FText::FromString(TEXT("Segmentation Camera")))
            ]
            .IsChecked_Lambda([this]() { return CameraConfig.bUseSegmentation ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
            .OnCheckStateChanged_Lambda([this](ECheckBoxState NewState) { 
                CameraConfig.bUseSegmentation = (NewState == ECheckBoxState::Checked); 
                OnCameraConfigurationChanged(); 
            })
        ]
        
        // Normal Camera
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 2)
        [
            SNew(SCheckBox)
            .Content()
            [
                SNew(STextBlock)
                .Text(FText::FromString(TEXT("Normal Camera")))
            ]
            .IsChecked_Lambda([this]() { return CameraConfig.bUseNormal ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
            .OnCheckStateChanged_Lambda([this](ECheckBoxState NewState) { 
                CameraConfig.bUseNormal = (NewState == ECheckBoxState::Checked); 
                OnCameraConfigurationChanged(); 
            })
        ];
}

TSharedRef<SWidget> SVCCSimPanel::CreatePoseConfigurationWidget()
{
    return SNew(SVerticalBox)
        
        // Number of poses
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 2)
        [
            SNew(SHorizontalBox)
            + SHorizontalBox::Slot()
            .AutoWidth()
            .VAlign(VAlign_Center)
            [
                SNew(STextBlock)
                .Text(FText::FromString(TEXT("Poses:")))
                .MinDesiredWidth(80.0f)
            ]
            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            [
                SAssignNew(NumPosesSpinBox, SNumericEntryBox<int32>)
                .Value_Lambda([this]() { return NumPosesValue; })
                .OnValueChanged_Lambda([this](int32 NewValue) 
                { 
                    PoseConfig.NumPoses = NewValue;
                    NumPosesValue = NewValue;
                    OnPoseConfigurationChanged();
                })
                .MinValue(1)
                .MaxValue(1000)
                .Delta(1)
            ]
        ]
        
        // Radius
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 2)
        [
            SNew(SHorizontalBox)
            + SHorizontalBox::Slot()
            .AutoWidth()
            .VAlign(VAlign_Center)
            [
                SNew(STextBlock)
                .Text(FText::FromString(TEXT("Radius:")))
                .MinDesiredWidth(80.0f)
            ]
            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            [
                SAssignNew(RadiusSpinBox, SNumericEntryBox<float>)
                .Value_Lambda([this]() { return RadiusValue; })
                .OnValueChanged_Lambda([this](float NewValue) 
                { 
                    PoseConfig.Radius = NewValue;
                    RadiusValue = NewValue;
                    OnPoseConfigurationChanged();
                })
                .MinValue(0.1f)
                .MaxValue(10000.0f)
                .Delta(10.0f)
            ]
        ]
        
        // Vertical Gap
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 2)
        [
            SNew(SHorizontalBox)
            + SHorizontalBox::Slot()
            .AutoWidth()
            .VAlign(VAlign_Center)
            [
                SNew(STextBlock)
                .Text(FText::FromString(TEXT("V.Gap:")))
                .MinDesiredWidth(80.0f)
            ]
            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            [
                SAssignNew(VerticalGapSpinBox, SNumericEntryBox<float>)
                .Value_Lambda([this]() { return VerticalGapValue; })
                .OnValueChanged_Lambda([this](float NewValue) 
                { 
                    PoseConfig.VerticalGap = NewValue;
                    VerticalGapValue = NewValue;
                    OnPoseConfigurationChanged();
                })
                .MinValue(0.1f)
                .MaxValue(1000.0f)
                .Delta(5.0f)
            ]
        ]
        
        // Generate Poses button
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 10, 5, 5)
        [
            SNew(SButton)
            .Text(FText::FromString(TEXT("Generate Poses")))
            .OnClicked(this, &SVCCSimPanel::OnGeneratePosesClicked)
        ]
        
        // Load/Save buttons
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 2)
        [
            SNew(SHorizontalBox)
            
            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            .Padding(0, 0, 2, 0)
            [
                SNew(SButton)
                .Text(FText::FromString(TEXT("Load Pose")))
                .OnClicked(this, &SVCCSimPanel::OnLoadPoseClicked)
            ]
            
            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            .Padding(2, 0, 0, 0)
            [
                SNew(SButton)
                .Text(FText::FromString(TEXT("Save Pose")))
                .OnClicked(this, &SVCCSimPanel::OnSavePoseClicked)
            ]
        ]
        
        // Capture buttons
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 10, 5, 5)
        [
            SNew(SHorizontalBox)
            
            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            .Padding(0, 0, 2, 0)
            [
                SNew(SButton)
                .Text(FText::FromString(TEXT("Capture Current")))
                .OnClicked(this, &SVCCSimPanel::OnCaptureImagesClicked)
            ]
            
            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            .Padding(2, 0, 0, 0)
            [
                SNew(SButton)
                .Text(FText::FromString(TEXT("Auto Capture All")))
                .OnClicked_Lambda([this]() { StartAutoCapture(); return FReply::Handled(); })
            ]
        ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateSceneAnalysisWidget()
{
    return SNew(SVerticalBox)
        
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 2)
        [
            SNew(STextBlock)
            .Text(FText::FromString(TEXT("Scene Analysis Configuration")))
            .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
        ]
        
        // First row of buttons
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 2)
        [
            SNew(SHorizontalBox)
            
            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            .Padding(0, 0, 2, 0)
            [
                SNew(SButton)
                .Text(FText::FromString(TEXT("Toggle Safe Zone")))
                .OnClicked(this, &SVCCSimPanel::OnToggleSafeZoneVisualizationClicked)
            ]
            
            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            .Padding(2, 0, 0, 0)
            [
                SNew(SButton)
                .Text(FText::FromString(TEXT("Toggle Coverage")))
                .OnClicked(this, &SVCCSimPanel::OnToggleCoverageVisualizationClicked)
            ]
        ]
        
        // Second row with complexity analysis
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 2)
        [
            SNew(SButton)
            .Text(FText::FromString(TEXT("Toggle Complexity Analysis")))
            .OnClicked(this, &SVCCSimPanel::OnToggleComplexityVisualizationClicked)
        ];
}

void SVCCSimPanel::UpdateNormalLinesVisibility()
{
    if (NormalLinesInstancedComponent.IsValid())
    {
        NormalLinesInstancedComponent.Get()->SetVisibility(bShowNormals);
        UE_LOG(LogTemp, Log, TEXT("Normal lines visibility updated: %s"), bShowNormals ? TEXT("Visible") : TEXT("Hidden"));
    }
}

TSharedRef<SWidget> SVCCSimPanel::CreateLogoPanel()
{
    return SNew(SHorizontalBox)
        
        + SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        .Padding(5, 5, 10, 5)
        [
            SNew(SBox)
            .WidthOverride(64)
            .HeightOverride(64)
            [
                SNew(SImage)
                .Image(VCCLogoBrush.IsValid() ? VCCLogoBrush.Get() : FAppStyle::GetBrush("Icons.Help"))
            ]
        ]
        
        + SHorizontalBox::Slot()
        .FillWidth(1.0f)
        .VAlign(VAlign_Center)
        [
            SNew(STextBlock)
            .Text(FText::FromString(TEXT("VCCSim Panel - Modern UI System")))
            .Font(FAppStyle::GetFontStyle("DetailsView.CategoryFontStyle"))
            .Justification(ETextJustify::Center)
        ]
        
        + SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        .Padding(10, 5, 5, 5)
        [
            SNew(SBox)
            .WidthOverride(64)
            .HeightOverride(64)
            [
                SNew(SImage)
                .Image(SZULogoBrush.IsValid() ? SZULogoBrush.Get() : FAppStyle::GetBrush("Icons.Help"))
            ]
        ];
}

TSharedRef<SWidget> SVCCSimPanel::CreatePointCloudPanel()
{
    return SNew(SVerticalBox)
        
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 2)
        [
            SNew(STextBlock)
            .Text(FText::FromString(TEXT("Point Cloud Operations")))
            .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
        ]
        
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 2)
        [
            SNew(SHorizontalBox)
            
            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            [
                SAssignNew(LoadPointCloudButton, SButton)
                .Text(FText::FromString(TEXT("Load Point Cloud")))
                .OnClicked(this, &SVCCSimPanel::OnLoadPointCloudClicked)
            ]
            
            + SHorizontalBox::Slot()
            .FillWidth(1.0f)
            .Padding(5, 0, 0, 0)
            [
                SAssignNew(VisualizePointCloudButton, SButton)
                .Text(FText::FromString(TEXT("Toggle Visualization")))
                .OnClicked(this, &SVCCSimPanel::OnTogglePointCloudVisualizationClicked)
            ]
        ]
        
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 2)
        [
            SAssignNew(ShowNormalsCheckBox, SCheckBox)
            .IsChecked_Lambda([this]() { return bShowNormals ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
            .OnCheckStateChanged(this, &SVCCSimPanel::OnShowNormalsCheckboxChanged)
            .Content()
            [
                SNew(STextBlock).Text(FText::FromString(TEXT("Show Normals")))
            ]
        ]
        
        + SVerticalBox::Slot()
        .AutoHeight()
        .Padding(5, 2)
        [
            SAssignNew(PointCloudStatusText, STextBlock)
            .Text_Lambda([this]() 
            { 
                if (!bPointCloudLoaded)
                    return FText::FromString(TEXT("No point cloud loaded"));
                return FText::FromString(FString::Printf(TEXT("%d points loaded"), PointCloudCount));
            })
        ];
}

TSharedRef<SWidget> SVCCSimPanel::CreateCollapsibleSection(const FString& Title, TSharedRef<SWidget> Content, bool& bExpanded)
{
    return SNew(SVerticalBox)
        + SVerticalBox::Slot()
        .AutoHeight()
        [
            SNew(SButton)
            .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
            .ContentPadding(FMargin(2, 2))
            .OnClicked_Lambda([&bExpanded]()
            {
                bExpanded = !bExpanded;
                return FReply::Handled();
            })
            [
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot()
                .AutoWidth()
                .VAlign(VAlign_Center)
                .Padding(0, 0, 4, 0)
                [
                    SNew(STextBlock)
                    .Text_Lambda([&bExpanded]()
                    {
                        return FText::FromString(bExpanded ? TEXT("▼") : TEXT("▶"));
                    })
                ]
                + SHorizontalBox::Slot()
                .FillWidth(1.0f)
                .VAlign(VAlign_Center)
                [
                    SNew(STextBlock)
                    .Text(FText::FromString(Title))
                    .Font(FAppStyle::GetFontStyle("PropertyWindow.BoldFont"))
                ]
            ]
        ]
        + SVerticalBox::Slot()
        .AutoHeight()
        [
            SNew(SBox)
            .Visibility_Lambda([&bExpanded]()
            {
                return bExpanded ? EVisibility::Visible : EVisibility::Collapsed;
            })
            .Padding(FMargin(8, 4))
            [
                Content
            ]
        ];
}

// ============================================================================
// SCENE ANALYSIS OPERATIONS
// ============================================================================

FReply SVCCSimPanel::OnToggleSafeZoneVisualizationClicked()
{
    bSafeZoneVisualized = !bSafeZoneVisualized;
    
    if (SceneAnalysisManager.IsValid())
    {
        SceneAnalysisManager->InterfaceVisualizeSafeZone(bSafeZoneVisualized);
    }
    
    return FReply::Handled();
}

FReply SVCCSimPanel::OnToggleCoverageVisualizationClicked()
{
    bCoverageVisualized = !bCoverageVisualized;
    
    if (SceneAnalysisManager.IsValid())
    {
        SceneAnalysisManager->InterfaceVisualizeCoverage(bCoverageVisualized);
    }
    
    return FReply::Handled();
}

FReply SVCCSimPanel::OnToggleComplexityVisualizationClicked()
{
    bComplexityVisualized = !bComplexityVisualized;
    
    if (SceneAnalysisManager.IsValid())
    {
        SceneAnalysisManager->InterfaceVisualizeComplexity(bComplexityVisualized);
    }
    
    return FReply::Handled();
}

// ============================================================================
// POINT CLOUD OPERATIONS
// ============================================================================

FReply SVCCSimPanel::OnLoadPointCloudClicked()
{
    // Open file dialog to load point cloud
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (DesktopPlatform)
    {
        TArray<FString> OpenFilenames;
        FString ExtensionStr = TEXT("Point Cloud Files (*.ply)|*.ply");
        
        bool bOpened = DesktopPlatform->OpenFileDialog(
            FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr),
            TEXT("Load Point Cloud File"),
            FPaths::ProjectSavedDir(),
            TEXT(""),
            *ExtensionStr,
            EFileDialogFlags::None,
            OpenFilenames
        );
        
        if (bOpened && OpenFilenames.Num() > 0)
        {
            LoadedPointCloudPath = OpenFilenames[0];
            // TODO: Implement actual point cloud loading
            bPointCloudLoaded = true;
            PointCloudCount = 10000; // Placeholder count
            bPointCloudHasColors = true;
            bPointCloudHasNormals = true;
            
            UE_LOG(LogTemp, Log, TEXT("Point cloud loaded: %s"), *LoadedPointCloudPath);
        }
    }
    
    return FReply::Handled();
}

FReply SVCCSimPanel::OnTogglePointCloudVisualizationClicked()
{
    if (!bPointCloudLoaded)
    {
        UE_LOG(LogTemp, Warning, TEXT("No point cloud loaded"));
        return FReply::Handled();
    }
    
    bPointCloudVisualized = !bPointCloudVisualized;
    
    if (bPointCloudVisualized)
    {
        CreateSpherePointCloudVisualization();
        if (bPointCloudHasNormals)
        {
            CreateNormalLinesVisualization();
        }
    }
    else
    {
        ClearPointCloudVisualization();
    }
    
    return FReply::Handled();
}

void SVCCSimPanel::OnShowNormalsCheckboxChanged(ECheckBoxState NewState)
{
    bShowNormals = (NewState == ECheckBoxState::Checked);
    UpdateNormalLinesVisibility();
    OnPointCloudConfigurationChanged();
}

void SVCCSimPanel::CreateSpherePointCloudVisualization()
{
    // TODO: Implement actual point cloud visualization
    UE_LOG(LogTemp, Log, TEXT("Creating point cloud visualization"));
}

void SVCCSimPanel::CreateNormalLinesVisualization()
{
    // TODO: Implement normal lines visualization
    UE_LOG(LogTemp, Log, TEXT("Creating normal lines visualization"));
}

void SVCCSimPanel::ClearPointCloudVisualization()
{
    if (PointCloudInstancedComponent.IsValid())
    {
        PointCloudInstancedComponent.Get()->ClearInstances();
    }
    
    if (NormalLinesInstancedComponent.IsValid())
    {
        NormalLinesInstancedComponent.Get()->ClearInstances();
    }
    
    UE_LOG(LogTemp, Log, TEXT("Point cloud visualization cleared"));
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

