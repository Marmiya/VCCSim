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

DEFINE_LOG_CATEGORY_STATIC(LogPathImageCapture, Log, All);

#include "Editor/Panels/VCCSimPanelPathImageCapture.h"
#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Utils/VCCSimConfigManager.h"
#include "Utils/PathGenerator.h"
#include "Utils/ImageCaptureService.h"
#include "Pawns/FlashPawn.h"
#include "Pawns/SimLookAtPath.h"
#include "Sensors/RGBCamera.h"
#include "Sensors/DepthCamera.h"
#include "Components/PrimitiveComponent.h"
#include "Selection.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Views/STableRow.h"
#include "Styling/AppStyle.h"
#include "Styling/CoreStyle.h"
#include "Utils/TrajectoryViewer.h"
#include "DesktopPlatformModule.h"
#include "EngineUtils.h"
#include "LevelEditorViewport.h"
#include "Async/Async.h"

static bool EnsureGameView()
{
    for (FLevelEditorViewportClient* Client : GEditor->GetLevelViewportClients())
    {
        if (Client && Client->IsPerspective())
        {
            if (!Client->IsInGameView())
            {
                Client->SetGameView(true);
                return true;
            }
            return false;
        }
    }
    return false;
}

static void RestoreGameView(bool bWasChangedByUs)
{
    if (!bWasChangedByUs) return;
    for (FLevelEditorViewportClient* Client : GEditor->GetLevelViewportClients())
    {
        if (Client && Client->IsPerspective())
        {
            Client->SetGameView(false);
            return;
        }
    }
}

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

FVCCSimPanelPathImageCapture::FVCCSimPanelPathImageCapture()
{
    PathGenerator = MakeShared<FPathGenerator>();
    // ImageCaptureService is initialized in SetSelectionManager, as it depends on it.

    OrbitMarginValue      = OrbitMargin;
    OrbitStartHeightValue = OrbitStartHeight;
    OrbitCameraHFOVValue  = OrbitCameraHFOV;
    OrbitHOverlapValue    = OrbitHOverlap;
    OrbitVOverlapValue    = OrbitVOverlap;
    OrbitNadirAltValue    = OrbitNadirAlt;
    OrbitNadirTiltValue   = OrbitNadirTiltAngle;
}

FVCCSimPanelPathImageCapture::~FVCCSimPanelPathImageCapture()
{
    Cleanup();
}

// ============================================================================
// INITIALIZATION
// ============================================================================

void FVCCSimPanelPathImageCapture::Initialize()
{
    LoadOrbitActorList();
}

void FVCCSimPanelPathImageCapture::Cleanup()
{
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
}

void FVCCSimPanelPathImageCapture::SetSelectionManager(TSharedPtr<FVCCSimPanelSelection> InSelectionManager)
{
    SelectionManager = InSelectionManager;
    ImageCaptureService = MakeShared<FImageCaptureService>(InSelectionManager);
}

// ============================================================================
// POSE GENERATION AND MANAGEMENT
// ============================================================================

FReply FVCCSimPanelPathImageCapture::OnGeneratePosesClicked()
{
    if (bGenerationInProgress) return FReply::Handled();

    bGenerationInProgress = true;
    HidePathVisualization();
    bPathVisualized = false;
    bPathNeedsUpdate = false;

    GeneratePosesAroundTarget();
    return FReply::Handled();
}

FReply FVCCSimPanelPathImageCapture::OnAddOrbitActorsClicked()
{
    USelection* Sel = GEditor ? GEditor->GetSelectedActors() : nullptr;
    if (!Sel) return FReply::Handled();

    bool bAdded = false;
    for (int32 i = 0; i < Sel->Num(); ++i)
    {
        AActor* Actor = Cast<AActor>(Sel->GetSelectedObject(i));
        if (!Actor) continue;

        const FString Label = Actor->GetActorLabel();
        bool bDuplicate = OrbitActorListItems.ContainsByPredicate(
            [&Label](const TSharedPtr<FString>& P) { return P.IsValid() && *P == Label; });

        if (!bDuplicate)
        {
            OrbitActorListItems.Add(MakeShareable(new FString(Label)));
            bAdded = true;
        }
    }

    if (bAdded && OrbitActorListView.IsValid())
        OrbitActorListView->RequestListRefresh();
    if (bAdded)
        SaveOrbitActorList();

    return FReply::Handled();
}

FBox FVCCSimPanelPathImageCapture::ComputeCombinedBounds() const
{
    UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
    if (!World) return FBox(ForceInit);

    TMap<FString, AActor*> LabelMap;
    for (TActorIterator<AActor> It(World); It; ++It)
    {
        if (AActor* A = *It)
            LabelMap.Add(A->GetActorLabel(), A);
    }

    FBox Combined(ForceInit);
    for (const TSharedPtr<FString>& LabelPtr : OrbitActorListItems)
    {
        if (!LabelPtr.IsValid()) continue;
        AActor** Found = LabelMap.Find(*LabelPtr);
        if (!Found || !*Found) continue;

        TArray<UPrimitiveComponent*> Prims;
        (*Found)->GetComponents<UPrimitiveComponent>(Prims);
        for (UPrimitiveComponent* Prim : Prims)
        {
            if (!Prim || !Prim->IsRegistered()) continue;
            Combined += Prim->CalcBounds(Prim->GetComponentTransform()).GetBox();
        }
    }
    return Combined;
}

void FVCCSimPanelPathImageCapture::GeneratePosesAroundTarget()
{
    auto FailCleanup = [this](const TCHAR* Msg)
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("%s"), Msg);
        bGenerationInProgress = false;
    };

    if (!SelectionManager.IsValid() || !SelectionManager.Pin()->GetSelectedFlashPawn().IsValid())
        return FailCleanup(TEXT("No FlashPawn selected"));
    
    if (OrbitActorListItems.IsEmpty())
        return FailCleanup(TEXT("No target actors in orbit list"));
    
    UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
    if (!World)
        return FailCleanup(TEXT("Editor world is not available"));

    FPathGenerator::FConformalOrbitParams Params;
    Params.TargetBounds = ComputeCombinedBounds();
    if (!Params.TargetBounds.IsValid)
        return FailCleanup(TEXT("Could not compute valid bounds from selected actors"));

    for (const TSharedPtr<FString>& Label : OrbitActorListItems)
    {
        if (!Label.IsValid()) continue;
        for (TActorIterator<AActor> It(World); It; ++It)
        {
            if (It->GetActorLabel() == *Label)
            {
                It->GetComponents<UPrimitiveComponent>(Params.TargetPrimitives);
                break;
            }
        }
    }
    
    Params.World = World;
    Params.Margin = OrbitMargin;
    Params.StartHeight = OrbitStartHeight;
    Params.CameraHFOV = OrbitCameraHFOV;
    Params.HOverlap = OrbitHOverlap;
    Params.VOverlap = OrbitVOverlap;
    Params.bIncludeNadir = bOrbitIncludeNadir;
    Params.NadirAltitude = OrbitNadirAlt;
    Params.NadirTiltAngle = OrbitNadirTiltAngle;

    TArray<URGBCameraComponent*> RGBCameras;
    SelectionManager.Pin()->GetSelectedFlashPawn()->GetComponents<URGBCameraComponent>(RGBCameras);
    if (RGBCameras.Num() > 0)
    {
        Params.CameraResolution = RGBCameras[0]->FOV;
    }

    TWeakObjectPtr<AFlashPawn> FlashPawnWeak = SelectionManager.Pin()->GetSelectedFlashPawn();
    TWeakObjectPtr<AVCCSimLookAtPath> LookAtWeak  = SelectionManager.Pin()->GetSelectedLookAtPath();
    
    PathGenerator->GenerateConformalOrbit(Params, FPathGenerator::FOnPathGenerated::CreateLambda(
        [this, FlashPawnWeak, LookAtWeak](const FPathGenerator::FGeneratedPath& Path)
        {
            bGenerationInProgress = false;
            if (!FlashPawnWeak.IsValid())
            {
                UE_LOG(LogPathImageCapture, Warning, TEXT("FlashPawn became invalid after path generation."));
                return;
            }

            FlashPawnWeak->SetPathPanel(Path.Positions, Path.Rotations);
            FlashPawnWeak->MoveTo(0);

            if (AVCCSimLookAtPath* LookAt = LookAtWeak.Get())
            {
                if (LookAt->Spline)
                {
                    LookAt->Spline->ClearSplinePoints(false);
                    for (int32 i = 0; i < Path.Positions.Num(); ++i)
                    {
                        LookAt->Spline->AddSplinePoint(Path.Positions[i], ESplineCoordinateSpace::World, false);
                        LookAt->Spline->SetSplinePointType(i, ESplinePointType::Linear, false);
                    }
                    LookAt->Spline->UpdateSpline();
                    LookAt->FreeOrientations = Path.Rotations;
                    LookAt->OrientationMode  = EOrientationMode::FreeOrientation;
                    if (LookAt->TargetPoint)
                    {
                        FBox Bounds = ComputeCombinedBounds();
                        if(Bounds.IsValid)
                            LookAt->TargetPoint->SetWorldLocation(Bounds.GetCenter());
                    }
                }
            }

            UE_LOG(LogPathImageCapture, Log, TEXT("Conformal orbit generated with %d poses."), Path.Positions.Num());
        }
    ));
}

FReply FVCCSimPanelPathImageCapture::OnLoadPoseClicked()
{
    LoadPredefinedPose();
    return FReply::Handled();
}

FReply FVCCSimPanelPathImageCapture::OnSavePoseClicked()
{
    SaveGeneratedPose();
    return FReply::Handled();
}

void FVCCSimPanelPathImageCapture::LoadPredefinedPose()
{
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    if (SelectionManager.IsValid())
    {
        SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    }
    
    if (!SelectedFlashPawn.IsValid())
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("No FlashPawn selected"));
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
                        continue;
                    }

                    TArray<FString> Values;
                    Line.ParseIntoArray(Values, TEXT(" "), true);

                    if (Values.Num() == 8)
                    {
                        float X = FCString::Atof(*Values[1]);
                        float Y = FCString::Atof(*Values[2]);
                        float Z = FCString::Atof(*Values[3]);
                        float Qx = FCString::Atof(*Values[4]);
                        float Qy = FCString::Atof(*Values[5]);
                        float Qz = FCString::Atof(*Values[6]);
                        float Qw = FCString::Atof(*Values[7]);

                        Positions.Add(FVector(X, Y, Z));

                        FQuat Quaternion(Qx, Qy, Qz, Qw);
                        Quaternion.Normalize();
                        FRotator Rotation = Quaternion.Rotator();
                        Rotations.Add(Rotation);
                    }
                    else
                    {
                        UE_LOG(LogPathImageCapture, Warning, TEXT("Invalid pose line format (expected 8 values): %s"), *Line);
                    }
                }
                
                if (Positions.Num() > 0 && Positions.Num() == Rotations.Num())
                {
                    // Set the path on the FlashPawn
                    SelectedFlashPawn->SetPathPanel(Positions, Rotations);
                    
                    // Clean up any existing visualization
                    HidePathVisualization();
                    
                    // Allow path visualization after loading
                    bPathVisualized = false;
                    bPathNeedsUpdate = false;
                    
                    UE_LOG(LogPathImageCapture, Log, TEXT("Successfully loaded %d "
                                              "poses from file"), Positions.Num());
                }
                else
                {
                    UE_LOG(LogPathImageCapture, Warning, TEXT("Failed to parse pose file: "
                                                  "Invalid format or empty file"));
                }
            }
            else
            {
                UE_LOG(LogPathImageCapture, Warning, TEXT("Failed to load file"));
            }
        }
    }
}

void FVCCSimPanelPathImageCapture::SaveGeneratedPose()
{
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    if (SelectionManager.IsValid())
    {
        SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    }
    
    if (!SelectedFlashPawn.IsValid())
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("No FlashPawn selected"));
        return;
    }
    
    // Check if there are poses to save
    int32 PoseCount = SelectedFlashPawn->GetPoseCount();
    if (PoseCount <= 0)
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("No poses to save"));
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
            WritePosesToFile(Positions, Rotations, SelectedFile);
        }
    }
}

void FVCCSimPanelPathImageCapture::WritePosesToFile(
    const TArray<FVector>& Positions,
    const TArray<FRotator>& Rotations,
    const FString& FilePath)
{
    FString FileContent;
    FileContent += TEXT("# UE coordinate system poses (left-handed, cm)\n");
    FileContent += TEXT("# Coordinate axes: +X forward, +Y right, +Z up\n");
    FileContent += TEXT("# Format: Timestamp X Y Z Qx Qy Qz Qw\n");
    FileContent += TEXT("# Quaternion order: [x, y, z, w] (UE format, scalar last)\n");
    FileContent += TEXT("# Timestamp: Sequential pseudo timestamps for pose ordering\n");

    for (int32 i = 0; i < Positions.Num(); ++i)
    {
        const FQuat Quat = Rotations[i].Quaternion();
        FileContent += FString::Printf(
            TEXT("%.1f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n"),
            static_cast<double>(i),
            Positions[i].X, Positions[i].Y, Positions[i].Z,
            Quat.X, Quat.Y, Quat.Z, Quat.W
        );
    }

    if (!FFileHelper::SaveStringToFile(FileContent, *FilePath))
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("Failed to save poses to %s"), *FilePath);
    }
}

// ============================================================================
// PATH VISUALIZATION
// ============================================================================

FReply FVCCSimPanelPathImageCapture::OnTogglePathVisualizationClicked()
{
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    if (SelectionManager.IsValid())
    {
        SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    }
    
    if (!SelectedFlashPawn.IsValid())
    {
        return FReply::Handled();
    }
    
    // Toggle the visualization state
    bPathVisualized = !bPathVisualized;

    if (bPathVisualized)
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("Showing path visualization..."));
        ShowPathVisualization();
    }
    else
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("Hiding path visualization..."));
        HidePathVisualization();
    }

    VisualizePathButton->SetButtonStyle(bPathVisualized ? 
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") : 
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
    
    return FReply::Handled();
}

void FVCCSimPanelPathImageCapture::UpdatePathVisualization()
{
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    if (SelectionManager.IsValid())
    {
        SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    }
    
    if (!SelectedFlashPawn.IsValid()) return;
    
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

void FVCCSimPanelPathImageCapture::ShowPathVisualization()
{
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    if (SelectionManager.IsValid())
    {
        SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    }
    
    if (SelectedFlashPawn.IsValid())
    {
        UpdatePathVisualization();
    }
}

void FVCCSimPanelPathImageCapture::HidePathVisualization()
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

FReply FVCCSimPanelPathImageCapture::OnCaptureImagesClicked()
{
    const bool bChanged = EnsureGameView();
    CaptureImageFromCurrentPose();
    RestoreGameView(bChanged);
    return FReply::Handled();
}

void FVCCSimPanelPathImageCapture::CaptureImageFromCurrentPose()
{
    if (!ImageCaptureService.IsValid())
    {
        UE_LOG(LogPathImageCapture, Error, TEXT("ImageCaptureService is not valid."));
        return;
    }
    
    // Create a directory for saving images if it doesn't exist yet
    if (SaveDirectory.IsEmpty())
    {
        SaveDirectory = FPaths::ProjectSavedDir() / TEXT("VCCSimCaptures") / GetTimestampedFilename();
        IFileManager::Get().MakeDirectory(*SaveDirectory, true);
    }

    bool bAnyCaptured = false;
    int32 PoseIndex = -1;
    if (SelectionManager.IsValid() && SelectionManager.Pin()->GetSelectedFlashPawn().IsValid())
    {
        PoseIndex = SelectionManager.Pin()->GetSelectedFlashPawn()->GetCurrentIndex();
    }
    
    ImageCaptureService->CaptureImageFromCurrentPose(PoseIndex, SaveDirectory, bAnyCaptured);
}

void FVCCSimPanelPathImageCapture::StartAutoCapture()
{
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    if (SelectionManager.IsValid())
    {
        SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    }
    
    if (!SelectedFlashPawn.IsValid())
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("No FlashPawn selected"));
        return;
    }
    
    // Create a directory for saving images
    SaveDirectory = FPaths::ProjectSavedDir() / TEXT("VCCSimCaptures") / GetTimestampedFilename();
    IFileManager::Get().MakeDirectory(*SaveDirectory, true);

    TArray<FVector> Positions;
    TArray<FRotator> Rotations;
    SelectedFlashPawn->GetCurrentPath(Positions, Rotations);
    WritePosesToFile(Positions, Rotations, SaveDirectory / TEXT("poses.txt"));

    bGameViewChangedForCapture = EnsureGameView();
    bAutoCaptureInProgress = true;

    SelectedFlashPawn->MoveTo(0);
    
    // Set up a timer to check if the FlashPawn is ready for capture
    GEditor->GetTimerManager()->SetTimer(
        AutoCaptureTimerHandle,
        [this]()
        {
            TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
            if (SelectionManager.IsValid())
            {
                SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
            }
            
            if (!bAutoCaptureInProgress || !SelectedFlashPawn.IsValid())
            {
                GEditor->GetTimerManager()->ClearTimer(AutoCaptureTimerHandle);
                bAutoCaptureInProgress = false;
                RestoreGameView(bGameViewChangedForCapture);
                bGameViewChangedForCapture = false;
                if (AutoCaptureButton.IsValid())
                {
                    AutoCaptureButton->SetButtonStyle(&FAppStyle::Get().
                        GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
                }
                return;
            }
            
            // Check if the FlashPawn is ready to capture
            if (SelectedFlashPawn->IsReady())
            {
                CaptureImageFromCurrentPose();
                SelectedFlashPawn->MoveToNext();
                
                TArray<FVector> PathPos; TArray<FRotator> PathRot;
                SelectedFlashPawn->GetCurrentPath(PathPos, PathRot);
                if (SelectedFlashPawn->GetCurrentIndex() == PathPos.Num() - 1)
                {
                    SaveDirectory.Empty();
                    bAutoCaptureInProgress = false;
                    GEditor->GetTimerManager()->ClearTimer(AutoCaptureTimerHandle);
                    RestoreGameView(bGameViewChangedForCapture);
                    bGameViewChangedForCapture = false;
                    if (AutoCaptureButton.IsValid())
                    {
                        AutoCaptureButton->SetButtonStyle(&FAppStyle::Get().
                            GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
                    }
                }
            }
            else
            {
                SelectedFlashPawn->MoveForward();
            }
        },
        0.2f,
        true
    );
}

void FVCCSimPanelPathImageCapture::StopAutoCapture()
{
    if (bAutoCaptureInProgress)
    {
        bAutoCaptureInProgress = false;
        if (GEditor)
            GEditor->GetTimerManager()->ClearTimer(AutoCaptureTimerHandle);
        RestoreGameView(bGameViewChangedForCapture);
        bGameViewChangedForCapture = false;
        SaveDirectory.Empty();
        UE_LOG(LogPathImageCapture, Log, TEXT("Auto-capture stopped by user"));
    }
}

// ============================================================================
// UTILITIES
// ============================================================================

FString FVCCSimPanelPathImageCapture::GetTimestampedFilename()
{
    FDateTime Now = FDateTime::Now();
    return FString::Printf(TEXT("%04d-%02d-%02d_%02d-%02d-%02d"),
        Now.GetYear(), Now.GetMonth(), Now.GetDay(),
        Now.GetHour(), Now.GetMinute(), Now.GetSecond());
}

void FVCCSimPanelPathImageCapture::SaveOrbitActorList()
{
    FVCCSimConfigManager::FPathImageCaptureConfig Config = FVCCSimConfigManager::Get().GetPathImageCaptureConfig();
    Config.OrbitActorLabels.Empty();
    for (const TSharedPtr<FString>& Label : OrbitActorListItems)
    {
        if (Label.IsValid())
            Config.OrbitActorLabels.Add(*Label);
    }
    FVCCSimConfigManager::Get().SetPathImageCaptureConfig(Config);
}

void FVCCSimPanelPathImageCapture::LoadOrbitActorList()
{
    const auto& Config = FVCCSimConfigManager::Get().GetPathImageCaptureConfig();
    OrbitActorListItems.Empty();
    for (const FString& Label : Config.OrbitActorLabels)
        OrbitActorListItems.Add(MakeShareable(new FString(Label)));
    if (OrbitActorListView.IsValid())
        OrbitActorListView->RequestListRefresh();
}

void FVCCSimPanelPathImageCapture::LoadFromConfigManager()
{
    LoadOrbitActorList();
}