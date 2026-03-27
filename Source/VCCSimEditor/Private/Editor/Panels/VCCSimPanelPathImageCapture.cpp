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
#include "Utils/VCCSimUIHelpers.h"
#include "Utils/VCCSimConfigManager.h"
#include "Pawns/FlashPawn.h"
#include "Pawns/SimLookAtPath.h"
#include "Sensors/RGBCamera.h"
#include "Sensors/DepthCamera.h"
#include "Sensors/SegmentationCamera.h"
#include "Sensors/NormalCamera.h"
#include "Engine/StaticMeshActor.h"
#include "Components/PrimitiveComponent.h"
#include "Selection.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Views/STableRow.h"
#include "Styling/AppStyle.h"
#include "Styling/CoreStyle.h"
#include "Utils/ImageProcesser.h"
#include "Utils/TrajectoryViewer.h"
#include "DesktopPlatformModule.h"
#include "EngineUtils.h"
#include "HighResScreenshot.h"
#include "LevelEditorViewport.h"

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
    OrbitMarginValue      = OrbitMargin;
    OrbitStartHeightValue = OrbitStartHeight;
    OrbitCameraHFOVValue  = OrbitCameraHFOV;
    OrbitHOverlapValue    = OrbitHOverlap;
    OrbitVOverlapValue    = OrbitVOverlap;
    OrbitNadirAltValue    = OrbitNadirAlt;
    OrbitNadirCountValue  = OrbitNadirCount;
    JobNum = MakeShared<std::atomic<int32>>(0);
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
    JobNum = MakeShared<std::atomic<int32>>(0);
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
}

// ============================================================================
// UI CONSTRUCTION
// ============================================================================

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreatePathImageCapturePanel()
{
    return FVCCSimUIHelpers::CreateCollapsibleSection(
        "Path Configuration & Image Capture", 
        SNew(SVerticalBox)
        
        // Path Configuration Section
        +SVerticalBox::Slot()
        .AutoHeight()
        [
            CreatePathConfigSection()
        ]
        
        +SVerticalBox::Slot()
        .MaxHeight(1)
        .Padding(FMargin(0, 8, 0, 8))
        [
           FVCCSimUIHelpers::CreateSeparator()
        ]
        
        // Image Capture Section
        +SVerticalBox::Slot()
        .AutoHeight()
        [
            CreateImageCaptureSection()
        ],
        
        bPathImageCaptureSectionExpanded
    );
}

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreatePathConfigSection()
{
    return SNew(SVerticalBox)

    // Target actor list
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 0, 0, 2))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot().FillWidth(1.f)
        [
            SNew(SButton)
            .ButtonStyle(FAppStyle::Get(), "FlatButton.Success")
            .ContentPadding(FMargin(5, 2))
            .Text(FText::FromString(TEXT("+ Add Selected Actors")))
            .ToolTipText(FText::FromString(TEXT("Add selected viewport actors to the orbit target list")))
            .OnClicked_Lambda([this]() { return OnAddOrbitActorsClicked(); })
        ]
        +SHorizontalBox::Slot().AutoWidth().Padding(FMargin(4, 0, 0, 0))
        [
            SNew(SButton)
            .ButtonStyle(FAppStyle::Get(), "FlatButton.Danger")
            .ContentPadding(FMargin(5, 2))
            .Text(FText::FromString(TEXT("Clear All")))
            .OnClicked_Lambda([this]() -> FReply
            {
                OrbitActorListItems.Empty();
                if (OrbitActorListView.IsValid())
                    OrbitActorListView->RequestListRefresh();
                SaveOrbitActorList();
                return FReply::Handled();
            })
        ]
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 0, 0, 4))
    [
        SNew(SBox)
        .HeightOverride(80.f)
        [
            SNew(SBorder)
            .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
            .BorderBackgroundColor(FColor(10, 10, 10, 255))
            .Padding(2)
            [
                SAssignNew(OrbitActorListView, SListView<TSharedPtr<FString>>)
                .ListItemsSource(&OrbitActorListItems)
                .SelectionMode(ESelectionMode::None)
                .OnGenerateRow_Lambda([this](TSharedPtr<FString> Item,
                    const TSharedRef<STableViewBase>& Owner) -> TSharedRef<ITableRow>
                {
                    return SNew(STableRow<TSharedPtr<FString>>, Owner)
                    [
                        SNew(SHorizontalBox)
                        +SHorizontalBox::Slot().FillWidth(1.f).VAlign(VAlign_Center).Padding(FMargin(2, 0))
                        [
                            SNew(STextBlock)
                            .Text(FText::FromString(*Item))
                            .ColorAndOpacity(FLinearColor(0.8f, 0.9f, 0.8f))
                            .Font(FCoreStyle::GetDefaultFontStyle("Mono", 8))
                        ]
                        +SHorizontalBox::Slot().AutoWidth()
                        [
                            SNew(SButton)
                            .ButtonStyle(FAppStyle::Get(), "FlatButton.Danger")
                            .ContentPadding(FMargin(4, 1))
                            .Text(FText::FromString(TEXT("×")))
                            .OnClicked_Lambda([this, Item]() -> FReply
                            {
                                if (Item.IsValid())
                                {
                                    const FString S = *Item;
                                    OrbitActorListItems.RemoveAll([&S](const TSharedPtr<FString>& P)
                                    {
                                        return P.IsValid() && *P == S;
                                    });
                                    if (OrbitActorListView.IsValid())
                                        OrbitActorListView->RequestListRefresh();
                                    SaveOrbitActorList();
                                }
                                return FReply::Handled();
                            })
                        ]
                    ];
                })
            ]
        ]
    ]

    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        FVCCSimUIHelpers::CreateSeparator()
    ]

    // Orbit parameters
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 4, 0, 4))
    [
        FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
            TEXT("Margin (cm)"), OrbitMarginSpinBox, OrbitMarginValue, OrbitMargin, 50.f, 50.f)
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 0, 0, 4))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 4, 0))
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
                TEXT("H-FOV (deg)"), OrbitCameraHFOVSpinBox, OrbitCameraHFOVValue, OrbitCameraHFOV, 5.f, 5.f)
        ]
        +SHorizontalBox::Slot().FillWidth(1.f)
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
                TEXT("Start H (cm)"), OrbitStartHeightSpinBox, OrbitStartHeightValue, OrbitStartHeight, 0.f, 50.f)
        ]
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 0, 0, 4))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 4, 0))
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
                TEXT("H-Overlap"), OrbitHOverlapSpinBox, OrbitHOverlapValue, OrbitHOverlap, 0.f, 0.05f)
        ]
        +SHorizontalBox::Slot().FillWidth(1.f)
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
                TEXT("V-Overlap"), OrbitVOverlapSpinBox, OrbitVOverlapValue, OrbitVOverlap, 0.f, 0.05f)
        ]
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 0, 0, 4))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(0, 0, 4, 0))
        [
            SNew(SCheckBox)
            .IsChecked_Lambda([this]() { return bOrbitIncludeNadir ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
            .OnCheckStateChanged_Lambda([this](ECheckBoxState S) { bOrbitIncludeNadir = (S == ECheckBoxState::Checked); })
        ]
        +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 4, 0))
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowInt32(
                TEXT("Nadir Count"), OrbitNadirCountSpinBox, OrbitNadirCountValue, OrbitNadirCount, 1, 1)
        ]
        +SHorizontalBox::Slot().FillWidth(1.f)
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
                TEXT("Nadir Alt (cm)"), OrbitNadirAltSpinBox, OrbitNadirAltValue, OrbitNadirAlt, 100.f, 100.f)
        ]
    ]

    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        FVCCSimUIHelpers::CreateSeparator()
    ]

    // Load/Save Pose buttons
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 4, 0, 4))
    [
        CreatePoseFileButtons()
    ]

    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        FVCCSimUIHelpers::CreateSeparator()
    ]

    // Action buttons
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 4, 0, 2))
    [
        CreatePoseActionButtons()
    ];
}

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreateImageCaptureSection()
{
    return SNew(SVerticalBox)
    
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(0, 0, 0, 4)
    [
        CreateMovementButtons()
    ]
    
    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        FVCCSimUIHelpers::CreateSeparator()
    ]
    
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(0, 4, 0, 0)
    [
        CreateCaptureButtons()
    ];
}

// ============================================================================
// POSE GENERATION AND MANAGEMENT
// ============================================================================

FReply FVCCSimPanelPathImageCapture::OnGeneratePosesClicked()
{
    GeneratePosesAroundTarget();
    
    // Clean up any existing visualization
    HidePathVisualization();
    
    // Allow path visualization after generating poses
    bPathVisualized = false;
    bPathNeedsUpdate = false;
    
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
    if (!SelectionManager.IsValid() || !SelectionManager.Pin()->GetSelectedFlashPawn().IsValid())
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("No FlashPawn selected"));
        return;
    }

    if (OrbitActorListItems.IsEmpty())
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("No target actors. Add actors to the orbit list first."));
        return;
    }

    FBox CombinedBox = ComputeCombinedBounds();
    if (!CombinedBox.IsValid)
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("Could not compute valid bounds from selected actors."));
        return;
    }

    const FVector BoxCenter = CombinedBox.GetCenter();
    const FVector BoxExtent = CombinedBox.GetExtent();
    const float   HX        = BoxExtent.X;
    const float   HY        = BoxExtent.Y;
    const float   BoxMinZ   = CombinedBox.Min.Z + OrbitStartHeight;
    const float   BoxMaxZ   = CombinedBox.Max.Z;

    // Coverage-based ring/pose computation
    const float HFovRad  = FMath::DegreesToRadians(FMath::Max(OrbitCameraHFOV, 5.f));
    const FIntPoint CamRes = SelectionManager.Pin()->GetActiveCameraResolution();
    const float AspectRatio = (CamRes.Y > 0) ? (float)CamRes.X / CamRes.Y : 16.f / 9.f;
    const float VFovRad  = 2.f * FMath::Atan(FMath::Tan(HFovRad * 0.5f) / AspectRatio);

    const float AvgDist = (HX + HY) * 0.5f + OrbitMargin;

    const float FootprintH = 2.f * AvgDist * FMath::Tan(HFovRad * 0.5f);
    const float FootprintV = 2.f * AvgDist * FMath::Tan(VFovRad * 0.5f);

    const float StepH = FootprintH * FMath::Max(1.f - OrbitHOverlap, 0.05f);
    const float StepV = FootprintV * FMath::Max(1.f - OrbitVOverlap, 0.05f);

    const float Perimeter  = 4.f * (HX + HY + 2.f * OrbitMargin);
    const int32 PosesPerRing = FMath::Max(4, FMath::CeilToInt(Perimeter / StepH));

    const float BuildingH = FMath::Max(BoxMaxZ - BoxMinZ, 0.f);
    const int32 NumRings  = FMath::Max(1, FMath::CeilToInt(BuildingH / StepV));

    UE_LOG(LogPathImageCapture, Log,
        TEXT("Orbit coverage: HFov=%.0f VFov=%.0f AvgDist=%.0f Footprint=%.0fx%.0f Step=%.0fx%.0f -> %d rings x %d poses"),
        FMath::RadiansToDegrees(HFovRad), FMath::RadiansToDegrees(VFovRad),
        AvgDist, FootprintH, FootprintV, StepH, StepV, NumRings, PosesPerRing);

    TArray<FVector>  Positions;
    TArray<FRotator> Rotations;

    for (int32 Ring = 0; Ring < NumRings; ++Ring)
    {
        const float T = (NumRings > 1) ? (float)Ring / (NumRings - 1) : 0.5f;
        const float Z = BoxMinZ + T * BuildingH;

        for (int32 PoseIdx = 0; PoseIdx < PosesPerRing; ++PoseIdx)
        {
            const float AngleRad = FMath::DegreesToRadians(360.f * PoseIdx / PosesPerRing);
            const float CosA = FMath::Cos(AngleRad);
            const float SinA = FMath::Sin(AngleRad);

            float DistToEdge;
            if (FMath::Abs(CosA) < KINDA_SMALL_NUMBER)
                DistToEdge = HY;
            else if (FMath::Abs(SinA) < KINDA_SMALL_NUMBER)
                DistToEdge = HX;
            else
                DistToEdge = FMath::Min(HX / FMath::Abs(CosA), HY / FMath::Abs(SinA));

            const FVector CamPos(BoxCenter.X + (DistToEdge + OrbitMargin) * CosA,
                                 BoxCenter.Y + (DistToEdge + OrbitMargin) * SinA, Z);
            const FVector LookTarget(BoxCenter.X, BoxCenter.Y, CamPos.Z);
            const FVector LookDir = (LookTarget - CamPos).GetSafeNormal();

            Positions.Add(CamPos);
            Rotations.Add(LookDir.ToOrientationRotator());
        }
    }

    if (bOrbitIncludeNadir && OrbitNadirCount > 0)
    {
        const float NadirZ = BoxMaxZ + OrbitNadirAlt;
        const float NadirR = FMath::Max(HX, HY) * 0.5f;
        for (int32 i = 0; i < OrbitNadirCount; ++i)
        {
            const float AngleRad = FMath::DegreesToRadians(360.f * i / OrbitNadirCount);
            const FVector CamPos(BoxCenter.X + NadirR * FMath::Cos(AngleRad),
                                 BoxCenter.Y + NadirR * FMath::Sin(AngleRad), NadirZ);
            Positions.Add(CamPos);
            Rotations.Add(FRotator(-90.f, FMath::RadiansToDegrees(AngleRad), 0.f));
        }
    }

    // Set path on FlashPawn
    TWeakObjectPtr<AFlashPawn> FlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    FlashPawn->SetPathPanel(Positions, Rotations);
    FlashPawn->MoveTo(0);

    // Set spline on selected LookAt actor if available
    AVCCSimLookAtPath* LookAt = SelectionManager.Pin()->GetSelectedLookAtPath().Get();
    if (LookAt && LookAt->Spline)
    {
        LookAt->Spline->ClearSplinePoints(false);
        for (int32 i = 0; i < Positions.Num(); ++i)
        {
            LookAt->Spline->AddSplinePoint(Positions[i], ESplineCoordinateSpace::World, false);
            LookAt->Spline->SetSplinePointType(i, ESplinePointType::Linear, false);
        }
        LookAt->Spline->UpdateSpline();
        LookAt->FreeOrientations  = Rotations;
        LookAt->OrientationMode   = EOrientationMode::FreeOrientation;
        if (LookAt->TargetPoint)
            LookAt->TargetPoint->SetWorldLocation(BoxCenter);
    }

    bAutoCaptureInProgress = false;
    GEditor->GetTimerManager()->ClearTimer(AutoCaptureTimerHandle);

    UE_LOG(LogPathImageCapture, Log, TEXT("Bounding-box orbit: %d poses (%d rings x %d + %d nadir)"),
        Positions.Num(), NumRings, PosesPerRing, bOrbitIncludeNadir ? OrbitNadirCount : 0);
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
            
            FString FileContent;
            FileContent += TEXT("# UE coordinate system poses (left-handed, cm)\n");
            FileContent += TEXT("# Coordinate axes: +X forward, +Y right, +Z up\n");
            FileContent += TEXT("# Format: Timestamp X Y Z Qx Qy Qz Qw\n");
            FileContent += TEXT("# Quaternion order: [x, y, z, w] (UE format, scalar last)\n");
            FileContent += TEXT("# Timestamp: Sequential pseudo timestamps for pose ordering\n");

            for (int32 i = 0; i < Positions.Num(); ++i)
            {
                const FVector& Pos = Positions[i];
                const FRotator& Rot = Rotations[i];

                // Convert rotator to quaternion
                FQuat Quat = Rot.Quaternion();

                double PseudoTimestamp = static_cast<double>(i);
                FileContent += FString::Printf(
                    TEXT("%.1f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n"),
                    PseudoTimestamp,
                    Pos.X, Pos.Y, Pos.Z,
                    Quat.X, Quat.Y, Quat.Z, Quat.W
                );
            }
            
            // Save to file
            if (!FFileHelper::SaveStringToFile(FileContent, *SelectedFile))
            {
                UE_LOG(LogPathImageCapture, Warning, TEXT("Failed to save file"));
            }
        }
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
        
        auto SelectionManagerPin = SelectionManager.Pin();
        if (!SelectionManagerPin.IsValid()) return;
        
        // Capture with RGB cameras if enabled
        if (SelectionManagerPin->IsUsingRGBCamera() && SelectionManagerPin->HasRGBCamera())
        {
            UE_LOG(LogPathImageCapture, Log, TEXT("Capturing RGB camera - Using: %s, Has: %s"), 
                SelectionManagerPin->IsUsingRGBCamera() ? TEXT("Yes") : TEXT("No"),
                SelectionManagerPin->HasRGBCamera() ? TEXT("Yes") : TEXT("No"));
            SaveRGB(PoseIndex, bAnyCaptured);
        }
        
        // Capture with Depth cameras if enabled
        if (SelectionManagerPin->IsUsingDepthCamera() && SelectionManagerPin->HasDepthCamera())
        {
            SaveDepth(PoseIndex, bAnyCaptured);
        }
        
        // Capture with Normal cameras if enabled
        if (SelectionManagerPin->IsUsingNormalCamera() && SelectionManagerPin->HasNormalCamera())
        {
            SaveNormal(PoseIndex, bAnyCaptured);
        }
        
        // Capture with Segmentation cameras if enabled
        if (SelectionManagerPin->IsUsingSegmentationCamera() &&
            SelectionManagerPin->HasSegmentationCamera())
        {
            SaveSeg(PoseIndex, bAnyCaptured);
        }
        
        // Log if no images were captured
        if (!bAnyCaptured)
        {
            UE_LOG(LogPathImageCapture, Warning, TEXT("No images captured. "
                                          "Ensure cameras are enabled."));
        }
    }
    else
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("FlashPawn not ready for capture. "
                                      "Wait for it to reach position."));
    }
}

void FVCCSimPanelPathImageCapture::SaveRGB(int32 PoseIndex, bool& bAnyCaptured)
{
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    if (SelectionManager.IsValid())
    {
        SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    }
    
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
        UE_LOG(LogPathImageCapture, Error, TEXT("No valid editor viewport found"));
        *JobNum -= RGBCameras.Num();
        return;
    }
    
    for (int32 i = 0; i < RGBCameras.Num(); ++i)
    {
        URGBCameraComponent* Camera = RGBCameras[i];
        
        if (Camera)
        {
            // Ensure camera is active for capture
            if (!Camera->IsActive())
            {
                Camera->SetActive(true);
                UE_LOG(LogPathImageCapture, Log, TEXT("SaveRGB: Activated camera[%d]"), i);
            }
            // Get camera index or use iterator index
            int32 CameraIndex = Camera->GetSensorIndex();
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

void FVCCSimPanelPathImageCapture::SaveDepth(int32 PoseIndex, bool& bAnyCaptured)
{
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    if (SelectionManager.IsValid())
    {
        SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    }
    
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
        
        if (Camera)
        {
            // Ensure camera is active for capture
            if (!Camera->IsActive())
            {
                Camera->SetActive(true);
                UE_LOG(LogPathImageCapture, Log, TEXT("SaveDepth: Activated camera[%d]"), i);
            }
            // Get camera index or use iterator index
            int32 CameraIndex = Camera->GetSensorIndex();
            if (CameraIndex < 0) CameraIndex = i;
            
            FString DepthFilename = SaveDirectory / FString::Printf(
                TEXT("Depth16_Cam%02d_Pose%03d.png"), 
                CameraIndex, 
                PoseIndex
            );
            
            FIntPoint Size = {Camera->GetImageSize().first,
                Camera->GetImageSize().second};

            Camera->AsyncGetDepthImageData(
           [DepthFilename, Size, JobNum = this->JobNum]
           (const TArray<FFloat16Color>& ImageData)
           {
               TArray<float> DepthValues;
               DepthValues.SetNum(ImageData.Num());
               for (int32 idx = 0; idx < ImageData.Num(); ++idx)
               {
                   DepthValues[idx] = ImageData[idx].A;
               }

               (new FAutoDeleteAsyncTask<FAsyncDepthSaveTask>
                   (DepthValues, Size, DepthFilename))->StartBackgroundTask();

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

void FVCCSimPanelPathImageCapture::SaveSeg(int32 PoseIndex, bool& bAnyCaptured)
{
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    if (SelectionManager.IsValid())
    {
        SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    }
    
    if (!SelectedFlashPawn.IsValid())
    {
        return;
    }
    
    TArray<USegCameraComponent*> SegmentationCameras;
    SelectedFlashPawn->GetComponents<USegCameraComponent>(SegmentationCameras);
    *JobNum += SegmentationCameras.Num();

    for (int32 i = 0; i < SegmentationCameras.Num(); ++i)
    {
        USegCameraComponent* Camera = SegmentationCameras[i];
        
        if (Camera)
        {
            // Ensure camera is active for capture
            if (!Camera->IsActive())
            {
                Camera->SetActive(true);
                UE_LOG(LogPathImageCapture, Log, TEXT("SaveSeg: Activated camera[%d]"), i);
            }
            // Get camera index or use iterator index
            int32 CameraIndex = Camera->GetSensorIndex();
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
                    (new FAutoDeleteAsyncTask<FAsyncImageSaveTask>(ImageData, Size, Filename))
                    ->StartBackgroundTask();
                    *JobNum -= 1;
                });
                    
            bAnyCaptured = true;
        }
    }
}

void FVCCSimPanelPathImageCapture::SaveNormal(int32 PoseIndex, bool& bAnyCaptured)
{
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    if (SelectionManager.IsValid())
    {
        SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    }
    
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
        
        if (Camera)
        {
            // Ensure camera is active for capture
            if (!Camera->IsActive())
            {
                Camera->SetActive(true);
                UE_LOG(LogPathImageCapture, Log, TEXT("SaveNormal: Activated camera[%d]"), i);
            }
            // Get camera index or use iterator index
            int32 CameraIndex = Camera->GetSensorIndex();
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
                (const TArray<FFloat16Color>& NormalData)
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
    
    bGameViewChangedForCapture = EnsureGameView();
    bAutoCaptureInProgress = true;
    *JobNum = 0;

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
            else if (*JobNum == 0)
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
// UI BUTTON GROUPS
// ============================================================================

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreatePoseFileButtons()
{
    return SNew(SHorizontalBox)
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(4, 2))
        .Text(FText::FromString("Load Predefined Pose"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() { return OnLoadPoseClicked(); })
        .IsEnabled_Lambda([this]() {
            return SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid();
        })
    ]
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(4, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(4, 2))
        .HAlign(HAlign_Center)
        .Text(FText::FromString("Save Generated Pose"))
        .OnClicked_Lambda([this]() { return OnSavePoseClicked(); })
        .IsEnabled_Lambda([this]() {
            return SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid() && 
                   SelectionManager.Pin()->GetSelectedFlashPawn()->GetPoseCount() > 0;
        })
    ];
}
TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreatePoseActionButtons()
{
    return SNew(SHorizontalBox)
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Success")
        .ContentPadding(FMargin(5, 2))
        .Text(FText::FromString("Generate Poses"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() { return OnGeneratePosesClicked(); })
        .IsEnabled_Lambda([this]() {
            return SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid() &&
                !OrbitActorListItems.IsEmpty();
        })
    ]
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(4, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SAssignNew(VisualizePathButton, SButton)
        .ButtonStyle(bPathVisualized ? 
           &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") : 
           &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"))
        .ContentPadding(FMargin(5, 2))
        .HAlign(HAlign_Center)
        .Text_Lambda([this]() {
            return FText::FromString(bPathVisualized ? "Hide Path" : "Show Path");
        })
        .OnClicked_Lambda([this]() { return OnTogglePathVisualizationClicked(); })
        .IsEnabled_Lambda([this]() {
            return SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid() && !bPathNeedsUpdate;
        })
    ];
}

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreateMovementButtons()
{
    return SNew(SHorizontalBox)
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(5, 2))
        .Text(FText::FromString("Move Back"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() {
            if (SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid())
            {
                SelectionManager.Pin()->GetSelectedFlashPawn()->MoveBackward();
            }
            return FReply::Handled();
        })
        .IsEnabled_Lambda([this]() {
            return SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid();
        })
    ]
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(4, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(5, 2))
        .Text(FText::FromString("Move Next"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() {
            if (SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid())
            {
                SelectionManager.Pin()->GetSelectedFlashPawn()->MoveForward();
            }
            return FReply::Handled();
        })
        .IsEnabled_Lambda([this]() {
            return SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid();
        })
    ];
}

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreateCaptureButtons()
{
    return SNew(SHorizontalBox)
    
    // Single Capture button
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Success")
        .ContentPadding(FMargin(5, 2))
        .Text(FText::FromString("Capture This Pose"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() { return OnCaptureImagesClicked(); })
        .IsEnabled_Lambda([this]() {
            if (!SelectionManager.IsValid()) return false;
            auto SelectionManagerPin = SelectionManager.Pin();
            if (!SelectionManagerPin.IsValid() || !SelectionManagerPin->GetSelectedFlashPawn().IsValid()) return false;
            
            return (SelectionManagerPin->IsUsingRGBCamera() && SelectionManagerPin->HasRGBCamera()) || 
                   (SelectionManagerPin->IsUsingDepthCamera() && SelectionManagerPin->HasDepthCamera()) || 
                   (SelectionManagerPin->IsUsingNormalCamera() && SelectionManagerPin->HasNormalCamera()) ||
                   (SelectionManagerPin->IsUsingSegmentationCamera() && SelectionManagerPin->HasSegmentationCamera());
        })
    ]
    
    // Auto Capture button
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(4, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SAssignNew(AutoCaptureButton, SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Primary")
        .ContentPadding(FMargin(5, 2))
        .Text_Lambda([this]() {
            return bAutoCaptureInProgress ? 
                FText::FromString("Stop Capture") :
                FText::FromString("Capture All Poses");
        })
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() {
            if (bAutoCaptureInProgress)
            {
                StopAutoCapture();
                AutoCaptureButton->SetButtonStyle(&FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
            }
            else
            {
                StartAutoCapture();
                AutoCaptureButton->SetButtonStyle(&FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger"));
            }
            return FReply::Handled();
        })
        .IsEnabled_Lambda([this]() {
            if (!SelectionManager.IsValid()) return false;
            auto SelectionManagerPin = SelectionManager.Pin();
            if (!SelectionManagerPin.IsValid() || !SelectionManagerPin->GetSelectedFlashPawn().IsValid()) return false;
            
            return (SelectionManagerPin->IsUsingRGBCamera() && SelectionManagerPin->HasRGBCamera()) || 
                   (SelectionManagerPin->IsUsingDepthCamera() && SelectionManagerPin->HasDepthCamera()) || 
                   (SelectionManagerPin->IsUsingSegmentationCamera() && SelectionManagerPin->HasSegmentationCamera()) ||
                   (SelectionManagerPin->IsUsingNormalCamera() && SelectionManagerPin->HasNormalCamera());
        })
    ];
}

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