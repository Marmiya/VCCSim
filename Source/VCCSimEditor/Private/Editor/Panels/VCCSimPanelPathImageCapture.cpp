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
#include "Sensors/BaseColorCamera.h"
#include "Sensors/MaterialPropertiesCamera.h"
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

    const float HFovRad  = FMath::DegreesToRadians(FMath::Max(OrbitCameraHFOV, 5.f));
    TArray<URGBCameraComponent*> RGBCameras;
    SelectionManager.Pin()->GetSelectedFlashPawn()->GetComponents<URGBCameraComponent>(RGBCameras);
    if (RGBCameras.IsEmpty())
    {
        UE_LOG(LogPathImageCapture, Warning, 
            TEXT("No RGBCamera found on the selected FlashPawn. "
                 "Using default resolution (1920*1080) for FOV calculations."));
    }
    const FIntPoint CamRes = (RGBCameras.Num() > 0) ? 
        RGBCameras[0]->FOV : FIntPoint(1920, 1080);
    const float AspectRatio = (CamRes.Y > 0) ? (float)CamRes.X / CamRes.Y : 16.f / 9.f;
    const float VFovRad  = 2.f * FMath::Atan(FMath::Tan(HFovRad * 0.5f) / AspectRatio);

    const float FootprintH = 2.f * OrbitMargin * FMath::Tan(HFovRad * 0.5f);
    const float FootprintV = 2.f * OrbitMargin * FMath::Tan(VFovRad * 0.5f);
    const float StepH = FootprintH * FMath::Max(1.f - OrbitHOverlap, 0.05f);
    const float StepV = FootprintV * FMath::Max(1.f - OrbitVOverlap, 0.05f);

    const float BuildingH = FMath::Max(BoxMaxZ - BoxMinZ, 0.f);
    const int32 NumRings  = FMath::Max(1, FMath::CeilToInt(BuildingH / StepV));

    UWorld* World = GEditor->GetEditorWorldContext().World();
    TArray<AActor*> OrbitActors;
    for (const TSharedPtr<FString>& Label : OrbitActorListItems)
    {
        if (!Label.IsValid()) continue;
        for (TActorIterator<AActor> It(World); It; ++It)
        {
            if (It->GetActorLabel() == *Label)
            {
                OrbitActors.Add(*It);
                break;
            }
        }
    }

    TArray<UPrimitiveComponent*> OrbitPrimComps;
    for (AActor* Actor : OrbitActors)
        Actor->GetComponents<UPrimitiveComponent>(OrbitPrimComps);

    const int32 NumSampleAngles = 360;
    const float SearchRadius = (FMath::Max(HX, HY) + OrbitMargin) * 4.f + 2000.f;

    FCollisionQueryParams QueryParams;
    QueryParams.bTraceComplex = true;

    TArray<FVector>  Positions;
    TArray<FRotator> Rotations;

    for (int32 Ring = 0; Ring < NumRings; ++Ring)
    {
        const float T = (NumRings > 1) ? (float)Ring / (NumRings - 1) : 0.5f;
        const float Z = BoxMinZ + T * BuildingH;
        const FVector CenterAtZ(BoxCenter.X, BoxCenter.Y, Z);

        TArray<FVector> FineSamples;
        FineSamples.Reserve(NumSampleAngles);

        for (int32 AngleIdx = 0; AngleIdx < NumSampleAngles; ++AngleIdx)
        {
            const float AngleRad = FMath::DegreesToRadians(360.f * AngleIdx / NumSampleAngles);
            const FVector Dir(FMath::Cos(AngleRad), FMath::Sin(AngleRad), 0.f);
            const FVector TraceStart = CenterAtZ + Dir * SearchRadius;
            const FVector TraceEnd   = CenterAtZ;

            FHitResult BestHit;
            float BestDistFromStart = FLT_MAX;
            bool bFoundHit = false;

            for (UPrimitiveComponent* Comp : OrbitPrimComps)
            {
                FHitResult CompHit;
                if (Comp && Comp->LineTraceComponent(CompHit, TraceStart, TraceEnd, QueryParams))
                {
                    const float D = FVector::Dist(TraceStart, CompHit.Location);
                    if (D < BestDistFromStart)
                    {
                        BestDistFromStart = D;
                        BestHit = CompHit;
                        bFoundHit = true;
                    }
                }
            }

            FVector CamPos;
            if (bFoundHit)
            {
                const FVector N2D(BestHit.Normal.X, BestHit.Normal.Y, 0.f);
                const FVector HitNormal2D = N2D.IsNearlyZero() ? Dir : N2D.GetSafeNormal();
                CamPos = FVector(BestHit.Location.X + HitNormal2D.X * OrbitMargin,
                                 BestHit.Location.Y + HitNormal2D.Y * OrbitMargin,
                                 Z);
            }
            else
            {
                const float AbsCosA = FMath::Abs(Dir.X);
                const float AbsSinA = FMath::Abs(Dir.Y);
                const float BoxDist = (AbsCosA < KINDA_SMALL_NUMBER) ? HY :
                                      (AbsSinA < KINDA_SMALL_NUMBER) ? HX :
                                      FMath::Min(HX / AbsCosA, HY / AbsSinA);
                CamPos = FVector(CenterAtZ.X + (BoxDist + OrbitMargin) * Dir.X,
                                 CenterAtZ.Y + (BoxDist + OrbitMargin) * Dir.Y,
                                 Z);
            }
            FineSamples.Add(CamPos);
        }

        auto AddPose = [&](const FVector& CamPos)
        {
            const FVector LookDir(BoxCenter.X - CamPos.X, BoxCenter.Y - CamPos.Y, 0.f);
            Positions.Add(CamPos);
            Rotations.Add(LookDir.GetSafeNormal().Rotation());
        };

        AddPose(FineSamples[0]);
        float AccumLen = 0.f;
        for (int32 i = 0; i < NumSampleAngles; ++i)
        {
            const int32 Next = (i + 1) % NumSampleAngles;
            AccumLen += FVector::Dist2D(FineSamples[i], FineSamples[Next]);
            if (AccumLen >= StepH)
            {
                AddPose(FVector(FineSamples[Next].X, FineSamples[Next].Y, Z));
                AccumLen = 0.f;
            }
        }
    }

    if (bOrbitIncludeNadir)
    {
        const float NadirZ     = CombinedBox.Max.Z + OrbitNadirAlt;
        const float NadirFootH = 2.f * OrbitNadirAlt * FMath::Tan(HFovRad * 0.5f);
        const float NadirFootV = 2.f * OrbitNadirAlt * FMath::Tan(VFovRad * 0.5f);
        const float CrossStep  = NadirFootH * FMath::Max(1.f - OrbitHOverlap, 0.05f);
        const float AlongStep  = NadirFootV * FMath::Max(1.f - OrbitVOverlap, 0.05f);

        const float MinX = CombinedBox.Min.X - OrbitMargin;
        const float MaxX = CombinedBox.Max.X + OrbitMargin;
        const float MinY = CombinedBox.Min.Y - OrbitMargin;
        const float MaxY = CombinedBox.Max.Y + OrbitMargin;

        int32 StripIdx = 0;
        for (float X = MinX; X <= MaxX + KINDA_SMALL_NUMBER; X += CrossStep, ++StripIdx)
        {
            const float Yaw = (StripIdx % 2 == 0) ? 90.f : -90.f;
            if (StripIdx % 2 == 0)
            {
                for (float Y = MinY; Y <= MaxY + KINDA_SMALL_NUMBER; Y += AlongStep)
                {
                    Positions.Add(FVector(X, Y, NadirZ));
                    Rotations.Add(FRotator(-90.f, Yaw, 0.f));
                }
            }
            else
            {
                for (float Y = MaxY; Y >= MinY - KINDA_SMALL_NUMBER; Y -= AlongStep)
                {
                    Positions.Add(FVector(X, Y, NadirZ));
                    Rotations.Add(FRotator(-90.f, Yaw, 0.f));
                }
            }
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

    UE_LOG(LogPathImageCapture, Log, TEXT("Conformal orbit: %d total poses (%d rings%s)"),
        Positions.Num(), NumRings, bOrbitIncludeNadir ? TEXT(" + nadir zigzag") : TEXT(""));
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
        
        if (SelectionManagerPin->IsUsingRGBCamera() && SelectionManagerPin->HasRGBCamera())
        {
            SaveRGB(PoseIndex, bAnyCaptured);
        }
        
        if (SelectionManagerPin->IsUsingDepthCamera() && SelectionManagerPin->HasDepthCamera())
        {
            SaveDepth(PoseIndex, bAnyCaptured);
        }
        
        if (SelectionManagerPin->IsUsingNormalCamera() && SelectionManagerPin->HasNormalCamera())
        {
            SaveNormal(PoseIndex, bAnyCaptured);
        }
        
        if (SelectionManagerPin->IsUsingSegmentationCamera() &&
            SelectionManagerPin->HasSegmentationCamera())
        {
            SaveSeg(PoseIndex, bAnyCaptured);
        }

        if (SelectionManagerPin->IsUsingBaseColorCamera() &&
            SelectionManagerPin->HasBaseColorCamera())
        {
            SaveBaseColor(PoseIndex, bAnyCaptured);
        }

        if (SelectionManagerPin->IsUsingMaterialPropertiesCamera() &&
            SelectionManagerPin->HasMaterialPropertiesCamera())
        {
            SaveMaterialProperties(PoseIndex, bAnyCaptured);
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
    auto SelectionManagerPin = SelectionManager.Pin();
    if (!SelectionManagerPin.IsValid()) return;

    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn = SelectionManagerPin->GetSelectedFlashPawn();
    if (!SelectedFlashPawn.IsValid()) return;

    if (SelectionManagerPin->ShouldUseRGBCameraClass())
    {
        TArray<URGBCameraComponent*> RGBCameras;
        SelectedFlashPawn->GetComponents<URGBCameraComponent>(RGBCameras);
        *JobNum += RGBCameras.Num();

        for (int32 i = 0; i < RGBCameras.Num(); ++i)
        {
            URGBCameraComponent* Camera = RGBCameras[i];
            if (Camera)
            {
                int32 CameraIndex = Camera->GetSensorIndex();
                if (CameraIndex < 0) CameraIndex = i;

                FString Filename = SaveDirectory /
                    FString::Printf(TEXT("RGB_Cam%02d_Pose%03d.png"), CameraIndex, PoseIndex);
                FIntPoint Size = {Camera->GetImageSize().first, Camera->GetImageSize().second};

                Camera->AsyncGetRGBImageData(
                    [Filename, Size, JobNum = this->JobNum](const TArray<FColor>& ImageData)
                    {
                        (new FAutoDeleteAsyncTask<FAsyncImageSaveTask>(ImageData, Size, Filename))
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
    else
    {
        // Existing logic: Use high-res screenshot
        TArray<URGBCameraComponent*> RGBCameras;
        SelectedFlashPawn->GetComponents<URGBCameraComponent>(RGBCameras);
        *JobNum += RGBCameras.Num();
        
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
                int32 CameraIndex = Camera->GetSensorIndex();
                if (CameraIndex < 0) CameraIndex = i;
                
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
                ViewportClient->ViewFOV = Camera->FOV;
                ViewportClient->Invalidate();
                ViewportClient->Viewport->Draw();
                
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
            int32 CameraIndex = Camera->GetSensorIndex();
            if (CameraIndex < 0) CameraIndex = i;

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
            int32 CameraIndex = Camera->GetSensorIndex();
            if (CameraIndex < 0) CameraIndex = i;

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

void FVCCSimPanelPathImageCapture::SaveBaseColor(int32 PoseIndex, bool& bAnyCaptured)
{
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    if (SelectionManager.IsValid())
        SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();

    if (!SelectedFlashPawn.IsValid()) return;

    TArray<UBaseColorCameraComponent*> Cameras;
    SelectedFlashPawn->GetComponents<UBaseColorCameraComponent>(Cameras);
    *JobNum += Cameras.Num();

    for (int32 i = 0; i < Cameras.Num(); ++i)
    {
        UBaseColorCameraComponent* Camera = Cameras[i];
        if (Camera)
        {
            int32 CameraIndex = Camera->GetSensorIndex();
            if (CameraIndex < 0) CameraIndex = i;

            FString Filename = SaveDirectory / FString::Printf(
                TEXT("BaseColor_Cam%02d_Pose%03d.png"), CameraIndex, PoseIndex);

            FIntPoint Size = {Camera->GetImageSize().first, Camera->GetImageSize().second};

            Camera->AsyncGetBaseColorImageData(
                [Filename, Size, JobNum = this->JobNum](const TArray<FColor>& ImageData)
                {
                    TArray<FColor> DataCopy = ImageData;
                    for (FColor& Pixel : DataCopy) Pixel.A = 255;
                    (new FAutoDeleteAsyncTask<FAsyncImageSaveTask>(DataCopy, Size, Filename))
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

void FVCCSimPanelPathImageCapture::SaveMaterialProperties(int32 PoseIndex, bool& bAnyCaptured)
{
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    if (SelectionManager.IsValid())
        SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();

    if (!SelectedFlashPawn.IsValid()) return;

    TArray<UMaterialPropertiesCameraComponent*> Cameras;
    SelectedFlashPawn->GetComponents<UMaterialPropertiesCameraComponent>(Cameras);
    *JobNum += Cameras.Num();

    for (int32 i = 0; i < Cameras.Num(); ++i)
    {
        UMaterialPropertiesCameraComponent* Camera = Cameras[i];
        if (Camera)
        {
            int32 CameraIndex = Camera->GetSensorIndex();
            if (CameraIndex < 0) CameraIndex = i;

            FString Filename = SaveDirectory / FString::Printf(
                TEXT("MatProps_Cam%02d_Pose%03d.png"), CameraIndex, PoseIndex);

            FIntPoint Size = {Camera->GetImageSize().first, Camera->GetImageSize().second};

            Camera->AsyncGetMaterialPropertiesImageData(
                [Filename, Size, JobNum = this->JobNum](const TArray<FColor>& ImageData)
                {
                    TArray<FColor> DataCopy = ImageData;
                    for (FColor& Pixel : DataCopy) Pixel.A = 255;
                    (new FAutoDeleteAsyncTask<FAsyncImageSaveTask>(DataCopy, Size, Filename))
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

    TArray<FVector> Positions;
    TArray<FRotator> Rotations;
    SelectedFlashPawn->GetCurrentPath(Positions, Rotations);
    WritePosesToFile(Positions, Rotations, SaveDirectory / TEXT("poses.txt"));

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
                   (SelectionManagerPin->IsUsingSegmentationCamera() && SelectionManagerPin->HasSegmentationCamera()) ||
                   (SelectionManagerPin->IsUsingBaseColorCamera() && SelectionManagerPin->HasBaseColorCamera()) ||
                   (SelectionManagerPin->IsUsingMaterialPropertiesCamera() && SelectionManagerPin->HasMaterialPropertiesCamera());
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

void FVCCSimPanelPathImageCapture::LoadFromConfigManager()
{
    LoadOrbitActorList();
}