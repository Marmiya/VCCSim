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
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program. If not, see <https://www.gnu.org/licenses/>.
*/

DEFINE_LOG_CATEGORY_STATIC(LogSelection, Log, All);

#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Utils/VCCSimConfigManager.h"
#include "Engine/World.h"
#include "Engine/Engine.h"
#include "EngineUtils.h"
#include "Selection.h"
#include "Pawns/FlashPawn.h"
#include "Pawns/SimLookAtPath.h"
#include "Sensors/SensorBase.h"
#include "Sensors/RGBCamera.h"
#include "Sensors/DepthCamera.h"
#include "Sensors/SegmentationCamera.h"
#include "Sensors/NormalCamera.h"
#include "Sensors/RGBLinearCamera.h"
#include "Sensors/BaseColorCamera.h"
#include "Sensors/BaseColorLinearCamera.h"
#include "Sensors/MaterialPropertiesCamera.h"
#include "Utils/GTMaterialExporter.h"
#include "Utils/VCCSimUIHelpers.h"
#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#include "Framework/Application/SlateApplication.h"
#include "Misc/Paths.h"
#include "Components/MeshComponent.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInterface.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Utils/PathGenerator.h"
#include "ScopedTransaction.h"

FVCCSimPanelSelection::FVCCSimPanelSelection()
{
}

FVCCSimPanelSelection::~FVCCSimPanelSelection()
{
    Cleanup();
}

void FVCCSimPanelSelection::Initialize()
{
    UE_LOG(LogSelection, Log, TEXT("VCCSimPanelSelection initialized"));
}

void FVCCSimPanelSelection::Cleanup()
{
    ClearSelections();

    if (AActor* HA = HighlightActor.Get())
        HA->Destroy();
    HighlightActor.Reset();

    for (const TWeakObjectPtr<AActor>& WA : HiddenGroundActors)
        if (AActor* A = WA.Get())
            A->SetIsTemporarilyHiddenInEditor(false);
    HiddenGroundActors.Reset();

    for (const TWeakObjectPtr<AActor>& WA : HiddenUnmatchedActors)
        if (AActor* A = WA.Get())
            A->SetIsTemporarilyHiddenInEditor(false);
    HiddenUnmatchedActors.Reset();

    SelectedFlashPawnText.Reset();
    SelectFlashPawnToggle.Reset();
    SelectedLookAtText.Reset();
    SelectLookAtToggle.Reset();
    TargetActorListView.Reset();
}

void FVCCSimPanelSelection::HandleActorSelection(AActor* Actor)
{
    if (!Actor || !IsValid(Actor))
    {
        return;
    }

    if (bSelectingFlashPawn)
    {
        AFlashPawn* FlashPawn = Cast<AFlashPawn>(Actor);
        if (FlashPawn)
        {
            SelectedFlashPawn = FlashPawn;
            if (SelectedFlashPawnText.IsValid())
            {
                SelectedFlashPawnText->SetText(FText::FromString(FlashPawn->GetActorLabel()));
            }

            bSelectingFlashPawn = false;
            if (SelectFlashPawnToggle.IsValid())
            {
                SelectFlashPawnToggle->SetIsChecked(ECheckBoxState::Unchecked);
            }

            RefreshCameraAvailability();
            bIsWarmedUp = false;

            UE_LOG(LogSelection, Log, TEXT("Selected FlashPawn: %s"), *FlashPawn->GetActorLabel());
        }
    }
    else if (bSelectingLookAtPath)
    {
        AVCCSimLookAtPath* LookAt = Cast<AVCCSimLookAtPath>(Actor);
        if (LookAt)
        {
            SelectedLookAtPath = LookAt;
            if (SelectedLookAtText.IsValid())
            {
                SelectedLookAtText->SetText(FText::FromString(LookAt->GetActorLabel()));
            }

            bSelectingLookAtPath = false;
            if (SelectLookAtToggle.IsValid())
            {
                SelectLookAtToggle->SetIsChecked(ECheckBoxState::Unchecked);
            }

            UE_LOG(LogSelection, Log, TEXT("Selected LookAtPath: %s"), *LookAt->GetActorLabel());
        }
        else
        {
            UE_LOG(LogSelection, Warning, TEXT("Selected actor is not a VCCSimLookAtPath"));
        }
    }
}

void FVCCSimPanelSelection::AutoSelectFlashPawn()
{
    UWorld* World = GEditor->GetEditorWorldContext().World();
    if (!World)
    {
        return;
    }

    SelectedFlashPawn = nullptr;

    AFlashPawn* FirstFoundFlashPawn = nullptr;
    for (TActorIterator<AFlashPawn> ActorIterator(World); ActorIterator; ++ActorIterator)
    {
        AFlashPawn* FlashPawn = *ActorIterator;
        if (FlashPawn && IsValid(FlashPawn))
        {
            FirstFoundFlashPawn = FlashPawn;
            break;
        }
    }

    if (FirstFoundFlashPawn)
    {
        SelectedFlashPawn = FirstFoundFlashPawn;

        if (SelectedFlashPawnText.IsValid())
        {
            SelectedFlashPawnText->SetText(FText::FromString(FirstFoundFlashPawn->GetActorLabel()));
        }

        RefreshCameraAvailability();
        bIsWarmedUp = false;

        UE_LOG(LogSelection, Log, TEXT("Auto-selected FlashPawn: %s"), *FirstFoundFlashPawn->GetActorLabel());
    }
    else
    {
        if (SelectedFlashPawnText.IsValid())
        {
            SelectedFlashPawnText->SetText(FText::FromString("None selected"));
        }
        UE_LOG(LogSelection, Log, TEXT("No FlashPawn found in the scene for auto-selection"));
    }
}

void FVCCSimPanelSelection::AutoSelectLookAtPath()
{
    UWorld* World = GEditor->GetEditorWorldContext().World();
    if (!World)
    {
        return;
    }

    SelectedLookAtPath = nullptr;

    for (TActorIterator<AVCCSimLookAtPath> It(World); It; ++It)
    {
        AVCCSimLookAtPath* LookAt = *It;
        if (LookAt && IsValid(LookAt))
        {
            SelectedLookAtPath = LookAt;
            if (SelectedLookAtText.IsValid())
            {
                SelectedLookAtText->SetText(FText::FromString(LookAt->GetActorLabel()));
            }
            UE_LOG(LogSelection, Log, TEXT("Auto-selected LookAtPath: %s"), *LookAt->GetActorLabel());
            return;
        }
    }

    if (SelectedLookAtText.IsValid())
    {
        SelectedLookAtText->SetText(FText::FromString("None selected"));
    }
    UE_LOG(LogSelection, Log, TEXT("No VCCSimLookAtPath found in the scene for auto-selection"));
}

TArray<FString> FVCCSimPanelSelection::GetEnabledTargetActorLabels() const
{
    TArray<FString> Labels;
    for (const TSharedPtr<FVCCSimTargetActorItem>& Item : TargetActorItems)
    {
        if (Item.IsValid() && Item->bEnabled)
        {
            Labels.Add(Item->Label);
        }
    }
    return Labels;
}

bool FVCCSimPanelSelection::HasEnabledTargetActors() const
{
    for (const TSharedPtr<FVCCSimTargetActorItem>& Item : TargetActorItems)
    {
        if (Item.IsValid() && Item->bEnabled)
        {
            return true;
        }
    }
    return false;
}

void FVCCSimPanelSelection::LoadFromConfigManager()
{
    const auto& Config = FVCCSimConfigManager::Get().GetTargetActorsConfig();

    BoundsMin = Config.BoundsMin;
    BoundsMax = Config.BoundsMax;
    MinBuildingHeight = Config.MinBuildingHeight;
    MinBuildingFootprint = Config.MinBuildingFootprint;
    ConnectGap = Config.ConnectGap;

    TargetActorItems.Empty();
    for (int32 i = 0; i < Config.Labels.Num(); ++i)
    {
        TSharedPtr<FVCCSimTargetActorItem> Item = MakeShared<FVCCSimTargetActorItem>();
        Item->Label = Config.Labels[i];
        Item->bEnabled = Config.EnabledFlags.IsValidIndex(i) ? Config.EnabledFlags[i] : true;
        TargetActorItems.Add(Item);
    }

    GroundActorItems.Empty();
    for (const FString& Label : Config.GroundLabels)
        GroundActorItems.Add(MakeShared<FString>(Label));

    if (TargetActorListView.IsValid())
    {
        TargetActorListView->RequestListRefresh();
    }
    if (GroundActorListView.IsValid())
    {
        GroundActorListView->RequestListRefresh();
    }
}

void FVCCSimPanelSelection::SaveTargetActorsToConfig() const
{
    FVCCSimConfigManager::FTargetActorsConfig Config;
    for (const TSharedPtr<FVCCSimTargetActorItem>& Item : TargetActorItems)
    {
        if (!Item.IsValid()) continue;
        Config.Labels.Add(Item->Label);
        Config.EnabledFlags.Add(Item->bEnabled);
    }
    for (const TSharedPtr<FString>& Item : GroundActorItems)
        if (Item.IsValid()) Config.GroundLabels.Add(*Item);
    Config.BoundsMin = BoundsMin;
    Config.BoundsMax = BoundsMax;
    Config.MinBuildingHeight = MinBuildingHeight;
    Config.MinBuildingFootprint = MinBuildingFootprint;
    Config.ConnectGap = ConnectGap;
    FVCCSimConfigManager::Get().SetTargetActorsConfig(Config);
}

FReply FVCCSimPanelSelection::OnAddTargetActorsClicked()
{
    USelection* Sel = GEditor ? GEditor->GetSelectedActors() : nullptr;
    if (!Sel) return FReply::Handled();

    bool bAdded = false;
    for (int32 i = 0; i < Sel->Num(); ++i)
    {
        AActor* Actor = Cast<AActor>(Sel->GetSelectedObject(i));
        if (!Actor) continue;

        const FString Label = Actor->GetActorLabel();
        const bool bDuplicate = TargetActorItems.ContainsByPredicate(
            [&Label](const TSharedPtr<FVCCSimTargetActorItem>& Item)
            {
                return Item.IsValid() && Item->Label == Label;
            });

        if (!bDuplicate)
        {
            TSharedPtr<FVCCSimTargetActorItem> Item = MakeShared<FVCCSimTargetActorItem>();
            Item->Label = Label;
            TargetActorItems.Add(Item);
            bAdded = true;
        }
    }

    if (bAdded)
    {
        if (TargetActorListView.IsValid())
        {
            TargetActorListView->RequestListRefresh();
        }
        SaveTargetActorsToConfig();
    }

    return FReply::Handled();
}

FReply FVCCSimPanelSelection::OnAddGroundActorsClicked()
{
    USelection* Sel = GEditor ? GEditor->GetSelectedActors() : nullptr;
    if (!Sel) return FReply::Handled();

    bool bAdded = false;
    for (int32 i = 0; i < Sel->Num(); ++i)
    {
        AActor* Actor = Cast<AActor>(Sel->GetSelectedObject(i));
        if (!Actor) continue;

        const FString Label = Actor->GetActorLabel();
        const bool bDuplicate = GroundActorItems.ContainsByPredicate(
            [&Label](const TSharedPtr<FString>& Item) { return Item.IsValid() && *Item == Label; });
        if (!bDuplicate)
        {
            GroundActorItems.Add(MakeShared<FString>(Label));
            bAdded = true;
        }
    }

    if (bAdded)
    {
        if (GroundActorListView.IsValid())
            GroundActorListView->RequestListRefresh();
        SaveTargetActorsToConfig();
    }
    return FReply::Handled();
}

TSet<const AActor*> FVCCSimPanelSelection::GetForcedGroundActors(UWorld* World) const
{
    TSet<const AActor*> Result;
    if (!World || GroundActorItems.Num() == 0) return Result;

    TSet<FString> Wanted;
    for (const TSharedPtr<FString>& Item : GroundActorItems)
        if (Item.IsValid()) Wanted.Add(*Item);

    for (TActorIterator<AActor> It(World); It; ++It)
        if (AActor* A = *It)
            if (Wanted.Contains(A->GetActorLabel()))
                Result.Add(A);

    return Result;
}

FReply FVCCSimPanelSelection::OnAddTargetActorsInBoundsClicked()
{
    UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
    if (!World)
    {
        return FReply::Handled();
    }

    const FBox Query(BoundsMin.ComponentMin(BoundsMax), BoundsMin.ComponentMax(BoundsMax));

    int32 Added = 0;
    int32 SkippedInfra = 0;
    for (TActorIterator<AActor> It(World); It; ++It)
    {
        AActor* Actor = *It;
        if (!Actor || !IsValid(Actor) || Actor->HasAnyFlags(RF_Transient)) continue;
        if (!FGTMaterialExporter::HasExportableMeshGeometry(Actor)) continue;

        // Use the actor's bounds CENTRE for containment, not AABB overlap: an out-of-range actor
        // whose rotated/elongated AABB merely clips a corner of the box is no longer pulled in.
        FVector Origin, Extent;
        Actor->GetActorBounds(false, Origin, Extent);
        if (!Query.IsInside(Origin)) continue;

        // Add every in-box mesh actor (buildings, ground, props alike); only skip our own capture
        // infrastructure. Buildings vs ground is decided geometrically later, not here.
        if (FGTMaterialExporter::IsCaptureInfraActor(Actor))
        {
            ++SkippedInfra;
            continue;
        }

        const FString Label = Actor->GetActorLabel();
        const bool bDuplicate = TargetActorItems.ContainsByPredicate(
            [&Label](const TSharedPtr<FVCCSimTargetActorItem>& Item)
            {
                return Item.IsValid() && Item->Label == Label;
            });
        if (bDuplicate) continue;

        TSharedPtr<FVCCSimTargetActorItem> Item = MakeShared<FVCCSimTargetActorItem>();
        Item->Label = Label;
        TargetActorItems.Add(Item);
        ++Added;
    }

    if (Added > 0)
    {
        if (TargetActorListView.IsValid())
        {
            TargetActorListView->RequestListRefresh();
        }
        SaveTargetActorsToConfig();
    }

    UE_LOG(LogSelection, Log,
        TEXT("Add In Bounds: added %d actor(s), skipped %d capture-infra."), Added, SkippedInfra);
    return FReply::Handled();
}

FReply FVCCSimPanelSelection::OnHighlightTargetsClicked()
{
    UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
    if (!World)
        return FReply::Handled();

    // Toggle: a second click tears the highlight down.
    if (AActor* Existing = HighlightActor.Get())
    {
        Existing->Destroy();
        HighlightActor.Reset();
        return FReply::Handled();
    }

    // Highlight is drawn as instanced cube "beams" (box edges) on a transient actor, NOT debug lines:
    // ULineBatchComponent re-processes every persistent line every frame (the lifetime-ageing pass),
    // which with thousands of boxes stalls the editor. Instanced static meshes are GPU-resident and cost
    // nothing per frame once built — the same mesh-based approach as the path visualisation.
    FActorSpawnParameters SpawnParams;
    SpawnParams.ObjectFlags = RF_Transient;
    SpawnParams.bNoFail = true;
    AActor* VizActor = World->SpawnActor<AActor>(AActor::StaticClass(), FTransform::Identity, SpawnParams);
    if (!VizActor)
        return FReply::Handled();
    VizActor->SetActorLabel(TEXT("VCCSimTargetHighlight"));
    USceneComponent* Root = NewObject<USceneComponent>(VizActor);
    Root->SetMobility(EComponentMobility::Movable);
    VizActor->SetRootComponent(Root);
    Root->RegisterComponent();
    HighlightActor = VizActor;

    UStaticMesh* CubeMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
    UMaterialInterface* BaseMat =
        LoadObject<UMaterialInterface>(nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
    if (!CubeMesh)
        return FReply::Handled();

    // One instanced-mesh component per colour (only a handful), each with a coloured material instance.
    TMap<uint32, UInstancedStaticMeshComponent*> ColorISMCs;
    auto GetISMC = [&](const FLinearColor& Color) -> UInstancedStaticMeshComponent*
    {
        const uint32 Key = Color.ToFColor(false).ToPackedARGB();
        if (UInstancedStaticMeshComponent** Found = ColorISMCs.Find(Key))
            return *Found;
        UInstancedStaticMeshComponent* ISMC = NewObject<UInstancedStaticMeshComponent>(VizActor);
        ISMC->SetStaticMesh(CubeMesh);
        ISMC->SetMobility(EComponentMobility::Movable);
        ISMC->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        ISMC->SetCastShadow(false);
        ISMC->SetupAttachment(Root);
        if (BaseMat)
        {
            UMaterialInstanceDynamic* MID = UMaterialInstanceDynamic::Create(BaseMat, VizActor);
            MID->SetVectorParameterValue(FName("Color"), Color);
            ISMC->SetMaterial(0, MID);
        }
        ISMC->RegisterComponent();
        ColorISMCs.Add(Key, ISMC);
        return ISMC;
    };

    // The BasicShapes cube is a 100 cm box centred on its pivot, so a beam A->B is the cube scaled to
    // (length, thickness, thickness) along local X, rotated onto B-A, centred at the segment midpoint.
    auto AddEdge = [&](const FVector& A, const FVector& B, const FLinearColor& Color, float Thickness)
    {
        const FVector Dir = B - A;
        const float Len = Dir.Size();
        if (Len < KINDA_SMALL_NUMBER) return;
        const FQuat Rot = FRotationMatrix::MakeFromX(Dir).ToQuat();
        const FVector Scale(Len / 100.f, Thickness / 100.f, Thickness / 100.f);
        GetISMC(Color)->AddInstance(FTransform(Rot, (A + B) * 0.5f, Scale), /*bWorldSpace=*/true);
    };
    auto AddEdges = [&](const FVector C[8], const FLinearColor& Color, float Thickness)
    {
        const int32 E[12][2] = {
            {0,1},{1,2},{2,3},{3,0}, {4,5},{5,6},{6,7},{7,4}, {0,4},{1,5},{2,6},{3,7} };
        for (const auto& Edge : E) AddEdge(C[Edge[0]], C[Edge[1]], Color, Thickness);
    };
    auto AddBox = [&](const FVector& Origin, const FVector& Extent, const FLinearColor& Color, float Thickness)
    {
        const FVector C[8] = {
            Origin + FVector(-Extent.X,-Extent.Y,-Extent.Z), Origin + FVector( Extent.X,-Extent.Y,-Extent.Z),
            Origin + FVector( Extent.X, Extent.Y,-Extent.Z), Origin + FVector(-Extent.X, Extent.Y,-Extent.Z),
            Origin + FVector(-Extent.X,-Extent.Y, Extent.Z), Origin + FVector( Extent.X,-Extent.Y, Extent.Z),
            Origin + FVector( Extent.X, Extent.Y, Extent.Z), Origin + FVector(-Extent.X, Extent.Y, Extent.Z) };
        AddEdges(C, Color, Thickness);
    };
    auto AddOBB = [&](const FPathGenerator::FVerticalOBB& O, const FLinearColor& Color, float Thickness)
    {
        const FVector2D AX = O.AxisX;
        const FVector2D AY(-AX.Y, AX.X);
        auto Corner = [&](double sx, double sy, double z) -> FVector
        {
            return FVector(
                O.Center.X + sx * O.HalfXY.X * AX.X + sy * O.HalfXY.Y * AY.X,
                O.Center.Y + sx * O.HalfXY.X * AX.Y + sy * O.HalfXY.Y * AY.Y, z);
        };
        const FVector C[8] = {
            Corner(-1,-1,O.MinZ), Corner(1,-1,O.MinZ), Corner(1,1,O.MinZ), Corner(-1,1,O.MinZ),
            Corner(-1,-1,O.MaxZ), Corner(1,-1,O.MaxZ), Corner(1,1,O.MaxZ), Corner(-1,1,O.MaxZ) };
        AddEdges(C, Color, Thickness);
    };

    // Building-detection params drive the per-building colouring and the brown ground boxes below.
    FPathGenerator::FBuildingDetectParams BParams;
    BParams.MinBuildingHeight = MinBuildingHeight;
    BParams.MinBuildingFootprint = MinBuildingFootprint;
    BParams.ConnectGap = ConnectGap;
    BParams.ForcedGroundActors = GetForcedGroundActors(World);
    const FLinearColor GroundColor(0.55f, 0.30f, 0.08f);   // brown = ground / terrain

    TMap<FString, AActor*> LabelMap;
    for (TActorIterator<AActor> It(World); It; ++It)
        if (AActor* A = *It) LabelMap.Add(A->GetActorLabel(), A);

    // Detect buildings up front so each list actor can be coloured by the building it was merged into.
    TArray<AActor*> EnabledActors;
    for (const TSharedPtr<FVCCSimTargetActorItem>& Item : TargetActorItems)
        if (Item.IsValid() && Item->bEnabled)
            if (AActor** Found = LabelMap.Find(Item->Label))
                if (*Found) EnabledActors.Add(*Found);

    const TArray<FPathGenerator::FOrbitTarget> Buildings =
        FPathGenerator::DetectBuildingsCached(World, EnabledActors, BParams);

    TMap<const AActor*, int32> ActorBuilding;
    for (int32 b = 0; b < Buildings.Num(); ++b)
        for (const TWeakObjectPtr<AActor>& WA : Buildings[b].Actors)
            if (const AActor* A = WA.Get()) ActorBuilding.Add(A, b);

    // One distinct colour per building (members of the same building share it); palette avoids the
    // reserved green (non-building structure) / brown (ground) / gray (disabled) / red+orange (audit).
    auto BuildingColor = [](int32 Idx) -> FLinearColor
    {
        static const FLinearColor Palette[] = {
            FLinearColor(0.00f, 0.80f, 1.00f), FLinearColor(1.00f, 0.00f, 1.00f),
            FLinearColor(1.00f, 0.85f, 0.00f), FLinearColor(0.30f, 0.45f, 1.00f),
            FLinearColor(0.70f, 0.20f, 1.00f), FLinearColor(0.00f, 0.85f, 0.55f),
            FLinearColor(1.00f, 0.45f, 0.75f), FLinearColor(0.50f, 0.80f, 1.00f),
            FLinearColor(0.85f, 0.50f, 1.00f), FLinearColor(0.95f, 0.75f, 0.20f) };
        return Palette[Idx % (sizeof(Palette) / sizeof(Palette[0]))];
    };

    // 1) Every list entry: building members take their building's colour; enabled structures not merged
    //    into any building = green; enabled ground = brown; disabled = gray.
    TSet<FString> ListedLabels;
    int32 Enabled = 0, Disabled = 0, Missing = 0, Ground = 0;
    for (const TSharedPtr<FVCCSimTargetActorItem>& Item : TargetActorItems)
    {
        if (!Item.IsValid()) continue;
        ListedLabels.Add(Item->Label);
        AActor** Found = LabelMap.Find(Item->Label);
        if (!Found || !*Found) { ++Missing; continue; }

        const bool bGround = FPathGenerator::IsGroundLikeActor(World, *Found, BParams);
        FLinearColor Color = FLinearColor(0.3f, 0.3f, 0.3f);   // disabled gray
        if (Item->bEnabled)
        {
            if (bGround) Color = GroundColor;
            else if (const int32* BIdx = ActorBuilding.Find(*Found)) Color = BuildingColor(*BIdx);
            else Color = FLinearColor::Green;
        }
        // Draw the min-area OBB — the exact box DetectBuildings tests for connectivity. AABB fallback
        // only if the OBB degenerates.
        const FPathGenerator::FVerticalOBB ItemOBB = FPathGenerator::BuildActorOBB(*Found);
        if (ItemOBB.bValid)
        {
            AddOBB(ItemOBB, Color, 8.f);
        }
        else
        {
            FVector Origin, Extent;
            (*Found)->GetActorBounds(false, Origin, Extent);
            AddBox(Origin, Extent, Color, 8.f);
        }
        if (!Item->bEnabled) ++Disabled;
        else if (bGround) ++Ground;
        else ++Enabled;
    }

    // 2) Audit every actor inside the configured box that is NOT in the list and has mesh geometry:
    //    red    = has an exportable static mesh + materials (a valid candidate that was clutter-filtered
    //             or simply never added); orange = has mesh geometry but NO exportable static mesh
    //             (dynamic / procedural / material-less meshes). Orange actors are invisible to the
    //             bounds auto-add AND to the red audit, which is why such an actor can end up with no
    //             box at all. Skipped when no box is configured (the default box spans the whole scene).
    const bool bBoxConfigured =
        !(BoundsMin.Equals(FVector(-100000.0), 1.0) && BoundsMax.Equals(FVector(100000.0), 1.0));
    int32 Red = 0, Orange = 0;
    if (bBoxConfigured)
    {
        const FBox Query(BoundsMin.ComponentMin(BoundsMax), BoundsMin.ComponentMax(BoundsMax));
        const int32 MaxAudit = 2000;
        for (TActorIterator<AActor> It(World); It && (Red + Orange) < MaxAudit; ++It)
        {
            AActor* Actor = *It;
            if (!Actor || !IsValid(Actor) || Actor == VizActor || Actor->HasAnyFlags(RF_Transient)) continue;
            if (Actor->IsA<AFlashPawn>() || Actor->IsA<AVCCSimLookAtPath>()) continue;
            if (ListedLabels.Contains(Actor->GetActorLabel())) continue;

            TArray<UMeshComponent*> MeshComps;
            Actor->GetComponents<UMeshComponent>(MeshComps);
            if (MeshComps.Num() == 0) continue;

            FVector Origin, Extent;
            Actor->GetActorBounds(false, Origin, Extent);
            if (!Query.IsInside(Origin)) continue;

            const bool bExportable = FGTMaterialExporter::HasExportableMeshGeometry(Actor);
            AddBox(Origin, Extent, bExportable ? FLinearColor::Red : FLinearColor(1.f, 0.5f, 0.f), 8.f);
            if (bExportable) ++Red; else ++Orange;
        }
    }

    // 3) One thick merged outline per detected building, in that building's colour — the bold envelope
    //    around its same-coloured members.
    for (int32 b = 0; b < Buildings.Num(); ++b)
    {
        const FPathGenerator::FOrbitTarget& Building = Buildings[b];
        const FLinearColor Color = BuildingColor(b);
        if (Building.OBB.bValid)
        {
            AddOBB(Building.OBB, Color, 20.f);
        }
        else if (Building.Bounds.IsValid)
        {
            const FBox B = Building.Bounds.ExpandBy(40.f);
            AddBox(B.GetCenter(), B.GetExtent(), Color, 20.f);
        }
    }

    UE_LOG(LogSelection, Log,
        TEXT("Highlight ON: %d enabled, %d ground, %d disabled, %d building(s); in-box audit %d red (exportable) + %d orange (mesh, not exportable); %d missing. Click again to hide."),
        Enabled, Ground, Disabled, Buildings.Num(), Red, Orange, Missing);
    return FReply::Handled();
}

FReply FVCCSimPanelSelection::OnHideUnmatchedActorsClicked()
{
    // Toggle: a second click restores everything this tool hid.
    if (HiddenUnmatchedActors.Num() > 0)
    {
        for (const TWeakObjectPtr<AActor>& WA : HiddenUnmatchedActors)
            if (AActor* A = WA.Get())
                A->SetIsTemporarilyHiddenInEditor(false);
        const int32 Restored = HiddenUnmatchedActors.Num();
        HiddenUnmatchedActors.Reset();
        UE_LOG(LogSelection, Log, TEXT("Hide Unmatched OFF: restored %d actor(s)."), Restored);
        return FReply::Handled();
    }

    UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
    if (!World)
        return FReply::Handled();

    FPathGenerator::FBuildingDetectParams BParams;
    BParams.MinBuildingHeight = MinBuildingHeight;
    BParams.MinBuildingFootprint = MinBuildingFootprint;
    BParams.ConnectGap = ConnectGap;
    BParams.ForcedGroundActors = GetForcedGroundActors(World);

    TMap<FString, AActor*> LabelMap;
    for (TActorIterator<AActor> It(World); It; ++It)
        if (AActor* A = *It) LabelMap.Add(A->GetActorLabel(), A);

    TArray<AActor*> EnabledActors;
    for (const TSharedPtr<FVCCSimTargetActorItem>& Item : TargetActorItems)
        if (Item.IsValid() && Item->bEnabled)
            if (AActor** Found = LabelMap.Find(Item->Label))
                if (*Found) EnabledActors.Add(*Found);

    const TArray<FPathGenerator::FOrbitTarget> Buildings =
        FPathGenerator::DetectBuildingsCached(World, EnabledActors, BParams);
    TSet<const AActor*> InBuilding;
    for (const FPathGenerator::FOrbitTarget& B : Buildings)
        for (const TWeakObjectPtr<AActor>& WA : B.Actors)
            if (const AActor* A = WA.Get()) InBuilding.Add(A);

    // Green = enabled, not ground, not merged into any building.
    int32 Hidden = 0;
    for (AActor* A : EnabledActors)
    {
        if (FPathGenerator::IsGroundLikeActor(World, A, BParams)) continue;   // brown, not green
        if (InBuilding.Contains(A)) continue;                                 // matched into a building
        A->SetIsTemporarilyHiddenInEditor(true);
        HiddenUnmatchedActors.Add(A);
        ++Hidden;
    }

    UE_LOG(LogSelection, Log,
        TEXT("Hide Unmatched ON: hid %d unmatched (green) actor(s). Click again to restore."), Hidden);
    return FReply::Handled();
}

FReply FVCCSimPanelSelection::OnForceRecomputeClicked()
{
    FPathGenerator::InvalidateBuildingCache();
    UE_LOG(LogSelection, Log, TEXT("Building-detection cache invalidated; next detection recomputes."));

    // If a highlight is currently shown, rebuild it so the recompute shows immediately. The highlight
    // handler is a toggle: once tears the current one down, once draws a fresh one — and the fresh draw
    // calls DetectBuildingsCached, which now recomputes because the cache was just cleared.
    if (HighlightActor.IsValid())
    {
        OnHighlightTargetsClicked();
        OnHighlightTargetsClicked();
    }
    return FReply::Handled();
}

FReply FVCCSimPanelSelection::OnHideGroundActorsClicked()
{
    // Toggle: a second click restores everything this tool hid.
    if (HiddenGroundActors.Num() > 0)
    {
        for (const TWeakObjectPtr<AActor>& WA : HiddenGroundActors)
            if (AActor* A = WA.Get())
                A->SetIsTemporarilyHiddenInEditor(false);
        const int32 Restored = HiddenGroundActors.Num();
        HiddenGroundActors.Reset();
        UE_LOG(LogSelection, Log, TEXT("Hide Ground OFF: restored %d actor(s)."), Restored);
        return FReply::Handled();
    }

    UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
    if (!World)
        return FReply::Handled();

    // Same ground test and enabled-actor set DetectBuildings runs on, so what stays visible is exactly
    // the structure set being clustered.
    FPathGenerator::FBuildingDetectParams BParams;
    BParams.MinBuildingHeight = MinBuildingHeight;
    BParams.MinBuildingFootprint = MinBuildingFootprint;
    BParams.ConnectGap = ConnectGap;
    BParams.ForcedGroundActors = GetForcedGroundActors(World);

    TMap<FString, AActor*> LabelMap;
    for (TActorIterator<AActor> It(World); It; ++It)
        if (AActor* A = *It) LabelMap.Add(A->GetActorLabel(), A);

    int32 Hidden = 0;
    for (const TSharedPtr<FVCCSimTargetActorItem>& Item : TargetActorItems)
    {
        if (!Item.IsValid() || !Item->bEnabled) continue;
        AActor** Found = LabelMap.Find(Item->Label);
        if (!Found || !*Found) continue;
        if (!FPathGenerator::IsGroundLikeActor(World, *Found, BParams)) continue;

        (*Found)->SetIsTemporarilyHiddenInEditor(true);
        HiddenGroundActors.Add(*Found);
        ++Hidden;
    }

    UE_LOG(LogSelection, Log,
        TEXT("Hide Ground ON: hid %d ground-classified actor(s); remaining visible meshes are the clustered structures. Click again to restore."),
        Hidden);
    return FReply::Handled();
}

FReply FVCCSimPanelSelection::OnFixDuplicateLabelsClicked()
{
    UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
    if (!World)
        return FReply::Handled();

    // Snapshot every non-transient actor and its label up front so generated names dodge ALL
    // existing labels, not just the duplicates seen so far.
    TArray<AActor*> AllActors;
    TSet<FString> UsedLabels;
    for (TActorIterator<AActor> It(World); It; ++It)
    {
        AActor* A = *It;
        if (!A || !IsValid(A) || A->HasAnyFlags(RF_Transient)) continue;
        AllActors.Add(A);
        UsedLabels.Add(A->GetActorLabel());
    }

    const FScopedTransaction Transaction(
        NSLOCTEXT("VCCSim", "FixDuplicateLabels", "Fix Duplicate Actor Labels"));

    TSet<FString> Seen;
    int32 Renamed = 0;
    for (AActor* A : AllActors)
    {
        const FString Label = A->GetActorLabel();
        if (!Seen.Contains(Label))
        {
            Seen.Add(Label);
            continue;
        }

        FString NewLabel;
        int32 Suffix = 1;
        do { NewLabel = FString::Printf(TEXT("%s_%d"), *Label, Suffix++); }
        while (UsedLabels.Contains(NewLabel));

        A->Modify();
        A->SetActorLabel(NewLabel);
        UsedLabels.Add(NewLabel);
        Seen.Add(NewLabel);
        ++Renamed;
    }

    const FString Msg = FString::Printf(
        TEXT("Fixed %d duplicate actor label(s). Re-run Add In Bounds to pick up the now-unique actors, then save the level."),
        Renamed);
    UE_LOG(LogSelection, Log, TEXT("%s"), *Msg);
    FVCCSimUIHelpers::ShowNotification(Msg, false);
    return FReply::Handled();
}

void FVCCSimPanelSelection::OnSelectFlashPawnToggleChanged(ECheckBoxState NewState)
{
    bSelectingFlashPawn = (NewState == ECheckBoxState::Checked);

    if (bSelectingFlashPawn && bSelectingLookAtPath)
    {
        bSelectingLookAtPath = false;
        if (SelectLookAtToggle.IsValid())
        {
            SelectLookAtToggle->SetIsChecked(ECheckBoxState::Unchecked);
        }
    }
}

void FVCCSimPanelSelection::OnSelectLookAtToggleChanged(ECheckBoxState NewState)
{
    bSelectingLookAtPath = (NewState == ECheckBoxState::Checked);

    if (bSelectingLookAtPath && bSelectingFlashPawn)
    {
        bSelectingFlashPawn = false;
        if (SelectFlashPawnToggle.IsValid())
        {
            SelectFlashPawnToggle->SetIsChecked(ECheckBoxState::Unchecked);
        }
    }
}

void FVCCSimPanelSelection::OnRGBCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseRGBCamera = (NewState == ECheckBoxState::Checked);
}

void FVCCSimPanelSelection::OnDepthCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseDepthCamera = (NewState == ECheckBoxState::Checked);
}

void FVCCSimPanelSelection::OnSegmentationCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseSegmentationCamera = (NewState == ECheckBoxState::Checked);
}

void FVCCSimPanelSelection::OnNormalCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseNormalCamera = (NewState == ECheckBoxState::Checked);
}

void FVCCSimPanelSelection::OnRGBLinearCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseRGBLinearCamera = (NewState == ECheckBoxState::Checked);
}

void FVCCSimPanelSelection::OnBaseColorCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseBaseColorCamera = (NewState == ECheckBoxState::Checked);
}

void FVCCSimPanelSelection::OnBaseColorLinearCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseBaseColorLinearCamera = (NewState == ECheckBoxState::Checked);
}

void FVCCSimPanelSelection::OnMaterialPropertiesCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseMaterialPropertiesCamera = (NewState == ECheckBoxState::Checked);
}

void FVCCSimPanelSelection::OnUseRGBCameraClassCheckboxChanged(ECheckBoxState NewState)
{
    bUseRGBCameraClass = (NewState == ECheckBoxState::Checked);
}

void FVCCSimPanelSelection::RefreshCameraAvailability()
{
    bHasRGBCamera = false;
    bHasDepthCamera = false;
    bHasSegmentationCamera = false;
    bHasNormalCamera = false;
    bHasRGBLinearCamera = false;
    bHasBaseColorCamera = false;
    bHasBaseColorLinearCamera = false;
    bHasMaterialPropertiesCamera = false;

    if (!SelectedFlashPawn.IsValid())
    {
        return;
    }

    TArray<URGBCameraComponent*> RGBCameras;
    TArray<UDepthCameraComponent*> DepthCameras;
    TArray<USegCameraComponent*> SegmentationCameras;
    TArray<UNormalCameraComponent*> NormalCameras;
    TArray<URGBLinearCameraComponent*> RGBLinearCameras;
    TArray<UBaseColorCameraComponent*> BaseColorCameras;
    TArray<UBaseColorLinearCameraComponent*> BaseColorLinearCameras;
    TArray<UMaterialPropertiesCameraComponent*> MatPropsCameras;

    SelectedFlashPawn->GetComponents<URGBCameraComponent>(RGBCameras);
    SelectedFlashPawn->GetComponents<UDepthCameraComponent>(DepthCameras);
    SelectedFlashPawn->GetComponents<USegCameraComponent>(SegmentationCameras);
    SelectedFlashPawn->GetComponents<UNormalCameraComponent>(NormalCameras);
    SelectedFlashPawn->GetComponents<URGBLinearCameraComponent>(RGBLinearCameras);
    SelectedFlashPawn->GetComponents<UBaseColorCameraComponent>(BaseColorCameras);
    SelectedFlashPawn->GetComponents<UBaseColorLinearCameraComponent>(BaseColorLinearCameras);
    SelectedFlashPawn->GetComponents<UMaterialPropertiesCameraComponent>(MatPropsCameras);

    bHasRGBCamera = (RGBCameras.Num() > 0);
    bHasDepthCamera = (DepthCameras.Num() > 0);
    bHasSegmentationCamera = (SegmentationCameras.Num() > 0);
    bHasNormalCamera = (NormalCameras.Num() > 0);
    bHasRGBLinearCamera = (RGBLinearCameras.Num() > 0);
    bHasBaseColorCamera = (BaseColorCameras.Num() > 0);
    bHasBaseColorLinearCamera = (BaseColorLinearCameras.Num() > 0);
    bHasMaterialPropertiesCamera = (MatPropsCameras.Num() > 0);
}

void FVCCSimPanelSelection::WarmupCameras()
{
    if (!SelectedFlashPawn.IsValid())
    {
        return;
    }

    TArray<UCameraBaseComponent*> AllCameras;
    SelectedFlashPawn->GetComponents<UCameraBaseComponent>(AllCameras);

    for (UCameraBaseComponent* Camera : AllCameras)
    {
        if (Camera)
        {
            Camera->WarmupCapture();
        }
    }

    bIsWarmedUp = true;
    UE_LOG(LogSelection, Log, TEXT("Warmed up %d camera(s) on FlashPawn"), AllCameras.Num());
}

void FVCCSimPanelSelection::ClearSelections()
{
    SelectedFlashPawn.Reset();
    SelectedLookAtPath.Reset();
    bSelectingFlashPawn = false;
    bSelectingLookAtPath = false;

    bHasRGBCamera = false;
    bHasDepthCamera = false;
    bHasSegmentationCamera = false;
    bHasNormalCamera = false;
    bHasRGBLinearCamera = false;
    bHasBaseColorCamera = false;
    bHasBaseColorLinearCamera = false;
    bHasMaterialPropertiesCamera = false;
    bUseRGBCamera = true;
    bUseDepthCamera = false;
    bUseSegmentationCamera = false;
    bUseNormalCamera = false;
    bUseRGBLinearCamera = false;
    bUseBaseColorCamera = false;
    bUseBaseColorLinearCamera = false;
    bUseMaterialPropertiesCamera = false;
    bUseRGBCameraClass = false;
}

bool FVCCSimPanelSelection::HasAnyActiveCamera() const
{
    return (bHasRGBCamera && bUseRGBCamera) ||
           (bHasDepthCamera && bUseDepthCamera) ||
           (bHasSegmentationCamera && bUseSegmentationCamera) ||
           (bHasNormalCamera && bUseNormalCamera) ||
           (bHasRGBLinearCamera && bUseRGBLinearCamera) ||
           (bHasBaseColorCamera && bUseBaseColorCamera) ||
           (bHasBaseColorLinearCamera && bUseBaseColorLinearCamera) ||
           (bHasMaterialPropertiesCamera && bUseMaterialPropertiesCamera);
}
