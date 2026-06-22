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
#include "Sensors/BaseColorCamera.h"
#include "Sensors/MaterialPropertiesCamera.h"
#include "Utils/GTMaterialExporter.h"
#include "Utils/VCCSimUIHelpers.h"
#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#include "Framework/Application/SlateApplication.h"
#include "Misc/Paths.h"
#include "Components/LineBatchComponent.h"
#include "Components/MeshComponent.h"
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
    bExcludeClutter = Config.bExcludeClutter;

    TargetActorItems.Empty();
    for (int32 i = 0; i < Config.Labels.Num(); ++i)
    {
        TSharedPtr<FVCCSimTargetActorItem> Item = MakeShared<FVCCSimTargetActorItem>();
        Item->Label = Config.Labels[i];
        Item->bEnabled = Config.EnabledFlags.IsValidIndex(i) ? Config.EnabledFlags[i] : true;
        TargetActorItems.Add(Item);
    }

    if (TargetActorListView.IsValid())
    {
        TargetActorListView->RequestListRefresh();
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
    Config.BoundsMin = BoundsMin;
    Config.BoundsMax = BoundsMax;
    Config.bExcludeClutter = bExcludeClutter;
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

bool FVCCSimPanelSelection::IsClutterActor(const AActor* Actor) const
{
    if (!Actor) return true;
    if (Actor->IsA<AFlashPawn>() || Actor->IsA<AVCCSimLookAtPath>()) return true;

    static const TCHAR* ClutterTokens[] = {
        TEXT("tree"), TEXT("foliage"), TEXT("plant"), TEXT("bush"), TEXT("grass"),
        TEXT("leaf"), TEXT("branch"), TEXT("hedge"), TEXT("shrub"),
        TEXT("car"), TEXT("vehicle"), TEXT("truck"), TEXT("bus"), TEXT("van"),
        TEXT("sedan"), TEXT("suv"), TEXT("taxi"), TEXT("traffic"),
        TEXT("pedestrian"), TEXT("people"), TEXT("person"), TEXT("npc"), TEXT("character"),
        TEXT("fence"), TEXT("hydrant"), TEXT("parking_meter"), TEXT("parkingmeter"),
        TEXT("road"), TEXT("street"), TEXT("ground"), TEXT("terrain"), TEXT("pavement"),
        TEXT("sidewalk"), TEXT("asphalt"), TEXT("curb"), TEXT("crosswalk"),
        TEXT("landscape"), TEXT("instancedfoliage")
    };

    const FString Label = Actor->GetActorLabel().ToLower();
    const FString ClassName = Actor->GetClass()->GetName().ToLower();
    for (const TCHAR* Token : ClutterTokens)
    {
        if (Label.Contains(Token) || ClassName.Contains(Token))
        {
            return true;
        }
    }
    return false;
}

FReply FVCCSimPanelSelection::OnFillBoundsFromSelectionClicked()
{
    USelection* Sel = GEditor ? GEditor->GetSelectedActors() : nullptr;
    if (!Sel || Sel->Num() == 0)
    {
        UE_LOG(LogSelection, Warning, TEXT("Fill Bounds: no actor selected in the viewport"));
        return FReply::Handled();
    }

    FBox Box(ForceInit);
    for (int32 i = 0; i < Sel->Num(); ++i)
    {
        if (AActor* Actor = Cast<AActor>(Sel->GetSelectedObject(i)))
        {
            FVector Origin, Extent;
            Actor->GetActorBounds(false, Origin, Extent);
            Box += FBox(Origin - Extent, Origin + Extent);
        }
    }

    if (Box.IsValid)
    {
        BoundsMin = Box.Min;
        BoundsMax = Box.Max;
        SaveTargetActorsToConfig();
        UE_LOG(LogSelection, Log, TEXT("Fill Bounds: [%s .. %s]"),
            *BoundsMin.ToString(), *BoundsMax.ToString());
    }
    return FReply::Handled();
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
    int32 SkippedClutter = 0;
    for (TActorIterator<AActor> It(World); It; ++It)
    {
        AActor* Actor = *It;
        if (!Actor || !IsValid(Actor)) continue;
        if (!FGTMaterialExporter::HasExportableMeshMaterials(Actor)) continue;

        FVector Origin, Extent;
        Actor->GetActorBounds(false, Origin, Extent);
        if (!Query.Intersect(FBox(Origin - Extent, Origin + Extent))) continue;

        if (bExcludeClutter && IsClutterActor(Actor))
        {
            ++SkippedClutter;
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

    const FString Msg = FString::Printf(
        TEXT("Add In Bounds: added %d actor(s), skipped %d clutter."), Added, SkippedClutter);
    UE_LOG(LogSelection, Log, TEXT("%s"), *Msg);
    FVCCSimUIHelpers::ShowNotification(Msg, false);
    return FReply::Handled();
}

FReply FVCCSimPanelSelection::OnExportGTMeshClicked()
{
    if (bGTExportInProgress)
        return FReply::Handled();

    UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
    if (!World)
        return FReply::Handled();

    const TArray<FString> Labels = GetEnabledTargetActorLabels();
    if (Labels.Num() == 0)
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("No enabled target actors to export."), true);
        return FReply::Handled();
    }

    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (!DesktopPlatform)
        return FReply::Handled();

    const void* ParentWindowHandle = FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr);
    FString ChosenDir;
    if (!DesktopPlatform->OpenDirectoryDialog(
            ParentWindowHandle, TEXT("Choose GT mesh export folder"),
            FPaths::ProjectSavedDir(), ChosenDir) || ChosenDir.IsEmpty())
        return FReply::Handled();

    const FString BaseDir = ChosenDir / TEXT("gt_materials");
    const FString SceneName = TEXT("GTMesh");
    const int32 TextureResolution = 2048;   // vestigial (no baking); kept for manifest/signature
    const FString Signature = FGTMaterialExporter::ComputeSignature(
        World, Labels, SceneName, TextureResolution, false, 0.f, false);

    if (!GTMaterialExporter.IsValid())
        GTMaterialExporter = MakeShared<FGTMaterialExporter>();

    bGTExportInProgress = true;
    UE_LOG(LogSelection, Log, TEXT("GT mesh export: %d actor(s) -> %s"), Labels.Num(), *BaseDir);
    GTMaterialExporter->ExportMaterials(
        Labels, TArray<FGTFoliageExportEntry>(), World, BaseDir, SceneName,
        TextureResolution, Signature,
        FSimpleDelegate::CreateLambda([this]() { bGTExportInProgress = false; }));

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

    // Boxes go into a ULineBatchComponent on a transient actor — self-contained, so toggling off
    // just destroys the actor. (DrawDebug's persistent lines are shared with the capture / path-viz
    // flush and could not be cleared in isolation.)
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

    ULineBatchComponent* LineBatch = NewObject<ULineBatchComponent>(VizActor);
    LineBatch->SetupAttachment(Root);
    LineBatch->SetHiddenInGame(true);
    LineBatch->RegisterComponent();
    HighlightActor = VizActor;

    // Lines persist until the actor is destroyed; the line-batch component does not tick down in the
    // editor, and the huge lifetime is a belt-and-suspenders guard in case it ever does.
    const float BoxLifeTime = 1.0e8f;
    auto AddBox = [&](const FVector& Origin, const FVector& Extent, const FLinearColor& Color, float Thickness)
    {
        LineBatch->DrawBox(Origin, Extent, Color, BoxLifeTime, 0, Thickness);
    };

    // 1) Every entry of the shared list: green = enabled, gray = disabled.
    TMap<FString, AActor*> LabelMap;
    for (TActorIterator<AActor> It(World); It; ++It)
        if (AActor* A = *It) LabelMap.Add(A->GetActorLabel(), A);

    TSet<FString> ListedLabels;
    int32 Enabled = 0, Disabled = 0, Missing = 0;
    for (const TSharedPtr<FVCCSimTargetActorItem>& Item : TargetActorItems)
    {
        if (!Item.IsValid()) continue;
        ListedLabels.Add(Item->Label);
        AActor** Found = LabelMap.Find(Item->Label);
        if (!Found || !*Found) { ++Missing; continue; }

        FVector Origin, Extent;
        (*Found)->GetActorBounds(false, Origin, Extent);
        AddBox(Origin, Extent, Item->bEnabled ? FLinearColor::Green : FLinearColor(0.3f, 0.3f, 0.3f), 8.f);
        if (Item->bEnabled) ++Enabled; else ++Disabled;
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
            if (!Actor || !IsValid(Actor) || Actor == VizActor) continue;
            if (Actor->IsA<AFlashPawn>() || Actor->IsA<AVCCSimLookAtPath>()) continue;
            if (ListedLabels.Contains(Actor->GetActorLabel())) continue;

            TArray<UMeshComponent*> MeshComps;
            Actor->GetComponents<UMeshComponent>(MeshComps);
            if (MeshComps.Num() == 0) continue;

            FVector Origin, Extent;
            Actor->GetActorBounds(false, Origin, Extent);
            if (!Query.Intersect(FBox(Origin - Extent, Origin + Extent))) continue;

            const bool bExportable = FGTMaterialExporter::HasExportableMeshMaterials(Actor);
            AddBox(Origin, Extent, bExportable ? FLinearColor::Red : FLinearColor(1.f, 0.5f, 0.f), 8.f);
            if (bExportable) ++Red; else ++Orange;
        }
    }

    // 3) Building clusters: the same union-find grouping path generation orbits as one building.
    //    One thick cyan box per cluster, slightly inflated so it reads around the green actor boxes.
    TArray<AActor*> EnabledActors;
    for (const TSharedPtr<FVCCSimTargetActorItem>& Item : TargetActorItems)
        if (Item.IsValid() && Item->bEnabled)
            if (AActor** Found = LabelMap.Find(Item->Label))
                if (*Found) EnabledActors.Add(*Found);

    const TArray<FPathGenerator::FOrbitTarget> Clusters =
        FPathGenerator::ClusterTargetsByProximity(EnabledActors);
    for (const FPathGenerator::FOrbitTarget& Cluster : Clusters)
    {
        if (!Cluster.Bounds.IsValid) continue;
        const FBox B = Cluster.Bounds.ExpandBy(40.f);
        AddBox(B.GetCenter(), B.GetExtent(), FLinearColor(0.f, 0.8f, 1.f), 20.f);
    }

    const FString Msg = FString::Printf(
        TEXT("Highlight ON: %d enabled, %d disabled, %d building cluster(s); in-box audit %d red (exportable) + %d orange (mesh, not exportable); %d missing. Click again to hide."),
        Enabled, Disabled, Clusters.Num(), Red, Orange, Missing);
    UE_LOG(LogSelection, Log, TEXT("%s"), *Msg);
    FVCCSimUIHelpers::ShowNotification(Msg, Missing > 0);
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

void FVCCSimPanelSelection::OnBaseColorCameraCheckboxChanged(ECheckBoxState NewState)
{
    bUseBaseColorCamera = (NewState == ECheckBoxState::Checked);
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
    bHasBaseColorCamera = false;
    bHasMaterialPropertiesCamera = false;

    if (!SelectedFlashPawn.IsValid())
    {
        return;
    }

    TArray<URGBCameraComponent*> RGBCameras;
    TArray<UDepthCameraComponent*> DepthCameras;
    TArray<USegCameraComponent*> SegmentationCameras;
    TArray<UNormalCameraComponent*> NormalCameras;
    TArray<UBaseColorCameraComponent*> BaseColorCameras;
    TArray<UMaterialPropertiesCameraComponent*> MatPropsCameras;

    SelectedFlashPawn->GetComponents<URGBCameraComponent>(RGBCameras);
    SelectedFlashPawn->GetComponents<UDepthCameraComponent>(DepthCameras);
    SelectedFlashPawn->GetComponents<USegCameraComponent>(SegmentationCameras);
    SelectedFlashPawn->GetComponents<UNormalCameraComponent>(NormalCameras);
    SelectedFlashPawn->GetComponents<UBaseColorCameraComponent>(BaseColorCameras);
    SelectedFlashPawn->GetComponents<UMaterialPropertiesCameraComponent>(MatPropsCameras);

    bHasRGBCamera = (RGBCameras.Num() > 0);
    bHasDepthCamera = (DepthCameras.Num() > 0);
    bHasSegmentationCamera = (SegmentationCameras.Num() > 0);
    bHasNormalCamera = (NormalCameras.Num() > 0);
    bHasBaseColorCamera = (BaseColorCameras.Num() > 0);
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
    bHasBaseColorCamera = false;
    bHasMaterialPropertiesCamera = false;
    bUseRGBCamera = true;
    bUseDepthCamera = false;
    bUseSegmentationCamera = false;
    bUseNormalCamera = false;
    bUseBaseColorCamera = false;
    bUseMaterialPropertiesCamera = false;
    bUseRGBCameraClass = false;
}

bool FVCCSimPanelSelection::HasAnyActiveCamera() const
{
    return (bHasRGBCamera && bUseRGBCamera) ||
           (bHasDepthCamera && bUseDepthCamera) ||
           (bHasSegmentationCamera && bUseSegmentationCamera) ||
           (bHasNormalCamera && bUseNormalCamera) ||
           (bHasBaseColorCamera && bUseBaseColorCamera) ||
           (bHasMaterialPropertiesCamera && bUseMaterialPropertiesCamera);
}
