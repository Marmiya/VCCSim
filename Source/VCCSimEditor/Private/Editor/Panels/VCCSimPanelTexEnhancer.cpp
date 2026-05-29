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

#include "Editor/Panels/VCCSimPanelTexEnhancer.h"
#include "Utils/VCCSimSunPositionHelper.h"
#include "Utils/VCCSimUIHelpers.h"
#include "Utils/VCCSimConfigManager.h"
#include "Utils/ConfigParser.h"
#include "Utils/LightingManager.h"
#include "Utils/GTMaterialExporter.h"
#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/StaticMesh.h"
#include "Engine/Texture.h"
#include "Engine/Texture2D.h"
#include "Materials/MaterialInterface.h"
#include "Components/StaticMeshComponent.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "InstancedFoliageActor.h"
#include "EngineUtils.h"
#include "DrawDebugHelpers.h"
#include "IMeshMergeUtilities.h"
#include "MeshMergeModule.h"
#include "Engine/MeshMerging.h"
#include "Misc/ScopeExit.h"
#include "Selection.h"
#include "MeshDescription.h"
#include "Modules/ModuleManager.h"

#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "HAL/PlatformFilemanager.h"
#include "HAL/PlatformProcess.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#include "Framework/Application/SlateApplication.h"
#include "Editor.h"
#include "TimerManager.h"
#include "Async/Async.h"

DEFINE_LOG_CATEGORY_STATIC(LogTexEnhancerPanel, Log, All);

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

FVCCSimPanelTexEnhancer::FVCCSimPanelTexEnhancer()
{
    GTMaterialExporter = MakeShared<FGTMaterialExporter>();
    NanobananaManager  = MakeShared<FNanobananaManager>();

    for (int32 i = 0; i < MaxLightingEntries; ++i)
    {
        SetAElevationValue[i] = SetAElevation[i];
        SetAAzimuthValue[i]   = SetAAzimuth[i];
        SetBElevationValue[i] = SetBElevation[i];
        SetBAzimuthValue[i]   = SetBAzimuth[i];
    }
    SunCalcLatValue      = SunCalcLatitude;
    SunCalcLonValue      = SunCalcLongitude;
    SunCalcTZValue       = SunCalcTimeZone;
    SunCalcYearValue     = SunCalcYear;
    SunCalcMonthValue    = SunCalcMonth;
    SunCalcDayValue      = SunCalcDay;
    SunCalcHourValue     = SunCalcHour;
    SunCalcMinuteValue   = SunCalcMinute;
    SunCalcFillSlotValue = SunCalcFillSlot;
    GTTexResValue        = GTTextureResolution;
    NearbyRadiusValue    = NearbyRadius;
    DayCycleSpeedValue   = DayCycleSpeed;

    NanobananaHFOVValue         = NanobananaHFOV;
    NanobananaImageWidthValue   = NanobananaImageWidth;
    NanobananaImageHeightValue  = NanobananaImageHeight;
    NanobananaOverlayAlphaValue = NanobananaOverlayAlpha;
}

FVCCSimPanelTexEnhancer::~FVCCSimPanelTexEnhancer()
{
    Cleanup();
}

// ============================================================================
// INITIALIZATION
// ============================================================================

void FVCCSimPanelTexEnhancer::Initialize()
{
    LightingManager = MakeShared<FLightingManager>(GEditor->GetEditorWorldContext().World());
    LoadPaths();
    UE_LOG(LogTexEnhancerPanel, Log, TEXT("TexEnhancer panel initialized"));
}

void FVCCSimPanelTexEnhancer::Cleanup()
{
    LightingManager.Reset();
    if (GEditor)
    {
        GEditor->GetTimerManager()->ClearTimer(StatusTimerHandle);
        GEditor->GetTimerManager()->ClearTimer(ExpansionVizTimerHandle);
        if (UWorld* World = GEditor->GetEditorWorldContext().World())
        {
            FlushPersistentDebugLines(World);
        }
    }
    bVisualizeExpansion = false;
    bNanobananaInProgress = false;

    if (PipelineProcHandle.IsValid())
    {
        FPlatformProcess::TerminateProc(PipelineProcHandle, false);
        FPlatformProcess::CloseProc(PipelineProcHandle);
    }

    SavePaths();
    UE_LOG(LogTexEnhancerPanel, Log, TEXT("TexEnhancer panel cleaned up"));
}

void FVCCSimPanelTexEnhancer::SetSelectionManager(TSharedPtr<FVCCSimPanelSelection> InSelectionManager)
{
    SelectionManager = InSelectionManager;
}

void FVCCSimPanelTexEnhancer::LoadFromConfigManager()
{
    const auto& Config = FVCCSimConfigManager::Get().GetTexEnhancerConfig();

    if (!Config.OutputDirectory.IsEmpty())
    {
        OutputDirectory = Config.OutputDirectory;
        if (OutputDirTextBox.IsValid())
        {
            OutputDirTextBox->SetText(FText::FromString(OutputDirectory));
        }
    }
    if (!Config.SceneName.IsEmpty())
    {
        SceneName = Config.SceneName;
        if (SceneNameTextBox.IsValid())
        {
            SceneNameTextBox->SetText(FText::FromString(SceneName));
        }
    }
    if (!Config.TexEnhancerScriptPath.IsEmpty())
    {
        TexEnhancerScriptPath = Config.TexEnhancerScriptPath;
        if (TexEnhancerScriptTextBox.IsValid())
        {
            TexEnhancerScriptTextBox->SetText(FText::FromString(TexEnhancerScriptPath));
        }
    }
    if (!Config.EstimatedMaterialsDir.IsEmpty())
    {
        EstimatedMaterialsDir = Config.EstimatedMaterialsDir;
        if (EstimatedMaterialsDirTextBox.IsValid())
        {
            EstimatedMaterialsDirTextBox->SetText(FText::FromString(EstimatedMaterialsDir));
        }
    }

    if (!Config.GTActorLabels.IsEmpty())
    {
        GTActorListItems.Empty();
        for (const FString& Label : Config.GTActorLabels)
            GTActorListItems.Add(MakeShareable(new FString(Label)));
        if (GTActorListView.IsValid())
            GTActorListView->RequestListRefresh();
    }

    if (!Config.NanobananaResultDir.IsEmpty())
    {
        NanobananaResultDir = Config.NanobananaResultDir;
        if (NanobananaResultDirTextBox.IsValid())
            NanobananaResultDirTextBox->SetText(FText::FromString(NanobananaResultDir));
    }
    if (!Config.NanobananaPosesFile.IsEmpty())
    {
        NanobananaPosesFile = Config.NanobananaPosesFile;
        if (NanobananaPosesFileTextBox.IsValid())
            NanobananaPosesFileTextBox->SetText(FText::FromString(NanobananaPosesFile));
    }
    if (!Config.NanobananaManifestFile.IsEmpty())
    {
        NanobananaManifestFile = Config.NanobananaManifestFile;
        if (NanobananaManifestFileTextBox.IsValid())
            NanobananaManifestFileTextBox->SetText(FText::FromString(NanobananaManifestFile));
    }
    NanobananaHFOV         = Config.NanobananaHFOV;
    NanobananaImageWidth   = Config.NanobananaImageWidth;
    NanobananaImageHeight  = Config.NanobananaImageHeight;
    NanobananaOverlayAlpha = Config.NanobananaOverlayAlpha;
    NanobananaHFOVValue         = NanobananaHFOV;
    NanobananaImageWidthValue   = NanobananaImageWidth;
    NanobananaImageHeightValue  = NanobananaImageHeight;
    NanobananaOverlayAlphaValue = NanobananaOverlayAlpha;

    bIncludeNearbyMeshes = Config.bIncludeNearbyMeshes;
    bMergeNearbyMeshes   = Config.bMergeNearbyMeshes;
    NearbyRadius         = Config.NearbyRadius;
    NearbyRadiusValue    = NearbyRadius;
}

// ============================================================================
// SECTION 1: DATASET CONFIGURATION
// ============================================================================

FReply FVCCSimPanelTexEnhancer::OnBrowseOutputDirClicked()
{
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (!DesktopPlatform) return FReply::Handled();

    FString SelectedDir;
    void* ParentWindowHandle = const_cast<void*>(FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr));

    if (DesktopPlatform->OpenDirectoryDialog(ParentWindowHandle, TEXT("Select Output Directory"), OutputDirectory, SelectedDir))
    {
        OutputDirectory = SelectedDir;
        if (OutputDirTextBox.IsValid())
        {
            OutputDirTextBox->SetText(FText::FromString(OutputDirectory));
        }
        SavePaths();
    }

    return FReply::Handled();
}

// ============================================================================
// SECTION 2: LIGHTING SCHEDULE
// ============================================================================

FReply FVCCSimPanelTexEnhancer::OnApplySetALightingClicked(int32 Index)
{
    if (Index < 0 || Index >= NumLightingSetA || !LightingManager.IsValid()) return FReply::Handled();
    LightingManager->ApplyLightingCondition(SetAElevation[Index], SetAAzimuth[Index]);
    if (LightingStatusTextBlock.IsValid())
        LightingStatusTextBlock->SetText(FText::FromString(FString::Printf(TEXT("Set-A%d Applied: Elev=%.1f\u00B0 Az=%.1f\u00B0"), Index + 1, SetAElevation[Index], SetAAzimuth[Index])));
    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnApplySetBLightingClicked(int32 Index)
{
    if (Index < 0 || Index >= NumLightingSetB || !LightingManager.IsValid()) return FReply::Handled();
    LightingManager->ApplyLightingCondition(SetBElevation[Index], SetBAzimuth[Index]);
    if (LightingStatusTextBlock.IsValid())
        LightingStatusTextBlock->SetText(FText::FromString(FString::Printf(TEXT("Set-B%d Applied: Elev=%.1f\u00B0 Az=%.1f\u00B0"), Index + 1, SetBElevation[Index], SetBAzimuth[Index])));
    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnCalculateSunPositionClicked()
{
    if (!LightingManager.IsValid()) return FReply::Handled();

    FVCCSimSunPositionHelper::FSunParams Params;
    Params.Latitude  = SunCalcLatitude;
    Params.Longitude = SunCalcLongitude;
    Params.TimeZone  = SunCalcTimeZone;
    Params.Year      = SunCalcYear;
    Params.Month     = SunCalcMonth;
    Params.Day       = SunCalcDay;
    Params.Hour      = SunCalcHour;
    Params.Minute    = SunCalcMinute;

    TPair<float, float> SunPos = LightingManager->CalculateAndApplySunPosition(Params);
    SunCalcElevation = SunPos.Key;
    SunCalcAzimuth = SunPos.Value;
    
    if (LightingStatusTextBlock.IsValid())
        LightingStatusTextBlock->SetText(FText::FromString(FString::Printf(TEXT("Calculated & Applied: Elev=%.1f\u00B0 Az=%.1f\u00B0"), SunCalcElevation, SunCalcAzimuth)));

    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnFillSetAFromSunPositionClicked()
{
    int32 SlotIdx = FMath::Clamp(SunCalcFillSlot - 1, 0, NumLightingSetA - 1);
    SetAElevation[SlotIdx]      = SunCalcElevation;
    SetAAzimuth[SlotIdx]        = SunCalcAzimuth;
    SetAElevationValue[SlotIdx] = SunCalcElevation;
    SetAAzimuthValue[SlotIdx]   = SunCalcAzimuth;

    if (SetAElevationSpinBox[SlotIdx].IsValid())
    {
        SetAElevationSpinBox[SlotIdx]->Invalidate(EInvalidateWidgetReason::Paint);
    }
    if (SetAAzimuthSpinBox[SlotIdx].IsValid())
    {
        SetAAzimuthSpinBox[SlotIdx]->Invalidate(EInvalidateWidgetReason::Paint);
    }

    UE_LOG(LogTexEnhancerPanel, Log,
        TEXT("Sun position filled into Set-A slot %d: Elevation=%.1f°  Azimuth=%.1f°"),
        SunCalcFillSlot, SunCalcElevation, SunCalcAzimuth);

    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnFillSetBFromSunPositionClicked()
{
    int32 SlotIdx = FMath::Clamp(SunCalcFillSlot - 1, 0, NumLightingSetB - 1);
    SetBElevation[SlotIdx]      = SunCalcElevation;
    SetBAzimuth[SlotIdx]        = SunCalcAzimuth;
    SetBElevationValue[SlotIdx] = SunCalcElevation;
    SetBAzimuthValue[SlotIdx]   = SunCalcAzimuth;

    if (SetBElevationSpinBox[SlotIdx].IsValid())
    {
        SetBElevationSpinBox[SlotIdx]->Invalidate(EInvalidateWidgetReason::Paint);
    }
    if (SetBAzimuthSpinBox[SlotIdx].IsValid())
    {
        SetBAzimuthSpinBox[SlotIdx]->Invalidate(EInvalidateWidgetReason::Paint);
    }

    UE_LOG(LogTexEnhancerPanel, Log,
        TEXT("Sun position filled into Set-B slot %d: Elevation=%.1f°  Azimuth=%.1f°"),
        SunCalcFillSlot, SunCalcElevation, SunCalcAzimuth);

    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnToggleDayCycleClicked()
{
    if (!LightingManager.IsValid()) return FReply::Handled();

    bDayCycleActive = !bDayCycleActive;
    
    FVCCSimSunPositionHelper::FSunParams Params;
    Params.Latitude  = SunCalcLatitude;
    Params.Longitude = SunCalcLongitude;
    Params.TimeZone  = SunCalcTimeZone;
    Params.Year      = SunCalcYear;
    Params.Month     = SunCalcMonth;
    Params.Day       = SunCalcDay;

    LightingManager->ToggleDayCycle(bDayCycleActive, Params, DayCycleSpeed);

    return FReply::Handled();
}

// ============================================================================
// SECTION 4: GT MATERIAL EXPORT
// ============================================================================

FReply FVCCSimPanelTexEnhancer::OnAddSelectedActorsClicked()
{
    USelection* Sel = GEditor ? GEditor->GetSelectedActors() : nullptr;
    if (!Sel) return FReply::Handled();

    bool bAdded = false;
    for (int32 i = 0; i < Sel->Num(); ++i)
    {
        AStaticMeshActor* SMA = Cast<AStaticMeshActor>(Sel->GetSelectedObject(i));
        if (!SMA) continue;

        const FString Label = SMA->GetActorLabel();
        bool bDuplicate = GTActorListItems.ContainsByPredicate(
            [&Label](const TSharedPtr<FString>& P) { return P.IsValid() && *P == Label; });

        if (!bDuplicate)
        {
            GTActorListItems.Add(MakeShareable(new FString(Label)));
            bAdded = true;
        }
    }

    if (bAdded && GTActorListView.IsValid())
        GTActorListView->RequestListRefresh();
    if (bAdded)
        SavePaths();

    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnRemoveFromGTListClicked()
{
    if (!GTActorListView.IsValid()) return FReply::Handled();

    TArray<TSharedPtr<FString>> Selected = GTActorListView->GetSelectedItems();
    if (Selected.IsEmpty()) return FReply::Handled();

    for (const TSharedPtr<FString>& Item : Selected)
    {
        if (!Item.IsValid()) continue;
        const FString ItemStr = *Item;
        GTActorListItems.RemoveAll([&ItemStr](const TSharedPtr<FString>& P)
        {
            return P.IsValid() && *P == ItemStr;
        });
    }

    GTActorListView->ClearSelection();
    GTActorListView->RequestListRefresh();
    return FReply::Handled();
}

void FVCCSimPanelTexEnhancer::BuildSeedShapes(
    UWorld* World,
    const TArray<AStaticMeshActor*>& Seeds,
    float ExpandCm,
    int32 NumProbes,
    TArray<FSeedShape>& OutShapes) const
{
    OutShapes.Reset();
    if (!World || Seeds.IsEmpty() || NumProbes < 8) return;

    TSet<AActor*> SeedSet;
    for (AStaticMeshActor* S : Seeds) if (IsValid(S)) SeedSet.Add(S);

    const int32 NumSlices = 3;
    const float SliceT[NumSlices] = { 0.15f, 0.5f, 0.85f };

    FCollisionQueryParams BaseQueryParams;
    BaseQueryParams.bTraceComplex = true;

    for (AStaticMeshActor* Seed : Seeds)
    {
        if (!IsValid(Seed)) continue;

        FVector Origin, Extent;
        Seed->GetActorBounds(false, Origin, Extent);
        const FBox SeedBox(Origin - Extent, Origin + Extent);
        const FVector2D CenterXY(Origin.X, Origin.Y);
        const float AABBRadius = FMath::Sqrt(Extent.X * Extent.X + Extent.Y * Extent.Y);
        const float SearchRadius = AABBRadius * 4.f + 2000.f;

        TArray<float> MaxHitDist;
        MaxHitDist.Init(-1.f, NumProbes);

        for (int32 s = 0; s < NumSlices; ++s)
        {
            const float Z = FMath::Lerp(SeedBox.Min.Z, SeedBox.Max.Z, SliceT[s]);
            const FVector SliceCenter(CenterXY.X, CenterXY.Y, Z);

            for (int32 a = 0; a < NumProbes; ++a)
            {
                const float AngleRad = (2.f * PI * a) / NumProbes;
                const FVector Dir(FMath::Cos(AngleRad), FMath::Sin(AngleRad), 0.f);
                const FVector TraceStart = SliceCenter + Dir * SearchRadius;

                FCollisionQueryParams ProbeParams = BaseQueryParams;
                int32 MaxPenetrations = 16;
                float HitDist = -1.f;

                while (MaxPenetrations-- > 0)
                {
                    FHitResult Hit;
                    if (!World->LineTraceSingleByChannel(
                            Hit, TraceStart, SliceCenter, ECC_Visibility, ProbeParams))
                        break;
                    AActor* HitActor = Hit.GetActor();
                    if (!HitActor) break;
                    if (SeedSet.Contains(HitActor))
                    {
                        const FVector Delta = Hit.ImpactPoint - SliceCenter;
                        HitDist = FMath::Sqrt(Delta.X * Delta.X + Delta.Y * Delta.Y);
                        break;
                    }
                    ProbeParams.AddIgnoredActor(HitActor);
                }

                if (HitDist > MaxHitDist[a]) MaxHitDist[a] = HitDist;
            }
        }

        FSeedShape Shape;
        Shape.Polygon.Reserve(NumProbes);
        Shape.MinZ = SeedBox.Min.Z;
        Shape.MaxZ = SeedBox.Max.Z;
        Shape.VizCenter = Origin;

        for (int32 a = 0; a < NumProbes; ++a)
        {
            const float AngleRad = (2.f * PI * a) / NumProbes;
            const FVector2D Dir2D(FMath::Cos(AngleRad), FMath::Sin(AngleRad));

            float Dist;
            if (MaxHitDist[a] >= 0.f)
            {
                Dist = MaxHitDist[a] + ExpandCm;
            }
            else
            {
                const float tX = FMath::Abs(Dir2D.X) > KINDA_SMALL_NUMBER ? Extent.X / FMath::Abs(Dir2D.X) : FLT_MAX;
                const float tY = FMath::Abs(Dir2D.Y) > KINDA_SMALL_NUMBER ? Extent.Y / FMath::Abs(Dir2D.Y) : FLT_MAX;
                Dist = FMath::Min(tX, tY) + ExpandCm;
            }
            Shape.Polygon.Add(CenterXY + Dir2D * Dist);
        }

        OutShapes.Add(MoveTemp(Shape));
    }
}

static bool PointInPoly2D(const FVector2D& P, const TArray<FVector2D>& Poly)
{
    bool bInside = false;
    const int32 N = Poly.Num();
    for (int32 i = 0, j = N - 1; i < N; j = i++)
    {
        const FVector2D& A = Poly[i];
        const FVector2D& B = Poly[j];
        if (((A.Y > P.Y) != (B.Y > P.Y)) &&
            (P.X < (B.X - A.X) * (P.Y - A.Y) / (B.Y - A.Y) + A.X))
        {
            bInside = !bInside;
        }
    }
    return bInside;
}

void FVCCSimPanelTexEnhancer::CollectNearbyTargets(
    UWorld* World,
    const TArray<AStaticMeshActor*>& SeedActors,
    float RadiusCm,
    TArray<FString>& InOutActorLabels,
    TArray<FGTFoliageExportEntry>& OutFoliageEntries) const
{
    if (!World || SeedActors.IsEmpty() || RadiusCm < 0.f) return;

    TSet<AActor*> SeedSet;
    for (AStaticMeshActor* Seed : SeedActors)
        if (IsValid(Seed)) SeedSet.Add(Seed);

    TArray<FSeedShape> Shapes;
    BuildSeedShapes(World, SeedActors, RadiusCm, /*NumProbes=*/180, Shapes);
    if (Shapes.IsEmpty()) return;

    auto OverlapsAnySeed = [&Shapes](const FBox& CandBox) -> bool
    {
        const FVector2D Min2D(CandBox.Min.X, CandBox.Min.Y);
        const FVector2D Max2D(CandBox.Max.X, CandBox.Max.Y);
        const FVector2D Ctr2D((Min2D.X + Max2D.X) * 0.5f, (Min2D.Y + Max2D.Y) * 0.5f);
        const FVector2D Corners[5] = {
            Min2D, { Max2D.X, Min2D.Y }, Max2D, { Min2D.X, Max2D.Y }, Ctr2D
        };

        for (const FSeedShape& S : Shapes)
        {
            if (CandBox.Max.Z < S.MinZ || CandBox.Min.Z > S.MaxZ) continue;

            for (const FVector2D& C : Corners)
            {
                if (PointInPoly2D(C, S.Polygon)) return true;
            }
            for (const FVector2D& V : S.Polygon)
            {
                if (V.X >= Min2D.X && V.X <= Max2D.X &&
                    V.Y >= Min2D.Y && V.Y <= Max2D.Y) return true;
            }
        }
        return false;
    };

    TSet<FString> ExistingLabels;
    for (const FString& L : InOutActorLabels) ExistingLabels.Add(L);

    for (TActorIterator<AStaticMeshActor> It(World); It; ++It)
    {
        AStaticMeshActor* Cand = *It;
        if (!IsValid(Cand) || SeedSet.Contains(Cand)) continue;

        UStaticMeshComponent* MC = Cand->GetStaticMeshComponent();
        if (!MC || MC->GetNumMaterials() == 0) continue;

        FVector Origin, Extent;
        Cand->GetActorBounds(false, Origin, Extent);
        const FBox CandBox(Origin - Extent, Origin + Extent);
        if (!OverlapsAnySeed(CandBox)) continue;

        const FString Label = Cand->GetActorLabel();
        if (!ExistingLabels.Contains(Label))
        {
            InOutActorLabels.Add(Label);
            ExistingLabels.Add(Label);
        }
    }

    int32 IFAIdx = 0;
    for (TActorIterator<AInstancedFoliageActor> It(World); It; ++It, ++IFAIdx)
    {
        AInstancedFoliageActor* IFA = *It;
        if (!IsValid(IFA)) continue;

        TArray<UInstancedStaticMeshComponent*> Components;
        IFA->GetComponents(Components);

        int32 CompIdx = 0;
        for (UInstancedStaticMeshComponent* ISM : Components)
        {
            if (!ISM) { ++CompIdx; continue; }

            UStaticMesh* Mesh = ISM->GetStaticMesh();
            if (!Mesh)   { ++CompIdx; continue; }

            const FBoxSphereBounds LocalBounds = Mesh->GetBounds();
            const int32 Count = ISM->GetInstanceCount();

            for (int32 i = 0; i < Count; ++i)
            {
                FTransform T;
                if (!ISM->GetInstanceTransform(i, T, /*bWorldSpace=*/true)) continue;

                const FBox InstBox = LocalBounds.TransformBy(T).GetBox();
                if (!OverlapsAnySeed(InstBox)) continue;

                FString Base = FString::Printf(TEXT("Foliage_%s_%d_%d_%d"),
                    *Mesh->GetName(), IFAIdx, CompIdx, i);
                FString Label = Base;
                int32 Suffix = 2;
                while (ExistingLabels.Contains(Label))
                {
                    Label = FString::Printf(TEXT("%s_%d"), *Base, Suffix++);
                }
                ExistingLabels.Add(Label);

                FGTFoliageExportEntry Entry;
                Entry.Mesh = Mesh;
                Entry.WorldTransform = T;
                Entry.Label = Label;
                OutFoliageEntries.Add(MoveTemp(Entry));
            }

            ++CompIdx;
        }
    }
}

bool FVCCSimPanelTexEnhancer::BuildMergedNearbyEntry(
    UWorld* World,
    const TArray<FString>& NearbyStaticLabels,
    const TArray<FGTFoliageExportEntry>& FoliageEntries,
    FGTFoliageExportEntry& OutMergedEntry) const
{
    if (!World) return false;
    if (NearbyStaticLabels.IsEmpty() && FoliageEntries.IsEmpty()) return false;

    TMap<FString, AStaticMeshActor*> LabelMap;
    for (TActorIterator<AStaticMeshActor> It(World); It; ++It)
        if (AStaticMeshActor* A = *It) LabelMap.Add(A->GetActorLabel(), A);

    TArray<UPrimitiveComponent*> Comps;
    TArray<AStaticMeshActor*> IntermediateActors;

    for (const FString& L : NearbyStaticLabels)
    {
        AStaticMeshActor** Found = LabelMap.Find(L);
        if (!Found || !IsValid(*Found)) continue;
        if (UStaticMeshComponent* MC = (*Found)->GetStaticMeshComponent())
        {
            if (MC->GetStaticMesh()) Comps.Add(MC);
        }
    }

    for (const FGTFoliageExportEntry& E : FoliageEntries)
    {
        UStaticMesh* M = E.Mesh.Get();
        if (!M) continue;

        FActorSpawnParameters SP;
        SP.ObjectFlags |= RF_Transient;
        SP.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
        AStaticMeshActor* TA = World->SpawnActor<AStaticMeshActor>(
            AStaticMeshActor::StaticClass(), E.WorldTransform, SP);
        if (!TA) continue;

        TA->SetMobility(EComponentMobility::Movable);
        if (UStaticMeshComponent* MC = TA->GetStaticMeshComponent())
        {
            MC->SetStaticMesh(M);
        }
        TA->SetActorEnableCollision(false);
        TA->SetActorHiddenInGame(true);
        IntermediateActors.Add(TA);
        Comps.Add(TA->GetStaticMeshComponent());
    }

    ON_SCOPE_EXIT
    {
        for (AStaticMeshActor* A : IntermediateActors)
            if (IsValid(A)) World->DestroyActor(A);
    };

    if (Comps.IsEmpty()) return false;

    const IMeshMergeUtilities& MeshUtils = FModuleManager::Get()
        .LoadModuleChecked<IMeshMergeModule>("MeshMergeUtilities").GetUtilities();

    FMeshMergingSettings Settings;
    Settings.bMergeMaterials        = false;
    Settings.bPivotPointAtZero      = false;
    Settings.LODSelectionType       = EMeshLODSelectionType::SpecificLOD;
    Settings.SpecificLOD            = 0;

    const FString GuidStr = FGuid::NewGuid().ToString(EGuidFormats::Short);
    const FString PackageName = FString::Printf(TEXT("/Engine/Transient/_VCCSimNearbyMerged_%s"), *GuidStr);
    UPackage* Package = CreatePackage(*PackageName);
    if (!Package) return false;
    Package->SetFlags(RF_Transient);
    Package->AddToRoot();
    ON_SCOPE_EXIT { Package->RemoveFromRoot(); };

    TArray<UObject*> CreatedAssets;
    FVector MergedLocation = FVector::ZeroVector;

    MeshUtils.MergeComponentsToStaticMesh(
        Comps, World, Settings, /*BaseMaterial=*/nullptr,
        Package, PackageName,
        CreatedAssets, MergedLocation,
        /*ScreenSize=*/TNumericLimits<float>::Max(),
        /*bSilent=*/true);

    UStaticMesh* MergedMesh = nullptr;
    int32 TotalTriangles = 0;
    for (UObject* Obj : CreatedAssets)
    {
        if (UStaticMesh* SM = Cast<UStaticMesh>(Obj))
        {
            MergedMesh = SM;
            if (SM->GetRenderData() && SM->GetRenderData()->LODResources.Num() > 0)
            {
                TotalTriangles = SM->GetRenderData()->LODResources[0].GetNumTriangles();
            }
            break;
        }
    }

    UE_LOG(LogTexEnhancerPanel, Log,
        TEXT("Merge: components=%d, assets=%d, mesh=%s, LOD0 tris=%d"),
        Comps.Num(), CreatedAssets.Num(),
        MergedMesh ? *MergedMesh->GetName() : TEXT("<null>"),
        TotalTriangles);

    if (!MergedMesh || TotalTriangles == 0) return false;

    OutMergedEntry.Mesh = MergedMesh;
    OutMergedEntry.WorldTransform = FTransform(MergedLocation);
    OutMergedEntry.Label = TEXT("Nearby_Merged");
    return true;
}

void FVCCSimPanelTexEnhancer::SetExpansionVisualization(bool bEnabled)
{
    bVisualizeExpansion = bEnabled;

    if (!GEditor) return;

    if (bEnabled)
    {
        if (!ExpansionVizTimerHandle.IsValid())
        {
            GEditor->GetTimerManager()->SetTimer(ExpansionVizTimerHandle,
                FTimerDelegate::CreateLambda([this]() { TickExpansionVisualization(); }),
                0.1f, /*bLoop=*/true);
        }
        TickExpansionVisualization();
    }
    else
    {
        GEditor->GetTimerManager()->ClearTimer(ExpansionVizTimerHandle);
        if (UWorld* World = GEditor->GetEditorWorldContext().World())
        {
            FlushPersistentDebugLines(World);
        }
    }
}

void FVCCSimPanelTexEnhancer::TickExpansionVisualization()
{
    if (!bVisualizeExpansion || !GEditor) return;
    UWorld* World = GEditor->GetEditorWorldContext().World();
    if (!World) return;

    TMap<FString, AStaticMeshActor*> LabelMap;
    for (TActorIterator<AStaticMeshActor> It(World); It; ++It)
        if (AStaticMeshActor* A = *It) LabelMap.Add(A->GetActorLabel(), A);

    TArray<AStaticMeshActor*> Seeds;
    for (const TSharedPtr<FString>& Item : GTActorListItems)
    {
        if (!Item.IsValid()) continue;
        if (AStaticMeshActor** Found = LabelMap.Find(*Item))
            if (IsValid(*Found)) Seeds.Add(*Found);
    }
    if (Seeds.IsEmpty()) return;

    TArray<FSeedShape> Shapes;
    BuildSeedShapes(World, Seeds, FMath::Max(0.f, NearbyRadius), /*NumProbes=*/60, Shapes);

    const FColor LineColor = FColor::Yellow;
    const float  LifeTime  = 0.15f;
    const float  Thickness = 2.f;

    for (const FSeedShape& S : Shapes)
    {
        const int32 N = S.Polygon.Num();
        if (N < 2) continue;
        const float LoZ = static_cast<float>(S.MinZ);
        const float HiZ = static_cast<float>(S.MaxZ);
        for (int32 i = 0; i < N; ++i)
        {
            const FVector2D& P0 = S.Polygon[i];
            const FVector2D& P1 = S.Polygon[(i + 1) % N];
            const FVector A_Lo(P0.X, P0.Y, LoZ);
            const FVector B_Lo(P1.X, P1.Y, LoZ);
            const FVector A_Hi(P0.X, P0.Y, HiZ);
            const FVector B_Hi(P1.X, P1.Y, HiZ);
            DrawDebugLine(World, A_Lo, B_Lo, LineColor, false, LifeTime, 0, Thickness);
            DrawDebugLine(World, A_Hi, B_Hi, LineColor, false, LifeTime, 0, Thickness);
            DrawDebugLine(World, A_Lo, A_Hi, LineColor, false, LifeTime, 0, Thickness);
        }
    }
}

FReply FVCCSimPanelTexEnhancer::OnExportGTMaterialsClicked()
{
    if (bGTExportInProgress)
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("GT material export already in progress."), true);
        return FReply::Handled();
    }
    if (OutputDirectory.IsEmpty())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Output directory is not set."), true);
        return FReply::Handled();
    }
    if (GTActorListItems.IsEmpty())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("GT actor list is empty. Select actors in viewport and click '+ Add Selected'."), true);
        return FReply::Handled();
    }
    UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
    if (!World)
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("No editor world available."), true);
        return FReply::Handled();
    }

    bGTExportInProgress = true;
    FVCCSimUIHelpers::ShowNotification(TEXT("Starting GT material export..."));

    FSimpleDelegate OnComplete = FSimpleDelegate::CreateLambda([this]()
    {
        bGTExportInProgress = false;
    });

    TArray<FString> ActorLabels;
    for (const auto& Item : GTActorListItems)
    {
        if (Item.IsValid()) ActorLabels.Add(*Item);
    }

    TArray<FGTFoliageExportEntry> FoliageEntries;

    if (bIncludeNearbyMeshes && NearbyRadius >= 0.f)
    {
        TMap<FString, AStaticMeshActor*> LabelMap;
        for (TActorIterator<AStaticMeshActor> It(World); It; ++It)
            if (AStaticMeshActor* A = *It) LabelMap.Add(A->GetActorLabel(), A);

        TArray<AStaticMeshActor*> SeedActors;
        for (const FString& Label : ActorLabels)
        {
            if (AStaticMeshActor** Found = LabelMap.Find(Label))
                SeedActors.Add(*Found);
        }

        const int32 BeforeActors = ActorLabels.Num();
        CollectNearbyTargets(World, SeedActors, NearbyRadius, ActorLabels, FoliageEntries);
        const int32 AddedActors = ActorLabels.Num() - BeforeActors;
        UE_LOG(LogTexEnhancerPanel, Log,
            TEXT("Nearby expansion (Expand=%.1fcm): +%d static actors, +%d foliage instances"),
            NearbyRadius, AddedActors, FoliageEntries.Num());

        if (bMergeNearbyMeshes && (AddedActors > 0 || !FoliageEntries.IsEmpty()))
        {
            TArray<FString> NearbyStaticLabels;
            NearbyStaticLabels.Reserve(AddedActors);
            for (int32 i = BeforeActors; i < ActorLabels.Num(); ++i)
                NearbyStaticLabels.Add(ActorLabels[i]);
            ActorLabels.SetNum(BeforeActors);

            TArray<FGTFoliageExportEntry> NearbyFoliage = MoveTemp(FoliageEntries);
            FoliageEntries.Reset();

            FGTFoliageExportEntry Merged;
            if (BuildMergedNearbyEntry(World, NearbyStaticLabels, NearbyFoliage, Merged))
            {
                FoliageEntries.Add(MoveTemp(Merged));
                UE_LOG(LogTexEnhancerPanel, Log, TEXT("Merged %d static + %d foliage into a single mesh"),
                    NearbyStaticLabels.Num(), NearbyFoliage.Num());
            }
            else
            {
                UE_LOG(LogTexEnhancerPanel, Warning, TEXT("Merge nearby failed; falling back to individual export"));
                for (const FString& L : NearbyStaticLabels) ActorLabels.Add(L);
                FoliageEntries = MoveTemp(NearbyFoliage);
            }
        }
    }

    const FString Timestamp = FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S"));
    const FString BaseDir = OutputDirectory / TEXT("gt_materials") / Timestamp;

    GTMaterialExporter->ExportMaterials(
        ActorLabels,
        FoliageEntries,
        World,
        BaseDir,
        SceneName,
        GTTextureResolution,
        OnComplete
    );

    return FReply::Handled();
}

// ============================================================================
// SECTION 6: TEXENHANCER PIPELINE
// ============================================================================

FReply FVCCSimPanelTexEnhancer::OnBrowseScriptClicked()
{
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (!DesktopPlatform) return FReply::Handled();

    TArray<FString> SelectedFiles;
    void* ParentWindowHandle = const_cast<void*>(FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr));

    if (DesktopPlatform->OpenFileDialog(
        ParentWindowHandle,
        TEXT("Select TexEnhancer Python Script"),
        FPaths::GetPath(TexEnhancerScriptPath),
        TEXT(""),
        TEXT("Python Scripts (*.py)|*.py"),
        EFileDialogFlags::None,
        SelectedFiles))
    {
        if (SelectedFiles.Num() > 0)
        {
            TexEnhancerScriptPath = SelectedFiles[0];
            if (TexEnhancerScriptTextBox.IsValid())
            {
                TexEnhancerScriptTextBox->SetText(FText::FromString(TexEnhancerScriptPath));
            }
            SavePaths();
        }
    }

    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnRunTexEnhancerClicked()
{
    if (TexEnhancerScriptPath.IsEmpty() || !FPaths::FileExists(TexEnhancerScriptPath))
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("TexEnhancer script not found. Please browse to the script."), true);
        return FReply::Handled();
    }

    if (OutputDirectory.IsEmpty())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Output directory is not set."), true);
        return FReply::Handled();
    }

    if (bPipelineInProgress)
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Pipeline is already running."), true);
        return FReply::Handled();
    }

    FString CameraInfoDir  = FPaths::Combine(GetSetACaptureDir(), TEXT("config"));
    FString GTMaterialsPath  = GetGTMaterialsPath();
    FString PipelineOutDir   = FPaths::Combine(OutputDirectory, TEXT("estimated"));

    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.DirectoryExists(*PipelineOutDir))
    {
        PlatformFile.CreateDirectoryTree(*PipelineOutDir);
    }

    FString Args = FString::Printf(
        TEXT("\"%s\" --camera_info_dir \"%s\" --image_dir \"%s\" --output_dir \"%s\" --gt_materials \"%s\""),
        *TexEnhancerScriptPath,
        *CameraInfoDir,
        *FPaths::Combine(GetSetACaptureDir(), TEXT("images")),
        *PipelineOutDir,
        *GTMaterialsPath
    );

    uint32 ProcessId = 0;
    PipelineProcHandle = FPlatformProcess::CreateProc(
        TEXT("python"),
        *Args,
        false, false, false,
        &ProcessId, 0, nullptr, nullptr
    );

    if (PipelineProcHandle.IsValid())
    {
        bPipelineInProgress = true;
        UE_LOG(LogTexEnhancerPanel, Log, TEXT("TexEnhancer pipeline started..."));

        GEditor->GetTimerManager()->SetTimer(StatusTimerHandle, [this]()
        {
            PollPipelineProcess();
        }, 2.f, true);
    }
    else
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Failed to launch TexEnhancer. Check that Python is in PATH."), true);
    }

    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnStopTexEnhancerClicked()
{
    if (PipelineProcHandle.IsValid())
    {
        FPlatformProcess::TerminateProc(PipelineProcHandle, false);
        FPlatformProcess::CloseProc(PipelineProcHandle);
    }

    if (GEditor)
    {
        GEditor->GetTimerManager()->ClearTimer(StatusTimerHandle);
    }

    bPipelineInProgress = false;
    UE_LOG(LogTexEnhancerPanel, Log, TEXT("TexEnhancer pipeline stopped by user."));

    return FReply::Handled();
}

void FVCCSimPanelTexEnhancer::PollPipelineProcess()
{
    if (!PipelineProcHandle.IsValid()) return;

    if (!FPlatformProcess::IsProcRunning(PipelineProcHandle))
    {
        int32 ReturnCode = 0;
        FPlatformProcess::GetProcReturnCode(PipelineProcHandle, &ReturnCode);
        FPlatformProcess::CloseProc(PipelineProcHandle);

        if (GEditor)
        {
            GEditor->GetTimerManager()->ClearTimer(StatusTimerHandle);
        }

        bPipelineInProgress = false;

        FString Msg = ReturnCode == 0
            ? TEXT("TexEnhancer pipeline completed successfully.")
            : FString::Printf(TEXT("TexEnhancer pipeline exited with code %d."), ReturnCode);

        FVCCSimUIHelpers::ShowNotification(Msg, ReturnCode != 0);
        UE_LOG(LogTexEnhancerPanel, Log, TEXT("%s"), *Msg);
    }
}

// ============================================================================
// SECTION 7: EVALUATION
// ============================================================================

FReply FVCCSimPanelTexEnhancer::OnBrowseEstimatedDirClicked()
{
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (!DesktopPlatform) return FReply::Handled();

    FString SelectedDir;
    void* ParentWindowHandle = const_cast<void*>(FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr));

    if (DesktopPlatform->OpenDirectoryDialog(
        ParentWindowHandle,
        TEXT("Select Estimated Materials Directory"),
        EstimatedMaterialsDir,
        SelectedDir))
    {
        EstimatedMaterialsDir = SelectedDir;
        if (EstimatedMaterialsDirTextBox.IsValid())
        {
            EstimatedMaterialsDirTextBox->SetText(FText::FromString(EstimatedMaterialsDir));
        }
        SavePaths();
    }

    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnRunEvaluationClicked()
{
    FString GTPath = GetGTMaterialsPath();
    if (!FPaths::FileExists(GTPath))
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("GT materials file not found. Run GT Export first."), true);
        return FReply::Handled();
    }

    if (EstimatedMaterialsDir.IsEmpty())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Estimated materials directory is not set."), true);
        return FReply::Handled();
    }

    RunBRDFEvaluation();
    return FReply::Handled();
}

void FVCCSimPanelTexEnhancer::RunBRDFEvaluation()
{
    FString GTPath = GetGTMaterialsPath();

    FString GTJsonString;
    if (!FFileHelper::LoadFileToString(GTJsonString, *GTPath))
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Failed to load GT materials JSON."), true);
        return;
    }

    TSharedPtr<FJsonObject> GTRoot;
    TSharedRef<TJsonReader<>> GTReader = TJsonReaderFactory<>::Create(GTJsonString);
    if (!FJsonSerializer::Deserialize(GTReader, GTRoot) || !GTRoot.IsValid())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Failed to parse GT materials JSON."), true);
        return;
    }

    FString EstJsonPath = FPaths::Combine(EstimatedMaterialsDir, TEXT("estimated_materials.json"));
    FString EstJsonString;
    if (!FFileHelper::LoadFileToString(EstJsonString, *EstJsonPath))
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("estimated_materials.json not found in estimated dir."), true);
        return;
    }

    TSharedPtr<FJsonObject> EstRoot;
    TSharedRef<TJsonReader<>> EstReader = TJsonReaderFactory<>::Create(EstJsonString);
    if (!FJsonSerializer::Deserialize(EstReader, EstRoot) || !EstRoot.IsValid())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Failed to parse estimated materials JSON."), true);
        return;
    }

    const TArray<TSharedPtr<FJsonValue>>* GTActors = nullptr;
    const TArray<TSharedPtr<FJsonValue>>* EstActors = nullptr;

    if (!GTRoot->TryGetArrayField(TEXT("actors"), GTActors) ||
        !EstRoot->TryGetArrayField(TEXT("actors"), EstActors))
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Invalid JSON format in materials files."), true);
        return;
    }

    TMap<FString, TSharedPtr<FJsonObject>> GTMap;
    for (auto& V : *GTActors)
    {
        if (V->Type == EJson::Object)
        {
            TSharedPtr<FJsonObject> Obj = V->AsObject();
            FString ActorName;
            Obj->TryGetStringField(TEXT("name"), ActorName);
            GTMap.Add(ActorName, Obj);
        }
    }

    float TotalRoughnessDiff = 0.f;
    float TotalMetallicDiff  = 0.f;
    int32 TotalSlots         = 0;

    for (auto& V : *EstActors)
    {
        if (V->Type != EJson::Object) continue;
        TSharedPtr<FJsonObject> EstActor = V->AsObject();

        FString ActorName;
        EstActor->TryGetStringField(TEXT("name"), ActorName);

        TSharedPtr<FJsonObject>* GTActorPtr = GTMap.Find(ActorName);
        if (!GTActorPtr) continue;

        const TArray<TSharedPtr<FJsonValue>>* EstMats = nullptr;
        const TArray<TSharedPtr<FJsonValue>>* GTMats  = nullptr;

        if (!EstActor->TryGetArrayField(TEXT("materials"), EstMats)) continue;
        if (!(*GTActorPtr)->TryGetArrayField(TEXT("materials"), GTMats)) continue;

        for (int32 i = 0; i < FMath::Min(EstMats->Num(), GTMats->Num()); ++i)
        {
            TSharedPtr<FJsonObject> EstMat = (*EstMats)[i]->AsObject();
            TSharedPtr<FJsonObject> GTMat  = (*GTMats)[i]->AsObject();

            double EstR = 0.0, GTR = 0.0, EstM = 0.0, GTM = 0.0;
            EstMat->TryGetNumberField(TEXT("roughness"), EstR);
            GTMat->TryGetNumberField(TEXT("roughness"), GTR);
            EstMat->TryGetNumberField(TEXT("metallic"),  EstM);
            GTMat->TryGetNumberField(TEXT("metallic"),  GTM);

            TotalRoughnessDiff += FMath::Abs((float)(EstR - GTR));
            TotalMetallicDiff  += FMath::Abs((float)(EstM - GTM));
            ++TotalSlots;
        }
    }

    FString Results;
    if (TotalSlots > 0)
    {
        float MAERoughness = TotalRoughnessDiff / TotalSlots;
        float MAEMetallic  = TotalMetallicDiff  / TotalSlots;
        Results = FString::Printf(
            TEXT("BRDF Evaluation Results\n"
                 "──────────────────────\n"
                 "Compared slots:       %d\n"
                 "Roughness MAE:        %.4f\n"
                 "Metallic  MAE:        %.4f\n"),
            TotalSlots, MAERoughness, MAEMetallic);
    }
    else
    {
        Results = TEXT("No matching actors found between GT and estimated materials.");
    }

    if (EvalResultsTextBlock.IsValid())
    {
        EvalResultsTextBlock->SetText(FText::FromString(Results));
    }
    UE_LOG(LogTexEnhancerPanel, Log, TEXT("BRDF evaluation complete."));

    FString EvalDir = GetEvaluationOutputDir();
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.DirectoryExists(*EvalDir))
    {
        PlatformFile.CreateDirectoryTree(*EvalDir);
    }

    FString CSVPath = FPaths::Combine(EvalDir, TEXT("brdf_accuracy.csv"));
    FString CSVContent = TEXT("metric,value\n");
    if (TotalSlots > 0)
    {
        CSVContent += FString::Printf(TEXT("roughness_mae,%.6f\n"), TotalRoughnessDiff / TotalSlots);
        CSVContent += FString::Printf(TEXT("metallic_mae,%.6f\n"),  TotalMetallicDiff  / TotalSlots);
        CSVContent += FString::Printf(TEXT("compared_slots,%d\n"),  TotalSlots);
    }
    FFileHelper::SaveStringToFile(CSVContent, *CSVPath);

    UE_LOG(LogTexEnhancerPanel, Log, TEXT("Evaluation complete. CSV saved: %s"), *CSVPath);
    FVCCSimUIHelpers::ShowNotification(TEXT("BRDF evaluation complete. Results saved to evaluation directory."), false);
}

// ============================================================================
// UTILITIES
// ============================================================================

FString FVCCSimPanelTexEnhancer::GetSetACaptureDir() const
{
    return FPaths::Combine(OutputDirectory, SceneName, TEXT("capture_setA"));
}

FString FVCCSimPanelTexEnhancer::GetSetBCaptureDir() const
{
    return FPaths::Combine(OutputDirectory, SceneName, TEXT("capture_setB"));
}

FString FVCCSimPanelTexEnhancer::GetGTMaterialsPath() const
{
    return FPaths::Combine(OutputDirectory, TEXT("gt_materials"), TEXT("manifest.json"));
}

FString FVCCSimPanelTexEnhancer::GetEvaluationOutputDir() const
{
    return FPaths::Combine(OutputDirectory, SceneName, TEXT("evaluation"));
}

// ============================================================================
// SECTION 5: NANOBANANA PROJECTION
// ============================================================================

FReply FVCCSimPanelTexEnhancer::OnBrowseNanobananaResultDirClicked()
{
    IDesktopPlatform* DP = FDesktopPlatformModule::Get();
    if (!DP) return FReply::Handled();
    FString Sel;
    void* Handle = const_cast<void*>(FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr));
    if (DP->OpenDirectoryDialog(Handle, TEXT("Select Nanobanana Result Directory"), NanobananaResultDir, Sel))
    {
        NanobananaResultDir = Sel;
        if (NanobananaResultDirTextBox.IsValid())
            NanobananaResultDirTextBox->SetText(FText::FromString(Sel));
        SavePaths();
    }
    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnBrowseNanobananaPosesFileClicked()
{
    IDesktopPlatform* DP = FDesktopPlatformModule::Get();
    if (!DP) return FReply::Handled();
    TArray<FString> Files;
    void* Handle = const_cast<void*>(FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr));
    if (DP->OpenFileDialog(Handle, TEXT("Select Poses File"), FPaths::GetPath(NanobananaPosesFile),
        TEXT(""), TEXT("Text Files (*.txt)|*.txt|All Files (*.*)|*.*"), EFileDialogFlags::None, Files) && Files.Num() > 0)
    {
        NanobananaPosesFile = Files[0];
        if (NanobananaPosesFileTextBox.IsValid())
            NanobananaPosesFileTextBox->SetText(FText::FromString(Files[0]));
        SavePaths();
    }
    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnBrowseNanobananaManifestFileClicked()
{
    IDesktopPlatform* DP = FDesktopPlatformModule::Get();
    if (!DP) return FReply::Handled();
    TArray<FString> Files;
    void* Handle = const_cast<void*>(FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr));
    if (DP->OpenFileDialog(Handle, TEXT("Select manifest.json"), FPaths::GetPath(NanobananaManifestFile),
        TEXT(""), TEXT("JSON Files (*.json)|*.json|All Files (*.*)|*.*"), EFileDialogFlags::None, Files) && Files.Num() > 0)
    {
        NanobananaManifestFile = Files[0];
        if (NanobananaManifestFileTextBox.IsValid())
            NanobananaManifestFileTextBox->SetText(FText::FromString(Files[0]));
        SavePaths();
    }
    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnRunNanobananaProjectionClicked()
{
    if (bNanobananaInProgress)
    {
        UE_LOG(LogTexEnhancerPanel, Warning, TEXT("Nanobanana projection already in progress."));
        return FReply::Handled();
    }
    if (NanobananaResultDir.IsEmpty() || !FPaths::DirectoryExists(NanobananaResultDir))
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Nanobanana result directory not set or missing."), true);
        return FReply::Handled();
    }
    if (NanobananaPosesFile.IsEmpty() || !FPaths::FileExists(NanobananaPosesFile))
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Poses file not set or missing."), true);
        return FReply::Handled();
    }
    if (NanobananaManifestFile.IsEmpty() || !FPaths::FileExists(NanobananaManifestFile))
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("manifest.json not set or missing."), true);
        return FReply::Handled();
    }

    NanobananaManager = MakeShared<FNanobananaManager>();

    FNanobananaManager::FProjectionParams Params;
    Params.ResultDir         = NanobananaResultDir;
    Params.PosesFile         = NanobananaPosesFile;
    Params.ManifestFile      = NanobananaManifestFile;
    Params.HFOV              = NanobananaHFOV;
    Params.ImageWidth        = NanobananaImageWidth;
    Params.ImageHeight       = NanobananaImageHeight;
    Params.OverlayAlpha      = NanobananaOverlayAlpha;
    Params.World             = GEditor->GetEditorWorldContext().World();
    Params.SceneName         = SceneName;
    Params.TextureResolution = GTTextureResolution;

    FOnNanobananaProgress OnProgressDelegate = FOnNanobananaProgress::CreateLambda(
        [this](const FString& Status, int32 Processed, int32 Total)
    {
        if (Total > 0)
        {
            UE_LOG(LogTexEnhancerPanel, Log, TEXT("%s %d / %d"), *Status, Processed, Total);
        }
        else
        {
            UE_LOG(LogTexEnhancerPanel, Log, TEXT("%s"), *Status);
        }
    });

    FOnNanobananaComplete OnCompleteDelegate = FOnNanobananaComplete::CreateLambda(
        [this](const FString& FinalStatus)
    {
        bNanobananaInProgress = false;
        UE_LOG(LogTexEnhancerPanel, Log, TEXT("%s"), *FinalStatus);
        FVCCSimUIHelpers::ShowNotification(TEXT("Nanobanana projection complete!"), false);
    });

    bNanobananaInProgress = true;
    NanobananaManager->RunProjection(Params, OnProgressDelegate, OnCompleteDelegate);

    return FReply::Handled();
}

// ============================================================================
// PATH PERSISTENCE
// ============================================================================

void FVCCSimPanelTexEnhancer::SavePaths()
{
    FVCCSimConfigManager::FTexEnhancerConfig Config;
    Config.OutputDirectory       = OutputDirectory;
    Config.SceneName             = SceneName;
    Config.TexEnhancerScriptPath = TexEnhancerScriptPath;
    Config.EstimatedMaterialsDir = EstimatedMaterialsDir;
    for (const TSharedPtr<FString>& Label : GTActorListItems)
    {
        if (Label.IsValid())
            Config.GTActorLabels.Add(*Label);
    }
    Config.NanobananaResultDir    = NanobananaResultDir;
    Config.NanobananaPosesFile    = NanobananaPosesFile;
    Config.NanobananaManifestFile = NanobananaManifestFile;
    Config.NanobananaHFOV         = NanobananaHFOV;
    Config.NanobananaImageWidth   = NanobananaImageWidth;
    Config.NanobananaImageHeight  = NanobananaImageHeight;
    Config.NanobananaOverlayAlpha = NanobananaOverlayAlpha;
    Config.bIncludeNearbyMeshes   = bIncludeNearbyMeshes;
    Config.bMergeNearbyMeshes     = bMergeNearbyMeshes;
    Config.NearbyRadius           = NearbyRadius;
    FVCCSimConfigManager::Get().SetTexEnhancerConfig(Config);
}

void FVCCSimPanelTexEnhancer::LoadPaths()
{
    const auto& Config = FVCCSimConfigManager::Get().GetTexEnhancerConfig();
    if (!Config.OutputDirectory.IsEmpty())       OutputDirectory       = Config.OutputDirectory;
    if (OutputDirectory.IsEmpty())               OutputDirectory       = GetVCCSimOutputRoot() / TEXT("TexEnhancer");
    if (!Config.SceneName.IsEmpty())             SceneName             = Config.SceneName;
    if (!Config.TexEnhancerScriptPath.IsEmpty()) TexEnhancerScriptPath = Config.TexEnhancerScriptPath;
    if (!Config.EstimatedMaterialsDir.IsEmpty()) EstimatedMaterialsDir = Config.EstimatedMaterialsDir;

    GTActorListItems.Empty();
    for (const FString& Label : Config.GTActorLabels)
        GTActorListItems.Add(MakeShareable(new FString(Label)));
    if (GTActorListView.IsValid())
        GTActorListView->RequestListRefresh();

    if (!Config.NanobananaResultDir.IsEmpty())    NanobananaResultDir    = Config.NanobananaResultDir;
    if (!Config.NanobananaPosesFile.IsEmpty())    NanobananaPosesFile    = Config.NanobananaPosesFile;
    if (!Config.NanobananaManifestFile.IsEmpty()) NanobananaManifestFile = Config.NanobananaManifestFile;
    NanobananaHFOV         = Config.NanobananaHFOV;
    NanobananaImageWidth   = Config.NanobananaImageWidth;
    NanobananaImageHeight  = Config.NanobananaImageHeight;
    NanobananaOverlayAlpha = Config.NanobananaOverlayAlpha;

    NanobananaHFOVValue         = NanobananaHFOV;
    NanobananaImageWidthValue   = NanobananaImageWidth;
    NanobananaImageHeightValue  = NanobananaImageHeight;
    NanobananaOverlayAlphaValue = NanobananaOverlayAlpha;

    bIncludeNearbyMeshes = Config.bIncludeNearbyMeshes;
    bMergeNearbyMeshes   = Config.bMergeNearbyMeshes;
    NearbyRadius         = Config.NearbyRadius;
    NearbyRadiusValue    = NearbyRadius;
}