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
#include "Utils/ColmapManager.h"
#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Pawns/FlashPawn.h"
#include "Engine/DirectionalLight.h"
#include "Components/DirectionalLightComponent.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/StaticMesh.h"
#include "Engine/Texture.h"
#include "Engine/Texture2D.h"
#include "Materials/MaterialInterface.h"
#include "EngineUtils.h"
#include "Selection.h"
#include "MeshDescription.h"
#include "StaticMeshAttributes.h"

#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
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

DEFINE_LOG_CATEGORY_STATIC(LogTexEnhancerPanel, Log, All);

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

FVCCSimPanelTexEnhancer::FVCCSimPanelTexEnhancer()
{
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
    DayCycleSpeedValue   = DayCycleSpeed;
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
    LoadPaths();
    UE_LOG(LogTexEnhancerPanel, Log, TEXT("TexEnhancer panel initialized"));
}

void FVCCSimPanelTexEnhancer::Cleanup()
{
    if (GEditor)
    {
        GEditor->GetTimerManager()->ClearTimer(StatusTimerHandle);
        GEditor->GetTimerManager()->ClearTimer(DayCycleTimerHandle);
    }
    bDayCycleActive = false;

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

void FVCCSimPanelTexEnhancer::ApplyLightingCondition(float ElevationDeg, float AzimuthDeg, bool bMarkDirty)
{
    if (!GEditor || !GEditor->GetEditorWorldContext().World())
    {
        UpdateStatus(TEXT("Error: No editor world available"));
        return;
    }

    UWorld* World = GEditor->GetEditorWorldContext().World();

    ADirectionalLight* DirectionalLight = nullptr;
    for (TActorIterator<ADirectionalLight> It(World); It; ++It)
    {
        ADirectionalLight* Candidate = *It;
        if (!Candidate) continue;

        UDirectionalLightComponent* LightComp = Candidate->GetComponent();
        if (LightComp && LightComp->bAtmosphereSunLight)
        {
            DirectionalLight = Candidate;
            break;
        }

        if (!DirectionalLight)
        {
            DirectionalLight = Candidate;
        }
    }

    if (!DirectionalLight)
    {
        UpdateStatus(TEXT("Warning: No Directional Light found in scene"));
        return;
    }

    if (bMarkDirty) DirectionalLight->Modify();
    FRotator NewRotation(-ElevationDeg, AzimuthDeg - 180.f, 0.f);
    DirectionalLight->SetActorRotation(NewRotation);
    GEditor->RedrawAllViewports();

    FString Msg = FString::Printf(TEXT("Elev=%.1f\u00B0  Az=%.1f\u00B0  applied"), ElevationDeg, AzimuthDeg);
    if (LightingStatusTextBlock.IsValid())
        LightingStatusTextBlock->SetText(FText::FromString(Msg));
    UE_LOG(LogTexEnhancerPanel, Log, TEXT("Lighting applied: Elevation=%.1f Az=%.1f"), ElevationDeg, AzimuthDeg);
}

FReply FVCCSimPanelTexEnhancer::OnApplySetALightingClicked(int32 Index)
{
    if (Index < 0 || Index >= NumLightingSetA) return FReply::Handled();
    ApplyLightingCondition(SetAElevation[Index], SetAAzimuth[Index]);
    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnApplySetBLightingClicked(int32 Index)
{
    if (Index < 0 || Index >= NumLightingSetB) return FReply::Handled();
    ApplyLightingCondition(SetBElevation[Index], SetBAzimuth[Index]);
    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnCalculateSunPositionClicked()
{
    FVCCSimSunPositionHelper::FSunParams Params;
    Params.Latitude  = SunCalcLatitude;
    Params.Longitude = SunCalcLongitude;
    Params.TimeZone  = SunCalcTimeZone;
    Params.Year      = SunCalcYear;
    Params.Month     = SunCalcMonth;
    Params.Day       = SunCalcDay;
    Params.Hour      = SunCalcHour;
    Params.Minute    = SunCalcMinute;

    bool bAboveHorizon = FVCCSimSunPositionHelper::Calculate(Params, SunCalcElevation, SunCalcAzimuth);

    ApplyLightingCondition(SunCalcElevation, SunCalcAzimuth);

    if (!bAboveHorizon)
    {
        UpdateStatus(FString::Printf(TEXT("Night: Sun %.1f below horizon"), -SunCalcElevation));
    }

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
    if (!GEditor) return FReply::Handled();

    bDayCycleActive = !bDayCycleActive;

    if (bDayCycleActive)
    {
        DayCycleSimMinute = 0.f;
        FTimerDelegate Del = FTimerDelegate::CreateRaw(this, &FVCCSimPanelTexEnhancer::TickDayCycle);
        GEditor->GetTimerManager()->SetTimer(DayCycleTimerHandle, Del, 0.1f, true);
    }
    else
    {
        GEditor->GetTimerManager()->ClearTimer(DayCycleTimerHandle);
    }

    return FReply::Handled();
}

void FVCCSimPanelTexEnhancer::TickDayCycle()
{
    const float MinutesPerTick = 1440.f * 0.1f / FMath::Max(DayCycleSpeed, 1.f);
    DayCycleSimMinute += MinutesPerTick;
    if (DayCycleSimMinute >= 1440.f) DayCycleSimMinute -= 1440.f;

    const int32 SimH = FMath::FloorToInt(DayCycleSimMinute / 60.f) % 24;
    const int32 SimM = FMath::FloorToInt(DayCycleSimMinute) % 60;

    FVCCSimSunPositionHelper::FSunParams Params;
    Params.Latitude  = SunCalcLatitude;
    Params.Longitude = SunCalcLongitude;
    Params.TimeZone  = SunCalcTimeZone;
    Params.Year      = SunCalcYear;
    Params.Month     = SunCalcMonth;
    Params.Day       = SunCalcDay;
    Params.Hour      = SimH;
    Params.Minute    = SimM;

    float Elev = 0.f, Az = 0.f;
    FVCCSimSunPositionHelper::Calculate(Params, Elev, Az);

    ApplyLightingCondition(Elev, Az, false);
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

FReply FVCCSimPanelTexEnhancer::OnExportGTMaterialsClicked()
{
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

    ExportGTMaterialsFromScene();
    return FReply::Handled();
}

void FVCCSimPanelTexEnhancer::ExportGTMaterialsFromScene()
{
    if (!GEditor || !GEditor->GetEditorWorldContext().World())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("No editor world available."), true);
        return;
    }

    const FString Timestamp = FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S"));
    const FString BaseDir = OutputDirectory / TEXT("gt_materials") / Timestamp;
    FPlatformFileManager::Get().GetPlatformFile().CreateDirectoryTree(*BaseDir);

    ExportMergedGTMaterials(BaseDir);
}

void FVCCSimPanelTexEnhancer::ExportMergedGTMaterials(const FString& BaseDir)
{
    UWorld* World = GEditor->GetEditorWorldContext().World();

    TMap<FString, AStaticMeshActor*> LabelMap;
    for (TActorIterator<AStaticMeshActor> It(World); It; ++It)
    {
        if (AStaticMeshActor* A = *It)
            LabelMap.Add(A->GetActorLabel(), A);
    }

    TArray<AStaticMeshActor*> Actors;
    TArray<int32>             SlotCounts;
    TArray<FString>           Labels;

    for (const TSharedPtr<FString>& LabelPtr : GTActorListItems)
    {
        if (!LabelPtr.IsValid()) continue;
        AStaticMeshActor** Found = LabelMap.Find(*LabelPtr);
        if (!Found)
        {
            UE_LOG(LogTexEnhancerPanel, Warning, TEXT("GT Export: actor '%s' not found in world"), **LabelPtr);
            continue;
        }
        AStaticMeshActor* Actor = *Found;
        UStaticMeshComponent* MC = Actor->GetStaticMeshComponent();
        const int32 NSlots = MC ? MC->GetNumMaterials() : 0;
        if (NSlots == 0) continue;

        Actors.Add(Actor);
        SlotCounts.Add(NSlots);
        Labels.Add(*LabelPtr);
    }

    if (Actors.IsEmpty())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("No valid actors to export."), true);
        return;
    }

    int32 TotalTiles = 0;
    TArray<int32> ActorTileOffsets;
    for (int32 i = 0; i < Actors.Num(); ++i)
    {
        ActorTileOffsets.Add(TotalTiles);
        TotalTiles += SlotCounts[i];
    }

    const int32 AtlasCols = FMath::Max(1, FMath::CeilToInt(FMath::Sqrt((float)TotalTiles)));
    const int32 AtlasRows = FMath::Max(1, FMath::CeilToInt((float)TotalTiles / AtlasCols));

    WriteMergedOBJ(Actors, SlotCounts, ActorTileOffsets, AtlasCols, AtlasRows, BaseDir / TEXT("merged_mesh.obj"));

    TArray<TArray<FColor>> RoughnessTiles, MetallicTiles;
    RoughnessTiles.SetNum(TotalTiles);
    MetallicTiles.SetNum(TotalTiles);

    TSharedPtr<FJsonObject> RootJson = MakeShareable(new FJsonObject);
    TArray<TSharedPtr<FJsonValue>> ActorArray;

    for (int32 ai = 0; ai < Actors.Num(); ++ai)
    {
        UStaticMeshComponent* MC = Actors[ai]->GetStaticMeshComponent();
        TSharedPtr<FJsonObject> ActorJson = MakeShareable(new FJsonObject);
        ActorJson->SetStringField(TEXT("label"),     Labels[ai]);
        ActorJson->SetStringField(TEXT("mesh_file"), TEXT("merged_mesh.obj"));

        TArray<TSharedPtr<FJsonValue>> SlotArray;
        for (int32 si = 0; si < SlotCounts[ai]; ++si)
        {
            const int32 TileIdx = ActorTileOffsets[ai] + si;
            UMaterialInterface* Mat = MC->GetMaterial(si);

            RoughnessTiles[TileIdx] = ReadMaterialChannelPixels(Mat, true,  GTTextureResolution);
            MetallicTiles[TileIdx]  = ReadMaterialChannelPixels(Mat, false, GTTextureResolution);

            TSharedPtr<FJsonObject> SlotJson = MakeShareable(new FJsonObject);
            SlotJson->SetNumberField(TEXT("slot"),          si);
            SlotJson->SetStringField(TEXT("material_name"), Mat ? Mat->GetName() : TEXT(""));
            SlotJson->SetNumberField(TEXT("atlas_tile"),    TileIdx);
            SlotArray.Add(MakeShareable(new FJsonValueObject(SlotJson)));
        }
        ActorJson->SetArrayField(TEXT("slots"), SlotArray);
        ActorArray.Add(MakeShareable(new FJsonValueObject(ActorJson)));
    }

    WriteAtlasPNG(RoughnessTiles, GTTextureResolution, AtlasCols, AtlasRows, BaseDir / TEXT("roughness_atlas.png"));
    WriteAtlasPNG(MetallicTiles,  GTTextureResolution, AtlasCols, AtlasRows, BaseDir / TEXT("metallic_atlas.png"));

    TSharedPtr<FJsonObject> MetaJson = MakeShareable(new FJsonObject);
    MetaJson->SetStringField(TEXT("scene_name"),        SceneName);
    MetaJson->SetStringField(TEXT("exported_at"),       FDateTime::Now().ToString());
    MetaJson->SetNumberField(TEXT("actor_count"),       Actors.Num());
    MetaJson->SetNumberField(TEXT("texture_resolution"), GTTextureResolution);
    MetaJson->SetNumberField(TEXT("atlas_cols"),        AtlasCols);
    MetaJson->SetNumberField(TEXT("atlas_rows"),        AtlasRows);
    MetaJson->SetStringField(TEXT("roughness_atlas"),   TEXT("roughness_atlas.png"));
    MetaJson->SetStringField(TEXT("metallic_atlas"),    TEXT("metallic_atlas.png"));
    MetaJson->SetStringField(TEXT("mesh_file"),         TEXT("merged_mesh.obj"));
    RootJson->SetObjectField(TEXT("metadata"), MetaJson);
    RootJson->SetArrayField(TEXT("actors"),    ActorArray);

    FString JsonStr;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&JsonStr);
    FJsonSerializer::Serialize(RootJson.ToSharedRef(), Writer);

    if (!FFileHelper::SaveStringToFile(JsonStr, *(BaseDir / TEXT("manifest.json"))))
        UE_LOG(LogTexEnhancerPanel, Warning, TEXT("GT Export: failed to write manifest.json -> %s"), *BaseDir);

    const FString Msg = FString::Printf(
        TEXT("GT merged export done: %d actors, %d tiles (%dx%d atlas) -> %s"),
        Actors.Num(), TotalTiles, AtlasCols, AtlasRows, *BaseDir);
    UpdateStatus(Msg);
    FVCCSimUIHelpers::ShowNotification(Msg, false);
    UE_LOG(LogTexEnhancerPanel, Log, TEXT("%s"), *Msg);
}

bool FVCCSimPanelTexEnhancer::WriteMergedOBJ(
    const TArray<AStaticMeshActor*>& Actors,
    const TArray<int32>& SlotCounts,
    const TArray<int32>& ActorTileOffsets,
    int32 AtlasCols, int32 AtlasRows,
    const FString& ObjPath)
{
    FString VertBuf, UVBuf, FaceBuf;
    int32 GlobalVertexOffset = 0;
    int32 GlobalUVIndex      = 1;

    for (int32 ai = 0; ai < Actors.Num(); ++ai)
    {
        UStaticMesh* SM = Actors[ai]->GetStaticMeshComponent()->GetStaticMesh();
        if (!SM) continue;
        SM->ConditionalPostLoad();

        const FMeshDescription* MD = SM->GetMeshDescription(0);
        if (!MD) continue;

        FStaticMeshConstAttributes Attrs(*MD);
        TVertexAttributesConstRef<FVector3f>         Positions = Attrs.GetVertexPositions();
        TVertexInstanceAttributesConstRef<FVector2f> UVs       = Attrs.GetVertexInstanceUVs();

        const FTransform& Transform = Actors[ai]->GetActorTransform();

        for (const FVertexID VID : MD->Vertices().GetElementIDs())
        {
            const FVector3f& LP = Positions[VID];
            const FVector WP = Transform.TransformPosition(FVector(LP.X, LP.Y, LP.Z));
            VertBuf += FString::Printf(TEXT("v %f %f %f\n"), WP.X, WP.Y, WP.Z);
        }

        TMap<FPolygonGroupID, int32> GroupToSlot;
        {
            int32 SlotIdx = 0;
            for (const FPolygonGroupID PGID : MD->PolygonGroups().GetElementIDs())
                GroupToSlot.Add(PGID, SlotIdx++);
        }

        TMap<FVertexInstanceID, int32> InstanceToTile;
        for (const FTriangleID TID : MD->Triangles().GetElementIDs())
        {
            const FPolygonGroupID GID = MD->GetTrianglePolygonGroup(TID);
            const int32 Tile = ActorTileOffsets[ai] + GroupToSlot.FindRef(GID);
            for (const FVertexInstanceID IID : MD->GetTriangleVertexInstances(TID))
            {
                if (!InstanceToTile.Contains(IID))
                    InstanceToTile.Add(IID, Tile);
            }
        }

        TMap<FVertexInstanceID, int32> InstanceToUVIdx;
        for (const FVertexInstanceID IID : MD->VertexInstances().GetElementIDs())
        {
            const int32 Tile = InstanceToTile.FindRef(IID);
            const int32 Col  = Tile % AtlasCols;
            const int32 Row  = Tile / AtlasCols;
            const FVector2f& UV = UVs.Get(IID, 0);
            const float U = UV.X / AtlasCols + (float)Col / AtlasCols;
            const float V = (1.f - UV.Y) / AtlasRows + (float)Row / AtlasRows;
            UVBuf += FString::Printf(TEXT("vt %f %f\n"), U, V);
            InstanceToUVIdx.Add(IID, GlobalUVIndex++);
        }

        for (const FTriangleID TID : MD->Triangles().GetElementIDs())
        {
            FaceBuf += TEXT("f");
            for (const FVertexInstanceID IID : MD->GetTriangleVertexInstances(TID))
            {
                const int32 VI = MD->GetVertexInstanceVertex(IID).GetValue() + 1 + GlobalVertexOffset;
                const int32 UI = InstanceToUVIdx[IID];
                FaceBuf += FString::Printf(TEXT(" %d/%d"), VI, UI);
            }
            FaceBuf += TEXT("\n");
        }

        GlobalVertexOffset += MD->Vertices().Num();
    }

    return FFileHelper::SaveStringToFile(VertBuf + UVBuf + FaceBuf, *ObjPath,
        FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), FILEWRITE_EvenIfReadOnly);
}

TArray<FColor> FVCCSimPanelTexEnhancer::SampleTexture(UTexture2D* Tex, int32 Channel, int32 TargetSize)
{
    TArray<FColor> Empty;
    if (!Tex) return Empty;

    FTextureSource& Source = Tex->Source;
    if (!Source.IsValid() || Source.GetFormat() != TSF_BGRA8) return Empty;

    const int32 SrcW = Source.GetSizeX();
    const int32 SrcH = Source.GetSizeY();

    TArray64<uint8> Raw;
    if (!Source.GetMipData(Raw, 0)) return Empty;

    TArray<FColor> SrcPixels;
    SrcPixels.SetNumUninitialized(SrcW * SrcH);
    for (int32 i = 0; i < SrcW * SrcH; ++i)
    {
        const uint8 B = Raw[i*4+0], G = Raw[i*4+1], R = Raw[i*4+2], A = Raw[i*4+3];
        if      (Channel == 0) SrcPixels[i] = FColor(R, R, R, 255);
        else if (Channel == 1) SrcPixels[i] = FColor(G, G, G, 255);
        else if (Channel == 2) SrcPixels[i] = FColor(B, B, B, 255);
        else                   SrcPixels[i] = FColor(R, G, B, A);
    }

    if (SrcW == TargetSize && SrcH == TargetSize) return SrcPixels;

    TArray<FColor> DstPixels;
    DstPixels.SetNumUninitialized(TargetSize * TargetSize);
    for (int32 Dy = 0; Dy < TargetSize; ++Dy)
    {
        for (int32 Dx = 0; Dx < TargetSize; ++Dx)
        {
            const int32 Sx = FMath::Clamp(Dx * SrcW / TargetSize, 0, SrcW - 1);
            const int32 Sy = FMath::Clamp(Dy * SrcH / TargetSize, 0, SrcH - 1);
            DstPixels[Dy * TargetSize + Dx] = SrcPixels[Sy * SrcW + Sx];
        }
    }
    return DstPixels;
}

TArray<FColor> FVCCSimPanelTexEnhancer::ReadMaterialChannelPixels(
    UMaterialInterface* Mat, bool bRoughness, int32 TargetSize)
{
    const float DefaultScalar = bRoughness ? 1.f : 0.f;

    auto MakeSolid = [&](float Value) -> TArray<FColor>
    {
        const uint8 V = (uint8)FMath::Clamp(FMath::RoundToInt(Value * 255.f), 0, 255);
        TArray<FColor> Pixels;
        Pixels.Init(FColor(V, V, V, 255), TargetSize * TargetSize);
        return Pixels;
    };

    if (!Mat) return MakeSolid(DefaultScalar);

    auto TryGetTex2D = [&](const FString& ParamName) -> UTexture2D*
    {
        UTexture* T = nullptr;
        Mat->GetTextureParameterValue(FHashedMaterialParameterInfo(FName(*ParamName)), T);
        return T ? Cast<UTexture2D>(T) : nullptr;
    };

    if (UTexture2D* ORM = TryGetTex2D(TEXT("ORM")))
    {
        TArray<FColor> Pixels = SampleTexture(ORM, bRoughness ? 1 : 2, TargetSize);
        if (Pixels.Num() > 0) return Pixels;
    }

    static const TArray<FString> RoughnessNames = { TEXT("Roughness"), TEXT("RoughnessMap"), TEXT("T_Roughness"), TEXT("MetallicRoughnessTexture") };
    static const TArray<FString> MetallicNames  = { TEXT("Metallic"),  TEXT("MetallicMap"),  TEXT("T_Metallic"),  TEXT("MetallicRoughnessTexture") };
    for (const FString& Name : (bRoughness ? RoughnessNames : MetallicNames))
    {
        if (UTexture2D* T = TryGetTex2D(Name))
        {
            TArray<FColor> Pixels = SampleTexture(T, -1, TargetSize);
            if (Pixels.Num() > 0) return Pixels;
        }
    }

    float ScalarVal = DefaultScalar;
    Mat->GetScalarParameterValue(
        FHashedMaterialParameterInfo(FName(bRoughness ? TEXT("Roughness") : TEXT("Metallic"))), ScalarVal);
    return MakeSolid(ScalarVal);
}

bool FVCCSimPanelTexEnhancer::WriteAtlasPNG(
    const TArray<TArray<FColor>>& Tiles,
    int32 TileSize, int32 Cols, int32 Rows,
    const FString& PngPath)
{
    const int32 W = TileSize * Cols;
    const int32 H = TileSize * Rows;
    TArray<FColor> Atlas;
    Atlas.SetNumZeroed(W * H);

    for (int32 TileIdx = 0; TileIdx < Tiles.Num(); ++TileIdx)
    {
        const int32 Col = TileIdx % Cols;
        const int32 Row = TileIdx / Cols;
        const TArray<FColor>& Tile = Tiles[TileIdx];
        for (int32 Py = 0; Py < TileSize; ++Py)
        {
            for (int32 Px = 0; Px < TileSize; ++Px)
            {
                const int32 SrcIdx = Py * TileSize + Px;
                const int32 DstIdx = (Row * TileSize + Py) * W + (Col * TileSize + Px);
                if (SrcIdx < Tile.Num())
                    Atlas[DstIdx] = Tile[SrcIdx];
            }
        }
    }

    IImageWrapperModule& IWM = FModuleManager::LoadModuleChecked<IImageWrapperModule>("ImageWrapper");
    TSharedPtr<IImageWrapper> Wrapper = IWM.CreateImageWrapper(EImageFormat::PNG);
    if (!Wrapper.IsValid()) return false;

    Wrapper->SetRaw(Atlas.GetData(), Atlas.Num() * sizeof(FColor), W, H, ERGBFormat::BGRA, 8);
    TArray64<uint8> PngData = Wrapper->GetCompressed();
    if (PngData.IsEmpty()) return false;

    return FFileHelper::SaveArrayToFile(PngData, *PngPath);
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
        UpdateStatus(TEXT("TexEnhancer pipeline started..."));

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
    UpdateStatus(TEXT("TexEnhancer pipeline stopped by user."));

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

        UpdateStatus(Msg);
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
    UpdateStatus(TEXT("BRDF evaluation complete."));

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

void FVCCSimPanelTexEnhancer::UpdateStatus(const FString& Message)
{
    StatusMessage = Message;
    if (StatusTextBlock.IsValid())
    {
        StatusTextBlock->SetText(FText::FromString(Message));
    }
}

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
    FVCCSimConfigManager::Get().SetTexEnhancerConfig(Config);
}

void FVCCSimPanelTexEnhancer::LoadPaths()
{
    const auto& Config = FVCCSimConfigManager::Get().GetTexEnhancerConfig();
    if (!Config.OutputDirectory.IsEmpty())       OutputDirectory       = Config.OutputDirectory;
    if (!Config.SceneName.IsEmpty())             SceneName             = Config.SceneName;
    if (!Config.TexEnhancerScriptPath.IsEmpty()) TexEnhancerScriptPath = Config.TexEnhancerScriptPath;
    if (!Config.EstimatedMaterialsDir.IsEmpty()) EstimatedMaterialsDir = Config.EstimatedMaterialsDir;

    GTActorListItems.Empty();
    for (const FString& Label : Config.GTActorLabels)
        GTActorListItems.Add(MakeShareable(new FString(Label)));
    if (GTActorListView.IsValid())
        GTActorListView->RequestListRefresh();
}

