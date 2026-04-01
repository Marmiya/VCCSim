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

#include "Utils/GTMaterialExporter.h"
#include "Utils/VCCSimUIHelpers.h"
#include "Utils/VCCSimDataConverter.h"
#include "Engine/StaticMeshActor.h"
#include "Materials/MaterialInterface.h"
#include "MeshDescription.h"
#include "StaticMeshAttributes.h"
#include "EngineUtils.h"
#include "Async/Async.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Modules/ModuleManager.h"
#include "HAL/PlatformFileManager.h"
#include "IMaterialBakingModule.h"
#include "MaterialBakingStructures.h"

DEFINE_LOG_CATEGORY_STATIC(LogGTMaterialExporter, Log, All);

// ============================================================================
// INTERNAL POD STRUCTURES
// ============================================================================

struct FGTMaterialExporter::FGTMeshRaw
{
    FString    Label;
    FTransform WorldTransform;
    int32      ActorTileOffset = 0;

    TArray<FVector3f> LocalVertPos;
    TArray<int32>     InstVertIdx;
    TArray<FVector2f> InstUV0;
    TArray<int32>     TriInstFlat;
    TArray<int32>     TriSlotFlat;

    struct FSlotRaw
    {
        FString        MatName;
        TArray<FColor> BakedColor;
        TArray<FColor> BakedRough;
        TArray<FColor> BakedMetal;
        int32          TileIdx = 0;
    };
    TArray<FSlotRaw> Slots;
};

struct FGTMaterialExporter::FGTActorBuilt
{
    TArray<FVector>   WorldVerts;
    TArray<FVector2f> AtlasUVs;
    TArray<int32>     FaceVerts;
    TArray<int32>     FaceUVs;
};

// ============================================================================
// CONSTRUCTOR
// ============================================================================

FGTMaterialExporter::FGTMaterialExporter() {}

// ============================================================================
// PUBLIC EXPORT METHOD
// ============================================================================

void FGTMaterialExporter::ExportMaterials(
    const TArray<FString>& ActorLabels,
    UWorld* World,
    const FString& BaseDir,
    const FString& SceneName,
    int32 TextureResolution,
    FSimpleDelegate OnComplete)
{
    if (!World)
    {
        const FString Msg = TEXT("GT Exporter: World is not valid.");
        UE_LOG(LogGTMaterialExporter, Error, TEXT("%s"), *Msg);
        FVCCSimUIHelpers::ShowNotification(Msg, true);
        OnComplete.ExecuteIfBound();
        return;
    }

    FPlatformFileManager::Get().GetPlatformFile().CreateDirectoryTree(*BaseDir);

    TMap<FString, AStaticMeshActor*> LabelMap;
    for (TActorIterator<AStaticMeshActor> It(World); It; ++It)
    {
        if (AStaticMeshActor* A = *It) LabelMap.Add(A->GetActorLabel(), A);
    }

    TArray<AStaticMeshActor*> Actors;
    TArray<int32> SlotCounts;
    TArray<FString> Labels;
    for (const FString& Label : ActorLabels)
    {
        AStaticMeshActor** Found = LabelMap.Find(Label);
        if (!Found)
        {
            UE_LOG(LogGTMaterialExporter, Warning, TEXT("GT Export: actor '%s' not found"), *Label);
            continue;
        }
        UStaticMeshComponent* MC = (*Found)->GetStaticMeshComponent();
        const int32 NS = MC ? MC->GetNumMaterials() : 0;
        if (NS > 0)
        {
            Actors.Add(*Found);
            SlotCounts.Add(NS);
            Labels.Add(Label);
        }
    }

    if (Actors.IsEmpty())
    {
        const FString Msg = TEXT("No valid actors with materials to export.");
        UE_LOG(LogGTMaterialExporter, Warning, TEXT("%s"), *Msg);
        FVCCSimUIHelpers::ShowNotification(Msg, true);
        OnComplete.ExecuteIfBound();
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

    // ── Game thread: mesh extraction + material baking via MaterialBaking module ──

    IMaterialBakingModule& BakingModule = FModuleManager::LoadModuleChecked<IMaterialBakingModule>("MaterialBaking");

    TArray<FGTMeshRaw> RawMeshes;
    RawMeshes.Reserve(Actors.Num());
    TSharedPtr<FJsonObject> RootJson = MakeShareable(new FJsonObject);
    TArray<TSharedPtr<FJsonValue>> ActorArray;

    for (int32 ai = 0; ai < Actors.Num(); ++ai)
    {
        UStaticMeshComponent* MC = Actors[ai]->GetStaticMeshComponent();
        UStaticMesh* SM = MC->GetStaticMesh();
        if (!SM) continue;
        SM->ConditionalPostLoad();

        const FMeshDescription* MD = SM->GetMeshDescription(0);
        if (!MD) continue;

        FGTMeshRaw Raw;
        Raw.Label           = Labels[ai];
        Raw.WorldTransform  = Actors[ai]->GetActorTransform();
        Raw.ActorTileOffset = ActorTileOffsets[ai];

        FStaticMeshConstAttributes Attrs(*MD);
        TVertexAttributesConstRef<FVector3f>         Positions = Attrs.GetVertexPositions();
        TVertexInstanceAttributesConstRef<FVector2f> UVs       = Attrs.GetVertexInstanceUVs();

        int32 MaxVID = -1;
        for (const FVertexID VID : MD->Vertices().GetElementIDs()) MaxVID = FMath::Max(MaxVID, VID.GetValue());

        TArray<int32> VIDToLocal;
        VIDToLocal.Init(-1, MaxVID + 1);
        Raw.LocalVertPos.Reserve(MD->Vertices().Num());
        for (const FVertexID VID : MD->Vertices().GetElementIDs())
        {
            VIDToLocal[VID.GetValue()] = Raw.LocalVertPos.Num();
            Raw.LocalVertPos.Add(Positions[VID]);
        }

        int32 MaxIID = -1;
        for (const FVertexInstanceID IID : MD->VertexInstances().GetElementIDs()) MaxIID = FMath::Max(MaxIID, IID.GetValue());

        TArray<int32> IIDToLocal;
        IIDToLocal.Init(-1, MaxIID + 1);
        Raw.InstVertIdx.Reserve(MD->VertexInstances().Num());
        Raw.InstUV0.Reserve(MD->VertexInstances().Num());
        for (const FVertexInstanceID IID : MD->VertexInstances().GetElementIDs())
        {
            IIDToLocal[IID.GetValue()] = Raw.InstVertIdx.Num();
            Raw.InstVertIdx.Add(VIDToLocal[MD->GetVertexInstanceVertex(IID).GetValue()]);
            Raw.InstUV0.Add(UVs.Get(IID, 0));
        }

        TMap<int32, int32> GroupToSlot;
        { int32 Idx = 0; for (const FPolygonGroupID GID : MD->PolygonGroups().GetElementIDs()) GroupToSlot.Add(GID.GetValue(), Idx++); }

        const int32 NumTris = MD->Triangles().Num();
        Raw.TriInstFlat.Reserve(NumTris * 3);
        Raw.TriSlotFlat.Reserve(NumTris);
        for (const FTriangleID TID : MD->Triangles().GetElementIDs())
        {
            Raw.TriSlotFlat.Add(GroupToSlot.FindRef(MD->GetTrianglePolygonGroup(TID).GetValue()));
            for (const FVertexInstanceID IID : MD->GetTriangleVertexInstances(TID))
                Raw.TriInstFlat.Add(IIDToLocal[IID.GetValue()]);
        }

        TSharedPtr<FJsonObject> ActorJson = MakeShareable(new FJsonObject);
        ActorJson->SetStringField(TEXT("label"), Labels[ai]);
        ActorJson->SetStringField(TEXT("mesh_file"), TEXT("merged_mesh.obj"));

        {
            auto MakeVec3Json = [](FVector V) -> TArray<TSharedPtr<FJsonValue>> {
                return { MakeShareable(new FJsonValueNumber(V.X)), MakeShareable(new FJsonValueNumber(V.Y)), MakeShareable(new FJsonValueNumber(V.Z)) };
            };
            const FVector Loc = Actors[ai]->GetActorLocation(), Scale = Actors[ai]->GetActorScale3D();
            const FRotator Rot = Actors[ai]->GetActorRotation();
            TSharedPtr<FJsonObject> TransformJson = MakeShareable(new FJsonObject);
            TransformJson->SetArrayField(TEXT("location"), MakeVec3Json(Loc));
            TransformJson->SetArrayField(TEXT("rotation"), TArray<TSharedPtr<FJsonValue>>{MakeShareable(new FJsonValueNumber(Rot.Pitch)), MakeShareable(new FJsonValueNumber(Rot.Yaw)), MakeShareable(new FJsonValueNumber(Rot.Roll))});
            TransformJson->SetArrayField(TEXT("scale"), MakeVec3Json(Scale));
            ActorJson->SetObjectField(TEXT("actor_transform"), TransformJson);
            FVector Origin, Extent;
            Actors[ai]->GetActorBounds(false, Origin, Extent);
            TSharedPtr<FJsonObject> AABBJson = MakeShareable(new FJsonObject);
            AABBJson->SetArrayField(TEXT("min"), MakeVec3Json(Origin - Extent));
            AABBJson->SetArrayField(TEXT("max"), MakeVec3Json(Origin + Extent));
            ActorJson->SetObjectField(TEXT("world_aabb"), AABBJson);
        }

        const int32 TilePixels = TextureResolution * TextureResolution;
        TArray<TSharedPtr<FJsonValue>> SlotArray;

        for (int32 si = 0; si < SlotCounts[ai]; ++si)
        {
            UMaterialInterface* Mat = MC->GetMaterial(si);
            const int32 TileIdx = ActorTileOffsets[ai] + si;

            FGTMeshRaw::FSlotRaw Slot;
            Slot.MatName = Mat ? Mat->GetName() : TEXT("");
            Slot.TileIdx = TileIdx;

            if (Mat)
            {
                FMaterialData MatData;
                MatData.Material = Mat;
                MatData.PropertySizes.Add(MP_BaseColor, FIntPoint(TextureResolution, TextureResolution));
                MatData.PropertySizes.Add(MP_Roughness, FIntPoint(TextureResolution, TextureResolution));
                MatData.PropertySizes.Add(MP_Metallic,  FIntPoint(TextureResolution, TextureResolution));

                FMeshData MeshData;
                MeshData.Mesh = SM;
                MeshData.TextureCoordinateIndex = 0;
                MeshData.MaterialIndices = { si };

                TArray<FMaterialData*> MatDataArr = { &MatData };
                TArray<FMeshData*>     MeshDataArr = { &MeshData };
                TArray<FBakeOutput>    BakeOutputs;
                BakingModule.BakeMaterials(MatDataArr, MeshDataArr, BakeOutputs);

                if (BakeOutputs.Num() > 0)
                {
                    if (TArray<FColor>* P = BakeOutputs[0].PropertyData.Find(MP_BaseColor)) Slot.BakedColor = MoveTemp(*P);
                    if (TArray<FColor>* P = BakeOutputs[0].PropertyData.Find(MP_Roughness)) Slot.BakedRough = MoveTemp(*P);
                    if (TArray<FColor>* P = BakeOutputs[0].PropertyData.Find(MP_Metallic))  Slot.BakedMetal = MoveTemp(*P);
                }
            }

            if (Slot.BakedColor.Num() != TilePixels) Slot.BakedColor.Init(FColor::White,                   TilePixels);
            if (Slot.BakedRough.Num() != TilePixels) Slot.BakedRough.Init(FColor(255, 255, 255, 255),       TilePixels);
            if (Slot.BakedMetal.Num() != TilePixels) Slot.BakedMetal.Init(FColor::Black,                    TilePixels);

            TSharedPtr<FJsonObject> SlotJson = MakeShareable(new FJsonObject);
            SlotJson->SetNumberField(TEXT("slot"), si);
            SlotJson->SetStringField(TEXT("material_name"), Slot.MatName);
            SlotJson->SetNumberField(TEXT("atlas_tile"), TileIdx);
            SlotArray.Add(MakeShareable(new FJsonValueObject(SlotJson)));
            Raw.Slots.Add(MoveTemp(Slot));
        }
        ActorJson->SetArrayField(TEXT("slots"), SlotArray);
        ActorArray.Add(MakeShareable(new FJsonValueObject(ActorJson)));
        RawMeshes.Add(MoveTemp(Raw));
    }

    if (RawMeshes.IsEmpty())
    {
        const FString Msg = TEXT("No valid mesh data to export.");
        UE_LOG(LogGTMaterialExporter, Warning, TEXT("%s"), *Msg);
        FVCCSimUIHelpers::ShowNotification(Msg, true);
        OnComplete.ExecuteIfBound();
        return;
    }

    TSharedPtr<FJsonObject> MetaJson = MakeShareable(new FJsonObject);
    MetaJson->SetStringField(TEXT("scene_name"), SceneName);
    MetaJson->SetStringField(TEXT("exported_at"), FDateTime::Now().ToString());
    MetaJson->SetNumberField(TEXT("actor_count"), RawMeshes.Num());
    MetaJson->SetNumberField(TEXT("texture_resolution"), TextureResolution);
    MetaJson->SetNumberField(TEXT("atlas_cols"), AtlasCols);
    MetaJson->SetNumberField(TEXT("atlas_rows"), AtlasRows);
    MetaJson->SetStringField(TEXT("basecolor_atlas"), TEXT("basecolor_atlas.png"));
    MetaJson->SetStringField(TEXT("orm_atlas"), TEXT("orm_atlas.png"));
    MetaJson->SetStringField(TEXT("mesh_file"), TEXT("merged_mesh.obj"));
    RootJson->SetObjectField(TEXT("metadata"), MetaJson);
    RootJson->SetArrayField(TEXT("actors"), ActorArray);

    FString JsonStr;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&JsonStr);
    FJsonSerializer::Serialize(RootJson.ToSharedRef(), Writer);

    // ── Dispatch geometry, atlas assembly and file I/O to background thread ──
    Async(EAsyncExecution::Thread,
        [RawMeshes = MoveTemp(RawMeshes), JsonStr = MoveTemp(JsonStr), BaseDir, OnComplete, TextureResolution, AtlasCols, AtlasRows, TotalTiles, SceneName]()
    {
        // ── Step 1: Build geometry and UV data ─────────────────────────────
        TArray<FGTActorBuilt> BuiltActors;
        BuiltActors.Reserve(RawMeshes.Num());
        int32 GlobalVertBase = 1, GlobalUVBase = 1;

        for (const FGTMeshRaw& Raw : RawMeshes)
        {
            const int32 NumVerts = Raw.LocalVertPos.Num(), NumInsts = Raw.InstUV0.Num(), NumTris = Raw.TriSlotFlat.Num();
            FGTActorBuilt Built;
            Built.WorldVerts.SetNumUninitialized(NumVerts);
            for (int32 vi = 0; vi < NumVerts; ++vi)
                Built.WorldVerts[vi] = FVCCSimDataConverter::ConvertLocation(Raw.WorldTransform.TransformPosition(FVector(Raw.LocalVertPos[vi])));

            TArray<int32> InstTile;
            InstTile.Init(-1, NumInsts);
            for (int32 ti = 0; ti < NumTris; ++ti)
            {
                const int32 TileIdx = Raw.ActorTileOffset + Raw.TriSlotFlat[ti];
                for (int32 k = 0; k < 3; ++k)
                {
                    const int32 IIdx = Raw.TriInstFlat[ti * 3 + k];
                    if (IIdx >= 0 && IIdx < NumInsts && InstTile[IIdx] < 0) InstTile[IIdx] = TileIdx;
                }
            }

            Built.AtlasUVs.SetNumUninitialized(NumInsts);
            for (int32 ii = 0; ii < NumInsts; ++ii)
            {
                const int32 Tile = (InstTile[ii] >= 0) ? InstTile[ii] : Raw.ActorTileOffset;
                const int32 Col  = Tile % AtlasCols, Row = Tile / AtlasCols;
                const FVector2f SrcUV = Raw.InstUV0[ii];
                const float U = SrcUV.X / AtlasCols + (float)Col / AtlasCols;
                const float V = (1.f - SrcUV.Y) / AtlasRows + (float)(AtlasRows - Row - 1) / AtlasRows;
                Built.AtlasUVs[ii] = FVector2f(U, V);
            }

            Built.FaceVerts.SetNumUninitialized(NumTris * 3);
            Built.FaceUVs.SetNumUninitialized(NumTris * 3);
            for (int32 ti = 0; ti < NumTris; ++ti)
            {
                for (int32 k = 0; k < 3; ++k)
                {
                    const int32 IIdx = Raw.TriInstFlat[ti * 3 + k];
                    Built.FaceVerts[ti * 3 + k] = GlobalVertBase + Raw.InstVertIdx[IIdx];
                    Built.FaceUVs  [ti * 3 + k] = GlobalUVBase + IIdx;
                }
            }
            GlobalVertBase += NumVerts;
            GlobalUVBase += NumInsts;
            BuiltActors.Add(MoveTemp(Built));
        }

        // ── Step 2: Assemble atlas tiles from baked data ───────────────────
        TArray<TArray<FColor>> ORMTiles, ColorTiles;
        ORMTiles.SetNum(TotalTiles); ColorTiles.SetNum(TotalTiles);
        for (const FGTMeshRaw& Raw : RawMeshes)
        {
            for (const FGTMeshRaw::FSlotRaw& S : Raw.Slots)
            {
                ColorTiles[S.TileIdx] = S.BakedColor;

                const int32 N = S.BakedRough.Num();
                TArray<FColor> ORM;
                ORM.SetNumUninitialized(N);
                for (int32 i = 0; i < N; ++i)
                    ORM[i] = FColor(0, S.BakedRough[i].R, S.BakedMetal[i].R, 255);
                ORMTiles[S.TileIdx] = MoveTemp(ORM);
            }
        }

        // ── Step 3: Write all output files ─────────────────────────────────
        const FString ObjContent = BG_BuildOBJContent(BuiltActors);
        const bool bObjOk   = FFileHelper::SaveStringToFile(ObjContent, *(BaseDir / TEXT("merged_mesh.obj")), FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), FILEWRITE_EvenIfReadOnly);
        const bool bMtlOk   = WriteMTLFile(BaseDir / TEXT("merged_mesh.mtl"));
        const bool bColorOk = WriteAtlasPNG(ColorTiles, TextureResolution, AtlasCols, AtlasRows, BaseDir / TEXT("basecolor_atlas.png"));
        const bool bORMOk   = WriteAtlasPNG(ORMTiles,   TextureResolution, AtlasCols, AtlasRows, BaseDir / TEXT("orm_atlas.png"));
        const bool bJsonOk  = FFileHelper::SaveStringToFile(JsonStr, *(BaseDir / TEXT("manifest.json")));

        if (!bObjOk)   UE_LOG(LogGTMaterialExporter, Warning, TEXT("GT Export: failed merged_mesh.obj"));
        if (!bMtlOk)   UE_LOG(LogGTMaterialExporter, Warning, TEXT("GT Export: failed merged_mesh.mtl"));
        if (!bColorOk) UE_LOG(LogGTMaterialExporter, Warning, TEXT("GT Export: failed basecolor_atlas.png"));
        if (!bORMOk)   UE_LOG(LogGTMaterialExporter, Warning, TEXT("GT Export: failed orm_atlas.png"));
        if (!bJsonOk)  UE_LOG(LogGTMaterialExporter, Warning, TEXT("GT Export: failed manifest.json"));

        const bool bSuccess = bObjOk && bMtlOk && bColorOk && bORMOk && bJsonOk;
        const FString Msg = bSuccess
            ? FString::Printf(TEXT("GT export done: %d actors, %d tiles (%dx%d atlas) -> %s"), RawMeshes.Num(), TotalTiles, AtlasCols, AtlasRows, *BaseDir)
            : FString::Printf(TEXT("GT export completed with errors -> %s"), *BaseDir);
        UE_LOG(LogGTMaterialExporter, Log, TEXT("%s"), *Msg);

        AsyncTask(ENamedThreads::GameThread, [OnComplete, Msg, bSuccess]() {
            FVCCSimUIHelpers::ShowNotification(Msg, !bSuccess);
            OnComplete.ExecuteIfBound();
        });
    });
}

// ============================================================================
// BACKGROUND-SAFE HELPERS
// ============================================================================

FString FGTMaterialExporter::BG_BuildOBJContent(const TArray<FGTActorBuilt>& Built)
{
    FString V = TEXT("mtllib merged_mesh.mtl\nusemtl merged_material\n");
    FString UV_str, F_str;
    for (const FGTActorBuilt& A : Built)
    {
        for (const FVector& P : A.WorldVerts) V += FString::Printf(TEXT("v %f %f %f\n"), P.X, P.Y, P.Z);
        for (const FVector2f& UV : A.AtlasUVs) UV_str += FString::Printf(TEXT("vt %f %f\n"), UV.X, UV.Y);
        for (int32 fi = 0; fi < A.FaceVerts.Num(); fi += 3)
            F_str += FString::Printf(TEXT("f %d/%d %d/%d %d/%d\n"), A.FaceVerts[fi], A.FaceUVs[fi], A.FaceVerts[fi+2], A.FaceUVs[fi+2], A.FaceVerts[fi+1], A.FaceUVs[fi+1]);
    }
    return V + UV_str + F_str;
}

bool FGTMaterialExporter::WriteMTLFile(const FString& MtlPath)
{
    FString Mtl;
    Mtl += TEXT("newmtl merged_material\n");
    Mtl += TEXT("Ka 1.0 1.0 1.0\n");
    Mtl += TEXT("Kd 1.0 1.0 1.0\n");
    Mtl += TEXT("Ks 0.0 0.0 0.0\n");
    Mtl += TEXT("map_Kd basecolor_atlas.png\n");
    Mtl += TEXT("map_ORM orm_atlas.png\n");
    return FFileHelper::SaveStringToFile(Mtl, *MtlPath);
}

bool FGTMaterialExporter::WriteAtlasPNG(const TArray<TArray<FColor>>& Tiles, int32 TileSize, int32 Cols, int32 Rows, const FString& PngPath)
{
    const int32 W = TileSize * Cols, H = TileSize * Rows;
    TArray<FColor> Atlas;
    Atlas.SetNumZeroed(W * H);
    for (int32 TileIdx = 0; TileIdx < Tiles.Num(); ++TileIdx)
    {
        const int32 Col = TileIdx % Cols, Row = TileIdx / Cols;
        const TArray<FColor>& Tile = Tiles[TileIdx];
        for (int32 Py = 0; Py < TileSize; ++Py)
        {
            for (int32 Px = 0; Px < TileSize; ++Px)
            {
                const int32 SrcIdx = Py * TileSize + Px;
                const int32 DstIdx = (Row * TileSize + Py) * W + (Col * TileSize + Px);
                if (SrcIdx < Tile.Num()) Atlas[DstIdx] = Tile[SrcIdx];
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
