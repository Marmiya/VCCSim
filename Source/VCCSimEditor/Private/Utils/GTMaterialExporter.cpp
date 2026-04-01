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
#include "Engine/Texture2D.h"
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

DEFINE_LOG_CATEGORY_STATIC(LogGTMaterialExporter, Log, All);

// ============================================================================
// INTERNAL POD STRUCTURES
// ============================================================================

struct FGTMaterialExporter::FGTRawTex
{
    TArray64<uint8> Bytes;
    int32  W = 0, H = 0, Ch = -1;
    float  Fallback = 0.f;
};

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
        FString   MatName;
        FGTRawTex Rough, Metal, Color;
        int32     TileIdx = 0;
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

    // ── Game thread: minimal UObject access, copy to plain arrays ──────────

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

        TArray<TSharedPtr<FJsonValue>> SlotArray;
        for (int32 si = 0; si < SlotCounts[ai]; ++si)
        {
            UMaterialInterface* Mat = MC->GetMaterial(si);
            const int32 TileIdx = ActorTileOffsets[ai] + si;

            FGTMeshRaw::FSlotRaw Slot;
            Slot.MatName = Mat ? Mat->GetName() : TEXT("");
            Slot.TileIdx = TileIdx;
            GT_CollectMatChannel(Mat, true,  Slot.Rough);
            GT_CollectMatChannel(Mat, false, Slot.Metal);
            GT_CollectBaseColor (Mat, Slot.Color);

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

    // ── Dispatch ALL heavy computation and I/O to background thread ────────
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
                for (int32 k=0; k<3; ++k)
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

        // ── Step 2: Sample atlas tiles ─────────────────────────────────────
        TArray<TArray<FColor>> ORMTiles, ColorTiles;
        ORMTiles.SetNum(TotalTiles); ColorTiles.SetNum(TotalTiles);
        for (const FGTMeshRaw& Raw : RawMeshes)
        {
            for (const FGTMeshRaw::FSlotRaw& S : Raw.Slots)
            {
                ORMTiles[S.TileIdx]   = BG_BuildORMTile(S.Rough, S.Metal, TextureResolution);
                ColorTiles[S.TileIdx] = BG_SampleFromRaw(S.Color, TextureResolution);
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
// GAME-THREAD HELPERS (UE OBJECT ACCESS)
// ============================================================================

void FGTMaterialExporter::GT_ExtractRawTex(UTexture2D* Tex, int32 Ch, FGTRawTex& Out)
{
    if (!Tex) return;
    FTextureSource& S = Tex->Source;
    if (!S.IsValid() || S.GetFormat() != TSF_BGRA8) return;
    Out.W = S.GetSizeX(); Out.H = S.GetSizeY(); Out.Ch = Ch;
    S.GetMipData(Out.Bytes, 0);
}

void FGTMaterialExporter::GT_CollectMatChannel(UMaterialInterface* Mat, bool bRough, FGTRawTex& Out)
{
    Out.Fallback = bRough ? 1.f : 0.f;
    if (!Mat) return;

    auto Get = [&](const FString& N) -> UTexture2D* {
        UTexture* T = nullptr;
        Mat->GetTextureParameterValue(FHashedMaterialParameterInfo(FName(*N)), T);
        return T ? Cast<UTexture2D>(T) : nullptr;
    };

    if (UTexture2D* ORM = Get(TEXT("ORM")))
    {
        GT_ExtractRawTex(ORM, bRough ? 1 : 2, Out);
        if (Out.Bytes.Num() > 0) return;
    }
    static const TArray<FString> RN = { TEXT("Roughness"), TEXT("RoughnessMap"), TEXT("T_Roughness"), TEXT("MetallicRoughnessTexture") };
    static const TArray<FString> MN = { TEXT("Metallic"),  TEXT("MetallicMap"),  TEXT("T_Metallic"),  TEXT("MetallicRoughnessTexture") };
    for (const FString& N : (bRough ? RN : MN))
    {
        if (UTexture2D* T = Get(N)) { GT_ExtractRawTex(T, -1, Out); if (Out.Bytes.Num() > 0) return; }
    }
    float V = Out.Fallback;
    Mat->GetScalarParameterValue(FHashedMaterialParameterInfo(FName(bRough ? TEXT("Roughness") : TEXT("Metallic"))), V);
    Out.Fallback = V;
}

void FGTMaterialExporter::GT_CollectBaseColor(UMaterialInterface* Mat, FGTRawTex& Out)
{
    Out.Fallback = 1.f;
    if (!Mat) return;

    auto Get = [&](const FString& N) -> UTexture2D* {
        UTexture* T = nullptr;
        Mat->GetTextureParameterValue(FHashedMaterialParameterInfo(FName(*N)), T);
        return T ? Cast<UTexture2D>(T) : nullptr;
    };
    static const TArray<FString> TexNames = { TEXT("BaseColor"), TEXT("Base Color"), TEXT("BaseColorMap"), TEXT("Albedo"), TEXT("AlbedoMap"), TEXT("DiffuseColor"), TEXT("Diffuse"), TEXT("T_BaseColor"), TEXT("Color") };
    for (const FString& N : TexNames)
    {
        if (UTexture2D* T = Get(N)) { GT_ExtractRawTex(T, -1, Out); if (Out.Bytes.Num() > 0) return; }
    }

    FLinearColor Vec = FLinearColor::White;
    for (const FString& N : { FString(TEXT("BaseColor")), FString(TEXT("Base Color")), FString(TEXT("Color")) })
    {
        if (Mat->GetVectorParameterValue(FHashedMaterialParameterInfo(FName(*N)), Vec))
        {
            const FColor C = Vec.ToFColor(true);
            Out.Bytes.SetNumUninitialized(4);
            Out.Bytes[0] = C.B; Out.Bytes[1] = C.G; Out.Bytes[2] = C.R; Out.Bytes[3] = C.A;
            Out.W = Out.H = 1; Out.Ch = -1;
            return;
        }
    }
}

// ============================================================================
// BACKGROUND-SAFE HELPERS
// ============================================================================

TArray<FColor> FGTMaterialExporter::BG_SampleFromRaw(const FGTRawTex& R, int32 TargetSize)
{
    TArray<FColor> Out;
    if (R.Bytes.IsEmpty())
    {
        const uint8 V = (uint8)FMath::Clamp(FMath::RoundToInt(R.Fallback * 255.f), 0, 255);
        Out.Init(FColor(V, V, V, 255), TargetSize * TargetSize);
        return Out;
    }
    const int32 N = R.W * R.H;
    TArray<FColor> Src;
    Src.SetNumUninitialized(N);
    for (int32 i = 0; i < N; ++i)
    {
        const uint8 B = R.Bytes[i*4], G = R.Bytes[i*4+1], Rv = R.Bytes[i*4+2], A = R.Bytes[i*4+3];
        if      (R.Ch == 0) Src[i] = FColor(Rv, Rv, Rv, 255);
        else if (R.Ch == 1) Src[i] = FColor(G,  G,  G,  255);
        else if (R.Ch == 2) Src[i] = FColor(B,  B,  B,  255);
        else                Src[i] = FColor(Rv, G,  B,  A);
    }

    if (R.W == TargetSize && R.H == TargetSize) return Src;
    Out.SetNumUninitialized(TargetSize * TargetSize);
    for (int32 Dy = 0; Dy < TargetSize; ++Dy)
    for (int32 Dx = 0; Dx < TargetSize; ++Dx)
    {
        const int32 Sx = FMath::Clamp(Dx * R.W / TargetSize, 0, R.W - 1);
        const int32 Sy = FMath::Clamp(Dy * R.H / TargetSize, 0, R.H - 1);
        Out[Dy * TargetSize + Dx] = Src[Sy * R.W + Sx];
    }
    return Out;
}

TArray<FColor> FGTMaterialExporter::BG_BuildORMTile(const FGTRawTex& Rough, const FGTRawTex& Metal, int32 TargetSize)
{
    const TArray<FColor> RoughSampled = BG_SampleFromRaw(Rough, TargetSize);
    const TArray<FColor> MetalSampled = BG_SampleFromRaw(Metal, TargetSize);
    const int32 N = TargetSize * TargetSize;
    TArray<FColor> Out;
    Out.SetNumUninitialized(N);
    for (int32 i = 0; i < N; ++i)
        Out[i] = FColor(0, RoughSampled[i].R, MetalSampled[i].R, 255);
    return Out;
}

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
