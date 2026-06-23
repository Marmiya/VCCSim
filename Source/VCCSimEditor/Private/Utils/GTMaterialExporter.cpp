#include "Utils/GTMaterialExporter.h"
#include "Utils/VCCSimUIHelpers.h"
#include "IO/GLTFUtils.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/StaticMesh.h"
#include "Components/StaticMeshComponent.h"
#include "EngineUtils.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/SecureHash.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonWriter.h"
#include "Materials/MaterialInterface.h"
#include "Materials/Material.h"
#include "HAL/PlatformFileManager.h"
#include "Misc/ScopedSlowTask.h"
#include "Misc/ScopeExit.h"
#include "Exporters/GLTFExporter.h"
#include "Options/GLTFExportOptions.h"
#include "UserData/GLTFMaterialUserData.h"
#include "Modules/ModuleManager.h"
#include "Components/DynamicMeshComponent.h"
#include "UDynamicMesh.h"
#include "DynamicMesh/DynamicMesh3.h"
#include "DynamicMeshToMeshDescription.h"
#include "StaticMeshAttributes.h"
#include "MeshDescription.h"
#include "Pawns/FlashPawn.h"
#include "Pawns/SimLookAtPath.h"

DEFINE_LOG_CATEGORY_STATIC(LogGTMaterialExporter, Log, All);

FGTMaterialExporter::FGTMaterialExporter() {}

bool FGTMaterialExporter::HasExportableMeshMaterials(const AActor* Actor)
{
    if (!Actor) return false;

    TArray<UStaticMeshComponent*> MeshComps;
    Actor->GetComponents<UStaticMeshComponent>(MeshComps);
    for (UStaticMeshComponent* MC : MeshComps)
    {
        if (MC && MC->GetStaticMesh() && MC->GetNumMaterials() > 0)
        {
            return true;
        }
    }
    return false;
}

bool FGTMaterialExporter::HasExportableMeshGeometry(const AActor* Actor)
{
    if (!Actor) return false;
    if (HasExportableMeshMaterials(Actor)) return true;

    TArray<UDynamicMeshComponent*> DynComps;
    Actor->GetComponents<UDynamicMeshComponent>(DynComps);
    for (UDynamicMeshComponent* DMC : DynComps)
    {
        if (!DMC) continue;
        UDynamicMesh* DynObj = DMC->GetDynamicMesh();
        if (DynObj && DynObj->GetTriangleCount() > 0)
        {
            return true;
        }
    }
    return false;
}

UStaticMesh* FGTMaterialExporter::BuildStaticMeshFromDynamic(UDynamicMeshComponent* DMC)
{
    if (!DMC) return nullptr;
    UDynamicMesh* DynObj = DMC->GetDynamicMesh();
    if (!DynObj || DynObj->GetTriangleCount() == 0) return nullptr;

    FMeshDescription MeshDesc;
    FStaticMeshAttributes Attributes(MeshDesc);
    Attributes.Register();

    DynObj->ProcessMesh([&MeshDesc](const UE::Geometry::FDynamicMesh3& ReadMesh)
    {
        FDynamicMeshToMeshDescription Converter;
        Converter.Convert(&ReadMesh, MeshDesc, /*bCopyTangents=*/true);
    });

    if (MeshDesc.Vertices().Num() == 0 || MeshDesc.Triangles().Num() == 0)
        return nullptr;

    UStaticMesh* StaticMesh = NewObject<UStaticMesh>(GetTransientPackage(), NAME_None, RF_Transient);
    StaticMesh->NeverStream = true;

    // Mirror the component's material slots so per-slot is_glass classification survives.
    const int32 NumMats = FMath::Max(DMC->GetNumMaterials(), 1);
    for (int32 i = 0; i < NumMats; ++i)
    {
        UMaterialInterface* Mat = DMC->GetMaterial(i);
        if (!Mat) Mat = UMaterial::GetDefaultMaterial(MD_Surface);
        StaticMesh->GetStaticMaterials().Add(FStaticMaterial(Mat, *FString::Printf(TEXT("Slot_%d"), i)));
    }

    UStaticMesh::FBuildMeshDescriptionsParams BuildParams;
    BuildParams.bBuildSimpleCollision = false;
    BuildParams.bFastBuild = true;

    TArray<const FMeshDescription*> Descriptions;
    Descriptions.Add(&MeshDesc);
    if (!StaticMesh->BuildFromMeshDescriptions(Descriptions, BuildParams))
        return nullptr;

    return StaticMesh;
}

bool FGTMaterialExporter::IsGlassMaterial(const UMaterialInterface* Mat)
{
    if (!Mat) return false;

    const FString Path = Mat->GetPathName().ToLower();
    static const TCHAR* GlassTokens[] = {
        TEXT("window"), TEXT("glass"), TEXT("interior")
    };
    for (const TCHAR* Token : GlassTokens)
    {
        if (Path.Contains(Token)) return true;
    }

    if (Mat->GetBlendMode() == BLEND_Translucent) return true;
    if (Mat->GetShadingModels().HasShadingModel(MSM_ThinTranslucent)) return true;
    return false;
}

bool FGTMaterialExporter::IsCaptureInfraActor(const AActor* Actor)
{
    if (!Actor) return true;
    // Our own capture / visualization infrastructure — never a scene target.
    return Actor->IsA<AFlashPawn>() || Actor->IsA<AVCCSimLookAtPath>();
}

bool FGTMaterialExporter::WriteManifest(
    AActor* Actor,
    const FString& Label,
    const FString& SceneName,
    int32 TextureResolution,
    const FString& ActorDir)
{
    FGLTFMeshData GltfData;
    if (!FGLTFUtils::LoadMeshData(ActorDir / TEXT("mesh.gltf"), GltfData))
    {
        UE_LOG(LogGTMaterialExporter, Warning, TEXT("GT Manifest '%s': cannot read or parse mesh.gltf"), *Label);
        return false;
    }

    auto MakeVec3Json = [](FVector V) -> TArray<TSharedPtr<FJsonValue>>
    {
        return {
            MakeShareable(new FJsonValueNumber(V.X)),
            MakeShareable(new FJsonValueNumber(V.Y)),
            MakeShareable(new FJsonValueNumber(V.Z))
        };
    };

    TSharedPtr<FJsonObject> Root = MakeShareable(new FJsonObject);
    Root->SetStringField(TEXT("label"), Label);
    Root->SetStringField(TEXT("scene_name"), SceneName);
    Root->SetStringField(TEXT("exported_at"), FDateTime::Now().ToString());
    Root->SetStringField(TEXT("gltf_file"), TEXT("mesh.gltf"));
    Root->SetNumberField(TEXT("texture_resolution"), TextureResolution);

    {
        TSharedPtr<FJsonObject> T = MakeShareable(new FJsonObject);
        const FRotator Rot = Actor->GetActorRotation();
        T->SetArrayField(TEXT("location"), MakeVec3Json(Actor->GetActorLocation()));
        T->SetArrayField(TEXT("rotation"), TArray<TSharedPtr<FJsonValue>>{
            MakeShareable(new FJsonValueNumber(Rot.Pitch)),
            MakeShareable(new FJsonValueNumber(Rot.Yaw)),
            MakeShareable(new FJsonValueNumber(Rot.Roll))
        });
        T->SetArrayField(TEXT("scale"), MakeVec3Json(Actor->GetActorScale3D()));
        Root->SetObjectField(TEXT("ue_transform"), T);
    }

    {
        FVector Origin, Extent;
        Actor->GetActorBounds(false, Origin, Extent);
        TSharedPtr<FJsonObject> AABB = MakeShareable(new FJsonObject);
        AABB->SetArrayField(TEXT("min"), MakeVec3Json(Origin - Extent));
        AABB->SetArrayField(TEXT("max"), MakeVec3Json(Origin + Extent));
        Root->SetObjectField(TEXT("world_aabb_ue"), AABB);
    }

    TArray<UMaterialInterface*> Materials;
    {
        TArray<UStaticMeshComponent*> MeshComps;
        Actor->GetComponents<UStaticMeshComponent>(MeshComps);
        for (UStaticMeshComponent* MC : MeshComps)
        {
            if (!MC || !MC->GetStaticMesh()) continue;
            for (int32 m = 0; m < MC->GetNumMaterials(); ++m)
            {
                Materials.Add(MC->GetMaterial(m));
            }
        }
    }

    TArray<TSharedPtr<FJsonValue>> SlotsArray;

    for (int32 PrimIdx = 0; PrimIdx < GltfData.PrimToMaterial.Num(); ++PrimIdx)
    {
        const int32 MatIdx = GltfData.PrimToMaterial[PrimIdx];

        TSharedPtr<FJsonObject> Slot = MakeShareable(new FJsonObject);
        Slot->SetNumberField(TEXT("slot"), PrimIdx);

        FString MatName;
        bool bIsGlass = false;
        if (Materials.IsValidIndex(PrimIdx) && Materials[PrimIdx])
        {
            MatName = Materials[PrimIdx]->GetName();
            bIsGlass = IsGlassMaterial(Materials[PrimIdx]);
        }
        Slot->SetStringField(TEXT("material_name"), MatName);
        Slot->SetBoolField(TEXT("is_glass"), bIsGlass);

        if (MatIdx >= 0 && MatIdx < GltfData.Materials.Num())
        {
            const FGLTFMeshData::FMaterialTextures& MatTex = GltfData.Materials[MatIdx];
            if (!MatTex.BaseColor.IsEmpty())
                Slot->SetStringField(TEXT("basecolor"), MatTex.BaseColor);
            if (!MatTex.MetallicRoughness.IsEmpty())
                Slot->SetStringField(TEXT("metallic_roughness"), MatTex.MetallicRoughness);
            if (!MatTex.Normal.IsEmpty())
                Slot->SetStringField(TEXT("normal"), MatTex.Normal);
            if (!MatTex.Occlusion.IsEmpty())
                Slot->SetStringField(TEXT("occlusion"), MatTex.Occlusion);
        }

        SlotsArray.Add(MakeShareable(new FJsonValueObject(Slot)));
    }

    Root->SetArrayField(TEXT("slots"), SlotsArray);

    FString JsonStr;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&JsonStr);
    FJsonSerializer::Serialize(Root.ToSharedRef(), Writer);
    return FFileHelper::SaveStringToFile(JsonStr, *(ActorDir / TEXT("manifest.json")));
}

void FGTMaterialExporter::ExportMaterials(
    const TArray<FString>& ActorLabels,
    const TArray<FGTFoliageExportEntry>& FoliageEntries,
    UWorld* World,
    const FString& BaseDir,
    const FString& SceneName,
    int32 TextureResolution,
    const FString& Signature,
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

    TMap<FString, AActor*> LabelMap;
    for (TActorIterator<AActor> It(World); It; ++It)
        if (AActor* A = *It) LabelMap.Add(A->GetActorLabel(), A);

    TArray<AActor*> Actors;
    TArray<FString> Labels;
    TArray<AActor*> TempFoliageActors;

    for (const FString& Label : ActorLabels)
    {
        AActor** Found = LabelMap.Find(Label);
        if (!Found)
        {
            UE_LOG(LogGTMaterialExporter, Warning, TEXT("GT Export: actor '%s' not found"), *Label);
            continue;
        }
        if (HasExportableMeshMaterials(*Found))
        {
            Actors.Add(*Found);
            Labels.Add(Label);
            continue;
        }

        // DynamicMeshActor roofs: convert each dynamic mesh component to a transient static-mesh
        // actor so it exports through the normal static-mesh path.
        TArray<UDynamicMeshComponent*> DynComps;
        (*Found)->GetComponents<UDynamicMeshComponent>(DynComps);
        int32 Converted = 0;
        for (int32 di = 0; di < DynComps.Num(); ++di)
        {
            UStaticMesh* SM = BuildStaticMeshFromDynamic(DynComps[di]);
            if (!SM) continue;

            FActorSpawnParameters SpawnParams;
            SpawnParams.ObjectFlags |= RF_Transient;
            SpawnParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
            AStaticMeshActor* Temp = World->SpawnActor<AStaticMeshActor>(
                AStaticMeshActor::StaticClass(), DynComps[di]->GetComponentTransform(), SpawnParams);
            if (!Temp) continue;

            Temp->SetMobility(EComponentMobility::Movable);
            if (UStaticMeshComponent* MC = Temp->GetStaticMeshComponent())
                MC->SetStaticMesh(SM);
            Temp->SetActorEnableCollision(false);
            Temp->SetActorHiddenInGame(true);

            const FString OutLabel = (DynComps.Num() > 1) ? FString::Printf(TEXT("%s_%d"), *Label, di) : Label;
            Temp->SetActorLabel(OutLabel);
            Actors.Add(Temp);
            Labels.Add(OutLabel);
            TempFoliageActors.Add(Temp);
            ++Converted;
        }
        if (Converted == 0)
        {
            UE_LOG(LogGTMaterialExporter, Warning,
                TEXT("GT Export: actor '%s' has no exportable static or dynamic mesh, skipped"), *Label);
        }
    }

    for (const FGTFoliageExportEntry& Entry : FoliageEntries)
    {
        UStaticMesh* Mesh = Entry.Mesh.Get();
        if (!Mesh)
        {
            UE_LOG(LogGTMaterialExporter, Warning, TEXT("GT Export: foliage '%s' mesh is null"), *Entry.Label);
            continue;
        }

        FActorSpawnParameters SpawnParams;
        SpawnParams.ObjectFlags |= RF_Transient;
        SpawnParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
        AStaticMeshActor* TempActor = World->SpawnActor<AStaticMeshActor>(
            AStaticMeshActor::StaticClass(),
            Entry.WorldTransform,
            SpawnParams);
        if (!TempActor)
        {
            UE_LOG(LogGTMaterialExporter, Warning, TEXT("GT Export: failed to spawn temp actor for foliage '%s'"), *Entry.Label);
            continue;
        }

        TempActor->SetMobility(EComponentMobility::Movable);
        if (UStaticMeshComponent* MC = TempActor->GetStaticMeshComponent())
        {
            MC->SetStaticMesh(Mesh);
        }
        TempActor->SetActorEnableCollision(false);
        TempActor->SetActorHiddenInGame(true);
        TempActor->SetActorLabel(Entry.Label);

        if (HasExportableMeshMaterials(TempActor))
        {
            Actors.Add(TempActor);
            Labels.Add(Entry.Label);
            TempFoliageActors.Add(TempActor);
        }
        else
        {
            World->DestroyActor(TempActor);
        }
    }

    ON_SCOPE_EXIT
    {
        for (AActor* TempActor : TempFoliageActors)
        {
            if (IsValid(TempActor))
            {
                World->DestroyActor(TempActor);
            }
        }
    };

    if (Actors.IsEmpty())
    {
        const FString Msg = TEXT("No valid actors with materials to export.");
        UE_LOG(LogGTMaterialExporter, Warning, TEXT("%s"), *Msg);
        FVCCSimUIHelpers::ShowNotification(Msg, true);
        OnComplete.ExecuteIfBound();
        return;
    }

    UGLTFExportOptions* Options = NewObject<UGLTFExportOptions>();
    Options->ExportUniformScale = 0.01f;
    Options->BakeMaterialInputs = EGLTFMaterialBakeMode::Disabled;
    Options->bExportLights = false;
    Options->bExportCameras = false;
    Options->bExportAnimationSequences = false;
    Options->bExportLevelSequences = false;
    Options->DefaultLevelOfDetail = 0;

    int32 SuccessCount = 0;
    FScopedSlowTask SlowTask(Actors.Num(), FText::FromString(TEXT("Exporting GT materials...")));
    SlowTask.MakeDialog(true);

    for (int32 i = 0; i < Actors.Num(); ++i)
    {
        const FString& Label = Labels[i];
        AActor* Actor = Actors[i];

        SlowTask.EnterProgressFrame(1.0f, FText::FromString(FString::Printf(TEXT("Exporting: %s (%d/%d)"), *Label, i + 1, Actors.Num())));
        if (SlowTask.ShouldCancel())
        {
            UE_LOG(LogGTMaterialExporter, Warning, TEXT("GT Export cancelled by user after %d/%d actors"), i, Actors.Num());
            break;
        }

        const FString ActorDir = BaseDir / Label;
        IFileManager::Get().MakeDirectory(*ActorDir, true);
        const FString GltfPath = ActorDir / TEXT("mesh.gltf");

        TSet<AActor*> SelectedActors = { Actor };
        FGLTFExportMessages Messages;
        const bool bExportOk = UGLTFExporter::ExportToGLTF(World, GltfPath, Options, SelectedActors, Messages);

        for (const FString& W : Messages.Warnings)
            UE_LOG(LogGTMaterialExporter, Warning, TEXT("GLTF '%s': %s"), *Label, *W);
        for (const FString& E : Messages.Errors)
            UE_LOG(LogGTMaterialExporter, Error, TEXT("GLTF '%s': %s"), *Label, *E);

        if (!bExportOk)
        {
            UE_LOG(LogGTMaterialExporter, Warning, TEXT("GT Export '%s': GLTFExporter failed"), *Label);
            continue;
        }

        const bool bManifestOk = WriteManifest(Actor, Label, SceneName, TextureResolution, ActorDir);
        if (!bManifestOk)
        {
            UE_LOG(LogGTMaterialExporter, Warning, TEXT("GT Export '%s': manifest write failed"), *Label);
        }
        else
        {
            ++SuccessCount;
        }
    }

    {
        TSharedPtr<FJsonObject> SceneRoot = MakeShareable(new FJsonObject);
        SceneRoot->SetStringField(TEXT("scene_name"), SceneName);
        SceneRoot->SetStringField(TEXT("exported_at"), FDateTime::Now().ToString());
        SceneRoot->SetStringField(TEXT("signature"), Signature);
        TArray<TSharedPtr<FJsonValue>> ActorsJson;
        for (int32 i = 0; i < Actors.Num(); ++i)
        {
            TSharedPtr<FJsonObject> A = MakeShareable(new FJsonObject);
            A->SetStringField(TEXT("label"), Labels[i]);
            ActorsJson.Add(MakeShareable(new FJsonValueObject(A)));
        }
        SceneRoot->SetArrayField(TEXT("actors"), ActorsJson);
        FString SceneJson;
        TSharedRef<TJsonWriter<>> SceneWriter = TJsonWriterFactory<>::Create(&SceneJson);
        FJsonSerializer::Serialize(SceneRoot.ToSharedRef(), SceneWriter);
        FFileHelper::SaveStringToFile(SceneJson, *(BaseDir / TEXT("manifest.json")));
    }

    const int32 Total = Actors.Num();
    const FString Msg = (SuccessCount == Total)
        ? FString::Printf(TEXT("GT export done: %d/%d actors -> %s"), SuccessCount, Total, *BaseDir)
        : FString::Printf(TEXT("GT export completed with errors: %d/%d succeeded -> %s"), SuccessCount, Total, *BaseDir);
    UE_LOG(LogGTMaterialExporter, Log, TEXT("%s"), *Msg);
    FVCCSimUIHelpers::ShowNotification(Msg, SuccessCount < Total);
    OnComplete.ExecuteIfBound();
}

FString FGTMaterialExporter::ComputeSignature(
    UWorld* World,
    const TArray<FString>& SeedLabels,
    const FString& SceneName,
    int32 TextureResolution)
{
    if (!World)
    {
        return FString();
    }

    TMap<FString, AActor*> LabelMap;
    for (TActorIterator<AActor> It(World); It; ++It)
        if (AActor* A = *It) LabelMap.Add(A->GetActorLabel(), A);

    TArray<FString> Sorted = SeedLabels;
    Sorted.Sort();

    FString Canon = FString::Printf(
        TEXT("scene=%s;res=%d"), *SceneName, TextureResolution);

    for (const FString& Label : Sorted)
    {
        Canon += TEXT(";A=") + Label;
        AActor** Found = LabelMap.Find(Label);
        if (!Found || !*Found)
        {
            Canon += TEXT("|missing");
            continue;
        }
        AActor* Actor = *Found;
        Canon += TEXT("|loc=") + Actor->GetActorLocation().ToString();
        Canon += TEXT("|rot=") + Actor->GetActorRotation().ToString();
        Canon += TEXT("|scl=") + Actor->GetActorScale3D().ToString();

        TArray<UStaticMeshComponent*> MeshComps;
        Actor->GetComponents<UStaticMeshComponent>(MeshComps);
        for (UStaticMeshComponent* MC : MeshComps)
        {
            if (!MC) continue;
            UStaticMesh* SM = MC->GetStaticMesh();
            Canon += TEXT("|m=") + (SM ? SM->GetPathName() : FString(TEXT("none")));
            for (int32 m = 0; m < MC->GetNumMaterials(); ++m)
            {
                UMaterialInterface* Mat = MC->GetMaterial(m);
                Canon += TEXT("|mat=") + (Mat ? Mat->GetPathName() : FString(TEXT("none")));
            }
        }
    }

    return FMD5::HashAnsiString(*Canon);
}

FString FGTMaterialExporter::FindReusableExport(
    const FString& CapturesRoot,
    const FString& ExcludeCaptureDirName,
    const FString& Signature)
{
    if (Signature.IsEmpty())
    {
        return FString();
    }

    IFileManager& FileManager = IFileManager::Get();
    TArray<FString> CaptureDirs;
    FileManager.FindFiles(CaptureDirs, *(CapturesRoot / TEXT("capture_*")), false, true);
    CaptureDirs.Sort();

    for (const FString& Dir : CaptureDirs)
    {
        if (Dir == ExcludeCaptureDirName) continue;

        const FString GTDir = CapturesRoot / Dir / TEXT("gt_materials");
        FString JsonStr;
        if (!FFileHelper::LoadFileToString(JsonStr, *(GTDir / TEXT("manifest.json"))))
            continue;

        TSharedPtr<FJsonObject> Root;
        TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonStr);
        if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
            continue;

        FString ExistingSig;
        if (!Root->TryGetStringField(TEXT("signature"), ExistingSig) || ExistingSig != Signature)
            continue;

        // Only reuse a complete export: every listed actor must still have its mesh.gltf.
        const TArray<TSharedPtr<FJsonValue>>* Actors = nullptr;
        if (!Root->TryGetArrayField(TEXT("actors"), Actors) || !Actors || Actors->Num() == 0)
            continue;

        bool bComplete = true;
        for (const TSharedPtr<FJsonValue>& Value : *Actors)
        {
            const TSharedPtr<FJsonObject> AObj = Value.IsValid() ? Value->AsObject() : nullptr;
            FString Label;
            if (!AObj.IsValid() || !AObj->TryGetStringField(TEXT("label"), Label)
                || !FPaths::FileExists(GTDir / Label / TEXT("mesh.gltf")))
            {
                bComplete = false;
                break;
            }
        }

        if (bComplete)
        {
            return GTDir;
        }
    }

    return FString();
}
