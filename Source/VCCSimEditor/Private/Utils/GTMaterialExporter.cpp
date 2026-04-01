#include "Utils/GTMaterialExporter.h"
#include "Utils/VCCSimUIHelpers.h"
#include "Engine/StaticMeshActor.h"
#include "Components/StaticMeshComponent.h"
#include "EngineUtils.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "HAL/PlatformFileManager.h"
#include "Misc/ScopedSlowTask.h"
#include "Exporters/GLTFExporter.h"
#include "Options/GLTFExportOptions.h"
#include "UserData/GLTFMaterialUserData.h"

DEFINE_LOG_CATEGORY_STATIC(LogGTMaterialExporter, Log, All);

FGTMaterialExporter::FGTMaterialExporter() {}

bool FGTMaterialExporter::WriteManifest(
    AStaticMeshActor* Actor,
    const FString& Label,
    const FString& SceneName,
    int32 TextureResolution,
    const FString& ActorDir)
{
    FString GltfContent;
    if (!FFileHelper::LoadFileToString(GltfContent, *(ActorDir / TEXT("mesh.gltf"))))
    {
        UE_LOG(LogGTMaterialExporter, Warning, TEXT("GT Manifest '%s': cannot read mesh.gltf"), *Label);
        return false;
    }

    TSharedPtr<FJsonObject> GltfRoot;
    TSharedRef<TJsonReader<>> GltfReader = TJsonReaderFactory<>::Create(GltfContent);
    if (!FJsonSerializer::Deserialize(GltfReader, GltfRoot) || !GltfRoot.IsValid())
    {
        UE_LOG(LogGTMaterialExporter, Warning, TEXT("GT Manifest '%s': failed to parse mesh.gltf"), *Label);
        return false;
    }

    TArray<FString> ImageUris;
    const TArray<TSharedPtr<FJsonValue>>* ImagesArr = nullptr;
    if (GltfRoot->TryGetArrayField(TEXT("images"), ImagesArr))
    {
        for (const TSharedPtr<FJsonValue>& Img : *ImagesArr)
        {
            FString Uri;
            if (Img->AsObject().IsValid()) Img->AsObject()->TryGetStringField(TEXT("uri"), Uri);
            ImageUris.Add(Uri);
        }
    }

    TArray<int32> TextureSources;
    const TArray<TSharedPtr<FJsonValue>>* TexturesArr = nullptr;
    if (GltfRoot->TryGetArrayField(TEXT("textures"), TexturesArr))
    {
        for (const TSharedPtr<FJsonValue>& Tex : *TexturesArr)
        {
            int32 Source = -1;
            if (Tex->AsObject().IsValid()) Tex->AsObject()->TryGetNumberField(TEXT("source"), Source);
            TextureSources.Add(Source);
        }
    }

    auto GetImageUri = [&](int32 TexIdx) -> FString
    {
        if (TexIdx < 0 || TexIdx >= TextureSources.Num()) return TEXT("");
        const int32 SrcIdx = TextureSources[TexIdx];
        if (SrcIdx < 0 || SrcIdx >= ImageUris.Num()) return TEXT("");
        return ImageUris[SrcIdx];
    };

    TArray<TSharedPtr<FJsonObject>> GltfMaterials;
    const TArray<TSharedPtr<FJsonValue>>* MaterialsArr = nullptr;
    if (GltfRoot->TryGetArrayField(TEXT("materials"), MaterialsArr))
        for (const TSharedPtr<FJsonValue>& M : *MaterialsArr)
            if (M->AsObject().IsValid()) GltfMaterials.Add(M->AsObject());

    TArray<int32> PrimToMaterial;
    const TArray<TSharedPtr<FJsonValue>>* MeshesArr = nullptr;
    if (GltfRoot->TryGetArrayField(TEXT("meshes"), MeshesArr) && MeshesArr->Num() > 0)
    {
        const TArray<TSharedPtr<FJsonValue>>* PrimsArr = nullptr;
        if ((*MeshesArr)[0]->AsObject().IsValid() &&
            (*MeshesArr)[0]->AsObject()->TryGetArrayField(TEXT("primitives"), PrimsArr))
        {
            for (const TSharedPtr<FJsonValue>& Prim : *PrimsArr)
            {
                int32 MatIdx = -1;
                if (Prim->AsObject().IsValid()) Prim->AsObject()->TryGetNumberField(TEXT("material"), MatIdx);
                PrimToMaterial.Add(MatIdx);
            }
        }
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

    UStaticMeshComponent* MC = Actor->GetStaticMeshComponent();
    TArray<TSharedPtr<FJsonValue>> SlotsArray;

    for (int32 PrimIdx = 0; PrimIdx < PrimToMaterial.Num(); ++PrimIdx)
    {
        const int32 MatIdx = PrimToMaterial[PrimIdx];

        TSharedPtr<FJsonObject> Slot = MakeShareable(new FJsonObject);
        Slot->SetNumberField(TEXT("slot"), PrimIdx);

        FString MatName;
        if (MC && PrimIdx < MC->GetNumMaterials())
        {
            UMaterialInterface* Mat = MC->GetMaterial(PrimIdx);
            MatName = Mat ? Mat->GetName() : TEXT("");
        }
        Slot->SetStringField(TEXT("material_name"), MatName);

        if (MatIdx >= 0 && MatIdx < GltfMaterials.Num())
        {
            const TSharedPtr<FJsonObject>& GltfMat = GltfMaterials[MatIdx];

            const TSharedPtr<FJsonObject>* PBR = nullptr;
            if (GltfMat->TryGetObjectField(TEXT("pbrMetallicRoughness"), PBR))
            {
                const TSharedPtr<FJsonObject>* BCTex = nullptr;
                if ((*PBR)->TryGetObjectField(TEXT("baseColorTexture"), BCTex))
                {
                    int32 TexIdx = -1;
                    (*BCTex)->TryGetNumberField(TEXT("index"), TexIdx);
                    Slot->SetStringField(TEXT("basecolor"), GetImageUri(TexIdx));
                }

                const TSharedPtr<FJsonObject>* MRTex = nullptr;
                if ((*PBR)->TryGetObjectField(TEXT("metallicRoughnessTexture"), MRTex))
                {
                    int32 TexIdx = -1;
                    (*MRTex)->TryGetNumberField(TEXT("index"), TexIdx);
                    Slot->SetStringField(TEXT("metallic_roughness"), GetImageUri(TexIdx));
                }
            }

            const TSharedPtr<FJsonObject>* NTex = nullptr;
            if (GltfMat->TryGetObjectField(TEXT("normalTexture"), NTex))
            {
                int32 TexIdx = -1;
                (*NTex)->TryGetNumberField(TEXT("index"), TexIdx);
                Slot->SetStringField(TEXT("normal"), GetImageUri(TexIdx));
            }

            const TSharedPtr<FJsonObject>* OTex = nullptr;
            if (GltfMat->TryGetObjectField(TEXT("occlusionTexture"), OTex))
            {
                int32 TexIdx = -1;
                (*OTex)->TryGetNumberField(TEXT("index"), TexIdx);
                Slot->SetStringField(TEXT("occlusion"), GetImageUri(TexIdx));
            }
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
        if (AStaticMeshActor* A = *It) LabelMap.Add(A->GetActorLabel(), A);

    TArray<AStaticMeshActor*> Actors;
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
        if (MC && MC->GetNumMaterials() > 0)
        {
            Actors.Add(*Found);
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

    UGLTFExportOptions* Options = NewObject<UGLTFExportOptions>();
    Options->ExportUniformScale = 0.01f;
    Options->BakeMaterialInputs = EGLTFMaterialBakeMode::UseMeshData;
    Options->DefaultMaterialBakeSize = FGLTFMaterialBakeSize{ TextureResolution, TextureResolution, false };
    Options->TextureImageFormat = EGLTFTextureImageFormat::PNG;
    Options->bAdjustNormalmaps = true;
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
        AStaticMeshActor* Actor = Actors[i];

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

    const int32 Total = Actors.Num();
    const FString Msg = (SuccessCount == Total)
        ? FString::Printf(TEXT("GT export done: %d/%d actors -> %s"), SuccessCount, Total, *BaseDir)
        : FString::Printf(TEXT("GT export completed with errors: %d/%d succeeded -> %s"), SuccessCount, Total, *BaseDir);
    UE_LOG(LogGTMaterialExporter, Log, TEXT("%s"), *Msg);
    FVCCSimUIHelpers::ShowNotification(Msg, SuccessCount < Total);
    OnComplete.ExecuteIfBound();
}
