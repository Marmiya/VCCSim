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

#include "IO/GLTFUtils.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonSerializer.h"
#include "Dom/JsonObject.h"

FString FGLTFMeshData::ResolveTextureUri(int32 TexIdx) const
{
    if (TexIdx < 0 || TexIdx >= TextureSources.Num()) return TEXT("");
    const int32 SrcIdx = TextureSources[TexIdx];
    if (SrcIdx < 0 || SrcIdx >= ImageUris.Num()) return TEXT("");
    return ImageUris[SrcIdx];
}

bool FGLTFUtils::LoadMeshData(const FString& GltfPath, FGLTFMeshData& OutData)
{
    FString GltfContent;
    if (!FFileHelper::LoadFileToString(GltfContent, *GltfPath)) return false;

    TSharedPtr<FJsonObject> Root;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(GltfContent);
    if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid()) return false;

    const TArray<TSharedPtr<FJsonValue>>* ImagesArr = nullptr;
    if (Root->TryGetArrayField(TEXT("images"), ImagesArr))
        for (const TSharedPtr<FJsonValue>& Img : *ImagesArr)
        {
            FString Uri;
            if (Img->AsObject().IsValid()) Img->AsObject()->TryGetStringField(TEXT("uri"), Uri);
            OutData.ImageUris.Add(Uri);
        }

    const TArray<TSharedPtr<FJsonValue>>* TexturesArr = nullptr;
    if (Root->TryGetArrayField(TEXT("textures"), TexturesArr))
        for (const TSharedPtr<FJsonValue>& Tex : *TexturesArr)
        {
            int32 Src = -1;
            if (Tex->AsObject().IsValid()) Tex->AsObject()->TryGetNumberField(TEXT("source"), Src);
            OutData.TextureSources.Add(Src);
        }

    const TArray<TSharedPtr<FJsonValue>>* MatsArr = nullptr;
    if (Root->TryGetArrayField(TEXT("materials"), MatsArr))
        for (const TSharedPtr<FJsonValue>& M : *MatsArr)
        {
            FGLTFMeshData::FMaterialTextures MatTex;
            if (M->AsObject().IsValid())
            {
                const TSharedPtr<FJsonObject>& MatObj = M->AsObject();
                const TSharedPtr<FJsonObject>* PBR = nullptr;
                if (MatObj->TryGetObjectField(TEXT("pbrMetallicRoughness"), PBR))
                {
                    const TSharedPtr<FJsonObject>* BCTex = nullptr;
                    if ((*PBR)->TryGetObjectField(TEXT("baseColorTexture"), BCTex))
                    {
                        int32 TIdx = -1;
                        (*BCTex)->TryGetNumberField(TEXT("index"), TIdx);
                        MatTex.BaseColor = OutData.ResolveTextureUri(TIdx);
                    }
                    const TSharedPtr<FJsonObject>* MRTex = nullptr;
                    if ((*PBR)->TryGetObjectField(TEXT("metallicRoughnessTexture"), MRTex))
                    {
                        int32 TIdx = -1;
                        (*MRTex)->TryGetNumberField(TEXT("index"), TIdx);
                        MatTex.MetallicRoughness = OutData.ResolveTextureUri(TIdx);
                    }
                }
                const TSharedPtr<FJsonObject>* NTex = nullptr;
                if (MatObj->TryGetObjectField(TEXT("normalTexture"), NTex))
                {
                    int32 TIdx = -1;
                    (*NTex)->TryGetNumberField(TEXT("index"), TIdx);
                    MatTex.Normal = OutData.ResolveTextureUri(TIdx);
                }
                const TSharedPtr<FJsonObject>* OTex = nullptr;
                if (MatObj->TryGetObjectField(TEXT("occlusionTexture"), OTex))
                {
                    int32 TIdx = -1;
                    (*OTex)->TryGetNumberField(TEXT("index"), TIdx);
                    MatTex.Occlusion = OutData.ResolveTextureUri(TIdx);
                }
            }
            OutData.Materials.Add(MoveTemp(MatTex));
        }

    const TArray<TSharedPtr<FJsonValue>>* MeshesArr = nullptr;
    if (Root->TryGetArrayField(TEXT("meshes"), MeshesArr) && MeshesArr->Num() > 0)
    {
        const TArray<TSharedPtr<FJsonValue>>* PrimsArr = nullptr;
        if ((*MeshesArr)[0]->AsObject().IsValid() &&
            (*MeshesArr)[0]->AsObject()->TryGetArrayField(TEXT("primitives"), PrimsArr))
        {
            for (const TSharedPtr<FJsonValue>& Prim : *PrimsArr)
            {
                int32 MatIdx = -1;
                if (Prim->AsObject().IsValid()) Prim->AsObject()->TryGetNumberField(TEXT("material"), MatIdx);
                OutData.PrimToMaterial.Add(MatIdx);
            }
        }
    }

    return true;
}
