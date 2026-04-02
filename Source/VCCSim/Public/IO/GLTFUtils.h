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
#pragma once

#include "CoreMinimal.h"

struct VCCSIM_API FGLTFMeshData
{
    TArray<FString> ImageUris;
    TArray<int32>   TextureSources;
    TArray<int32>   PrimToMaterial;

    struct FMaterialTextures
    {
        FString BaseColor;
        FString MetallicRoughness;
        FString Normal;
        FString Occlusion;
    };
    TArray<FMaterialTextures> Materials;

    FString ResolveTextureUri(int32 TexIdx) const;
};

class VCCSIM_API FGLTFUtils
{
public:
    static bool LoadMeshData(const FString& GltfPath, FGLTFMeshData& OutData);
};
