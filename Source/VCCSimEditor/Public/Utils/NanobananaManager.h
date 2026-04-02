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
#include "Dom/JsonObject.h"

class UWorld;
class UMaterial;
class UMaterialInstanceConstant;
class UTexture2D;

DECLARE_DELEGATE_ThreeParams(FOnNanobananaProgress, const FString& /* Status */, int32 /* Processed */, int32 /* Total */);
DECLARE_DELEGATE_OneParam(FOnNanobananaComplete, const FString& /* FinalStatus */);

/**
 * Projects nanobanana segmentation masks onto a UStaticMeshActor via SceneCapture2D ID pass,
 * votes per material slot across all frames, creates labeled UE materials, and exports
 * mesh_labeled.gltf via GTMaterialExporter. Runs synchronously on the GameThread with
 * FScopedSlowTask progress display.
 */
class VCCSIMEDITOR_API FNanobananaManager : public TSharedFromThis<FNanobananaManager>
{
public:
    FNanobananaManager();
    ~FNanobananaManager();

    struct FProjectionParams
    {
        FString ResultDir;
        FString PosesFile;
        FString ManifestFile;
        float   HFOV              = 90.f;
        int32   ImageWidth        = 1920;
        int32   ImageHeight       = 1080;
        float   OverlayAlpha      = 0.4f;
        UWorld* World             = nullptr;
        FString SceneName;
        int32   TextureResolution = 2048;
    };

    bool RunProjection(
        const FProjectionParams& InParams,
        FOnNanobananaProgress    InOnProgress,
        FOnNanobananaComplete    InOnComplete);

    void Cancel();
    bool IsInProgress() const { return bIsInProgress; }

private:
    struct FSlotPngInfo { FString PngPath; FString PngUri; };
    struct FActorData   { FString Label; FString Dir; TArray<FSlotPngInfo> Slots; };

    void RunOnGameThread();

    static bool LoadSlotPngInfos(
        const FString&          ActorDir,
        TArray<FSlotPngInfo>&   OutSlots,
        int32&                  OutGltfPrimCount,
        int32&                  OutSkippedNoPng);

    static UMaterial*                 GetOrCreateSlotIDMaterial();
    static UMaterial*                 GetOrCreateLabeledOverlayMaterial();
    static UTexture2D*                ImportPngAsTexture(
        const FString& PngPath,
        const FString& PackagePath,
        const FString& AssetName);
    static UMaterialInstanceConstant* CreateLabeledMIC(
        UMaterial*     ParentMat,
        UTexture2D*    BaseTex,
        uint8 CR, uint8 CG, uint8 CB,
        float          OverlayAlpha,
        const FString& PackagePath,
        const FString& AssetName);

    bool              bIsInProgress = false;
    FProjectionParams Params;
    FString           OutputDir;

    FOnNanobananaProgress OnProgress;
    FOnNanobananaComplete OnComplete;
};
