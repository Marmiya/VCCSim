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

class AActor;
class UStaticMesh;

struct FGTFoliageExportEntry
{
    TWeakObjectPtr<UStaticMesh> Mesh;
    FTransform WorldTransform;
    FString    Label;
};

class VCCSIMEDITOR_API FGTMaterialExporter
{
public:
    FGTMaterialExporter();

    void ExportMaterials(
        const TArray<FString>& ActorLabels,
        const TArray<FGTFoliageExportEntry>& FoliageEntries,
        UWorld* World,
        const FString& BaseDir,
        const FString& SceneName,
        int32 TextureResolution,
        const FString& Signature,
        FSimpleDelegate OnComplete
    );

public:
    /** True if the actor owns at least one StaticMeshComponent (incl. ISM/HISM) with a mesh and materials. */
    static bool HasExportableMeshMaterials(const AActor* Actor);

    /**
     * Hash of everything that determines the GT export output — the enabled target
     * actors (labels, transforms, meshes, materials), texture resolution, scene name
     * and nearby-mesh config. GT materials are lighting-independent, so two captures
     * with the same signature produce identical gt_materials and the export can be
     * reused. Note: keyed on material *asset paths*, not content — an in-place edit to
     * a material asset will not change the signature.
     */
    static FString ComputeSignature(
        UWorld* World,
        const TArray<FString>& SeedLabels,
        const FString& SceneName,
        int32 TextureResolution,
        bool bIncludeNearby,
        float NearbyRadius,
        bool bMergeNearby);

    /**
     * Searches sibling capture_* dirs under CapturesRoot for a complete gt_materials
     * export whose manifest signature matches Signature, skipping ExcludeCaptureDirName.
     * Returns that gt_materials directory (to copy from), or empty if none qualifies.
     */
    static FString FindReusableExport(
        const FString& CapturesRoot,
        const FString& ExcludeCaptureDirName,
        const FString& Signature);

private:
    static bool WriteManifest(
        AActor* Actor,
        const FString& Label,
        const FString& SceneName,
        int32 TextureResolution,
        const FString& ActorDir
    );
};
