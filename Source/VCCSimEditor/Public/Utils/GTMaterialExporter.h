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
class UMaterialInterface;
class UPrimitiveComponent;
class UDynamicMeshComponent;
class UWorld;

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

    /** True while a RunExport() call on this instance is still running. */
    bool IsExportInProgress() const { return bExportInProgress; }

    /** Guarded entry point: no-ops (still firing OnComplete) if this instance is already exporting,
     *  otherwise runs ExportMaterials and clears the in-progress flag when it completes. The single
     *  copy of "guard + run", shared by every caller instead of each one tracking its own flag. */
    void RunExport(
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

    /** Broader gate for the target pipeline: true if the actor has any surveyable/exportable mesh
     *  geometry — a StaticMeshComponent (mesh+materials) OR a DynamicMeshComponent with triangles
     *  (e.g. ADynamicMeshActor roofs). */
    static bool HasExportableMeshGeometry(const AActor* Actor);

    /** Build a transient (RF_Transient) UStaticMesh from a DynamicMeshComponent, mirroring its
     *  material slots, so dynamic-mesh geometry (e.g. ADynamicMeshActor roofs) can flow through the
     *  per-actor static-mesh export. Returns nullptr if the component is empty or conversion fails. */
    static UStaticMesh* BuildStaticMeshFromDynamic(UDynamicMeshComponent* DMC);

    /** True for our own capture / visualization infrastructure (FlashPawn / LookAtPath) — never a
     *  scene target. Buildings vs ground/clutter is now a geometric decision (FPathGenerator). */
    static bool IsCaptureInfraActor(const AActor* Actor);

    /**
     * Hash of everything that determines the GT export output — the enabled target
     * actors (labels, transforms, meshes, materials), texture resolution and scene name.
     * GT materials are lighting-independent, so two captures with the same signature
     * produce identical gt_materials and the export can be reused. Note: keyed on
     * material *asset paths*, not content — an in-place edit to a material asset will
     * not change the signature.
     */
    static FString ComputeSignature(
        UWorld* World,
        const TArray<FString>& SeedLabels,
        const FString& SceneName,
        int32 TextureResolution);

    /**
     * Best-effort hash of the whole visible scene that affects the per-view GT image channels
     * (BaseColor/MatProps/Normal): every mesh actor's label, transform, mesh asset path,
     * material asset paths, and instance/triangle counts. Lighting actors carry no mesh
     * component and are excluded, so the sun position never changes it. Two captures with the
     * same scene_key (and same pose_key) render identical GT channels and can share them.
     * Best-effort like ComputeSignature: keyed on asset paths, not asset content.
     */
    static FString ComputeSceneSignature(UWorld* World);

private:
    bool bExportInProgress = false;

    static bool WriteManifest(
        AActor* Actor,
        const FString& Label,
        const FString& SceneName,
        int32 TextureResolution,
        const FString& ActorDir
    );

    /**
     * Classify a material as glazing for the Glass Intrinsic Decomposition pipeline.
     * Opaque interior-cubemap windows are matched by asset name/path (window/glass/interior);
     * future physical glass is matched by Translucent blend mode or the ThinTranslucent shading
     * model. Written per material slot as "is_glass" in manifest.json so the preprocess can
     * rasterize a per-face glass mask without isolating glass as a separate actor.
     */
    static bool IsGlassMaterial(const UMaterialInterface* Mat);
};
