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

class AStaticMeshActor;

/**
 * Handles the logic for exporting ground truth PBR materials and mesh data from selected actors.
 */
class VCCSIMEDITOR_API FGTMaterialExporter
{
public:
    FGTMaterialExporter();

    /**
     * Asynchronously exports the materials and combined mesh for a list of specified actors.
     *
     * @param ActorsToExport The list of actor labels to export.
     * @param World The world context to find actors in.
     * @param BaseDir The root directory for the export.
     * @param SceneName The name for the scene.
     * @param TextureResolution The resolution for the exported texture atlases.
     * @param OnComplete A delegate called on the game thread when the export is finished.
     */
    void ExportMaterials(
        const TArray<FString>& ActorLabels,
        UWorld* World,
        const FString& BaseDir,
        const FString& SceneName,
        int32 TextureResolution,
        FSimpleDelegate OnComplete
    );

private:
    // Internal data structures for processing
    struct FGTRawTex;
    struct FGTMeshRaw;
    struct FGTActorBuilt;

    // Game-thread helpers to extract data from UObjects
    static void GT_ExtractRawTex(class UTexture2D* Tex, int32 Ch, FGTRawTex& Out);
    static void GT_CollectMatChannel(class UMaterialInterface* Mat, bool bRough, FGTRawTex& Out);
    static void GT_CollectBaseColor(class UMaterialInterface* Mat, FGTRawTex& Out);
    
    // Background-thread helpers for processing and file I/O
    static TArray<FColor> BG_SampleFromRaw(const FGTRawTex& R, int32 TargetSize);
    static TArray<FColor> BG_BuildORMTile(const FGTRawTex& Rough, const FGTRawTex& Metal, int32 TargetSize);
    static FString BG_BuildOBJContent(const TArray<FGTActorBuilt>& Built);
    static bool WriteAtlasPNG(const TArray<TArray<FColor>>& Tiles, int32 TileSize, int32 Cols, int32 Rows, const FString& PngPath);
    static bool WriteMTLFile(const FString& MtlPath);
};
