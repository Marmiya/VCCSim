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

/**
 * One capture's reuse record in <captures>/reuse.json. A capture "owns" a channel group when its
 * Owner field is empty (the files live in its own directory); otherwise the group is shared from
 * the named owner capture (the files are NOT in this directory — a Python resolve step hardlinks
 * them in before preprocessing).
 *
 *   view_gt      = BaseColor_*.png + MatProps_*.png + Normal_*.exr, keyed on (pose_key, scene_key)
 *   gt_materials = gt_materials/ directory,                        keyed on gt_materials_key
 */
struct FCaptureReuseEntry
{
    FString PoseKey;
    FString SceneKey;
    FString GtMaterialsKey;
    FString ViewGtOwner;        // empty => this capture owns BaseColor/MatProps/Normal
    FString GtMaterialsOwner;   // empty => this capture owns gt_materials/
};

/**
 * Read/query/write helper for <captures>/reuse.json — the C++/Python contract that lets multiple
 * captures (e.g. the same path under different lighting) share their lighting-independent GT
 * channels instead of re-rendering and re-storing them. C++ records owner references here; the
 * Python resolve step materialises the shares as hardlinks.
 */
class VCCSIMEDITOR_API FCaptureReuseManifest
{
public:
    /** Load <CapturesRoot>/reuse.json (empty manifest if missing or unparseable). */
    static FCaptureReuseManifest Load(const FString& CapturesRoot);

    /** Persist back to <CapturesRoot>/reuse.json. Returns false on write failure. */
    bool Save() const;

    /** Owner capture name whose (pose_key, scene_key) match and which physically owns the GT
     *  image channels and still exists on disk, or empty if none qualifies. */
    FString FindViewGtOwner(const FString& PoseKey, const FString& SceneKey) const;

    /** Owner capture name whose gt_materials_key matches and which physically owns a gt_materials/
     *  directory that still exists, or empty if none qualifies. */
    FString FindGtMaterialsOwner(const FString& GtMaterialsKey) const;

    void AddOrUpdate(const FString& CaptureName, const FCaptureReuseEntry& Entry);
    void Remove(const FString& CaptureName);

private:
    FString CapturesRoot;
    TMap<FString, FCaptureReuseEntry> Entries;
};
