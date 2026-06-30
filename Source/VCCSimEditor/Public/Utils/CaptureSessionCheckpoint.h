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
 * One planned capture window inside a dataset run: a single lighting condition captured into its own
 * directory. A batch run has one per selected lighting slot; a single (no-lighting) run has exactly
 * one window with Slot == -1.
 */
struct FCaptureWindow
{
    int32   Slot = -1;          // lighting slot index, or -1 for a single (no-lighting) capture
    float   Elevation = 0.f;    // applied sun elevation/azimuth, so resume re-applies the same lighting
    float   Azimuth = 0.f;      // even if the panel's slot values changed since
    FString DirName;            // clean filename of the capture directory (e.g. capture_<ts>_L1)
    bool    bRgbOnly = false;   // the channel mode this window was started with (RGB-only vs full GT)
};

/**
 * On-disk record of an in-progress dataset capture, persisted to <captures>/capture_session.json so a
 * capture interrupted by Stop or an editor crash can be resumed. Mirrors FCaptureReuseManifest's
 * load/save style. Its mere presence (Exists) means there is something to resume; it is cleared only
 * when the whole run finishes successfully.
 *
 * PoseKey / SceneKey are validated against the current path and scene before resuming, so a resume
 * never mixes images from a different camera path or a changed scene into the same directories.
 */
class VCCSIMEDITOR_API FCaptureSessionCheckpoint
{
public:
    /** Load <CapturesRoot>/capture_session.json (an empty, !IsValid() checkpoint if missing/unparseable). */
    static FCaptureSessionCheckpoint Load(const FString& CapturesRoot);

    /** Persist back to <CapturesRoot>/capture_session.json (atomic temp+rename). False on write failure. */
    bool Save() const;

    /** True if a checkpoint file is present on disk (a resumable run exists). */
    static bool Exists(const FString& CapturesRoot);

    /** Delete the checkpoint file (called when the whole run completes). */
    static void Clear(const FString& CapturesRoot);

    /** A loaded checkpoint is usable only if it describes at least one window. */
    bool IsValid() const { return Windows.Num() > 0; }

    /** Set the resolved channel mode for the window with the given directory name (no-op if absent). */
    void SetWindowRgbOnly(const FString& DirName, bool bInRgbOnly);

    FString BatchTimestamp;
    FString PoseKey;
    FString SceneKey;
    bool    bOutputMesh = false;
    int32   GTTextureResolution = 0;
    bool    bUseCaptureReuse = false;
    TArray<FCaptureWindow> Windows;

    // Capture task description, recorded so a resume is self-contained even after an editor crash where
    // the level (and the FlashPawn's in-memory path) was never saved: the path can be restored onto the
    // pawn from PathPositions/PathRotations, and the target/scene can be validated and logged.
    FString          SceneName;       // dataset scene name
    TArray<FString>  TargetLabels;    // enabled target actor labels (what the capture is aimed at)
    TArray<FVector>  PathPositions;   // raw FlashPawn path (PendingPositions) at capture start
    TArray<FRotator> PathRotations;   // matching rotations

    /** The captures root this checkpoint was loaded from / will save to. */
    FString CapturesRoot;
};
