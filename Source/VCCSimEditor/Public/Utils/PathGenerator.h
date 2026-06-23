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

/**
 * Generates complex camera paths for scene capture.
 */
class VCCSIMEDITOR_API FPathGenerator
{
public:
    /** One building to orbit: its combined world-space bounds and ALL its source
     *  actors (a building is often several static-mesh actors). Surface probing accepts
     *  a hit on ANY of these actors; they are never treated as occluders of their own
     *  building. Actors are only dereferenced on the game thread during ray-casting. */
    struct FOrbitTarget
    {
        FBox Bounds;
        TArray<TWeakObjectPtr<AActor>> Actors;
    };

    /** Parameters for generating a conformal orbit path. */
    struct FConformalOrbitParams
    {
        TArray<FOrbitTarget> Buildings;
        UWorld* World;

        float Margin = 500.0f;
        float StartHeight = 200.0f;
        float CameraHFOV = 90.0f;
        FIntPoint CameraResolution = FIntPoint(1920, 1080);
        float HOverlap = 0.6f;
        float VOverlap = 0.6f;

        /** Max yaw change between consecutive ring poses; larger turns get interpolated poses inserted. */
        float CornerYawStepDeg = 15.0f;

        /** Extra downward-pitched rings above the top ring, filling the pitch gap
         *  between the horizontal orbit rings and the nadir grid. */
        int32 NumObliqueRings = 2;

        bool bIncludeNadir = true;
        float NadirAltitude = 500.0f;
        float NadirTiltAngle = 45.0f;

        /** Opt-in: also generate the per-building facade orbit rings (default off => region survey only). */
        bool bSideOrbit = false;
    };

    /** The output of a path generation operation. */
    struct FGeneratedPath
    {
        TArray<FVector> Positions;
        TArray<FRotator> Rotations;
    };

    /** Delegate called upon completion of asynchronous path generation. */
    DECLARE_DELEGATE_OneParam(FOnPathGenerated, const FGeneratedPath&);

    /**
     * Asynchronously generates a conformal orbit path around the specified target.
     * Performs ray-casting on the game thread, then heavy geometry processing on a background thread,
     * and finally calls the completion delegate back on the game thread.
     *
     * @param Params The parameters for path generation.
     * @param OnComplete The delegate to call when path generation is finished.
     */
    void GenerateConformalOrbit(const FConformalOrbitParams& Params, FOnPathGenerated OnComplete);

    /** Cluster actors into buildings by AABB proximity (union-find): pieces whose bounds, expanded
     *  by GroupGap cm, intersect become one building; separate buildings stay separate. Shared by
     *  path generation and the Object-Selection highlight so both group targets identically. */
    static TArray<FOrbitTarget> ClusterTargetsByProximity(const TArray<AActor*>& Actors, float GroupGap = 100.0f);
};
