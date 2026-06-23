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
    /** A vertical oriented bounding box: a rectangle in the XY plane that can yaw about the vertical
     *  axis, extruded over [MinZ, MaxZ]. Buildings are vertical prisms, so this hugs a rotated
     *  building far better than a world-axis AABB. AxisX is a unit horizontal vector; AxisY is its
     *  perpendicular; HalfXY are the half-extents along AxisX / AxisY. */
    struct FVerticalOBB
    {
        FVector   Center = FVector::ZeroVector;
        FVector2D AxisX  = FVector2D(1.f, 0.f);
        FVector2D HalfXY = FVector2D::ZeroVector;
        double    MinZ = 0.0;
        double    MaxZ = 0.0;
        bool      bValid = false;
    };

    /** One building to orbit: its world-space bounds (AABB + oriented OBB) and ALL its source
     *  actors (a building is often several static-mesh actors). Surface probing accepts
     *  a hit on ANY of these actors; they are never treated as occluders of their own
     *  building. Actors are only dereferenced on the game thread during ray-casting. */
    struct FOrbitTarget
    {
        FBox Bounds;
        FVerticalOBB OBB;
        TArray<TWeakObjectPtr<AActor>> Actors;
    };

    /** Geometric thresholds for telling buildings (orbit these) from ground/clutter (survey only),
     *  replacing the old name-based clutter filter. All distances in cm. */
    struct FBuildingDetectParams
    {
        float ConnectGap = 15.0f;            // two structure pieces merge only if their OBBs are
                                             // within this of touching — near-contact, not 1 m-near,
                                             // so gapped road clutter no longer joins a building.
        float MinBuildingHeight = 300.0f;    // cluster Z-extent to qualify as a building
        float MinBuildingFootprint = 300.0f; // min horizontal extent to qualify as a building
        // Ground = a mesh that is all three of: flat (not chunky/tall), wide (a real surface, not a
        // small prop), and low (its edge sits level with the adjacent terrain). All cm.
        float GroundFlatRatio = 0.5f;        // Z-extent / max-horizontal-extent above this => chunky
                                             // or tall (trash can, post, wall, tower) => NOT ground.
        float GroundMinFootprint = 500.0f;   // flat pieces narrower than this are small props
                                             // (flower bed, sign, bin) => NOT ground.
        float GroundMaxRise = 200.0f;        // edge surface within this of the adjacent terrain =>
                                             // ground; an elevated flat roof rises far above => structure.
    };

    /** Parameters for generating a conformal orbit path. */
    struct FConformalOrbitParams
    {
        /** Building clusters that receive facade/oblique orbit rings (geometric subset of the targets). */
        TArray<FOrbitTarget> Buildings;

        /** Every capture target (incl. ground/clutter) the region nadir survey covers. Drives the
         *  height-map occupancy + region extent independently of Buildings. */
        TArray<AActor*> SurveyTargets;
        FBox SurveyRegion = FBox(ForceInit);

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

    /** True for a ground / terrain / road mesh. Three tests, all must hold: (1) flat — not chunky or
     *  tall (rejects bins, posts, walls, towers); (2) wide — a real surface, not a small prop
     *  (rejects flower beds, signs); (3) low — its surface at each footprint EDGE is level with the
     *  terrain just outside that edge (rejects elevated flat roofs; handles big sloped tiles, which
     *  are level with their neighbours, and buildings with no ground modelled beneath them, since the
     *  terrain is read just OUTSIDE the footprint via penetrating traces). Ground bridges buildings
     *  into one cluster, so it is kept out of DetectBuildings (still surveyed). Needs World. */
    static bool IsGroundLikeActor(UWorld* World, const AActor* Actor, const FBuildingDetectParams& Params);

    /** Drop ground-like meshes, cluster the remaining structures by proximity, and return only the
     *  clusters whose merged bounds pass the building height/footprint thresholds. These are the
     *  buildings to orbit; everything else is covered by the region survey only. */
    static TArray<FOrbitTarget> DetectBuildings(
        UWorld* World, const TArray<AActor*>& Actors, const FBuildingDetectParams& Params);
};
