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

#include "Utils/PathGenerator.h"
#include "Algo/Sort.h"
#include "Async/Async.h"
#include "Engine/World.h"
#include "GameFramework/Actor.h"
#include "Components/PrimitiveComponent.h"

TArray<FPathGenerator::FOrbitTarget> FPathGenerator::ClusterTargetsByProximity(
    const TArray<AActor*>& Actors, float GroupGap)
{
    struct FActorBox { AActor* Actor; FBox Box; };
    TArray<FActorBox> Items;
    Items.Reserve(Actors.Num());
    for (AActor* A : Actors)
    {
        if (!A) continue;

        TArray<UPrimitiveComponent*> Prims;
        A->GetComponents<UPrimitiveComponent>(Prims);
        FBox Box(ForceInit);
        for (UPrimitiveComponent* Prim : Prims)
        {
            if (!Prim || !Prim->IsRegistered()) continue;
            Box += Prim->CalcBounds(Prim->GetComponentTransform()).GetBox();
        }
        if (Box.IsValid)
            Items.Add({ A, Box });
    }

    const int32 N = Items.Num();
    TArray<int32> Parent;
    Parent.SetNum(N);
    for (int32 i = 0; i < N; ++i) Parent[i] = i;
    auto Find = [&Parent](int32 x)
    {
        while (Parent[x] != x) { Parent[x] = Parent[Parent[x]]; x = Parent[x]; }
        return x;
    };
    for (int32 i = 0; i < N; ++i)
        for (int32 j = i + 1; j < N; ++j)
            if (Items[i].Box.ExpandBy(GroupGap).Intersect(Items[j].Box))
                Parent[Find(i)] = Find(j);

    TArray<FOrbitTarget> Buildings;
    TMap<int32, int32> RootToBuilding;
    for (int32 i = 0; i < N; ++i)
    {
        const int32 Root = Find(i);
        if (int32* Existing = RootToBuilding.Find(Root))
        {
            FOrbitTarget& T = Buildings[*Existing];
            T.Bounds += Items[i].Box;
            T.Actors.Add(Items[i].Actor);
        }
        else
        {
            FOrbitTarget T;
            T.Bounds = Items[i].Box;
            T.Actors.Add(Items[i].Actor);
            RootToBuilding.Add(Root, Buildings.Add(MoveTemp(T)));
        }
    }
    return Buildings;
}

namespace
{
    // Region occupancy raster — a coarse 2D "is target geometry under this column" mask that the
    // region-wide oblique/nadir survey is laid out over (the "2D height map"). POD so it crosses the
    // Phase-1 (game thread) -> Phase-2 (worker thread) boundary by value.
    struct FRegionGrid
    {
        FVector2D Origin = FVector2D::ZeroVector;
        float CellSize = 100.f;
        int32 NumX = 0;
        int32 NumY = 0;
        TArray<uint8> Occupied;
        TArray<float> TopZ;   // target surface height per cell (valid where Occupied) — the DSM

        int32 CellIndex(float X, float Y) const
        {
            if (NumX <= 0 || NumY <= 0) return INDEX_NONE;
            const int32 ix = FMath::FloorToInt((X - Origin.X) / CellSize);
            const int32 iy = FMath::FloorToInt((Y - Origin.Y) / CellSize);
            if (ix < 0 || ix >= NumX || iy < 0 || iy >= NumY) return INDEX_NONE;
            return iy * NumX + ix;
        }
        bool IsOccupied(float X, float Y) const
        {
            const int32 i = CellIndex(X, Y);
            return i != INDEX_NONE && Occupied[i] != 0;
        }
        float HeightAt(float X, float Y) const
        {
            const int32 i = CellIndex(X, Y);
            return i != INDEX_NONE ? TopZ[i] : 0.f;
        }
    };

    // One downward ray per cell, penetrating non-targets, marks a cell occupied when a TARGET column
    // exists beneath it. Grid is clamped to <=128 cells/axis to bound the raycast count.
    FRegionGrid BuildRegionHeightMap(UWorld* World, const FBox& RegionBox, const TSet<AActor*>& Targets)
    {
        FRegionGrid Grid;
        const float ExtentX = FMath::Max(RegionBox.Max.X - RegionBox.Min.X, 1.f);
        const float ExtentY = FMath::Max(RegionBox.Max.Y - RegionBox.Min.Y, 1.f);
        Grid.CellSize = FMath::Clamp(FMath::Max(ExtentX, ExtentY) / 128.f, 100.f, 2000.f);
        Grid.Origin = FVector2D(RegionBox.Min.X, RegionBox.Min.Y);
        Grid.NumX = FMath::Max(1, FMath::CeilToInt(ExtentX / Grid.CellSize));
        Grid.NumY = FMath::Max(1, FMath::CeilToInt(ExtentY / Grid.CellSize));
        Grid.Occupied.SetNumZeroed(Grid.NumX * Grid.NumY);
        Grid.TopZ.SetNumZeroed(Grid.NumX * Grid.NumY);

        const float ZTop = RegionBox.Max.Z + 1000.f;
        const float ZBot = RegionBox.Min.Z - 1000.f;
        for (int32 iy = 0; iy < Grid.NumY; ++iy)
        {
            for (int32 ix = 0; ix < Grid.NumX; ++ix)
            {
                const float X = Grid.Origin.X + (ix + 0.5f) * Grid.CellSize;
                const float Y = Grid.Origin.Y + (iy + 0.5f) * Grid.CellSize;

                FCollisionQueryParams QP;
                QP.bTraceComplex = true;
                int32 MaxPen = 8;
                bool bOcc = false;
                float HitZ = 0.f;
                while (MaxPen-- > 0)
                {
                    FHitResult Hit;
                    if (!World->LineTraceSingleByChannel(
                            Hit, FVector(X, Y, ZTop), FVector(X, Y, ZBot), ECC_Visibility, QP))
                        break;
                    AActor* HA = Hit.GetActor();
                    if (!HA) break;
                    if (Targets.Contains(HA)) { bOcc = true; HitZ = Hit.ImpactPoint.Z; break; }
                    QP.AddIgnoredActor(HA);
                }
                const int32 CellIdx = iy * Grid.NumX + ix;
                Grid.Occupied[CellIdx] = bOcc ? 1 : 0;
                Grid.TopZ[CellIdx] = HitZ;
            }
        }
        return Grid;
    }
}

void FPathGenerator::GenerateConformalOrbit(const FConformalOrbitParams& Params, FOnPathGenerated OnComplete)
{
    if (!Params.World)
    {
        return;
    }

    // Camera frustum constants are building-independent.
    const float HFovRad = FMath::DegreesToRadians(FMath::Max(Params.CameraHFOV, 5.f));
    const float AspectRatio = (Params.CameraResolution.Y > 0) ? (float)Params.CameraResolution.X / Params.CameraResolution.Y : 16.f / 9.f;
    const float VFovRad = 2.f * FMath::Atan(FMath::Tan(HFovRad * 0.5f) / AspectRatio);

    // Region = union of all building clusters. The oblique/nadir survey is planned ONCE over the
    // whole region at a constant altitude above the tallest structure (commercial oblique-survey
    // style), so a short block beside a tall one is no longer skipped.
    FBox RegionBox(ForceInit);
    TSet<AActor*> AllTargets;
    for (const FOrbitTarget& B : Params.Buildings)
    {
        if (B.Bounds.IsValid) RegionBox += B.Bounds;
        for (const TWeakObjectPtr<AActor>& WA : B.Actors)
            if (AActor* A = WA.Get()) AllTargets.Add(A);
    }
    if (!RegionBox.IsValid)
    {
        return;
    }
    // 2D height/occupancy map over the region (game thread) — the nadir survey reads its surface
    // height per cell to raise each waypoint above the local geometry.
    FRegionGrid Grid = BuildRegionHeightMap(Params.World, RegionBox, AllTargets);

    // Per-building facade payload — only built when the optional side orbit is requested.
    struct FBuildingRings
    {
        TArray<TArray<FVector>> RingOrbitPoints;
        TArray<TArray<FVector>> RingSurfacePoints;
        TArray<TArray<bool>> RingValid;
        float StepH;
    };
    TArray<FBuildingRings> AllBuildings;

    const int32 NumProbes = 360;
    FCollisionQueryParams QueryParams;
    QueryParams.bTraceComplex = true;

    // Phase 1 (game thread): facade surface probing — ONLY when side orbit is enabled. Each building
    // is orbited individually so inner facades facing the gaps get their own ring of cameras; the
    // other selected buildings act as occluders that drop cameras jammed in tight gaps.
    if (Params.bSideOrbit)
    {
        AllBuildings.Reserve(Params.Buildings.Num());
        for (int32 BIdx = 0; BIdx < Params.Buildings.Num(); ++BIdx)
        {
            const FOrbitTarget& Building = Params.Buildings[BIdx];
            if (!Building.Bounds.IsValid)
            {
                continue;
            }
            TSet<AActor*> TargetActors;
            for (const TWeakObjectPtr<AActor>& WA : Building.Actors)
            {
                if (AActor* A = WA.Get())
                    TargetActors.Add(A);
            }

            const FVector BoxCenter = Building.Bounds.GetCenter();
            const FVector BoxExtent = Building.Bounds.GetExtent();
            const float BoxMinZ = Building.Bounds.Min.Z + Params.StartHeight;
            const float BoxMaxZ = Building.Bounds.Max.Z;

            const float StepH = FMath::Max(
                2.f * Params.Margin * FMath::Tan(HFovRad * 0.5f) * FMath::Max(1.f - Params.HOverlap, 0.05f), 10.f);
            const float StepV = FMath::Max(
                2.f * Params.Margin * FMath::Tan(VFovRad * 0.5f) * FMath::Max(1.f - Params.VOverlap, 0.05f), 10.f);

            // Lower the highest ring to position the building top at the 1/4 mark of the frame (reserving 1/4 for the sky).
            const float BoxMaxZ_Rings = FMath::Max(BoxMaxZ - Params.Margin * FMath::Tan(VFovRad * 0.25f), BoxMinZ + StepV);
            const float BuildingH = FMath::Max(BoxMaxZ_Rings - BoxMinZ, 0.f);
            const int32 NumRings = FMath::Max(1, FMath::CeilToInt(BuildingH / StepV));

            const float SearchRadius = (FMath::Max(BoxExtent.X, BoxExtent.Y) + Params.Margin) * 4.f + 2000.f;

            FBuildingRings Rings;
            Rings.StepH = StepH;
            Rings.RingOrbitPoints.Reserve(NumRings);
            Rings.RingSurfacePoints.Reserve(NumRings);
            Rings.RingValid.Reserve(NumRings);

            for (int32 Ring = 0; Ring < NumRings; ++Ring)
            {
                const float T = (NumRings > 1) ? (float)Ring / (NumRings - 1) : 0.5f;
                const float Z = BoxMinZ + T * BuildingH;
                const FVector CenterAtZ(BoxCenter.X, BoxCenter.Y, Z);

                TArray<float> SurfaceDistances;
                SurfaceDistances.Reserve(NumProbes);

                for (int32 a = 0; a < NumProbes; ++a)
                {
                    const float AngleRad = (2.f * PI * a) / NumProbes;
                    const FVector Dir(FMath::Cos(AngleRad), FMath::Sin(AngleRad), 0.f);
                    const FVector TraceStart = CenterAtZ + Dir * SearchRadius;

                    bool bFound = false;
                    float HitDistFromCenter = 0.f;
                    FVector HitPoint = FVector::ZeroVector;

                    // World-level raycast from outside toward the building center, penetrating
                    // every actor that is NOT this building until this building's surface is hit.
                    {
                        FCollisionQueryParams ProbeParams = QueryParams;
                        int32 MaxPenetrations = 16;

                        while (!bFound && MaxPenetrations-- > 0)
                        {
                            FHitResult Hit;
                            if (!Params.World->LineTraceSingleByChannel(
                                    Hit, TraceStart, CenterAtZ, ECC_Visibility, ProbeParams))
                                break;

                            AActor* HitActor = Hit.GetActor();
                            if (!HitActor)
                                break;

                            if (TargetActors.Contains(HitActor))
                            {
                                HitPoint = Hit.ImpactPoint;
                                bFound = true;
                            }
                            else
                            {
                                ProbeParams.AddIgnoredActor(HitActor);
                            }
                        }

                        if (bFound)
                        {
                            HitDistFromCenter = FMath::Sqrt(
                                FMath::Square(HitPoint.X - CenterAtZ.X) +
                                FMath::Square(HitPoint.Y - CenterAtZ.Y));
                        }
                    }

                    if (bFound)
                    {
                        SurfaceDistances.Add(HitDistFromCenter);
                    }
                    else
                    {
                        // Bounding box fallback based on angle: distance to the ray/AABB intersection.
                        const float tX = FMath::Abs(Dir.X) > KINDA_SMALL_NUMBER ? BoxExtent.X / FMath::Abs(Dir.X) : FLT_MAX;
                        const float tY = FMath::Abs(Dir.Y) > KINDA_SMALL_NUMBER ? BoxExtent.Y / FMath::Abs(Dir.Y) : FLT_MAX;
                        SurfaceDistances.Add(FMath::Min(tX, tY));
                    }
                }

                // Circular median filter (window 5) over the radial distances: kills single-probe
                // outliers from railings/protrusions before the orbit polygon is built.
                TArray<float> FilteredDistances;
                FilteredDistances.Reserve(NumProbes);
                for (int32 a = 0; a < NumProbes; ++a)
                {
                    float Window[5];
                    for (int32 k = -2; k <= 2; ++k)
                    {
                        Window[k + 2] = SurfaceDistances[(a + k + NumProbes) % NumProbes];
                    }
                    Algo::Sort(MakeArrayView(Window, 5));
                    FilteredDistances.Add(Window[2]);
                }

                TArray<FVector> OrbitPoints;
                TArray<FVector> SurfacePoints;
                TArray<bool> Valid;
                OrbitPoints.Reserve(NumProbes);
                SurfacePoints.Reserve(NumProbes);
                Valid.Reserve(NumProbes);
                for (int32 a = 0; a < NumProbes; ++a)
                {
                    const float AngleRad = (2.f * PI * a) / NumProbes;
                    const FVector Dir(FMath::Cos(AngleRad), FMath::Sin(AngleRad), 0.f);
                    const float D = FilteredDistances[a];
                    const FVector SurfacePt(
                        CenterAtZ.X + Dir.X * D,
                        CenterAtZ.Y + Dir.Y * D,
                        Z);

                    const float OrbitDist = D + Params.Margin;
                    const FVector OrbitPt(
                        CenterAtZ.X + Dir.X * OrbitDist,
                        CenterAtZ.Y + Dir.Y * OrbitDist,
                        Z);

                    // Validity: the facade must be visible from the camera. Trace from just off the
                    // surface out to the orbit position, ignoring THIS building; any hit means a
                    // neighbour (selected or not) or generic clutter sits in the gap between camera
                    // and wall — or the camera would embed inside it. Such a pose is dropped rather
                    // than squeezed in at a grazing angle. Phase 2 breaks the orbit arc at dropped
                    // points so the path is never interpolated through the gap.
                    bool bValid = true;
                    {
                        FCollisionQueryParams LosParams = QueryParams;
                        for (AActor* TA : TargetActors) LosParams.AddIgnoredActor(TA);
                        FHitResult LosHit;
                        if (Params.World->LineTraceSingleByChannel(
                                LosHit, SurfacePt + Dir * 2.f, OrbitPt, ECC_Visibility, LosParams))
                        {
                            bValid = false;
                        }
                    }

                    SurfacePoints.Add(SurfacePt);
                    OrbitPoints.Add(OrbitPt);
                    Valid.Add(bValid);
                }

                Rings.RingOrbitPoints.Add(MoveTemp(OrbitPoints));
                Rings.RingSurfacePoints.Add(MoveTemp(SurfacePoints));
                Rings.RingValid.Add(MoveTemp(Valid));
            }

            AllBuildings.Add(MoveTemp(Rings));
        }
    }

    // Phase 2 (background): build facade poses (optional) and the region survey grid (default);
    // Phase 3 (game thread): single callback. Only scalars and POD payloads cross the boundary.
    Async(EAsyncExecution::LargeThreadPool,
        [OnComplete,
         AllBuildings = MoveTemp(AllBuildings),
         Grid = MoveTemp(Grid),
         HFovRad, VFovRad,
         Margin = Params.Margin,
         CornerYawStepDeg = Params.CornerYawStepDeg,
         NumObliqueRings = Params.NumObliqueRings,
         bIncludeNadir = Params.bIncludeNadir,
         bSideOrbit = Params.bSideOrbit,
         NadirAltitude = Params.NadirAltitude,
         NadirTiltAngle = Params.NadirTiltAngle,
         HOverlap = Params.HOverlap,
         VOverlap = Params.VOverlap,
         RegionBox]() mutable
        {
            FGeneratedPath GeneratedPath;

            const float MaxYawStep = FMath::Max(CornerYawStepDeg, 1.f);

            // --- Optional side / facade orbit ---
            if (bSideOrbit)
            {
                for (FBuildingRings& BR : AllBuildings)
                {
                    TArray<TArray<FVector>>& AllRingOrbitPoints = BR.RingOrbitPoints;
                    TArray<TArray<FVector>>& AllRingSurfacePoints = BR.RingSurfacePoints;
                    TArray<TArray<bool>>& AllRingValid = BR.RingValid;
                    const float StepH = BR.StepH;

                    if (AllRingOrbitPoints.Num() == 0)
                    {
                        continue;
                    }

                    const TArray<FVector> TopRingPolygon = AllRingOrbitPoints.Last();
                    const TArray<FVector> TopSurfacePolygon = AllRingSurfacePoints.Last();
                    const TArray<bool> TopValidPolygon = AllRingValid.Last();

                    // Oblique rings: copies of the top ring raised so the look-at to the top-ring
                    // surface points lands between the horizontal rings (pitch 0) and the nadir
                    // pitch, closing the stereo coverage gap on roof slopes and upper facades.
                    const int32 NumOblique = FMath::Clamp(NumObliqueRings, 0, 8);
                    if (NumOblique > 0)
                    {
                        const TArray<FVector>& TopSurface = TopSurfacePolygon;
                        const float MaxPitchDeg = 90.f - FMath::Clamp(NadirTiltAngle, 10.f, 70.f);
                        for (int32 i = 0; i < NumOblique; ++i)
                        {
                            const float Blend = (float)(i + 1) / (NumOblique + 1);
                            const float PitchDeg = FMath::Lerp(20.f, FMath::Max(MaxPitchDeg, 25.f), Blend);
                            const float Raise = Margin * FMath::Tan(FMath::DegreesToRadians(PitchDeg));

                            TArray<FVector> Orbit = TopRingPolygon;
                            for (FVector& P : Orbit)
                            {
                                P.Z += Raise;
                            }
                            AllRingOrbitPoints.Add(MoveTemp(Orbit));
                            AllRingSurfacePoints.Add(TopSurface);
                            AllRingValid.Add(TopValidPolygon);
                        }
                    }

                    const int32 NumRings = AllRingOrbitPoints.Num();
                    for (int32 r = 0; r < NumRings; ++r)
                    {
                        const TArray<FVector>& OrbitPoints = AllRingOrbitPoints[r];
                        const TArray<FVector>& SurfacePoints = AllRingSurfacePoints[r];
                        const TArray<bool>& Valid = AllRingValid[r];
                        const int32 M = OrbitPoints.Num();
                        if (M < 4) continue;

                        TArray<FVector> RingPositions;
                        TArray<FRotator> RingRotations;
                        TArray<bool> BreakAfter;

                        float DistUntilNext = 0.f;
                        for (int32 i = 0; i < M; ++i)
                        {
                            // Drop segments touching an invalid (occluded / gap) orbit point and mark
                            // a break, so corner smoothing never bridges a path across the dropped span.
                            if (!Valid[i] || !Valid[(i + 1) % M])
                            {
                                if (RingPositions.Num() > 0) BreakAfter.Last() = true;
                                DistUntilNext = 0.f;
                                continue;
                            }

                            const FVector& A = OrbitPoints[i];
                            const FVector& B = OrbitPoints[(i + 1) % M];
                            const FVector& SurfA = SurfacePoints[i];
                            const FVector& SurfB = SurfacePoints[(i + 1) % M];
                            const float SegLen = FMath::Sqrt(FMath::Square(B.X - A.X) + FMath::Square(B.Y - A.Y));
                            if (SegLen < KINDA_SMALL_NUMBER) continue;

                            const FVector EdgeDir = FVector(B.X - A.X, B.Y - A.Y, 0.f) / SegLen;
                            const FRotator SegRot = FVector(-EdgeDir.Y, EdgeDir.X, 0.f).Rotation();

                            while (DistUntilNext <= SegLen)
                            {
                                const float t = DistUntilNext / SegLen;
                                const FVector Pos = FMath::Lerp(A, B, t);
                                const FVector LookTarget = FMath::Lerp(SurfA, SurfB, t);
                                const FVector LookDir = LookTarget - Pos;

                                RingPositions.Add(Pos);
                                RingRotations.Add(
                                    LookDir.SizeSquared2D() > KINDA_SMALL_NUMBER
                                        ? LookDir.Rotation()
                                        : SegRot);
                                BreakAfter.Add(false);
                                DistUntilNext += StepH;
                            }
                            DistUntilNext -= SegLen;
                        }

                        // Corner smoothing: where the view yaw turns faster than MaxYawStep between
                        // consecutive poses, insert interpolated poses so the sweep stays gradual.
                        for (int32 s = 0; s < RingPositions.Num(); ++s)
                        {
                            GeneratedPath.Positions.Add(RingPositions[s]);
                            GeneratedPath.Rotations.Add(RingRotations[s]);

                            if (s + 1 >= RingPositions.Num()) break;
                            if (BreakAfter[s]) continue;

                            const float YawDelta = FMath::Abs(FMath::FindDeltaAngleDegrees(
                                RingRotations[s].Yaw, RingRotations[s + 1].Yaw));
                            const int32 NumInserts = FMath::FloorToInt(YawDelta / MaxYawStep);
                            if (NumInserts <= 0) continue;

                            const FQuat QA = RingRotations[s].Quaternion();
                            const FQuat QB = RingRotations[s + 1].Quaternion();
                            for (int32 k = 1; k <= NumInserts; ++k)
                            {
                                const float t = (float)k / (NumInserts + 1);
                                GeneratedPath.Positions.Add(
                                    FMath::Lerp(RingPositions[s], RingPositions[s + 1], t));
                                GeneratedPath.Rotations.Add(FQuat::Slerp(QA, QB, t).Rotator());
                            }
                        }
                    }
                }
            }

            // --- Region-wide nadir survey: a single N-S boustrophedon over the height map ---
            // Strips run North-South (along X), stepped West->East (along Y), serpentine so the whole
            // survey is one continuous path. Every waypoint looks straight down at NadirAltitude above
            // the height map's local surface (the region floor where no target sits beneath). No
            // culling — the full region bounding box is swept, so nothing is left uncovered.
            if (bIncludeNadir)
            {
                const float NFootH = 2.f * NadirAltitude * FMath::Tan(HFovRad * 0.5f);
                const float NFootV = 2.f * NadirAltitude * FMath::Tan(VFovRad * 0.5f);
                const float CrossStep = FMath::Max(NFootH * FMath::Max(1.f - HOverlap, 0.05f), 10.f);
                const float AlongStep = FMath::Max(NFootV * FMath::Max(1.f - VOverlap, 0.05f), 10.f);

                const float MinX = RegionBox.Min.X, MaxX = RegionBox.Max.X;
                const float MinY = RegionBox.Min.Y, MaxY = RegionBox.Max.Y;
                const float FloorZ = RegionBox.Min.Z;

                int32 StripIdx = 0;
                for (float Y = MinY; Y <= MaxY + KINDA_SMALL_NUMBER; Y += CrossStep, ++StripIdx)
                {
                    TArray<FVector> StripPos;
                    for (float X = MinX; X <= MaxX + KINDA_SMALL_NUMBER; X += AlongStep)
                    {
                        const float SurfZ = Grid.IsOccupied(X, Y) ? Grid.HeightAt(X, Y) : FloorZ;
                        StripPos.Add(FVector(X, Y, SurfZ + NadirAltitude));
                    }

                    if (StripIdx % 2 == 1)
                    {
                        for (int32 i = StripPos.Num() - 1; i >= 0; --i)
                        {
                            GeneratedPath.Positions.Add(StripPos[i]);
                            GeneratedPath.Rotations.Add(FRotator(-90.f, 0.f, 0.f));
                        }
                    }
                    else
                    {
                        for (const FVector& P : StripPos)
                        {
                            GeneratedPath.Positions.Add(P);
                            GeneratedPath.Rotations.Add(FRotator(-90.f, 0.f, 0.f));
                        }
                    }
                }
            }

            AsyncTask(ENamedThreads::GameThread,
                [OnComplete, GeneratedPath = MoveTemp(GeneratedPath)]()
                {
                    OnComplete.ExecuteIfBound(GeneratedPath);
                });
        });
}
