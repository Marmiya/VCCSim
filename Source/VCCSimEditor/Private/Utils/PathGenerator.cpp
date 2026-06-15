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

    // Per-building Phase 1→Phase 2 payload (geometry only — no UObject crosses the
    // thread boundary). TopZ is the building roof height, the nadir altitude base.
    struct FBuildingRings
    {
        TArray<TArray<FVector>> RingOrbitPoints;
        TArray<TArray<FVector>> RingSurfacePoints;
        float StepH;
        float TopZ;
    };
    TArray<FBuildingRings> AllBuildings;
    AllBuildings.Reserve(Params.Buildings.Num());

    const int32 NumProbes = 360;
    FCollisionQueryParams QueryParams;
    QueryParams.bTraceComplex = true;

    // Phase 1 (game thread): orbit EACH building individually so inner facades
    // (walls facing the gaps between buildings) get their own ring of cameras,
    // which a single combined-center orbit can never reach. The other selected
    // buildings act as occluders that push cameras off their faces in tight gaps,
    // but they never block probing of this building's own surface.
    for (int32 BIdx = 0; BIdx < Params.Buildings.Num(); ++BIdx)
    {
        const FOrbitTarget& Building = Params.Buildings[BIdx];
        if (!Building.Bounds.IsValid)
        {
            continue;
        }
        AActor* TargetActor = Building.Actor.Get();

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

        // Occluder set: every OTHER selected building. Cameras may not sit inside these.
        TSet<AActor*> Occluders;
        for (int32 k = 0; k < Params.Buildings.Num(); ++k)
        {
            if (k == BIdx) continue;
            if (AActor* OA = Params.Buildings[k].Actor.Get())
            {
                Occluders.Add(OA);
            }
        }

        const float Standoff = FMath::Min(Params.Margin * 0.5f, 100.f);
        const float MinMargin = Params.Margin * 0.25f;

        FBuildingRings Rings;
        Rings.StepH = StepH;
        Rings.TopZ = BoxMaxZ;
        Rings.RingOrbitPoints.Reserve(NumRings);
        Rings.RingSurfacePoints.Reserve(NumRings);

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

                        if (TargetActor && HitActor == TargetActor)
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
            OrbitPoints.Reserve(NumProbes);
            SurfacePoints.Reserve(NumProbes);
            for (int32 a = 0; a < NumProbes; ++a)
            {
                const float AngleRad = (2.f * PI * a) / NumProbes;
                const FVector Dir(FMath::Cos(AngleRad), FMath::Sin(AngleRad), 0.f);
                const float D = FilteredDistances[a];
                const FVector SurfacePt(
                    CenterAtZ.X + Dir.X * D,
                    CenterAtZ.Y + Dir.Y * D,
                    Z);

                float OrbitDist = D + Params.Margin;

                // Occluder standoff: if a neighbouring building lies within Margin of this
                // surface point along the outward radial, pull the camera in to its face
                // (minus a standoff) so it never embeds in the neighbour. Generic scene
                // clutter is penetrated, so only target buildings trigger a clamp.
                if (Occluders.Num() > 0)
                {
                    const FVector ClampStart = SurfacePt + Dir * 2.f;
                    const FVector OrbitPtFull = SurfacePt + Dir * Params.Margin;
                    FCollisionQueryParams ClampParams = QueryParams;
                    if (TargetActor) ClampParams.AddIgnoredActor(TargetActor);

                    int32 MaxClampPen = 8;
                    while (MaxClampPen-- > 0)
                    {
                        FHitResult CHit;
                        if (!Params.World->LineTraceSingleByChannel(
                                CHit, ClampStart, OrbitPtFull, ECC_Visibility, ClampParams))
                            break;

                        AActor* CHitActor = CHit.GetActor();
                        if (!CHitActor)
                            break;

                        if (Occluders.Contains(CHitActor))
                        {
                            const float dHit = FVector::DotProduct(CHit.ImpactPoint - SurfacePt, Dir);
                            OrbitDist = D + FMath::Max(dHit - Standoff, MinMargin);
                            break;
                        }

                        ClampParams.AddIgnoredActor(CHitActor);
                    }
                }

                SurfacePoints.Add(SurfacePt);
                OrbitPoints.Add(FVector(
                    CenterAtZ.X + Dir.X * OrbitDist,
                    CenterAtZ.Y + Dir.Y * OrbitDist,
                    Z));
            }

            Rings.RingOrbitPoints.Add(MoveTemp(OrbitPoints));
            Rings.RingSurfacePoints.Add(MoveTemp(SurfacePoints));
        }

        AllBuildings.Add(MoveTemp(Rings));
    }

    // Phase 2 (background): sample along each building's orbital polygons and
    // concatenate; Phase 3 (game thread): single callback. Only scalars and the
    // geometry payload cross the thread boundary — no UObject access here.
    Async(EAsyncExecution::LargeThreadPool,
        [OnComplete,
         AllBuildings = MoveTemp(AllBuildings),
         HFovRad, VFovRad,
         Margin = Params.Margin,
         CornerYawStepDeg = Params.CornerYawStepDeg,
         NumObliqueRings = Params.NumObliqueRings,
         bIncludeNadir = Params.bIncludeNadir,
         NadirAltitude = Params.NadirAltitude,
         NadirTiltAngle = Params.NadirTiltAngle,
         HOverlap = Params.HOverlap,
         VOverlap = Params.VOverlap]() mutable
        {
            FGeneratedPath GeneratedPath;

            const float MaxYawStep = FMath::Max(CornerYawStepDeg, 1.f);

            for (FBuildingRings& BR : AllBuildings)
            {
                TArray<TArray<FVector>>& AllRingOrbitPoints = BR.RingOrbitPoints;
                TArray<TArray<FVector>>& AllRingSurfacePoints = BR.RingSurfacePoints;
                const float StepH = BR.StepH;

                if (AllRingOrbitPoints.Num() == 0)
                {
                    continue;
                }

                const TArray<FVector> TopRingPolygon = AllRingOrbitPoints.Last();
                const TArray<FVector> TopSurfacePolygon = AllRingSurfacePoints.Last();

                // Oblique rings: copies of the top ring raised so the look-at to the
                // top-ring surface points lands between the horizontal rings (pitch 0)
                // and the nadir grid pitch, closing the stereo coverage gap on roof
                // slopes and upper facades.
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
                    }
                }

                const int32 NumRings = AllRingOrbitPoints.Num();
                for (int32 r = 0; r < NumRings; ++r)
                {
                    const TArray<FVector>& OrbitPoints = AllRingOrbitPoints[r];
                    const TArray<FVector>& SurfacePoints = AllRingSurfacePoints[r];
                    const int32 M = OrbitPoints.Num();
                    if (M < 4) continue;

                    TArray<FVector> RingPositions;
                    TArray<FRotator> RingRotations;

                    float DistUntilNext = 0.f;
                    for (int32 i = 0; i < M; ++i)
                    {
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

                if (bIncludeNadir)
                {
                    const float TiltRad = FMath::DegreesToRadians(FMath::Clamp(NadirTiltAngle, 0.f, 80.f));
                    const float SlantDist = NadirAltitude / FMath::Max(FMath::Cos(TiltRad), 0.01f);
                    const float NadirFootH = 2.f * SlantDist * FMath::Tan(HFovRad * 0.5f);
                    const float NadirFootV = 2.f * SlantDist * FMath::Tan(VFovRad * 0.5f);

                    const float CrossStep = FMath::Max(NadirFootH * FMath::Max(1.f - HOverlap, 0.05f), 10.f);
                    const float AlongStep = FMath::Max(NadirFootV * FMath::Max(1.f - VOverlap, 0.05f), 10.f);
                    const float CamPitch = -(90.f - NadirTiltAngle);
                    const float NadirZ = BR.TopZ + NadirAltitude;

                    const TArray<FVector>& TopRingPoints = TopRingPolygon;
                    float NadirMinX = FLT_MAX, NadirMaxX = -FLT_MAX;
                    float NadirMinY = FLT_MAX, NadirMaxY = -FLT_MAX;
                    for (const FVector& P : TopRingPoints)
                    {
                        NadirMinX = FMath::Min(NadirMinX, P.X);
                        NadirMaxX = FMath::Max(NadirMaxX, P.X);
                        NadirMinY = FMath::Min(NadirMinY, P.Y);
                        NadirMaxY = FMath::Max(NadirMaxY, P.Y);
                    }

                    const FVector LastOrbitPos = GeneratedPath.Positions.Num() > 0
                        ? GeneratedPath.Positions.Last()
                        : FVector((NadirMinX + NadirMaxX) * 0.5f, (NadirMinY + NadirMaxY) * 0.5f, 0.f);

                    auto IsPointInPolygon = [](const FVector& Pt, const TArray<FVector>& Polygon) -> bool
                    {
                        bool bInside = false;
                        for (int32 i = 0, j = Polygon.Num() - 1; i < Polygon.Num(); j = i++)
                        {
                            if (((Polygon[i].Y > Pt.Y) != (Polygon[j].Y > Pt.Y)) &&
                                (Pt.X < (Polygon[j].X - Polygon[i].X) * (Pt.Y - Polygon[i].Y) / (Polygon[j].Y - Polygon[i].Y) + Polygon[i].X))
                            {
                                bInside = !bInside;
                            }
                        }
                        return bInside;
                    };

                    float DistToXMin = FMath::Abs(LastOrbitPos.X - NadirMinX);
                    float DistToXMax = FMath::Abs(LastOrbitPos.X - NadirMaxX);
                    float DistToYMin = FMath::Abs(LastOrbitPos.Y - NadirMinY);
                    float DistToYMax = FMath::Abs(LastOrbitPos.Y - NadirMaxY);

                    float MinDist = FMath::Min(DistToXMin, DistToXMax, DistToYMin, DistToYMax);
                    bool bStripsAlongY = (MinDist == DistToXMin || MinDist == DistToXMax);

                    struct FGridPoint { FVector Pos; FRotator Rot; };
                    TArray<FGridPoint> GridPoints;

                    // A grid pose is kept iff its view-centre ground point — the camera
                    // position pushed FootprintAhead along the view direction — lands on
                    // the building outline, so strip-end frames never frame empty ground.
                    const float FootprintAhead = NadirAltitude * FMath::Tan(TiltRad);
                    auto FootprintOnTarget = [&](const FVector& CamPt, const FVector2D& ViewDir2D)
                    {
                        const FVector Probe(
                            CamPt.X + ViewDir2D.X * FootprintAhead,
                            CamPt.Y + ViewDir2D.Y * FootprintAhead,
                            CamPt.Z);
                        return IsPointInPolygon(Probe, TopSurfacePolygon);
                    };

                    // Cross-hatch: one pass of serpentine strips per axis, so roof
                    // surfaces get two view sweeps with orthogonal baselines.
                    auto AppendStrips = [&](bool bAlongY)
                    {
                    if (bAlongY)
                    {
                        bool bStartXMax = (DistToXMax < DistToXMin);
                        float XStart = bStartXMax ? NadirMaxX : NadirMinX;
                        float XEnd = bStartXMax ? NadirMinX : NadirMaxX;
                        float XStepDir = bStartXMax ? -1.f : 1.f;

                        bool bFirstStripGoesNegY = (LastOrbitPos.Y >= (NadirMinY + NadirMaxY) * 0.5f);

                        int32 StripIdx = 0;
                        for (float X = XStart;
                             bStartXMax ? (X >= XEnd - KINDA_SMALL_NUMBER) : (X <= XEnd + KINDA_SMALL_NUMBER);
                             X += XStepDir * CrossStep, ++StripIdx)
                        {
                            const bool bGoNegY = (StripIdx % 2 == 0) ? bFirstStripGoesNegY : !bFirstStripGoesNegY;
                            const float Yaw = bGoNegY ? -90.f : 90.f;
                            const FVector2D ViewDir(0.f, bGoNegY ? -1.f : 1.f);

                            TArray<FGridPoint> StripPoints;
                            for (float Y = NadirMinY - FootprintAhead;
                                 Y <= NadirMaxY + FootprintAhead + KINDA_SMALL_NUMBER; Y += AlongStep)
                            {
                                FVector Pt(X, Y, NadirZ);
                                if (FootprintOnTarget(Pt, ViewDir))
                                {
                                    StripPoints.Add({Pt, FRotator(CamPitch, Yaw, 0.f)});
                                }
                            }

                            if (bGoNegY)
                            {
                                for (int32 i = StripPoints.Num() - 1; i >= 0; --i)
                                {
                                    GridPoints.Add(StripPoints[i]);
                                }
                            }
                            else
                            {
                                GridPoints.Append(StripPoints);
                            }
                        }
                    }
                    else
                    {
                        bool bStartYMax = (DistToYMax < DistToYMin);
                        float YStart = bStartYMax ? NadirMaxY : NadirMinY;
                        float YEnd = bStartYMax ? NadirMinY : NadirMaxY;
                        float YStepDir = bStartYMax ? -1.f : 1.f;

                        bool bFirstStripGoesNegX = (LastOrbitPos.X >= (NadirMinX + NadirMaxX) * 0.5f);

                        int32 StripIdx = 0;
                        for (float Y = YStart;
                             bStartYMax ? (Y >= YEnd - KINDA_SMALL_NUMBER) : (Y <= YEnd + KINDA_SMALL_NUMBER);
                             Y += YStepDir * CrossStep, ++StripIdx)
                        {
                            const bool bGoNegX = (StripIdx % 2 == 0) ? bFirstStripGoesNegX : !bFirstStripGoesNegX;
                            const float Yaw = bGoNegX ? 180.f : 0.f;
                            const FVector2D ViewDir(bGoNegX ? -1.f : 1.f, 0.f);

                            TArray<FGridPoint> StripPoints;
                            for (float X = NadirMinX - FootprintAhead;
                                 X <= NadirMaxX + FootprintAhead + KINDA_SMALL_NUMBER; X += AlongStep)
                            {
                                FVector Pt(X, Y, NadirZ);
                                if (FootprintOnTarget(Pt, ViewDir))
                                {
                                    StripPoints.Add({Pt, FRotator(CamPitch, Yaw, 0.f)});
                                }
                            }

                            if (bGoNegX)
                            {
                                for (int32 i = StripPoints.Num() - 1; i >= 0; --i)
                                {
                                    GridPoints.Add(StripPoints[i]);
                                }
                            }
                            else
                            {
                                GridPoints.Append(StripPoints);
                            }
                        }
                    }
                    };

                    AppendStrips(bStripsAlongY);
                    AppendStrips(!bStripsAlongY);

                    for (const FGridPoint& GP : GridPoints)
                    {
                        GeneratedPath.Positions.Add(GP.Pos);
                        GeneratedPath.Rotations.Add(GP.Rot);
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
