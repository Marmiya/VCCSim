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
#include "Async/Async.h"
#include "Components/PrimitiveComponent.h"
#include "Engine/World.h"
#include "GameFramework/Actor.h"

void FPathGenerator::GenerateConformalOrbit(const FConformalOrbitParams& Params, FOnPathGenerated OnComplete)
{
    if (!Params.World)
    {
        return;
    }

    const FVector BoxCenter = Params.TargetBounds.GetCenter();
    const FVector BoxExtent = Params.TargetBounds.GetExtent();
    const float BoxMinZ = Params.TargetBounds.Min.Z + Params.StartHeight;
    const float BoxMaxZ = Params.TargetBounds.Max.Z;

    const float HFovRad = FMath::DegreesToRadians(FMath::Max(Params.CameraHFOV, 5.f));
    const float AspectRatio = (Params.CameraResolution.Y > 0) ? (float)Params.CameraResolution.X / Params.CameraResolution.Y : 16.f / 9.f;
    const float VFovRad = 2.f * FMath::Atan(FMath::Tan(HFovRad * 0.5f) / AspectRatio);

    const float StepH = FMath::Max(
        2.f * Params.Margin * FMath::Tan(HFovRad * 0.5f) * FMath::Max(1.f - Params.HOverlap, 0.05f), 10.f);
    const float StepV = FMath::Max(
        2.f * Params.Margin * FMath::Tan(VFovRad * 0.5f) * FMath::Max(1.f - Params.VOverlap, 0.05f), 10.f);

    // Lower the highest ring to position the building top at the 1/4 mark of the frame (reserving 1/4 for the sky).
    const float BoxMaxZ_Rings = FMath::Max(BoxMaxZ - Params.Margin * FMath::Tan(VFovRad * 0.25f), BoxMinZ + StepV);
    const float BuildingH = FMath::Max(BoxMaxZ_Rings - BoxMinZ, 0.f);
    const int32 NumRings = FMath::Max(1, FMath::CeilToInt(BuildingH / StepV));

    const float SearchRadius = (FMath::Max(BoxExtent.X, BoxExtent.Y) + Params.Margin) * 4.f + 2000.f;
    const int32 NumProbes = 360;
    FCollisionQueryParams QueryParams;
    QueryParams.bTraceComplex = true;

    // Phase 1 (game thread): For each ring and each direction, cast a ray to get an orbit point (hit surface + Margin).
    TArray<TArray<FVector>> AllRingOrbitPoints;   // Camera orbit positions, in angular order, 360 per ring.
    TArray<float> AllRingZs;
    AllRingOrbitPoints.Reserve(NumRings);
    AllRingZs.Reserve(NumRings);

    // Build a fast lookup set of Actors; if the list is empty, fall back to bounding box filtering.
    TSet<AActor*> TargetActorSet;
    for (AActor* A : Params.TargetActors)
    {
        if (A) TargetActorSet.Add(A);
    }
    const bool bUseActorFilter = TargetActorSet.Num() > 0;
    const FBox FallbackBounds = Params.TargetBounds.ExpandBy(Params.Margin * 0.5f);

    for (int32 Ring = 0; Ring < NumRings; ++Ring)
    {
        const float T = (NumRings > 1) ? (float)Ring / (NumRings - 1) : 0.5f;
        const float Z = BoxMinZ + T * BuildingH;
        const FVector CenterAtZ(BoxCenter.X, BoxCenter.Y, Z);

        TArray<FVector> OrbitPoints;
        OrbitPoints.Reserve(NumProbes);

        for (int32 a = 0; a < NumProbes; ++a)
        {
            const float AngleRad = (2.f * PI * a) / NumProbes;
            const FVector Dir(FMath::Cos(AngleRad), FMath::Sin(AngleRad), 0.f);
            const FVector TraceStart = CenterAtZ + Dir * SearchRadius;

            bool bFound = false;
            float HitDistFromCenter = 0.f;
            FVector HitPoint;

            // Main method: World-level raycast, penetrating non-target Actors until a target Actor is hit.
            // LineTraceMultiByChannel stops at the first blocking hit, so we iterate, ignoring non-target Actors, 
            // and tracing multiple times until a target is hit or there are no more hits.
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

                    if (bUseActorFilter)
                    {
                        if (TargetActorSet.Contains(HitActor))
                        {
                            HitPoint = Hit.ImpactPoint;
                            bFound = true;
                        }
                        else
                        {
                            // Non-target, ignore and continue to penetrate.
                            ProbeParams.AddIgnoredActor(HitActor);
                        }
                    }
                    else
                    {
                        // When there is no Actor list, fall back to bounding box filtering.
                        if (FallbackBounds.IsInsideOrOn(Hit.ImpactPoint))
                        {
                            HitPoint = Hit.ImpactPoint;
                            bFound = true;
                        }
                        else
                        {
                            ProbeParams.AddIgnoredActor(HitActor);
                        }
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
                // Orbit point = hit point moved outwards by Margin in the same direction.
                OrbitPoints.Add(FVector(
                    CenterAtZ.X + Dir.X * (HitDistFromCenter + Params.Margin),
                    CenterAtZ.Y + Dir.Y * (HitDistFromCenter + Params.Margin),
                    Z));
            }
            else
            {
                // Bounding box fallback based on angle: find the distance to the intersection of the ray and the AABB boundary.
                const float tX = FMath::Abs(Dir.X) > KINDA_SMALL_NUMBER ? BoxExtent.X / FMath::Abs(Dir.X) : FLT_MAX;
                const float tY = FMath::Abs(Dir.Y) > KINDA_SMALL_NUMBER ? BoxExtent.Y / FMath::Abs(Dir.Y) : FLT_MAX;
                const float FallbackDist = FMath::Min(tX, tY);
                OrbitPoints.Add(FVector(
                    CenterAtZ.X + Dir.X * (FallbackDist + Params.Margin),
                    CenterAtZ.Y + Dir.Y * (FallbackDist + Params.Margin),
                    Z));
            }
        }

        AllRingZs.Add(Z);
        AllRingOrbitPoints.Add(MoveTemp(OrbitPoints));
    }

    // Phase 2 (background): Sample along the orbital polygon; Phase 3 (game thread): Callback.
    Async(EAsyncExecution::LargeThreadPool,
        [Params, OnComplete,
         AllRingOrbitPoints = MoveTemp(AllRingOrbitPoints),
         AllRingZs = MoveTemp(AllRingZs),
         BoxCenter, StepH, HFovRad, VFovRad]() mutable
        {
            FGeneratedPath GeneratedPath;

            const int32 NumRings = AllRingZs.Num();
            for (int32 r = 0; r < NumRings; ++r)
            {
                const TArray<FVector>& OrbitPoints = AllRingOrbitPoints[r];
                const int32 M = OrbitPoints.Num();
                if (M < 4) continue;

                float DistUntilNext = 0.f;
                for (int32 i = 0; i < M; ++i)
                {
                    const FVector& A = OrbitPoints[i];
                    const FVector& B = OrbitPoints[(i + 1) % M];
                    const float SegLen = FMath::Sqrt(FMath::Square(B.X - A.X) + FMath::Square(B.Y - A.Y));
                    if (SegLen < KINDA_SMALL_NUMBER) continue;

                    const FVector EdgeDir = FVector(B.X - A.X, B.Y - A.Y, 0.f) / SegLen;
                    const FRotator SegRot = FVector(-EdgeDir.Y, EdgeDir.X, 0.f).Rotation();

                    while (DistUntilNext <= SegLen)
                    {
                        const float t = DistUntilNext / SegLen;
                        GeneratedPath.Positions.Add(FMath::Lerp(A, B, t));
                        GeneratedPath.Rotations.Add(SegRot);
                        DistUntilNext += StepH;
                    }
                    DistUntilNext -= SegLen;
                }
            }

            if (Params.bIncludeNadir)
            {
                const float TiltRad = FMath::DegreesToRadians(FMath::Clamp(Params.NadirTiltAngle, 0.f, 80.f));
                const float SlantDist = Params.NadirAltitude / FMath::Max(FMath::Cos(TiltRad), 0.01f);
                const float NadirFootH = 2.f * SlantDist * FMath::Tan(HFovRad * 0.5f);
                const float NadirFootV = 2.f * SlantDist * FMath::Tan(VFovRad * 0.5f);

                const float CrossStep = FMath::Max(NadirFootH * FMath::Max(1.f - Params.HOverlap, 0.05f), 10.f);
                const float AlongStep = FMath::Max(NadirFootV * FMath::Max(1.f - Params.VOverlap, 0.05f), 10.f);
                const float CamPitch = -(90.f - Params.NadirTiltAngle);
                const float NadirZ = Params.TargetBounds.Max.Z + Params.NadirAltitude;

                const TArray<FVector>& TopRingPoints = AllRingOrbitPoints.Last();
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

                if (bStripsAlongY)
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
                        
                        TArray<FGridPoint> StripPoints;
                        for (float Y = NadirMinY; Y <= NadirMaxY + KINDA_SMALL_NUMBER; Y += AlongStep)
                        {
                            FVector Pt(X, Y, NadirZ);
                            if (IsPointInPolygon(Pt, TopRingPoints))
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
                        
                        TArray<FGridPoint> StripPoints;
                        for (float X = NadirMinX; X <= NadirMaxX + KINDA_SMALL_NUMBER; X += AlongStep)
                        {
                            FVector Pt(X, Y, NadirZ);
                            if (IsPointInPolygon(Pt, TopRingPoints))
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

                for (const FGridPoint& GP : GridPoints)
                {
                    GeneratedPath.Positions.Add(GP.Pos);
                    GeneratedPath.Rotations.Add(GP.Rot);
                }
            }

            AsyncTask(ENamedThreads::GameThread,
                [OnComplete, GeneratedPath = MoveTemp(GeneratedPath)]()
                {
                    OnComplete.ExecuteIfBound(GeneratedPath);
                });
        });
}
