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
#include "DrawDebugHelpers.h"

DEFINE_LOG_CATEGORY_STATIC(LogPathGenerator, Log, All);

// ============================================================================
// CONVEX HULL AND POLYGON EXPANSION HELPERS (Copied from VCCSimPanelPathImageCapture.cpp)
// ============================================================================

namespace PathGeneratorHelpers
{
    TArray<FVector2D> ComputeConvexHull2D(TArray<FVector2D> Points)
    {
        const int32 N = Points.Num();
        if (N < 3) return Points;

        int32 Pivot = 0;
        for (int32 i = 1; i < N; i++)
        {
            if (Points[i].Y < Points[Pivot].Y ||
                (Points[i].Y == Points[Pivot].Y && Points[i].X < Points[Pivot].X))
                Pivot = i;
        }
        Points.Swap(0, Pivot);
        const FVector2D P0 = Points[0];

        Points.Sort([P0](const FVector2D& A, const FVector2D& B)
        {
            const FVector2D dA = A - P0, dB = B - P0;
            const float Cross = dA.X * dB.Y - dA.Y * dB.X;
            if (FMath::Abs(Cross) > KINDA_SMALL_NUMBER) return Cross > 0.f;
            return (dA.X * dA.X + dA.Y * dA.Y) < (dB.X * dB.X + dB.Y * dB.Y);
        });

        TArray<FVector2D> Stack;
        Stack.Reserve(N);
        for (const FVector2D& P : Points)
        {
            while (Stack.Num() >= 2)
            {
                const FVector2D O = Stack[Stack.Num() - 2];
                const FVector2D A = Stack[Stack.Num() - 1];
                if ((A.X - O.X) * (P.Y - O.Y) - (A.Y - O.Y) * (P.X - O.X) <= 0.f)
                    Stack.Pop(EAllowShrinking::No);
                else break;
            }
            Stack.Add(P);
        }
        return Stack;
    }

    TArray<FVector2D> ExpandConvexPolygon(const TArray<FVector2D>& Hull, float D)
    {
        const int32 N = Hull.Num();
        if (N < 2) return Hull;

        TArray<FVector2D> Result;
        Result.Reserve(N);
        for (int32 i = 0; i < N; i++)
        {
            const FVector2D Prev = Hull[(i - 1 + N) % N];
            const FVector2D Curr = Hull[i];
            const FVector2D Next = Hull[(i + 1) % N];

            const FVector2D E1 = (Curr - Prev).GetSafeNormal();
            const FVector2D E2 = (Next - Curr).GetSafeNormal();
            const FVector2D N1(E1.Y, -E1.X);
            const FVector2D N2(E2.Y, -E2.X);

            const FVector2D Bisector = (N1 + N2).GetSafeNormal();
            const float SinHalf = FMath::Abs(N1.X * Bisector.Y - N1.Y * Bisector.X);
            const float Scale = (SinHalf > 0.02f) ? FMath::Min(D / SinHalf, D * 10.f) : D;

            Result.Add(Curr + Bisector * Scale);
        }
        return Result;
    }
}


void FPathGenerator::GenerateConformalOrbit(const FConformalOrbitParams& Params, FOnPathGenerated OnComplete)
{
    if (!Params.World)
    {
        UE_LOG(LogPathGenerator, Error, TEXT("World is null, cannot generate path."));
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

    const float BuildingH = FMath::Max(BoxMaxZ - BoxMinZ, 0.f);
    const int32 NumRings = FMath::Max(1, FMath::CeilToInt(BuildingH / StepV));

    const float SearchRadius = (FMath::Max(BoxExtent.X, BoxExtent.Y) + Params.Margin) * 4.f + 2000.f;
    const int32 NumProbes = 360;
    FCollisionQueryParams QueryParams;
    QueryParams.bTraceComplex = true;

    // Phase 1 (game thread): cast probe rays per ring to discover building outline
    TArray<TArray<FVector2D>> AllRingHits;
    TArray<float> AllRingZs;
    AllRingHits.Reserve(NumRings);
    AllRingZs.Reserve(NumRings);

    for (int32 Ring = 0; Ring < NumRings; ++Ring)
    {
        const float T = (NumRings > 1) ? (float)Ring / (NumRings - 1) : 0.5f;
        const float Z = BoxMinZ + T * BuildingH;
        const FVector CenterAtZ(BoxCenter.X, BoxCenter.Y, Z);

        TArray<FVector2D> Hits;
        Hits.Reserve(NumProbes);

        for (int32 a = 0; a < NumProbes; ++a)
        {
            const float AngleRad = (2.f * PI * a) / NumProbes;
            const FVector Dir(FMath::Cos(AngleRad), FMath::Sin(AngleRad), 0.f);
            const FVector TraceStart = CenterAtZ + Dir * SearchRadius;

            FHitResult BestHit;
            float BestDist = FLT_MAX;
            bool bFound = false;

            for (UPrimitiveComponent* Comp : Params.TargetPrimitives)
            {
                if (!Comp || !Comp->IsRegistered()) continue;
                FHitResult Hit;
                if (Comp->LineTraceComponent(Hit, TraceStart, CenterAtZ, QueryParams))
                {
                    const float D = FVector::Dist(TraceStart, Hit.ImpactPoint);
                    if (D < BestDist) { BestDist = D; BestHit = Hit; bFound = true; }
                }
            }

            if (bFound)
            {
                Hits.Add(FVector2D(BestHit.ImpactPoint.X, BestHit.ImpactPoint.Y));
            }
        }

        if (Hits.Num() < 4)
        {
            Hits = {
                FVector2D(Params.TargetBounds.Min.X, Params.TargetBounds.Min.Y),
                FVector2D(Params.TargetBounds.Max.X, Params.TargetBounds.Min.Y),
                FVector2D(Params.TargetBounds.Max.X, Params.TargetBounds.Max.Y),
                FVector2D(Params.TargetBounds.Min.X, Params.TargetBounds.Max.Y)
            };
        }

        AllRingZs.Add(Z);
        AllRingHits.Add(MoveTemp(Hits));
    }

    UWorld* World = Params.World;

    // Phase 2 (background): hull + expand + sample; Phase 3 (game thread): call delegate
    Async(EAsyncExecution::LargeThreadPool,
        [World, Params, OnComplete,
         AllRingHits = MoveTemp(AllRingHits),
         AllRingZs = MoveTemp(AllRingZs),
         BoxCenter, StepH, HFovRad, VFovRad]() mutable
        {
            FGeneratedPath GeneratedPath;
            TArray<TArray<FVector>> DebugHulls, DebugExpandedPolygons;

            const int32 NumRings = AllRingZs.Num();
            for (int32 r = 0; r < NumRings; ++r)
            {
                const float Z = AllRingZs[r];
                TArray<FVector2D> Hull2D = PathGeneratorHelpers::ComputeConvexHull2D(AllRingHits[r]);
                TArray<FVector2D> Expanded2D = PathGeneratorHelpers::ExpandConvexPolygon(Hull2D, Params.Margin);

                TArray<FVector> Hull3D, Expanded3D;
                for(const FVector2D& P : Hull2D) Hull3D.Add(FVector(P.X, P.Y, Z));
                for(const FVector2D& P : Expanded2D) Expanded3D.Add(FVector(P.X, P.Y, Z));
                DebugHulls.Add(MoveTemp(Hull3D));
                DebugExpandedPolygons.Add(MoveTemp(Expanded3D));
                
                const int32 M = Expanded2D.Num();
                if (M < 2) continue;

                auto LookAtCenter = [&](const FVector2D& P) -> FRotator
                {
                    return (FVector(BoxCenter.X, BoxCenter.Y, BoxCenter.Z) -
                            FVector(P.X, P.Y, Z)).GetSafeNormal().Rotation();
                };

                float DistUntilNext = 0.f;
                for (int32 i = 0; i < M; ++i)
                {
                    const FVector2D A = Expanded2D[i];
                    const FVector2D B = Expanded2D[(i + 1) % M];
                    const float SegLen = FVector2D::Distance(A, B);
                    if (SegLen < KINDA_SMALL_NUMBER) continue;

                    while (DistUntilNext <= SegLen)
                    {
                        const FVector2D P = A + (B - A) * (DistUntilNext / SegLen);
                        GeneratedPath.Positions.Add(FVector(P.X, P.Y, Z));
                        GeneratedPath.Rotations.Add(LookAtCenter(P));
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
                const float FarAngle = FMath::Min(TiltRad + VFovRad * 0.5f, PI * 0.5f - 0.017f);
                const float NadirFootV = FMath::Max(
                    Params.NadirAltitude * (FMath::Tan(FarAngle) - FMath::Tan(TiltRad - VFovRad * 0.5f)), 50.f);

                const float CrossStep = FMath::Max(NadirFootH * FMath::Max(1.f - Params.HOverlap, 0.05f), 10.f);
                const float AlongStep = FMath::Max(NadirFootV * FMath::Max(1.f - Params.VOverlap, 0.05f), 10.f);
                const float CamPitch = -(90.f - Params.NadirTiltAngle);
                const float NadirZ = Params.TargetBounds.Max.Z + Params.NadirAltitude;

                const float MinX = Params.TargetBounds.Min.X - Params.Margin;
                const float MaxX = Params.TargetBounds.Max.X + Params.Margin;
                const float MinY = Params.TargetBounds.Min.Y - Params.Margin;
                const float MaxY = Params.TargetBounds.Max.Y + Params.Margin;

                int32 StripIdx = 0;
                for (float X = MinX; X <= MaxX + KINDA_SMALL_NUMBER; X += CrossStep, ++StripIdx)
                {
                    const float Yaw = (StripIdx % 2 == 0) ? 90.f : -90.f;
                    if (StripIdx % 2 == 0)
                    {
                        for (float Y = MinY; Y <= MaxY + KINDA_SMALL_NUMBER; Y += AlongStep)
                        {
                            GeneratedPath.Positions.Add(FVector(X, Y, NadirZ));
                            GeneratedPath.Rotations.Add(FRotator(CamPitch, Yaw, 0.f));
                        }
                    }
                    else
                    {
                        for (float Y = MaxY; Y >= MinY - KINDA_SMALL_NUMBER; Y -= AlongStep)
                        {
                            GeneratedPath.Positions.Add(FVector(X, Y, NadirZ));
                            GeneratedPath.Rotations.Add(FRotator(CamPitch, Yaw, 0.f));
                        }
                    }
                }
            }

            AsyncTask(ENamedThreads::GameThread, [OnComplete, GeneratedPath = MoveTemp(GeneratedPath), DebugHulls, DebugExpandedPolygons, World]()
            {
                for (const auto& Hull : DebugHulls)
                {
                    for (int32 i = 0; i < Hull.Num(); ++i)
                    {
                        DrawDebugLine(World, Hull[i], Hull[(i + 1) % Hull.Num()], FColor::Cyan, false, 30.f, 0, 10.f);
                    }
                }
                for (const auto& Expanded : DebugExpandedPolygons)
                {
                    for (int32 i = 0; i < Expanded.Num(); ++i)
                    {
                        DrawDebugLine(World, Expanded[i], Expanded[(i + 1) % Expanded.Num()], FColor::Magenta, false, 30.f, 0, 10.f);
                    }
                }
                
                OnComplete.ExecuteIfBound(GeneratedPath);
            });
        });
}
