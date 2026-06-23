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

namespace
{
    using FVerticalOBB = FPathGenerator::FVerticalOBB;

    // Largest registered primitive component of an actor (by local-box volume). Its world yaw sets
    // the building's footprint orientation.
    const UPrimitiveComponent* LargestComponent(const AActor* A, double& OutVolume)
    {
        const UPrimitiveComponent* Best = nullptr;
        OutVolume = -1.0;
        if (!A) return nullptr;
        TArray<UPrimitiveComponent*> Prims;
        const_cast<AActor*>(A)->GetComponents<UPrimitiveComponent>(Prims);
        for (UPrimitiveComponent* P : Prims)
        {
            if (!P || !P->IsRegistered()) continue;
            const FVector S = P->CalcBounds(FTransform::Identity).GetBox().GetSize();
            const double Vol = (double)S.X * (double)S.Y * (double)S.Z;
            if (Vol > OutVolume) { OutVolume = Vol; Best = P; }
        }
        return Best;
    }

    // Accumulate an actor's component corners, projected into the (AxisX, AxisY, Z) frame.
    void AccumulateActorFrame(const AActor* A, const FVector2D& AxisX, const FVector2D& AxisY,
        double& MinU, double& MaxU, double& MinV, double& MaxV, double& MinZ, double& MaxZ, bool& bAny)
    {
        if (!A) return;
        TArray<UPrimitiveComponent*> Prims;
        const_cast<AActor*>(A)->GetComponents<UPrimitiveComponent>(Prims);
        for (UPrimitiveComponent* P : Prims)
        {
            if (!P || !P->IsRegistered()) continue;
            const FBox LB = P->CalcBounds(FTransform::Identity).GetBox();
            if (!LB.IsValid) continue;
            const FTransform Xf = P->GetComponentTransform();
            const FVector Mn = LB.Min, Mx = LB.Max;
            const FVector LC[8] = {
                {Mn.X,Mn.Y,Mn.Z},{Mx.X,Mn.Y,Mn.Z},{Mn.X,Mx.Y,Mn.Z},{Mx.X,Mx.Y,Mn.Z},
                {Mn.X,Mn.Y,Mx.Z},{Mx.X,Mn.Y,Mx.Z},{Mn.X,Mx.Y,Mx.Z},{Mx.X,Mx.Y,Mx.Z} };
            for (const FVector& L : LC)
            {
                const FVector W = Xf.TransformPosition(L);
                const double U = W.X * AxisX.X + W.Y * AxisX.Y;
                const double V = W.X * AxisY.X + W.Y * AxisY.Y;
                MinU = FMath::Min(MinU, U); MaxU = FMath::Max(MaxU, U);
                MinV = FMath::Min(MinV, V); MaxV = FMath::Max(MaxV, V);
                MinZ = FMath::Min(MinZ, (double)W.Z); MaxZ = FMath::Max(MaxZ, (double)W.Z);
                bAny = true;
            }
        }
    }

    // Vertical OBB enclosing a set of actors, oriented to the yaw of their single largest component.
    FVerticalOBB BuildOBBForActors(const TArray<AActor*>& Members)
    {
        FVerticalOBB OBB;
        const UPrimitiveComponent* Largest = nullptr;
        double BestVol = -1.0;
        for (AActor* A : Members)
        {
            double Vol;
            if (const UPrimitiveComponent* P = LargestComponent(A, Vol))
                if (Vol > BestVol) { BestVol = Vol; Largest = P; }
        }
        if (!Largest) return OBB;

        const double Yaw = FMath::DegreesToRadians(Largest->GetComponentRotation().Yaw);
        const FVector2D AxisX(FMath::Cos(Yaw), FMath::Sin(Yaw));
        const FVector2D AxisY(-AxisX.Y, AxisX.X);

        double MinU = TNumericLimits<double>::Max(), MaxU = TNumericLimits<double>::Lowest();
        double MinV = MinU, MaxV = MaxU, MinZ = MinU, MaxZ = MaxU;
        bool bAny = false;
        for (AActor* A : Members)
            AccumulateActorFrame(A, AxisX, AxisY, MinU, MaxU, MinV, MaxV, MinZ, MaxZ, bAny);
        if (!bAny) return OBB;

        const double CU = 0.5 * (MinU + MaxU), CV = 0.5 * (MinV + MaxV);
        OBB.Center = FVector(CU * AxisX.X + CV * AxisY.X, CU * AxisX.Y + CV * AxisY.Y, 0.5 * (MinZ + MaxZ));
        OBB.AxisX = AxisX;
        OBB.HalfXY = FVector2D(0.5 * (MaxU - MinU), 0.5 * (MaxV - MinV));
        OBB.MinZ = MinZ; OBB.MaxZ = MaxZ;
        OBB.bValid = true;
        return OBB;
    }

    // Do two vertical OBBs come within Gap of touching? 2D separating-axis test on the four box
    // edge normals, plus a Z-range overlap check.
    bool OBBOverlapXY(const FVerticalOBB& A, const FVerticalOBB& B, double Gap)
    {
        if (A.MinZ > B.MaxZ + Gap || B.MinZ > A.MaxZ + Gap) return false;

        const FVector2D AY(-A.AxisX.Y, A.AxisX.X);
        const FVector2D BY(-B.AxisX.Y, B.AxisX.X);
        const FVector2D dC(B.Center.X - A.Center.X, B.Center.Y - A.Center.Y);
        const FVector2D Axes[4] = { A.AxisX, AY, B.AxisX, BY };
        for (const FVector2D& L : Axes)
        {
            const double dc = FMath::Abs(dC.X * L.X + dC.Y * L.Y);
            const double rA = FMath::Abs(A.HalfXY.X * (A.AxisX.X * L.X + A.AxisX.Y * L.Y))
                            + FMath::Abs(A.HalfXY.Y * (AY.X * L.X + AY.Y * L.Y));
            const double rB = FMath::Abs(B.HalfXY.X * (B.AxisX.X * L.X + B.AxisX.Y * L.Y))
                            + FMath::Abs(B.HalfXY.Y * (BY.X * L.X + BY.Y * L.Y));
            if (dc > rA + rB + Gap) return false;
        }
        return true;
    }
}

bool FPathGenerator::IsGroundLikeActor(UWorld* World, const AActor* Actor, const FBuildingDetectParams& Params)
{
    if (!Actor) return false;

    FVector Origin, Extent;
    Actor->GetActorBounds(false, Origin, Extent);
    const double Zext = Extent.Z * 2.0;
    const double HalfXY = FMath::Max(Extent.X, Extent.Y);
    const double MaxXY = HalfXY * 2.0;
    if (MaxXY <= KINDA_SMALL_NUMBER) return false;

    // (1) flat: chunky or tall pieces (bins, posts, walls, towers) are structures — cheap, no traces.
    if (Zext / MaxXY > Params.GroundFlatRatio) return false;
    // (2) wide: small flat props (flower beds, signs, debris) are not ground surfaces.
    if (MaxXY < Params.GroundMinFootprint) return false;
    if (!World) return false;

    // (3) low: compare the actor's own surface just INSIDE each footprint edge against the terrain
    // just OUTSIDE that edge. Sampling locally at the edges (not global Max.Z) keeps a big SLOPED
    // ground tile — level with its neighbours all the way round — from looking like a structure,
    // while an elevated flat roof shows a large step at every edge. Median over the ring.
    const double RIn = HalfXY * 0.7;
    const double ROut = HalfXY * 1.3 + 50.0;
    const double ZTop = Origin.Z + Extent.Z + 1000.0;
    const double ZBot = Origin.Z - Extent.Z - 100000.0;

    TArray<double> Rises;
    const int32 K = 8;
    for (int32 i = 0; i < K; ++i)
    {
        const double Ang = (2.0 * PI * i) / K;
        const double C = FMath::Cos(Ang), S = FMath::Sin(Ang);

        // terrain just outside the footprint (ignore A): lowest penetrating hit
        FCollisionQueryParams OutQP; OutQP.bTraceComplex = true; OutQP.AddIgnoredActor(Actor);
        TArray<FHitResult> OutHits;
        const FVector OutA(Origin.X + ROut * C, Origin.Y + ROut * S, ZTop);
        const FVector OutB(Origin.X + ROut * C, Origin.Y + ROut * S, ZBot);
        if (!World->LineTraceMultiByChannel(OutHits, OutA, OutB, ECC_Visibility, OutQP) || OutHits.Num() == 0)
            continue;
        double GroundZ = ZTop;
        for (const FHitResult& H : OutHits) GroundZ = FMath::Min(GroundZ, (double)H.ImpactPoint.Z);

        // this actor's own surface just inside the edge: highest hit that belongs to A
        FCollisionQueryParams InQP; InQP.bTraceComplex = true;
        TArray<FHitResult> InHits;
        const FVector InA(Origin.X + RIn * C, Origin.Y + RIn * S, ZTop);
        const FVector InB(Origin.X + RIn * C, Origin.Y + RIn * S, ZBot);
        World->LineTraceMultiByChannel(InHits, InA, InB, ECC_Visibility, InQP);
        bool bSurf = false;
        double SurfZ = 0.0;
        for (const FHitResult& H : InHits)
            if (H.GetActor() == Actor && (!bSurf || H.ImpactPoint.Z > SurfZ))
            {
                SurfZ = H.ImpactPoint.Z;
                bSurf = true;
            }
        if (!bSurf) continue;

        Rises.Add(SurfZ - GroundZ);
    }

    if (Rises.Num() == 0) return false;   // couldn't measure -> treat as structure, don't drop it
    Rises.Sort();
    const double MedianRise = Rises[Rises.Num() / 2];
    return MedianRise <= Params.GroundMaxRise;
}

TArray<FPathGenerator::FOrbitTarget> FPathGenerator::DetectBuildings(
    UWorld* World, const TArray<AActor*>& Actors, const FBuildingDetectParams& Params)
{
    // Drop ground (else it bridges every building into one). No per-piece size filter — connectivity
    // decides membership, so small genuine attachments (rooftop AC units, parapets) are kept.
    TArray<AActor*> Structures;
    Structures.Reserve(Actors.Num());
    for (AActor* A : Actors)
        if (A && !IsGroundLikeActor(World, A, Params))
            Structures.Add(A);

    const int32 N = Structures.Num();
    TArray<FVerticalOBB> OBBs;
    OBBs.SetNum(N);
    for (int32 i = 0; i < N; ++i)
        OBBs[i] = BuildOBBForActors(TArray<AActor*>{ Structures[i] });

    // Union-find: two pieces merge only if their oriented boxes are within ConnectGap of touching.
    // Ground is not in this graph, so it can never act as a connector between buildings.
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
            if (OBBs[i].bValid && OBBs[j].bValid && OBBOverlapXY(OBBs[i], OBBs[j], Params.ConnectGap))
                Parent[Find(i)] = Find(j);

    TMap<int32, TArray<int32>> Groups;
    for (int32 i = 0; i < N; ++i) Groups.FindOrAdd(Find(i)).Add(i);

    TArray<FOrbitTarget> Buildings;
    Buildings.Reserve(Groups.Num());
    for (const TPair<int32, TArray<int32>>& KV : Groups)
    {
        TArray<AActor*> Members;
        FOrbitTarget T;
        for (int32 Idx : KV.Value)
        {
            AActor* A = Structures[Idx];
            Members.Add(A);
            T.Actors.Add(A);
            FVector O, E;
            A->GetActorBounds(false, O, E);
            T.Bounds += FBox(O - E, O + E);
        }
        T.OBB = BuildOBBForActors(Members);

        const double H = T.OBB.bValid ? (T.OBB.MaxZ - T.OBB.MinZ)
                                      : (T.Bounds.IsValid ? T.Bounds.GetSize().Z : 0.0);
        const double Wd = T.OBB.bValid ? 2.0 * FMath::Max(T.OBB.HalfXY.X, T.OBB.HalfXY.Y)
                                       : (T.Bounds.IsValid ? FMath::Max(T.Bounds.GetSize().X, T.Bounds.GetSize().Y) : 0.0);
        if (H >= Params.MinBuildingHeight && Wd >= Params.MinBuildingFootprint)
            Buildings.Add(MoveTemp(T));
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
        float FallbackZ = 0.f;   // used only where a whole neighbourhood is empty (no geometry at all)
        TArray<uint8> Occupied;
        TArray<float> TopZ;      // top surface height per cell (valid where Occupied) — the DSM

        // Conservative surface height for a camera placed at (X,Y): the MAX DSM height over the cell
        // AND its 8 neighbours. Taking the neighbourhood max (not just the one cell) keeps a waypoint
        // above a tall feature whose cell its position merely borders, so a grid coarser than the
        // survey step can never drop a pose below nearby geometry. An entirely empty neighbourhood
        // (nothing around) falls back to FallbackZ — there is no geometry there to collide with.
        float HeightAt(float X, float Y) const
        {
            if (NumX <= 0 || NumY <= 0) return FallbackZ;
            const int32 cx = FMath::FloorToInt((X - Origin.X) / CellSize);
            const int32 cy = FMath::FloorToInt((Y - Origin.Y) / CellSize);
            float Best = -FLT_MAX;
            for (int32 dy = -1; dy <= 1; ++dy)
            {
                for (int32 dx = -1; dx <= 1; ++dx)
                {
                    const int32 ix = cx + dx, iy = cy + dy;
                    if (ix < 0 || ix >= NumX || iy < 0 || iy >= NumY) continue;
                    const int32 i = iy * NumX + ix;
                    if (Occupied[i] != 0) Best = FMath::Max(Best, TopZ[i]);
                }
            }
            return Best > -FLT_MAX ? Best : FallbackZ;
        }
    };

    // One downward ray per cell, penetrating non-targets, records the topmost TARGET surface beneath
    // the column (the DSM) and marks the cell occupied. Grid is clamped to <=256 cells/axis so it is
    // at least as fine as the survey step (a coarse map was dropping poses below geometry), while
    // still bounding the raycast count.
    FRegionGrid BuildRegionHeightMap(UWorld* World, const FBox& RegionBox, const TSet<AActor*>& Targets)
    {
        FRegionGrid Grid;
        const float ExtentX = FMath::Max(RegionBox.Max.X - RegionBox.Min.X, 1.f);
        const float ExtentY = FMath::Max(RegionBox.Max.Y - RegionBox.Min.Y, 1.f);
        Grid.CellSize = FMath::Clamp(FMath::Max(ExtentX, ExtentY) / 256.f, 50.f, 1500.f);
        Grid.Origin = FVector2D(RegionBox.Min.X, RegionBox.Min.Y);
        Grid.FallbackZ = RegionBox.Min.Z;
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

        // Fill DSM holes by bounded max-dilation. A cell whose single down-ray found no target — a gap
        // between meshes, or (the boundary case) a cell whose centre fell just past the outermost mesh
        // or inside an AABB corner off the actual surface — inherits the highest surface among its
        // occupied neighbours, propagated outward up to MaxDilation rings. This lifts a boundary pose
        // onto the adjacent real surface instead of dropping it to the global floor and clipping
        // underground. Areas with no geometry within reach stay empty and use FallbackZ (nothing to hit).
        {
            const int32 MaxDilation = 8;
            for (int32 Pass = 0; Pass < MaxDilation; ++Pass)
            {
                TArray<uint8> NextOcc = Grid.Occupied;
                TArray<float> NextTop = Grid.TopZ;
                bool bChanged = false;
                for (int32 iy = 0; iy < Grid.NumY; ++iy)
                {
                    for (int32 ix = 0; ix < Grid.NumX; ++ix)
                    {
                        const int32 i = iy * Grid.NumX + ix;
                        if (Grid.Occupied[i] != 0) continue;
                        float Best = -FLT_MAX;
                        for (int32 dy = -1; dy <= 1; ++dy)
                        {
                            for (int32 dx = -1; dx <= 1; ++dx)
                            {
                                const int32 jx = ix + dx, jy = iy + dy;
                                if (jx < 0 || jx >= Grid.NumX || jy < 0 || jy >= Grid.NumY) continue;
                                const int32 j = jy * Grid.NumX + jx;
                                if (Grid.Occupied[j] != 0) Best = FMath::Max(Best, Grid.TopZ[j]);
                            }
                        }
                        if (Best > -FLT_MAX)
                        {
                            NextTop[i] = Best;
                            NextOcc[i] = 1;
                            bChanged = true;
                        }
                    }
                }
                Grid.Occupied = MoveTemp(NextOcc);
                Grid.TopZ = MoveTemp(NextTop);
                if (!bChanged) break;
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

    // The nadir/oblique survey covers EVERY capture target (incl. ground/clutter), independently of
    // which clusters got facade orbits — so nothing in range is left uncovered. Region + occupancy
    // come from SurveyTargets/SurveyRegion; fall back to the building clusters if those are unset.
    FBox RegionBox = Params.SurveyRegion;
    const bool bComputeRegionFromTargets = !RegionBox.IsValid;
    TSet<AActor*> AllTargets;
    for (AActor* A : Params.SurveyTargets)
    {
        if (!A) continue;
        AllTargets.Add(A);
        if (bComputeRegionFromTargets)
        {
            FVector O, E;
            A->GetActorBounds(false, O, E);
            RegionBox += FBox(O - E, O + E);
        }
    }
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

            // Frame the orbit by the oriented box (tighter for rotated buildings); the radial probe
            // below still follows the real surface. Fall back to the AABB if the OBB is invalid.
            const bool bUseOBB = Building.OBB.bValid;
            const FVector BoxCenter = bUseOBB ? Building.OBB.Center : Building.Bounds.GetCenter();
            const float MaxHalfXY = bUseOBB
                ? (float)FMath::Max(Building.OBB.HalfXY.X, Building.OBB.HalfXY.Y)
                : (float)FMath::Max(Building.Bounds.GetExtent().X, Building.Bounds.GetExtent().Y);
            const FVector BoxExtent(MaxHalfXY, MaxHalfXY, 0.f);
            const float BoxMinZ = (bUseOBB ? (float)Building.OBB.MinZ : (float)Building.Bounds.Min.Z) + Params.StartHeight;
            const float BoxMaxZ = bUseOBB ? (float)Building.OBB.MaxZ : (float)Building.Bounds.Max.Z;

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
         bIncludeOblique = Params.bIncludeOblique,
         bSideOrbit = Params.bSideOrbit,
         NadirAltitude = Params.NadirAltitude,
         NadirTiltAngle = Params.NadirTiltAngle,
         HOverlap = Params.HOverlap,
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

            // --- Region-wide survey over the height map: tilt-aware oblique strips ---
            // Single lens (default): ONE pass, pitched by NadirTiltAngle off vertical (straight down only
            // when Tilt == 0). 5-lens oblique (bIncludeOblique): a straight-down nadir pass PLUS four
            // passes pitched by NadirTiltAngle, leaning +X/-X (along the strips) and +Y/-Y (across them).
            //
            // Each pass is its own N-S boustrophedon (strips along X, stepped along Y, serpentine), with
            // step sizes from the TILTED ground footprint so the requested overlap actually holds at the
            // tilt — NOT the nadir footprint:
            //   - along the lean axis: the VFOV footprint trapezoid alt*(tan(t+vf/2)-tan(t-vf/2));
            //   - across it: the HFOV footprint at the cone edge nearest vertical (its narrowest point,
            //     so the overlap is guaranteed along the whole strip).
            // BOTH directions use H-Overlap: a flat survey grid has no vertical stacking, so V-Overlap
            // (the facade ring-to-ring overlap) does not apply here.
            // At Tilt == 0 this reduces exactly to the plain nadir footprints. Every waypoint sits
            // NadirAltitude above the local surface, so AGL stays constant. No culling — full box swept.
            if (bIncludeNadir)
            {
                const float HalfHFov = HFovRad * 0.5f;
                const float HalfVFov = VFovRad * 0.5f;

                // (along-VFOV, across-HFOV) ground footprint for a camera tilted Tau rad off nadir.
                auto TiltedFootprint = [&](float Tau, float& OutAlong, float& OutAcross)
                {
                    const float ANear = FMath::Clamp(Tau - HalfVFov, -1.5f, 1.5f);
                    const float AFar  = FMath::Clamp(Tau + HalfVFov, -1.5f, 1.5f);
                    OutAlong = NadirAltitude * FMath::Max(FMath::Tan(AFar) - FMath::Tan(ANear), 0.01f);
                    const float PhiMin = (ANear <= 0.f && AFar >= 0.f)
                        ? 0.f : FMath::Min(FMath::Abs(ANear), FMath::Abs(AFar));
                    OutAcross = 2.f * NadirAltitude * FMath::Tan(HalfHFov) / FMath::Max(FMath::Cos(PhiMin), 0.05f);
                };

                const float MinX = RegionBox.Min.X, MaxX = RegionBox.Max.X;
                const float MinY = RegionBox.Min.Y, MaxY = RegionBox.Max.Y;
                const float TiltDeg = FMath::Clamp(NadirTiltAngle, 0.f, 85.f);

                // Yaw = direction the camera leans; bLeanAlongX = lean axis is X (else Y); TiltDeg per pass.
                struct FSurveyPass { float Yaw; bool bLeanAlongX; float TiltDeg; };
                TArray<FSurveyPass> Passes;
                if (bIncludeOblique)
                {
                    Passes.Add({ 0.f,   true,  0.f });        // nadir (straight down)
                    Passes.Add({ 0.f,   true,  TiltDeg });    // lean +X
                    Passes.Add({ 180.f, true,  TiltDeg });    // lean -X
                    Passes.Add({ 90.f,  false, TiltDeg });    // lean +Y
                    Passes.Add({ 270.f, false, TiltDeg });    // lean -Y
                }
                else
                {
                    Passes.Add({ 0.f, true, TiltDeg });        // single (tilted) lens
                }

                for (const FSurveyPass& Pass : Passes)
                {
                    float FAlong, FAcross;
                    TiltedFootprint(FMath::DegreesToRadians(Pass.TiltDeg), FAlong, FAcross);
                    const float LeanStep = FMath::Max(FAlong  * FMath::Max(1.f - HOverlap, 0.05f), 10.f);
                    const float PerpStep = FMath::Max(FAcross * FMath::Max(1.f - HOverlap, 0.05f), 10.f);
                    const float StepX = Pass.bLeanAlongX ? LeanStep : PerpStep;
                    const float StepY = Pass.bLeanAlongX ? PerpStep : LeanStep;
                    const FRotator PoseRot(-(90.f - Pass.TiltDeg), Pass.Yaw, 0.f);

                    int32 StripIdx = 0;
                    for (float Y = MinY; Y <= MaxY + KINDA_SMALL_NUMBER; Y += StepY, ++StripIdx)
                    {
                        TArray<FVector> StripPos;
                        for (float X = MinX; X <= MaxX + KINDA_SMALL_NUMBER; X += StepX)
                        {
                            const float SurfZ = Grid.HeightAt(X, Y);
                            StripPos.Add(FVector(X, Y, SurfZ + NadirAltitude));
                        }

                        if (StripIdx % 2 == 1)
                        {
                            for (int32 i = StripPos.Num() - 1; i >= 0; --i)
                            {
                                GeneratedPath.Positions.Add(StripPos[i]);
                                GeneratedPath.Rotations.Add(PoseRot);
                            }
                        }
                        else
                        {
                            for (const FVector& P : StripPos)
                            {
                                GeneratedPath.Positions.Add(P);
                                GeneratedPath.Rotations.Add(PoseRot);
                            }
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
