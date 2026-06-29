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
#include "Async/ParallelFor.h"
#include "Engine/World.h"
#include "GameFramework/Actor.h"
#include "Components/PrimitiveComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "StaticMeshResources.h"

namespace
{
    using FVerticalOBB = FPathGenerator::FVerticalOBB;

    // 2D convex hull of a footprint corner cloud — defined further down, forward-declared so the OBB
    // builders below can use it.
    TArray<FVector2D> BuildFootprintHull(TArray<FVector2D> Pts);

    // Minimum-area enclosing rectangle of a 2D footprint (its true principal direction), extruded over
    // [MinZ,MaxZ]. By the Freeman–Shapira theorem the min-area rectangle of a convex polygon has one
    // side collinear with a hull edge, so testing every hull-edge orientation and keeping the smallest
    // box finds the optimum. This hugs a rotated building tightly regardless of how its meshes were
    // authored — unlike orienting to a single component's yaw, which a mis-modelled roof/podium slab
    // can rotate enough that the box balloons across a street and false-merges two buildings.
    FVerticalOBB MinAreaOBB(const TArray<FVector2D>& Pts, double MinZ, double MaxZ)
    {
        FVerticalOBB OBB;
        if (Pts.Num() == 0) return OBB;

        const TArray<FVector2D> Hull = BuildFootprintHull(Pts);
        if (Hull.Num() < 3)
        {
            // Collinear / too few points: fall back to a world-axis box over the raw points.
            double MinX = TNumericLimits<double>::Max(), MaxX = TNumericLimits<double>::Lowest();
            double MinY = MinX, MaxY = MaxX;
            for (const FVector2D& P : Pts)
            {
                MinX = FMath::Min(MinX, (double)P.X); MaxX = FMath::Max(MaxX, (double)P.X);
                MinY = FMath::Min(MinY, (double)P.Y); MaxY = FMath::Max(MaxY, (double)P.Y);
            }
            OBB.Center = FVector(0.5 * (MinX + MaxX), 0.5 * (MinY + MaxY), 0.5 * (MinZ + MaxZ));
            OBB.AxisX = FVector2D(1.f, 0.f);
            OBB.HalfXY = FVector2D(0.5 * (MaxX - MinX), 0.5 * (MaxY - MinY));
            OBB.MinZ = MinZ; OBB.MaxZ = MaxZ;
            OBB.bValid = true;
            return OBB;
        }

        const int32 H = Hull.Num();
        double BestArea = TNumericLimits<double>::Max();
        for (int32 i = 0; i < H; ++i)
        {
            const FVector2D Edge = (Hull[(i + 1) % H] - Hull[i]).GetSafeNormal();
            if (Edge.IsNearlyZero()) continue;
            const FVector2D Perp(-Edge.Y, Edge.X);

            double MinU = TNumericLimits<double>::Max(), MaxU = TNumericLimits<double>::Lowest();
            double MinV = MinU, MaxV = MaxU;
            for (const FVector2D& P : Hull)
            {
                const double U = P.X * Edge.X + P.Y * Edge.Y;
                const double V = P.X * Perp.X + P.Y * Perp.Y;
                MinU = FMath::Min(MinU, U); MaxU = FMath::Max(MaxU, U);
                MinV = FMath::Min(MinV, V); MaxV = FMath::Max(MaxV, V);
            }
            const double Area = (MaxU - MinU) * (MaxV - MinV);
            if (Area < BestArea)
            {
                BestArea = Area;
                const double CU = 0.5 * (MinU + MaxU), CV = 0.5 * (MinV + MaxV);
                OBB.Center = FVector(CU * Edge.X + CV * Perp.X, CU * Edge.Y + CV * Perp.Y, 0.5 * (MinZ + MaxZ));
                OBB.AxisX = Edge;
                OBB.HalfXY = FVector2D(0.5 * (MaxU - MinU), 0.5 * (MaxV - MinV));
                OBB.MinZ = MinZ; OBB.MaxZ = MaxZ;
                OBB.bValid = true;
            }
        }
        return OBB;
    }

    // World-space XY corners (and accumulated Z range) of one primitive component's local box.
    void CollectComponentCornersXY(const UPrimitiveComponent* P, TArray<FVector2D>& OutXY, double& MinZ, double& MaxZ)
    {
        if (!P) return;
        const FBox LB = P->CalcBounds(FTransform::Identity).GetBox();
        if (!LB.IsValid) return;
        const FTransform Xf = P->GetComponentTransform();
        const FVector Mn = LB.Min, Mx = LB.Max;
        const FVector LC[8] = {
            {Mn.X,Mn.Y,Mn.Z},{Mx.X,Mn.Y,Mn.Z},{Mn.X,Mx.Y,Mn.Z},{Mx.X,Mx.Y,Mn.Z},
            {Mn.X,Mn.Y,Mx.Z},{Mx.X,Mn.Y,Mx.Z},{Mn.X,Mx.Y,Mx.Z},{Mx.X,Mx.Y,Mx.Z} };
        for (const FVector& L : LC)
        {
            const FVector W = Xf.TransformPosition(L);
            OutXY.Add(FVector2D(W.X, W.Y));
            MinZ = FMath::Min(MinZ, (double)W.Z);
            MaxZ = FMath::Max(MaxZ, (double)W.Z);
        }
    }

    // Vertical OBB enclosing a set of actors, oriented to the minimum-area footprint direction.
    FVerticalOBB BuildOBBForActors(const TArray<AActor*>& Members)
    {
        TArray<FVector2D> XY;
        double MinZ = TNumericLimits<double>::Max(), MaxZ = TNumericLimits<double>::Lowest();
        for (AActor* A : Members)
        {
            if (!A) continue;
            TArray<UPrimitiveComponent*> Prims;
            A->GetComponents<UPrimitiveComponent>(Prims);
            for (UPrimitiveComponent* P : Prims)
                if (P && P->IsRegistered())
                    CollectComponentCornersXY(P, XY, MinZ, MaxZ);
        }
        return MinAreaOBB(XY, MinZ, MaxZ);
    }

    // Vertical OBB of a SINGLE primitive component, oriented to its minimum-area footprint direction.
    // Used to test orbit cameras against each neighbour component's box rather than a whole-building box.
    FVerticalOBB BuildOBBForComponent(const UPrimitiveComponent* P)
    {
        if (!P) return FVerticalOBB();
        TArray<FVector2D> XY;
        double MinZ = TNumericLimits<double>::Max(), MaxZ = TNumericLimits<double>::Lowest();
        CollectComponentCornersXY(P, XY, MinZ, MaxZ);
        return MinAreaOBB(XY, MinZ, MaxZ);
    }

    // 2D convex hull (Andrew's monotone chain), returned in counter-clockwise order. Used to turn a
    // building's cross-section corner cloud at a given height into a stable footprint polygon.
    TArray<FVector2D> BuildFootprintHull(TArray<FVector2D> Pts)
    {
        TArray<FVector2D> Hull;
        if (Pts.Num() < 3)
        {
            return Hull;
        }
        Pts.Sort([](const FVector2D& A, const FVector2D& B)
        {
            return A.X < B.X || (A.X == B.X && A.Y < B.Y);
        });
        auto Cross = [](const FVector2D& O, const FVector2D& A, const FVector2D& B)
        {
            return (A.X - O.X) * (B.Y - O.Y) - (A.Y - O.Y) * (B.X - O.X);
        };
        const int32 N = Pts.Num();
        TArray<FVector2D> H;
        H.SetNum(2 * N);
        int32 k = 0;
        for (int32 i = 0; i < N; ++i)
        {
            while (k >= 2 && Cross(H[k - 2], H[k - 1], Pts[i]) <= 0.0) --k;
            H[k++] = Pts[i];
        }
        for (int32 i = N - 2, t = k + 1; i >= 0; --i)
        {
            while (k >= t && Cross(H[k - 2], H[k - 1], Pts[i]) <= 0.0) --k;
            H[k++] = Pts[i];
        }
        H.SetNum(FMath::Max(k - 1, 0));
        return H;
    }

    // Is P inside the convex polygon Poly? Consistent-sign-of-cross-product test; winding-agnostic.
    bool PointInConvexPoly(const TArray<FVector2D>& Poly, const FVector2D& P)
    {
        const int32 N = Poly.Num();
        if (N < 3) return false;
        bool bPos = false, bNeg = false;
        for (int32 i = 0; i < N; ++i)
        {
            const FVector2D& A = Poly[i];
            const FVector2D& B = Poly[(i + 1) % N];
            const double Cr = (B.X - A.X) * (P.Y - A.Y) - (B.Y - A.Y) * (P.X - A.X);
            if (Cr > 0.0) bPos = true; else if (Cr < 0.0) bNeg = true;
            if (bPos && bNeg) return false;
        }
        return true;
    }

    // Is P inside the (possibly concave) polygon Poly? Crossing-number ray test; winding-agnostic.
    bool PointInPolygon2D(const TArray<FVector2D>& Poly, const FVector2D& P)
    {
        const int32 N = Poly.Num();
        if (N < 3) return false;
        bool bIn = false;
        for (int32 i = 0, j = N - 1; i < N; j = i++)
        {
            const FVector2D& A = Poly[i];
            const FVector2D& B = Poly[j];
            if (((A.Y > P.Y) != (B.Y > P.Y)) &&
                (P.X < (B.X - A.X) * (P.Y - A.Y) / (B.Y - A.Y) + A.X))
                bIn = !bIn;
        }
        return bIn;
    }

    // Exact 1D squared-distance transform (Felzenszwalb & Huttenlocher): D[q] = min_p (F[p] + (q-p)^2),
    // with Arg[q] = the argmin p. The 2D builder below runs this along columns then rows to get an exact
    // Euclidean distance field plus, via Arg, the nearest occupied cell (feature transform).
    void DistanceTransform1D(const TArray<double>& F, TArray<double>& D, TArray<int32>& Arg)
    {
        const int32 N = F.Num();
        D.SetNumUninitialized(N);
        Arg.SetNumUninitialized(N);
        if (N == 0) return;
        const double BIG = 1.0e19;
        TArray<int32> V; V.SetNumUninitialized(N);
        TArray<double> Z; Z.SetNumUninitialized(N + 1);
        int32 K = 0;
        V[0] = 0; Z[0] = -BIG; Z[1] = BIG;
        for (int32 Q = 1; Q < N; ++Q)
        {
            double S = ((F[Q] + (double)Q * Q) - (F[V[K]] + (double)V[K] * V[K])) / (2.0 * Q - 2.0 * V[K]);
            while (S <= Z[K])
            {
                --K;
                S = ((F[Q] + (double)Q * Q) - (F[V[K]] + (double)V[K] * V[K])) / (2.0 * Q - 2.0 * V[K]);
            }
            ++K; V[K] = Q; Z[K] = S; Z[K + 1] = BIG;
        }
        K = 0;
        for (int32 Q = 0; Q < N; ++Q)
        {
            while (Z[K + 1] < (double)Q) ++K;
            const int32 P = V[K];
            D[Q] = (double)(Q - P) * (Q - P) + F[P];
            Arg[Q] = P;
        }
    }

    // Concave-aware standoff rings around a footprint, at constant distance Margin. The footprint is
    // rasterised into an occupancy grid from the component rectangles spanning this height (filled) PLUS
    // the building's mesh vertices in the height band (point hits; the distance field bridges their
    // sparsity). An exact Euclidean distance transform then yields the Margin iso-contour via marching
    // squares — which hugs L / U / concave plans, enters their inner corners, and rounds smoothly over any
    // notch narrower than 2*Margin (a camera at standoff Margin physically cannot enter it). Each ordered
    // loop is resampled to ResampleStep; Feet carry the nearest footprint point per ring point (the
    // look-at target, so the downstream camera faces the wall squarely).
    struct FStandoffContours
    {
        TArray<TArray<FVector2D>> Loops;
        TArray<TArray<FVector2D>> Feet;
    };

    FStandoffContours BuildStandoffContours(
        const TArray<TArray<FVector2D>>& CompQuads, const TArray<FVector2D>& VertsXY,
        double Margin, double ResampleStep)
    {
        FStandoffContours Out;

        // Source bounds.
        double MinX = TNumericLimits<double>::Max(), MaxX = TNumericLimits<double>::Lowest();
        double MinY = MinX, MaxY = MaxX;
        auto Acc = [&](const FVector2D& P)
        {
            MinX = FMath::Min(MinX, P.X); MaxX = FMath::Max(MaxX, P.X);
            MinY = FMath::Min(MinY, P.Y); MaxY = FMath::Max(MaxY, P.Y);
        };
        for (const TArray<FVector2D>& Q : CompQuads) for (const FVector2D& P : Q) Acc(P);
        for (const FVector2D& P : VertsXY) Acc(P);
        if (MaxX < MinX) return Out;   // no sources

        double SpanX = MaxX - MinX, SpanY = MaxY - MinY;
        double Cell = FMath::Clamp(FMath::Max(SpanX, SpanY) / 200.0, 25.0, FMath::Max(Margin * 0.5, 25.0));
        const double Halo = Margin + 2.0 * Cell;
        MinX -= Halo; MinY -= Halo; MaxX += Halo; MaxY += Halo;
        SpanX = MaxX - MinX; SpanY = MaxY - MinY;
        int32 NumX = FMath::CeilToInt(SpanX / Cell);
        int32 NumY = FMath::CeilToInt(SpanY / Cell);
        if (NumX > 256 || NumY > 256)
        {
            Cell = FMath::Max(SpanX / 256.0, SpanY / 256.0);
            NumX = FMath::CeilToInt(SpanX / Cell);
            NumY = FMath::CeilToInt(SpanY / Cell);
        }
        NumX = FMath::Max(NumX, 1); NumY = FMath::Max(NumY, 1);

        const FVector2D Origin(MinX, MinY);
        auto CellCenter = [&](int32 ix, int32 iy)
        {
            return FVector2D(Origin.X + (ix + 0.5) * Cell, Origin.Y + (iy + 0.5) * Cell);
        };

        // Occupancy: fill cells inside any component rectangle, mark cells holding a mesh vertex.
        TArray<uint8> Occ; Occ.SetNumZeroed(NumX * NumY);
        int32 NumOcc = 0;
        for (const TArray<FVector2D>& Quad : CompQuads)
        {
            double QMinX = TNumericLimits<double>::Max(), QMaxX = TNumericLimits<double>::Lowest();
            double QMinY = QMinX, QMaxY = QMaxX;
            for (const FVector2D& P : Quad)
            {
                QMinX = FMath::Min(QMinX, P.X); QMaxX = FMath::Max(QMaxX, P.X);
                QMinY = FMath::Min(QMinY, P.Y); QMaxY = FMath::Max(QMaxY, P.Y);
            }
            const int32 Lx = FMath::Clamp(FMath::FloorToInt((QMinX - Origin.X) / Cell), 0, NumX - 1);
            const int32 Hx = FMath::Clamp(FMath::FloorToInt((QMaxX - Origin.X) / Cell), 0, NumX - 1);
            const int32 Ly = FMath::Clamp(FMath::FloorToInt((QMinY - Origin.Y) / Cell), 0, NumY - 1);
            const int32 Hy = FMath::Clamp(FMath::FloorToInt((QMaxY - Origin.Y) / Cell), 0, NumY - 1);
            for (int32 iy = Ly; iy <= Hy; ++iy)
                for (int32 ix = Lx; ix <= Hx; ++ix)
                {
                    const int32 Idx = iy * NumX + ix;
                    if (Occ[Idx]) continue;
                    if (PointInConvexPoly(Quad, CellCenter(ix, iy))) { Occ[Idx] = 1; ++NumOcc; }
                }
        }
        for (const FVector2D& P : VertsXY)
        {
            const int32 ix = FMath::FloorToInt((P.X - Origin.X) / Cell);
            const int32 iy = FMath::FloorToInt((P.Y - Origin.Y) / Cell);
            if (ix < 0 || ix >= NumX || iy < 0 || iy >= NumY) continue;
            const int32 Idx = iy * NumX + ix;
            if (!Occ[Idx]) { Occ[Idx] = 1; ++NumOcc; }
        }
        if (NumOcc == 0) return Out;

        // Fill enclosed voids before measuring distance: only the footprint's OUTER boundary should spawn
        // an orbit ring. A hollow shell (walls modelled as separate meshes around an empty interior) or
        // stray interior meshes otherwise leave interior cells farther than Margin from any wall, which
        // would trace a spurious ring INSIDE the building — and an interior camera has clear line-of-sight
        // to the inner wall, so the LOS cull cannot drop it. Flood-fill the exterior empty cells from the
        // grid border (the Margin halo guarantees the border is outside); any empty cell the flood cannot
        // reach is enclosed, so mark it occupied. An L / U notch stays open because the exterior flood
        // reaches into it from outside.
        {
            TArray<uint8> Reach; Reach.SetNumZeroed(NumX * NumY);
            TArray<int32> Stack;
            auto PushIf = [&](int32 ix, int32 iy)
            {
                if (ix < 0 || ix >= NumX || iy < 0 || iy >= NumY) return;
                const int32 Idx = iy * NumX + ix;
                if (Occ[Idx] || Reach[Idx]) return;
                Reach[Idx] = 1; Stack.Add(Idx);
            };
            for (int32 ix = 0; ix < NumX; ++ix) { PushIf(ix, 0); PushIf(ix, NumY - 1); }
            for (int32 iy = 0; iy < NumY; ++iy) { PushIf(0, iy); PushIf(NumX - 1, iy); }
            for (int32 Head = 0; Head < Stack.Num(); ++Head)
            {
                const int32 Idx = Stack[Head];
                const int32 ix = Idx % NumX, iy = Idx / NumX;
                PushIf(ix + 1, iy); PushIf(ix - 1, iy);
                PushIf(ix, iy + 1); PushIf(ix, iy - 1);
            }
            for (int32 i = 0; i < NumX * NumY; ++i)
                if (!Occ[i] && !Reach[i]) { Occ[i] = 1; ++NumOcc; }
        }

        // Exact Euclidean distance transform (squared, cell units) + feature (nearest occupied cell).
        const double BIG = 1.0e19;
        TArray<double> G; G.SetNumUninitialized(NumX * NumY);
        for (int32 i = 0; i < NumX * NumY; ++i) G[i] = Occ[i] ? 0.0 : BIG;

        TArray<double> ColD; ColD.SetNumUninitialized(NumX * NumY);
        TArray<int32> ColArgY; ColArgY.SetNumUninitialized(NumX * NumY);
        {
            TArray<double> Fcol, Dcol; TArray<int32> Acol;
            Fcol.SetNumUninitialized(NumY);
            for (int32 ix = 0; ix < NumX; ++ix)
            {
                for (int32 iy = 0; iy < NumY; ++iy) Fcol[iy] = G[iy * NumX + ix];
                DistanceTransform1D(Fcol, Dcol, Acol);
                for (int32 iy = 0; iy < NumY; ++iy)
                {
                    ColD[iy * NumX + ix] = Dcol[iy];
                    ColArgY[iy * NumX + ix] = Acol[iy];   // nearest occupied row in this column
                }
            }
        }

        TArray<double> Dist2; Dist2.SetNumUninitialized(NumX * NumY);
        TArray<int32> Feature; Feature.SetNumUninitialized(NumX * NumY);   // nearest occupied cell, linear
        {
            TArray<double> Frow, Drow; TArray<int32> Arow;
            Frow.SetNumUninitialized(NumX);
            for (int32 iy = 0; iy < NumY; ++iy)
            {
                for (int32 ix = 0; ix < NumX; ++ix) Frow[ix] = ColD[iy * NumX + ix];
                DistanceTransform1D(Frow, Drow, Arow);
                for (int32 ix = 0; ix < NumX; ++ix)
                {
                    const int32 BestX = Arow[ix];
                    const int32 BestY = ColArgY[iy * NumX + BestX];
                    Dist2[iy * NumX + ix] = Drow[ix];
                    Feature[iy * NumX + ix] = BestY * NumX + BestX;
                }
            }
        }

        // Scalar field S = worldDist - Margin (zero at the standoff ring), sampled at cell centres.
        TArray<double> S; S.SetNumUninitialized(NumX * NumY);
        for (int32 i = 0; i < NumX * NumY; ++i) S[i] = FMath::Sqrt(Dist2[i]) * Cell - Margin;

        auto FeatureCenter = [&](int32 LinIdx)
        {
            const int32 f = Feature[LinIdx];
            return CellCenter(f % NumX, f / NumX);
        };

        // Marching squares over the cell-centre lattice. Each crossing carries its foot (the feature of
        // the corner inside the dilated region — the one nearer the wall).
        struct FMSPt { FVector2D P; FVector2D Foot; };
        struct FSeg { FMSPt A; FMSPt B; };
        TArray<FSeg> Segs;

        auto Crossing = [&](int32 ixa, int32 iya, int32 ixb, int32 iyb) -> FMSPt
        {
            const int32 ia = iya * NumX + ixa, ib = iyb * NumX + ixb;
            const double Sa = S[ia], Sb = S[ib];
            const double Denom = (Sa - Sb);
            const double t = FMath::Abs(Denom) > KINDA_SMALL_NUMBER ? Sa / Denom : 0.5;
            const FVector2D Pa = CellCenter(ixa, iya), Pb = CellCenter(ixb, iyb);
            FMSPt Pt;
            Pt.P = FMath::Lerp(Pa, Pb, FMath::Clamp(t, 0.0, 1.0));
            Pt.Foot = FeatureCenter(Sa < Sb ? ia : ib);   // inside (more negative S) corner -> nearer wall
            return Pt;
        };

        for (int32 iy = 0; iy < NumY - 1; ++iy)
        {
            for (int32 ix = 0; ix < NumX - 1; ++ix)
            {
                // Corners: BL(ix,iy) BR(ix+1,iy) TR(ix+1,iy+1) TL(ix,iy+1). Inside = S < 0.
                const double SBL = S[iy * NumX + ix];
                const double SBR = S[iy * NumX + (ix + 1)];
                const double STR = S[(iy + 1) * NumX + (ix + 1)];
                const double STL = S[(iy + 1) * NumX + ix];
                int32 Code = 0;
                if (SBL < 0.0) Code |= 1;
                if (SBR < 0.0) Code |= 2;
                if (STR < 0.0) Code |= 4;
                if (STL < 0.0) Code |= 8;
                if (Code == 0 || Code == 15) continue;

                // Edge crossings: B=bottom(BL-BR) R=right(BR-TR) T=top(TL-TR) L=left(BL-TL).
                auto EB = [&]() { return Crossing(ix, iy, ix + 1, iy); };
                auto ER = [&]() { return Crossing(ix + 1, iy, ix + 1, iy + 1); };
                auto ET = [&]() { return Crossing(ix, iy + 1, ix + 1, iy + 1); };
                auto EL = [&]() { return Crossing(ix, iy, ix, iy + 1); };
                auto Add = [&](const FMSPt& P0, const FMSPt& P1) { Segs.Add({ P0, P1 }); };

                switch (Code)
                {
                case 1:  Add(EL(), EB()); break;
                case 2:  Add(EB(), ER()); break;
                case 3:  Add(EL(), ER()); break;
                case 4:  Add(ER(), ET()); break;
                case 5:  Add(EL(), ET()); Add(EB(), ER()); break;   // saddle
                case 6:  Add(EB(), ET()); break;
                case 7:  Add(EL(), ET()); break;
                case 8:  Add(ET(), EL()); break;
                case 9:  Add(ET(), EB()); break;
                case 10: Add(EB(), EL()); Add(ET(), ER()); break;   // saddle
                case 11: Add(ET(), ER()); break;
                case 12: Add(ER(), EL()); break;
                case 13: Add(ER(), EB()); break;
                case 14: Add(EB(), EL()); break;
                default: break;
                }
            }
        }
        if (Segs.Num() == 0) return Out;

        // Stitch segments sharing endpoints into ordered loops. Endpoints from a shared cell edge are
        // computed identically by both squares, so quantising to a fine key matches them exactly.
        const double KeyScale = 1.0 / FMath::Max(Cell * 0.05, 0.01);
        auto KeyOf = [&](const FVector2D& P)
        {
            return FIntPoint(FMath::RoundToInt(P.X * KeyScale), FMath::RoundToInt(P.Y * KeyScale));
        };
        TMultiMap<FIntPoint, int32> EndMap;   // point key -> 2*seg + end(0/1)
        for (int32 s = 0; s < Segs.Num(); ++s)
        {
            EndMap.Add(KeyOf(Segs[s].A.P), 2 * s + 0);
            EndMap.Add(KeyOf(Segs[s].B.P), 2 * s + 1);
        }
        TArray<uint8> Used; Used.SetNumZeroed(Segs.Num());

        for (int32 s0 = 0; s0 < Segs.Num(); ++s0)
        {
            if (Used[s0]) continue;
            TArray<FVector2D> LoopPts, LoopFeet;
            Used[s0] = 1;
            LoopPts.Add(Segs[s0].A.P);   LoopFeet.Add(Segs[s0].A.Foot);
            LoopPts.Add(Segs[s0].B.P);   LoopFeet.Add(Segs[s0].B.Foot);
            const FIntPoint StartKey = KeyOf(Segs[s0].A.P);
            FIntPoint CurKey = KeyOf(Segs[s0].B.P);

            for (int32 Guard = 0; Guard < Segs.Num(); ++Guard)
            {
                if (CurKey == StartKey) break;
                TArray<int32> Cand;
                EndMap.MultiFind(CurKey, Cand);
                int32 NextSeg = -1, NextEnd = -1;
                for (int32 Ref : Cand)
                {
                    const int32 Seg = Ref / 2;
                    if (Used[Seg]) continue;
                    NextSeg = Seg; NextEnd = Ref & 1; break;
                }
                if (NextSeg < 0) break;
                Used[NextSeg] = 1;
                const FMSPt& Far = (NextEnd == 0) ? Segs[NextSeg].B : Segs[NextSeg].A;
                LoopPts.Add(Far.P); LoopFeet.Add(Far.Foot);
                CurKey = KeyOf(Far.P);
            }

            if (LoopPts.Num() < 3) continue;

            // Drop hole loops: marching squares also rings every void INSIDE the occupied region (an
            // enclosed courtyard, the gap around a stray interior mesh, a cell farther than Margin from any
            // wall reached through a door/window opening or a sparse-vertex seam). A camera there sits
            // inside the building with clear line-of-sight to the inner wall, so the LOS cull cannot drop
            // it. Each foot (nearest occupied cell) lies INSIDE an outer-boundary loop but OUTSIDE a hole
            // loop — so keep the loop only when its feet fall inside it. L / U notches survive because they
            // belong to the outer boundary, whose feet are inside.
            {
                int32 Votes = 0, InCount = 0;
                const int32 Stride = FMath::Max(1, LoopFeet.Num() / 15);
                for (int32 i = 0; i < LoopFeet.Num(); i += Stride)
                {
                    ++Votes;
                    if (PointInPolygon2D(LoopPts, LoopFeet[i])) ++InCount;
                }
                if (Votes > 0 && InCount * 2 <= Votes) continue;   // mostly outside -> hole, drop
            }

            // Resample the closed loop to ResampleStep, interpolating foot alongside.
            double Perim = 0.0;
            const int32 LN = LoopPts.Num();
            for (int32 i = 0; i < LN; ++i) Perim += (LoopPts[(i + 1) % LN] - LoopPts[i]).Size();
            if (Perim < KINDA_SMALL_NUMBER) continue;

            const int32 NSamp = FMath::Max(3, FMath::FloorToInt(Perim / FMath::Max(ResampleStep, 1.0)));
            const double Step = Perim / NSamp;
            TArray<FVector2D> RP, RF;
            RP.Reserve(NSamp); RF.Reserve(NSamp);
            double Target = 0.0, Acc2 = 0.0;
            int32 Seg = 0;
            double SegLen = (LoopPts[1 % LN] - LoopPts[0]).Size();
            for (int32 n = 0; n < NSamp; ++n)
            {
                while (Acc2 + SegLen < Target && Seg < LN)
                {
                    Acc2 += SegLen;
                    ++Seg;
                    SegLen = (LoopPts[(Seg + 1) % LN] - LoopPts[Seg % LN]).Size();
                }
                const double t = SegLen > KINDA_SMALL_NUMBER ? (Target - Acc2) / SegLen : 0.0;
                const int32 a = Seg % LN, b = (Seg + 1) % LN;
                RP.Add(FMath::Lerp(LoopPts[a], LoopPts[b], FMath::Clamp(t, 0.0, 1.0)));
                RF.Add(FMath::Lerp(LoopFeet[a], LoopFeet[b], FMath::Clamp(t, 0.0, 1.0)));
                Target += Step;
            }
            Out.Loops.Add(MoveTemp(RP));
            Out.Feet.Add(MoveTemp(RF));
        }
        return Out;
    }
}

FPathGenerator::FVerticalOBB FPathGenerator::BuildActorOBB(const AActor* Actor)
{
    if (!Actor) return FVerticalOBB();
    return BuildOBBForActors(TArray<AActor*>{ const_cast<AActor*>(Actor) });
}

bool FPathGenerator::IsGroundLikeActor(UWorld* World, const AActor* Actor, const FBuildingDetectParams& Params)
{
    if (!Actor) return false;

    // Manual override wins outright — the user marked this actor as ground.
    if (Params.ForcedGroundActors.Contains(Actor)) return true;

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

namespace
{
    // Order-independent signature of (actor set + their transforms + detection params). Locations are
    // quantised to 1 cm and rotations to 1 degree, so sub-threshold jitter does not bust the cache
    // while any real move/add/remove/param change does. Order independence (summation) lets the two
    // callers pass the enabled actors in different orders and still hit the same cache entry.
    uint32 ComputeDetectSignature(const TArray<AActor*>& Actors,
        const FPathGenerator::FBuildingDetectParams& P)
    {
        uint32 Sum = 0;
        for (const AActor* A : Actors)
        {
            if (!A) continue;
            const FTransform Xf = A->GetActorTransform();
            const FVector T = Xf.GetLocation();
            const FRotator R = Xf.Rotator();
            uint32 H = GetTypeHash(A->GetFName());
            H = HashCombine(H, (uint32)(int32)FMath::RoundToFloat(T.X));
            H = HashCombine(H, (uint32)(int32)FMath::RoundToFloat(T.Y));
            H = HashCombine(H, (uint32)(int32)FMath::RoundToFloat(T.Z));
            H = HashCombine(H, (uint32)(int32)FMath::RoundToFloat(R.Yaw));
            H = HashCombine(H, (uint32)(int32)FMath::RoundToFloat(R.Pitch));
            H = HashCombine(H, (uint32)(int32)FMath::RoundToFloat(R.Roll));
            Sum += H;
        }
        // Fold the manual ground overrides in (order-independent) so editing that list busts the cache.
        uint32 GroundSum = 0;
        for (const AActor* A : P.ForcedGroundActors)
            if (A) GroundSum += GetTypeHash(A->GetFName());

        uint32 Sig = HashCombine(Sum, (uint32)Actors.Num());
        Sig = HashCombine(Sig, GroundSum);
        Sig = HashCombine(Sig, (uint32)P.ForcedGroundActors.Num());
        Sig = HashCombine(Sig, (uint32)(int32)FMath::RoundToFloat(P.ConnectGap));
        Sig = HashCombine(Sig, (uint32)(int32)FMath::RoundToFloat(P.MinBuildingHeight));
        Sig = HashCombine(Sig, (uint32)(int32)FMath::RoundToFloat(P.MinBuildingFootprint));
        return Sig;
    }

    FCriticalSection GBuildingCacheCS;
    bool GBuildingCacheValid = false;
    uint32 GBuildingCacheSig = 0;
    TArray<FPathGenerator::FOrbitTarget> GBuildingCache;
}

void FPathGenerator::InvalidateBuildingCache()
{
    FScopeLock Lock(&GBuildingCacheCS);
    GBuildingCacheValid = false;
    GBuildingCache.Reset();
}

TArray<FPathGenerator::FOrbitTarget> FPathGenerator::DetectBuildingsCached(
    UWorld* World, const TArray<AActor*>& Actors, const FBuildingDetectParams& Params)
{
    const uint32 Sig = ComputeDetectSignature(Actors, Params);
    {
        FScopeLock Lock(&GBuildingCacheCS);
        if (GBuildingCacheValid && GBuildingCacheSig == Sig)
        {
            // A stale entry can still reference actors destroyed since (e.g. a map reload that reused
            // the signature); validate before trusting the memo.
            bool bAllValid = true;
            for (const FOrbitTarget& T : GBuildingCache)
            {
                for (const TWeakObjectPtr<AActor>& WA : T.Actors)
                    if (!WA.IsValid()) { bAllValid = false; break; }
                if (!bAllValid) break;
            }
            if (bAllValid) return GBuildingCache;
        }
    }

    TArray<FOrbitTarget> Result = DetectBuildings(World, Actors, Params);
    {
        FScopeLock Lock(&GBuildingCacheCS);
        GBuildingCacheSig = Sig;
        GBuildingCache = Result;
        GBuildingCacheValid = true;
    }
    return Result;
}

TArray<FPathGenerator::FOrbitTarget> FPathGenerator::DetectBuildings(
    UWorld* World, const TArray<AActor*>& Actors, const FBuildingDetectParams& Params)
{
    // Drop ground (else it bridges every building into one). No per-piece size filter — connectivity
    // decides membership, so small genuine attachments (rooftop AC units, parapets) are kept. The
    // ground test ray-casts per actor, so classify in parallel (the game thread is parked in
    // ParallelFor → the read-only line traces and actor-bounds reads are safe), then compact serially.
    TArray<uint8> bIsGround;
    bIsGround.SetNumZeroed(Actors.Num());
    ParallelFor(Actors.Num(), [&](int32 i)
    {
        const AActor* A = Actors[i];
        bIsGround[i] = (A && IsGroundLikeActor(World, A, Params)) ? 1 : 0;
    });

    TArray<AActor*> Structures;
    Structures.Reserve(Actors.Num());
    for (int32 i = 0; i < Actors.Num(); ++i)
        if (Actors[i] && !bIsGround[i])
            Structures.Add(Actors[i]);

    const int32 N = Structures.Num();

    // Connectivity by REAL collision geometry instead of bounding boxes: two structures merge only when
    // their actual meshes touch/interpenetrate or come within ConnectGap of each other. Each structure
    // component is queried against the world with its own collision shape (complex trace, so it hits the
    // true triangle mesh); an overlap, or a short cardinal sweep of length ConnectGap reaching another
    // structure, unions the two. A street leaves a real gap wider than ConnectGap, so the buildings it
    // separates stay distinct — which a coarse whole-building box (even a min-area OBB) cannot guarantee
    // for L-shaped / concave / rotated plans. Ground is not in this graph, so it never bridges buildings.
    TArray<int32> Parent;
    Parent.SetNum(N);
    for (int32 i = 0; i < N; ++i) Parent[i] = i;
    auto Find = [&Parent](int32 x)
    {
        while (Parent[x] != x) { Parent[x] = Parent[Parent[x]]; x = Parent[x]; }
        return x;
    };

    TMap<const AActor*, int32> ActorToStruct;
    ActorToStruct.Reserve(N);
    for (int32 i = 0; i < N; ++i) ActorToStruct.Add(Structures[i], i);

    const float Gap = FMath::Max(Params.ConnectGap, 1.f);

    // Sample each structure's ACTUAL surface (static-mesh vertices, coarsest LOD), deduplicated to a Gap
    // grid to bound the count. These are real geometry points — NOT a component's simple collision shape,
    // which for building meshes is typically an auto-fitted box/convex larger than the mesh and would
    // merge two buildings whose boxes (not their walls) come within Gap. UObject access happens here on
    // the game thread; the parallel pass below only reads these points and the read-only physics scene.
    const float CellSize = FMath::Max(Gap, 25.f);
    TArray<TArray<FVector>> StructPoints;
    StructPoints.SetNum(N);
    for (int32 i = 0; i < N; ++i)
    {
        TArray<UStaticMeshComponent*> SMCs;
        Structures[i]->GetComponents<UStaticMeshComponent>(SMCs);
        TSet<FIntVector> Seen;
        for (UStaticMeshComponent* SMC : SMCs)
        {
            if (!SMC || !SMC->IsRegistered()) continue;
            UStaticMesh* SM = SMC->GetStaticMesh();
            if (!SM || !SM->GetRenderData() || SM->GetRenderData()->LODResources.Num() == 0) continue;
            const int32 LOD = SM->GetRenderData()->LODResources.Num() - 1;   // coarsest LOD = fewest verts
            const FPositionVertexBuffer& PVB =
                SM->GetRenderData()->LODResources[LOD].VertexBuffers.PositionVertexBuffer;
            const FTransform Xf = SMC->GetComponentTransform();
            const uint32 NumV = PVB.GetNumVertices();
            for (uint32 v = 0; v < NumV; ++v)
            {
                const FVector P = Xf.TransformPosition((FVector)PVB.VertexPosition(v));
                const FIntVector Cell(
                    FMath::FloorToInt(P.X / CellSize), FMath::FloorToInt(P.Y / CellSize),
                    FMath::FloorToInt(P.Z / CellSize));
                bool bAlready = false;
                Seen.Add(Cell, &bAlready);
                if (!bAlready) StructPoints[i].Add(P);
            }
        }
    }

    // Connectivity: from each surface sample, a Gap-radius sphere sweep against COMPLEX (triangle)
    // geometry. The query shape is a small sphere at a real surface point — no oversized collision box —
    // and the swept geometry is the real mesh, so two structures merge only if their actual surfaces come
    // within Gap. Each task writes only its own Neighbors slot, so the parallel pass is race-free.
    TArray<TArray<int32>> Neighbors;
    Neighbors.SetNum(N);
    ParallelFor(N, [&](int32 i)
    {
        FCollisionQueryParams QP;
        QP.bTraceComplex = true;
        QP.AddIgnoredActor(Structures[i]);   // the sphere sits on i's own surface; only neighbours matter
        const FCollisionShape Sphere = FCollisionShape::MakeSphere(Gap);

        TSet<int32> Local;
        for (const FVector& P : StructPoints[i])
        {
            TArray<FHitResult> Hits;
            World->SweepMultiByChannel(Hits, P, P + FVector(0, 0, 0.5f), FQuat::Identity,
                ECC_Visibility, Sphere, QP);
            for (const FHitResult& H : Hits)
                if (const int32* Idx = ActorToStruct.Find(H.GetActor()))
                    if (*Idx != i) Local.Add(*Idx);
        }
        Neighbors[i] = Local.Array();
    });

    for (int32 i = 0; i < N; ++i)
        for (int32 j : Neighbors[i])
            Parent[Find(i)] = Find(j);

    // Rooftop / sits-on-top merge: an appendage resting on a roof (AC unit, water house, parapet, spire)
    // touches the building only through a thin horizontal seam that the surface-sphere pass can miss
    // (small mounting gap, or its footprint centre is far from any roof vertex). Merge structure i into j
    // when i's footprint centre lies inside j's footprint AND i starts above j's mid-height — i.e. it
    // sits on j's upper part. Cheap OBB-only test (no ray-casts); a ground-level prop inside a courtyard
    // is excluded by the height condition, and two side-by-side buildings never contain each other.
    {
        TArray<FVerticalOBB> SOBB;
        SOBB.SetNum(N);
        for (int32 i = 0; i < N; ++i)
            SOBB[i] = BuildOBBForActors(TArray<AActor*>{ Structures[i] });

        auto PointInOBBXY = [](const FVerticalOBB& O, double X, double Y) -> bool
        {
            const FVector2D D(X - O.Center.X, Y - O.Center.Y);
            const FVector2D AY(-O.AxisX.Y, O.AxisX.X);
            return FMath::Abs(FVector2D::DotProduct(D, O.AxisX)) <= O.HalfXY.X
                && FMath::Abs(FVector2D::DotProduct(D, AY)) <= O.HalfXY.Y;
        };

        for (int32 i = 0; i < N; ++i)
        {
            if (!SOBB[i].bValid) continue;
            for (int32 j = 0; j < N; ++j)
            {
                if (i == j || !SOBB[j].bValid) continue;
                if (Find(i) == Find(j)) continue;
                const double JMid = 0.5 * (SOBB[j].MinZ + SOBB[j].MaxZ);
                if (SOBB[i].Center.Z < JMid) continue;                                   // i not on j's upper part
                if (!PointInOBBXY(SOBB[j], SOBB[i].Center.X, SOBB[i].Center.Y)) continue; // i's centre over j
                Parent[Find(i)] = Find(j);
            }
        }
    }

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
        TArray<uint8> Occupied;  // DILATED target occupancy — drives HeightAt's DSM hole-filling
        TArray<float> TopZ;      // top target surface height per cell (valid where Occupied) — the DSM
        TArray<uint8> RawOccupied; // pre-dilation occupancy (genuine target hits only) — drives the void cull
        TArray<float> TopAnyZ;   // topmost surface of ANY geometry per column (not just targets) — drives clearance

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

        // Topmost surface of ANY geometry (target, terrain, other building) over the cell AND its 8
        // neighbours — the physical ceiling a survey camera must clear at (X,Y). The target-only DSM
        // (TopZ) says nothing about un-selected geometry, so a nadir/oblique waypoint placed purely
        // from TopZ can sit inside a tree or a neighbouring structure; this lets the caller lift it
        // clear. Returns -FLT_MAX where no geometry is within reach (genuine void: nothing to hit).
        // Neighbourhood max for the same coarse-grid safety reason as HeightAt.
        float ClearanceAt(float X, float Y) const
        {
            if (NumX <= 0 || NumY <= 0) return -FLT_MAX;
            const int32 cx = FMath::FloorToInt((X - Origin.X) / CellSize);
            const int32 cy = FMath::FloorToInt((Y - Origin.Y) / CellSize);
            float Best = -FLT_MAX;
            for (int32 dy = -1; dy <= 1; ++dy)
            {
                for (int32 dx = -1; dx <= 1; ++dx)
                {
                    const int32 ix = cx + dx, iy = cy + dy;
                    if (ix < 0 || ix >= NumX || iy < 0 || iy >= NumY) continue;
                    Best = FMath::Max(Best, TopAnyZ[iy * NumX + ix]);
                }
            }
            return Best;
        }

        // Fraction of a ground footprint rectangle that overlaps occupied (mesh) cells. The rectangle
        // is centred at (Cx,Cy), oriented so its "along" axis points along Yaw, with the given half
        // sizes. Samples a small NxN grid. Used to cull survey poses aimed mostly at empty background:
        // outside the scene's real geometry the cells are unoccupied, so the fraction drops toward 0.
        // Tests RAW occupancy, NOT the dilated Occupied: the DSM dilation inflates the occupied mask
        // outward by up to 8 cells to lift boundary poses, which would mask genuine void and let
        // void-facing poses survive — exactly what this cull exists to stop.
        float OccupiedFraction(float Cx, float Cy, float Yaw, float HalfAlong, float HalfAcross) const
        {
            if (NumX <= 0 || NumY <= 0) return 0.f;
            const float C = FMath::Cos(Yaw), S = FMath::Sin(Yaw);
            const int32 N = 5;
            int32 Total = 0, Occ = 0;
            for (int32 a = 0; a < N; ++a)
            {
                const float U = ((a + 0.5f) / N * 2.f - 1.f) * HalfAlong;
                for (int32 b = 0; b < N; ++b)
                {
                    const float V = ((b + 0.5f) / N * 2.f - 1.f) * HalfAcross;
                    const float X = Cx + U * C - V * S;
                    const float Y = Cy + U * S + V * C;
                    const int32 cx = FMath::FloorToInt((X - Origin.X) / CellSize);
                    const int32 cy = FMath::FloorToInt((Y - Origin.Y) / CellSize);
                    ++Total;
                    if (cx >= 0 && cx < NumX && cy >= 0 && cy < NumY && RawOccupied[cy * NumX + cx] != 0) ++Occ;
                }
            }
            return Total > 0 ? (float)Occ / (float)Total : 0.f;
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
        Grid.TopAnyZ.Init(-FLT_MAX, Grid.NumX * Grid.NumY);

        const float ZTop = RegionBox.Max.Z + 1000.f;
        const float ZBot = RegionBox.Min.Z - 1000.f;

        // One penetrating down-column per cell. The columns are independent and each writes only its own
        // slot, so run them in parallel: the game thread parks in ParallelFor, making the read-only line
        // traces safe (same pattern as DetectBuildings). This is the heaviest game-thread cost in path
        // generation (up to ~256x256 columns), so parallelising it is the biggest single speed-up.
        ParallelFor(Grid.NumX * Grid.NumY, [&](int32 CellIdx)
        {
            const int32 ix = CellIdx % Grid.NumX;
            const int32 iy = CellIdx / Grid.NumX;
            const float X = Grid.Origin.X + (ix + 0.5f) * Grid.CellSize;
            const float Y = Grid.Origin.Y + (iy + 0.5f) * Grid.CellSize;

            FCollisionQueryParams QP;
            QP.bTraceComplex = true;
            int32 MaxPen = 8;
            bool bOcc = false;
            float HitZ = 0.f;
            float TopAny = -FLT_MAX;
            while (MaxPen-- > 0)
            {
                FHitResult Hit;
                if (!World->LineTraceSingleByChannel(
                        Hit, FVector(X, Y, ZTop), FVector(X, Y, ZBot), ECC_Visibility, QP))
                    break;
                AActor* HA = Hit.GetActor();
                if (!HA) break;
                // First hit down the column is the topmost surface of any kind — the clearance ceiling.
                if (TopAny == -FLT_MAX) TopAny = Hit.ImpactPoint.Z;
                if (Targets.Contains(HA)) { bOcc = true; HitZ = Hit.ImpactPoint.Z; break; }
                QP.AddIgnoredActor(HA);
            }
            Grid.Occupied[CellIdx] = bOcc ? 1 : 0;
            Grid.TopZ[CellIdx] = HitZ;
            Grid.TopAnyZ[CellIdx] = TopAny;
        });

        // Snapshot genuine occupancy BEFORE dilation: the void cull (OccupiedFraction) needs the true
        // geometry footprint, while the dilation below only serves HeightAt's hole-filling.
        Grid.RawOccupied = Grid.Occupied;

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

    FCollisionQueryParams QueryParams;
    QueryParams.bTraceComplex = true;

    // Phase 1 (game thread): for each building, rasterise its per-height footprint and trace the
    // constant-standoff (Margin) iso-contour into concave-aware orbit rings — ONLY when side orbit is
    // enabled. Each building is orbited individually; other selected buildings drop cameras inside them.
    if (Params.bSideOrbit)
    {
        // A camera that lands inside ANOTHER selected building is meaningless. Test each orbit
        // position against every OTHER building's per-COMPONENT oriented boxes (not the coarse merged
        // building box) and drop it. The raised oblique rings copy this mask, inheriting the rejection.
        auto PointInVerticalOBB = [](const FVerticalOBB& O, const FVector& P) -> bool
        {
            if (!O.bValid || P.Z < O.MinZ || P.Z > O.MaxZ) return false;
            const FVector2D D(P.X - O.Center.X, P.Y - O.Center.Y);
            const FVector2D AxisY(-O.AxisX.Y, O.AxisX.X);
            return FMath::Abs(FVector2D::DotProduct(D, O.AxisX)) <= O.HalfXY.X
                && FMath::Abs(FVector2D::DotProduct(D, AxisY)) <= O.HalfXY.Y;
        };

        AllBuildings.Reserve(Params.Buildings.Num());
        for (int32 BIdx = 0; BIdx < Params.Buildings.Num(); ++BIdx)
        {
            const FOrbitTarget& Building = Params.Buildings[BIdx];
            if (!Building.Bounds.IsValid)
            {
                continue;
            }

            TArray<FVerticalOBB> OtherCompOBBs;
            for (int32 OIdx = 0; OIdx < Params.Buildings.Num(); ++OIdx)
            {
                if (OIdx == BIdx) continue;
                for (const TWeakObjectPtr<AActor>& WA : Params.Buildings[OIdx].Actors)
                {
                    AActor* A = WA.Get();
                    if (!A) continue;
                    TArray<UPrimitiveComponent*> Prims;
                    A->GetComponents<UPrimitiveComponent>(Prims);
                    for (UPrimitiveComponent* P : Prims)
                    {
                        if (!P || !P->IsRegistered()) continue;
                        FVerticalOBB CompOBB = BuildOBBForComponent(P);
                        if (CompOBB.bValid) OtherCompOBBs.Add(CompOBB);
                    }
                }
            }

            TSet<AActor*> TargetActors;
            for (const TWeakObjectPtr<AActor>& WA : Building.Actors)
            {
                if (AActor* A = WA.Get())
                    TargetActors.Add(A);
            }

            // Cross-section source: each member primitive contributes its world-space box (8 corners
            // + a Z-range). At each ring height we hull the corners of the components spanning that
            // height into a stable footprint polygon — no flaky per-ray surface probing.
            struct FCompBox { TArray<FVector2D> CornersXY; float MinZ; float MaxZ; };
            TArray<FCompBox> CompBoxes;
            for (const TWeakObjectPtr<AActor>& WA : Building.Actors)
            {
                AActor* A = WA.Get();
                if (!A) continue;
                TArray<UPrimitiveComponent*> Prims;
                A->GetComponents<UPrimitiveComponent>(Prims);
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
                    FCompBox CB;
                    CB.MinZ = FLT_MAX; CB.MaxZ = -FLT_MAX;
                    for (const FVector& L : LC)
                    {
                        const FVector W = Xf.TransformPosition(L);
                        CB.CornersXY.Add(FVector2D(W.X, W.Y));
                        CB.MinZ = FMath::Min(CB.MinZ, (float)W.Z);
                        CB.MaxZ = FMath::Max(CB.MaxZ, (float)W.Z);
                    }
                    CompBoxes.Add(MoveTemp(CB));
                }
            }

            // Actual mesh vertices (coarsest LOD, world space) of every member. Folded into the per-ring
            // footprint raster so a concave plan modelled as ONE mesh (whose component box is a plain
            // rectangle) still recovers its true outline — component boxes alone cannot.
            TArray<FVector> MeshVertsWorld;
            for (const TWeakObjectPtr<AActor>& WA : Building.Actors)
            {
                AActor* A = WA.Get();
                if (!A) continue;
                TArray<UStaticMeshComponent*> SMCs;
                A->GetComponents<UStaticMeshComponent>(SMCs);
                for (UStaticMeshComponent* SMC : SMCs)
                {
                    if (!SMC || !SMC->IsRegistered()) continue;
                    UStaticMesh* SM = SMC->GetStaticMesh();
                    if (!SM || !SM->GetRenderData() || SM->GetRenderData()->LODResources.Num() == 0) continue;
                    const int32 LOD = SM->GetRenderData()->LODResources.Num() - 1;
                    const FPositionVertexBuffer& PVB =
                        SM->GetRenderData()->LODResources[LOD].VertexBuffers.PositionVertexBuffer;
                    const FTransform Xf = SMC->GetComponentTransform();
                    const uint32 NumV = PVB.GetNumVertices();
                    MeshVertsWorld.Reserve(MeshVertsWorld.Num() + (int32)NumV);
                    for (uint32 v = 0; v < NumV; ++v)
                        MeshVertsWorld.Add(Xf.TransformPosition((FVector)PVB.VertexPosition(v)));
                }
            }

            const bool bUseOBB = Building.OBB.bValid;
            const float BoxMaxZ = bUseOBB ? (float)Building.OBB.MaxZ : (float)Building.Bounds.Max.Z;

            // StartHeight is measured above the LOCAL ground, not an absolute Z: ground heights vary
            // across the scene, so trace straight down through this building to the terrain beneath it
            // and start the lowest ring StartHeight above that. Falls back to the building base if no
            // ground is hit (e.g. a building modelled without ground under it).
            float GroundZ = bUseOBB ? (float)Building.OBB.MinZ : (float)Building.Bounds.Min.Z;
            {
                const FVector C = bUseOBB ? Building.OBB.Center : Building.Bounds.GetCenter();
                FCollisionQueryParams GroundParams = QueryParams;
                for (AActor* TA : TargetActors) GroundParams.AddIgnoredActor(TA);
                FHitResult GroundHit;
                if (Params.World->LineTraceSingleByChannel(
                        GroundHit, FVector(C.X, C.Y, BoxMaxZ + 1000.f),
                        FVector(C.X, C.Y, GroundZ - 100000.f), ECC_Visibility, GroundParams))
                {
                    GroundZ = (float)GroundHit.ImpactPoint.Z;
                }
            }
            const float BoxMinZ = GroundZ + Params.StartHeight;

            const float StepH = FMath::Max(
                2.f * Params.Margin * FMath::Tan(HFovRad * 0.5f) * FMath::Max(1.f - Params.HOverlap, 0.05f), 10.f);
            const float StepV = FMath::Max(
                2.f * Params.Margin * FMath::Tan(VFovRad * 0.5f) * FMath::Max(1.f - Params.VOverlap, 0.05f), 10.f);

            // Lower the highest ring to position the building top at the 1/4 mark of the frame (reserving 1/4 for the sky).
            const float BoxMaxZ_Rings = FMath::Max(BoxMaxZ - Params.Margin * FMath::Tan(VFovRad * 0.25f), BoxMinZ + StepV);
            const float BuildingH = FMath::Max(BoxMaxZ_Rings - BoxMinZ, 0.f);
            const int32 NumRings = FMath::Max(1, FMath::CeilToInt(BuildingH / StepV));

            // Sample the offset contour finer than StepH so Phase 2 can resample to StepH and the
            // LOS / neighbour-OBB rejection has enough granularity.
            const float ContourStep = FMath::Max(StepH * 0.5f, 50.f);

            // Validate and record one ring pose. OrbitXY is the standoff point, FootXY the look-at target
            // on the wall (so the camera faces it squarely). Drops poses with no clear line to the wall,
            // poses embedded in another selected building, and poses below the local ground.
            auto Emit = [&](const FVector2D& OrbitXY, const FVector2D& FootXY, float Z,
                            TArray<FVector>& OrbitPoints, TArray<FVector>& SurfacePoints, TArray<bool>& Valid)
            {
                const FVector SurfacePt(FootXY.X, FootXY.Y, Z);
                const FVector OrbitPt(OrbitXY.X, OrbitXY.Y, Z);
                FVector2D Nrm = (OrbitXY - FootXY).GetSafeNormal();
                if (Nrm.IsNearlyZero()) Nrm = FVector2D(1.f, 0.f);

                bool bValid = true;
                {
                    FCollisionQueryParams LosParams = QueryParams;
                    for (AActor* TA : TargetActors) LosParams.AddIgnoredActor(TA);
                    FHitResult LosHit;
                    const FVector LosStart(FootXY.X + Nrm.X * 2.f, FootXY.Y + Nrm.Y * 2.f, Z);
                    if (Params.World->LineTraceSingleByChannel(
                            LosHit, LosStart, OrbitPt, ECC_Visibility, LosParams))
                    {
                        bValid = false;
                    }
                }
                if (bValid)
                {
                    for (const FVerticalOBB& O : OtherCompOBBs)
                        if (PointInVerticalOBB(O, OrbitPt)) { bValid = false; break; }
                }

                // Underground guard: the ring height is set from the ground under the building CENTRE, but
                // a camera stands Margin away over possibly higher terrain (a slope), so its height can fall
                // below the LOCAL ground. Trace down at the camera XY (ignoring this building) and drop the
                // pose if it is not clearly above the local surface.
                if (bValid)
                {
                    FCollisionQueryParams GroundParams = QueryParams;
                    for (AActor* TA : TargetActors) GroundParams.AddIgnoredActor(TA);
                    FHitResult GroundHit;
                    if (Params.World->LineTraceSingleByChannel(
                            GroundHit, FVector(OrbitPt.X, OrbitPt.Y, RegionBox.Max.Z + 1000.f),
                            FVector(OrbitPt.X, OrbitPt.Y, RegionBox.Min.Z - 100000.f), ECC_Visibility, GroundParams))
                    {
                        const float GroundClearance = FMath::Clamp(Params.StartHeight * 0.5f, 50.f, 300.f);
                        if (OrbitPt.Z < (float)GroundHit.ImpactPoint.Z + GroundClearance)
                        {
                            bValid = false;
                        }
                    }
                }

                // Six-direction enclosure test: a pose inside the building is boxed in by walls / floor /
                // ceiling within a short radius, whereas an exterior orbit pose always has at least one open
                // direction (open sky above, or the open side away from the facade). These rays must NOT
                // ignore this building — its own walls and roof are exactly what box an interior pose in.
                // If all six axes hit geometry within EncloseDist, treat the pose as interior and drop it.
                if (bValid)
                {
                    const float EncloseDist = FMath::Max(Params.Margin * 1.5f, 200.f);
                    const FVector Dirs[6] = {
                        FVector(1, 0, 0), FVector(-1, 0, 0), FVector(0, 1, 0),
                        FVector(0, -1, 0), FVector(0, 0, 1), FVector(0, 0, -1) };
                    bool bBoxedIn = true;
                    for (const FVector& D : Dirs)
                    {
                        FHitResult H;
                        if (!Params.World->LineTraceSingleByChannel(
                                H, OrbitPt, OrbitPt + D * EncloseDist, ECC_Visibility, QueryParams))
                        {
                            bBoxedIn = false;
                            break;
                        }
                    }
                    if (bBoxedIn) bValid = false;
                }

                // Require something solid below: a pose with nothing at all beneath it sits in the void
                // (off the edge of the scene, or over a hole). Trace far down, hitting ANYTHING (terrain or
                // the building base — do not ignore it); drop the pose when the column is empty.
                if (bValid)
                {
                    FHitResult H;
                    if (!Params.World->LineTraceSingleByChannel(
                            H, OrbitPt, FVector(OrbitPt.X, OrbitPt.Y, RegionBox.Min.Z - 100000.f),
                            ECC_Visibility, QueryParams))
                    {
                        bValid = false;
                    }
                }

                SurfacePoints.Add(SurfacePt);
                OrbitPoints.Add(OrbitPt);
                Valid.Add(bValid);
            };

            FBuildingRings Rings;
            Rings.StepH = StepH;
            Rings.RingOrbitPoints.Reserve(NumRings);
            Rings.RingSurfacePoints.Reserve(NumRings);
            Rings.RingValid.Reserve(NumRings);

            for (int32 Ring = 0; Ring < NumRings; ++Ring)
            {
                const float T = (NumRings > 1) ? (float)Ring / (NumRings - 1) : 0.5f;
                const float Z = BoxMinZ + T * BuildingH;

                // Footprint at this height: component rectangles spanning Z (filled) PLUS mesh vertices in
                // the height band. BuildStandoffContours rasterises these and traces the Margin iso-contour,
                // so the ring hugs concave (L / U) plans and enters their inner corners, rounding smoothly
                // over notches narrower than 2*Margin. Falls back to the building OBB rectangle when neither
                // a spanning component nor a band vertex exists at this height.
                TArray<TArray<FVector2D>> CompQuads;
                for (const FCompBox& CB : CompBoxes)
                {
                    if (Z >= CB.MinZ - 1.f && Z <= CB.MaxZ + 1.f)
                    {
                        TArray<FVector2D> Quad = BuildFootprintHull(CB.CornersXY);
                        if (Quad.Num() >= 3) CompQuads.Add(MoveTemp(Quad));
                    }
                }
                TArray<FVector2D> BandVerts;
                {
                    const float ZLo = Z - 0.5f * StepV, ZHi = Z + 0.5f * StepV;
                    for (const FVector& V : MeshVertsWorld)
                        if (V.Z >= ZLo && V.Z <= ZHi) BandVerts.Add(FVector2D(V.X, V.Y));
                }
                if (CompQuads.Num() == 0 && BandVerts.Num() == 0 && bUseOBB)
                {
                    const FVerticalOBB& O = Building.OBB;
                    const FVector2D AxX = O.AxisX;
                    const FVector2D AxY(-AxX.Y, AxX.X);
                    const FVector2D C2(O.Center.X, O.Center.Y);
                    TArray<FVector2D> Quad;
                    Quad.Add(C2 + AxX * (float)O.HalfXY.X + AxY * (float)O.HalfXY.Y);
                    Quad.Add(C2 - AxX * (float)O.HalfXY.X + AxY * (float)O.HalfXY.Y);
                    Quad.Add(C2 - AxX * (float)O.HalfXY.X - AxY * (float)O.HalfXY.Y);
                    Quad.Add(C2 + AxX * (float)O.HalfXY.X - AxY * (float)O.HalfXY.Y);
                    CompQuads.Add(MoveTemp(Quad));
                }

                const FStandoffContours SC = BuildStandoffContours(
                    CompQuads, BandVerts, Params.Margin, ContourStep);

                TArray<FVector> OrbitPoints;
                TArray<FVector> SurfacePoints;
                TArray<bool> Valid;

                const bool bMultiLoop = SC.Loops.Num() > 1;
                for (int32 L = 0; L < SC.Loops.Num(); ++L)
                {
                    const TArray<FVector2D>& Loop = SC.Loops[L];
                    const TArray<FVector2D>& Ft = SC.Feet[L];
                    if (Loop.Num() < 3) continue;
                    for (int32 i = 0; i < Loop.Num(); ++i)
                        Emit(Loop[i], Ft[i], Z, OrbitPoints, SurfacePoints, Valid);

                    // Multiple disconnected footprints (e.g. two towers in one cluster): close this loop,
                    // then append an invalid sentinel so Phase 2's cyclic walk never bridges one loop to
                    // the next (nor wraps the last loop back to the first).
                    if (bMultiLoop)
                    {
                        Emit(Loop[0], Ft[0], Z, OrbitPoints, SurfacePoints, Valid);
                        OrbitPoints.Add(FVector(Loop[0].X, Loop[0].Y, Z));
                        SurfacePoints.Add(FVector(Ft[0].X, Ft[0].Y, Z));
                        Valid.Add(false);
                    }
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
         SurveyHOverlap = Params.SurveyHOverlap,
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
            // BOTH directions use the dedicated Survey overlap (independent of the facade orbit's
            // H-Overlap): a flat survey grid has no vertical stacking, so V-Overlap does not apply.
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

                // Single (default) lens: the camera leans along its travel direction, so the lean flips
                // with each serpentine row. The 5-lens oblique set keeps a fixed lean per pass (its
                // ±X/±Y coverage is the whole point) and is never flipped.
                const bool bAlternateLean = !bIncludeOblique;
                // The target-only DSM cannot see terrain / un-selected structures, so a survey waypoint
                // placed purely from it can sit inside them. Keep every camera at least this far above
                // the column's true geometry ceiling (ClearanceAt). Scales with the requested AGL.
                const float SurveyClearance = FMath::Max(0.25f * NadirAltitude, 100.f);

                for (const FSurveyPass& Pass : Passes)
                {
                    float FAlong, FAcross;
                    TiltedFootprint(FMath::DegreesToRadians(Pass.TiltDeg), FAlong, FAcross);
                    const float LeanStep = FMath::Max(FAlong  * FMath::Max(1.f - SurveyHOverlap, 0.05f), 10.f);
                    const float PerpStep = FMath::Max(FAcross * FMath::Max(1.f - SurveyHOverlap, 0.05f), 10.f);
                    const float StepX = Pass.bLeanAlongX ? LeanStep : PerpStep;
                    const float StepY = Pass.bLeanAlongX ? PerpStep : LeanStep;
                    const float TiltShift = NadirAltitude * FMath::Tan(FMath::DegreesToRadians(Pass.TiltDeg));
                    const float MinMeshFraction = 0.8f;

                    int32 StripIdx = 0;
                    for (float Y = MinY; Y <= MaxY + KINDA_SMALL_NUMBER; Y += StepY, ++StripIdx)
                    {
                        // Serpentine: odd strips are walked in reverse. Flip the lean 180° on them (single
                        // lens only) so it tracks travel — one row leans one way, the next the other,
                        // giving forward+backward oblique stereo of the same ground.
                        const bool bReverse = (StripIdx % 2 == 1);
                        const float StripYaw = (bAlternateLean && bReverse) ? Pass.Yaw + 180.f : Pass.Yaw;
                        const float StripYawRad = FMath::DegreesToRadians(StripYaw);
                        const FRotator PoseRot(-(90.f - Pass.TiltDeg), StripYaw, 0.f);

                        // Cull poses aimed mostly at empty background (the scene isn't a clean rectangle, so
                        // the region box overhangs into void at the edges/corners). The footprint sits a
                        // tilt-shift downrange of the camera along this strip's lean; if under 80% overlaps
                        // mesh (>20% empty), drop the pose.
                        const float ShiftX = TiltShift * FMath::Cos(StripYawRad);
                        const float ShiftY = TiltShift * FMath::Sin(StripYawRad);

                        TArray<FVector> StripPos;
                        for (float X = MinX; X <= MaxX + KINDA_SMALL_NUMBER; X += StepX)
                        {
                            const float MeshFraction = Grid.OccupiedFraction(
                                X + ShiftX, Y + ShiftY, StripYawRad, FAlong * 0.5f, FAcross * 0.5f);
                            if (MeshFraction < MinMeshFraction) continue;

                            // AGL over the capture target, lifted clear of any geometry at this column.
                            const float SurfZ = Grid.HeightAt(X, Y);
                            float CamZ = SurfZ + NadirAltitude;
                            const float ClearZ = Grid.ClearanceAt(X, Y);
                            if (ClearZ > -FLT_MAX) CamZ = FMath::Max(CamZ, ClearZ + SurveyClearance);
                            StripPos.Add(FVector(X, Y, CamZ));
                        }

                        if (bReverse)
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
