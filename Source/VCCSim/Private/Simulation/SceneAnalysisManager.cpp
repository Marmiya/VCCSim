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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "Simulation/SceneAnalysisManager.h"
#include "Components/StaticMeshComponent.h"
#include "TimerManager.h"
#include "Sensors/CameraSensor.h"
#include "EngineUtils.h"
#include "Engine/StaticMesh.h"
#include "Engine/StaticMeshActor.h"
#include "DrawDebugHelpers.h"

ASceneAnalysisManager::ASceneAnalysisManager()
{
    World = nullptr;
    TotalPointsInScene = 0;
    TotalTrianglesInScene = 0;
    CurrentCoveragePercentage = 0.0f;
    LogPath = FPaths::ProjectLogDir();
    SafeZoneMaterial = nullptr;
    SafeZoneVisualizationMesh = nullptr;
    CoverageVisualizationMesh = nullptr;
    CoverageMaterial = nullptr;

    // Set no collision and no tick
    PrimaryActorTick.bCanEverTick = false;
    SetActorEnableCollision(false);
}

bool ASceneAnalysisManager::Initialize(UWorld* InWorld, FString&& Path)
{
    if (!InWorld)
        return false;
    
    World = InWorld;
    LogPath = std::move(Path) + "/SceneAnalysisLog";
    return true;
}

void ASceneAnalysisManager::ScanScene()
{
    // Call implementation without region bounds
    ScanSceneImpl(TOptional<FBox>());
}

void ASceneAnalysisManager::ScanSceneRegion3D(
    float MinX, float MaxX, float MinY, float MaxY, float MinZ, float MaxZ)
{
    // Create a 3D region bounds
    FBox RegionBounds(
        FVector(MinX, MinY, MinZ),
        FVector(MaxX, MaxY, MaxZ)
    );
    
    ScanSceneImpl(RegionBounds);
}

void ASceneAnalysisManager::ScanSceneImpl(const TOptional<FBox>& RegionBounds)
{
    if (!World)
        return;
    
    // Log region info if bounds are specified
    if (RegionBounds.IsSet())
    {
        const FBox& Bounds = RegionBounds.GetValue();
        UE_LOG(LogTemp, Display, TEXT("Scanning scene within region bounds: "
                                      "X(%.2f to %.2f), Y(%.2f to %.2f), Z(%.2f to %.2f)"),
               Bounds.Min.X, Bounds.Max.X, Bounds.Min.Y, Bounds.Max.Y, Bounds.Min.Z, Bounds.Max.Z);
    }
    else
    {
        UE_LOG(LogTemp, Display, TEXT("Scanning entire scene (no bounds)"));
    }
    
    // Clear previous data
    SceneMeshes.Empty();
    TotalPointsInScene = 0;
    TotalTrianglesInScene = 0;
    
    int32 TotalActorsFound = 0;
    int32 TotalMeshComponentsFound = 0;
    int32 TotalMeshesInRegion = 0;
    
    // Iterate through ALL actors in the world instead of just StaticMeshActors
    for (TActorIterator<AActor> ActorItr(World); ActorItr; ++ActorItr)
    {
        AActor* Actor = *ActorItr;
        TotalActorsFound++;
        
        // Skip actors with the "NotSMActor" tag
        if (Actor->ActorHasTag(FName("NotSMActor")))
        {
            continue;
        }
        
        // Find all static mesh components on this actor
        TArray<UStaticMeshComponent*> MeshComponents;
        Actor->GetComponents<UStaticMeshComponent>(MeshComponents);
        
        for (UStaticMeshComponent* MeshComp : MeshComponents)
        {
            if (MeshComp && MeshComp->GetStaticMesh())
            {
                TotalMeshComponentsFound++;
                
                // If region bounds are specified, check if mesh intersects with region
                bool bShouldIncludeMesh = true;
                
                if (RegionBounds.IsSet())
                {
                    const FBox& RegionBox = RegionBounds.GetValue();
                    FBoxSphereBounds MeshBoundsS = MeshComp->Bounds;
                    FBox MeshBox = MeshBoundsS.GetBox();
                    
                    // Explicit overlap test instead of using Intersect
                    bool bOverlapsX = (MeshBox.Max.X >= RegionBox.Min.X) &&
                        (MeshBox.Min.X <= RegionBox.Max.X);
                    bool bOverlapsY = (MeshBox.Max.Y >= RegionBox.Min.Y) &&
                        (MeshBox.Min.Y <= RegionBox.Max.Y);
                    bool bOverlapsZ = (MeshBox.Max.Z >= RegionBox.Min.Z) &&
                        (MeshBox.Min.Z <= RegionBox.Max.Z);
                    
                    bShouldIncludeMesh = bOverlapsX && bOverlapsY && bOverlapsZ;
                    
                    if (bShouldIncludeMesh)
                    {
                        TotalMeshesInRegion++;
                        UE_LOG(LogTemp, Verbose,
                            TEXT("Mesh from %s is within region - Bounds: X(%.2f to %.2f), "
                                 "Y(%.2f to %.2f), Z(%.2f to %.2f)"),
                              *Actor->GetName(),
                              MeshBox.Min.X, MeshBox.Max.X,
                              MeshBox.Min.Y, MeshBox.Max.Y,
                              MeshBox.Min.Z, MeshBox.Max.Z);
                    }
                    else
                    {
                        UE_LOG(LogTemp, Verbose,
                            TEXT("Mesh from %s is outside region - Bounds:"
                                 " X(%.2f to %.2f), Y(%.2f to %.2f), Z(%.2f to %.2f)"),
                              *Actor->GetName(),
                              MeshBox.Min.X, MeshBox.Max.X,
                              MeshBox.Min.Y, MeshBox.Max.Y,
                              MeshBox.Min.Z, MeshBox.Max.Z);
                    }
                }
                
                if (bShouldIncludeMesh)
                {
                    FMeshInfo MeshInfo;
                    ExtractMeshData(MeshComp, MeshInfo);
                    SceneMeshes.Add(MeshInfo);
                    
                    TotalTrianglesInScene += MeshInfo.NumTriangles;
                    TotalPointsInScene += MeshInfo.NumVertices;
                }
            }
        }
    }
    
    UE_LOG(LogTemp, Display, TEXT("Scan complete: Found %d actors, "
                                  "%d with mesh components, %d within region bounds. "
                                  "Added %d meshes with %d triangles and %d vertices."),
          TotalActorsFound, TotalMeshComponentsFound, TotalMeshesInRegion,
          SceneMeshes.Num(), TotalTrianglesInScene, TotalPointsInScene);
    
    // If no meshes were found, provide a warning
    if (SceneMeshes.Num() == 0)
    {
        if (TotalActorsFound == 0)
        {
            UE_LOG(LogTemp, Warning, TEXT("No actors found in the world!"));
        }
        else if (TotalMeshComponentsFound == 0)
        {
            UE_LOG(LogTemp, Warning,
                TEXT("Found %d actors, but none have valid static mesh components!"), 
                TotalActorsFound);
        }
        else if (RegionBounds.IsSet())
        {
            UE_LOG(LogTemp, Warning,
                TEXT("Found %d actors with valid static mesh components, "
                     "but none intersect with the specified region!"), 
                  TotalMeshComponentsFound);
        }
    }
    
    ResetCoverage();
}

void ASceneAnalysisManager::RegisterCamera(URGBCameraComponent* CameraComponent)
{
    CameraIntrinsics.Add(CameraComponent->CameraName,
    CameraComponent->GetCameraIntrinsics());

    // CameraComponent->OnKeyPointCaptured.BindUFunction(
    //     this, "");
}

FMeshInfo ASceneAnalysisManager::GetMeshInfo(int32 MeshID) const
{
    for (const FMeshInfo& MeshInfo : SceneMeshes)
    {
        if (MeshInfo.MeshID == MeshID)
            return MeshInfo;
    }
    
    return FMeshInfo();
}

TArray<FMeshInfo> ASceneAnalysisManager::GetAllMeshInfo() const
{
    return SceneMeshes;
}

FCoverageData ASceneAnalysisManager::ComputeCoverage(
    const TArray<FTransform>& CameraTransforms, const FString& CameraName)
{
    FCoverageData CoverageData;
    CoverageData.CoveragePercentage = 0.0f;
    CoverageData.TotalVisibleTriangles = 0;
    
    if (!World || CameraTransforms.Num() == 0 || CoverageMap.Num() == 0)
        return CoverageData;
    
    // Get camera intrinsic for specified camera name
    FMatrix44f CameraIntrinsic;
    
    if (CameraIntrinsics.Contains(CameraName))
    {
        CameraIntrinsic = CameraIntrinsics[CameraName];
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("ComputeCoverage: No intrinsics found for camera %s"), *CameraName);
    }
    
    // Reset visibility of all points
    VisiblePoints.Empty(CoverageMap.Num() / 2); // Pre-allocate with estimated capacity
    InvisiblePoints.Empty(CoverageMap.Num() / 2);
    
    // Create a spatial hash grid for quickly finding which mesh a point belongs to
    // This is a simple implementation - could be replaced with a more sophisticated structure
    constexpr float SpatialGridSize = 500.0f; // Adjust based on scene scale
    TMap<FIntVector, TArray<int32>> SpatialGrid;
    
    // Populate spatial grid with mesh IDs
    for (int32 MeshIdx = 0; MeshIdx < SceneMeshes.Num(); ++MeshIdx)
    {
        const FMeshInfo& MeshInfo = SceneMeshes[MeshIdx];
        FBox MeshBox = MeshInfo.Bounds.GetBox();
        
        // Calculate grid cell range for this mesh
        FIntVector MinCell(
            FMath::FloorToInt(MeshBox.Min.X / SpatialGridSize),
            FMath::FloorToInt(MeshBox.Min.Y / SpatialGridSize),
            FMath::FloorToInt(MeshBox.Min.Z / SpatialGridSize)
        );
        
        FIntVector MaxCell(
            FMath::FloorToInt(MeshBox.Max.X / SpatialGridSize),
            FMath::FloorToInt(MeshBox.Max.Y / SpatialGridSize),
            FMath::FloorToInt(MeshBox.Max.Z / SpatialGridSize)
        );
        
        // Add mesh to all cells it overlaps
        for (int32 X = MinCell.X; X <= MaxCell.X; ++X)
        {
            for (int32 Y = MinCell.Y; Y <= MaxCell.Y; ++Y)
            {
                for (int32 Z = MinCell.Z; Z <= MaxCell.Z; ++Z)
                {
                    FIntVector Cell(X, Y, Z);
                    SpatialGrid.FindOrAdd(Cell).Add(MeshIdx);
                }
            }
        }
    }
    
    // Reset all points to invisible first and create a copy for parallel processing
    struct FPointVisibilityData
    {
        FVector Point;
        bool bIsVisible;
        
        FPointVisibilityData(const FVector& InPoint) : Point(InPoint), bIsVisible(false) {}
    };
    
    TArray<FPointVisibilityData> PointsData;
    PointsData.Reserve(CoverageMap.Num());
    
    for (auto& Pair : CoverageMap)
    {
        Pair.Value = false; // Reset to invisible
        PointsData.Emplace(Pair.Key);
    }
    
    // OPTIMIZATION: Lower threshold for parallel processing and use better work distribution
    const int32 ParallelThreshold = 1000; // Lowered from 10000
    
    // For each camera, check which points are visible
    for (const FTransform& CameraTransform : CameraTransforms)
    {
        // Construct frustum for this camera (this could be further optimized)
        FConvexVolume Frustum;
        ConstructFrustum(Frustum, CameraTransform, CameraIntrinsic);
        
        // OPTIMIZATION: Extract camera properties once outside the loops
        const FVector CameraLocation = CameraTransform.GetLocation();
        const FVector CameraForward = CameraTransform.GetRotation().GetForwardVector();
        
        // Process points in parallel for better performance
        if (PointsData.Num() > ParallelThreshold)
        {
            // OPTIMIZATION: Improved parallel processing with better work distribution
            const int32 NumBatches = FMath::Max(1, FMath::Min(FPlatformMisc::NumberOfCores() * 2, PointsData.Num() / 100));
            const int32 PointsPerBatch = FMath::DivideAndRoundUp(PointsData.Num(), NumBatches);
            
            ParallelFor(NumBatches, [&](int32 BatchIndex)
            {
                const int32 StartIdx = BatchIndex * PointsPerBatch;
                const int32 EndIdx = FMath::Min(StartIdx + PointsPerBatch, PointsData.Num());
                
                // Pre-allocate trace params outside the loop
                FCollisionQueryParams TraceParams;
                TraceParams.bTraceComplex = true;
                
                // OPTIMIZATION: Use batched visibility checks for better performance
                for (int32 PointIdx = StartIdx; PointIdx < EndIdx; ++PointIdx)
                {
                    FPointVisibilityData& PointData = PointsData[PointIdx];
                    
                    // Skip points that are already visible
                    if (PointData.bIsVisible)
                        continue;
                    
                    // OPTIMIZATION: Early directional culling (cheaper than frustum test)
                    FVector PointDir = (PointData.Point - CameraLocation).GetSafeNormal();
                    if (FVector::DotProduct(CameraForward, PointDir) <= 0.0f)
                        continue; // Point is behind camera
                    
                    // Early frustum culling
                    bool bInFrustum = true;
                    for (const FPlane& Plane : Frustum.Planes)
                    {
                        if (Plane.PlaneDot(PointData.Point) < 0.0f)
                        {
                            bInFrustum = false;
                            break;
                        }
                    }
                    
                    if (!bInFrustum)
                        continue;
                    
                    // OPTIMIZATION: Could implement occlusion culling here
                    
                    // Visibility check with line trace
                    FHitResult HitResult;
                    if (World->LineTraceSingleByChannel(
                            HitResult,
                            CameraLocation,
                            PointData.Point,
                            ECC_Visibility,
                            TraceParams))
                    {
                        // If we hit something near our target point, consider it visible
                        PointData.bIsVisible = (HitResult.Location - PointData.Point).SizeSquared() < 1.0f;
                    }
                    else
                    {
                        // If nothing was hit, point is visible
                        PointData.bIsVisible = true;
                    }
                }
            });
        }
        else
        {
            // Sequential processing for smaller point sets
            // Pre-allocate trace params outside the loop
            FCollisionQueryParams TraceParams;
            TraceParams.bTraceComplex = true;
            
            for (FPointVisibilityData& PointData : PointsData)
            {
                // Skip points that are already visible
                if (PointData.bIsVisible)
                    continue;
                
                // Early directional culling
                FVector PointDir = (PointData.Point - CameraLocation).GetSafeNormal();
                if (FVector::DotProduct(CameraForward, PointDir) <= 0.0f)
                    continue; // Point is behind camera
                
                // Early frustum culling
                bool bInFrustum = true;
                for (const FPlane& Plane : Frustum.Planes)
                {
                    if (Plane.PlaneDot(PointData.Point) < 0.0f)
                    {
                        bInFrustum = false;
                        break;
                    }
                }
                
                if (!bInFrustum)
                    continue;
                
                // Visibility check with line trace
                FHitResult HitResult;
                if (World->LineTraceSingleByChannel(
                        HitResult,
                        CameraLocation,
                        PointData.Point,
                        ECC_Visibility,
                        TraceParams))
                {
                    // If we hit something near our target point, consider it visible
                    PointData.bIsVisible = (HitResult.Location - PointData.Point).SizeSquared() < 1.0f;
                }
                else
                {
                    // If nothing was hit, point is visible
                    PointData.bIsVisible = true;
                }
            }
        }
    }
    
    // Update the CoverageMap with results and build visible/invisible point lists
    TSet<int32> VisibleMeshIDs;
    int32 VisiblePointCount = 0;
    
    for (int32 PointIdx = 0; PointIdx < PointsData.Num(); ++PointIdx)
    {
        const FPointVisibilityData& PointData = PointsData[PointIdx];
        
        // Update the coverage map
        CoverageMap[PointData.Point] = PointData.bIsVisible;
        
        if (PointData.bIsVisible)
        {
            VisiblePoints.Add(PointData.Point);
            VisiblePointCount++;
            
            // OPTIMIZATION: Use spatial grid to quickly find which meshes contain this point
            FIntVector GridCell(
                FMath::FloorToInt(PointData.Point.X / SpatialGridSize),
                FMath::FloorToInt(PointData.Point.Y / SpatialGridSize),
                FMath::FloorToInt(PointData.Point.Z / SpatialGridSize)
            );
            
            // Get meshes potentially containing this point
            if (SpatialGrid.Contains(GridCell))
            {
                const TArray<int32>& PotentialMeshIndices = SpatialGrid[GridCell];
                
                for (int32 MeshIdx : PotentialMeshIndices)
                {
                    // Final check if point is inside mesh bounds
                    if (SceneMeshes[MeshIdx].Bounds.GetBox().IsInsideOrOn(PointData.Point))
                    {
                        VisibleMeshIDs.Add(SceneMeshes[MeshIdx].MeshID);
                    }
                }
            }
        }
        else
        {
            InvisiblePoints.Add(PointData.Point);
        }
    }
    
    // Calculate coverage percentage
    int32 TotalPoints = CoverageMap.Num();
    float CoveragePercentage = TotalPoints > 0 ? (float)VisiblePointCount / (float)TotalPoints * 100.0f : 0.0f;
    
    // OPTIMIZATION: More efficient visible triangle calculation
    struct FMeshVisibilityData
    {
        int32 TotalPoints;
        int32 VisiblePoints;
        
        FMeshVisibilityData() : TotalPoints(0), VisiblePoints(0) {}
    };
    
    TMap<int32, FMeshVisibilityData> MeshVisibilityMap;
    
    // Pre-allocate with estimated capacity
    MeshVisibilityMap.Reserve(VisibleMeshIDs.Num());
    
    // First pass: count points per mesh
    for (const auto& Pair : CoverageMap)
    {
        const FVector& Point = Pair.Key;
        bool bIsVisible = Pair.Value;
        
        // OPTIMIZATION: Use spatial grid to find mesh
        FIntVector GridCell(
            FMath::FloorToInt(Point.X / SpatialGridSize),
            FMath::FloorToInt(Point.Y / SpatialGridSize),
            FMath::FloorToInt(Point.Z / SpatialGridSize)
        );
        
        if (SpatialGrid.Contains(GridCell))
        {
            for (int32 MeshIdx : SpatialGrid[GridCell])
            {
                const FMeshInfo& MeshInfo = SceneMeshes[MeshIdx];
                
                if (MeshInfo.Bounds.GetBox().IsInsideOrOn(Point))
                {
                    FMeshVisibilityData& VisData = MeshVisibilityMap.FindOrAdd(MeshInfo.MeshID);
                    VisData.TotalPoints++;
                    
                    if (bIsVisible)
                    {
                        VisData.VisiblePoints++;
                    }
                    
                    break; // Point can only be in one mesh
                }
            }
        }
    }
    
    // Calculate visible triangles
    int32 VisibleTriangles = 0;
    
    for (const FMeshInfo& MeshInfo : SceneMeshes)
    {
        if (VisibleMeshIDs.Contains(MeshInfo.MeshID))
        {
            // Get visibility data for this mesh
            if (MeshVisibilityMap.Contains(MeshInfo.MeshID))
            {
                const FMeshVisibilityData& VisData = MeshVisibilityMap[MeshInfo.MeshID];
                
                if (VisData.TotalPoints > 0)
                {
                    float MeshVisiblePointRatio = (float)VisData.VisiblePoints / (float)VisData.TotalPoints;
                    VisibleTriangles += FMath::RoundToInt(MeshInfo.NumTriangles * MeshVisiblePointRatio);
                }
            }
        }
    }
    
    // Update class members
    CurrentlyVisibleMeshIDs = VisibleMeshIDs;
    CurrentCoveragePercentage = CoveragePercentage;
    bCoverageVisualizationDirty = true;
    
    // Populate return data structure
    CoverageData.CoveragePercentage = CoveragePercentage;
    CoverageData.VisibleMeshIDs = VisibleMeshIDs;
    CoverageData.VisiblePoints = VisiblePoints;
    CoverageData.TotalVisibleTriangles = VisibleTriangles;
    
    UE_LOG(LogTemp, Display, TEXT("Coverage computed for camera %s: %.2f%% of points visible (%d/%d), %d visible meshes, ~%d visible triangles"),
        *CameraName, CoveragePercentage, VisiblePointCount, TotalPoints, VisibleMeshIDs.Num(), VisibleTriangles);
    
    if (bGridInitialized)
    {
        UpdateCoverageGrid();
    }
    
    return CoverageData;
}

FCoverageData ASceneAnalysisManager::ComputeCoverage(
    const FTransform& CameraTransform, const FString& CameraName)
{
    // Create array with single transform and call the multi-transform version
    TArray<FTransform> CameraTransforms;
    CameraTransforms.Add(CameraTransform);
    return ComputeCoverage(CameraTransforms, CameraName);
}

void ASceneAnalysisManager::InitializeCoverageVisualization()
{
    if (!World)
        return;
    
    // Initialize the procedural mesh component if it doesn't exist
    if (!CoverageVisualizationMesh)
    {
        CoverageVisualizationMesh = NewObject<UProceduralMeshComponent>(this);
        CoverageVisualizationMesh->RegisterComponent();
        CoverageVisualizationMesh->SetMobility(EComponentMobility::Movable);
        CoverageVisualizationMesh->AttachToComponent(GetRootComponent(), 
                                                   FAttachmentTransformRules::KeepWorldTransform);
        CoverageVisualizationMesh->SetCollisionEnabled(ECollisionEnabled::Type::NoCollision);
    }
    
    // Load or create the coverage material if not already set
    if (!CoverageMaterial)
    {
        // Try to load the coverage material
        CoverageMaterial = LoadObject<UMaterialInterface>(nullptr, 
            TEXT("/Script/Engine.Material'/VCCSim/Materials/M_Coverage.M_Coverage'"));
        
        // Only log error if material failed to load
        if (!CoverageMaterial)
        {
            UE_LOG(LogTemp, Error, TEXT("InitializeCoverageVisualization: "
                                       "Failed to load coverage material."));
        }
    }
}

void ASceneAnalysisManager::VisualizeCoverage(bool bShow)
{
    if (!World)
        return;
    
    // Check if we have coverage data
    if (!bGridInitialized)
    {
        UE_LOG(LogTemp, Warning,
            TEXT("VisualizeCoverage: No coverage grid initialized"));
        return;
    }
    
    // If not showing, clear visualization and return
    if (!bShow)
    {
        if (CoverageVisualizationMesh)
        {
            CoverageVisualizationMesh->SetVisibility(false);
        }
        return;
    }
    
    // Initialize visualization components if needed
    InitializeCoverageVisualization();
    
    // If mesh component failed to initialize, return
    if (!CoverageVisualizationMesh)
    {
        UE_LOG(LogTemp, Error, TEXT("VisualizeCoverage: Coverage "
                                    "mesh component not initialized"));
        return;
    }
    
    // Only update mesh if dirty or visibility is changing
    if (bCoverageVisualizationDirty)
    {
        CreateCoverageMesh();
        bCoverageVisualizationDirty = false;
    }
    
    // Set visibility
    CoverageVisualizationMesh->SetVisibility(true);
}

void ASceneAnalysisManager::ClearCoverageVisualization()
{
    if (CoverageVisualizationMesh)
    {
        CoverageVisualizationMesh->ClearAllMeshSections();
        CoverageVisualizationMesh->SetVisibility(false);
    }
}

// Update the ResetCoverage function to initialize the point arrays
void ASceneAnalysisManager::ResetCoverage()
{
    CoverageMap.Empty();
    CurrentlyVisibleMeshIDs.Empty();
    CurrentCoveragePercentage = 0.0f;
    VisiblePoints.Empty();
    InvisiblePoints.Empty();
    
    // Initialize coverage map with all points set to not visible
    for (const FMeshInfo& MeshInfo : SceneMeshes)
    {
        TArray<FVector> SampledPoints = SamplePointsOnMesh(MeshInfo);
        for (const FVector& Point : SampledPoints)
        {
            CoverageMap.Add(Point, false);
            InvisiblePoints.Add(Point);
        }
    }
    
    // Initialize the unified grid
    InitializeUnifiedGrid();
    
    bCoverageVisualizationDirty = true;
}

void ASceneAnalysisManager::ExtractMeshData(
    UStaticMeshComponent* MeshComponent, FMeshInfo& OutMeshInfo)
{
    if (!MeshComponent || !MeshComponent->GetStaticMesh())
        return;
    
    UStaticMesh* StaticMesh = MeshComponent->GetStaticMesh();
    
    // Basic mesh info
    OutMeshInfo.MeshID = MeshComponent->GetUniqueID();
    OutMeshInfo.MeshName = MeshComponent->GetName();
    OutMeshInfo.Mesh = StaticMesh;
    OutMeshInfo.Transform = MeshComponent->GetComponentTransform();
    OutMeshInfo.Bounds = MeshComponent->Bounds;
    OutMeshInfo.bIsVisible = MeshComponent->IsVisible();
    
    // Get mesh data
    if (StaticMesh->GetRenderData() && StaticMesh->GetRenderData()->LODResources.Num() > 0)
    {
        const FStaticMeshLODResources& LODModel = StaticMesh->GetRenderData()->LODResources[0];
        
        // Get vertices and indices
        OutMeshInfo.NumVertices = LODModel.VertexBuffers.PositionVertexBuffer.GetNumVertices();
        OutMeshInfo.NumTriangles = LODModel.IndexBuffer.GetNumIndices() / 3;
        
        // Extract vertex positions
        OutMeshInfo.VertexPositions.Reserve(OutMeshInfo.NumVertices);
        for (int32 VertIdx = 0; VertIdx < OutMeshInfo.NumVertices; ++VertIdx)
        {
            // Get the FVector3f from the vertex buffer
            FVector3f VertexPos3f = LODModel.VertexBuffers.
            PositionVertexBuffer.VertexPosition(VertIdx);
            
            // Convert to FVector (explicit conversion)
            FVector VertexPos(VertexPos3f.X, VertexPos3f.Y, VertexPos3f.Z);
            
            // Transform to world space
            VertexPos = OutMeshInfo.Transform.TransformPosition(VertexPos);
            OutMeshInfo.VertexPositions.Add(VertexPos);
        }
        
        // Extract indices
        OutMeshInfo.Indices.Reserve(LODModel.IndexBuffer.GetNumIndices());
        for (int32 IndexIdx = 0; IndexIdx < LODModel.IndexBuffer.GetNumIndices(); ++IndexIdx)
        {
            OutMeshInfo.Indices.Add(LODModel.IndexBuffer.GetIndex(IndexIdx));
        }
    }
}

FIntVector ASceneAnalysisManager::WorldToGridCoordinates(const FVector& WorldPos) const
{
    return FIntVector(
        FMath::FloorToInt((WorldPos.X - GridOrigin.X) / GridResolution),
        FMath::FloorToInt((WorldPos.Y - GridOrigin.Y) / GridResolution),
        FMath::FloorToInt((WorldPos.Z - GridOrigin.Z) / GridResolution)
    );
}

void ASceneAnalysisManager::InitializeUnifiedGrid()
{
    if (!World || SceneMeshes.Num() == 0)
        return;
    
    // Calculate bounds of the scene
    FBox SceneBounds(EForceInit::ForceInit);
    for (const FMeshInfo& MeshInfo : SceneMeshes)
    {
        SceneBounds += MeshInfo.Bounds.GetBox();
    }
    
    // Expand bounds slightly
    SceneBounds = SceneBounds.ExpandBy(GridResolution * 2.0f);
    ExpandedSceneBounds = SceneBounds;
    
    // Store grid origin
    GridOrigin = SceneBounds.Min;
    
    FVector BoundsSize = SceneBounds.GetSize();
    int32 GridSizeX = FMath::Max(1, FMath::CeilToInt(BoundsSize.X / GridResolution));
    int32 GridSizeY = FMath::Max(1, FMath::CeilToInt(BoundsSize.Y / GridResolution));
    int32 GridSizeZ = FMath::Max(1, FMath::CeilToInt(BoundsSize.Z / GridResolution));
    
    GridSize = FVector(GridSizeX, GridSizeY, GridSizeZ);
    
    // Clear the unified grid (sparse structure)
    UnifiedGrid.Empty();
    
    // Pre-populate the grid with cells that contain sampled points
    for (const auto& Pair : CoverageMap)
    {
        const FVector& Point = Pair.Key;
        FIntVector GridCoords = WorldToGridCoordinates(Point);
        
        // Add cell with no coverage yet
        FUnifiedGridCell& Cell = UnifiedGrid.FindOrAdd(GridCoords);
        Cell.TotalPoints++;
        Cell.bIsSafe = false; // Default to safe
    }
    
    bGridInitialized = true;
    
    UE_LOG(LogTemp, Display, TEXT("Unified grid initialized: "
                                  "theoretical grid %dx%dx%d, actual populated cells: %d"), 
           GridSizeX, GridSizeY, GridSizeZ, UnifiedGrid.Num());
}

void ASceneAnalysisManager::GenerateSafeZone(
    const float& SafeDistance, const float& SafeHeight)
{
    // Try to find a valid world
    if (GEngine)
    {
        for (const FWorldContext& Context : GEngine->GetWorldContexts())
        {
            if (Context.WorldType == EWorldType::Game || Context.WorldType == EWorldType::PIE)
            {
                World = Context.World();
            }
        }
    }
    
    if (!World)
    {
        UE_LOG(LogTemp, Error, TEXT("GenerateSafeZone: No valid World found"));
        return;
    }
    
    // Get scene mesh data
    if (SceneMeshes.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("GenerateSafeZone: No meshes in scene"));
        return;
    }
    
    // Initialize grid if not done already
    if (!bGridInitialized)
    {
        InitializeUnifiedGrid();
    }
    
    // Mark all cells as safe initially
    for (auto& Pair : UnifiedGrid)
    {
        Pair.Value.bIsSafe = true;
    }
    
    // Calculate expanded scene bounds for safety
    FBox SceneBounds = ExpandedSceneBounds;
    FVector BoundsMin = SceneBounds.Min;
    FVector BoundsSize = SceneBounds.GetSize();
    
    int32 NumX = FMath::Max(1, FMath::CeilToInt(BoundsSize.X / GridResolution));
    int32 NumY = FMath::Max(1, FMath::CeilToInt(BoundsSize.Y / GridResolution));
    int32 NumZ = FMath::Max(1, FMath::CeilToInt(BoundsSize.Z / GridResolution));
    
    UE_LOG(LogTemp, Display, TEXT("Generating safe zone grid: %dx%dx%d "
                                  "with cell size %.2f"), NumX, NumY, NumZ, GridResolution);
    
    // OPTIMIZATION: Use parallel processing for large meshes
    const bool bUseParallelProcessing = SceneMeshes.Num() > 10 || TotalTrianglesInScene > 100000;
    FCriticalSection CriticalSection;
    
    // For each mesh in the scene
    for (const FMeshInfo& MeshInfo : SceneMeshes)
    {
        // Process triangles in parallel for large meshes
        if (bUseParallelProcessing && MeshInfo.Indices.Num() > 30000)
        {
            ParallelFor(MeshInfo.Indices.Num() / 3, [&](int32 TriangleIndex)
            {
                int32 i = TriangleIndex * 3;
                if (i + 2 >= MeshInfo.Indices.Num()) return;
                
                // Get triangle vertices
                const FVector& V0 = MeshInfo.VertexPositions[MeshInfo.Indices[i]];
                const FVector& V1 = MeshInfo.VertexPositions[MeshInfo.Indices[i + 1]];
                const FVector& V2 = MeshInfo.VertexPositions[MeshInfo.Indices[i + 2]];
                
                // Calculate triangle bounds
                FBox TriBounds(EForceInit::ForceInit);
                TriBounds += V0;
                TriBounds += V1;
                TriBounds += V2;
                
                // Expand bounds by safe distance
                TriBounds = TriBounds.ExpandBy(FVector(SafeDistance,
                    SafeDistance, SafeHeight));
                
                // Convert to grid coordinates
                int32 MinX = FMath::Max(0, FMath::FloorToInt(
                    (TriBounds.Min.X - BoundsMin.X) / GridResolution));
                int32 MinY = FMath::Max(0, FMath::FloorToInt(
                    (TriBounds.Min.Y - BoundsMin.Y) / GridResolution));
                int32 MinZ = FMath::Max(0, FMath::FloorToInt(
                    (TriBounds.Min.Z - BoundsMin.Z) / GridResolution));
                
                int32 MaxX = FMath::Min(NumX - 1, FMath::CeilToInt(
                    (TriBounds.Max.X - BoundsMin.X) / GridResolution));
                int32 MaxY = FMath::Min(NumY - 1, FMath::CeilToInt(
                    (TriBounds.Max.Y - BoundsMin.Y) / GridResolution));
                int32 MaxZ = FMath::Min(NumZ - 1, FMath::CeilToInt(
                    (TriBounds.Max.Z - BoundsMin.Z) / GridResolution));
                
                // Process cells that need checking
                TArray<FIntVector> CellsToUpdate;
                
                // Check each cell in the expanded bounds
                for (int32 X = MinX; X <= MaxX; ++X)
                {
                    for (int32 Y = MinY; Y <= MaxY; ++Y)
                    {
                        for (int32 Z = MinZ; Z <= MaxZ; ++Z)
                        {
                            FIntVector GridCoords(X, Y, Z);
                            
                            // Calculate cell center
                            FVector CellCenter(
                                BoundsMin.X + (X + 0.5f) * GridResolution,
                                BoundsMin.Y + (Y + 0.5f) * GridResolution,
                                BoundsMin.Z + (Z + 0.5f) * GridResolution
                            );
                            
                            // Find closest point on triangle
                            FVector ClosestPoint = FMath::ClosestPointOnTriangleToPoint(
                                CellCenter, V0, V1, V2);
                            
                            // Check horizontal distance (XY plane)
                            FVector2D CellCenter2D(CellCenter.X, CellCenter.Y);
                            FVector2D ClosestPoint2D(ClosestPoint.X, ClosestPoint.Y);
                            float HorizontalDist = FVector2D::Distance(CellCenter2D, ClosestPoint2D);
                            
                            // Check vertical distance (Z axis)
                            float VerticalDist = FMath::Abs(CellCenter.Z - ClosestPoint.Z);
                            
                            // Mark as unsafe if too close in both horizontal and vertical directions
                            if (HorizontalDist < SafeDistance && VerticalDist < SafeHeight)
                            {
                                CellsToUpdate.Add(GridCoords);
                            }
                        }
                    }
                }
                // Update grid cells in a thread-safe manner
                if (CellsToUpdate.Num() > 0)
                {
                    FScopeLock Lock(&CriticalSection);
                    for (const FIntVector& Coords : CellsToUpdate)
                    {
                        FUnifiedGridCell& Cell = UnifiedGrid.FindOrAdd(Coords);
                        Cell.bIsSafe = false;
                    }
                }
            });
        }
        else
        {
            // Sequential processing for smaller meshes
            for (int32 i = 0; i < MeshInfo.Indices.Num(); i += 3)
            {
                if (i + 2 >= MeshInfo.Indices.Num()) continue;
                
                // Get triangle vertices
                const FVector& V0 = MeshInfo.VertexPositions[MeshInfo.Indices[i]];
                const FVector& V1 = MeshInfo.VertexPositions[MeshInfo.Indices[i + 1]];
                const FVector& V2 = MeshInfo.VertexPositions[MeshInfo.Indices[i + 2]];
                
                // Calculate triangle bounds
                FBox TriBounds(EForceInit::ForceInit);
                TriBounds += V0;
                TriBounds += V1;
                TriBounds += V2;
                
                // Expand bounds by safe distance
                TriBounds = TriBounds.ExpandBy(FVector(SafeDistance,
                    SafeDistance, SafeHeight));
                
                // Convert to grid coordinates
                int32 MinX = FMath::Max(0, FMath::FloorToInt(
                    (TriBounds.Min.X - BoundsMin.X) / GridResolution));
                int32 MinY = FMath::Max(0, FMath::FloorToInt(
                    (TriBounds.Min.Y - BoundsMin.Y) / GridResolution));
                int32 MinZ = FMath::Max(0, FMath::FloorToInt(
                    (TriBounds.Min.Z - BoundsMin.Z) / GridResolution));
                
                int32 MaxX = FMath::Min(NumX - 1, FMath::CeilToInt(
                    (TriBounds.Max.X - BoundsMin.X) / GridResolution));
                int32 MaxY = FMath::Min(NumY - 1, FMath::CeilToInt(
                    (TriBounds.Max.Y - BoundsMin.Y) / GridResolution));
                int32 MaxZ = FMath::Min(NumZ - 1, FMath::CeilToInt(
                    (TriBounds.Max.Z - BoundsMin.Z) / GridResolution));
                
                // OPTIMIZATION: Use bounding box check first
                for (int32 X = MinX; X <= MaxX; ++X)
                {
                    for (int32 Y = MinY; Y <= MaxY; ++Y)
                    {
                        for (int32 Z = MinZ; Z <= MaxZ; ++Z)
                        {
                            FIntVector GridCoords(X, Y, Z);
                            
                            // Find or add cell to unified grid
                            FUnifiedGridCell& Cell = UnifiedGrid.FindOrAdd(GridCoords);
                            
                            // Skip if already marked unsafe (optimization)
                            if (!Cell.bIsSafe) continue;
                            
                            // Calculate cell center
                            FVector CellCenter(
                                BoundsMin.X + (X + 0.5f) * GridResolution,
                                BoundsMin.Y + (Y + 0.5f) * GridResolution,
                                BoundsMin.Z + (Z + 0.5f) * GridResolution
                            );
                            
                            // Find closest point on triangle
                            FVector ClosestPoint = FMath::ClosestPointOnTriangleToPoint(
                                CellCenter, V0, V1, V2);
                            
                            // Check horizontal distance (XY plane)
                            FVector2D CellCenter2D(CellCenter.X, CellCenter.Y);
                            FVector2D ClosestPoint2D(ClosestPoint.X, ClosestPoint.Y);
                            float HorizontalDist = FVector2D::Distance(CellCenter2D, ClosestPoint2D);
                            
                            // Check vertical distance (Z axis)
                            float VerticalDist = FMath::Abs(CellCenter.Z - ClosestPoint.Z);
                            
                            // Mark as unsafe if too close in both horizontal and vertical directions
                            if (HorizontalDist < SafeDistance && VerticalDist < SafeHeight)
                            {
                                Cell.bIsSafe = false;
                            }
                        }
                    }
                }
            }
        }
    }

    UE_LOG(LogTemp, Display, TEXT("Safe zone generated: Over"));
    bSafeZoneDirty = true;
}

void ASceneAnalysisManager::InitializeSafeZoneVisualization()
{
    if (!World)
        return;
    
    // Initialize the procedural mesh component if it doesn't exist
    if (!SafeZoneVisualizationMesh)
    {
        SafeZoneVisualizationMesh = NewObject<UProceduralMeshComponent>(this);
        SafeZoneVisualizationMesh->RegisterComponent();
        SafeZoneVisualizationMesh->SetMobility(EComponentMobility::Movable);
        SafeZoneVisualizationMesh->AttachToComponent(GetRootComponent(), 
                                                  FAttachmentTransformRules::KeepWorldTransform);
        SafeZoneVisualizationMesh->SetCollisionEnabled(ECollisionEnabled::Type::NoCollision);
    }
    
    // Load or create the safe zone material if not already set
    if (!SafeZoneMaterial)
    {
        // Try to load the safe zone material
        SafeZoneMaterial = LoadObject<UMaterialInterface>(nullptr, 
            TEXT("/Script/Engine.Material'/VCCSim/Materials/M_SafeZone.M_SafeZone'"));
        
        if (!SafeZoneMaterial)
        {
            UE_LOG(LogTemp, Error, TEXT("InitializeSafeZoneVisualization: "
                                       "Failed to load safe zone material."));
        }
    }
}

void ASceneAnalysisManager::VisualizeSafeZone(bool Vis)
{
    if (!World)
        return;
    
    // Check if we have a valid grid
    if (!bGridInitialized)
    {
        UE_LOG(LogTemp, Warning, TEXT("VisualizeSafeZone: No grid initialized"));
        return;
    }
    
    // If not showing, clear visualization and return
    if (!Vis)
    {
        if (SafeZoneVisualizationMesh)
        {
            SafeZoneVisualizationMesh->SetVisibility(false);
        }
        return;
    }
    
    // Initialize visualization components if needed
    InitializeSafeZoneVisualization();
    
    // If mesh component failed to initialize, return
    if (!SafeZoneVisualizationMesh)
    {
        UE_LOG(LogTemp, Error, TEXT("VisualizeSafeZone: Safe zone "
                                   "mesh component not initialized"));
        return;
    }
    
    // Only update mesh if dirty or visibility is changing
    if (bSafeZoneDirty)
    {
        CreateSafeZoneMesh();
        bSafeZoneDirty = false;
    }
    
    // Set visibility
    SafeZoneVisualizationMesh->SetVisibility(true);
}

void ASceneAnalysisManager::ClearSafeZoneVisualization()
{
    if (SafeZoneVisualizationMesh)
    {
        SafeZoneVisualizationMesh->ClearAllMeshSections();
        SafeZoneVisualizationMesh->SetVisibility(false);
    }
}

bool ASceneAnalysisManager::IsPointVisibleFromCamera(
    const FVector& Point, const FTransform& CameraPose) const
{
    if (!World)
    {
        return false;
    }
    
    // Get camera position and forward vector (optimized)
    const FVector CameraPos = CameraPose.GetLocation();
    const FVector CameraForward = CameraPose.GetRotation().GetForwardVector();
    
    // Check if point is in front of camera (dot product optimization)
    const FVector PointDir = (Point - CameraPos).GetSafeNormal();
    
    if (FVector::DotProduct(CameraForward, PointDir) <= 0.0f)
        return false; // Point is behind camera
    
    // Use pre-allocated trace params for better performance
    static FCollisionQueryParams TraceParams;
    static bool bTraceParamsInitialized = false;
    
    if (!bTraceParamsInitialized)
    {
        TraceParams.bTraceComplex = true;
        bTraceParamsInitialized = true;
    }
    
    // Trace from camera to point
    FHitResult HitResult;
    
    if (World->LineTraceSingleByChannel(
            HitResult,
            CameraPos,
            Point,
            ECC_Visibility,
            TraceParams))
    {
        // If we hit something near our target point, consider it visible
        // Optimized by using SizeSquared() instead of Size()
        return (HitResult.Location - Point).SizeSquared() < 1.0f;
    }
    
    // If nothing was hit, point is visible
    return true;
}

void ASceneAnalysisManager::ConstructFrustum(
    FConvexVolume& OutFrustum, const FTransform& CameraPose,
    const FMatrix44f& CameraIntrinsic)
{
    // Cache frequently used values
    const float fx = CameraIntrinsic.M[0][0];
    const float fy = CameraIntrinsic.M[1][1];
    const float cx = CameraIntrinsic.M[0][2];
    const float cy = CameraIntrinsic.M[1][2];
    
    // Calculate image width and height from principal points (assuming centered)
    const float width = cx * 2.0f;
    const float height = cy * 2.0f;
    
    // Calculate FOV correctly and cache reciprocals for efficiency
    const float fxRecip = (fx > KINDA_SMALL_NUMBER) ? 1.0f / fx : 0.0f;
    const float fyRecip = (fy > KINDA_SMALL_NUMBER) ? 1.0f / fy : 0.0f;
    
    float HorizontalFOV = (fx > KINDA_SMALL_NUMBER) ? 2.0f * FMath::Atan(width * 0.5f * fxRecip) : FMath::DegreesToRadians(90.0f);
    float VerticalFOV = (fy > KINDA_SMALL_NUMBER) ? 2.0f * FMath::Atan(height * 0.5f * fyRecip) : FMath::DegreesToRadians(60.0f);
    float AspectRatio = width / height;
    
    // If values are extreme, use reasonable defaults
    if (FMath::IsNaN(HorizontalFOV) || HorizontalFOV < FMath::DegreesToRadians(1.0f)) {
        HorizontalFOV = FMath::DegreesToRadians(90.0f);
    }
    
    // Extract camera transform components once
    const FVector ForwardVector = CameraPose.GetRotation().GetForwardVector();
    const FVector RightVector = CameraPose.GetRotation().GetRightVector();
    const FVector UpVector = CameraPose.GetRotation().GetUpVector();
    const FVector Position = CameraPose.GetLocation();
    
    // Use constants for near and far plane distances
    constexpr float NearPlaneDistance = 10.0f;
    constexpr float FarPlaneDistance = 5000.0f;
    
    // Pre-calculate common values
    const float HalfVFOV = VerticalFOV * 0.5f;
    const float FarHeight = FarPlaneDistance * FMath::Tan(HalfVFOV);
    const float FarWidth = FarHeight * AspectRatio;
    
    // Calculate plane centers once
    const FVector NearCenter = Position + ForwardVector * NearPlaneDistance;
    const FVector FarCenter = Position + ForwardVector * FarPlaneDistance;
    
    // Calculate far plane corners efficiently
    const FVector UpScaled = UpVector * FarHeight;
    const FVector RightScaled = RightVector * FarWidth;
    
    const FVector FarTopLeft = FarCenter + UpScaled - RightScaled;
    const FVector FarTopRight = FarCenter + UpScaled + RightScaled;
    const FVector FarBottomLeft = FarCenter - UpScaled - RightScaled;
    const FVector FarBottomRight = FarCenter - UpScaled + RightScaled;
    
    // Pre-allocate planes array with exact capacity
    OutFrustum.Planes.Empty(6);
    OutFrustum.Planes.Reserve(6);
    
    // Near plane (normal points backward)
    OutFrustum.Planes.Add(FPlane(NearCenter, ForwardVector));
    
    // Far plane (normal points forward)
    OutFrustum.Planes.Add(FPlane(FarCenter, -ForwardVector));
    
    // Calculate plane normals more efficiently
    FVector LeftNormal = FVector::CrossProduct(FarTopLeft - Position, FarBottomLeft - Position).GetSafeNormal();
    OutFrustum.Planes.Add(FPlane(Position, -LeftNormal));
    
    FVector RightNormal = FVector::CrossProduct(FarBottomRight - Position, FarTopRight - Position).GetSafeNormal();
    OutFrustum.Planes.Add(FPlane(Position, -RightNormal));
    
    FVector TopNormal = FVector::CrossProduct(FarTopRight - Position, FarTopLeft - Position).GetSafeNormal();
    OutFrustum.Planes.Add(FPlane(Position, -TopNormal));
    
    FVector BottomNormal = FVector::CrossProduct(FarBottomLeft - Position, FarBottomRight - Position).GetSafeNormal();
    OutFrustum.Planes.Add(FPlane(Position, -BottomNormal));
}

TArray<FVector> ASceneAnalysisManager::SamplePointsOnMesh(const FMeshInfo& MeshInfo)
{
    // Pre-allocate result array based on whether we're using vertex sampling
    TArray<FVector> SampledPoints;
    
    if (bUseVertexSampling)
    {
        // When using vertex sampling, just return a copy of vertex positions
        return MeshInfo.VertexPositions;
    }
    
    // Calculate total triangle area for more accurate pre-allocation
    float TotalArea = 0.0f;
    for (int32 i = 0; i < MeshInfo.Indices.Num(); i += 3)
    {
        if (i + 2 < MeshInfo.Indices.Num())
        {
            const FVector& V0 = MeshInfo.VertexPositions[MeshInfo.Indices[i]];
            const FVector& V1 = MeshInfo.VertexPositions[MeshInfo.Indices[i + 1]];
            const FVector& V2 = MeshInfo.VertexPositions[MeshInfo.Indices[i + 2]];
            
            // Calculate triangle area
            float TriangleArea = 0.5f * FVector::CrossProduct(V1 - V0, V2 - V0).Size() / 10000.0f;
            TotalArea += TriangleArea;
        }
    }
    
    // Estimate total number of samples
    int32 EstimatedSampleCount = MeshInfo.Indices.Num() +
        FMath::Max(1, FMath::RoundToInt(TotalArea * SamplingDensity));
    SampledPoints.Reserve(EstimatedSampleCount);
    
    // Process triangles in batches for better cache coherency
    const int32 BatchSize = 64;
    const int32 NumTriangles = MeshInfo.Indices.Num() / 3;
    const int32 NumBatches = FMath::DivideAndRoundUp(NumTriangles, BatchSize);
    
    // Pre-compute random values for barycentric coordinates
    const int32 MaxSamplesPerTriangle = FMath::Max(10, SamplingDensity);
    TArray<float> RandomValues;
    RandomValues.Reserve(MaxSamplesPerTriangle * 2);
    
    for (int32 i = 0; i < MaxSamplesPerTriangle * 2; ++i)
    {
        RandomValues.Add(FMath::SRand());
    }
    
    for (int32 BatchIdx = 0; BatchIdx < NumBatches; ++BatchIdx)
    {
        int32 StartTriangle = BatchIdx * BatchSize;
        int32 EndTriangle = FMath::Min(StartTriangle + BatchSize, NumTriangles);
        
        for (int32 TriIdx = StartTriangle; TriIdx < EndTriangle; ++TriIdx)
        {
            int32 BaseIdx = TriIdx * 3;
            
            // Skip if invalid indices
            if (BaseIdx + 2 >= MeshInfo.Indices.Num())
                continue;
            
            const FVector& V0 = MeshInfo.VertexPositions[MeshInfo.Indices[BaseIdx]];
            const FVector& V1 = MeshInfo.VertexPositions[MeshInfo.Indices[BaseIdx + 1]];
            const FVector& V2 = MeshInfo.VertexPositions[MeshInfo.Indices[BaseIdx + 2]];
            
            // Add triangle vertices
            SampledPoints.Add(V0);
            SampledPoints.Add(V1);
            SampledPoints.Add(V2);
            
            // Calculate triangle area efficiently
            FVector Cross = FVector::CrossProduct(V1 - V0, V2 - V0);
            float TriangleArea = 0.5f * Cross.Size() / 10000.0f;
            
            // Calculate number of samples based on SamplingDensity and triangle area
            int32 NumSamples = FMath::Max(1, FMath::RoundToInt(TriangleArea * SamplingDensity));
            NumSamples = FMath::Min(NumSamples, MaxSamplesPerTriangle);
            
            // Add additional samples within the triangle using pre-computed random values
            for (int32 SampleIdx = 0; SampleIdx < NumSamples; ++SampleIdx)
            {
                // Use pre-computed random values
                float r1 = RandomValues[SampleIdx * 2 % RandomValues.Num()];
                float r2 = RandomValues[(SampleIdx * 2 + 1) % RandomValues.Num()];
                
                // Convert to barycentric coordinates - use sqrt for better distribution
                float u = 1.0f - FMath::Sqrt(r1);
                float v = r2 * FMath::Sqrt(r1);
                float w = 1.0f - u - v;
                
                // Compute point using barycentric coordinates
                SampledPoints.Add(u * V0 + v * V1 + w * V2);
            }
        }
    }
    
    return SampledPoints;
}

void ASceneAnalysisManager::UpdateCoverageGrid()
{
    if (!bGridInitialized)
        return;
    
    // Reset visible points and coverage in existing cells
    ParallelFor(UnifiedGrid.Num(), [this](int32 Index)
    {
        auto It = UnifiedGrid.CreateIterator();
        for (int32 i = 0; i < Index; ++i)
        {
            ++It;
        }
        
        It.Value().VisiblePoints = 0;
        It.Value().Coverage = 0.0f;
        // Note: we don't reset safe zone data here
    });
    
    // Create a spatial hash structure for faster grid cell lookup
    struct FGridCellInfo
    {
        FIntVector Coords;
        int32 VisiblePoints;
        
        FGridCellInfo(const FIntVector& InCoords) : Coords(InCoords), VisiblePoints(0) {}
    };
    
    // Calculate the number of cells in each dimension for the hash
    constexpr int32 HashGridSize = 10; // Adjust based on scene scale
    TMap<FIntVector, TArray<FGridCellInfo*>> GridHashMap;
    
    // Create a thread-local copy of cell data to avoid lock contention
    TMap<FIntVector, int32> LocalCellVisibility;
    LocalCellVisibility.Reserve(UnifiedGrid.Num());
    
    // Process points in parallel batches
    const int32 BatchSize = 1000;
    const int32 NumBatches = FMath::DivideAndRoundUp(CoverageMap.Num(), BatchSize);
    
    ParallelFor(NumBatches, [&](int32 BatchIndex)
    {
        // Set up batch processing range
        auto It = CoverageMap.CreateConstIterator();
        for (int32 i = 0; i < BatchIndex * BatchSize && It; ++i, ++It) {}
        
        // Process this batch
        TMap<FIntVector, int32> ThreadLocalVisibility;
        
        for (int32 i = 0; i < BatchSize && It; ++i, ++It)
        {
            const FVector& Point = It.Key();
            bool bIsVisible = It.Value();
            
            // Convert world position to grid coordinates
            FIntVector GridCoords = WorldToGridCoordinates(Point);
            
            // If point is visible, increment local counter
            if (bIsVisible)
            {
                ThreadLocalVisibility.FindOrAdd(GridCoords)++;
            }
        }
        
        // Merge thread-local results with global map using atomic operations
        FCriticalSection CriticalSection;
        FScopeLock Lock(&CriticalSection);
        
        for (const auto& Pair : ThreadLocalVisibility)
        {
            LocalCellVisibility.FindOrAdd(Pair.Key) += Pair.Value;
        }
    });
    
    // Update the grid with local visibility data
    for (const auto& Pair : LocalCellVisibility)
    {
        const FIntVector& GridCoords = Pair.Key;
        int32 VisiblePointsCount = Pair.Value;
        
        if (UnifiedGrid.Contains(GridCoords))
        {
            FUnifiedGridCell& Cell = UnifiedGrid[GridCoords];
            Cell.VisiblePoints = VisiblePointsCount;
            
            // Calculate coverage
            if (Cell.TotalPoints > 0)
            {
                Cell.Coverage = (float)Cell.VisiblePoints / (float)Cell.TotalPoints;
            }
        }
    }
}

void ASceneAnalysisManager::CreateCoverageMesh()
{
    // Clear existing mesh sections
    CoverageVisualizationMesh->ClearAllMeshSections();
    
    // Create arrays for procedural mesh
    TArray<FVector> Vertices;
    TArray<int32> Triangles;
    TArray<FVector> Normals;
    TArray<FVector2D> UV0;
    TArray<FColor> VertexColors;
    TArray<FProcMeshTangent> Tangents;
    
    // Pre-allocate memory based on number of cells with sample points
    int32 EstimatedCells = UnifiedGrid.Num();
    Vertices.Reserve(EstimatedCells * 8);
    Triangles.Reserve(EstimatedCells * 36);
    Normals.Reserve(EstimatedCells * 8);
    UV0.Reserve(EstimatedCells * 8);
    VertexColors.Reserve(EstimatedCells * 8);
    Tangents.Reserve(EstimatedCells * 8);
    
    // Create cubes only for cells with sample points and coverage above threshold
    int32 NumCellsVisualized = 0;
    for (const auto& Pair : UnifiedGrid)
    {
        const FIntVector& GridCoords = Pair.Key;
        const FUnifiedGridCell& Cell = Pair.Value;
        
        // Skip cells with no sample points or low coverage
        if (Cell.TotalPoints == 0 || Cell.Coverage < VisualizationThreshold)
            continue;
        
        // Calculate cell center in world space
        FVector CellCenter(
            GridOrigin.X + (GridCoords.X + 0.5f) * GridResolution,
            GridOrigin.Y + (GridCoords.Y + 0.5f) * GridResolution,
            GridOrigin.Z + (GridCoords.Z + 0.5f) * GridResolution
        );
        
        // Calculate cell size (slightly smaller than grid resolution to see cell boundaries)
        float CellSize = GridResolution * 0.9f;
        float HalfSize = CellSize * 0.5f;
        
        // Convert coverage to color (green = 1.0, red = 0.0)
        FColor CellColor = FLinearColor::LerpUsingHSV(
            FLinearColor(1.0f, 0.0f, 0.0f), // Red (0% coverage)
            FLinearColor(0.0f, 1.0f, 0.0f), // Green (100% coverage)
            Cell.Coverage
        ).ToFColor(false);
        
        // Adjust alpha based on coverage
        CellColor.A = 128 + FMath::FloorToInt(127.0f * Cell.Coverage); // 128-255 range for alpha
        
        // Add a cube for this cell
        int32 BaseVertexIndex = Vertices.Num();
        
        // Define the 8 corners of the cube
        Vertices.Add(CellCenter + FVector(-HalfSize, -HalfSize, -HalfSize)); // 0: bottom left back
        Vertices.Add(CellCenter + FVector(HalfSize, -HalfSize, -HalfSize));  // 1: bottom right back
        Vertices.Add(CellCenter + FVector(HalfSize, HalfSize, -HalfSize));   // 2: bottom right front
        Vertices.Add(CellCenter + FVector(-HalfSize, HalfSize, -HalfSize));  // 3: bottom left front
        Vertices.Add(CellCenter + FVector(-HalfSize, -HalfSize, HalfSize));  // 4: top left back
        Vertices.Add(CellCenter + FVector(HalfSize, -HalfSize, HalfSize));   // 5: top right back
        Vertices.Add(CellCenter + FVector(HalfSize, HalfSize, HalfSize));    // 6: top right front
        Vertices.Add(CellCenter + FVector(-HalfSize, HalfSize, HalfSize));   // 7: top left front
        
        // Add colors for all 8 vertices
        for (int32 i = 0; i < 8; ++i)
        {
            VertexColors.Add(CellColor);
        }
        
        // Add texture coordinates
        UV0.Add(FVector2D(0, 0)); // 0
        UV0.Add(FVector2D(1, 0)); // 1
        UV0.Add(FVector2D(1, 1)); // 2
        UV0.Add(FVector2D(0, 1)); // 3
        UV0.Add(FVector2D(0, 0)); // 4
        UV0.Add(FVector2D(1, 0)); // 5
        UV0.Add(FVector2D(1, 1)); // 6
        UV0.Add(FVector2D(0, 1)); // 7
        
        // Add normals
        Normals.Add(FVector(-1, -1, -1).GetSafeNormal()); // 0
        Normals.Add(FVector(1, -1, -1).GetSafeNormal());  // 1
        Normals.Add(FVector(1, 1, -1).GetSafeNormal());   // 2
        Normals.Add(FVector(-1, 1, -1).GetSafeNormal());  // 3
        Normals.Add(FVector(-1, -1, 1).GetSafeNormal());  // 4
        Normals.Add(FVector(1, -1, 1).GetSafeNormal());   // 5
        Normals.Add(FVector(1, 1, 1).GetSafeNormal());    // 6
        Normals.Add(FVector(-1, 1, 1).GetSafeNormal());   // 7
        
        // Add tangents (simplified)
        for (int32 i = 0; i < 8; ++i)
        {
            Tangents.Add(FProcMeshTangent(1, 0, 0));
        }
        
        // Add triangles for each face (12 triangles total)
        // Bottom face (0,1,2,3)
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 1);
        Triangles.Add(BaseVertexIndex + 2);
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 2);
        Triangles.Add(BaseVertexIndex + 3);
        
        // Top face (4,5,6,7)
        Triangles.Add(BaseVertexIndex + 4);
        Triangles.Add(BaseVertexIndex + 6);
        Triangles.Add(BaseVertexIndex + 5);
        Triangles.Add(BaseVertexIndex + 4);
        Triangles.Add(BaseVertexIndex + 7);
        Triangles.Add(BaseVertexIndex + 6);
        
        // Front face (3,2,6,7)
        Triangles.Add(BaseVertexIndex + 3);
        Triangles.Add(BaseVertexIndex + 2);
        Triangles.Add(BaseVertexIndex + 6);
        Triangles.Add(BaseVertexIndex + 3);
        Triangles.Add(BaseVertexIndex + 6);
        Triangles.Add(BaseVertexIndex + 7);
        
        // Back face (0,1,5,4)
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 5);
        Triangles.Add(BaseVertexIndex + 1);
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 4);
        Triangles.Add(BaseVertexIndex + 5);
        
        // Left face (0,3,7,4)
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 3);
        Triangles.Add(BaseVertexIndex + 7);
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 7);
        Triangles.Add(BaseVertexIndex + 4);
        
        // Right face (1,2,6,5)
        Triangles.Add(BaseVertexIndex + 1);
        Triangles.Add(BaseVertexIndex + 6);
        Triangles.Add(BaseVertexIndex + 2);
        Triangles.Add(BaseVertexIndex + 1);
        Triangles.Add(BaseVertexIndex + 5);
        Triangles.Add(BaseVertexIndex + 6);
        
        NumCellsVisualized++;
    }
    
    // Only create mesh if we have cells to visualize
    if (NumCellsVisualized > 0)
    {
        // Create the mesh section
        CoverageVisualizationMesh->CreateMeshSection(0, Vertices, Triangles, Normals,
            UV0, VertexColors, Tangents, false);
        
        // Set the material
        if (CoverageMaterial)
        {
            CoverageVisualizationMesh->SetMaterial(0, CoverageMaterial);
        }
    }
    
    UE_LOG(LogTemp, Display, TEXT("Coverage visualization: Created mesh with %d cells "
                                  "visible out of %d populated cells (%d vertices, %d triangles)"), 
           NumCellsVisualized, UnifiedGrid.Num(), Vertices.Num(), Triangles.Num() / 3);
}

void ASceneAnalysisManager::CreateSafeZoneMesh()
{
    // Clear existing mesh sections
    SafeZoneVisualizationMesh->ClearAllMeshSections();
    
    // Create arrays for procedural mesh
    TArray<FVector> Vertices;
    TArray<int32> Triangles;
    TArray<FVector> Normals;
    TArray<FVector2D> UV0;
    TArray<FColor> VertexColors;
    TArray<FProcMeshTangent> Tangents;
    
    // Pre-allocate memory based on estimated number of unsafe cells
    int32 EstimatedUnsafeCells = UnifiedGrid.Num() / 2; // Rough estimate
    Vertices.Reserve(EstimatedUnsafeCells * 8);
    Triangles.Reserve(EstimatedUnsafeCells * 36);
    Normals.Reserve(EstimatedUnsafeCells * 8);
    UV0.Reserve(EstimatedUnsafeCells * 8);
    VertexColors.Reserve(EstimatedUnsafeCells * 8);
    Tangents.Reserve(EstimatedUnsafeCells * 8);
    
    // Find the Z range of unsafe cells to create proper gradient mapping
    float MinZ = FLT_MAX;
    float MaxZ = -FLT_MAX;
    
    for (const auto& Pair : UnifiedGrid)
    {
        if (!Pair.Value.bIsSafe)
        {
            const FIntVector& GridCoords = Pair.Key;
            float CellZ = GridOrigin.Z + (GridCoords.Z + 0.5f) * GridResolution;
            MinZ = FMath::Min(MinZ, CellZ);
            MaxZ = FMath::Max(MaxZ, CellZ);
        }
    }
    
    // Prevent division by zero if all cells are at same height
    if (FMath::IsNearlyEqual(MaxZ, MinZ))
    {
        MaxZ = MinZ + 1.0f;
    }
    
    // Create cubes only for unsafe cells
    int32 NumCellsVisualized = 0;
    for (const auto& Pair : UnifiedGrid)
    {
        const FIntVector& GridCoords = Pair.Key;
        const FUnifiedGridCell& Cell = Pair.Value;
        
        // Skip safe cells - only visualize unsafe cells
        if (Cell.bIsSafe)
            continue;
        
        // Calculate cell center in world space
        FVector CellCenter(
            GridOrigin.X + (GridCoords.X + 0.5f) * GridResolution,
            GridOrigin.Y + (GridCoords.Y + 0.5f) * GridResolution,
            GridOrigin.Z + (GridCoords.Z + 0.5f) * GridResolution
        );
        
        // Calculate cell size (slightly smaller than grid resolution to see cell boundaries)
        float CellSize = GridResolution * 0.9f;
        float HalfSize = CellSize * 0.5f;
        
        // Calculate color based on height
        float NormalizedHeight = (CellCenter.Z - MinZ) / (MaxZ - MinZ);
        
        FLinearColor CellLinearColor = FLinearColor::LerpUsingHSV(
            SafeZoneDarkColor, SafeZoneLightColor, NormalizedHeight);
        
        // Convert to FColor and add transparency
        FColor CellColor = CellLinearColor.ToFColor(false);
        CellColor.A = 180; // Semi-transparent
        
        // Add a cube for this cell
        int32 BaseVertexIndex = Vertices.Num();
        
        // Define the 8 corners of the cube
        Vertices.Add(CellCenter + FVector(-HalfSize, -HalfSize, -HalfSize)); // 0: bottom left back
        Vertices.Add(CellCenter + FVector(HalfSize, -HalfSize, -HalfSize));  // 1: bottom right back
        Vertices.Add(CellCenter + FVector(HalfSize, HalfSize, -HalfSize));   // 2: bottom right front
        Vertices.Add(CellCenter + FVector(-HalfSize, HalfSize, -HalfSize));  // 3: bottom left front
        Vertices.Add(CellCenter + FVector(-HalfSize, -HalfSize, HalfSize));  // 4: top left back
        Vertices.Add(CellCenter + FVector(HalfSize, -HalfSize, HalfSize));   // 5: top right back
        Vertices.Add(CellCenter + FVector(HalfSize, HalfSize, HalfSize));    // 6: top right front
        Vertices.Add(CellCenter + FVector(-HalfSize, HalfSize, HalfSize));   // 7: top left front
        
        // Add colors for all 8 vertices
        for (int32 i = 0; i < 8; ++i)
        {
            VertexColors.Add(CellColor);
        }
        
        // Add texture coordinates
        UV0.Add(FVector2D(0, 0)); // 0
        UV0.Add(FVector2D(1, 0)); // 1
        UV0.Add(FVector2D(1, 1)); // 2
        UV0.Add(FVector2D(0, 1)); // 3
        UV0.Add(FVector2D(0, 0)); // 4
        UV0.Add(FVector2D(1, 0)); // 5
        UV0.Add(FVector2D(1, 1)); // 6
        UV0.Add(FVector2D(0, 1)); // 7
        
        // Add normals
        Normals.Add(FVector(-1, -1, -1).GetSafeNormal()); // 0
        Normals.Add(FVector(1, -1, -1).GetSafeNormal());  // 1
        Normals.Add(FVector(1, 1, -1).GetSafeNormal());   // 2
        Normals.Add(FVector(-1, 1, -1).GetSafeNormal());  // 3
        Normals.Add(FVector(-1, -1, 1).GetSafeNormal());  // 4
        Normals.Add(FVector(1, -1, 1).GetSafeNormal());   // 5
        Normals.Add(FVector(1, 1, 1).GetSafeNormal());    // 6
        Normals.Add(FVector(-1, 1, 1).GetSafeNormal());   // 7
        
        // Add tangents (simplified)
        for (int32 i = 0; i < 8; ++i)
        {
            Tangents.Add(FProcMeshTangent(1, 0, 0));
        }
        
        // Add triangles for each face (12 triangles total)
        // Bottom face (0,1,2,3)
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 1);
        Triangles.Add(BaseVertexIndex + 2);
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 2);
        Triangles.Add(BaseVertexIndex + 3);
        
        // Top face (4,5,6,7)
        Triangles.Add(BaseVertexIndex + 4);
        Triangles.Add(BaseVertexIndex + 6);
        Triangles.Add(BaseVertexIndex + 5);
        Triangles.Add(BaseVertexIndex + 4);
        Triangles.Add(BaseVertexIndex + 7);
        Triangles.Add(BaseVertexIndex + 6);
        
        // Front face (3,2,6,7)
        Triangles.Add(BaseVertexIndex + 3);
        Triangles.Add(BaseVertexIndex + 2);
        Triangles.Add(BaseVertexIndex + 6);
        Triangles.Add(BaseVertexIndex + 3);
        Triangles.Add(BaseVertexIndex + 6);
        Triangles.Add(BaseVertexIndex + 7);
        
        // Back face (0,1,5,4)
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 5);
        Triangles.Add(BaseVertexIndex + 1);
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 4);
        Triangles.Add(BaseVertexIndex + 5);
        
        // Left face (0,3,7,4)
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 3);
        Triangles.Add(BaseVertexIndex + 7);
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 7);
        Triangles.Add(BaseVertexIndex + 4);
        
        // Right face (1,2,6,5)
        Triangles.Add(BaseVertexIndex + 1);
        Triangles.Add(BaseVertexIndex + 6);
        Triangles.Add(BaseVertexIndex + 2);
        Triangles.Add(BaseVertexIndex + 1);
        Triangles.Add(BaseVertexIndex + 5);
        Triangles.Add(BaseVertexIndex + 6);
        
        NumCellsVisualized++;
    }
    
    // Only create mesh if we have cells to visualize
    if (NumCellsVisualized > 0)
    {
        // Create the mesh section
        SafeZoneVisualizationMesh->CreateMeshSection(0, Vertices, Triangles, Normals,
            UV0, VertexColors, Tangents, false);
        
        // Set the material
        if (SafeZoneMaterial)
        {
            SafeZoneVisualizationMesh->SetMaterial(0, SafeZoneMaterial);
        }
    }
    
    UE_LOG(LogTemp, Display, TEXT("Safe zone visualization: Created mesh with %d unsafe cells "
                                  "out of %d populated cells (%d vertices, %d triangles)"), 
           NumCellsVisualized, UnifiedGrid.Num(), Vertices.Num(), Triangles.Num() / 3);
}


// Add these implementations to SceneAnalysisManager.cpp

void ASceneAnalysisManager::AnalyzeGeometricComplexity()
{
    if (!World || !bGridInitialized || SceneMeshes.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("AnalyzeGeometricComplexity: "
                                      "World not initialized or no meshes in scene"));
        return;
    }
    
    // Apply preset parameters if not using custom
    if (SceneComplexityPreset != ESceneComplexityPreset::Custom)
    {
        ApplyComplexityPreset(SceneComplexityPreset);
    }
    
    UE_LOG(LogTemp, Display, TEXT("Analyzing geometric complexity with weights: "
                                  "Curvature=%.2f, EdgeDensity=%.2f, AngleVariation=%.2f"),
           CurvatureWeight, EdgeDensityWeight, AngleVariationWeight);
    
    // Reset complexity scores
    for (auto& Pair : UnifiedGrid)
    {
        FUnifiedGridCell& Cell = Pair.Value;
        Cell.CurvatureScore = 0.0f;
        Cell.EdgeDensityScore = 0.0f;
        Cell.AngleVariationScore = 0.0f;
        Cell.ComplexityScore = 0.0f;
    }
    
    // Thread-safe storage for collected data
    FCriticalSection DataLock;
    
    // More efficient data structures
    struct FCellData {
        TArray<FVector> Normals;
        int32 EdgeCount;
        TArray<float> DihedralAngles;
        
        FCellData() : EdgeCount(0) {}
    };
    
    TMap<FIntVector, FCellData> CellDataMap;
    
    // Prepare the map with expected capacity to avoid rehashing
    CellDataMap.Reserve(UnifiedGrid.Num());
    for (auto& Pair : UnifiedGrid)
    {
        CellDataMap.Add(Pair.Key, FCellData());
    }
    
    // Calculate min/max values for adaptive normalization
    float MinCurvature = FLT_MAX, MaxCurvature = -FLT_MAX;
    float MinEdgeDensity = FLT_MAX, MaxEdgeDensity = -FLT_MAX;
    float MinAngleVar = FLT_MAX, MaxAngleVar = -FLT_MAX;
    
    // Track processed edges to avoid duplicates
    TSet<TPair<int32, int32>> ProcessedEdges;
    ProcessedEdges.Reserve(TotalTrianglesInScene * 3);
    
    // Process meshes in parallel
    ParallelFor(SceneMeshes.Num(), [&](int32 MeshIndex)
    {
        const FMeshInfo& MeshInfo = SceneMeshes[MeshIndex];
        
        // Thread-local data to minimize lock contention
        TMap<FIntVector, FCellData> LocalCellData;
        TSet<TPair<int32, int32>> LocalProcessedEdges;
        
        // Process each triangle in this mesh
        for (int32 i = 0; i < MeshInfo.Indices.Num(); i += 3)
        {
            if (i + 2 >= MeshInfo.Indices.Num()) continue;
            
            // Get triangle vertices
            const int32 Idx0 = MeshInfo.Indices[i];
            const int32 Idx1 = MeshInfo.Indices[i + 1];
            const int32 Idx2 = MeshInfo.Indices[i + 2];
            
            if (Idx0 >= MeshInfo.VertexPositions.Num() || 
                Idx1 >= MeshInfo.VertexPositions.Num() || 
                Idx2 >= MeshInfo.VertexPositions.Num())
            {
                continue; // Skip invalid indices
            }
            
            const FVector& V0 = MeshInfo.VertexPositions[Idx0];
            const FVector& V1 = MeshInfo.VertexPositions[Idx1];
            const FVector& V2 = MeshInfo.VertexPositions[Idx2];
            
            if ((V0 - V1).IsNearlyZero() || (V1 - V2).IsNearlyZero() || (V2 - V0).IsNearlyZero())
            {
                continue;
            }
            
            // Calculate triangle normal (optimized)
            FVector Edge1 = V1 - V0;
            FVector Edge2 = V2 - V0;
            FVector TriangleNormal = FVector::CrossProduct(Edge1, Edge2);
            
            // Skip triangles with zero area
            float TriangleArea = TriangleNormal.Size();
            if (TriangleArea < KINDA_SMALL_NUMBER)
            {
                continue;
            }
            
            // Normalize the normal
            TriangleNormal /= TriangleArea;
            
            // Calculate triangle centroid
            FVector Centroid = (V0 + V1 + V2) / 3.0f;
            
            // Get grid cell for centroid
            FIntVector CellCoords = WorldToGridCoordinates(Centroid);
            
            // Add triangle normal to cell's normal list
            FCellData& CellData = LocalCellData.FindOrAdd(CellCoords);
            CellData.Normals.Add(TriangleNormal);
            
            // Process edges for edge density calculation
            // Use an ordered pair to ensure consistent edge representation
            TPair<int32, int32> Edge1Pair(FMath::Min(Idx0, Idx1), FMath::Max(Idx0, Idx1));
            TPair<int32, int32> Edge2Pair(FMath::Min(Idx1, Idx2), FMath::Max(Idx1, Idx2));
            TPair<int32, int32> Edge3Pair(FMath::Min(Idx2, Idx0), FMath::Max(Idx2, Idx0));
            
            if (!LocalProcessedEdges.Contains(Edge1Pair))
            {
                LocalProcessedEdges.Add(Edge1Pair);
                CellData.EdgeCount++;
            }
            
            if (!LocalProcessedEdges.Contains(Edge2Pair))
            {
                LocalProcessedEdges.Add(Edge2Pair);
                CellData.EdgeCount++;
            }
            
            if (!LocalProcessedEdges.Contains(Edge3Pair))
            {
                LocalProcessedEdges.Add(Edge3Pair);
                CellData.EdgeCount++;
            }
            
            // OPTIMIZATION: Find adjacent triangles more efficiently
            // We'll batch process dihedral angles in a second pass
        }
        
        // Second pass for dihedral angles
        // Instead of nested loops, use an optimized lookup approach
        
        // Create a map of edges to triangles for faster lookup
        TMap<TPair<int32, int32>, TArray<TPair<int32, FVector>>> EdgeToTriangles;
        
        for (int32 TriIdx = 0; TriIdx < MeshInfo.Indices.Num() / 3; TriIdx++)
        {
            int32 BaseIdx = TriIdx * 3;
            if (BaseIdx + 2 >= MeshInfo.Indices.Num()) continue;
            
            int32 Idx0 = MeshInfo.Indices[BaseIdx];
            int32 Idx1 = MeshInfo.Indices[BaseIdx + 1];
            int32 Idx2 = MeshInfo.Indices[BaseIdx + 2];
            
            if (Idx0 >= MeshInfo.VertexPositions.Num() || 
                Idx1 >= MeshInfo.VertexPositions.Num() || 
                Idx2 >= MeshInfo.VertexPositions.Num())
            {
                continue;
            }
            
            // Calculate normal once
            const FVector& V0 = MeshInfo.VertexPositions[Idx0];
            const FVector& V1 = MeshInfo.VertexPositions[Idx1];
            const FVector& V2 = MeshInfo.VertexPositions[Idx2];
            
            FVector Edge1 = V1 - V0;
            FVector Edge2 = V2 - V0;
            FVector Normal = FVector::CrossProduct(Edge1, Edge2);
            
            // Skip degenerate triangles
            if (Normal.IsNearlyZero())
                continue;
                
            Normal.Normalize();
            
            // Get triangle edges
            TPair<int32, int32> Edge1Pair(FMath::Min(Idx0, Idx1), FMath::Max(Idx0, Idx1));
            TPair<int32, int32> Edge2Pair(FMath::Min(Idx1, Idx2), FMath::Max(Idx1, Idx2));
            TPair<int32, int32> Edge3Pair(FMath::Min(Idx2, Idx0), FMath::Max(Idx2, Idx0));
            
            // Add this triangle to each edge's list
            EdgeToTriangles.FindOrAdd(Edge1Pair).Add(TPair<int32, FVector>(TriIdx, Normal));
            EdgeToTriangles.FindOrAdd(Edge2Pair).Add(TPair<int32, FVector>(TriIdx, Normal));
            EdgeToTriangles.FindOrAdd(Edge3Pair).Add(TPair<int32, FVector>(TriIdx, Normal));
        }
        
        // Now find adjacent triangles by shared edges
        for (auto& EdgePair : EdgeToTriangles)
        {
            const TArray<TPair<int32, FVector>>& TriangleList = EdgePair.Value;
            
            // Only edges shared by exactly 2 triangles can form dihedral angles
            if (TriangleList.Num() != 2)
                continue;
                
            // Get both triangle's normals
            const FVector& Normal1 = TriangleList[0].Value;
            const FVector& Normal2 = TriangleList[1].Value;
            
            // Calculate dihedral angle
            float CosAngle = FVector::DotProduct(Normal1, Normal2);
            CosAngle = FMath::Clamp(CosAngle, -1.0f, 1.0f); // Avoid acos domain errors
            float AngleRadians = FMath::Acos(CosAngle);
            float AngleDegrees = FMath::RadiansToDegrees(AngleRadians);
            
            // Get triangle indices
            int32 TriIdx1 = TriangleList[0].Key;
            int32 TriIdx2 = TriangleList[1].Key;
            
            // Calculate centroids of both triangles
            FVector Centroid1 = FVector::ZeroVector;
            FVector Centroid2 = FVector::ZeroVector;
            
            for (int32 i = 0; i < 3; i++)
            {
                int32 Idx = MeshInfo.Indices[TriIdx1 * 3 + i];
                if (Idx < MeshInfo.VertexPositions.Num())
                    Centroid1 += MeshInfo.VertexPositions[Idx];
                    
                Idx = MeshInfo.Indices[TriIdx2 * 3 + i];
                if (Idx < MeshInfo.VertexPositions.Num())
                    Centroid2 += MeshInfo.VertexPositions[Idx];
            }
            
            Centroid1 /= 3.0f;
            Centroid2 /= 3.0f;
            
            // Get grid cells for both centroids
            FIntVector CellCoords1 = WorldToGridCoordinates(Centroid1);
            FIntVector CellCoords2 = WorldToGridCoordinates(Centroid2);
            
            // Add angle to both cells' angle lists
            LocalCellData.FindOrAdd(CellCoords1).DihedralAngles.Add(AngleDegrees);
            
            // Only add to second cell if it's different from the first
            if (CellCoords1 != CellCoords2)
            {
                LocalCellData.FindOrAdd(CellCoords2).DihedralAngles.Add(AngleDegrees);
            }
        }
        
        // Merge thread-local data with global data
        FScopeLock Lock(&DataLock);
        
        for (auto& Pair : LocalCellData)
        {
            FCellData& GlobalCellData = CellDataMap.FindOrAdd(Pair.Key);
            GlobalCellData.Normals.Append(Pair.Value.Normals);
            GlobalCellData.EdgeCount += Pair.Value.EdgeCount;
            GlobalCellData.DihedralAngles.Append(Pair.Value.DihedralAngles);
        }
    });
    
    // Calculate complexity scores for each cell with data
    for (auto& Pair : CellDataMap)
    {
        const FIntVector& CellCoords = Pair.Key;
        
        // Skip if cell not in unified grid (shouldn't happen)
        if (!UnifiedGrid.Contains(CellCoords))
            continue;
            
        FCellData& CellData = Pair.Value;
        FUnifiedGridCell& Cell = UnifiedGrid[CellCoords];
        
        // Calculate cell volume (for normalizing edge density)
        float CellVolume = FMath::Pow(GridResolution, 3);
        
        // Calculate raw scores only if we have data
        float RawCurvatureScore = 0.0f;
        if (CellData.Normals.Num() >= 2)
        {
            RawCurvatureScore = CalculateCurvatureScore(CellData.Normals);
        }
        
        float RawEdgeDensityScore = 0.0f;
        if (CellData.EdgeCount > 0)
        {
            RawEdgeDensityScore = CalculateEdgeDensityScore(CellData.EdgeCount, CellVolume);
        }
        
        float RawAngleVariationScore = 0.0f;
        if (CellData.DihedralAngles.Num() >= 2)
        {
            RawAngleVariationScore = CalculateAngleVariationScore(CellData.DihedralAngles);
        }
        
        // Track min/max for normalization if using adaptive normalization
        if (bUseAdaptiveNormalization)
        {
            if (CellData.Normals.Num() >= 2)
            {
                MinCurvature = FMath::Min(MinCurvature, RawCurvatureScore);
                MaxCurvature = FMath::Max(MaxCurvature, RawCurvatureScore);
            }
            
            if (CellData.EdgeCount > 0)
            {
                MinEdgeDensity = FMath::Min(MinEdgeDensity, RawEdgeDensityScore);
                MaxEdgeDensity = FMath::Max(MaxEdgeDensity, RawEdgeDensityScore);
            }
            
            if (CellData.DihedralAngles.Num() >= 2)
            {
                MinAngleVar = FMath::Min(MinAngleVar, RawAngleVariationScore);
                MaxAngleVar = FMath::Max(MaxAngleVar, RawAngleVariationScore);
            }
        }
        
        // Store raw scores
        Cell.CurvatureScore = RawCurvatureScore;
        Cell.EdgeDensityScore = RawEdgeDensityScore;
        Cell.AngleVariationScore = RawAngleVariationScore;
    }
    
    // Ensure valid min/max values for normalization
    if (bUseAdaptiveNormalization)
    {
        // Handle cases where no valid data was found
        if (MinCurvature == FLT_MAX || MaxCurvature == -FLT_MAX)
        {
            MinCurvature = 0.0f;
            MaxCurvature = 1.0f;
        }
        
        if (MinEdgeDensity == FLT_MAX || MaxEdgeDensity == -FLT_MAX)
        {
            MinEdgeDensity = 0.0f;
            MaxEdgeDensity = 1.0f;
        }
        
        if (MinAngleVar == FLT_MAX || MaxAngleVar == -FLT_MAX)
        {
            MinAngleVar = 0.0f;
            MaxAngleVar = 1.0f;
        }
        
        // Prevent division by zero
        if (FMath::IsNearlyEqual(MaxCurvature, MinCurvature))
            MaxCurvature = MinCurvature + 1.0f;
            
        if (FMath::IsNearlyEqual(MaxEdgeDensity, MinEdgeDensity))
            MaxEdgeDensity = MinEdgeDensity + 1.0f;
            
        if (FMath::IsNearlyEqual(MaxAngleVar, MinAngleVar))
            MaxAngleVar = MinAngleVar + 1.0f;
    }
    
    // Second pass: normalize scores and calculate combined complexity
    for (auto& Pair : UnifiedGrid)
    {
        FUnifiedGridCell& Cell = Pair.Value;
        
        // Normalize individual scores if using adaptive normalization
        if (bUseAdaptiveNormalization)
        {
            // Normalize each score between 0 and 1 based on min/max values
            Cell.CurvatureScore = (Cell.CurvatureScore - MinCurvature) / (MaxCurvature - MinCurvature);
            Cell.EdgeDensityScore = (Cell.EdgeDensityScore - MinEdgeDensity) / (MaxEdgeDensity - MinEdgeDensity);
            Cell.AngleVariationScore = (Cell.AngleVariationScore - MinAngleVar) / (MaxAngleVar - MinAngleVar);
        }
        
        // Clamp normalized values between 0 and 1
        Cell.CurvatureScore = FMath::Clamp(Cell.CurvatureScore, 0.0f, 1.0f);
        Cell.EdgeDensityScore = FMath::Clamp(Cell.EdgeDensityScore, 0.0f, 1.0f);
        Cell.AngleVariationScore = FMath::Clamp(Cell.AngleVariationScore, 0.0f, 1.0f);
        
        // Calculate weighted complexity score
        Cell.ComplexityScore = 
            Cell.CurvatureScore * CurvatureWeight +
            Cell.EdgeDensityScore * EdgeDensityWeight +
            Cell.AngleVariationScore * AngleVariationWeight;
        
        // Ensure total weight is 1.0
        float TotalWeight = CurvatureWeight + EdgeDensityWeight + AngleVariationWeight;
        if (!FMath::IsNearlyEqual(TotalWeight, 1.0f))
        {
            Cell.ComplexityScore /= TotalWeight;
        }
    }
    
    bComplexityVisualizationDirty = true;
    
    // Log results
    int32 TotalAnalyzedCells = CellDataMap.Num();
    int32 HighComplexityCells = 0;
    
    for (const auto& Pair : UnifiedGrid)
    {
        if (Pair.Value.ComplexityScore > 0.7f)
            HighComplexityCells++;
    }
    
    UE_LOG(LogTemp, Display, TEXT("Geometric complexity analysis complete: %d cells analyzed, %d high-complexity cells identified"),
           TotalAnalyzedCells, HighComplexityCells);
}

TArray<FIntVector> ASceneAnalysisManager::GetHighComplexityRegions(float ComplexityThreshold)
{
    TArray<FIntVector> HighComplexityRegions;
    
    for (const auto& Pair : UnifiedGrid)
    {
        if (Pair.Value.ComplexityScore >= ComplexityThreshold)
        {
            HighComplexityRegions.Add(Pair.Key);
        }
    }
    
    return HighComplexityRegions;
}

float ASceneAnalysisManager::CalculateCurvatureScore(const TArray<FVector>& Normals)
{
    // Optimized calculation of variance of surface normals
    if (Normals.Num() < 2)
        return 0.0f;
    
    // Use a more efficient approach to calculate the average normal
    FVector AvgNormal(0, 0, 0);
    
    // Process normals in batches for better cache coherency
    const int32 BatchSize = 64;
    int32 NumBatches = FMath::DivideAndRoundUp(Normals.Num(), BatchSize);
    
    for (int32 BatchIdx = 0; BatchIdx < NumBatches; BatchIdx++)
    {
        FVector BatchSum(0, 0, 0);
        int32 StartIdx = BatchIdx * BatchSize;
        int32 EndIdx = FMath::Min(StartIdx + BatchSize, Normals.Num());
        
        for (int32 i = StartIdx; i < EndIdx; i++)
        {
            BatchSum += Normals[i];
        }
        
        AvgNormal += BatchSum;
    }
    
    float InvNum = 1.0f / Normals.Num();
    AvgNormal *= InvNum;
    
    // Normalize only if not close to zero
    if (!AvgNormal.IsNearlyZero())
    {
        AvgNormal.Normalize();
    }
    else
    {
        // Default direction if average is zero
        AvgNormal = FVector(0, 0, 1);
    }
    
    // Calculate variation from average normal efficiently
    float TotalVariation = 0.0f;
    
    for (int32 BatchIdx = 0; BatchIdx < NumBatches; BatchIdx++)
    {
        float BatchVariation = 0.0f;
        int32 StartIdx = BatchIdx * BatchSize;
        int32 EndIdx = FMath::Min(StartIdx + BatchSize, Normals.Num());
        
        for (int32 i = StartIdx; i < EndIdx; i++)
        {
            // 1 - dot product gives deviation from average normal
            float Deviation = 1.0f - FVector::DotProduct(Normals[i], AvgNormal);
            BatchVariation += Deviation;
        }
        
        TotalVariation += BatchVariation;
    }
    
    return TotalVariation * InvNum * CurvatureSensitivity;
}

float ASceneAnalysisManager::CalculateEdgeDensityScore(int32 EdgeCount, float CellVolume)
{
    // Skip division if cell volume is too small
    if (CellVolume < KINDA_SMALL_NUMBER)
        return EdgeCount > 0 ? 1.0f : 0.0f;
        
    // More efficient normalization
    float NormalizedDensity = (float)EdgeCount / (CellVolume * EdgeDensityNormalizationFactor);
    
    // Early clamping
    return FMath::Min(1.0f, NormalizedDensity);
}

float ASceneAnalysisManager::CalculateAngleVariationScore(const TArray<float>& Angles)
{
    if (Angles.Num() < 2)
        return 0.0f;
        
    // More efficient calculation of angle statistics
    float Sum = 0.0f;
    float SumOfSquares = 0.0f;
    
    // Calculate sum and sum of squares in one pass
    for (float Angle : Angles)
    {
        Sum += Angle;
        SumOfSquares += Angle * Angle;
    }
    
    float InvCount = 1.0f / Angles.Num();
    float Mean = Sum * InvCount;
    
    // Calculate variance efficiently
    float Variance = (SumOfSquares * InvCount) - (Mean * Mean);
    
    // Standard deviation as a measure of variation
    float StdDev = FMath::Sqrt(FMath::Max(0.0f, Variance));
    
    // Normalize by threshold
    return FMath::Min(1.0f, StdDev / AngleVariationThreshold);
}

FVector ASceneAnalysisManager::CalculateTriangleNormal(const FVector& V0, const FVector& V1, const FVector& V2)
{
    // More efficient normal calculation
    FVector Edge1 = V1 - V0;
    FVector Edge2 = V2 - V0;
    FVector Normal = FVector::CrossProduct(Edge1, Edge2);
    
    // Get length once to avoid repeated calculation
    float Length = Normal.Size();
    
    // Handle degenerate triangles efficiently
    if (Length > KINDA_SMALL_NUMBER)
    {
        // Normalize directly using the length we already calculated
        Normal *= (1.0f / Length);
    }
    else
    {
        // Default normal if triangle is degenerate
        Normal = FVector(0, 0, 1);
    }
    
    return Normal;
}

void ASceneAnalysisManager::ApplyComplexityPreset(ESceneComplexityPreset Preset)
{
    switch(Preset)
    {
        case ESceneComplexityPreset::UrbanOutdoor:
            // Urban scenes value edge density more than curvature
            CurvatureWeight = 0.25f;
            EdgeDensityWeight = 0.5f;
            AngleVariationWeight = 0.25f;
            CurvatureSensitivity = 0.6f;
            EdgeDensityNormalizationFactor = 15.0f;
            AngleVariationThreshold = 30.0f;
            break;
            
        case ESceneComplexityPreset::IndoorCluttered:
            // Indoor scenes benefit from balanced approach
            CurvatureWeight = 0.33f;
            EdgeDensityWeight = 0.33f;
            AngleVariationWeight = 0.33f;
            CurvatureSensitivity = 1.0f;
            EdgeDensityNormalizationFactor = 20.0f;
            AngleVariationThreshold = 40.0f;
            break;
            
        case ESceneComplexityPreset::NaturalTerrain:
            // Natural terrain has smooth curves but complex angle variations
            CurvatureWeight = 0.2f;
            EdgeDensityWeight = 0.3f;
            AngleVariationWeight = 0.5f;
            CurvatureSensitivity = 0.7f;
            EdgeDensityNormalizationFactor = 7.0f;
            AngleVariationThreshold = 60.0f;
            break;
            
        case ESceneComplexityPreset::MechanicalParts:
            // Mechanical parts have sharp edges and precise angles
            CurvatureWeight = 0.3f;
            EdgeDensityWeight = 0.4f;
            AngleVariationWeight = 0.3f;
            CurvatureSensitivity = 1.2f;
            EdgeDensityNormalizationFactor = 25.0f;
            AngleVariationThreshold = 15.0f;  // Very sensitive to angle changes
            break;
            
        case ESceneComplexityPreset::Generic:
        default:
            // Balanced default parameters
            CurvatureWeight = 0.4f;
            EdgeDensityWeight = 0.3f;
            AngleVariationWeight = 0.3f;
            CurvatureSensitivity = 0.8f;
            EdgeDensityNormalizationFactor = 10.0f;
            AngleVariationThreshold = 45.0f;
            break;
    }
    
    UE_LOG(LogTemp, Display, TEXT("Applied complexity preset: %s"), 
        *UEnum::GetValueAsString(Preset));
}

void ASceneAnalysisManager::InitializeComplexityVisualization()
{
    if (!World)
        return;
    
    // Initialize the procedural mesh component if it doesn't exist
    if (!ComplexityVisualizationMesh)
    {
        ComplexityVisualizationMesh = NewObject<UProceduralMeshComponent>(this);
        ComplexityVisualizationMesh->RegisterComponent();
        ComplexityVisualizationMesh->SetMobility(EComponentMobility::Movable);
        ComplexityVisualizationMesh->AttachToComponent(GetRootComponent(), 
                                                  FAttachmentTransformRules::KeepWorldTransform);
        ComplexityVisualizationMesh->SetCollisionEnabled(ECollisionEnabled::Type::NoCollision);
    }
    
    // Load or create the complexity material if not already set
    if (!ComplexityMaterial)
    {
        // Try to load the complexity material
        ComplexityMaterial = LoadObject<UMaterialInterface>(nullptr, 
            TEXT("/Script/Engine.Material'/VCCSim/Materials/M_Complexity.M_Complexity'"));
        
        // Only log error if material failed to load
        if (!ComplexityMaterial)
        {
            UE_LOG(LogTemp, Error, TEXT("InitializeComplexityVisualization: "
                                      "Failed to load complexity material."));
        }
    }
}

void ASceneAnalysisManager::VisualizeComplexity(bool bShow)
{
    if (!World)
        return;
    
    // Check if we have complexity data
    bool bHasComplexityData = false;
    for (const auto& Pair : UnifiedGrid)
    {
        if (Pair.Value.ComplexityScore > 0.0f)
        {
            bHasComplexityData = true;
            break;
        }
    }
    
    if (!bHasComplexityData)
    {
        UE_LOG(LogTemp, Warning, TEXT("VisualizeComplexity: No complexity data found."
                                      " Run AnalyzeGeometricComplexity first."));
        return;
    }
    
    // If not showing, clear visualization and return
    if (!bShow)
    {
        if (ComplexityVisualizationMesh)
        {
            ComplexityVisualizationMesh->SetVisibility(false);
        }
        return;
    }
    
    // Initialize visualization components if needed
    InitializeComplexityVisualization();
    
    // If mesh component failed to initialize, return
    if (!ComplexityVisualizationMesh)
    {
        UE_LOG(LogTemp, Error, TEXT("VisualizeComplexity: "
                                    "Complexity mesh component not initialized"));
        return;
    }
    
    // Only update mesh if dirty or visibility is changing
    if (bComplexityVisualizationDirty)
    {
        CreateComplexityMesh();
        bComplexityVisualizationDirty = false;
    }
    
    // Set visibility
    ComplexityVisualizationMesh->SetVisibility(true);
}

void ASceneAnalysisManager::ClearComplexityVisualization()
{
    if (ComplexityVisualizationMesh)
    {
        ComplexityVisualizationMesh->ClearAllMeshSections();
        ComplexityVisualizationMesh->SetVisibility(false);
    }
}

void ASceneAnalysisManager::CreateComplexityMesh()
{
    // Clear existing mesh sections
    ComplexityVisualizationMesh->ClearAllMeshSections();
    
    // Create arrays for procedural mesh
    TArray<FVector> Vertices;
    TArray<int32> Triangles;
    TArray<FVector> Normals;
    TArray<FVector2D> UV0;
    TArray<FColor> VertexColors;
    TArray<FProcMeshTangent> Tangents;
    
    // Pre-allocate memory based on number of cells
    int32 EstimatedCells = UnifiedGrid.Num();
    Vertices.Reserve(EstimatedCells * 8);
    Triangles.Reserve(EstimatedCells * 36);
    Normals.Reserve(EstimatedCells * 8);
    UV0.Reserve(EstimatedCells * 8);
    VertexColors.Reserve(EstimatedCells * 8);
    Tangents.Reserve(EstimatedCells * 8);
    
    // Create cubes only for cells with complexity above zero
    int32 NumCellsVisualized = 0;
    for (const auto& Pair : UnifiedGrid)
    {
        const FIntVector& GridCoords = Pair.Key;
        const FUnifiedGridCell& Cell = Pair.Value;
        
        // Skip cells with no complexity
        if (Cell.ComplexityScore <= 0.0f)
            continue;
        
        // Calculate cell center in world space
        FVector CellCenter(
            GridOrigin.X + (GridCoords.X + 0.5f) * GridResolution,
            GridOrigin.Y + (GridCoords.Y + 0.5f) * GridResolution,
            GridOrigin.Z + (GridCoords.Z + 0.5f) * GridResolution
        );
        
        // Calculate cell size (slightly smaller than grid resolution to see cell boundaries)
        float CellSize = GridResolution * 0.9f;
        float HalfSize = CellSize * 0.5f;
        
        // Convert complexity to color (blue = low, green = medium, red = high)
        FLinearColor ComplexityColor;
        
        if (Cell.ComplexityScore < 0.33f)
        {
            // Blue to Cyan
            float T = Cell.ComplexityScore / 0.33f;
            ComplexityColor = FLinearColor::LerpUsingHSV(
                FLinearColor(0.0f, 0.0f, 1.0f), // Blue (low complexity)
                FLinearColor(0.0f, 1.0f, 1.0f), // Cyan
                T
            );
        }
        else if (Cell.ComplexityScore < 0.66f)
        {
            // Cyan to Yellow
            float T = (Cell.ComplexityScore - 0.33f) / 0.33f;
            ComplexityColor = FLinearColor::LerpUsingHSV(
                FLinearColor(0.0f, 1.0f, 1.0f), // Cyan
                FLinearColor(1.0f, 1.0f, 0.0f), // Yellow
                T
            );
        }
        else
        {
            // Yellow to Red
            float T = (Cell.ComplexityScore - 0.66f) / 0.34f;
            ComplexityColor = FLinearColor::LerpUsingHSV(
                FLinearColor(1.0f, 1.0f, 0.0f), // Yellow
                FLinearColor(1.0f, 0.0f, 0.0f), // Red (high complexity)
                T
            );
        }
        
        FColor CellColor = ComplexityColor.ToFColor(false);
        
        // Adjust alpha based on complexity
        CellColor.A = 128 + FMath::FloorToInt(127.0f * Cell.ComplexityScore); // 128-255 range
        
        // Add a cube for this cell
        int32 BaseVertexIndex = Vertices.Num();
        
        // Define the 8 corners of the cube
        Vertices.Add(CellCenter + FVector(-HalfSize, -HalfSize, -HalfSize)); // 0: bottom left back
        Vertices.Add(CellCenter + FVector(HalfSize, -HalfSize, -HalfSize));  // 1: bottom right back
        Vertices.Add(CellCenter + FVector(HalfSize, HalfSize, -HalfSize));   // 2: bottom right front
        Vertices.Add(CellCenter + FVector(-HalfSize, HalfSize, -HalfSize));  // 3: bottom left front
        Vertices.Add(CellCenter + FVector(-HalfSize, -HalfSize, HalfSize));  // 4: top left back
        Vertices.Add(CellCenter + FVector(HalfSize, -HalfSize, HalfSize));   // 5: top right back
        Vertices.Add(CellCenter + FVector(HalfSize, HalfSize, HalfSize));    // 6: top right front
        Vertices.Add(CellCenter + FVector(-HalfSize, HalfSize, HalfSize));   // 7: top left front
        
        // Add colors for all 8 vertices
        for (int32 i = 0; i < 8; ++i)
        {
            VertexColors.Add(CellColor);
        }
        
        // Add texture coordinates
        UV0.Add(FVector2D(0, 0)); // 0
        UV0.Add(FVector2D(1, 0)); // 1
        UV0.Add(FVector2D(1, 1)); // 2
        UV0.Add(FVector2D(0, 1)); // 3
        UV0.Add(FVector2D(0, 0)); // 4
        UV0.Add(FVector2D(1, 0)); // 5
        UV0.Add(FVector2D(1, 1)); // 6
        UV0.Add(FVector2D(0, 1)); // 7
        
        // Add normals
        Normals.Add(FVector(-1, -1, -1).GetSafeNormal()); // 0
        Normals.Add(FVector(1, -1, -1).GetSafeNormal());  // 1
        Normals.Add(FVector(1, 1, -1).GetSafeNormal());   // 2
        Normals.Add(FVector(-1, 1, -1).GetSafeNormal());  // 3
        Normals.Add(FVector(-1, -1, 1).GetSafeNormal());  // 4
        Normals.Add(FVector(1, -1, 1).GetSafeNormal());   // 5
        Normals.Add(FVector(1, 1, 1).GetSafeNormal());    // 6
        Normals.Add(FVector(-1, 1, 1).GetSafeNormal());   // 7
        
        // Add tangents (simplified)
        for (int32 i = 0; i < 8; ++i)
        {
            Tangents.Add(FProcMeshTangent(1, 0, 0));
        }
        
        // Add triangles for each face (12 triangles total)
        // Bottom face (0,1,2,3)
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 1);
        Triangles.Add(BaseVertexIndex + 2);
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 2);
        Triangles.Add(BaseVertexIndex + 3);
        
        // Top face (4,5,6,7)
        Triangles.Add(BaseVertexIndex + 4);
        Triangles.Add(BaseVertexIndex + 6);
        Triangles.Add(BaseVertexIndex + 5);
        Triangles.Add(BaseVertexIndex + 4);
        Triangles.Add(BaseVertexIndex + 7);
        Triangles.Add(BaseVertexIndex + 6);
        
        // Front face (3,2,6,7)
        Triangles.Add(BaseVertexIndex + 3);
        Triangles.Add(BaseVertexIndex + 2);
        Triangles.Add(BaseVertexIndex + 6);
        Triangles.Add(BaseVertexIndex + 3);
        Triangles.Add(BaseVertexIndex + 6);
        Triangles.Add(BaseVertexIndex + 7);
        
        // Back face (0,1,5,4)
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 5);
        Triangles.Add(BaseVertexIndex + 1);
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 4);
        Triangles.Add(BaseVertexIndex + 5);
        
        // Left face (0,3,7,4)
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 3);
        Triangles.Add(BaseVertexIndex + 7);
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 7);
        Triangles.Add(BaseVertexIndex + 4);
        
        // Right face (1,2,6,5)
        Triangles.Add(BaseVertexIndex + 1);
        Triangles.Add(BaseVertexIndex + 6);
        Triangles.Add(BaseVertexIndex + 2);
        Triangles.Add(BaseVertexIndex + 1);
        Triangles.Add(BaseVertexIndex + 5);
        Triangles.Add(BaseVertexIndex + 6);
        
        NumCellsVisualized++;
    }
    
    // Only create mesh if we have cells to visualize
    if (NumCellsVisualized > 0)
    {
        // Create the mesh section
        ComplexityVisualizationMesh->CreateMeshSection(0, Vertices, Triangles, Normals,
            UV0, VertexColors, Tangents, false);
        
        // Set the material
        if (ComplexityMaterial)
        {
            ComplexityVisualizationMesh->SetMaterial(0, ComplexityMaterial);
        }
    }
    
    UE_LOG(LogTemp, Display, TEXT("Complexity visualization: "
                                  "Created mesh with %d cells visible, %d vertices, %d triangles"),
          NumCellsVisualized, Vertices.Num(), Triangles.Num() / 3);
}

/* ----------------------------- Test ----------------------------- */

void ASceneAnalysisManager::ExportMeshesToPly()
{
    if (!World)
    {
        UE_LOG(LogTemp, Warning, TEXT("USceneAnalysisManager::ExportMeshesToPly:"
            " World not set!"));
        return;
    }

    if (SceneMeshes.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("USceneAnalysisManager::ExportMeshesToPly:"
            " No meshes found in the scene!"));
        return;
    }
    
    // Create export directory if it doesn't exist
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.DirectoryExists(*LogPath))
    {
        PlatformFile.CreateDirectory(*LogPath);
    }

    
    // Export each mesh to a separate PLY file
    for (const FMeshInfo& MeshInfo : SceneMeshes)
    {
        FString FilePath = FPaths::Combine(LogPath, FString::Printf(TEXT("%s_%d.ply"),
            *MeshInfo.MeshName, MeshInfo.MeshID));
        FFileHelper::SaveStringToFile(GeneratePlyContent(MeshInfo), *FilePath);
    }
}

FString ASceneAnalysisManager::GeneratePlyContent(const FMeshInfo& MeshInfo)
{
    FString PLYContent;
    
    // PLY Header
    PLYContent += TEXT("ply\n");
    PLYContent += TEXT("format ascii 1.0\n");
    PLYContent += FString::Printf(TEXT("element vertex %d\n"), MeshInfo.VertexPositions.Num());
    PLYContent += TEXT("property float x\n");
    PLYContent += TEXT("property float y\n");
    PLYContent += TEXT("property float z\n");
    PLYContent += FString::Printf(TEXT("element face %d\n"), MeshInfo.NumTriangles);
    PLYContent += TEXT("property list uchar int vertex_indices\n");
    PLYContent += TEXT("end_header\n");
    
    // Vertex data
    for (const FVector& Vertex : MeshInfo.VertexPositions)
    {
        PLYContent += FString::Printf(TEXT("%f %f %f\n"), Vertex.X, Vertex.Y, Vertex.Z);
    }
    
    // Face data
    for (int32 i = 0; i < MeshInfo.Indices.Num(); i += 3)
    {
        if (i + 2 < MeshInfo.Indices.Num())
        {
            PLYContent += FString::Printf(TEXT("3 %d %d %d\n"), 
                MeshInfo.Indices[i], 
                MeshInfo.Indices[i + 1], 
                MeshInfo.Indices[i + 2]);
        }
    }
    
    return PLYContent;
}

void ASceneAnalysisManager::VisualizeSceneMeshes(
    float Duration, bool bShowWireframe, bool bShowVertices, float VertexSize)
{
    if (!World || SceneMeshes.Num() == 0)
        return;
    
    // Generate a unique color for each mesh for easier distinction
    TArray<FColor> MeshColors;
    for (int32 i = 0; i < SceneMeshes.Num(); ++i)
    {
        // Create visually distinct colors using golden ratio
        const float Hue = fmodf(i * 0.618033988749895f, 1.0f);
        FLinearColor LinearColor = FLinearColor::MakeFromHSV8(Hue * 255.0f, 200, 200);
        MeshColors.Add(LinearColor.ToFColor(false));
    }
    
    // Visualize each mesh
    for (int32 MeshIdx = 0; MeshIdx < SceneMeshes.Num(); ++MeshIdx)
    {
        const FMeshInfo& MeshInfo = SceneMeshes[MeshIdx];
        const FColor& Color = MeshColors[MeshIdx];
        
        // Draw mesh bounds
        DrawDebugBox(World, MeshInfo.Bounds.Origin, MeshInfo.Bounds.BoxExtent,
            Color, false, Duration, 0, 2.0f);
        
        // Draw mesh ID text
        FString MeshText = FString::Printf(TEXT("Mesh ID: %d\nName: %s\nTriangles: %d"), 
            MeshInfo.MeshID, *MeshInfo.MeshName, MeshInfo.NumTriangles);
        DrawDebugString(World, MeshInfo.Bounds.Origin, MeshText, nullptr, Color, Duration);
        
        // Draw wireframe if requested
        if (bShowWireframe)
        {
            for (int32 i = 0; i < MeshInfo.Indices.Num(); i += 3)
            {
                if (i + 2 < MeshInfo.Indices.Num())
                {
                    const FVector& V0 = MeshInfo.VertexPositions[MeshInfo.Indices[i]];
                    const FVector& V1 = MeshInfo.VertexPositions[MeshInfo.Indices[i + 1]];
                    const FVector& V2 = MeshInfo.VertexPositions[MeshInfo.Indices[i + 2]];
                    
                    DrawDebugLine(World, V0, V1, Color, false,
                        Duration, 0, 1.0f);
                    DrawDebugLine(World, V1, V2, Color, false,
                        Duration, 0, 1.0f);
                    DrawDebugLine(World, V2, V0, Color, false,
                        Duration, 0, 1.0f);
                }
            }
        }
        
        // Draw vertices if requested
        if (bShowVertices)
        {
            for (const FVector& Vertex : MeshInfo.VertexPositions)
            {
                DrawDebugPoint(World, Vertex, VertexSize, Color,
                    false, Duration);
            }
        }
    }
    
    // Log some statistics
    UE_LOG(LogTemp, Display, TEXT("Visualized %d meshes with %d total "
                                  "triangles and %d total vertices"), 
        SceneMeshes.Num(), TotalTrianglesInScene, TotalPointsInScene);
}

void ASceneAnalysisManager::VisualizeSampledPoints(float Duration, float VertexSize)
{
    for (const auto& Point : CoverageMap)
    {
        DrawDebugPoint(World, Point.Key, VertexSize, FColor::Green,
            false, Duration);
    }
}
