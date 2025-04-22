#include "Simulation/CoverageAnalyzer.h"
#include "ProceduralMeshComponent.h"

FCoverageData UCoverageAnalyzer::ComputeCoverage(
    const TArray<FTransform>& CameraTransforms, const FString& CameraName)
{
    FCoverageData CoverageData;
    CoverageData.CoveragePercentage = 0.0f;
    CoverageData.TotalVisibleTriangles = 0;
    
    if (CameraTransforms.Num() == 0 || CoverageMap.Num() == 0)
        return CoverageData;
    
    // Get camera intrinsic for specified camera name
    FMatrix44f CameraIntrinsic;
    
    if (CameraIntrinsicsPtr->Contains(CameraName))
    {
        CameraIntrinsic = (*CameraIntrinsicsPtr)[CameraName];
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("ComputeCoverage: "
                                      "No intrinsics found for camera %s"), *CameraName);
    }
    
    // Reset visibility of all points
    VisiblePoints.Empty(CoverageMap.Num() / 2); // Pre-allocate with estimated capacity
    InvisiblePoints.Empty(CoverageMap.Num() / 2);
    
    // Create a spatial hash grid for quickly finding which mesh a point belongs to
    // This is a simple implementation - could be replaced with a more sophisticated structure
    constexpr float SpatialGridSize = 500.0f; // Adjust based on scene scale
    TMap<FIntVector, TArray<int32>> SpatialGrid;
    
    // Populate spatial grid with mesh IDs
    for (int32 MeshIdx = 0; MeshIdx < MeshInfos->Num(); ++MeshIdx)
    {
        const FMeshInfo& MeshInfo = (*MeshInfos)[MeshIdx];
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
                    if ((*MeshInfos)[MeshIdx].Bounds.GetBox().IsInsideOrOn(PointData.Point))
                    {
                        VisibleMeshIDs.Add((*MeshInfos)[MeshIdx].MeshID);
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
    float CoveragePercentage = TotalPoints > 0 ? (float)VisiblePointCount /
        (float)TotalPoints * 100.0f : 0.0f;
    
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
                const FMeshInfo& MeshInfo = (*MeshInfos)[MeshIdx];
                
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
    
    for (const FMeshInfo& MeshInfo : *MeshInfos)
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
    
    if (*GridInitializedPtr)
    {
        UpdateCoverageGrid();
    }
    
    return CoverageData;
}

FCoverageData UCoverageAnalyzer::ComputeCoverage(
    const FTransform& CameraTransform, const FString& CameraName)
{
    // Create array with single transform and call the multi-transform version
    TArray<FTransform> CameraTransforms;
    CameraTransforms.Add(CameraTransform);
    return ComputeCoverage(CameraTransforms, CameraName);
}

void UCoverageAnalyzer::InitializeCoverageVisualization()
{
    if (!World)
        return;
    
    // Initialize the procedural mesh component if it doesn't exist
    if (!CoverageVisualizationMesh)
    {
        CoverageVisualizationMesh = NewObject<UProceduralMeshComponent>(this);
        CoverageVisualizationMesh->RegisterComponent();
        CoverageVisualizationMesh->SetMobility(EComponentMobility::Movable);
        CoverageVisualizationMesh->AttachToComponent(
            this, FAttachmentTransformRules::KeepWorldTransform);
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

void UCoverageAnalyzer::VisualizeCoverage(bool bShow)
{
    if (!World)
        return;
    
    // Check if we have coverage data
    if (!(*GridInitializedPtr))
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

void UCoverageAnalyzer::ClearCoverageVisualization()
{
    if (CoverageVisualizationMesh)
    {
        CoverageVisualizationMesh->ClearAllMeshSections();
        CoverageVisualizationMesh->SetVisibility(false);
    }
}

void UCoverageAnalyzer::UpdateCoverageGrid()
{
    if (!(*GridInitializedPtr))
        return;

    // Reset visible points and coverage in existing cells
    for (auto& Pair : *UnifiedGridPtr)
    {
        FUnifiedGridCell& Cell = Pair.Value;
        Cell.VisiblePoints = 0;
        Cell.Coverage = 0.0f;
        // Note: we don't reset safe zone data here
    }

    // Process each point in the coverage map
    for (const auto& Pair : CoverageMap)
    {
        const FVector& Point = Pair.Key;
        bool bIsVisible = Pair.Value;

        // Convert world position to grid coordinates
        FIntVector GridCoords = WorldToGridCoordinates(Point);

        // Get the cell (should already exist from initialization)
        if (UnifiedGridPtr->Contains(GridCoords))
        {
            FUnifiedGridCell& Cell = (*UnifiedGridPtr)[GridCoords];

            // If point is visible, increment visible points
            if (bIsVisible)
            {
                Cell.VisiblePoints++;
            }
        }
    }

    // Normalize coverage values
    for (auto& Pair : *UnifiedGridPtr)
    {
        FUnifiedGridCell& Cell = Pair.Value;
        if (Cell.TotalPoints > 0)
        {
            Cell.Coverage = (float)Cell.VisiblePoints / (float)Cell.TotalPoints;
        }
    }
}


void UCoverageAnalyzer::CreateCoverageMesh()
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
    int32 EstimatedCells = UnifiedGridPtr->Num();
    Vertices.Reserve(EstimatedCells * 8);
    Triangles.Reserve(EstimatedCells * 36);
    Normals.Reserve(EstimatedCells * 8);
    UV0.Reserve(EstimatedCells * 8);
    VertexColors.Reserve(EstimatedCells * 8);
    Tangents.Reserve(EstimatedCells * 8);
    
    // Create cubes only for cells with sample points and coverage above threshold
    int32 NumCellsVisualized = 0;
    for (const auto& Pair : *UnifiedGridPtr)
    {
        const FIntVector& GridCoords = Pair.Key;
        const FUnifiedGridCell& Cell = Pair.Value;
        
        // Skip cells with no sample points or low coverage
        if (Cell.TotalPoints == 0 || Cell.Coverage < VisualizationThreshold)
            continue;
        
        // Calculate cell center in world space
        FVector CellCenter(
            GridOriginPtr->X + (GridCoords.X + 0.5f) * (*GridResolutionPtr),
            GridOriginPtr->Y + (GridCoords.Y + 0.5f) * (*GridResolutionPtr),
            GridOriginPtr->Z + (GridCoords.Z + 0.5f) * (*GridResolutionPtr)
        );
        
        // Calculate cell size (slightly smaller than grid resolution to see cell boundaries)
        float CellSize = (*GridResolutionPtr) * 0.9f;
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
           NumCellsVisualized, UnifiedGridPtr->Num(), Vertices.Num(), Triangles.Num() / 3);
}

void UCoverageAnalyzer::VisualizeSampledPoints(float Duration, float VertexSize)
{
    for (const auto& Point : CoverageMap)
    {
        DrawDebugPoint(World, Point.Key, VertexSize, FColor::Green,
            false, Duration);
    }
}

void UCoverageAnalyzer::ConstructFrustum(FConvexVolume& OutFrustum, const FTransform& CameraPose,
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
    
    float HorizontalFOV = (fx > KINDA_SMALL_NUMBER) ? 2.0f *
        FMath::Atan(width * 0.5f * fxRecip) : FMath::DegreesToRadians(90.0f);
    float VerticalFOV = (fy > KINDA_SMALL_NUMBER) ? 2.0f *
        FMath::Atan(height * 0.5f * fyRecip) : FMath::DegreesToRadians(60.0f);
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
    FVector LeftNormal = -FVector::CrossProduct(FarTopLeft - Position,
        FarBottomLeft - Position).GetSafeNormal();
    OutFrustum.Planes.Add(FPlane(Position, -LeftNormal));
    
    FVector RightNormal = -FVector::CrossProduct(FarBottomRight - Position,
        FarTopRight - Position).GetSafeNormal();
    OutFrustum.Planes.Add(FPlane(Position, -RightNormal));
    
    FVector TopNormal = -FVector::CrossProduct(FarTopRight - Position,
        FarTopLeft - Position).GetSafeNormal();
    OutFrustum.Planes.Add(FPlane(Position, -TopNormal));
    
    FVector BottomNormal = -FVector::CrossProduct(FarBottomLeft - Position,
        FarBottomRight - Position).GetSafeNormal();
    OutFrustum.Planes.Add(FPlane(Position, -BottomNormal));
}

TArray<FVector> UCoverageAnalyzer::SamplePointsOnMesh(const FMeshInfo& MeshInfo)
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

// Update the ResetCoverage function to initialize the point arrays
void UCoverageAnalyzer::ResetCoverage()
{
    CoverageMap.Empty();
    CurrentlyVisibleMeshIDs.Empty();
    CurrentCoveragePercentage = 0.0f;
    VisiblePoints.Empty();
    InvisiblePoints.Empty();
    
    // Initialize coverage map with all points set to not visible
    for (const FMeshInfo& MeshInfo : *MeshInfos)
    {
        TArray<FVector> SampledPoints = SamplePointsOnMesh(MeshInfo);
        for (const FVector& Point : SampledPoints)
        {
            CoverageMap.Add(Point, false);
            InvisiblePoints.Add(Point);
        }
    }
    
    bCoverageVisualizationDirty = true;
}

void UCoverageAnalyzer::PrepareCoverage()
{
    UnifiedGridPtr->Empty(CoverageMap.Num());
    for (const auto& Pair : CoverageMap)
    {
        const FVector& Point = Pair.Key;
        FIntVector GridCoords = WorldToGridCoordinates(Point);
        FUnifiedGridCell& Cell = UnifiedGridPtr->FindOrAdd(GridCoords);
        Cell.TotalPoints++;
    }
}

FIntVector UCoverageAnalyzer::WorldToGridCoordinates(const FVector& WorldPos) const
{
    return FIntVector(
        FMath::FloorToInt((WorldPos.X - GridOriginPtr->X) / (*GridResolutionPtr)),
        FMath::FloorToInt((WorldPos.Y - GridOriginPtr->Y) / (*GridResolutionPtr)),
        FMath::FloorToInt((WorldPos.Z - GridOriginPtr->Z) / (*GridResolutionPtr))
    );
}