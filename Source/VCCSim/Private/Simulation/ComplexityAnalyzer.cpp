#include "Simulation/ComplexityAnalyzer.h"
#include "ProceduralMeshComponent.h"

UComplexityAnalyzer::UComplexityAnalyzer()
{
}

void UComplexityAnalyzer::AnalyzeGeometricComplexity()
{
    if (!(*GridInitializedPtr) || MeshInfos->Num() == 0)
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
    for (auto& Pair : *UnifiedGridPtr)
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
    CellDataMap.Reserve(UnifiedGridPtr->Num());
    for (auto& Pair : *UnifiedGridPtr)
    {
        CellDataMap.Add(Pair.Key, FCellData());
    }
    
    // Calculate min/max values for adaptive normalization
    float MinCurvature = FLT_MAX, MaxCurvature = -FLT_MAX;
    float MinEdgeDensity = FLT_MAX, MaxEdgeDensity = -FLT_MAX;
    float MinAngleVar = FLT_MAX, MaxAngleVar = -FLT_MAX;
    
    // Track processed edges to avoid duplicates
    TSet<TPair<int32, int32>> ProcessedEdges;
    ProcessedEdges.Reserve(*TotalTrianglesInScenePtr * 3);
    
    // Process meshes in parallel
    ParallelFor(MeshInfos->Num(), [&](int32 MeshIndex)
    {
        const FMeshInfo& MeshInfo = (*MeshInfos)[MeshIndex];
        
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
        if (!UnifiedGridPtr->Contains(CellCoords))
            continue;
            
        FCellData& CellData = Pair.Value;
        FUnifiedGridCell& Cell = (*UnifiedGridPtr)[CellCoords];
        
        // Calculate cell volume (for normalizing edge density)
        float CellVolume = FMath::Pow((*GridResolutionPtr), 3);
        
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
    for (auto& Pair : *UnifiedGridPtr)
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
    
    for (const auto& Pair : *UnifiedGridPtr)
    {
        if (Pair.Value.ComplexityScore > 0.7f)
            HighComplexityCells++;
    }
    
    UE_LOG(LogTemp, Display, TEXT("Geometric complexity analysis complete: %d cells analyzed, %d high-complexity cells identified"),
           TotalAnalyzedCells, HighComplexityCells);
}

TArray<FIntVector> UComplexityAnalyzer::GetHighComplexityRegions(float ComplexityThreshold)
{
    TArray<FIntVector> HighComplexityRegions;
    
    for (const auto& Pair : *UnifiedGridPtr)
    {
        if (Pair.Value.ComplexityScore >= ComplexityThreshold)
        {
            HighComplexityRegions.Add(Pair.Key);
        }
    }
    
    return HighComplexityRegions;
}

float UComplexityAnalyzer::CalculateCurvatureScore(const TArray<FVector>& Normals)
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

float UComplexityAnalyzer::CalculateEdgeDensityScore(int32 EdgeCount, float CellVolume)
{
    // Skip division if cell volume is too small
    if (CellVolume < KINDA_SMALL_NUMBER)
        return EdgeCount > 0 ? 1.0f : 0.0f;
        
    // More efficient normalization
    float NormalizedDensity = (float)EdgeCount / (CellVolume * EdgeDensityNormalizationFactor);
    
    // Early clamping
    return FMath::Min(1.0f, NormalizedDensity);
}

float UComplexityAnalyzer::CalculateAngleVariationScore(const TArray<float>& Angles)
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

FVector UComplexityAnalyzer::CalculateTriangleNormal(
    const FVector& V0, const FVector& V1, const FVector& V2)
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




void UComplexityAnalyzer::ApplyComplexityPreset(ESceneComplexityPreset Preset)
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

void UComplexityAnalyzer::InitializeComplexityVisualization()
{    
    // Initialize the procedural mesh component if it doesn't exist
    if (!ComplexityVisualizationMesh)
    {
        ComplexityVisualizationMesh = NewObject<UProceduralMeshComponent>(this);
        ComplexityVisualizationMesh->RegisterComponent();
        ComplexityVisualizationMesh->SetMobility(EComponentMobility::Movable);
        ComplexityVisualizationMesh->AttachToComponent(
            this, FAttachmentTransformRules::KeepWorldTransform);
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

void UComplexityAnalyzer::VisualizeComplexity(bool bShow)
{// Check if we have complexity data
    bool bHasComplexityData = false;
    for (const auto& Pair : *UnifiedGridPtr)
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

void UComplexityAnalyzer::ClearComplexityVisualization()
{
    if (ComplexityVisualizationMesh)
    {
        ComplexityVisualizationMesh->ClearAllMeshSections();
        ComplexityVisualizationMesh->SetVisibility(false);
    }
}

void UComplexityAnalyzer::CreateComplexityMesh()
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
    int32 EstimatedCells = UnifiedGridPtr->Num();
    Vertices.Reserve(EstimatedCells * 8);
    Triangles.Reserve(EstimatedCells * 36);
    Normals.Reserve(EstimatedCells * 8);
    UV0.Reserve(EstimatedCells * 8);
    VertexColors.Reserve(EstimatedCells * 8);
    Tangents.Reserve(EstimatedCells * 8);
    
    // Create cubes only for cells with complexity above zero
    int32 NumCellsVisualized = 0;
    for (const auto& Pair : *UnifiedGridPtr)
    {
        const FIntVector& GridCoords = Pair.Key;
        const FUnifiedGridCell& Cell = Pair.Value;
        
        // Skip cells with no complexity
        if (Cell.ComplexityScore <= 0.0f)
            continue;
        
        // Calculate cell center in world space
        FVector CellCenter(
            (*GridOriginPtr).X + (GridCoords.X + 0.5f) * (*GridResolutionPtr),
            (*GridOriginPtr).Y + (GridCoords.Y + 0.5f) * (*GridResolutionPtr),
            (*GridOriginPtr).Z + (GridCoords.Z + 0.5f) * (*GridResolutionPtr)
        );
        
        // Calculate cell size (slightly smaller than grid resolution to see cell boundaries)
        float CellSize = (*GridResolutionPtr) * 0.9f;
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

FIntVector UComplexityAnalyzer::WorldToGridCoordinates(const FVector& WorldPos) const
{
    return FIntVector(
        FMath::FloorToInt((WorldPos.X - GridOriginPtr->X) / (*GridResolutionPtr)),
        FMath::FloorToInt((WorldPos.Y - GridOriginPtr->Y) / (*GridResolutionPtr)),
        FMath::FloorToInt((WorldPos.Z - GridOriginPtr->Z) / (*GridResolutionPtr))
    );
}