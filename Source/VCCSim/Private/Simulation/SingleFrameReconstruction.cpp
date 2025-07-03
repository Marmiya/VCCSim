#include "Simulation/SingleFrameReconstruction.h"
#include "Async/ParallelFor.h"
#include "Sensors/LidarSensor.h"
#include "HAL/PlatformMath.h"
#include "Engine/Engine.h"
#include "DrawDebugHelpers.h"
#include "HAL/PlatformFilemanager.h"
#include "Misc/DateTime.h"

USingleFrameReconstruction::USingleFrameReconstruction()
{
    PrimaryComponentTick.bCanEverTick = false;
    
    // Initialize default configuration
    ReconConfig.GHPRParam = 3.6f;
    ReconConfig.PseudoFaceThreshold = 0.01f;
    ReconConfig.SectorNum = 8;
    ReconConfig.MinPointsPerSector = 5;
    ReconConfig.ViewpointElevation = 0.0f;
    ReconConfig.bMultiThread = true;
    
    // Initialize performance tracking
    LastReconstructionTime = 0.0f;
    LastProcessedPointCount = 0;
    LastGeneratedTriangleCount = 0;
}

void USingleFrameReconstruction::BeginPlay()
{
    Super::BeginPlay();

    if (!ProceduralMeshComponent)
    {
        if (AActor* Owner = GetOwner())
        {
            ProceduralMeshComponent = Owner->FindComponentByClass<UProceduralMeshComponent>();
            if (!ProceduralMeshComponent)
            {
                ProceduralMeshComponent = NewObject<UProceduralMeshComponent>(Owner, 
                    UProceduralMeshComponent::StaticClass(), TEXT("ReconstructionMesh"));
                ProceduralMeshComponent->RegisterComponent();
                ProceduralMeshComponent->SetWorldTransform(FTransform::Identity);
                // Move z a bit to avoid z-fighting with ground
                ProceduralMeshComponent->SetRelativeLocation(FVector(0.0f, 0.0f, 1.0f));
                ProceduralMeshComponent->AttachToComponent(Owner->GetRootComponent(), 
                    FAttachmentTransformRules::KeepWorldTransform);
            }
        }
    }
}

void USingleFrameReconstruction::DoSingleRecon()
{
    double StartTime = FPlatformTime::Seconds();
    
    if (!LidarComponent)
    {
        UE_LOG(LogTemp, Error, TEXT("LidarComponent is not set!"));
        return;
    }
    
    if (!ProceduralMeshComponent)
    {
        UE_LOG(LogTemp, Error, TEXT("ProceduralMeshComponent is not set!"));
        return;
    }

    auto Data = LidarComponent->GetPointCloudDataAndOdom();
    
    // if (Data.Key.Num() == 0)
    // {
    //     UE_LOG(LogTemp, Warning, TEXT("No points from LiDAR!"));
    //     return;
    // }
    //
    // TArray<FVector> ConvertedPointCloud;
    // ConvertedPointCloud.Reserve(Data.Key.Num());
    //
    // for (const FVector3f& Point3f : Data.Key)
    // {
    //     ConvertedPointCloud.Add(FVector(Point3f));
    // }
    //
    // LastProcessedPointCount = ConvertedPointCloud.Num();
    // UE_LOG(LogTemp, Warning, TEXT("Processing %d points from LiDAR"), ConvertedPointCloud.Num());
    //
    // ReconstructFrame(ConvertedPointCloud, Data.Value.Location);
    //
    // LastGeneratedTriangleCount = Triangles.Num() / 3;
    // LastReconstructionTime = FPlatformTime::Seconds() - StartTime;
    //
    // UE_LOG(LogTemp, Warning, TEXT("Reconstruction complete: %d vertices, %d triangles in %.3f seconds"), 
    //     ReconstructedVertices.Num(), LastGeneratedTriangleCount, LastReconstructionTime);
    //
    // ApplyToProceduralMesh(ProceduralMeshComponent);
    // DrawDebugNormals(GetWorld());
    // DrawDebugSectors(GetWorld(), Data.Value.Location);
}

void USingleFrameReconstruction::ReconstructFrame(
    const TArray<FVector>& PointCloud, const FVector& ViewPoint)
{
    if (PointCloud.Num() < ReconConfig.MinPointsPerSector)
    {
        UE_LOG(LogTemp, Warning, TEXT("Not enough points for reconstruction: %d < %d"), 
            PointCloud.Num(), ReconConfig.MinPointsPerSector);
        return;
    }

    ClearReconstruction();
    
    // Step 1: Divide point cloud into sectors
    DivideSectors(PointCloud, ViewPoint);
    
    UE_LOG(LogTemp, Log, TEXT("Divided %d points into %d sectors"), 
        PointCloud.Num(), ReconConfig.SectorNum);

    // Step 2: Process each sector
    if (ReconConfig.bMultiThread)
    {
        ParallelFor(SectorData.Num(), [&](int32 Index)
        {
            ProcessSector(Index, ViewPoint);
        });
    }
    else
    {
        for (int32 i = 0; i < SectorData.Num(); i++)
        {
            ProcessSector(i, ViewPoint);
        }
    }

    // Step 3: Combine results from all sectors
    CombineSectorResults();

    // Step 4: Compute vertex normals
    ComputeVertexNormals(SectorData);
    
    // Step 5: Validate reconstruction data
    ValidateReconstructionData();
}

void USingleFrameReconstruction::DivideSectors(
    const TArray<FVector>& PointCloud, const FVector& ViewPoint)
{
    SectorData.SetNum(ReconConfig.SectorNum);
    
    // Clear existing data
    for (FSectorData& Sector : SectorData)
    {
        Sector.Points.Empty();
        Sector.Faces.Empty();
        Sector.PointIndices.Empty();
    }
    
    for (int32 i = 0; i < PointCloud.Num(); i++)
    {
        FVector RelativePoint = PointCloud[i] - ViewPoint;
        int32 SectorIndex = GetSectorIndex(RelativePoint, FVector::ZeroVector, ReconConfig.SectorNum);
        
        if (SectorIndex >= 0 && SectorIndex < ReconConfig.SectorNum)
        {
            SectorData[SectorIndex].Points.Add(PointCloud[i]);
            SectorData[SectorIndex].PointIndices.Add(i);
        }
    }
    
    // Log sector distribution
    for (int32 i = 0; i < SectorData.Num(); i++)
    {
        UE_LOG(LogTemp, VeryVerbose, TEXT("Sector %d: %d points"), i, SectorData[i].Points.Num());
    }
}

void USingleFrameReconstruction::ProcessSector(int32 SectorIndex, const FVector& ViewPoint)
{
    FSectorData& Sector = SectorData[SectorIndex];
    
    if (Sector.Points.Num() < ReconConfig.MinPointsPerSector)
    {
        UE_LOG(LogTemp, VeryVerbose, TEXT("Sector %d skipped: insufficient points (%d < %d)"), 
            SectorIndex, Sector.Points.Num(), ReconConfig.MinPointsPerSector);
        return;
    }

    // Add helper points if elevation is set
    TArray<FVector> SectorPoints = Sector.Points;
    if (ReconConfig.ViewpointElevation > 0.0f)
    {
        FVector GroundPoint = ViewPoint;
        GroundPoint.Z -= ReconConfig.ViewpointElevation;
        SectorPoints.Add(GroundPoint);
        
        FVector TopPoint = ViewPoint;
        TopPoint.Z += ReconConfig.ViewpointElevation * 2.0f;
        SectorPoints.Add(TopPoint);
    }

    // GHPR Algorithm Implementation
    TArray<FVector> TransformedPoints;
    TArray<float> OriginalDistances;
    float Radius;
    
    // Step 1: Transform points using proper GHPR
    TransformPointsForGHPR_Proper(SectorPoints, ViewPoint, TransformedPoints, OriginalDistances, Radius);
    
    // Step 2: Compute convex hull in transformed space
    TArray<FTriangleFace> HullFaces = ComputeConvexHull_Proper(TransformedPoints);
    
    // Step 3: Map faces back to original points
    TArray<int32> HullToOriginalMapping = MapHullToOriginalPoints(TransformedPoints, SectorPoints, ViewPoint);
    
    // Step 4: Filter faces and compute normals
    FilterAndOrientFaces(HullFaces, SectorPoints, ViewPoint, HullToOriginalMapping, Sector.Faces);
    
    UE_LOG(LogTemp, VeryVerbose, TEXT("Sector %d processed: %d points -> %d faces"), 
        SectorIndex, SectorPoints.Num(), Sector.Faces.Num());
}

void USingleFrameReconstruction::TransformPointsForGHPR_Proper(
    const TArray<FVector>& Points, 
    const FVector& ViewPoint,
    TArray<FVector>& OutTransformedPoints,
    TArray<float>& OutDistances,
    float& OutRadius)
{
    OutTransformedPoints.Empty();
    OutDistances.Empty();
    OutTransformedPoints.Reserve(Points.Num() + 1);
    OutDistances.Reserve(Points.Num());

    // Convert to local coordinates and compute distances
    float MaxDistance = 0.0f;
    TArray<FVector> LocalPoints;
    LocalPoints.Reserve(Points.Num());
    
    for (const FVector& Point : Points)
    {
        FVector LocalPoint = Point - ViewPoint;
        float Distance = LocalPoint.Size();
        OutDistances.Add(Distance);
        MaxDistance = FMath::Max(MaxDistance, Distance);
        LocalPoints.Add(LocalPoint);
    }

    // Calculate radius using GHPR formula
    OutRadius = FMath::Pow(10.0f, ReconConfig.GHPRParam) * MaxDistance;

    // IMPORTANT: Match the original implementation's transformation
    // The original code just normalizes and scales by 100, not the standard GHPR formula
    for (const FVector& LocalPoint : LocalPoints)
    {
        if (LocalPoint.SizeSquared() > SMALL_NUMBER)
        {
            // This matches the original: 100 * normalized_vector
            OutTransformedPoints.Add(LocalPoint.GetSafeNormal() * 100.0f);
        }
        else
        {
            OutTransformedPoints.Add(FVector::ZeroVector);
        }
    }

    // Add viewpoint at origin in transformed space
    OutTransformedPoints.Add(FVector::ZeroVector);
}


TArray<FTriangleFace> USingleFrameReconstruction::ComputeConvexHull_Proper(
    const TArray<FVector>& Points)
{
    TArray<FTriangleFace> Faces;
    
    if (Points.Num() < 4)
    {
        UE_LOG(LogTemp, VeryVerbose, TEXT("Not enough points for convex hull: %d"), Points.Num());
        return Faces;
    }

    // Use a simple but more complete convex hull algorithm
    // This is a basic incremental algorithm - for production use, 
    // consider using a proper 3D convex hull library
    
    // Step 1: Find initial tetrahedron
    TArray<int32> TetraIndices;
    if (!FindInitialTetrahedron(Points, TetraIndices))
    {
        UE_LOG(LogTemp, Warning, TEXT("Failed to find initial tetrahedron"));
        return Faces;
    }

    // Create initial faces from tetrahedron
    Faces.Add(CreateFace(TetraIndices[0], TetraIndices[1], TetraIndices[2], Points));
    Faces.Add(CreateFace(TetraIndices[0], TetraIndices[1], TetraIndices[3], Points));
    Faces.Add(CreateFace(TetraIndices[0], TetraIndices[2], TetraIndices[3], Points));
    Faces.Add(CreateFace(TetraIndices[1], TetraIndices[2], TetraIndices[3], Points));

    // Step 2: Add remaining points incrementally
    TSet<int32> ProcessedPoints;
    for (int32 idx : TetraIndices)
    {
        ProcessedPoints.Add(idx);
    }

    for (int32 i = 0; i < Points.Num() && Faces.Num() < 5000; i++)
    {
        if (ProcessedPoints.Contains(i))
            continue;

        // Find visible faces from this point
        TArray<FTriangleFace> VisibleFaces;
        TArray<int32> VisibleIndices;
        
        for (int32 j = 0; j < Faces.Num(); j++)
        {
            const FTriangleFace& Face = Faces[j];
            FVector FaceCenter = (Points[Face.V0] + Points[Face.V1] + Points[Face.V2]) / 3.0f;
            FVector ToPoint = Points[i] - FaceCenter;
            
            if (FVector::DotProduct(Face.Normal, ToPoint) > KINDA_SMALL_NUMBER)
            {
                VisibleFaces.Add(Face);
                VisibleIndices.Add(j);
            }
        }

        if (VisibleFaces.Num() > 0)
        {
            // Find horizon edges
            TMap<FIntPoint, int32> EdgeCount;
            for (const FTriangleFace& Face : VisibleFaces)
            {
                AddEdge(EdgeCount, Face.V0, Face.V1);
                AddEdge(EdgeCount, Face.V1, Face.V2);
                AddEdge(EdgeCount, Face.V2, Face.V0);
            }

            // Remove visible faces
            for (int32 k = VisibleIndices.Num() - 1; k >= 0; k--)
            {
                Faces.RemoveAt(VisibleIndices[k]);
            }

            // Add new faces from horizon edges
            for (const auto& EdgePair : EdgeCount)
            {
                if (EdgePair.Value == 1) // Horizon edge
                {
                    FTriangleFace NewFace = CreateFace(EdgePair.Key.X, EdgePair.Key.Y, i, Points);
                    if (NewFace.Normal.SizeSquared() > SMALL_NUMBER)
                    {
                        Faces.Add(NewFace);
                    }
                }
            }

            ProcessedPoints.Add(i);
        }
    }

    // Orient all faces consistently
    for (FTriangleFace& Face : Faces)
    {
        FVector Center = (Points[Face.V0] + Points[Face.V1] + Points[Face.V2]) / 3.0f;
        
        // Compute hull center (approximate)
        FVector HullCenter = FVector::ZeroVector;
        for (const FVector& P : Points)
        {
            HullCenter += P;
        }
        HullCenter /= Points.Num();

        // Make sure normal points outward
        FVector ToCenter = Center - HullCenter;
        if (FVector::DotProduct(Face.Normal, ToCenter) < 0)
        {
            // Flip face
            Swap(Face.V1, Face.V2);
            Face.Normal = -Face.Normal;
        }
    }

    return Faces;
}

bool USingleFrameReconstruction::IsTriangleOnConvexHull(
    const TArray<FVector>& Points, int32 V0, int32 V1, int32 V2)
{
    // Skip degenerate triangles
    if (V0 == V1 || V1 == V2 || V0 == V2)
    {
        return false;
    }
    
    if (V0 >= Points.Num() || V1 >= Points.Num() || V2 >= Points.Num())
    {
        return false;
    }

    FVector Normal = ComputeTriangleNormal(Points[V0], Points[V1], Points[V2]);
    if (Normal.SizeSquared() < SMALL_NUMBER)
    {
        return false; // Degenerate triangle
    }
    
    FVector Center = (Points[V0] + Points[V1] + Points[V2]) / 3.0f;
    
    // Check if all other points are on one side of this triangle
    // (or within a small tolerance for numerical stability)
    bool HasPositive = false;
    bool HasNegative = false;
    const float Tolerance = 1.0f; // cm tolerance
    
    int32 TestCount = 0;
    for (int32 i = 0; i < Points.Num() && TestCount < 50; i++) // Limit tests for performance
    {
        if (i != V0 && i != V1 && i != V2)
        {
            FVector ToPoint = Points[i] - Center;
            float Dot = FVector::DotProduct(Normal, ToPoint);
            
            if (Dot > Tolerance) HasPositive = true;
            if (Dot < -Tolerance) HasNegative = true;
            
            // If points exist on both sides, this is not a hull face
            if (HasPositive && HasNegative)
            {
                return false;
            }
            TestCount++;
        }
    }
    
    return true;
}

TArray<int32> USingleFrameReconstruction::MapHullToOriginalPoints(
    const TArray<FVector>& TransformedPoints,
    const TArray<FVector>& OriginalPoints,
    const FVector& ViewPoint)
{
    TArray<int32> Mapping;
    Mapping.Reserve(TransformedPoints.Num());

    // Map each transformed point back to original point using nearest neighbor
    for (int32 i = 0; i < TransformedPoints.Num(); i++)
    {
        if (i == TransformedPoints.Num() - 1) // Last point is viewpoint
        {
            Mapping.Add(-1); // Special marker for viewpoint
            continue;
        }

        // Find closest original point
        float MinDistance = FLT_MAX;
        int32 BestMatch = 0;
        
        // Compare in original space - transform back approximately
        FVector LocalTransformed = TransformedPoints[i];
        
        for (int32 j = 0; j < OriginalPoints.Num(); j++)
        {
            FVector LocalOriginal = OriginalPoints[j] - ViewPoint;
            
            // Simple distance comparison (could be improved with proper inverse transform)
            float Distance = (LocalTransformed.GetSafeNormal() - LocalOriginal.GetSafeNormal()).SizeSquared();
            
            if (Distance < MinDistance)
            {
                MinDistance = Distance;
                BestMatch = j;
            }
        }
        
        Mapping.Add(BestMatch);
    }

    return Mapping;
}

void USingleFrameReconstruction::FilterAndOrientFaces(
    const TArray<FTriangleFace>& HullFaces,
    const TArray<FVector>& Points,
    const FVector& ViewPoint,
    const TArray<int32>& Mapping,
    TArray<FTriangleFace>& OutFaces)
{
    OutFaces.Empty();
    OutFaces.Reserve(HullFaces.Num());

    for (const FTriangleFace& Face : HullFaces)
    {
        // Map indices back to original points
        int32 OrigV0 = (Face.V0 < Mapping.Num()) ? Mapping[Face.V0] : -1;
        int32 OrigV1 = (Face.V1 < Mapping.Num()) ? Mapping[Face.V1] : -1;
        int32 OrigV2 = (Face.V2 < Mapping.Num()) ? Mapping[Face.V2] : -1;

        // Skip faces connected to viewpoint or with invalid indices
        if (OrigV0 == -1 || OrigV1 == -1 || OrigV2 == -1 ||
            OrigV0 >= Points.Num() || OrigV1 >= Points.Num() || OrigV2 >= Points.Num() ||
            OrigV0 == OrigV1 || OrigV1 == OrigV2 || OrigV0 == OrigV2)
        {
            continue;
        }

        FTriangleFace NewFace;
        NewFace.V0 = OrigV0;
        NewFace.V1 = OrigV1;
        NewFace.V2 = OrigV2;

        FVector V0 = Points[NewFace.V0];
        FVector V1 = Points[NewFace.V1];
        FVector V2 = Points[NewFace.V2];
        
        // Check for degenerate triangle
        if (!IsValidTriangle(V0, V1, V2))
        {
            continue;
        }
        
        // Use proper normal calculation with orientation
        bool bShouldFlip;
        ComputeTriangleNormalWithOrientation(V0, V1, V2, ViewPoint, ViewPoint, 
                                           NewFace.Normal, bShouldFlip);
        
        // Flip winding order if needed
        if (bShouldFlip)
        {
            int32 Temp = NewFace.V1;
            NewFace.V1 = NewFace.V2;
            NewFace.V2 = Temp;
        }

        // Compute face weight (for pseudo-face removal)
        FVector FaceCenter = ComputeFaceCenter(V0, V1, V2);
        FVector ToCenter = (FaceCenter - ViewPoint).GetSafeNormal();
        float CosAngle = FMath::Abs(FVector::DotProduct(NewFace.Normal, ToCenter));
        NewFace.Weight = CosAngle;

        // Keep face if it's not too perpendicular to view direction
        if (CosAngle > ReconConfig.PseudoFaceThreshold)
        {
            OutFaces.Add(NewFace);
        }
    }
}

void USingleFrameReconstruction::CombineSectorResults()
{
    ReconstructedVertices.Empty();
    Triangles.Empty();
    
    int32 TotalVertices = 0;
    int32 TotalTriangles = 0;
    
    // Count total vertices and faces
    for (const FSectorData& Sector : SectorData)
    {
        TotalVertices += Sector.Points.Num();
        TotalTriangles += Sector.Faces.Num();
    }
    
    ReconstructedVertices.Reserve(TotalVertices);
    Triangles.Reserve(TotalTriangles * 3);
    
    int32 VertexOffset = 0;
    
    for (const FSectorData& Sector : SectorData)
    {
        // Add vertices
        for (const FVector& Point : Sector.Points)
        {
            ReconstructedVertices.Add(Point);
        }
        
        // Add triangles with offset
        for (const FTriangleFace& Face : Sector.Faces)
        {
            // Validate indices before adding
            if (Face.V0 + VertexOffset < TotalVertices &&
                Face.V1 + VertexOffset < TotalVertices &&
                Face.V2 + VertexOffset < TotalVertices)
            {
                Triangles.Add(Face.V0 + VertexOffset);
                Triangles.Add(Face.V1 + VertexOffset);
                Triangles.Add(Face.V2 + VertexOffset);
            }
        }
        
        VertexOffset += Sector.Points.Num();
    }
    
    // Validate final triangle indices
    ValidateTriangleIndices(Triangles, ReconstructedVertices.Num());
}

void USingleFrameReconstruction::ComputeVertexNormals(
    const TArray<FSectorData>& AllSectorData)
{
    VertexNormals.SetNum(ReconstructedVertices.Num());
    TArray<float> NormalWeights; // Use weights instead of counts
    NormalWeights.SetNum(ReconstructedVertices.Num());
    
    // Initialize
    for (int32 i = 0; i < VertexNormals.Num(); i++)
    {
        VertexNormals[i] = FVector::ZeroVector;
        NormalWeights[i] = 0.0f;
    }
    
    // Accumulate face normals with weights
    int32 VertexOffset = 0;
    for (const FSectorData& Sector : AllSectorData)
    {
        for (const FTriangleFace& Face : Sector.Faces)
        {
            int32 GlobalV0 = Face.V0 + VertexOffset;
            int32 GlobalV1 = Face.V1 + VertexOffset;
            int32 GlobalV2 = Face.V2 + VertexOffset;
            
            // Weight by face area and confidence
            FVector V0 = ReconstructedVertices[GlobalV0];
            FVector V1 = ReconstructedVertices[GlobalV1];
            FVector V2 = ReconstructedVertices[GlobalV2];
            
            float FaceArea = FVector::CrossProduct(V1 - V0, V2 - V0).Size() * 0.5f;
            float Weight = FaceArea * Face.Weight;
            
            // Validate indices and accumulate
            if (GlobalV0 < VertexNormals.Num() && GlobalV0 >= 0)
            {
                VertexNormals[GlobalV0] += Face.Normal * Weight;
                NormalWeights[GlobalV0] += Weight;
            }
            if (GlobalV1 < VertexNormals.Num() && GlobalV1 >= 0)
            {
                VertexNormals[GlobalV1] += Face.Normal * Weight;
                NormalWeights[GlobalV1] += Weight;
            }
            if (GlobalV2 < VertexNormals.Num() && GlobalV2 >= 0)
            {
                VertexNormals[GlobalV2] += Face.Normal * Weight;
                NormalWeights[GlobalV2] += Weight;
            }
        }
        VertexOffset += Sector.Points.Num();
    }
    
    // Normalize and handle vertices without faces
    for (int32 i = 0; i < VertexNormals.Num(); i++)
    {
        if (NormalWeights[i] > SMALL_NUMBER)
        {
            VertexNormals[i] = (VertexNormals[i] / NormalWeights[i]).GetSafeNormal();
        }
        else
        {
            // For vertices without faces, use default up vector
            // Original code would use ray from viewpoint, but this is simpler
            VertexNormals[i] = FVector::UpVector;
            
            UE_LOG(LogTemp, VeryVerbose, TEXT("Vertex %d has no associated faces, using default normal"), i);
        }
    }
}

void USingleFrameReconstruction::ApplyToProceduralMesh(
    UProceduralMeshComponent* MeshComponent)
{
    if (!MeshComponent)
    {
        UE_LOG(LogTemp, Error, TEXT("MeshComponent is null in ApplyToProceduralMesh!"));
        return;
    }
    
    if (ReconstructedVertices.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("No vertices to apply to mesh!"));
        MeshComponent->ClearAllMeshSections();
        return;
    }
    
    if (Triangles.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("No triangles to apply to mesh!"));
        MeshComponent->ClearAllMeshSections();
        return;
    }
    
    // Clear existing mesh
    MeshComponent->ClearAllMeshSections();
    
    // Create UV coordinates (simple planar projection)
    TArray<FVector2D> UVs;
    UVs.Reserve(ReconstructedVertices.Num());
    
    FVector MinBounds, MaxBounds;
    ComputeBoundingBox(ReconstructedVertices, MinBounds, MaxBounds);
    FVector BoundsSize = MaxBounds - MinBounds;
    
    for (const FVector& Vertex : ReconstructedVertices)
    {
        float U = (BoundsSize.X > 0) ? (Vertex.X - MinBounds.X) / BoundsSize.X : 0.5f;
        float V = (BoundsSize.Y > 0) ? (Vertex.Y - MinBounds.Y) / BoundsSize.Y : 0.5f;
        UVs.Add(FVector2D(U, V));
    }
    
    // Create tangents
    TArray<FProcMeshTangent> Tangents;
    Tangents.Reserve(VertexNormals.Num());
    
    for (const FVector& Normal : VertexNormals)
    {
        FVector Tangent = FVector::CrossProduct(Normal, FVector::UpVector).GetSafeNormal();
        if (Tangent.IsNearlyZero())
        {
            Tangent = FVector::CrossProduct(Normal, FVector::ForwardVector).GetSafeNormal();
        }
        Tangents.Add(FProcMeshTangent(Tangent, false));
    }
    
    // Create vertex colors (color by height)
    TArray<FLinearColor> VertexColors;
    VertexColors.Reserve(ReconstructedVertices.Num());
    
    float MinZ = MinBounds.Z;
    float MaxZ = MaxBounds.Z;
    
    for (const FVector& Vertex : ReconstructedVertices)
    {
        float NormalizedHeight = (MaxZ > MinZ) ? (Vertex.Z - MinZ) / (MaxZ - MinZ) : 0.5f;
        VertexColors.Add(FLinearColor::LerpUsingHSV(
            FLinearColor::Blue, FLinearColor::Red, NormalizedHeight));
    }
    
    // Create mesh section
    MeshComponent->CreateMeshSection_LinearColor(
        0, // Section index
        ReconstructedVertices,
        Triangles,
        VertexNormals,
        UVs,
        VertexColors,
        Tangents,
        false // Don't create collision for performance
    );
    
    // Set material if available
    if (MeshComponent->GetMaterial(0) == nullptr)
    {
        FString MaterialPath = TEXT("/VCCSim/Materials/M_Dynamic_mesh.M_Dynamic_mesh");
        auto MeshMaterial = Cast<UMaterialInterface>(StaticLoadObject(
            UMaterialInterface::StaticClass(), nullptr, *MaterialPath));

        if (MeshMaterial)
        {
            MeshComponent->SetMaterial(0, MeshMaterial);
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("Failed to load material at %s"), *MaterialPath);
        }
    }
    
    UE_LOG(LogTemp, Log, TEXT("Applied mesh with %d vertices and %d triangles"), 
        ReconstructedVertices.Num(), Triangles.Num() / 3);
}

void USingleFrameReconstruction::ClearReconstruction()
{
    ReconstructedVertices.Empty();
    VertexNormals.Empty();
    Triangles.Empty();
    SectorData.Empty();
}

// Legacy methods for compatibility
TArray<FVector> USingleFrameReconstruction::TransformPointsForGHPR(
    const TArray<FVector>& Points, const FVector& ViewPoint, float& OutRadius)
{
    TArray<FVector> TransformedPoints;
    TArray<float> Distances;
    TransformPointsForGHPR_Proper(Points, ViewPoint, TransformedPoints, Distances, OutRadius);
    return TransformedPoints;
}

TArray<FTriangleFace> USingleFrameReconstruction::ComputeConvexHull(
    const TArray<FVector>& TransformedPoints)
{
    return ComputeConvexHull_Proper(TransformedPoints);
}

void USingleFrameReconstruction::ComputeFaceNormals(
    TArray<FTriangleFace>& Faces, const TArray<FVector>& Points, const FVector& ViewPoint)
{
    for (FTriangleFace& Face : Faces)
    {
        if (Face.V0 < Points.Num() && Face.V1 < Points.Num() && Face.V2 < Points.Num())
        {
            FVector V0 = Points[Face.V0];
            FVector V1 = Points[Face.V1];
            FVector V2 = Points[Face.V2];
            
            bool bShouldFlip;
            ComputeTriangleNormalWithOrientation(V0, V1, V2, ViewPoint, ViewPoint,
                                               Face.Normal, bShouldFlip);
            
            if (bShouldFlip)
            {
                // Swap winding order
                int32 Temp = Face.V1;
                Face.V1 = Face.V2;
                Face.V2 = Temp;
            }
        }
    }
}

void USingleFrameReconstruction::RemovePseudoFaces(
    TArray<FTriangleFace>& Faces, const TArray<FVector>& Points, const FVector& ViewPoint)
{
    TArray<FTriangleFace> ValidFaces;
    ValidFaces.Reserve(Faces.Num());
    
    for (FTriangleFace& Face : Faces)
    {
        if (Face.V0 < Points.Num() && Face.V1 < Points.Num() && Face.V2 < Points.Num())
        {
            FVector FaceCenter = ComputeFaceCenter(Points[Face.V0], Points[Face.V1], Points[Face.V2]);
            FVector ToCenter = (FaceCenter - ViewPoint).GetSafeNormal();
            
            // Compute angle between normal and view direction
            float CosAngle = FMath::Abs(FVector::DotProduct(Face.Normal, ToCenter));
            Face.Weight = CosAngle;
            
            // Keep face if it's not too perpendicular to view direction
            if (CosAngle > ReconConfig.PseudoFaceThreshold)
            {
                ValidFaces.Add(Face);
            }
        }
    }
    
    Faces = ValidFaces;
}

TArray<FTriangleFace> USingleFrameReconstruction::ComputeConvexHullProper(
    const TArray<FVector>& Points, int32 ViewPointIndex)
{
    TArray<FTriangleFace> Faces = ComputeConvexHull_Proper(Points);
    
    // Remove faces connected to viewpoint if requested
    if (ViewPointIndex >= 0)
    {
        Faces.RemoveAll([ViewPointIndex](const FTriangleFace& Face)
        {
            return Face.V0 == ViewPointIndex || 
                   Face.V1 == ViewPointIndex || 
                   Face.V2 == ViewPointIndex;
        });
    }
    
    return Faces;
}

// Helper methods
FVector USingleFrameReconstruction::ComputeTriangleNormal(
    const FVector& V0, const FVector& V1, const FVector& V2)
{
    FVector EdgeA = V1 - V0;  // oAvec in original
    FVector EdgeB = V2 - V1;  // oBvec in original  
    return FVector::CrossProduct(EdgeA, EdgeB).GetSafeNormal();
}

void USingleFrameReconstruction::ComputeTriangleNormalWithOrientation(
    const FVector& V0, const FVector& V1, const FVector& V2,
    const FVector& ViewPoint, const FVector& ReferencePoint,
    FVector& OutNormal, bool& bShouldFlipWinding)
{
    // Calculate normal using original method
    FVector EdgeA = V1 - V0;
    FVector EdgeB = V2 - V1;
    FVector Normal = FVector::CrossProduct(EdgeA, EdgeB);
    
    float Length = Normal.Size();
    if (Length < SMALL_NUMBER)
    {
        OutNormal = FVector::UpVector;
        bShouldFlipWinding = false;
        return;
    }
    
    Normal = Normal / Length; // Normalize
    
    // Compute D parameter: D = -(A*x + B*y + C*z) using V0
    float DParam = -FVector::DotProduct(Normal, V0);
    
    // Check orientation relative to reference point (usually viewpoint)
    float Distance = FVector::DotProduct(Normal, ReferencePoint) + DParam;
    
    // If reference point is on positive side, flip normal
    bShouldFlipWinding = (Distance > 0);
    
    if (bShouldFlipWinding)
    {
        Normal = -Normal;
        DParam = -DParam;
    }
    
    OutNormal = Normal;
}


FVector USingleFrameReconstruction::ComputeFaceCenter(
    const FVector& V0, const FVector& V1, const FVector& V2)
{
    return (V0 + V1 + V2) / 3.0f;
}

float USingleFrameReconstruction::ComputeAngleCosine(const FVector& V1, const FVector& V2)
{
    return FVector::DotProduct(V1.GetSafeNormal(), V2.GetSafeNormal());
}

int32 USingleFrameReconstruction::GetSectorIndex(
    const FVector& Point, const FVector& Origin, int32 NumSectors)
{
    FVector RelativePoint = Point - Origin;
    float Angle = FMath::Atan2(RelativePoint.Y, RelativePoint.X);
    
    // Convert to [0, 2PI]
    if (Angle < 0)
    {
        Angle += 2.0f * PI;
    }
    
    float SectorWidth = 2.0f * PI / float(NumSectors);
    int32 SectorIndex = FMath::FloorToInt(Angle / SectorWidth);
    
    // Handle edge case
    if (SectorIndex >= NumSectors)
    {
        SectorIndex = NumSectors - 1;
    }
    
    return SectorIndex;
}


bool USingleFrameReconstruction::FindInitialTetrahedron(
    const TArray<FVector>& Points, TArray<int32>& OutIndices)
{
    OutIndices.Empty();
    
    // Find points with extreme coordinates
    int32 MinX = 0, MaxX = 0, MinY = 0, MaxY = 0, MinZ = 0, MaxZ = 0;
    
    for (int32 i = 1; i < Points.Num(); i++)
    {
        if (Points[i].X < Points[MinX].X) MinX = i;
        if (Points[i].X > Points[MaxX].X) MaxX = i;
        if (Points[i].Y < Points[MinY].Y) MinY = i;
        if (Points[i].Y > Points[MaxY].Y) MaxY = i;
        if (Points[i].Z < Points[MinZ].Z) MinZ = i;
        if (Points[i].Z > Points[MaxZ].Z) MaxZ = i;
    }

    // Get unique points
    TSet<int32> UniqueIndices;
    UniqueIndices.Add(MinX);
    UniqueIndices.Add(MaxX);
    UniqueIndices.Add(MinY);
    UniqueIndices.Add(MaxY);
    UniqueIndices.Add(MinZ);
    UniqueIndices.Add(MaxZ);

    if (UniqueIndices.Num() < 4)
    {
        // Add more points based on distance
        FVector Center = FVector::ZeroVector;
        for (int32 idx : UniqueIndices)
        {
            Center += Points[idx];
        }
        Center /= UniqueIndices.Num();

        TArray<TPair<float, int32>> Distances;
        for (int32 i = 0; i < Points.Num(); i++)
        {
            if (!UniqueIndices.Contains(i))
            {
                float Dist = (Points[i] - Center).SizeSquared();
                Distances.Add(TPair<float, int32>(Dist, i));
            }
        }

        Distances.Sort([](const TPair<float, int32>& A, const TPair<float, int32>& B) {
            return A.Key > B.Key;
        });

        for (const auto& Pair : Distances)
        {
            UniqueIndices.Add(Pair.Value);
            if (UniqueIndices.Num() >= 4)
                break;
        }
    }

    OutIndices = UniqueIndices.Array();
    
    // Make sure we have exactly 4 points
    if (OutIndices.Num() > 4)
    {
        OutIndices.SetNum(4);
    }

    return OutIndices.Num() == 4;
}

FTriangleFace USingleFrameReconstruction::CreateFace(
    int32 V0, int32 V1, int32 V2, const TArray<FVector>& Points)
{
    FTriangleFace Face;
    Face.V0 = V0;
    Face.V1 = V1;
    Face.V2 = V2;
    Face.Normal = ComputeTriangleNormal(Points[V0], Points[V1], Points[V2]);
    Face.Weight = 1.0f;
    return Face;
}

void USingleFrameReconstruction::AddEdge(TMap<FIntPoint, int32>& EdgeCount, int32 V0, int32 V1)
{
    FIntPoint Edge(FMath::Min(V0, V1), FMath::Max(V0, V1));
    if (EdgeCount.Contains(Edge))
    {
        EdgeCount[Edge]++;
    }
    else
    {
        EdgeCount.Add(Edge, 1);
    }
}

// Utility methods
void USingleFrameReconstruction::ValidateTriangleIndices(TArray<int32>& InTriangles, int32 VertexCount)
{
    TArray<int32> ValidTriangles;
    ValidTriangles.Reserve(InTriangles.Num());
    
    for (int32 i = 0; i < InTriangles.Num(); i += 3)
    {
        if (i + 2 < InTriangles.Num())
        {
            int32 V0 = InTriangles[i];
            int32 V1 = InTriangles[i + 1];
            int32 V2 = InTriangles[i + 2];
            
            // Check bounds and degeneracy
            if (V0 >= 0 && V0 < VertexCount &&
                V1 >= 0 && V1 < VertexCount &&
                V2 >= 0 && V2 < VertexCount &&
                V0 != V1 && V1 != V2 && V0 != V2)
            {
                ValidTriangles.Add(V0);
                ValidTriangles.Add(V1);
                ValidTriangles.Add(V2);
            }
        }
    }
    
    InTriangles = ValidTriangles;
}

void USingleFrameReconstruction::ComputeBoundingBox(
    const TArray<FVector>& Vertices, FVector& OutMin, FVector& OutMax)
{
    if (Vertices.Num() == 0)
    {
        OutMin = OutMax = FVector::ZeroVector;
        return;
    }
    
    OutMin = OutMax = Vertices[0];
    
    for (const FVector& Vertex : Vertices)
    {
        OutMin.X = FMath::Min(OutMin.X, Vertex.X);
        OutMin.Y = FMath::Min(OutMin.Y, Vertex.Y);
        OutMin.Z = FMath::Min(OutMin.Z, Vertex.Z);
        
        OutMax.X = FMath::Max(OutMax.X, Vertex.X);
        OutMax.Y = FMath::Max(OutMax.Y, Vertex.Y);
        OutMax.Z = FMath::Max(OutMax.Z, Vertex.Z);
    }
}

bool USingleFrameReconstruction::IsValidTriangle(
    const FVector& V0, const FVector& V1, const FVector& V2, float MinArea)
{
    FVector Edge1 = V1 - V0;
    FVector Edge2 = V2 - V0;
    FVector CrossProduct = FVector::CrossProduct(Edge1, Edge2);
    float Area = CrossProduct.Size() * 0.5f;
    
    return Area > MinArea;
}

// Debug and validation methods
void USingleFrameReconstruction::LogReconstructionStats() const
{
    UE_LOG(LogTemp, Warning, TEXT("=== Reconstruction Statistics ==="));
    UE_LOG(LogTemp, Warning, TEXT("Last reconstruction time: %.3f seconds"), LastReconstructionTime);
    UE_LOG(LogTemp, Warning, TEXT("Points processed: %d"), LastProcessedPointCount);
    UE_LOG(LogTemp, Warning, TEXT("Vertices generated: %d"), ReconstructedVertices.Num());
    UE_LOG(LogTemp, Warning, TEXT("Triangles generated: %d"), LastGeneratedTriangleCount);
    UE_LOG(LogTemp, Warning, TEXT("Sectors used: %d"), SectorData.Num());
    
    int32 ActiveSectors = 0;
    for (const FSectorData& Sector : SectorData)
    {
        if (Sector.Faces.Num() > 0) ActiveSectors++;
    }
    UE_LOG(LogTemp, Warning, TEXT("Active sectors: %d"), ActiveSectors);
    
    if (LastReconstructionTime > 0)
    {
        float PointsPerSecond = LastProcessedPointCount / LastReconstructionTime;
        UE_LOG(LogTemp, Warning, TEXT("Processing rate: %.1f points/second"), PointsPerSecond);
    }
}

void USingleFrameReconstruction::ValidateReconstructionData() const
{
    bool bIsValid = true;
    
    // Check vertex array
    if (ReconstructedVertices.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("Validation: No vertices generated"));
        bIsValid = false;
    }
    
    // Check normal array size
    if (VertexNormals.Num() != ReconstructedVertices.Num())
    {
        UE_LOG(LogTemp, Error, TEXT("Validation: Normal count mismatch - Vertices: %d, Normals: %d"), 
            ReconstructedVertices.Num(), VertexNormals.Num());
        bIsValid = false;
    }
    
    // Check triangle indices
    if (Triangles.Num() % 3 != 0)
    {
        UE_LOG(LogTemp, Error, TEXT("Validation: Triangle array size not multiple of 3: %d"), Triangles.Num());
        bIsValid = false;
    }
    
    // Check triangle index bounds
    for (int32 i = 0; i < Triangles.Num(); i++)
    {
        if (Triangles[i] >= ReconstructedVertices.Num() || Triangles[i] < 0)
        {
            UE_LOG(LogTemp, Error, TEXT("Validation: Invalid triangle index %d at position %d (vertex count: %d)"), 
                Triangles[i], i, ReconstructedVertices.Num());
            bIsValid = false;
            break;
        }
    }
    
    // Check for degenerate triangles
    int32 DegenerateCount = 0;
    for (int32 i = 0; i < Triangles.Num(); i += 3)
    {
        if (i + 2 < Triangles.Num())
        {
            int32 V0 = Triangles[i];
            int32 V1 = Triangles[i + 1];
            int32 V2 = Triangles[i + 2];
            
            if (V0 == V1 || V1 == V2 || V0 == V2)
            {
                DegenerateCount++;
            }
        }
    }
    
    if (DegenerateCount > 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("Validation: Found %d degenerate triangles"), DegenerateCount);
    }
    
    if (bIsValid)
    {
        UE_LOG(LogTemp, Log, TEXT("Validation: Reconstruction data is valid"));
    }
}

void USingleFrameReconstruction::DrawDebugSectors(UWorld* World, const FVector& ViewPoint, float Duration) const
{
    if (!World) return;
    
    float SectorAngle = 2.0f * PI / ReconConfig.SectorNum;
    float Radius = 1000.0f; // cm
    
    for (int32 i = 0; i < ReconConfig.SectorNum; i++)
    {
        float StartAngle = i * SectorAngle;
        float EndAngle = (i + 1) * SectorAngle;
        
        FVector Start = ViewPoint + FVector(
            FMath::Cos(StartAngle) * Radius,
            FMath::Sin(StartAngle) * Radius,
            0);
            
        FVector End = ViewPoint + FVector(
            FMath::Cos(EndAngle) * Radius,
            FMath::Sin(EndAngle) * Radius,
            0);
        
        // Draw sector boundary
        DrawDebugLine(World, ViewPoint, Start, FColor::Yellow, false, Duration, 0, 2.0f);
        DrawDebugLine(World, ViewPoint, End, FColor::Yellow, false, Duration, 0, 2.0f);
        
        // Draw arc
        int32 ArcSegments = 10;
        for (int32 j = 0; j < ArcSegments; j++)
        {
            float Alpha1 = float(j) / ArcSegments;
            float Alpha2 = float(j + 1) / ArcSegments;
            
            float Angle1 = FMath::Lerp(StartAngle, EndAngle, Alpha1);
            float Angle2 = FMath::Lerp(StartAngle, EndAngle, Alpha2);
            
            FVector Arc1 = ViewPoint + FVector(
                FMath::Cos(Angle1) * Radius,
                FMath::Sin(Angle1) * Radius,
                0);
                
            FVector Arc2 = ViewPoint + FVector(
                FMath::Cos(Angle2) * Radius,
                FMath::Sin(Angle2) * Radius,
                0);
            
            DrawDebugLine(World, Arc1, Arc2, FColor::Green, false, Duration, 0, 1.0f);
        }
        
        // Label sector
        FVector LabelPos = ViewPoint + FVector(
            FMath::Cos(StartAngle + SectorAngle * 0.5f) * Radius * 0.7f,
            FMath::Sin(StartAngle + SectorAngle * 0.5f) * Radius * 0.7f,
            50.0f);
            
        DrawDebugString(World, LabelPos, FString::Printf(TEXT("S%d"), i), 
            nullptr, FColor::White, Duration);
    }
}

void USingleFrameReconstruction::DrawDebugNormals(UWorld* World, float Scale, float Duration) const
{
    if (!World) return;
    
    for (int32 i = 0; i < ReconstructedVertices.Num() && i < VertexNormals.Num(); i++)
    {
        FVector Start = ReconstructedVertices[i];
        FVector End = Start + VertexNormals[i] * Scale;
        
        DrawDebugLine(World, Start, End, FColor::Blue, false, Duration, 0, 1.0f);
        DrawDebugPoint(World, Start, 3.0f, FColor::Red, false, Duration);
    }
}