#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "ProceduralMeshComponent.h"
#include "SingleFrameReconstruction.generated.h"

USTRUCT(BlueprintType)
struct FReconstructionConfig
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, Category = "Reconstruction")
    float GHPRParam = 3.6f;

    UPROPERTY(EditAnywhere, Category = "Reconstruction")
    float PseudoFaceThreshold = 0.01f;

    UPROPERTY(EditAnywhere, Category = "Reconstruction")
    int32 SectorNum = 8;

    UPROPERTY(EditAnywhere, Category = "Reconstruction")
    int32 MinPointsPerSector = 5;

    UPROPERTY(EditAnywhere, Category = "Reconstruction")
    float ViewpointElevation = 0.0f;

    UPROPERTY(EditAnywhere, Category = "Reconstruction")
    bool bMultiThread = true;
};

USTRUCT(BlueprintType)
struct FTriangleFace
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    int32 V0;
    
    UPROPERTY(BlueprintReadOnly)
    int32 V1;
    
    UPROPERTY(BlueprintReadOnly)
    int32 V2;
    
    UPROPERTY(BlueprintReadOnly)
    FVector Normal;
    
    UPROPERTY(BlueprintReadOnly)
    float Weight;

    FTriangleFace()
    {
        V0 = V1 = V2 = 0;
        Normal = FVector::ZeroVector;
        Weight = 1.0f;
    }

    FTriangleFace(int32 InV0, int32 InV1, int32 InV2)
    {
        V0 = InV0;
        V1 = InV1;
        V2 = InV2;
        Normal = FVector::ZeroVector;
        Weight = 1.0f;
    }
};

USTRUCT(BlueprintType)
struct FSectorData
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    TArray<FVector> Points;
    
    UPROPERTY(BlueprintReadOnly)
    TArray<FTriangleFace> Faces;
    
    UPROPERTY(BlueprintReadOnly)
    TArray<int32> PointIndices;

    FSectorData()
    {
        Points.Empty();
        Faces.Empty();
        PointIndices.Empty();
    }
};

// Forward declarations
class ULidarComponent;

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class VCCSIM_API USingleFrameReconstruction : public UActorComponent
{
    GENERATED_BODY()

public:
    USingleFrameReconstruction();

    // Main reconstruction functions
    UFUNCTION(BlueprintCallable, Category = "Reconstruction")
    void DoSingleRecon();
    
    UFUNCTION(BlueprintCallable, Category = "Reconstruction")
    void ReconstructFrame(const TArray<FVector>& PointCloud, const FVector& ViewPoint);

    UFUNCTION(BlueprintCallable, Category = "Reconstruction")
    void ApplyToProceduralMesh(UProceduralMeshComponent* MeshComponent);

    UFUNCTION(BlueprintCallable, Category = "Reconstruction")
    void ClearReconstruction();

    // Configuration
    UFUNCTION(BlueprintCallable, Category = "Reconstruction")
    void SetConfiguration(const FReconstructionConfig& Config) { ReconConfig = Config; }

    UFUNCTION(BlueprintCallable, Category = "Reconstruction")
    FReconstructionConfig GetConfiguration() const { return ReconConfig; }

    // Get reconstruction results
    UFUNCTION(BlueprintCallable, Category = "Reconstruction")
    TArray<FVector> GetReconstructedVertices() const { return ReconstructedVertices; }

    UFUNCTION(BlueprintCallable, Category = "Reconstruction")
    TArray<FVector> GetVertexNormals() const { return VertexNormals; }

    UFUNCTION(BlueprintCallable, Category = "Reconstruction")
    TArray<int32> GetTriangles() const { return Triangles; }

    UFUNCTION(BlueprintCallable, Category = "Reconstruction")
    TArray<FSectorData> GetSectorData() const { return SectorData; }

    UFUNCTION(BlueprintCallable, Category = "Reconstruction")
    int32 GetTriangleCount() const { return Triangles.Num() / 3; }

    UFUNCTION(BlueprintCallable, Category = "Reconstruction")
    int32 GetVertexCount() const { return ReconstructedVertices.Num(); }

    // Component setup
    UFUNCTION(BlueprintCallable, Category = "Reconstruction")
    void SetLiDAR(ULidarComponent* InLidarComponent)
    {
        LidarComponent = InLidarComponent;
    }

    UFUNCTION(BlueprintCallable, Category = "Reconstruction")
    void SetProceduralMeshComponent(UProceduralMeshComponent* InMeshComponent)
    {
        ProceduralMeshComponent = InMeshComponent;
    }

protected:
    virtual void BeginPlay() override;

private:
    // Configuration
    UPROPERTY(EditAnywhere, Category = "Reconstruction")
    FReconstructionConfig ReconConfig;

    // Reconstruction results
    UPROPERTY(VisibleAnywhere, Category = "Reconstruction Results")
    TArray<FVector> ReconstructedVertices;
    
    UPROPERTY(VisibleAnywhere, Category = "Reconstruction Results")
    TArray<FVector> VertexNormals;
    
    UPROPERTY(VisibleAnywhere, Category = "Reconstruction Results")
    TArray<int32> Triangles;
    
    UPROPERTY(VisibleAnywhere, Category = "Reconstruction Results")
    TArray<FSectorData> SectorData;

    // Component references
    UPROPERTY()
    ULidarComponent* LidarComponent = nullptr;
    
    UPROPERTY()
    UProceduralMeshComponent* ProceduralMeshComponent = nullptr;

    // Core reconstruction pipeline methods
    void DivideSectors(const TArray<FVector>& PointCloud, const FVector& ViewPoint);
    void ProcessSector(int32 SectorIndex, const FVector& ViewPoint);
    void CombineSectorResults();
    void ComputeVertexNormals(const TArray<FSectorData>& AllSectorData);

    // GHPR Algorithm implementation
    void TransformPointsForGHPR_Proper(
        const TArray<FVector>& Points, 
        const FVector& ViewPoint,
        TArray<FVector>& OutTransformedPoints,
        TArray<float>& OutDistances,
        float& OutRadius);

    // Convex Hull computation
    TArray<FTriangleFace> ComputeConvexHull_Proper(const TArray<FVector>& Points);
    bool IsTriangleOnConvexHull(const TArray<FVector>& Points, int32 V0, int32 V1, int32 V2);

    // Point mapping and face processing
    TArray<int32> MapHullToOriginalPoints(
        const TArray<FVector>& TransformedPoints,
        const TArray<FVector>& OriginalPoints,
        const FVector& ViewPoint);

    void FilterAndOrientFaces(
        const TArray<FTriangleFace>& HullFaces,
        const TArray<FVector>& Points,
        const FVector& ViewPoint,
        const TArray<int32>& Mapping,
        TArray<FTriangleFace>& OutFaces);

    // Legacy methods (kept for compatibility, but improved internally)
    TArray<FVector> TransformPointsForGHPR(const TArray<FVector>& Points, const FVector& ViewPoint, float& OutRadius);
    TArray<FTriangleFace> ComputeConvexHull(const TArray<FVector>& TransformedPoints);
    void ComputeFaceNormals(TArray<FTriangleFace>& Faces, const TArray<FVector>& Points, const FVector& ViewPoint);
    void RemovePseudoFaces(TArray<FTriangleFace>& Faces, const TArray<FVector>& Points, const FVector& ViewPoint);
    TArray<FTriangleFace> ComputeConvexHullProper(const TArray<FVector>& Points, int32 ViewPointIndex);

    // Utility and helper methods
    FVector ComputeTriangleNormal(const FVector& V0, const FVector& V1, const FVector& V2);
    void ComputeTriangleNormalWithOrientation(
        const FVector& V0, const FVector& V1, const FVector& V2,
        const FVector& ViewPoint, const FVector& ReferencePoint,
        FVector& OutNormal, bool& bShouldFlipWinding);
    FVector ComputeFaceCenter(const FVector& V0, const FVector& V1, const FVector& V2);
    float ComputeAngleCosine(const FVector& V1, const FVector& V2);
    int32 GetSectorIndex(const FVector& Point, const FVector& Origin, int32 NumSectors);
    bool FindInitialTetrahedron(const TArray<FVector>& Points, TArray<int32>& OutIndices);
    FTriangleFace CreateFace(int32 V0, int32 V1, int32 V2, const TArray<FVector>& Points);
    void AddEdge(TMap<FIntPoint, int32>& EdgeCount, int32 V0, int32 V1);
    
    // Mesh processing utilities
    void ValidateTriangleIndices(TArray<int32>& InTriangles, int32 VertexCount);
    void ComputeBoundingBox(const TArray<FVector>& Vertices, FVector& OutMin, FVector& OutMax);
    bool IsValidTriangle(const FVector& V0, const FVector& V1, const FVector& V2, float MinArea = 0.01f);

    // Debugging and statistics
    UFUNCTION(BlueprintCallable, Category = "Reconstruction Debug")
    void LogReconstructionStats() const;

    UFUNCTION(BlueprintCallable, Category = "Reconstruction Debug")
    void ValidateReconstructionData() const;

public:
    // Debug visualization helpers
    UFUNCTION(BlueprintCallable, Category = "Reconstruction Debug")
    void DrawDebugSectors(UWorld* World, const FVector& ViewPoint, float Duration = 5.0f) const;

    UFUNCTION(BlueprintCallable, Category = "Reconstruction Debug")
    void DrawDebugNormals(UWorld* World, float Scale = 10.0f, float Duration = 5.0f) const;

    // Performance monitoring
    UPROPERTY(BlueprintReadOnly, Category = "Performance")
    float LastReconstructionTime = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "Performance")
    int32 LastProcessedPointCount = 0;

    UPROPERTY(BlueprintReadOnly, Category = "Performance")
    int32 LastGeneratedTriangleCount = 0;
};