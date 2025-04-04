#pragma once

#include "CoreMinimal.h"
#include "DataType/DataMesh.h"
#include "SceneAnalysisManager.generated.h"

class URGBCameraComponent;

struct FCoverageData
{
    float CoveragePercentage;
    TSet<int32> VisibleMeshIDs;
    TArray<FVector> VisiblePoints;
    int32 TotalVisibleTriangles;
};

UCLASS(BlueprintType, Blueprintable)
class VCCSIM_API USceneAnalysisManager : public UObject
{
    GENERATED_BODY()

public:
    USceneAnalysisManager();
    
    bool Initialize(UWorld* InWorld, FString&& Path);
    void ScanScene();
    
    void RegisterCamera(URGBCameraComponent* CameraComponent);
    
    FMeshInfo GetMeshInfo(int32 MeshID) const;
    TArray<FMeshInfo> GetAllMeshInfo() const;
    
    UFUNCTION(BlueprintCallable, Category = "SceneAnalysis")
    void UpdateAccumulatedCoverage(const FTransform& Transform, const FString& Name);
    float GetTotalCoveragePercentage() const;
    void ResetCoverage();
    void VisualizeCoverage(bool bShowVisiblePoints,
        bool bHighlightCoveredMeshes, float Duration = 5.0f);

    static void ExtractMeshData(UStaticMeshComponent* MeshComponent, FMeshInfo& OutMeshInfo);

private:

    void ConstructFrustum(FConvexVolume& OutFrustum,
        const FTransform& CameraPose, const FMatrix44f& CameraIntrinsic);

    bool IsPointVisibleFromCamera(const FVector& Point, const FTransform& CameraPose) const;
    
    TArray<FVector> SamplePointsOnMesh(const FMeshInfo& MeshInfo, int32 SamplesPerTriangle = 1);

private:
    UPROPERTY()
    UWorld* World;
    TMap<FString, FMatrix44f> CameraIntrinsics;
    TArray<FMeshInfo> SceneMeshes;
    TMap<FVector, bool> CoverageMap;
    int32 TotalPointsInScene;
    int32 TotalTrianglesInScene;
    TSet<int32> CurrentlyVisibleMeshIDs;
    float CurrentCoveragePercentage;
    float SamplingDensity;
    bool bUseVertexSampling;
    float GridResolution;

    /* ----------------------------- Test ----------------------------- */
    FString LogPath;
    void ExportMeshesToPly();
    FString GeneratePlyContent(const FMeshInfo& MeshInfo);
    void VisualizeSceneMeshes(float Duration, bool bShowWireframe,
        bool bShowVertices, float VertexSize);
};