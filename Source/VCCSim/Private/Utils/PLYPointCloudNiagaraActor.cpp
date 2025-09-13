#include "Utils/PLYPointCloudNiagaraActor.h"
#include "NiagaraDataInterfaceArrayFunctionLibrary.h"
#include "IO/PLYUtils.h"

DEFINE_LOG_CATEGORY_STATIC(LogPLYPointCloudNiagaraActor, Log, All);

APLYPointCloudNiagaraActor::APLYPointCloudNiagaraActor()
{
    PrimaryActorTick.bCanEverTick = false;
    
    NiagaraComp = CreateDefaultSubobject<UNiagaraComponent>(TEXT("NiagaraComp"));
    RootComponent = NiagaraComp;
    
    if (NiagaraComp)
    {
        NiagaraComp->bAutoActivate = false;
        NiagaraComp->SetMobility(EComponentMobility::Movable);
    }
}

bool APLYPointCloudNiagaraActor::LoadPly(TArray<FVector>& OutPositions, TArray<FLinearColor>& OutColors)
{
    if (PlyFilePath.IsEmpty())
    {
        UE_LOG(LogPLYPointCloudNiagaraActor, Error, TEXT("PLY file path is empty"));
        return false;
    }

    // Use VCCSim's existing PLY loader
    FPLYLoader::FPLYLoadResult LoadResult = FPLYLoader::LoadPLYFile(PlyFilePath, FLinearColor::White);
    
    if (!LoadResult.bSuccess || LoadResult.PointCount == 0)
    {
        UE_LOG(LogPLYPointCloudNiagaraActor, Error, TEXT("Failed to load PLY file: %s"), *PlyFilePath);
        return false;
    }

    OutPositions.Reserve(LoadResult.PointCount);
    OutColors.Reserve(LoadResult.PointCount);
    
    // Check point count and downsample if necessary
    const int32 MaxPointsLimit = 100000;
    TArray<FRatPoint> ProcessedPoints = LoadResult.Points;
    
    if (LoadResult.PointCount > MaxPointsLimit)
    {
        UE_LOG(LogPLYPointCloudNiagaraActor, Warning, TEXT("Point cloud has %d points, exceeding limit of %d. Downsampling..."), 
               LoadResult.PointCount, MaxPointsLimit);
        
        // Perform uniform downsampling
        TArray<FRatPoint> DownsampledPoints;
        const int32 Step = FMath::Max(1, LoadResult.PointCount / MaxPointsLimit);
        
        DownsampledPoints.Reserve(MaxPointsLimit);
        for (int32 i = 0; i < LoadResult.Points.Num() && DownsampledPoints.Num() < MaxPointsLimit; i += Step)
        {
            DownsampledPoints.Add(LoadResult.Points[i]);
        }
        
        ProcessedPoints = MoveTemp(DownsampledPoints);
        
        UE_LOG(LogPLYPointCloudNiagaraActor, Log, TEXT("Downsampled point cloud to %d points (%.1f%% reduction)"), 
               ProcessedPoints.Num(),
               (1.0f - (float)ProcessedPoints.Num() / (float)LoadResult.PointCount) * 100.0f);
    }

    for (const FRatPoint& Point : ProcessedPoints)
    {
        // Apply unit scale conversion (e.g., meters to centimeters)
        FVector ScaledPosition = Point.Position * UnitScaleToUE;
        OutPositions.Add(ScaledPosition);
        
        // Use point color (all points have color, may be default white)
        OutColors.Add(Point.Color);
    }

    UE_LOG(LogPLYPointCloudNiagaraActor, Log, TEXT("Final PLY data: %d points (Colors: %s)"), 
           OutPositions.Num(),
           LoadResult.bHasColors ? TEXT("Yes") : TEXT("No"));

    return true;
}

void APLYPointCloudNiagaraActor::BeginPlay()
{
    Super::BeginPlay();
    
    if (!NiagaraSystemAsset)
    {
        UE_LOG(LogPLYPointCloudNiagaraActor, Error, TEXT("NiagaraSystemAsset is not set!"));
        return;
    }
    
    if (!NiagaraComp)
    {
        UE_LOG(LogPLYPointCloudNiagaraActor, Error, TEXT("NiagaraComp is not valid!"));
        return;
    }
    
    NiagaraComp->SetAsset(NiagaraSystemAsset);
    NiagaraComp->Activate(true);

    // Load and set point cloud data
    TArray<FVector> Positions;
    TArray<FLinearColor> Colors;
    
    if (!LoadPly(Positions, Colors))
    {
        return;
    }

    if (Positions.Num() == 0)
    {
        UE_LOG(LogPLYPointCloudNiagaraActor, Warning, TEXT("No points loaded from PLY file"));
        return;
    }

    // Set Niagara parameters using the Array Data Interface
    UNiagaraDataInterfaceArrayFunctionLibrary::SetNiagaraArrayVector(NiagaraComp, FName(TEXT("Points")), Positions);
    
    if (Colors.Num() == Positions.Num())
    {
        // Convert LinearColor to Vector for Niagara (RGB as XYZ)
        TArray<FVector> ColorVectors;
        ColorVectors.Reserve(Colors.Num());
        for (const FLinearColor& Color : Colors)
        {
            ColorVectors.Add(FVector(Color.R, Color.G, Color.B));
        }
        UNiagaraDataInterfaceArrayFunctionLibrary::SetNiagaraArrayVector(NiagaraComp, FName(TEXT("Colors")), ColorVectors);
    }
    
    NiagaraComp->SetVariableFloat(TEXT("PointSize"), PointSize);

    // Calculate bounds for the point cloud (for logging/debugging)
    FBox BoundingBox(ForceInit);
    for (const FVector& Position : Positions)
    {
        BoundingBox += Position;
    }
    
    if (BoundingBox.IsValid)
    {
        UE_LOG(LogPLYPointCloudNiagaraActor, Log, TEXT("Point cloud bounds: %s"), *BoundingBox.ToString());
    }

    UE_LOG(LogPLYPointCloudNiagaraActor, Log, TEXT("Successfully initialized Niagara point cloud with %d points"), Positions.Num());
}