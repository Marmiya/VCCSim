#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "NiagaraComponent.h"
#include "NiagaraSystem.h"
#include "PLYPointCloudNiagaraActor.generated.h"

UCLASS()
class VCCSIM_API APLYPointCloudNiagaraActor : public AActor
{
    GENERATED_BODY()
public:
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Point Cloud")
    TObjectPtr<UNiagaraSystem> NiagaraSystemAsset;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Point Cloud")
    FString PlyFilePath;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Point Cloud", meta = (ClampMin = "0.1", ClampMax = "1000.0"))
    float UnitScaleToUE = 100.0f; // meters -> centimeters
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Point Cloud", meta = (ClampMin = "0.1", ClampMax = "10.0"))
    float PointSize = 1.5f;

    APLYPointCloudNiagaraActor();

protected:
    virtual void BeginPlay() override;

private:
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components", meta = (AllowPrivateAccess = "true"))
    TObjectPtr<UNiagaraComponent> NiagaraComp;
    
    bool LoadPly(TArray<FVector>& OutPositions, TArray<FLinearColor>& OutColors);
};