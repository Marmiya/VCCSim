#pragma once

#include "CoreMinimal.h"
#include "Components/SceneComponent.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "DataStructures/PointCloud.h"
#include "PointCloudRenderer.generated.h"

UCLASS(ClassGroup=(VCCSim), meta=(BlueprintSpawnableComponent))
class VCCSIM_API UPointCloudRenderer : public USceneComponent
{
    GENERATED_BODY()

public:
    UPointCloudRenderer();

protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    virtual void OnRegister() override;

public:
    UFUNCTION(BlueprintCallable, Category = "Point Cloud Rendering")
    void RenderPointCloud(const FPointCloudData& PointCloudData, bool bShowColors = true, float PointSize = 1.0f);

    UFUNCTION(BlueprintCallable, Category = "Point Cloud Rendering")
    void ClearPointCloud();

    UFUNCTION(BlueprintCallable, Category = "Point Cloud Rendering")
    void SetPointSize(float NewPointSize);

    UFUNCTION(BlueprintCallable, Category = "Point Cloud Rendering")
    void SetShowColors(bool bInShowColors);

    UFUNCTION(BlueprintCallable, BlueprintPure, Category = "Point Cloud Rendering")
    int32 GetRenderedPointCount() const { return RenderedPointCount; }

    UFUNCTION(BlueprintCallable, BlueprintPure, Category = "Point Cloud Rendering")
    bool HasColors() const { return bHasColors; }

protected:
    void SetupPointCloudMaterial();
    void CreateInstancedMeshSystem();
    void RenderPointCloudInstanced(const FPointCloudData& PointCloudData);
    void RenderPointCloudInstancedWithColors(const FPointCloudData& PointCloudData);
    void ClearColorInstancedComponents();
    UInstancedStaticMeshComponent* GetOrCreateColorISM(const FLinearColor& Color);

protected:
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
    TObjectPtr<UInstancedStaticMeshComponent> InstancedMeshComponent;

    UPROPERTY(Transient)
    TMap<uint32, TObjectPtr<UInstancedStaticMeshComponent>> ColorInstancedComponents;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Rendering")
    float PointSize = 1.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Rendering")
    bool bShowColors = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Rendering")
    FLinearColor DefaultColor = FLinearColor(1.0f, 0.5f, 0.0f, 1.0f);

    UPROPERTY()
    int32 RenderedPointCount = 0;

    UPROPERTY()
    bool bHasColors = false;

    UPROPERTY()
    bool bIsInitialized = false;

    TArray<FVector> CachedPositions;
    TArray<FLinearColor> CachedColors;

private:
    static uint32 PackColorKey(const FLinearColor& Color)
    {
        const uint8 R = (uint8)FMath::Clamp(FMath::RoundToInt(Color.R * 255.0f), 0, 255);
        const uint8 G = (uint8)FMath::Clamp(FMath::RoundToInt(Color.G * 255.0f), 0, 255);
        const uint8 B = (uint8)FMath::Clamp(FMath::RoundToInt(Color.B * 255.0f), 0, 255);
        return ((uint32)R << 16) | ((uint32)G << 8) | (uint32)B;
    }
};
