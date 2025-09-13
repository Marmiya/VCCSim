#pragma once

#include "CoreMinimal.h"
#include "Components/SceneComponent.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "PointCloudRenderComponent.generated.h"

UCLASS(ClassGroup=(VCCSim), meta=(BlueprintSpawnableComponent))
class VCCSIM_API UPointCloudRenderComponent : public USceneComponent
{
	GENERATED_BODY()

public:
	UPointCloudRenderComponent();

	// Set visible points
	UFUNCTION(BlueprintCallable, Category = "Point Cloud Rendering")
	void SetVisiblePoints(const TArray<FVector>& Points);
    
	// Set invisible points 
	UFUNCTION(BlueprintCallable, Category = "Point Cloud Rendering") 
	void SetInvisiblePoints(const TArray<FVector>& Points);
    
	// Clear all points
	UFUNCTION(BlueprintCallable, Category = "Point Cloud Rendering")
	void ClearPoints();

	// Point size for visible points
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Rendering", meta = (ClampMin = "0.1", ClampMax = "10.0"))
	float VisiblePointSize = 5.0f;
    
	// Point size for invisible points
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Rendering", meta = (ClampMin = "0.1", ClampMax = "10.0"))
	float InvisiblePointSize = 3.0f;

protected:
	virtual void BeginPlay() override;

private:
	// Instanced mesh component for rendering points
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components", meta = (AllowPrivateAccess = "true"))
	TObjectPtr<UInstancedStaticMeshComponent> VisiblePointsComponent;
	
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components", meta = (AllowPrivateAccess = "true"))
	TObjectPtr<UInstancedStaticMeshComponent> InvisiblePointsComponent;
	
	void SetupComponents();
};