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

#pragma once

#include "CoreMinimal.h"
#include "Components/SceneComponent.h"
#include "Materials/MaterialInstanceDynamic.h"

#include "ActorMarkerComponent.generated.h"

class UProceduralMeshComponent;

UCLASS(ClassGroup = (VCCSIM), meta = (BlueprintSpawnableComponent))
class VCCSIM_API UActorMarkerComponent : public USceneComponent
{
	GENERATED_BODY()

public:
	UActorMarkerComponent();

	virtual void OnRegister() override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType,
		FActorComponentTickFunction* ThisTickFunction) override;

	UFUNCTION(BlueprintCallable, Category = "VCCSim|Marker")
	void SetMarkerColor(const FLinearColor& NewColor);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Marker|Appearance")
	FLinearColor MarkerColor = FLinearColor(0.05f, 0.6f, 1.0f, 1.0f);

	// Optional material. Assign a material exposing a "BaseColor" vector
	// parameter and an "EmissiveStrength" scalar parameter to get the glow /
	// pulse effect. If left empty the marker still renders with vertex colors.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Marker|Appearance")
	UMaterialInterface* BaseMaterial = nullptr;

	// Half-width of the square base (cm).
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Marker|Geometry")
	float BaseHalfSize = 5.0f;

	// Distance from the downward apex to the base (cm).
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Marker|Geometry")
	float Height = 10.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Marker|Animation")
	bool bAnimate = true;

	// Yaw spin speed (degrees / second).
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Marker|Animation")
	float SpinSpeed = 60.0f;

	// Vertical bob amplitude (cm).
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Marker|Animation")
	float HoverAmplitude = 5.0f;

	// Vertical bob frequency (cycles / second).
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Marker|Animation")
	float HoverFrequency = 0.6f;

	// Emissive pulse frequency (cycles / second). 0 disables the pulse.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Marker|Animation")
	float PulseFrequency = 1.2f;

	// Emissive strength oscillates between Min and Max with the pulse.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Marker|Animation")
	float EmissiveMin = 2.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Marker|Animation")
	float EmissiveMax = 8.0f;

private:
	UPROPERTY()
	UProceduralMeshComponent* MeshComponent = nullptr;

	UPROPERTY()
	UMaterialInstanceDynamic* MaterialInstance = nullptr;

	void RebuildGeometry();
	void ApplyMaterial();
	void ApplyColor();

	float AnimTime = 0.0f;
	float CurrentYaw = 0.0f;

	float BuiltBaseHalfSize = -1.0f;
	float BuiltHeight = -1.0f;
};
