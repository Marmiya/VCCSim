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
#include "GameFramework/Actor.h"

#include "SimLookAtPath.generated.h"

class USplineComponent;
class UStaticMeshComponent;

UENUM(BlueprintType)
enum class ELookAtSamplingMode : uint8
{
	ControlPoints	UMETA(DisplayName = "Control Points"),
	EqualDivisions	UMETA(DisplayName = "Equal Divisions")
};

UENUM(BlueprintType)
enum class EOrientationMode : uint8
{
	LookAtTarget    UMETA(DisplayName = "Look At Target"),
	FreeOrientation UMETA(DisplayName = "Free Orientation")
};

UCLASS(HideCategories=("Default", "Replication", "Rendering", "Collision",
	"HLOD", "Input", "Physics", "Networking", "Actor", "Cooking", "Hidden",
	"World Partition", "Tick", "Events", "Data Layers"))
class VCCSIM_API AVCCSimLookAtPath : public AActor
{
	GENERATED_BODY()
public:
	AVCCSimLookAtPath();
	virtual void OnConstruction(const FTransform& Transform) override;

	UFUNCTION(BlueprintCallable, Category="Default")
	void DiscoverTraceIgnores();

	UFUNCTION(BlueprintCallable, meta=(CallInEditor=true), Category="Path")
	void FlattenAllPointsToSameHeight();

	UFUNCTION(BlueprintCallable, meta=(CallInEditor=true), Category="Path")
	void SnapAllPointsToGround();

	void GetSamplePoses(TArray<FVector>& OutPositions, TArray<FRotator>& OutRotations) const;
	int32 GetNumSamplePoints() const;

public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Default")
	TObjectPtr<USplineComponent> Spline;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Target")
	TObjectPtr<UStaticMeshComponent> TargetPoint;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Mode")
	ELookAtSamplingMode SamplingMode;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Mode",
		meta=(EditCondition="SamplingMode==ELookAtSamplingMode::EqualDivisions",
			  EditConditionHides, ClampMin="2", ClampMax="2000"))
	int32 NumDivisions;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Mode")
	EOrientationMode OrientationMode;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Mode",
		meta=(EditCondition="OrientationMode==EOrientationMode::FreeOrientation",
			  EditConditionHides))
	TArray<FRotator> FreeOrientations;

	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category="Read Only")
	float PathLength;

	UPROPERTY(BlueprintReadWrite, VisibleAnywhere, Category="Read Only")
	int32 NumSamplePoints;

	UPROPERTY(BlueprintReadWrite, EditInstanceOnly, Category="Hidden")
	TArray<AActor*> ExcludedInTrace;
};
