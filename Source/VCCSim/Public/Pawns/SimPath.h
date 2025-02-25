﻿#pragma once
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"

#include "SimPath.generated.h"

class USplineComponent;
struct FSplinePoint;

UCLASS(HideCategories=("Default", "Replication", "REndering", "Collision",
	"HLOD", "Input", "Physics", "Networking", "Actor", "Cooking", "Hidden",
	"World Partition", "Tick", "Events", "Data Layers"))
class VCCSIM_API AVCCSimPath : public AActor
{
	GENERATED_BODY()
public:	
	AVCCSimPath();
	virtual void OnConstruction(const FTransform& Transform) override; 
	
	UFUNCTION(BlueprintCallable, Category="Default")
	void DiscoverTraceIgnores();

	UFUNCTION(BlueprintCallable)
	void SnapAllPointsToGround();

	UFUNCTION(BlueprintCallable)
	void ReverseSpline();
	
	UFUNCTION(BlueprintCallable)
	void FlattenAllTangents();
	
	UFUNCTION(BlueprintCallable)
	void MovePivotToFirstPoint();

	UFUNCTION(BlueprintCallable)
	void SetNewTrajectory(const TArray<FVector>& Positions,
		const TArray<FRotator>& Rotations);

	int32 GetNumberOfSplinePoints() const;
	FVector GetLocationAtSplinePoint(int32 Index) const;
	FRotator GetRotationAtSplinePoint(int32 Index) const;
	
public:
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Default")
	TObjectPtr<USplineComponent> Spline;
	
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Path")
	bool SnapToGround;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Path")
	bool bReverseSpline;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Path")
	bool ClosedSpline;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Path")
	bool FlattenTangents;
	
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Hidden")
	TArray<FSplinePoint> SplinePoints;
	UPROPERTY(BlueprintReadWrite, EditInstanceOnly, Category="Hidden")
	TArray<AActor*> ExcludedInTrace;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Hidden")
	TMap<int32,FVector> NewSplinePointLocations;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Read Only")
	float PathLength;
};
