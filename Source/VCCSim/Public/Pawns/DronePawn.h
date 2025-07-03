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
#include "Pawns/PawnBase.h"
#include "InputActionValue.h"
#include "SimPath.h"
#include "DronePawn.generated.h"

class ARecorder;

UCLASS()
class VCCSIM_API ADronePawn : public APawnBase
{
	GENERATED_BODY()

public:

    UFUNCTION()
	virtual void OnConstruction(const FTransform& Transform) override;
	virtual void Tick(float DeltaSeconds) override;

	UFUNCTION(BlueprintCallable)
    void AddMapContext();

	UFUNCTION(BlueprintCallable)
	void CalculateDistance();

	void InitPose();
	
    UFUNCTION(BlueprintCallable, Category = "API")
    void SetTarget(FVector Location, FRotator Rotation);
    virtual bool IfCloseToTarget(FVector Location, FRotator Rotation) const;
	bool SetPath(TArray<FVector> Positions, TArray<FRotator> Rotations);

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Target")
    FVector TargetLocation;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Target")
    FRotator TargetRotation;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Target")
    bool bUseTarget = false;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Target")
    float PositionThreshold = 0.5f;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Target")
    float RotationThreshold = 0.5f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Target")
	bool bUsePath = false;
	
    // Enhanced Input Components
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "VCCSim|Input")
    class UInputMappingContext* DroneInputMappingContext;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "VCCSim|Input")
    class UInputAction* ThrottleAction;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "VCCSim|Input")
    class UInputAction* MovementAction;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "VCCSim|Input")
    class UInputAction* YawAction;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "VCCSim|Input")
    class UInputAction* LookAction;
    
    // Components
    UPROPERTY(VisibleAnywhere)
    class USceneComponent* DroneRoot;
    
    UPROPERTY(VisibleAnywhere)
    class UStaticMeshComponent* DroneMesh;

    UPROPERTY(VisibleAnywhere)
    class UEnhancedInputComponent* EnhancedInputComponent;

	UPROPERTY(BlueprintReadWrite, EditInstanceOnly, Category="VCCSim|Driving", Interp)
	TObjectPtr<AVCCSimPath> Path;
	
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="VCCSim|Driving", Interp, meta=(UIMin="0", ClampMin="0"))
	float DistanceTraveled;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="VCCSim|Debug")
	float CourseDistance;
	float LastCourseDistance;
	float Laps;
};
