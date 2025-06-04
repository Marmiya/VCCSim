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
#include "DronePawn.h"
#include "GameFramework/Pawn.h"
#include "InputActionValue.h"
#include "PreciseDrone.generated.h"

class ARecorder;

UCLASS()
class VCCSIM_API APreciseDrone : public ADronePawn
{
	GENERATED_BODY()

public:
    APreciseDrone();
    
    virtual void Tick(float DeltaTime) override;
    virtual void SetupPlayerInputComponent(
        class UInputComponent* PlayerInputComponent) override;

    UFUNCTION(BlueprintCallable)
    void FollowThePathAndSteer();
    
    UFUNCTION(BlueprintCallable, Category = "VCCSim|Sequencer")
    void UpdatePoseFromDistance();
    
    virtual bool IfCloseToTarget(FVector Location, FRotator Rotation) const override;
    
    // Movement parameters
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Movement")
    float MaxVerticalSpeed = 500.0f;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Movement")
    float MaxHorizontalSpeed = 800.0f;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Movement")
    float VerticalAcceleration = 100.0f;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Movement")
    float HorizontalAcceleration = 100.0f;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Movement")
    float YawAcceleration = 180.0f;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Movement")
    float MaxPitch = 20.0f;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Movement")
    float MaxRoll = 20.0f;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Movement")
    float PitchRollAcceleration = 5.0f;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VCCSim|Debug")
    float CurrentVerticalSpeed = 0.0f;
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VCCSim|Debug")
    float CurrentHorizontalSpeed = 0.0f;
    
protected:
    virtual void BeginPlay() override;
    virtual void OnConstruction(const FTransform& Transform) override;

    UPROPERTY(VisibleAnywhere)
    class UStaticMeshComponent* RotorMesh1;
    UPROPERTY(VisibleAnywhere)
    class UStaticMeshComponent* RotorMesh2;
    UPROPERTY(VisibleAnywhere)
    class UStaticMeshComponent* RotorMesh3;
    UPROPERTY(VisibleAnywhere)
    class UStaticMeshComponent* RotorMesh4;

    UPROPERTY(EditAnywhere, Category = "VCCSim|Physics")
    float RotorRotationSpeed = 3000.0f;

    void HandleThrottleInput(const FInputActionValue& Value);
    void HandleThrottleReleased(const FInputActionValue& Value);
    void HandleMovementInput(const FInputActionValue& Value);
    void HandleMovementReleased(const FInputActionValue& Value);
    void HandleYawInput(const FInputActionValue& Value);
    void HandleYawReleased(const FInputActionValue& Value);
    void HandleLookInput(const FInputActionValue& Value);

private:
    float VerticalInput = 0.0f;
    float YawInput = 0.0f;
    FVector2D MovementInput = FVector2D::ZeroVector;
    
    void MoveToTarget(float DeltaTime);
    FVector CalculateTargetPosition(float DeltaTime);
    FRotator CalculateTargetRotation(float DeltaTime);
    
    void HandleManualMovement(float DeltaTime);
    FVector CalculateNewPosition(float DeltaTime);
    FRotator CalculateNewRotation(float DeltaTime);
    FRotator CalculateMovementBasedRotation(
    float DeltaTime, const FVector& MovementDirection);
    
    UFUNCTION(BlueprintCallable)
    void RotateRotors(float DeltaTime);
};