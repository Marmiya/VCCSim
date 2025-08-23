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
#include "GameFramework/Pawn.h"
#include "PawnBase.h"
#include "FlashPawn.generated.h"

class AVCCSimPath;

UCLASS()
class VCCSIM_API AFlashPawn final: public APawnBase
{
	GENERATED_BODY()
	
public:
	AFlashPawn();
	
	UFUNCTION(BlueprintCallable, Category = "API")
	void SetTarget(FVector Location, FRotator Rotation);
	UFUNCTION(BlueprintCallable, Category = "API")
	void SetPath(const TArray<FVector>& Positions, const TArray<FRotator>& Rotations);
	UFUNCTION(BlueprintCallable, Category = "API")
	bool IsReady() const { return bAcqReady; }
	UFUNCTION(BlueprintCallable, Category = "API")
	void MoveToNext() { bMoveReady = true; bAcqReady = false; }

	UFUNCTION(BlueprintCallable, Category = "Panel")
	void SetPathPanel(const TArray<FVector>& Positions, const TArray<FRotator>& Rotations);
	UFUNCTION(BlueprintCallable, Category = "Panel")
	void MoveForward();
	UFUNCTION(BlueprintCallable, Category = "Panel")
	void MoveBackward();
	UFUNCTION(BlueprintCallable, Category = "Panel")
	void MoveTo(const int32& Index);
	UFUNCTION(BlueprintCallable, Category = "Panel")
	int32 GetCurrentIndex() const { return CurrentIndex; }
	UFUNCTION(BlueprintCallable, Category = "Panel")
	void GetCurrentPath(TArray<FVector>& Positions, TArray<FRotator>& Rotations) const;
	UFUNCTION(BlueprintCallable, Category = "Panel")
	int32 GetPoseCount() const { return PendingPositions.Num(); }
	UPROPERTY()
	TArray<FVector> PendingPositions;
	UPROPERTY()
	TArray<FRotator> PendingRotations;
	
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VCCSim|Target")
	bool bMoveReady = true;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VCCSim|Target")
	bool bAcqReady = false;
	
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Target")
	AVCCSimPath* Path;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Target")
	bool bUsePath = false;
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "VCCSim|Target")
	int32 CurrentPathIndex = 0;
	
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Target")
	FVector TargetLocation;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Target")
	FRotator TargetRotation;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "VCCSim|Target")
	bool bUseTarget = false;
	
protected:
	virtual void BeginPlay() override;
	virtual void Tick(float DeltaSeconds) override;

private:
	UPROPERTY()
	class USceneComponent* FlashRoot;
	UPROPERTY()
	int32 CurrentIndex = 0;
};
