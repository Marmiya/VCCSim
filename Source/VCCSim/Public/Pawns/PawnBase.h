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
#include "PawnBase.generated.h"

class ARecorder;

enum class EPawnType : uint8
{
	Drone,
	Car,
	Flash
};

UCLASS()
class VCCSIM_API APawnBase : public APawn
{
	GENERATED_BODY()
	
public:
	UFUNCTION(BlueprintCallable)
	void SetRecorder(ARecorder* InRecorder);
	UFUNCTION(BlueprintCallable)
	void SetRecordInterval(const float& Interval);
	UFUNCTION(BlueprintCallable)
	void SetRecordState(bool RState){ RecordState = RState; }
	UFUNCTION(BlueprintCallable)
	void SetManualControl(bool Manual){ ManualControl = Manual; }

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category = "VCCSim|Debug")
	bool bRecorded = false;
	bool RecordState = false;
	
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="VCCSim|Debug")
	bool ManualControl = false;

	UPROPERTY()
	ARecorder* Recorder;
	float RecordInterval = -1.f;
	float TimeSinceLastCapture = 0.f;
};