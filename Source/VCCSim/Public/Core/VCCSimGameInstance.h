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
#include "Engine/GameInstance.h"
#include "VCCSimGameInstance.generated.h"

UCLASS()
class VCCSIM_API UVCCSimGameInstance : public UGameInstance
{
	GENERATED_BODY()

public:
	UVCCSimGameInstance();

	virtual void Init() override;

	// Functions for map management
	UFUNCTION(BlueprintCallable, Category = "VCCSim|Maps")
	void LoadMap(const FString& MapName);

	UFUNCTION(BlueprintCallable, Category = "VCCSim|Maps")
	virtual void ReturnToMainMenu() override;

	// Functions for game state
	UFUNCTION(BlueprintCallable, Category = "VCCSim|GameState")
	void SaveGameState();

	UFUNCTION(BlueprintCallable, Category = "VCCSim|GameState")
	void LoadGameState();

	// Map names
	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category = "VCCSim|Maps")
	FString MainMenuMapName;

	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category = "VCCSim|Maps")
	TArray<FString> AvailableMaps;
	
	UPROPERTY()
	FString CurrentMapName;

	// Settings getters/setters
	UFUNCTION(BlueprintCallable, Category = "VCCSim|Settings")
	void SetSimulationSpeed(float Speed);

	UFUNCTION(BlueprintPure, Category = "VCCSim|Settings")
	float GetSimulationSpeed() const { return SimulationSpeed; }

	UFUNCTION(BlueprintCallable)
	FString GetCurrentMapName() const;

	UFUNCTION(BlueprintCallable)
	void ReloadCurrentMap();

	UFUNCTION()
	void DetectCurrentMapName();

private:
	UPROPERTY()
	float SimulationSpeed;
};