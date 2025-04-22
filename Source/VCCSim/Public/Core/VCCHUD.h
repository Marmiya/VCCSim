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
#include "GameFramework/HUD.h"
#include "EnhancedInputSubsystems.h"
#include "Utils/ConfigParser.h"
#include "VCCHUD.generated.h"

class UVCCSIMDisplayWidget;
class UPauseMenuWidget;
class UInputAction;
class UInputMappingContext;
class ARecorder;
class UFMeshManager;
class ASceneAnalysisManager;
struct FRobotGrpcMaps;

UCLASS()
class VCCSIM_API AVCCHUD : public AHUD
{
	GENERATED_BODY()

public:
	void SetupRecorder(FVCCSimConfig& Config);
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    
	// Original properties
	UPROPERTY(EditDefaultsOnly, Category = "UI")
	TSubclassOf<UUserWidget> WidgetClass;
    
	UPROPERTY(EditDefaultsOnly, Category = "UI")
	TSubclassOf<UPauseMenuWidget> PauseWidgetClass;

	UPROPERTY()
	TObjectPtr<APawn> MainCharacter;
    
	UPROPERTY(EditAnywhere, Category = "IA")
	TObjectPtr<UInputAction> PauseAction;
	UPROPERTY(EditAnywhere, Category = "IA")
	TObjectPtr<UInputAction> ToggleRecordingAction;

	UPROPERTY(EditAnywhere, Category = "IA")
	TObjectPtr<UInputMappingContext> DefaultMappingContext;

protected:
	UFUNCTION()
	void OnPauseActionTriggered();
	UFUNCTION()
	void OnToggleRecordingTriggered();
	void SetupEnhancedInput();
	void SetupWidgetsAndLS(const FVCCSimConfig& Config);
	void SetupMainCharacter(const FVCCSimConfig& Config, TArray<AActor*> FoundPawns);
	FRobotGrpcMaps SetupActors(const FVCCSimConfig& Config);
	APawn* CreatePawn(const FVCCSimConfig& Config, const FRobot& Robot);

private:
	UPROPERTY()
	TObjectPtr<UVCCSIMDisplayWidget> WidgetInstance;
	UPROPERTY()
	TObjectPtr<UPauseMenuWidget> CurrentPauseMenu;
	UPROPERTY()
	AActor* Holder = nullptr;
	UPROPERTY()
	ARecorder* Recorder;
	UPROPERTY()
	UFMeshManager* MeshManager = nullptr;
	UPROPERTY()
	ASceneAnalysisManager* SceneAnalysisManager = nullptr;

	AActor* FindPawnInTagAndName(const std::string& Target, TArray<AActor*> FoundPawns);
};
