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

#include "Core/VCCSimGameInstance.h"
#include "Kismet/GameplayStatics.h"

UVCCSimGameInstance::UVCCSimGameInstance()
{
	// Default values
	MainMenuMapName = TEXT("MainMenu");
	SimulationSpeed = 1.0f;
}

void UVCCSimGameInstance::Init()
{
	Super::Init();

	DetectCurrentMapName();
	UE_LOG(LogTemp, Log, TEXT("VCCSim GameInstance Initialized"));
}

void UVCCSimGameInstance::LoadMap(const FString& MapName)
{
	CurrentMapName = MapName;
	
	// Check if the map name is valid
	if (AvailableMaps.Contains(MapName))
	{
		UGameplayStatics::OpenLevel(this, *MapName);
	}
	else
	{
		UE_LOG(LogTemp, Warning,
			TEXT("Attempted to load invalid map: %s"), *MapName);
	}
}

void UVCCSimGameInstance::ReturnToMainMenu()
{
	UGameplayStatics::OpenLevel(this, *MainMenuMapName);
}

void UVCCSimGameInstance::SaveGameState()
{
}

void UVCCSimGameInstance::LoadGameState()
{
}

void UVCCSimGameInstance::SetSimulationSpeed(float Speed)
{
	SimulationSpeed = FMath::Clamp(Speed, 0.1f, 10.0f);
}

FString UVCCSimGameInstance::GetCurrentMapName() const
{
	return CurrentMapName;
}

void UVCCSimGameInstance::ReloadCurrentMap()
{
	// If CurrentMapName is empty (like when starting from editor), detect it
	if (CurrentMapName.IsEmpty())
	{
		DetectCurrentMapName();
	}
    
	if (!CurrentMapName.IsEmpty())
	{
		UE_LOG(LogTemp, Warning, TEXT("GameInstance: Reloading current map: %s"), *CurrentMapName);
		LoadMap(CurrentMapName);
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("GameInstance: Could not determine current map to reload"));
	}
}

void UVCCSimGameInstance::DetectCurrentMapName()
{
    UWorld* CurrentWorld = GetWorld();
    if (!CurrentWorld)
    {
        UE_LOG(LogTemp, Error, TEXT("GameInstance: Current world is null"));
        return;
    }
    
    // Try to get the map name from the world
    FString DetectedMapName = CurrentWorld->GetMapName();
    
    UE_LOG(LogTemp, Warning, TEXT("GameInstance: Raw detected map name: %s"), *DetectedMapName);
    
    // Clean up the map name
    if (DetectedMapName.StartsWith(TEXT("UEDPIE_0_")))
    {
        DetectedMapName = DetectedMapName.RightChop(9); // Remove "UEDPIE_0_" prefix
    }
    
    // Remove /Game/ path if present
    if (DetectedMapName.StartsWith(TEXT("/Game/")))
    {
        DetectedMapName = DetectedMapName.RightChop(6); // Remove "/Game/" prefix
    }
    
    // Try alternative method if still empty
    if (DetectedMapName.IsEmpty())
    {
        FString URL = CurrentWorld->URL.Map;
        DetectedMapName = FPaths::GetBaseFilename(URL);
        UE_LOG(LogTemp, Warning, TEXT("GameInstance: Using URL-based map name: %s"), *DetectedMapName);
    }
    
    // Try to match with available maps
    if (!DetectedMapName.IsEmpty())
    {
        // First, try exact matches
        for (const FString& AvailableMap : AvailableMaps)
        {
            if (DetectedMapName.Equals(AvailableMap, ESearchCase::IgnoreCase))
            {
                CurrentMapName = AvailableMap;
                UE_LOG(LogTemp, Warning, TEXT("GameInstance: Exact match - detected map '%s' matches available map '%s'"), *DetectedMapName, *AvailableMap);
                return;
            }
        }
        
        // Then try partial matches
        for (const FString& AvailableMap : AvailableMaps)
        {
            if (DetectedMapName.Contains(AvailableMap) || AvailableMap.Contains(DetectedMapName))
            {
                CurrentMapName = AvailableMap;
                UE_LOG(LogTemp, Warning, TEXT("GameInstance: Partial match - detected map '%s' matched to available map '%s'"), *DetectedMapName, *AvailableMap);
                return;
            }
        }
        
        // If no match found, use the detected name as-is
        CurrentMapName = DetectedMapName;
        UE_LOG(LogTemp, Warning, TEXT("GameInstance: No match found, using detected map name: %s"), *CurrentMapName);
    }
    else
    {
        // Last resort: if we have available maps and detection failed, use the first one
        if (AvailableMaps.Num() > 0)
        {
            CurrentMapName = AvailableMaps[0];
            UE_LOG(LogTemp, Warning, TEXT("GameInstance: Detection failed, defaulting to first available map: %s"), *CurrentMapName);
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("GameInstance: Could not detect current map name and no available maps"));
        }
    }
}