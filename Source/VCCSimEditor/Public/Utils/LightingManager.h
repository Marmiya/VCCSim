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
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program. If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "CoreMinimal.h"
#include "Engine/TimerHandle.h"
#include "Utils/VCCSimSunPositionHelper.h"

class UWorld;

/**
 * Manages scene lighting, including applying specific sun positions and simulating a day/night cycle.
 */
class VCCSIMEDITOR_API FLightingManager
{
public:
    FLightingManager(UWorld* InWorld);
    ~FLightingManager();

    /**
     * Applies a specific lighting condition to the scene's main directional light.
     * @param ElevationDeg Sun elevation in degrees.
     * @param AzimuthDeg Sun azimuth in degrees.
     * @param bMarkDirty Whether to mark the light actor as modified.
     */
    void ApplyLightingCondition(float ElevationDeg, float AzimuthDeg, bool bMarkDirty = true);

    /**
     * Calculates the sun position based on the provided parameters and applies it.
     * @param Params The geographic and time parameters for the sun position calculation.
     * @return The calculated sun elevation and azimuth.
     */
    TPair<float, float> CalculateAndApplySunPosition(const FVCCSimSunPositionHelper::FSunParams& Params);

    /**
     * Toggles the automatic day/night cycle simulation.
     * @param bIsActive Whether to activate or deactivate the cycle.
     * @param InCycleParams The parameters to use for the simulation.
     * @param InCycleSpeed The speed of the simulation (24h in N real-world seconds).
     */
    void ToggleDayCycle(bool bIsActive, const FVCCSimSunPositionHelper::FSunParams& InCycleParams, float InCycleSpeed);

private:
    /** Callback for the day cycle timer. */
    void TickDayCycle();

    /** The world context in which this manager operates. */
    TWeakObjectPtr<UWorld> World;
    
    /** Timer handle for the day cycle simulation. */
    FTimerHandle DayCycleTimerHandle;

    /** Current parameters for the active day cycle. */
    FVCCSimSunPositionHelper::FSunParams DayCycleParams;
    
    /** Speed of the day cycle simulation. */
    float DayCycleSpeed = 10.f;
    
    /** The current simulated minute of the day (0-1440). */
    float DayCycleSimMinute = 0.f;
};
