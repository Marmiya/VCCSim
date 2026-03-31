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

#include "Utils/LightingManager.h"
#include "Engine/World.h"
#include "Engine/DirectionalLight.h"
#include "Components/DirectionalLightComponent.h"
#include "EngineUtils.h"
#include "Editor.h"
#include "TimerManager.h"

DEFINE_LOG_CATEGORY_STATIC(LogLightingManager, Log, All);

FLightingManager::FLightingManager(UWorld* InWorld)
    : World(InWorld)
{
}

FLightingManager::~FLightingManager()
{
    // Ensure the timer is cleared if it's still active
    if (DayCycleTimerHandle.IsValid() && GEditor)
    {
        GEditor->GetTimerManager()->ClearTimer(DayCycleTimerHandle);
    }
}

void FLightingManager::ApplyLightingCondition(float ElevationDeg, float AzimuthDeg, bool bMarkDirty)
{
    if (!World.IsValid())
    {
        UE_LOG(LogLightingManager, Error, TEXT("World is not available to apply lighting condition."));
        return;
    }

    ADirectionalLight* DirectionalLight = nullptr;
    for (TActorIterator<ADirectionalLight> It(World.Get()); It; ++It)
    {
        ADirectionalLight* Candidate = *It;
        if (!Candidate) continue;

        UDirectionalLightComponent* LightComp = Candidate->GetComponent();
        if (LightComp && LightComp->bAtmosphereSunLight)
        {
            DirectionalLight = Candidate;
            break;
        }

        if (!DirectionalLight)
        {
            DirectionalLight = Candidate;
        }
    }

    if (!DirectionalLight)
    {
        UE_LOG(LogLightingManager, Warning, TEXT("No Directional Light found in scene."));
        return;
    }

    if (bMarkDirty)
    {
        DirectionalLight->Modify();
    }
    
    FRotator NewRotation(-ElevationDeg, AzimuthDeg - 180.f, 0.f);
    DirectionalLight->SetActorRotation(NewRotation);

    if (GEditor)
    {
        GEditor->RedrawAllViewports();
    }
    
    UE_LOG(LogLightingManager, Log, TEXT("Lighting applied: Elevation=%.1f Az=%.1f"), ElevationDeg, AzimuthDeg);
}

TPair<float, float> FLightingManager::CalculateAndApplySunPosition(const FVCCSimSunPositionHelper::FSunParams& Params)
{
    float Elevation = 0.f, Azimuth = 0.f;
    bool bAboveHorizon = FVCCSimSunPositionHelper::Calculate(Params, Elevation, Azimuth);

    ApplyLightingCondition(Elevation, Azimuth);

    if (!bAboveHorizon)
    {
        UE_LOG(LogLightingManager, Log, TEXT("Night time: Sun is %.1f degrees below the horizon."), -Elevation);
    }
    
    return TPair<float, float>(Elevation, Azimuth);
}

void FLightingManager::ToggleDayCycle(bool bIsActive, const FVCCSimSunPositionHelper::FSunParams& InCycleParams, float InCycleSpeed)
{
    if (!GEditor) return;

    if (bIsActive)
    {
        DayCycleParams = InCycleParams;
        DayCycleSpeed = InCycleSpeed;
        DayCycleSimMinute = (DayCycleParams.Hour * 60.f) + DayCycleParams.Minute;

        FTimerDelegate TimerDelegate = FTimerDelegate::CreateRaw(this, &FLightingManager::TickDayCycle);
        GEditor->GetTimerManager()->SetTimer(DayCycleTimerHandle, TimerDelegate, 0.1f, true);
    }
    else
    {
        GEditor->GetTimerManager()->ClearTimer(DayCycleTimerHandle);
    }
}

void FLightingManager::TickDayCycle()
{
    const float MinutesPerTick = 1440.f * 0.1f / FMath::Max(DayCycleSpeed, 1.f);
    DayCycleSimMinute += MinutesPerTick;
    if (DayCycleSimMinute >= 1440.f)
    {
        DayCycleSimMinute -= 1440.f;
    }

    FVCCSimSunPositionHelper::FSunParams CurrentParams = DayCycleParams;
    CurrentParams.Hour = FMath::FloorToInt(DayCycleSimMinute / 60.f) % 24;
    CurrentParams.Minute = FMath::FloorToInt(DayCycleSimMinute) % 60;

    float Elevation = 0.f, Azimuth = 0.f;
    FVCCSimSunPositionHelper::Calculate(CurrentParams, Elevation, Azimuth);

    ApplyLightingCondition(Elevation, Azimuth, false);
}
