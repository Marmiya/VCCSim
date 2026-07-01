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

DEFINE_LOG_CATEGORY_STATIC(LogLightingManager, Log, All);

FLightingManager::FLightingManager()
{
}

FLightingManager::~FLightingManager()
{
}

void FLightingManager::ApplyLightingCondition(float ElevationDeg, float AzimuthDeg, bool bMarkDirty)
{
    UWorld* CurrentWorld = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
    if (!CurrentWorld)
    {
        UE_LOG(LogLightingManager, Error, TEXT("World is not available to apply lighting condition."));
        return;
    }

    ADirectionalLight* DirectionalLight = nullptr;
    for (TActorIterator<ADirectionalLight> It(CurrentWorld); It; ++It)
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
