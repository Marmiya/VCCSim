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
#include "Sensors/SensorBase.h"
#include "Sensors/CameraSensor.h"
#include "Sensors/LidarSensor.h"
#include "Sensors/DepthCamera.h"
#include "Sensors/NormalCamera.h"
#include "Sensors/SegmentCamera.h"

class ISensorDataProvider;

DECLARE_LOG_CATEGORY_EXTERN(LogSensorRegistry, Log, All);

struct VCCSIM_API FSensorRegistryEntry
{
    TWeakObjectPtr<UObject> ProviderComponent;
    ESensorType SensorType;
    int32 SensorIndex;
    TWeakObjectPtr<AActor> OwnerActor;

    FSensorRegistryEntry() = default;

    FSensorRegistryEntry(UObject* InProvider, ESensorType InType,
                        int32 InIndex, AActor* InOwner)
        : ProviderComponent(InProvider)
        , SensorType(InType)
        , SensorIndex(InIndex)
        , OwnerActor(InOwner)
    {}

    bool IsValid() const
    {
        return ProviderComponent.IsValid() && OwnerActor.IsValid();
    }

    ISensorDataProvider* GetProvider() const
    {
        if (UObject* Component = ProviderComponent.Get())
        {
            // Try casting to each sensor type that implements ISensorDataProvider
            if (URGBCameraComponent* RGBCam = Cast<URGBCameraComponent>(Component))
            {
                return static_cast<ISensorDataProvider*>(RGBCam);
            }
            if (ULidarComponent* LidarCam = Cast<ULidarComponent>(Component))
            {
                return static_cast<ISensorDataProvider*>(LidarCam);
            }
            if (UDepthCameraComponent* DepthCam = Cast<UDepthCameraComponent>(Component))
            {
                return static_cast<ISensorDataProvider*>(DepthCam);
            }
            if (UNormalCameraComponent* NormalCam = Cast<UNormalCameraComponent>(Component))
            {
                return static_cast<ISensorDataProvider*>(NormalCam);
            }
            if (USegmentationCameraComponent* SegCam = Cast<USegmentationCameraComponent>(Component))
            {
                return static_cast<ISensorDataProvider*>(SegCam);
            }
        }
        return nullptr;
    }
};

class VCCSIM_API FSensorRegistry
{
public:
    FSensorRegistry() = default;
    ~FSensorRegistry() = default;

    void RegisterSensor(UObject* ProviderComponent);
    void UnregisterSensor(UObject* ProviderComponent);
    void UnregisterSensorsForActor(AActor* Actor);

    TArray<FSensorRegistryEntry> GetSensorsForActor(AActor* Actor) const;
    TArray<FSensorRegistryEntry> GetSensorsByType(ESensorType Type) const;
    TArray<FSensorRegistryEntry> GetAllSensors() const;

    void CleanupInvalidEntries();
    int32 GetSensorCount() const { return SensorEntries.Num(); }

private:
    mutable FCriticalSection RegistryLock;
    TArray<FSensorRegistryEntry> SensorEntries;

    void RemoveInvalidEntries_Internal();
};