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

#include "Simulation/SensorRegistry.h"
#include "Sensors/ISensorDataProvider.h"
#include "Sensors/CameraSensor.h"
#include "Sensors/LidarSensor.h"
#include "Sensors/DepthCamera.h"
#include "Sensors/NormalCamera.h"
#include "Sensors/SegmentCamera.h"

DEFINE_LOG_CATEGORY(LogSensorRegistry);

void FSensorRegistry::RegisterSensor(UObject* ProviderComponent)
{
    if (!ProviderComponent)
    {
        UE_LOG(LogSensorRegistry, Warning, TEXT("Attempted to register null sensor provider"));
        return;
    }

    ISensorDataProvider* Provider = nullptr;
    USensorBaseComponent* SensorBase = nullptr;

    // Try casting to each sensor type that implements ISensorDataProvider
    if (URGBCameraComponent* RGBCam = Cast<URGBCameraComponent>(ProviderComponent))
    {
        Provider = static_cast<ISensorDataProvider*>(RGBCam);
        SensorBase = RGBCam;
    }
    else if (ULidarComponent* LidarCam = Cast<ULidarComponent>(ProviderComponent))
    {
        Provider = static_cast<ISensorDataProvider*>(LidarCam);
        SensorBase = LidarCam;
    }
    else if (UDepthCameraComponent* DepthCam = Cast<UDepthCameraComponent>(ProviderComponent))
    {
        Provider = static_cast<ISensorDataProvider*>(DepthCam);
        SensorBase = DepthCam;
    }
    else if (UNormalCameraComponent* NormalCam = Cast<UNormalCameraComponent>(ProviderComponent))
    {
        Provider = static_cast<ISensorDataProvider*>(NormalCam);
        SensorBase = NormalCam;
    }
    else if (USegmentationCameraComponent* SegCam = Cast<USegmentationCameraComponent>(ProviderComponent))
    {
        Provider = static_cast<ISensorDataProvider*>(SegCam);
        SensorBase = SegCam;
    }

    if (!Provider || !SensorBase)
    {
        UE_LOG(LogSensorRegistry, Warning, TEXT("Object does not implement ISensorDataProvider"));
        return;
    }

    FScopeLock Lock(&RegistryLock);

    AActor* OwnerActor = Provider->GetOwnerActor();
    if (!OwnerActor)
    {
        UE_LOG(LogSensorRegistry, Warning, TEXT("Sensor provider has no owner actor"));
        return;
    }

    int32 SensorIndex = SensorBase->GetSensorIndex();
    FSensorRegistryEntry Entry(ProviderComponent, Provider->GetSensorType(), SensorIndex, OwnerActor);

    SensorEntries.Add(Entry);

    UE_LOG(LogSensorRegistry, Log, TEXT("Registered sensor: Type=%d, Index=%d, Actor=%s"),
           static_cast<int32>(Entry.SensorType), Entry.SensorIndex,
           OwnerActor ? *OwnerActor->GetName() : TEXT("None"));
}

void FSensorRegistry::UnregisterSensor(UObject* ProviderComponent)
{
    if (!ProviderComponent)
    {
        return;
    }

    FScopeLock Lock(&RegistryLock);

    for (int32 Index = SensorEntries.Num() - 1; Index >= 0; --Index)
    {
        if (SensorEntries[Index].ProviderComponent.Get() == ProviderComponent)
        {
            UE_LOG(LogSensorRegistry, Log, TEXT("Unregistered sensor: Type=%d, Index=%d"),
                   static_cast<int32>(SensorEntries[Index].SensorType),
                   SensorEntries[Index].SensorIndex);
            SensorEntries.RemoveAt(Index);
            break;
        }
    }
}

void FSensorRegistry::UnregisterSensorsForActor(AActor* Actor)
{
    if (!Actor)
    {
        return;
    }

    FScopeLock Lock(&RegistryLock);

    int32 RemovedCount = 0;
    for (int32 Index = SensorEntries.Num() - 1; Index >= 0; --Index)
    {
        if (SensorEntries[Index].OwnerActor.Get() == Actor)
        {
            SensorEntries.RemoveAt(Index);
            ++RemovedCount;
        }
    }

    if (RemovedCount > 0)
    {
        UE_LOG(LogSensorRegistry, Log, TEXT("Unregistered %d sensors for actor: %s"),
               RemovedCount, *Actor->GetName());
    }
}

TArray<FSensorRegistryEntry> FSensorRegistry::GetSensorsForActor(AActor* Actor) const
{
    TArray<FSensorRegistryEntry> Result;
    if (!Actor)
    {
        return Result;
    }

    FScopeLock Lock(&RegistryLock);

    for (const FSensorRegistryEntry& Entry : SensorEntries)
    {
        if (Entry.IsValid() && Entry.OwnerActor.Get() == Actor)
        {
            Result.Add(Entry);
        }
    }

    return Result;
}

TArray<FSensorRegistryEntry> FSensorRegistry::GetSensorsByType(ESensorType Type) const
{
    TArray<FSensorRegistryEntry> Result;

    FScopeLock Lock(&RegistryLock);

    for (const FSensorRegistryEntry& Entry : SensorEntries)
    {
        if (Entry.IsValid() && Entry.SensorType == Type)
        {
            Result.Add(Entry);
        }
    }

    return Result;
}

TArray<FSensorRegistryEntry> FSensorRegistry::GetAllSensors() const
{
    TArray<FSensorRegistryEntry> Result;

    FScopeLock Lock(&RegistryLock);

    for (const FSensorRegistryEntry& Entry : SensorEntries)
    {
        if (Entry.IsValid())
        {
            Result.Add(Entry);
        }
    }

    return Result;
}

void FSensorRegistry::CleanupInvalidEntries()
{
    FScopeLock Lock(&RegistryLock);
    RemoveInvalidEntries_Internal();
}

void FSensorRegistry::RemoveInvalidEntries_Internal()
{
    int32 RemovedCount = 0;
    for (int32 Index = SensorEntries.Num() - 1; Index >= 0; --Index)
    {
        if (!SensorEntries[Index].IsValid())
        {
            SensorEntries.RemoveAt(Index);
            ++RemovedCount;
        }
    }

    if (RemovedCount > 0)
    {
        UE_LOG(LogSensorRegistry, Warning, TEXT("Cleaned up %d invalid sensor entries"), RemovedCount);
    }
}