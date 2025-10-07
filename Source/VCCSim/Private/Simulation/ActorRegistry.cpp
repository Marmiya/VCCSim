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

#include "Simulation/ActorRegistry.h"

DEFINE_LOG_CATEGORY(LogActorRegistry);

void FActorRegistry::RegisterActor(AActor* InActor, const FString& InActorName, const FString& InPoseFilePath, double InRecordInterval)
{
	if (!InActor)
	{
		UE_LOG(LogActorRegistry, Warning, TEXT("Attempted to register null actor"));
		return;
	}

	FScopeLock Lock(&RegistryLock);

	for (const FActorRegistryEntry& Entry : ActorEntries)
	{
		if (Entry.Actor.Get() == InActor)
		{
			UE_LOG(LogActorRegistry, Warning, TEXT("Actor already registered: %s"), *InActorName);
			return;
		}
	}

	FActorRegistryEntry NewEntry(InActor, InActorName, InPoseFilePath, InRecordInterval);
	ActorEntries.Add(NewEntry);

	UE_LOG(LogActorRegistry, Log, TEXT("Registered actor: %s, Interval: %.4f, PoseFile: %s"),
		*InActorName, InRecordInterval, *InPoseFilePath);
}

void FActorRegistry::UnregisterActor(AActor* InActor)
{
	if (!InActor)
	{
		return;
	}

	FScopeLock Lock(&RegistryLock);

	for (int32 Index = ActorEntries.Num() - 1; Index >= 0; --Index)
	{
		if (ActorEntries[Index].Actor.Get() == InActor)
		{
			UE_LOG(LogActorRegistry, Log, TEXT("Unregistered actor: %s"), *ActorEntries[Index].ActorName);
			ActorEntries.RemoveAt(Index);
			break;
		}
	}
}

void FActorRegistry::UnregisterAllActors()
{
	FScopeLock Lock(&RegistryLock);

	int32 Count = ActorEntries.Num();
	ActorEntries.Empty();

	if (Count > 0)
	{
		UE_LOG(LogActorRegistry, Log, TEXT("Unregistered all %d actors"), Count);
	}
}

FActorRegistryEntry* FActorRegistry::FindEntry(AActor* InActor)
{
	if (!InActor)
	{
		return nullptr;
	}

	FScopeLock Lock(&RegistryLock);

	for (FActorRegistryEntry& Entry : ActorEntries)
	{
		if (Entry.Actor.Get() == InActor)
		{
			return &Entry;
		}
	}

	return nullptr;
}

TArray<FActorRegistryEntry> FActorRegistry::GetAllActors() const
{
	TArray<FActorRegistryEntry> Result;

	FScopeLock Lock(&RegistryLock);

	for (const FActorRegistryEntry& Entry : ActorEntries)
	{
		if (Entry.IsValid())
		{
			Result.Add(Entry);
		}
	}

	return Result;
}

void FActorRegistry::CleanupInvalidEntries()
{
	FScopeLock Lock(&RegistryLock);
	RemoveInvalidEntries_Internal();
}

void FActorRegistry::RemoveInvalidEntries_Internal()
{
	int32 RemovedCount = 0;
	for (int32 Index = ActorEntries.Num() - 1; Index >= 0; --Index)
	{
		if (!ActorEntries[Index].IsValid())
		{
			ActorEntries.RemoveAt(Index);
			++RemovedCount;
		}
	}

	if (RemovedCount > 0)
	{
		UE_LOG(LogActorRegistry, Warning, TEXT("Cleaned up %d invalid actor entries"), RemovedCount);
	}
}
