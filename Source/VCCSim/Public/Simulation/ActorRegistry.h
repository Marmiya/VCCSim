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

DECLARE_LOG_CATEGORY_EXTERN(LogActorRegistry, Log, All);

struct VCCSIM_API FActorRegistryEntry
{
	TWeakObjectPtr<AActor> Actor;
	FString ActorName;
	FString PoseFilePath;
	double RecordInterval;
	double LastRecordTime;

	FActorRegistryEntry() = default;

	FActorRegistryEntry(AActor* InActor, const FString& InActorName, const FString& InPoseFilePath, double InRecordInterval)
		: Actor(InActor)
		, ActorName(InActorName)
		, PoseFilePath(InPoseFilePath)
		, RecordInterval(InRecordInterval)
		, LastRecordTime(-1.f)
	{}

	bool IsValid() const
	{
		return Actor.IsValid();
	}

	AActor* GetActor() const
	{
		return Actor.Get();
	}
};

class VCCSIM_API FActorRegistry
{
public:
	FActorRegistry() = default;
	~FActorRegistry() = default;

	void RegisterActor(AActor* InActor, const FString& InActorName, const FString& InPoseFilePath, double InRecordInterval);
	void UnregisterActor(AActor* InActor);
	void UnregisterAllActors();

	FActorRegistryEntry* FindEntry(AActor* InActor);
	TArray<FActorRegistryEntry> GetAllActors() const;

	void CleanupInvalidEntries();
	int32 GetActorCount() const { return ActorEntries.Num(); }

private:
	mutable FCriticalSection RegistryLock;
	TArray<FActorRegistryEntry> ActorEntries;

	void RemoveInvalidEntries_Internal();
};
