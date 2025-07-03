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
#include "Components/SceneComponent.h"
#include "Materials/MaterialInstanceDynamic.h"

#include "InsMeshHolder.generated.h"


UCLASS()
class VCCSIM_API UInsMeshHolder : public USceneComponent
{
	GENERATED_BODY()

public:
	UInsMeshHolder();
	virtual void OnRegister() override;
	void CreateStaticMeshes();
	void ClearAndAddNewInstances(const TArray<FTransform>& Transforms) const;
	void ClearAndAddNewInstancesWithColors(const TArray<FTransform>& Transforms, 
		const TArray<FColor>& Colors) const;

	virtual void TickComponent(float DeltaTime, enum ELevelTick TickType,
		FActorComponentTickFunction* ThisTickFunction) override;
    
	// Thread-safe method to queue updates
	void QueueInstanceUpdate(const TArray<FTransform>& Transforms,
		const TArray<FColor>& Colors);
	
	// Thread-safe queue for pending updates
	FCriticalSection UpdateLock;
	TQueue<TPair<TArray<FTransform>, TArray<FColor>>> PendingUpdates;

	UInstancedStaticMeshComponent* GetInstancedMeshComponent() const;
	UInstancedStaticMeshComponent* GetInstancedMeshComponentColor() const;

private:
	UPROPERTY()
	UInstancedStaticMeshComponent* InstancedMeshComponent = nullptr;
	UPROPERTY()
	UInstancedStaticMeshComponent* InstancedMeshComponentColor = nullptr;
	float IMCCTimer = 0.f;
	UPROPERTY()
	UMaterialInstanceDynamic* ColorMaterialInstance = nullptr;
	void DisableAllShadows(UInstancedStaticMeshComponent* Component);
};