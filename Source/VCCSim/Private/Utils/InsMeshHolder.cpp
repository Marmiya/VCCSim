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

#include "Utils/InsMeshHolder.h"
#include "Components/InstancedStaticMeshComponent.h"

UInsMeshHolder::UInsMeshHolder()
{
	PrimaryComponentTick.bCanEverTick = true;
	bWantsInitializeComponent = true;
	bAutoActivate = true;
	Mobility = EComponentMobility::Movable;
}

void UInsMeshHolder::OnRegister()
{
	Super::OnRegister();

	InstancedMeshComponent = NewObject<UInstancedStaticMeshComponent>(this);
	if (!InstancedMeshComponent)
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to create InstancedMeshComponent"));
		return;
	}
	InstancedMeshComponent->SetUsingAbsoluteLocation(true);
	InstancedMeshComponent->SetUsingAbsoluteRotation(true);
	InstancedMeshComponent->SetUsingAbsoluteScale(true);
	InstancedMeshComponent->SetWorldTransform(FTransform::Identity);
	InstancedMeshComponent->RegisterComponent();

	UStaticMesh* MeshAsset = Cast<UStaticMesh>(StaticLoadObject(
		UStaticMesh::StaticClass(),
		nullptr,
		TEXT("/Engine/BasicShapes/Plane.Plane")
	));
	if (MeshAsset)
	{
		InstancedMeshComponent->SetStaticMesh(MeshAsset);
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to load MeshAsset"));
	}

	UMaterialInterface* MaterialAsset = Cast<UMaterialInterface>(StaticLoadObject(
		UMaterialInterface::StaticClass(),
		nullptr,
		TEXT("/VCCSim/Lidar/Materials/M_Point.M_Point")
	));
	if (MaterialAsset)
	{
		InstancedMeshComponent->SetMaterial(0, MaterialAsset);
	}

	InstancedMeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	InstancedMeshComponent->SetCollisionProfileName(UCollisionProfile::NoCollision_ProfileName);
	InstancedMeshComponent->SetCastShadow(false);
	DisableAllShadows(InstancedMeshComponent);
}

void UInsMeshHolder::CreateStaticMeshes()
{
	InstancedMeshComponentColor = NewObject<UInstancedStaticMeshComponent>(this);
	if (!InstancedMeshComponentColor)
	{
		UE_LOG(LogTemp, Error,
			TEXT("Failed to create InstancedMeshComponentColor"));
		return;
	}
	InstancedMeshComponentColor->SetUsingAbsoluteLocation(true);
	InstancedMeshComponentColor->SetUsingAbsoluteRotation(true);
	InstancedMeshComponentColor->SetUsingAbsoluteScale(true);
	InstancedMeshComponentColor->SetWorldTransform(FTransform::Identity);
	InstancedMeshComponentColor->SetVisibility(true);
	InstancedMeshComponentColor->bVisibleInSceneCaptureOnly = true;
	InstancedMeshComponentColor->RegisterComponent();

	UStaticMesh* MeshAsset = Cast<UStaticMesh>(StaticLoadObject(
		UStaticMesh::StaticClass(),
		nullptr,
		TEXT("/Engine/BasicShapes/Sphere.Sphere")
	));
	if (MeshAsset)
	{
		InstancedMeshComponentColor->SetStaticMesh(MeshAsset);
	}

	UMaterialInterface* BaseMaterial = Cast<UMaterialInterface>(StaticLoadObject(
		UMaterialInterface::StaticClass(),
		nullptr,
		TEXT("/VCCSim/Materials/M_Point_Color.M_Point_Color")
	));
    
	if (BaseMaterial)
	{
		// Create a dynamic material instance
		ColorMaterialInstance = UMaterialInstanceDynamic::Create(BaseMaterial, this);
		if (ColorMaterialInstance)
		{
			InstancedMeshComponentColor->SetMaterial(0, ColorMaterialInstance);
		}
		else
		{
			UE_LOG(LogTemp, Error,
				TEXT("Failed to create dynamic material instance"));
		}
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to load MaterialAsset"));
	}

	InstancedMeshComponentColor->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	InstancedMeshComponentColor->SetCollisionProfileName(UCollisionProfile::NoCollision_ProfileName);
	InstancedMeshComponentColor->SetCastShadow(false);
	DisableAllShadows(InstancedMeshComponentColor);
}

void UInsMeshHolder::ClearAndAddNewInstances(
	const TArray<FTransform>& Transforms) const
{
	InstancedMeshComponent->ClearInstances();
	InstancedMeshComponent->AddInstances(Transforms, false, true);
}

void UInsMeshHolder::ClearAndAddNewInstancesWithColors(
	const TArray<FTransform>& Transforms,
	const TArray<FColor>& Colors) const
{
	if (!InstancedMeshComponentColor || !ColorMaterialInstance)
	{
		UE_LOG(LogTemp, Error,
			TEXT("InstancedMeshComponentColor or ColorMaterialInstance is null"));
		return;
	}

	if (Transforms.Num() == 0)
	{
		UE_LOG(LogTemp, Error, TEXT("Transforms array must be non-empty"));
		return;
	}
		
	if (Transforms.Num() != Colors.Num())
	{
		UE_LOG(LogTemp, Error,
			TEXT("Transforms and Colors arrays must have the same length"));
		return;
	}

	InstancedMeshComponentColor->NumCustomDataFloats = 4;
	InstancedMeshComponentColor->ClearInstances();
	InstancedMeshComponentColor->PreAllocateInstancesMemory(Transforms.Num());

	for (int32 i = 0; i < Transforms.Num(); ++i)
	{
		const FTransform& Transform = Transforms[i];
		const FColor& Color = Colors[i];

		int32 InstanceIndex = InstancedMeshComponentColor->AddInstance(Transform);
        
		if (InstanceIndex != INDEX_NONE)
		{
			float ColorData[4] = {
				Color.R / 255.0f,  // Red
				Color.G / 255.0f,  // Green
				Color.B / 255.0f,  // Blue
				Color.A / 255.0f   // Alpha
			};
        
			InstancedMeshComponentColor->SetCustomData(InstanceIndex,
				TArrayView<const float>(ColorData, 4));
		}
	}

	InstancedMeshComponentColor->MarkRenderStateDirty();
}

void UInsMeshHolder::TickComponent(float DeltaTime,
	enum ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	
	IMCCTimer += DeltaTime;

	if (IMCCTimer >= 0.1f)
	{
		TPair<TArray<FTransform>, TArray<FColor>> Update;
		while (PendingUpdates.Dequeue(Update))
		{
			ClearAndAddNewInstancesWithColors(Update.Key, Update.Value);
		}
		IMCCTimer = 0.0f;
	}
}

void UInsMeshHolder::QueueInstanceUpdate(
	const TArray<FTransform>& Transforms,
	const TArray<FColor>& Colors)
{
	FScopeLock Lock(&UpdateLock);
	PendingUpdates.Empty();
	PendingUpdates.Enqueue(TPair<TArray<FTransform>, TArray<FColor>>(Transforms, Colors));
}

UInstancedStaticMeshComponent*	UInsMeshHolder::GetInstancedMeshComponent() const
{
	return InstancedMeshComponent;
}

UInstancedStaticMeshComponent* UInsMeshHolder::GetInstancedMeshComponentColor() const
{
	return InstancedMeshComponentColor;
}

void UInsMeshHolder::DisableAllShadows(UInstancedStaticMeshComponent* Component)
{
	if (!Component) return;
    
	Component->SetCastShadow(false);
	Component->bCastDynamicShadow = false;
	Component->bCastStaticShadow = false;
	Component->bCastVolumetricTranslucentShadow = false;
	Component->bCastContactShadow = false;
	Component->bSelfShadowOnly = false;
	Component->bCastFarShadow = false;
	Component->bCastInsetShadow = false;
	Component->bCastCinematicShadow = false;
	Component->bAffectDynamicIndirectLighting = false;
	Component->bAffectDistanceFieldLighting = false;
	Component->CastShadow = false;
}
