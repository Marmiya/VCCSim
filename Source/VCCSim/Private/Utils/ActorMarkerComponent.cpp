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

#include "Utils/ActorMarkerComponent.h"
#include "ProceduralMeshComponent.h"
#include "UObject/ConstructorHelpers.h"

DEFINE_LOG_CATEGORY_STATIC(LogActorMarker, Log, All);

namespace
{
	const FName BaseColorParam(TEXT("BaseColor"));
	const FName EmissiveStrengthParam(TEXT("EmissiveStrength"));
}

UActorMarkerComponent::UActorMarkerComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	bAutoActivate = true;
	SetMobility(EComponentMobility::Movable);

	static ConstructorHelpers::FObjectFinder<UMaterialInterface> MarkerMaterialFinder(
		TEXT("/Script/Engine.Material'/VCCSim/Materials/M_Actor_Marker.M_Actor_Marker'"));
	if (MarkerMaterialFinder.Succeeded())
	{
		BaseMaterial = MarkerMaterialFinder.Object;
	}
}

void UActorMarkerComponent::OnRegister()
{
	Super::OnRegister();

	if (!MeshComponent)
	{
		MeshComponent = NewObject<UProceduralMeshComponent>(this,
			UProceduralMeshComponent::StaticClass(), TEXT("MarkerMesh"));
		MeshComponent->SetupAttachment(this);
		MeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
		MeshComponent->SetCastShadow(false);
		MeshComponent->RegisterComponent();
	}

	RebuildGeometry();
	ApplyMaterial();
	ApplyColor();
}

void UActorMarkerComponent::RebuildGeometry()
{
	if (!MeshComponent)
	{
		return;
	}

	const float S = BaseHalfSize;
	const float H = Height;

	// Apex points down (toward the owner) at the local origin; the square base
	// sits at +Z. Faces are duplicated per-triangle for crisp flat shading.
	const FVector Apex(0.0f, 0.0f, 0.0f);
	const FVector Corners[4] = {
		FVector( S,  S, H),
		FVector(-S,  S, H),
		FVector(-S, -S, H),
		FVector( S, -S, H)
	};

	TArray<FVector> Vertices;
	TArray<int32> Triangles;
	TArray<FVector> Normals;
	TArray<FVector2D> UVs;
	TArray<FLinearColor> VertexColors;
	TArray<FProcMeshTangent> Tangents;

	auto AddTriangle = [&](const FVector& P0, const FVector& P1, const FVector& P2)
	{
		const int32 Start = Vertices.Num();
		const FVector Normal =
			FVector::CrossProduct(P1 - P0, P2 - P0).GetSafeNormal();

		Vertices.Add(P0);
		Vertices.Add(P1);
		Vertices.Add(P2);

		for (int32 i = 0; i < 3; ++i)
		{
			Normals.Add(Normal);
			VertexColors.Add(MarkerColor);
			Tangents.Add(FProcMeshTangent(1.0f, 0.0f, 0.0f));
		}

		UVs.Add(FVector2D(0.5f, 1.0f));
		UVs.Add(FVector2D(0.0f, 0.0f));
		UVs.Add(FVector2D(1.0f, 0.0f));

		Triangles.Add(Start);
		Triangles.Add(Start + 1);
		Triangles.Add(Start + 2);
	};

	// Four side faces (winding chosen so normals point outward / downward).
	for (int32 f = 0; f < 4; ++f)
	{
		AddTriangle(Apex, Corners[(f + 1) % 4], Corners[f]);
	}

	// Top square base (normal up).
	AddTriangle(Corners[0], Corners[1], Corners[2]);
	AddTriangle(Corners[0], Corners[2], Corners[3]);

	MeshComponent->CreateMeshSection_LinearColor(
		0, Vertices, Triangles, Normals, UVs, VertexColors, Tangents, false);

	BuiltBaseHalfSize = S;
	BuiltHeight = H;
}

void UActorMarkerComponent::ApplyMaterial()
{
	if (!MeshComponent || !BaseMaterial)
	{
		return;
	}

	MaterialInstance = UMaterialInstanceDynamic::Create(BaseMaterial, this);
	if (MaterialInstance)
	{
		MeshComponent->SetMaterial(0, MaterialInstance);
	}
}

void UActorMarkerComponent::ApplyColor()
{
	if (MaterialInstance)
	{
		MaterialInstance->SetVectorParameterValue(BaseColorParam, MarkerColor);
	}
}

void UActorMarkerComponent::SetMarkerColor(const FLinearColor& NewColor)
{
	MarkerColor = NewColor;

	if (MeshComponent && BuiltBaseHalfSize > 0.0f)
	{
		RebuildGeometry();
	}
	ApplyColor();
}

void UActorMarkerComponent::TickComponent(float DeltaTime, ELevelTick TickType,
	FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	if (!MeshComponent)
	{
		return;
	}

	if (!FMath::IsNearlyEqual(BuiltBaseHalfSize, BaseHalfSize) ||
		!FMath::IsNearlyEqual(BuiltHeight, Height))
	{
		RebuildGeometry();
	}

	float ZOffset = 0.0f;

	if (bAnimate)
	{
		AnimTime += DeltaTime;

		CurrentYaw = FMath::Fmod(CurrentYaw + SpinSpeed * DeltaTime, 360.0f);
		ZOffset += HoverAmplitude *
			FMath::Sin(AnimTime * HoverFrequency * 2.0f * PI);

		if (MaterialInstance && PulseFrequency > 0.0f)
		{
			const float Alpha =
				0.5f * (FMath::Sin(AnimTime * PulseFrequency * 2.0f * PI) + 1.0f);
			MaterialInstance->SetScalarParameterValue(EmissiveStrengthParam,
				FMath::Lerp(EmissiveMin, EmissiveMax, Alpha));
		}
	}

	MeshComponent->SetRelativeLocationAndRotation(
		FVector(0.0f, 0.0f, ZOffset), FRotator(0.0f, CurrentYaw, 0.0f));

	ApplyColor();
}
