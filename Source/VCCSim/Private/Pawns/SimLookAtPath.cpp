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

#include "Pawns/SimLookAtPath.h"
#include "Components/SplineComponent.h"
#include "Components/StaticMeshComponent.h"
#include "UObject/ConstructorHelpers.h"
#include "Kismet/GameplayStatics.h"

DEFINE_LOG_CATEGORY_STATIC(LogSimLookAtPath, Log, All);

AVCCSimLookAtPath::AVCCSimLookAtPath()
{
	PrimaryActorTick.bCanEverTick = false;

	Spline = CreateDefaultSubobject<USplineComponent>(TEXT("Spline"));
	RootComponent = Spline;

	TargetPoint = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("TargetPoint"));
	TargetPoint->SetupAttachment(RootComponent);

	static ConstructorHelpers::FObjectFinder<UStaticMesh> SphereMesh(
		TEXT("/Engine/BasicShapes/Sphere.Sphere"));
	if (SphereMesh.Succeeded())
	{
		TargetPoint->SetStaticMesh(SphereMesh.Object);
		TargetPoint->SetWorldScale3D(FVector(0.3f));
	}
	TargetPoint->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	TargetPoint->SetRelativeLocation(FVector(0.f, 0.f, 300.f));

	SamplingMode    = ELookAtSamplingMode::ControlPoints;
	OrientationMode = EOrientationMode::LookAtTarget;
	NumDivisions    = 50;
	PathLength      = 0.f;
	NumSamplePoints = 0;
}

void AVCCSimLookAtPath::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);

	DiscoverTraceIgnores();

	PathLength      = Spline->GetSplineLength();
	NumSamplePoints = GetNumSamplePoints();
}

void AVCCSimLookAtPath::DiscoverTraceIgnores()
{
	TArray<AActor*> FoundActors;
	UGameplayStatics::GetAllActorsWithTag(GetWorld(), "IgnoreTrace", FoundActors);
	for (AActor* Actor : FoundActors)
	{
		ExcludedInTrace.AddUnique(Actor);
	}
}

void AVCCSimLookAtPath::SnapAllPointsToGround()
{
	for (int32 i = 0; i < Spline->GetNumberOfSplinePoints(); i++)
	{
		FVector Location = Spline->GetLocationAtSplinePoint(i, ESplineCoordinateSpace::World);
		FHitResult HitResult;
		FVector Start = Location + FVector(0, 0, 150);
		FVector End   = Location - FVector(0, 0, 10000);
		FCollisionQueryParams TraceParams;
		TraceParams.AddIgnoredActors(ExcludedInTrace);
		if (GetWorld()->LineTraceSingleByChannel(HitResult, Start, End,
			ECC_Visibility, TraceParams))
		{
			Spline->SetLocationAtSplinePoint(i, HitResult.Location,
				ESplineCoordinateSpace::World);
		}
	}
	Spline->UpdateSpline();
}

void AVCCSimLookAtPath::FlattenAllPointsToSameHeight()
{
	const int32 NumPoints = Spline->GetNumberOfSplinePoints();
	if (NumPoints == 0) return;

	const float TargetZ =
		Spline->GetLocationAtSplinePoint(0, ESplineCoordinateSpace::World).Z;

	for (int32 i = 1; i < NumPoints; i++)
	{
		FVector Location = Spline->GetLocationAtSplinePoint(i, ESplineCoordinateSpace::World);
		Location.Z = TargetZ;
		Spline->SetLocationAtSplinePoint(i, Location, ESplineCoordinateSpace::World);
	}
	Spline->UpdateSpline();
	PathLength = Spline->GetSplineLength();
}

void AVCCSimLookAtPath::GetSamplePoses(
	TArray<FVector>& OutPositions, TArray<FRotator>& OutRotations) const
{
	OutPositions.Empty();
	OutRotations.Empty();

	if (!Spline) return;

	if (OrientationMode == EOrientationMode::FreeOrientation)
	{
		const int32 NumPoints = Spline->GetNumberOfSplinePoints();
		const bool bHasFreeRots = (FreeOrientations.Num() == NumPoints);
		OutPositions.Reserve(NumPoints);
		OutRotations.Reserve(NumPoints);

		for (int32 i = 0; i < NumPoints; i++)
		{
			const FVector Position =
				Spline->GetLocationAtSplinePoint(i, ESplineCoordinateSpace::World);
			OutPositions.Add(Position);

			if (bHasFreeRots)
			{
				OutRotations.Add(FreeOrientations[i]);
			}
			else if (TargetPoint)
			{
				const FVector Dir = (TargetPoint->GetComponentLocation() - Position).GetSafeNormal();
				OutRotations.Add(Dir.Rotation());
			}
			else
			{
				OutRotations.Add(FRotator::ZeroRotator);
			}
		}
		return;
	}

	if (!TargetPoint) return;
	const FVector TargetLocation = TargetPoint->GetComponentLocation();

	if (SamplingMode == ELookAtSamplingMode::ControlPoints)
	{
		const int32 NumPoints = Spline->GetNumberOfSplinePoints();
		OutPositions.Reserve(NumPoints);
		OutRotations.Reserve(NumPoints);

		for (int32 i = 0; i < NumPoints; i++)
		{
			const FVector Position =
				Spline->GetLocationAtSplinePoint(i, ESplineCoordinateSpace::World);
			OutPositions.Add(Position);
			OutRotations.Add((TargetLocation - Position).GetSafeNormal().Rotation());
		}
	}
	else
	{
		OutPositions.Reserve(NumDivisions);
		OutRotations.Reserve(NumDivisions);

		const float SplineLen = Spline->GetSplineLength();
		for (int32 i = 0; i < NumDivisions; i++)
		{
			const float Distance = SplineLen * static_cast<float>(i) /
				static_cast<float>(NumDivisions - 1);
			const FVector Position =
				Spline->GetLocationAtDistanceAlongSpline(Distance, ESplineCoordinateSpace::World);
			OutPositions.Add(Position);
			OutRotations.Add((TargetLocation - Position).GetSafeNormal().Rotation());
		}
	}
}

int32 AVCCSimLookAtPath::GetNumSamplePoints() const
{
	if (OrientationMode == EOrientationMode::FreeOrientation)
		return Spline ? Spline->GetNumberOfSplinePoints() : 0;

	if (SamplingMode == ELookAtSamplingMode::ControlPoints)
		return Spline ? Spline->GetNumberOfSplinePoints() : 0;
	return NumDivisions;
}
