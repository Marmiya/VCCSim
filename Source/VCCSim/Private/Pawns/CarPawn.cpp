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

#include "Pawns/CarPawn.h"
#include "Components/SplineComponent.h"
#include "Kismet/GameplayStatics.h"
#include "Pawns/SimPath.h"
#include <cmath>

float AsinD(float value)
{
	constexpr float RadToDeg = 180.0f / PI;
	return std::asin(value) * RadToDeg;
}

ACarPawn::ACarPawn()
{	
	PrimaryActorTick.bCanEverTick = true;
	Chassis = CreateDefaultSubobject<UChildActorComponent>(TEXT("Chassis"));
	RootComponent = Chassis;
	BodyAndAccessories = CreateDefaultSubobject<UChildActorComponent>(TEXT("BodyAndAccessories"));
	BodyAndAccessories->SetupAttachment(Chassis);
	Body = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Body"));
	Body->SetupAttachment(BodyAndAccessories);
	Wheels = CreateDefaultSubobject<UChildActorComponent>(TEXT("Wheels"));
	Wheels->SetupAttachment(Chassis);
	WheelFL = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("WheelFL"));
	WheelFL->SetupAttachment(Wheels);
	WheelFR = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("WheelFR"));
	WheelFR->SetupAttachment(Wheels);
	WheelRL = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("WheelRL"));
	WheelRL->SetupAttachment(Wheels);
	WheelRR = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("WheelRR"));
	WheelRR->SetupAttachment(Wheels);

	TraceGround = true;
}

void ACarPawn::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);
	
	DistanceTraveled = 0.f;
	CourseDistance = 0.f;
	LastCourseDistance = 0.f;
	Laps = 1.f;
	
	SetTraceIgnores();
	// CarMeshSetup();
	CalculateDistance();
	FollowThePathAndSteer();
	ActorGroundTrace();
	InitialState();

	if (TraceGround)
	{
		for (int32 i = 0; i < 2; i++)
		{
			WheelsGroundTrace();
			ChassisRollandPitch();
			ChassisRelocation();
		}
	}

	BodyAndAccessories->SetWorldLocationAndRotation(
		Chassis->GetComponentLocation(),
		Chassis->GetComponentRotation());

	WheelsDrive();
	Vibrations();
	CalculateSpeed();
}

void ACarPawn::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);

	if (bEnableManualControl)
	{
		return;
	}
	AutoMove(DeltaSeconds);
	SetTraceIgnores();
	CalculateDistance();
	FollowThePathAndSteer();
	ActorGroundTrace();
	InitialState();
	if (TraceGround)
	{
		for (int32 i = 0; i < 2; i++)
		{
			WheelsGroundTrace();
			ChassisRollandPitch();
			ChassisRelocation();
		}
	}
	WheelsDrive();
	Vibrations();
	CalculateSpeed();
}

void ACarPawn::SetTraceIgnores()
{
	// Find all actors which has Tag IgnoreTrace.
	TArray<AActor*> FoundActors;
	UGameplayStatics::GetAllActorsWithTag(GetWorld(), "IgnoreTrace", FoundActors);
	for (AActor* Actor : FoundActors)
	{
		ActorsToIgnore.Add(Actor);
	}
}

// void ACarPawn::CarMeshSetup()
// {
// 	if (BodyMesh)
// 	{
// 		Body->SetStaticMesh(BodyMesh);
// 	}
// 	if (FrontWheelsMesh)
// 	{
// 		WheelFL->SetStaticMesh(FrontWheelsMesh);
// 		WheelFR->SetStaticMesh(FrontWheelsMesh);
// 	}
// 	if (RearWheelsMesh)
// 	{
// 		WheelRL->SetStaticMesh(RearWheelsMesh);
// 		WheelRR->SetStaticMesh(RearWheelsMesh);
// 	}
// }

void ACarPawn::CalculateDistance()
{
	if (Path)
	{
		Laps = FMath::Floor(DistanceTraveled / Path->Spline->GetSplineLength());
		CourseDistance = DistanceTraveled - Laps * Path->Spline->GetSplineLength();
	}
	else
	{
		CourseDistance = DistanceTraveled;
	}
}

void ACarPawn::FollowThePathAndSteer()
{
	if (Path)
	{
		const auto NewLocation = Path->Spline->GetLocationAtDistanceAlongSpline(
			CourseDistance, ESplineCoordinateSpace::World);
		const auto NextRotation = Path->Spline->GetRotationAtDistanceAlongSpline(
			CourseDistance, ESplineCoordinateSpace::World);
		const auto NewRotation = FMath::RInterpTo(GetActorRotation(), NextRotation,
			1 / TickFPS, 8.f);
		if (TraceGround)
		{
			SetActorLocationAndRotation(NewLocation,
				FRotator(0, NewRotation.Yaw, 0));
		}
		else
		{
			SetActorLocationAndRotation(NewLocation, NewRotation);
		}
		
		const auto FrontRotation = Path->Spline->GetRotationAtDistanceAlongSpline(
			CourseDistance + FrontAxleOffset, ESplineCoordinateSpace::World);
		const auto DeltaYaw = -FMath::FindDeltaAngleDegrees(
			FrontRotation.Yaw, GetActorRotation().Yaw);
		Steering = FMath::Clamp(
			FMath::FInterpTo(Steering, DeltaYaw, 1 / TickFPS, 3.f),
			-45.f, 45.f);
	}
}

void ACarPawn::ActorGroundTrace()
{
	SetActorScale3D({1,1,1});
	if (TraceGround)
	{
		const auto CurrentPosition = GetActorLocation();
		const auto CurrentYaw = GetActorRotation().Yaw;
		const auto TraceStart = CurrentPosition + FVector(0, 0, 150);
		const auto TraceEnd = CurrentPosition - FVector(0, 0, 10000);
		FHitResult HitResult;
		FCollisionQueryParams TraceParams;
		TraceParams.AddIgnoredActors(ActorsToIgnore);
		if (GetWorld()->LineTraceSingleByChannel(HitResult, TraceStart, TraceEnd,
			ECollisionChannel::ECC_Visibility, TraceParams))
		{
			SetActorLocationAndRotation(HitResult.Location,
				FRotator(0, CurrentYaw, 0));
			return;
		}
	}
	ChassisPitch = 0.f;
	ChassisRoll = 0.f;
}

void ACarPawn::InitialState()
{
    // Get world position for chassis
    const auto ActorWorldPos = GetActorLocation();
    const auto ActorWorldRot = GetActorRotation();
    
    const auto NewLocationZ = WheelFR->Bounds.SphereRadius;
    const auto MappedSuspensionStiffness = FMath::GetMappedRangeValueClamped(
        FVector2D(0.0f, 10.0f), FVector2D(2.0f, 0.3f),
        SuspensionStiffness);
    const auto SteeringFactor = -0.1 * FMath::Clamp(Steering, -45.f, 45.f);
    const auto ClampedDrift = FMath::Clamp(Drift, -40.f, 40.f);
    const auto PerlinNoiseDrift = PerlinNoise(0., 0.5,
    	0.1, OverallVibrationMultiplier * ClampedDrift);
    const auto DriftFactor = ClampedDrift + PerlinNoiseDrift;
    
    // Calculate world rotation
    const auto NewRotationRoll = ChassisRoll + MappedSuspensionStiffness *
    	(SteeringFactor + DriftFactor * -0.1);
    const auto NewRotationPitch = ChassisPitch;
    const auto NewRotationYaw = DriftFactor + ActorWorldRot.Yaw;
    
    // Set chassis in world space
    Chassis->SetWorldLocation(FVector(ActorWorldPos.X, ActorWorldPos.Y,
    	ActorWorldPos.Z + NewLocationZ));
    Chassis->SetWorldRotation(FRotator(NewRotationPitch, NewRotationYaw, NewRotationRoll));

    const auto FormerFactor = FMath::Clamp(Steering, -45.f, 45.f) - DriftFactor;
    
    const FVector ForwardVector = GetActorForwardVector();
	const FVector RightVector = GetActorRightVector();
	
    FVector WheelFRPos = ActorWorldPos + ForwardVector * FrontAxleOffset + 
                        RightVector * FrontWheelsSideOffset;
    FRotator WheelFRRot = FRotator(0, FormerFactor + ActorWorldRot.Yaw,
        -FMath::Clamp(FrontCamber, 0.f, 10.f));
    WheelFR->SetWorldLocationAndRotation(WheelFRPos, WheelFRRot);

    FVector WheelFLPos = ActorWorldPos +ForwardVector * FrontAxleOffset - 
                        RightVector * FrontWheelsSideOffset;
    FRotator WheelFLRot = FRotator(0, FormerFactor + ActorWorldRot.Yaw + 180,
        -FMath::Clamp(FrontCamber, 0.f, 10.f));
    WheelFL->SetWorldLocationAndRotation(WheelFLPos, WheelFLRot);

    FVector WheelRRPos = ActorWorldPos - ForwardVector * RearAxleOffset + 
                        RightVector * RearWheelsSideOffset;
    FRotator WheelRRRot = FRotator(0, ActorWorldRot.Yaw,
        -FMath::Clamp(RearCamber, 0.f, 10.f));
    WheelRR->SetWorldLocationAndRotation(WheelRRPos, WheelRRRot);
    FVector WheelRLPos = ActorWorldPos - ForwardVector * RearAxleOffset - 
                        RightVector * RearWheelsSideOffset;
    FRotator WheelRLRot = FRotator(0, ActorWorldRot.Yaw + 180,
        -FMath::Clamp(RearCamber, 0.f, 10.f));
    WheelRL->SetWorldLocationAndRotation(WheelRLPos, WheelRLRot);
}

void ACarPawn::WheelsGroundTrace()
{
	for (auto Wheel : {WheelFL, WheelFR, WheelRL, WheelRR})
	{
		const auto CurrentPosition = Wheel->GetComponentLocation();
		const auto UpVector = Chassis->GetUpVector();
		const auto TraceStart = CurrentPosition + 100 * UpVector;
		const auto TraceEnd = CurrentPosition - 10000 * UpVector;
		FHitResult HitResult;
		FCollisionQueryParams TraceParams;
		TraceParams.AddIgnoredActors(ActorsToIgnore);
		TraceParams.AddIgnoredActor(this);
		if (GetWorld()->LineTraceSingleByChannel(HitResult, TraceStart, TraceEnd,
			ECollisionChannel::ECC_Visibility, TraceParams))
		{
			Wheel->SetWorldLocation(HitResult.Location + 0.95 *
				Wheel->GetStaticMesh()->GetBoundingBox().Max.Z * UpVector);
		}
		else
		{
			UStaticMesh* StaticMesh = Wheel->GetStaticMesh();
			if (StaticMesh)
			{
				Wheel->SetWorldLocation({CurrentPosition.X, CurrentPosition.Y,
					0.95 * StaticMesh->GetBoundingBox().Max.Z * UpVector.Z});
			}
			else
			{
				// Fallback if static mesh is null
				Wheel->SetWorldLocation({CurrentPosition.X, CurrentPosition.Y, 20.0f * UpVector.Z}); // Use a default value
			}
		}
	}
}

void ACarPawn::ChassisRollandPitch()
{
	const auto FLZ = WheelFL->GetComponentLocation().Z;
	const auto FRZ = WheelFR->GetComponentLocation().Z;
	const auto RLZ = WheelRL->GetComponentLocation().Z;
	const auto RRZ = WheelRR->GetComponentLocation().Z;
	ChassisRoll = FMath::Clamp(
		(AsinD((FLZ-FRZ)/(2*FrontWheelsSideOffset))
		+AsinD((RLZ-RRZ)/(2*FrontWheelsSideOffset)))
		/2.,
		-85., 85.);
	ChassisPitch = FMath::Clamp(
		(AsinD((FLZ-RLZ)/(RearAxleOffset+FrontAxleOffset)) +
		AsinD((FRZ-RRZ)/(RearAxleOffset+FrontAxleOffset)))
		/ 2.,
		-85., 85.);
}

void ACarPawn::ChassisRelocation()
{
	const auto AvgWorldLocation = (WheelFL->GetComponentLocation() +
		WheelFR->GetComponentLocation() +
		WheelRL->GetComponentLocation() +
		WheelRR->GetComponentLocation()) / 4;

	const auto ChassisOffset = GetActorForwardVector() *
		((FrontAxleOffset - RearAxleOffset) / 2);
    
	Chassis->SetWorldLocation(AvgWorldLocation - ChassisOffset);
}

void ACarPawn::WheelsDrive()
{
	if (RearBreaksLock)
	{
		if (FrontBreaksLock)
		{
			return;
		}
	}
	else
	{
		const auto WorldRotRL = FRotator(
			CalculateWheelRotation(WheelRL, -120), 0, 0);
		const auto WorldRotRR = FRotator(
			-CalculateWheelRotation(WheelRR, 0), 0, 0);
        
		WheelRL->AddLocalRotation(WorldRotRL);
		WheelRR->AddLocalRotation(WorldRotRR);
	}
    
	const auto WorldRotFL = FRotator(
		CalculateWheelRotation(WheelFL, 120), 0, 0);
	const auto WorldRotFR = FRotator(
		-CalculateWheelRotation(WheelFR, -80), 0, 0);
    
	WheelFL->AddLocalRotation(WorldRotFL);
	WheelFR->SetWorldRotation(WheelFR->GetComponentRotation() + WorldRotFR);
	
}

void ACarPawn::Vibrations()
{
	// Apply vibrations in world space
	SingleWheelVibration(WheelRR, 0);
	SingleWheelVibration(WheelRL, 1.);
	SingleWheelVibration(WheelFR, 2.);
	SingleWheelVibration(WheelFL, 3.);

	const auto Noise = PerlinNoise(0., BodyVibrationSpeed,
		BodyVibrationAmount, OverallVibrationMultiplier);
	const auto MappedSuspensionStiffness = FMath::GetMappedRangeValueClamped(
		FVector2D(0.0f, 10.0f), FVector2D(.0f, 5.f),
		SuspensionStiffness);

	// Convert body movement to world space
	const auto BodyWorldPos = Chassis->GetComponentLocation() + 
		GetActorUpVector() * (BodyHeight + Noise + MappedSuspensionStiffness);
	const auto BodyWorldRot = GetActorRotation() + 
		FRotator(BodyPitch + Noise / 1.5, 0, BodyRoll);

	Body->SetWorldLocationAndRotation(BodyWorldPos, BodyWorldRot);
}

void ACarPawn::CalculateSpeed()
{
	bSpeed = (CourseDistance - LastCourseDistance) / (1 / TickFPS) * 0.01;
	LastCourseDistance = CourseDistance;
}

void ACarPawn::AutoMove(double DeltaSec)
{
	if (IfAutoMove)
	{
		DistanceTraveled += DeltaSec * 100;
	}
}

double ACarPawn::PerlinNoise(double TimeOffset, double Speed, double Amount,
                             double Multiplier)
{
	const auto Noise = FMath::PerlinNoise1D(
		(GetWorld()->GetTimeSeconds() +TimeOffset) * Speed);
	return Amount * Multiplier * Noise;
}

double ACarPawn::CalculateWheelRotation(USceneComponent* WheelMesh,
	double Offset)
{
	return Offset + CourseDistance / (WheelMesh->Bounds.SphereRadius * 2. * PI) * 360.;
}

void ACarPawn::SingleWheelVibration(USceneComponent* WheelMesh, double Offset)
{
	const auto CurLocation = WheelMesh->GetComponentLocation();
	WheelMesh->SetWorldLocation(CurLocation + FVector{0, 0, WheelVibration(Offset)});
}

double ACarPawn::WheelVibration(double TimeOffset)
{
	return PerlinNoise(TimeOffset, WheelsVibrationSpeed,
		WheelsVibrationAmount, OverallVibrationMultiplier);
}

FVector ACarPawn::GetPhysicsLinearVelocity() const
{
	if (Body && Body->IsSimulatingPhysics())
	{
		return Body->GetPhysicsLinearVelocity();
	}
	return FVector::ZeroVector;
}

FVector ACarPawn::GetPhysicsAngularVelocityInDegrees() const
{
	if (Body && Body->IsSimulatingPhysics())
	{
		FVector AngularVelocityRad = Body->GetPhysicsAngularVelocityInRadians();
		return FMath::RadiansToDegrees(AngularVelocityRad);
	}
	return FVector::ZeroVector;
}

bool ACarPawn::SetTarget(const FVector& TargetPosition, const FRotator& TargetRotation)
{
	FVector CurrentPosition = GetActorLocation();
	FRotator CurrentRotation = GetActorRotation();

	TArray<FVector> Positions;
	TArray<FRotator> Rotations;

	Positions.Add(CurrentPosition);
	Rotations.Add(CurrentRotation);

	Positions.Add(TargetPosition);
	Rotations.Add(TargetRotation);

	if (!Path)
	{
		AsyncTask(ENamedThreads::GameThread, [this, Positions, Rotations]()
			{
				Path = GetWorld()->SpawnActor<AVCCSimPath>();
				Path->SetNewTrajectory(Positions, Rotations);
			});
		return true;
	}

	Path->SetNewTrajectory(Positions, Rotations);

	return true;
}

bool ACarPawn::SetPath(const TArray<FVector>& Positions, const TArray<FRotator>& Rotations)
{
	if (!Path)
	{
		AsyncTask(ENamedThreads::GameThread, [this, Positions, Rotations]()
			{
				Path = GetWorld()->SpawnActor<AVCCSimPath>();
				Path->SetNewTrajectory(Positions, Rotations);
			});
		return true;
	}
	Path->SetNewTrajectory(Positions, Rotations);
	return true;
}