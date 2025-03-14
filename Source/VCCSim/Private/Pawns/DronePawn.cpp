﻿/*
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

#include "Pawns/DronePawn.h"
#include "EnhancedInputSubsystems.h"
#include "Components/SplineComponent.h"
#include "Pawns/SimPath.h"

void ADronePawn::OnConstruction(const FTransform& Transform)
{
    Super::OnConstruction(Transform);
    DistanceTraveled = 0.f;
    CourseDistance = 0.f;
    LastCourseDistance = 0.f; 
    Laps = 1.f;
    CalculateDistance();
    FollowThePathAndSteer(1);
}
void ADronePawn::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);
    CalculateDistance();
    FollowThePathAndSteer(DeltaSeconds);
    AutoMove(DeltaSeconds);
}
void ADronePawn::AddMapContext()
{
    if (APlayerController* PlayerController = Cast<APlayerController>(Controller))
    {
        if (UEnhancedInputLocalPlayerSubsystem* Subsystem = ULocalPlayer::GetSubsystem<
            UEnhancedInputLocalPlayerSubsystem>(PlayerController->GetLocalPlayer()))
        {
            Subsystem->AddMappingContext(DroneInputMappingContext, 0);
        }
    }
}

void ADronePawn::SetTarget(FVector Location, FRotator Rotation)
{
    TargetLocation = Location;
    TargetRotation = Rotation;
    bUseTarget = true;
}

bool ADronePawn::IfCloseToTarget(FVector Location, FRotator Rotation) const
{
    // Calculate position error
    const FVector CurrentLocation = GetActorLocation();
    const FRotator CurrentRotation = GetActorRotation();
    const FVector PositionError = Location - CurrentLocation;
    
    // Calculate rotation error
    const FRotator RotationError = Rotation - CurrentRotation;
    
    return PositionError.Size() < PositionThreshold &&
           RotationError.Vector().Size() < RotationThreshold;
}

bool ADronePawn::SetPath(
    TArray<FVector> Positions, TArray<FRotator> Rotations)
{
    if (!Path)
    {
        AsyncTask(ENamedThreads::GameThread, [this ,Positions, Rotations]()
           {
                Path = GetWorld()->SpawnActor<AVCCSimPath>();
                Path->SetNewTrajectory(Positions, Rotations);
           });

        return true;
    }

    Path->SetNewTrajectory(Positions, Rotations);
    return true;
}

void ADronePawn::CalculateDistance()
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

void ADronePawn::FollowThePathAndSteer(float DeltaTime)
{
    if (Path)
    {
        const auto NewLocation = Path->Spline->GetLocationAtDistanceAlongSpline(
            CourseDistance, ESplineCoordinateSpace::World);
        const auto NextRotation = Path->Spline->GetRotationAtDistanceAlongSpline(
            CourseDistance, ESplineCoordinateSpace::World);
        const auto NewRotation = FMath::RInterpTo(GetActorRotation(), NextRotation,
            DeltaTime, 8.f);
        SetActorLocationAndRotation(NewLocation,
                FRotator(0, NewRotation.Yaw, 0));
    }
}

void ADronePawn::AutoMove(double DeltaSeconds)
{
    if (IfAutoMove)
    {
        DistanceTraveled += DeltaSeconds * 100;
    }
}
