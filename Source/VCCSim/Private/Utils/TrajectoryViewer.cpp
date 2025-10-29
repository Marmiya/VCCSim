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

#include "Utils/TrajectoryViewer.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "DrawDebugHelpers.h"
#include "Engine/World.h"

UTrajectoryViewer::UTrajectoryViewer()
{
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.TickInterval = 0.0333f;
}

void UTrajectoryViewer::BeginPlay()
{
    Super::BeginPlay();
    
    if (SplineComponent)
    {
        TotalLength = SplineComponent->GetSplineLength();
    }
    
    if (!PathMesh)
    {
        PathMesh = LoadObject<UStaticMesh>(nullptr,
            TEXT("/Engine/BasicShapes/Cylinder"));
    }

    const int32 EstimatedMeshCount = FMath::CeilToInt(DisplayDistance / StepSize);
    PathMeshes.Reserve(EstimatedMeshCount);
}

void UTrajectoryViewer::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    Super::EndPlay(EndPlayReason);
    ClearSplineMeshes();
}

void UTrajectoryViewer::TickComponent(float DeltaTime, enum ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    if (!SplineComponent || DisplayDistance <= 0.0f || !PathMesh || !PathMaterial)
    {
        return;
    }

    // Only update if the path has actually moved
    if (!FMath::IsNearlyEqual(LastTraveledDistance, TraveledDistance, 10))
    {
        UpdatePartialPath();
        LastTraveledDistance = TraveledDistance;
    }
}

AActor* UTrajectoryViewer::GenerateVisibleElements(
    UWorld* World,
    const TArray<FVector>& InPositions,
    const TArray<FRotator>& InRotations,
    float PathWidth,
    float ConeSize,
    float ConeLength)
{
    if (!World || InPositions.Num() == 0 || InPositions.Num() != InRotations.Num())
    {
        return nullptr;
    }
    
    // Create a container actor for tracking
    FActorSpawnParameters SpawnParams;
    SpawnParams.ObjectFlags = RF_Transient;
    SpawnParams.bNoFail = true;
    AActor* VisualizationActor = World->SpawnActor<AActor>(AActor::StaticClass(),
        FTransform::Identity, SpawnParams);
    #if WITH_EDITOR
        VisualizationActor->SetActorLabel(TEXT("PathVisualization"));
        // Make sure it's marked as deletable and transient
        VisualizationActor->SetFlags(RF_Transient);
        VisualizationActor->Tags.Add(FName("VCCSimPathViz"));
    #endif
    
    // Draw path lines
    for (int32 i = 0; i < InPositions.Num() - 1; ++i)
    {
        FVector Start = InPositions[i];
        FVector End = InPositions[i + 1];
        
        if (FVector::Dist(Start, End) > 1.0f)
        {
            DrawDebugLine(World, Start, End, FColor::Cyan, true, -1.0f, 0, PathWidth);
        }
    }
    
    // Draw position markers and direction arrows
    for (int32 i = 0; i < InPositions.Num(); ++i)
    {
        FVector Position = InPositions[i];
        FRotator Rotation = InRotations[i];
        
        // Choose color based on position
        FColor SphereColor = FColor::Yellow;
        if (i == 0) SphereColor = FColor::Green;
        else if (i == InPositions.Num() - 1) SphereColor = FColor::Blue;
        
        DrawDebugSphere(World, Position, ConeSize, 12, SphereColor, true, -1.0f, 0, 3.0f);
        
        // Draw direction arrow
        FVector ForwardVector = Rotation.RotateVector(FVector(1.f, .0f, .0f));
        FVector ArrowEnd = Position + ForwardVector * ConeLength;
        DrawDebugDirectionalArrow(World, Position, ArrowEnd, ConeSize * 0.6f, FColor::Red, true, -1.0f, 0, 4.0f);
    }
    
    return VisualizationActor;
}

void UTrajectoryViewer::UpdatePartialPath()
{
    const float ComponentZ = GetRelativeLocation().Z;
    const float EndDistance = FMath::Min(TraveledDistance + DisplayDistance, TotalLength);
    
    // Calculate segment counts
    const int32 RequiredMeshCount = FMath::CeilToInt((EndDistance - TraveledDistance) / StepSize);
    const int32 CurrentMeshCount = PathMeshes.Num();
    
    const float DistanceMoved = TraveledDistance - LastTraveledDistance;
    const int32 SegmentsToRemove = FMath::Max(0, FMath::FloorToInt(DistanceMoved / StepSize));
    const int32 SegmentsToAdd = RequiredMeshCount - (CurrentMeshCount - SegmentsToRemove);
    
    // Remove segments from front
    for (int32 i = 0; i < SegmentsToRemove && !PathMeshes.IsEmpty(); ++i)
    {
        if (PathMeshes[0])
        {
            PathMeshes[0]->DestroyComponent();
        }
        PathMeshes.PopFirst();
    }
    
    // Add new segments at the end
    float StartDistance = EndDistance - (SegmentsToAdd * StepSize);
    for (int32 i = 0; i < SegmentsToAdd; ++i)
    {
        float Distance = StartDistance + (i * StepSize);
        float NextDistance = FMath::Min(Distance + StepSize, EndDistance);
        
        FVector StartPos = GetPointAtDistance(Distance);
        StartPos.Z += ComponentZ;
        FVector StartTangent = SplineComponent->GetTangentAtDistanceAlongSpline(
            Distance, ESplineCoordinateSpace::World);
            
        FVector EndPos = GetPointAtDistance(NextDistance);
        EndPos.Z += ComponentZ;
        FVector EndTangent = SplineComponent->GetTangentAtDistanceAlongSpline(
            NextDistance, ESplineCoordinateSpace::World);

        USplineMeshComponent* SplineMesh = NewObject<USplineMeshComponent>(this);
        SplineMesh->SetMobility(EComponentMobility::Movable);
        SplineMesh->SetVisibility(true);
        SplineMesh->SetHiddenInGame(false);
        SplineMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        SplineMesh->SetStaticMesh(PathMesh);
        SplineMesh->SetMaterial(0, PathMaterial);
        SplineMesh->SetForwardAxis(ESplineMeshAxis::X);
        SplineMesh->SetUsingAbsoluteLocation(true);
        SplineMesh->SetUsingAbsoluteRotation(true);
        
        SplineMesh->SetStartAndEnd(StartPos, StartTangent, EndPos, EndTangent);   
        SplineMesh->SetStartScale(FVector2D(PathWidth, PathWidth));
        SplineMesh->SetEndScale(FVector2D(PathWidth, PathWidth));
        
        SplineMesh->RegisterComponent();
        SplineMesh->AttachToComponent(this, 
            FAttachmentTransformRules::KeepWorldTransform);
        
        PathMeshes.PushLast(SplineMesh);
    }
}

void UTrajectoryViewer::ClearSplineMeshes()
{
    while (!PathMeshes.IsEmpty())
    {
        if (PathMeshes[0])
        {
            PathMeshes[0]->DestroyComponent();
        }
        PathMeshes.PopFirst();
    }
}


FVector UTrajectoryViewer::GetPointAtDistance(float Distance) const
{
    return SplineComponent ? SplineComponent->GetLocationAtDistanceAlongSpline(
        Distance, ESplineCoordinateSpace::World) : FVector::ZeroVector;
}