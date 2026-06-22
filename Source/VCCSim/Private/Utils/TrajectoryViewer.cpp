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
#include "Components/InstancedStaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Engine/Engine.h"
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

    // Container actor that OWNS all visualization geometry, so destroying it
    // removes the whole preview. Geometry is drawn with instanced static meshes
    // (one draw call per role) instead of persistent debug lines, which the
    // engine's line batcher re-renders every frame — a heavy cost that scales
    // with pose count and tanks the editor framerate on large multi-building paths.
    FActorSpawnParameters SpawnParams;
    SpawnParams.ObjectFlags = RF_Transient;
    SpawnParams.bNoFail = true;
    AActor* VisualizationActor = World->SpawnActor<AActor>(AActor::StaticClass(),
        FTransform::Identity, SpawnParams);
    if (!VisualizationActor)
    {
        return nullptr;
    }
    #if WITH_EDITOR
        VisualizationActor->SetActorLabel(TEXT("PathVisualization"));
        VisualizationActor->SetFlags(RF_Transient);
        VisualizationActor->Tags.Add(FName("VCCSimPathViz"));
    #endif

    USceneComponent* Root = NewObject<USceneComponent>(VisualizationActor);
    Root->SetMobility(EComponentMobility::Movable);
    VisualizationActor->SetRootComponent(Root);
    Root->RegisterComponent();

    UStaticMesh* SphereMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Sphere"));
    UStaticMesh* ConeMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cone"));
    UStaticMesh* CylinderMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cylinder"));

    // M_PathViz exposes a "Color" vector parameter (set per role below) so each path element can be
    // tinted; the previous GEngine->ArrowMaterial ignored both candidate colour params and stayed black.
    UMaterialInterface* BaseMat = LoadObject<UMaterialInterface>(
        nullptr, TEXT("/VCCSim/Materials/M_PathViz.M_PathViz"));

    auto MakeISM = [&](UStaticMesh* Mesh, const FLinearColor& Color) -> UInstancedStaticMeshComponent*
    {
        UInstancedStaticMeshComponent* ISM = NewObject<UInstancedStaticMeshComponent>(VisualizationActor);
        ISM->SetStaticMesh(Mesh);
        ISM->SetMobility(EComponentMobility::Movable);
        ISM->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        ISM->SetCastShadow(false);
        // Preview-only: hidden in game view, which is what dataset capture runs in,
        // so the path geometry never bleeds into captured images.
        ISM->SetHiddenInGame(true);
        ISM->SetupAttachment(Root);
        ISM->RegisterComponent();
        if (BaseMat)
        {
            UMaterialInstanceDynamic* MID = UMaterialInstanceDynamic::Create(BaseMat, ISM);
            MID->SetVectorParameterValue(TEXT("Color"), Color);
            ISM->SetMaterial(0, MID);
        }
        return ISM;
    };

    // Path line: a thin cylinder per segment (basic-shape cylinder is Z-aligned,
    // 100cm tall, radius 50, centred — so place at the midpoint, align +Z to the
    // segment, and scale length to the segment distance).
    if (CylinderMesh && InPositions.Num() > 1)
    {
        UInstancedStaticMeshComponent* LineISM = MakeISM(CylinderMesh, FLinearColor(0.f, 1.f, 1.f));
        const float Radius = FMath::Max(PathWidth, 1.f);
        for (int32 i = 0; i < InPositions.Num() - 1; ++i)
        {
            const FVector Delta = InPositions[i + 1] - InPositions[i];
            const float Len = Delta.Size();
            if (Len < 1.0f) continue;
            const FQuat Rot = FRotationMatrix::MakeFromZ(Delta / Len).ToQuat();
            const FVector Scale(Radius / 50.f, Radius / 50.f, Len / 100.f);
            LineISM->AddInstance(FTransform(Rot, (InPositions[i] + InPositions[i + 1]) * 0.5f, Scale));
        }
    }

    // Direction arrows: a cone per pose, +Z aligned to the view forward, base at
    // the pose and tip ConeLength ahead.
    if (ConeMesh)
    {
        UInstancedStaticMeshComponent* ArrowISM = MakeISM(ConeMesh, FLinearColor(1.f, 0.f, 0.f));
        const FVector Scale(ConeSize / 50.f, ConeSize / 50.f, ConeLength / 100.f);
        for (int32 i = 0; i < InPositions.Num(); ++i)
        {
            const FVector Forward = InRotations[i].Vector();
            const FQuat Rot = FRotationMatrix::MakeFromZ(Forward).ToQuat();
            ArrowISM->AddInstance(FTransform(Rot, InPositions[i] + Forward * (ConeLength * 0.5f), Scale));
        }
    }

    // Pose markers: yellow spheres at every pose, green at the start, blue at the end.
    if (SphereMesh)
    {
        const float MScale = FMath::Max(ConeSize * 0.5f, 5.f) / 50.f;
        UInstancedStaticMeshComponent* MarkerISM = MakeISM(SphereMesh, FLinearColor(1.f, 1.f, 0.f));
        for (int32 i = 1; i < InPositions.Num() - 1; ++i)
        {
            MarkerISM->AddInstance(FTransform(FQuat::Identity, InPositions[i], FVector(MScale)));
        }
        UInstancedStaticMeshComponent* StartISM = MakeISM(SphereMesh, FLinearColor(0.f, 1.f, 0.f));
        StartISM->AddInstance(FTransform(FQuat::Identity, InPositions[0], FVector(MScale * 1.5f)));
        if (InPositions.Num() > 1)
        {
            UInstancedStaticMeshComponent* EndISM = MakeISM(SphereMesh, FLinearColor(0.f, 0.f, 1.f));
            EndISM->AddInstance(FTransform(FQuat::Identity, InPositions.Last(), FVector(MScale * 1.5f)));
        }
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