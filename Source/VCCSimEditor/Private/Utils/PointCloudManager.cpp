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
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program. If not, see <https://www.gnu.org/licenses/>.
*/

DEFINE_LOG_CATEGORY_STATIC(LogPointCloudManager, Log, All);

#include "Utils/PointCloudManager.h"
#include "Engine/World.h"
#include "Engine/Engine.h"
#include "EngineUtils.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "Components/SceneComponent.h"
#include "ProceduralMeshComponent.h"
#include "Materials/MaterialInterface.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "DataStructures/PointCloudRenderer.h"
#include "IO/PLYUtils.h"

FPointCloudManager::FPointCloudManager()
{
}

FPointCloudManager::~FPointCloudManager()
{
    ClearVisualization();
}

bool FPointCloudManager::LoadFromFile(const FString& FilePath)
{
    PointCloudData.Empty();
    bPointCloudLoaded = false;
    bPointCloudHasColors = false;
    bPointCloudHasNormals = false;
    PointCloudCount = 0;

    FPLYLoader::FPLYLoadResult LoadResult = FPLYLoader::LoadPLYFile(FilePath, FLinearColor::White);

    if (LoadResult.bSuccess && LoadResult.PointCount > 0)
    {
        PointCloudData = MoveTemp(LoadResult.Points);
        PointCloudCount = LoadResult.PointCount;
        bPointCloudHasColors = LoadResult.bHasColors;
        bPointCloudHasNormals = LoadResult.bHasNormals;

        const int32 MaxPointsLimit = 100000;
        if (PointCloudCount > MaxPointsLimit)
        {
            UE_LOG(LogPointCloudManager, Warning, TEXT("Point cloud has %d points, exceeding limit of %d. Downsampling..."),
                   PointCloudCount, MaxPointsLimit);

            TArray<FRatPoint> DownsampledPoints;
            const int32 Step = FMath::Max(1, PointCloudCount / MaxPointsLimit);

            DownsampledPoints.Reserve(MaxPointsLimit);
            for (int32 i = 0; i < PointCloudData.Num() && DownsampledPoints.Num() < MaxPointsLimit; i += Step)
            {
                DownsampledPoints.Add(PointCloudData[i]);
            }

            PointCloudData = MoveTemp(DownsampledPoints);
            PointCloudCount = PointCloudData.Num();

            UE_LOG(LogPointCloudManager, Log, TEXT("Downsampled point cloud to %d points (%.1f%% reduction)"),
                   PointCloudCount,
                   (1.0f - (float)PointCloudCount / (float)LoadResult.PointCount) * 100.0f);
        }

        bPointCloudLoaded = true;
        LoadedPointCloudPath = FilePath;

        UE_LOG(LogPointCloudManager, Log, TEXT("Final point cloud: %d points (Colors: %s, Normals: %s)"),
            PointCloudCount,
            bPointCloudHasColors ? TEXT("Yes") : TEXT("No"),
            bPointCloudHasNormals ? TEXT("Yes") : TEXT("No"));

        return true;
    }

    UE_LOG(LogPointCloudManager, Error, TEXT("Failed to load point cloud from: %s"), *FilePath);
    return false;
}

void FPointCloudManager::ShowVisualization(UWorld* World, bool bWithColors, bool bWithNormals)
{
    if (!World || PointCloudData.Num() == 0)
    {
        return;
    }
    CreateColoredPointCloudVisualization(World, bWithColors, bWithNormals);
}

void FPointCloudManager::UpdateNormalVisibility(bool bShow)
{
    if (!bPointCloudVisualized)
    {
        return;
    }

    AActor* Actor = PointCloudActor.Get();
    if (!Actor)
    {
        return;
    }

    if (UInstancedStaticMeshComponent* NormalComponent = NormalLinesInstancedComponent.Get())
    {
        NormalComponent->ClearInstances();
    }

    if (bShow && bPointCloudHasNormals)
    {
        CreateNormalVisualization(Actor);
    }
}

void FPointCloudManager::ClearVisualization()
{
    bool bHadAnythingToClear = false;

    if (ParticlePointCloudRenderer.IsValid())
    {
        ParticlePointCloudRenderer->ClearPointCloud();
        ParticlePointCloudRenderer.Reset();
        bHadAnythingToClear = true;
    }

    if (PointCloudInstancedComponent.IsValid())
    {
        PointCloudInstancedComponent->ClearInstances();
        PointCloudInstancedComponent.Reset();
        bHadAnythingToClear = true;
    }

    if (NormalLinesInstancedComponent.IsValid())
    {
        NormalLinesInstancedComponent->ClearInstances();
        NormalLinesInstancedComponent.Reset();
        bHadAnythingToClear = true;
    }

    if (PointCloudActor.IsValid())
    {
        PointCloudActor->Destroy();
        PointCloudActor.Reset();
        bHadAnythingToClear = true;
    }

    if (UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr)
    {
        for (TActorIterator<AActor> ActorIterator(World); ActorIterator; ++ActorIterator)
        {
            AActor* Actor = *ActorIterator;
            if (Actor && Actor->GetActorLabel() == TEXT("VCCSim_PointCloud"))
            {
                UE_LOG(LogPointCloudManager, Warning, TEXT("Cleaning up orphaned VCCSim_PointCloud actor: %s"), *Actor->GetName());
                Actor->Destroy();
                bHadAnythingToClear = true;
            }
        }
    }

    bPointCloudVisualized = false;

    if (bHadAnythingToClear)
    {
        UE_LOG(LogPointCloudManager, Log, TEXT("Cleared point cloud visualization"));
    }
}

void FPointCloudManager::ClearData()
{
    PointCloudData.Empty();
    bPointCloudLoaded = false;
    bPointCloudHasColors = false;
    bPointCloudHasNormals = false;
    LoadedPointCloudPath.Empty();
    PointCloudCount = 0;
}

void FPointCloudManager::CreateColoredPointCloudVisualization(UWorld* World, bool bWithColors, bool bWithNormals)
{
    if (!World || PointCloudData.Num() == 0)
    {
        return;
    }

    ClearVisualization();

    for (TActorIterator<AActor> ActorIterator(World); ActorIterator; ++ActorIterator)
    {
        AActor* Actor = *ActorIterator;
        if (Actor && Actor->GetActorLabel() == TEXT("VCCSim_PointCloud"))
        {
            UE_LOG(LogPointCloudManager, Warning, TEXT("Found existing VCCSim_PointCloud actor, removing it: %s"), *Actor->GetName());
            Actor->Destroy();
        }
    }

    FPointCloudData PointCloudDataStruct;
    PointCloudDataStruct.Points = PointCloudData;

    AActor* NewActor = World->SpawnActor<AActor>();
    if (NewActor)
    {
        NewActor->SetActorLabel(TEXT("VCCSim_PointCloud"));
        PointCloudActor = NewActor;

        if (!NewActor->GetRootComponent())
        {
            USceneComponent* RootComp = NewObject<USceneComponent>(NewActor, TEXT("PointCloudRoot"));
            NewActor->SetRootComponent(RootComp);
            RootComp->RegisterComponent();
        }

        UPointCloudRenderer* PointCloudRenderer = NewObject<UPointCloudRenderer>(NewActor, TEXT("PointCloudRenderer"));
        if (PointCloudRenderer)
        {
            PointCloudRenderer->SetupAttachment(NewActor->GetRootComponent());
            PointCloudRenderer->RegisterComponent();
            PointCloudRenderer->RenderPointCloud(PointCloudDataStruct, bWithColors, 1.0f);
            ParticlePointCloudRenderer = PointCloudRenderer;

            UE_LOG(LogPointCloudManager, Log, TEXT("Created point cloud visualization: %d points"), PointCloudData.Num());

            if (bWithNormals && bPointCloudHasNormals)
            {
                CreateNormalVisualization(NewActor);
            }
        }

        bPointCloudVisualized = true;
    }
}

void FPointCloudManager::CreateNormalVisualization(AActor* Owner)
{
    if (!Owner || PointCloudData.Num() == 0)
    {
        return;
    }

    UInstancedStaticMeshComponent* NormISM = NormalLinesInstancedComponent.Get();
    if (!NormISM)
    {
        NormISM = NewObject<UInstancedStaticMeshComponent>(Owner, TEXT("PointCloudNormalInstanced"));
        if (!NormISM)
        {
            return;
        }
        NormISM->SetupAttachment(Owner->GetRootComponent());
        NormISM->SetMobility(EComponentMobility::Movable);
        NormISM->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        NormISM->SetCastShadow(false);
        NormISM->SetReceivesDecals(false);
        NormISM->bCastDynamicShadow = false;
        NormISM->bCastStaticShadow = false;

        if (UStaticMesh* Cylinder = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cylinder.Cylinder")))
        {
            NormISM->SetStaticMesh(Cylinder);
        }

        NormISM->RegisterComponent();
        NormalLinesInstancedComponent = NormISM;

        UMaterialInterface* BaseMaterial = LoadObject<UMaterialInterface>(
            nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
        if (BaseMaterial)
        {
            UMaterialInstanceDynamic* MID = UMaterialInstanceDynamic::Create(BaseMaterial, NormISM);
            if (MID)
            {
                MID->SetVectorParameterValue(TEXT("Color"), FLinearColor(0.2f, 0.6f, 1.0f, 1.0f));
                NormISM->SetMaterial(0, MID);
            }
        }
    }
    else
    {
        NormISM->ClearInstances();
    }

    const float NormalLength = 25.0f;
    const float NormalRadiusScale = 0.01f;
    for (const FRatPoint& P : PointCloudData)
    {
        if (!P.bHasNormal || P.Normal.IsNearlyZero())
        {
            continue;
        }
        const FVector Dir = P.Normal.GetSafeNormal() * NormalLength;
        const FVector Mid = P.Position + 0.5f * Dir;
        const FQuat Rot = FRotationMatrix::MakeFromZ(Dir).ToQuat();
        FTransform Xform;
        Xform.SetLocation(Mid);
        Xform.SetRotation(Rot);
        Xform.SetScale3D(FVector(NormalRadiusScale, NormalRadiusScale, FMath::Max(0.01f, NormalLength / 100.0f)));
        NormISM->AddInstance(Xform);
    }

    NormISM->SetVisibility(true);
}

void FPointCloudManager::SetupPointCloudMaterial(UInstancedStaticMeshComponent* MeshComponent)
{
    if (!MeshComponent)
    {
        return;
    }

    UMaterialInterface* Material = LoadPointCloudMaterial();
    if (Material)
    {
        MeshComponent->SetMaterial(0, Material);
        UE_LOG(LogPointCloudManager, Log, TEXT("Applied point cloud material"));
    }
    else
    {
        CreateSimplePointCloudMaterial(MeshComponent);
    }
}

UMaterialInterface* FPointCloudManager::LoadPointCloudMaterial()
{
    UMaterialInterface* CustomMaterial = LoadObject<UMaterialInterface>(nullptr, TEXT("/VCCSim/Materials/M_Point_Color"));
    if (CustomMaterial)
    {
        return CustomMaterial;
    }
    return LoadObject<UMaterialInterface>(nullptr, TEXT("/VCCSim/Materials/M_PointCloud"));
}

void FPointCloudManager::CreateSimplePointCloudMaterial(UInstancedStaticMeshComponent* MeshComponent)
{
    UMaterialInterface* BaseMaterial = LoadObject<UMaterialInterface>(
        nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial"));

    if (BaseMaterial && MeshComponent)
    {
        UMaterialInstanceDynamic* DynamicMaterial = UMaterialInstanceDynamic::Create(BaseMaterial, MeshComponent);
        if (DynamicMaterial)
        {
            DynamicMaterial->SetVectorParameterValue(TEXT("Color"), FLinearColor(1.0f, 0.5f, 0.0f, 1.0f));
            MeshComponent->SetMaterial(0, DynamicMaterial);
            UE_LOG(LogPointCloudManager, Log, TEXT("Created simple point cloud material"));
        }
    }
}

void FPointCloudManager::CreateBasicPointCloudMaterial(UProceduralMeshComponent* MeshComponent)
{
    if (!MeshComponent)
    {
        return;
    }

    UMaterialInterface* BaseMaterial = LoadObject<UMaterialInterface>(
        nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial"));

    if (BaseMaterial)
    {
        UMaterialInstanceDynamic* DynamicMaterial = UMaterialInstanceDynamic::Create(BaseMaterial, MeshComponent);
        if (DynamicMaterial)
        {
            DynamicMaterial->SetVectorParameterValue(TEXT("Color"), FLinearColor(1.0f, 0.5f, 0.0f, 1.0f));
            MeshComponent->SetMaterial(0, DynamicMaterial);
            UE_LOG(LogPointCloudManager, Log, TEXT("Applied basic point cloud material to procedural mesh"));
        }
    }
}
