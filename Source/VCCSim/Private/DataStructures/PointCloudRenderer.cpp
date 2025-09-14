DEFINE_LOG_CATEGORY_STATIC(LogPointCloudRenderer, Log, All);

#include "DataStructures/PointCloudRenderer.h"
#include "Engine/World.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "UObject/ConstructorHelpers.h"

UPointCloudRenderer::UPointCloudRenderer()
{
    PrimaryComponentTick.bCanEverTick = false;

    InstancedMeshComponent = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("InstancedMeshComponent"));
    if (InstancedMeshComponent)
    {
        InstancedMeshComponent->SetupAttachment(this);
        InstancedMeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        InstancedMeshComponent->SetVisibility(false);
        InstancedMeshComponent->SetMobility(EComponentMobility::Movable);
        InstancedMeshComponent->SetCastShadow(false);
        InstancedMeshComponent->SetReceivesDecals(false);
        InstancedMeshComponent->bCastDynamicShadow = false;
        InstancedMeshComponent->bCastStaticShadow = false;
    }
}

void UPointCloudRenderer::BeginPlay()
{
    Super::BeginPlay();
    CreateInstancedMeshSystem();
}

void UPointCloudRenderer::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    ClearPointCloud();
    Super::EndPlay(EndPlayReason);
}


void UPointCloudRenderer::OnRegister()
{
    Super::OnRegister();

    if (InstancedMeshComponent)
    {
        if (!InstancedMeshComponent->GetAttachParent())
        {
            InstancedMeshComponent->SetupAttachment(this);
        }
        if (!InstancedMeshComponent->IsRegistered())
        {
            InstancedMeshComponent->RegisterComponent();
        }
        InstancedMeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        InstancedMeshComponent->SetVisibility(false);
        InstancedMeshComponent->SetMobility(EComponentMobility::Movable);
        InstancedMeshComponent->SetCastShadow(false);
        InstancedMeshComponent->SetReceivesDecals(false);
        InstancedMeshComponent->bCastDynamicShadow = false;
        InstancedMeshComponent->bCastStaticShadow = false;
    }
}

void UPointCloudRenderer::CreateInstancedMeshSystem()
{
    if (!InstancedMeshComponent)
    {
        UE_LOG(LogPointCloudRenderer, Error, TEXT("Instanced mesh component not available"));
        return;
    }

    if (!InstancedMeshComponent->GetStaticMesh())
    {
        UStaticMesh* SphereMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Sphere.Sphere"));
        if (SphereMesh)
        {
            InstancedMeshComponent->SetStaticMesh(SphereMesh);
        }
        else
        {
            UE_LOG(LogPointCloudRenderer, Error, TEXT("Failed to load sphere mesh"));
        }
    }

    if (InstancedMeshComponent->GetStaticMesh())
    {
        UMaterialInterface* PointMaterial = nullptr;
        
        PointMaterial = LoadObject<UMaterialInterface>(
            nullptr, 
            TEXT("/VCCSim/Materials/M_PointCloud")
        );
        
        if (!PointMaterial)
        {
            UMaterialInterface* BaseMaterial = LoadObject<UMaterialInterface>(
                nullptr, 
                TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial")
            );
            
            if (BaseMaterial)
            {
                UMaterialInstanceDynamic* MID = UMaterialInstanceDynamic::Create(BaseMaterial, this);
                if (MID)
                {
                    MID->SetVectorParameterValue(TEXT("Color"), FLinearColor(1.0f, 1.0f, 1.0f, 1.0f));
                    PointMaterial = MID;
                }
                else
                {
                    PointMaterial = BaseMaterial;
                }
            }
        }
        
        if (PointMaterial)
        {
            InstancedMeshComponent->SetMaterial(0, PointMaterial);
        }
    }

    bIsInitialized = true;
}


void UPointCloudRenderer::RenderPointCloud(const FPointCloudData& PointCloudData, bool bInShowColors, float InPointSize)
{
    bShowColors = bInShowColors;
    PointSize = InPointSize;
    bHasColors = PointCloudData.HasColors();

    ClearPointCloud();

    if (PointCloudData.GetPointCount() == 0)
    {
        return;
    }

    RenderedPointCount = PointCloudData.GetPointCount();

    if (bShowColors && bHasColors)
    {
        RenderPointCloudInstancedWithColors(PointCloudData);
    }
    else
    {
        RenderPointCloudInstanced(PointCloudData);
    }
}

void UPointCloudRenderer::RenderPointCloudInstanced(const FPointCloudData& PointCloudData)
{
    if (!InstancedMeshComponent || PointCloudData.GetPointCount() == 0)
    {
        return;
    }

    if (!InstancedMeshComponent->GetStaticMesh() || !InstancedMeshComponent->GetMaterial(0))
    {
        CreateInstancedMeshSystem();
    }

    InstancedMeshComponent->ClearInstances();
    ClearColorInstancedComponents();
    
    const TArray<FRatPoint>& Points = PointCloudData.Points;
    
    for (const FRatPoint& Point : Points)
    {
        FTransform InstanceTransform;
        InstanceTransform.SetLocation(Point.Position);
        float ScaleValue = FMath::Max(0.01f, PointSize * 0.1f);
        InstanceTransform.SetScale3D(FVector(ScaleValue));
        InstancedMeshComponent->AddInstance(InstanceTransform);
    }
    
    InstancedMeshComponent->SetVisibility(true);
    
    UE_LOG(LogPointCloudRenderer, Log, TEXT("Rendered %d points using instanced meshes"), Points.Num());
}

void UPointCloudRenderer::RenderPointCloudInstancedWithColors(const FPointCloudData& PointCloudData)
{
    if (PointCloudData.GetPointCount() == 0)
    {
        return;
    }

    if (InstancedMeshComponent)
    {
        InstancedMeshComponent->ClearInstances();
        InstancedMeshComponent->SetVisibility(false);
    }

    ClearColorInstancedComponents();

    const TArray<FRatPoint>& Points = PointCloudData.Points;
    
    const int32 MaxUniqueColors = 256;
    TMap<uint32, TArray<int32>> ColorToIndices;
    ColorToIndices.Reserve(FMath::Min(MaxUniqueColors, Points.Num()));

    for (int32 i = 0; i < Points.Num(); ++i)
    {
        const uint32 Key = PackColorKey(Points[i].Color);
        auto& Indices = ColorToIndices.FindOrAdd(Key);
        Indices.Add(i);
    }

    for (const TPair<uint32, TArray<int32>>& Pair : ColorToIndices)
    {
        const uint32 Key = Pair.Key;
        const TArray<int32>& Indices = Pair.Value;
        const FLinearColor Color(
            ((Key >> 16) & 0xFF) / 255.0f,
            ((Key >> 8) & 0xFF) / 255.0f,
            (Key & 0xFF) / 255.0f,
            1.0f
        );

        UInstancedStaticMeshComponent* ISM = GetOrCreateColorISM(Color);
        if (!ISM)
        {
            continue;
        }

        for (int32 Index : Indices)
        {
            const FRatPoint& Point = Points[Index];
            FTransform Xform;
            Xform.SetLocation(Point.Position);
            float ScaleValue = FMath::Max(0.01f, PointSize * 0.1f);
            Xform.SetScale3D(FVector(ScaleValue));
            ISM->AddInstance(Xform);
        }

        ISM->SetVisibility(true);
    }

    UE_LOG(LogPointCloudRenderer, Log, TEXT("Rendered %d points using color-grouped instanced meshes (%d unique colors)"), Points.Num(), ColorToIndices.Num());
}


void UPointCloudRenderer::ClearPointCloud()
{
    if (InstancedMeshComponent)
    {
        InstancedMeshComponent->ClearInstances();
        InstancedMeshComponent->SetVisibility(false);
    }
    ClearColorInstancedComponents();
    
    RenderedPointCount = 0;
    CachedPositions.Empty();
    CachedColors.Empty();
}

void UPointCloudRenderer::ClearColorInstancedComponents()
{
    if (ColorInstancedComponents.Num() == 0)
    {
        return;
    }
    for (TPair<uint32, TObjectPtr<UInstancedStaticMeshComponent>>& Pair : ColorInstancedComponents)
    {
        if (Pair.Value)
        {
            Pair.Value->ClearInstances();
            Pair.Value->SetVisibility(false);
            if (Pair.Value->IsRegistered())
            {
                Pair.Value->UnregisterComponent();
            }
            Pair.Value->DestroyComponent();
        }
    }
    ColorInstancedComponents.Empty();
}

UInstancedStaticMeshComponent* UPointCloudRenderer::GetOrCreateColorISM(const FLinearColor& Color)
{
    const uint32 Key = PackColorKey(Color);
    if (TObjectPtr<UInstancedStaticMeshComponent>* Found = ColorInstancedComponents.Find(Key))
    {
        return Found->Get();
    }

    AActor* Owner = GetOwner();
    if (!Owner)
    {
        return nullptr;
    }

    UInstancedStaticMeshComponent* NewISM = NewObject<UInstancedStaticMeshComponent>(Owner);
    if (!NewISM)
    {
        return nullptr;
    }
    NewISM->SetupAttachment(this);
    NewISM->SetMobility(EComponentMobility::Movable);
    NewISM->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    
    // Disable shadows for point cloud visualization
    NewISM->SetCastShadow(false);
    NewISM->SetReceivesDecals(false);
    NewISM->bCastDynamicShadow = false;
    NewISM->bCastStaticShadow = false;

    // Use Engine basic sphere - load at runtime
    UStaticMesh* SphereMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Sphere.Sphere"));
    if (SphereMesh)
    {
        NewISM->SetStaticMesh(SphereMesh);
    }

    // Create a dynamic material with the specified color
    UMaterialInterface* BaseMaterial = LoadObject<UMaterialInterface>(nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
    if (BaseMaterial)
    {
        UMaterialInstanceDynamic* MID = UMaterialInstanceDynamic::Create(BaseMaterial, NewISM);
        if (MID)
        {
            MID->SetVectorParameterValue(TEXT("Color"), Color);
            NewISM->SetMaterial(0, MID);
        }
    }

    NewISM->RegisterComponent();
    ColorInstancedComponents.Add(Key, NewISM);
    return NewISM;
}

void UPointCloudRenderer::SetPointSize(float NewPointSize)
{
    PointSize = NewPointSize;
}

void UPointCloudRenderer::SetShowColors(bool bInShowColors)
{
    bShowColors = bInShowColors;
}

void UPointCloudRenderer::SetupPointCloudMaterial()
{
}
