#include "Utils/PointCloudRenderComponent.h"
#include "Engine/StaticMesh.h"
#include "UObject/ConstructorHelpers.h"
#include "Materials/MaterialInterface.h"

UPointCloudRenderComponent::UPointCloudRenderComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
    
    // Create instanced mesh components for visible and invisible points
    VisiblePointsComponent = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("VisiblePointsComponent"));
    InvisiblePointsComponent = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("InvisiblePointsComponent"));
    
    SetupComponents();
}

void UPointCloudRenderComponent::BeginPlay()
{
    Super::BeginPlay();
    SetupComponents();
}

void UPointCloudRenderComponent::SetupComponents()
{
    if (!VisiblePointsComponent || !InvisiblePointsComponent)
        return;
        
    // Load a simple sphere mesh for point representation
    static ConstructorHelpers::FObjectFinder<UStaticMesh> SphereMesh(TEXT("/Engine/BasicShapes/Sphere"));
    if (SphereMesh.Succeeded())
    {
        VisiblePointsComponent->SetStaticMesh(SphereMesh.Object);
        InvisiblePointsComponent->SetStaticMesh(SphereMesh.Object);
    }
    
    // Load materials for visible and invisible points
    UMaterialInterface* VisibleMaterial = LoadObject<UMaterialInterface>(nullptr, TEXT("/VCCSim/Materials/M_Point_Color"));
    UMaterialInterface* InvisibleMaterial = LoadObject<UMaterialInterface>(nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial"));
    
    // Fallback to basic materials if custom ones aren't found
    if (!VisibleMaterial)
    {
        VisibleMaterial = LoadObject<UMaterialInterface>(nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial"));
    }
    
    if (VisibleMaterial)
    {
        VisiblePointsComponent->SetMaterial(0, VisibleMaterial);
    }
    
    if (InvisibleMaterial)
    {
        InvisiblePointsComponent->SetMaterial(0, InvisibleMaterial);
    }
    
    // Disable collision for better performance
    VisiblePointsComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    InvisiblePointsComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    
    // Set initial visibility
    VisiblePointsComponent->SetVisibility(true);
    InvisiblePointsComponent->SetVisibility(true);
}

void UPointCloudRenderComponent::SetVisiblePoints(const TArray<FVector>& Points)
{
    if (!VisiblePointsComponent)
        return;
        
    // Clear existing instances
    VisiblePointsComponent->ClearInstances();
    
    // Create instances for each point
    for (const FVector& Point : Points)
    {
        FTransform InstanceTransform;
        InstanceTransform.SetLocation(Point);
        
        // Scale based on point size (small spheres for points)
        float ScaleValue = FMath::Max(0.01f, VisiblePointSize * 0.1f);
        InstanceTransform.SetScale3D(FVector(ScaleValue));
        
        VisiblePointsComponent->AddInstance(InstanceTransform);
    }
    
    UE_LOG(LogTemp, Log, TEXT("PointCloudRenderComponent: Set %d visible points"), Points.Num());
}

void UPointCloudRenderComponent::SetInvisiblePoints(const TArray<FVector>& Points)
{
    if (!InvisiblePointsComponent)
        return;
        
    // Clear existing instances
    InvisiblePointsComponent->ClearInstances();
    
    // Create instances for each point
    for (const FVector& Point : Points)
    {
        FTransform InstanceTransform;
        InstanceTransform.SetLocation(Point);
        
        // Scale based on point size (small spheres for points)
        float ScaleValue = FMath::Max(0.01f, InvisiblePointSize * 0.1f);
        InstanceTransform.SetScale3D(FVector(ScaleValue));
        
        InvisiblePointsComponent->AddInstance(InstanceTransform);
    }
    
    UE_LOG(LogTemp, Log, TEXT("PointCloudRenderComponent: Set %d invisible points"), Points.Num());
}

void UPointCloudRenderComponent::ClearPoints()
{
    if (VisiblePointsComponent)
    {
        VisiblePointsComponent->ClearInstances();
    }
    
    if (InvisiblePointsComponent)
    {
        InvisiblePointsComponent->ClearInstances();
    }
    
    UE_LOG(LogTemp, Log, TEXT("PointCloudRenderComponent: Cleared all points"));
}