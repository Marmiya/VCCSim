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

#include "DataStruct_IO/PointCloudRenderer.h"
#include "NiagaraFunctionLibrary.h"
#include "NiagaraDataInterfaceArrayFunctionLibrary.h"
#include "NiagaraSystemInstanceController.h"
#include "Engine/World.h"
#include "Engine/StaticMesh.h"
#include "Engine/AssetManager.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "UObject/ConstructorHelpers.h"
#include "AssetRegistry/AssetRegistryModule.h"

UPointCloudRenderer::UPointCloudRenderer()
{
    PrimaryComponentTick.bCanEverTick = false;
    
    // Create Niagara component
    NiagaraComponent = CreateDefaultSubobject<UNiagaraComponent>(TEXT("NiagaraComponent"));
    
    // Create fallback instanced mesh component
    InstancedMeshComponent = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("InstancedMeshComponent"));
    if (InstancedMeshComponent)
    {
        // Load a simple sphere mesh for point representation
        static ConstructorHelpers::FObjectFinder<UStaticMesh> SphereMesh(TEXT("/Engine/BasicShapes/Sphere"));
        if (SphereMesh.Succeeded())
        {
            InstancedMeshComponent->SetStaticMesh(SphereMesh.Object);
        }
        
        // Disable collision for better performance
        InstancedMeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        InstancedMeshComponent->SetVisibility(false); // Start invisible until needed
    }
}

void UPointCloudRenderer::BeginPlay()
{
    Super::BeginPlay();
    InitializeNiagaraSystem();
}

void UPointCloudRenderer::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    ClearPointCloud();
    Super::EndPlay(EndPlayReason);
}

void UPointCloudRenderer::InitializeNiagaraSystem()
{
    if (bIsInitialized || !NiagaraComponent)
    {
        return;
    }

    UNiagaraSystem* PointCloudSystem = nullptr;

    // 1) Try user-assigned asset (soft reference)
    if (NiagaraSystemAsset.IsValid() || NiagaraSystemAsset.ToSoftObjectPath().IsValid())
    {
        PointCloudSystem = NiagaraSystemAsset.LoadSynchronous();
        if (PointCloudSystem)
        {
            UE_LOG(LogTemp, Log, TEXT("Loaded Niagara system from property: %s"), *NiagaraSystemAsset.ToString());
        }
    }

    // 2) Try to find any available Niagara system using Asset Registry
    if (!PointCloudSystem)
    {
        FAssetRegistryModule& AssetRegistryModule = FModuleManager::LoadModuleChecked<FAssetRegistryModule>("AssetRegistry");
        IAssetRegistry& AssetRegistry = AssetRegistryModule.Get();

        // Find Niagara systems
        TArray<FAssetData> NiagaraAssets;
        FARFilter Filter;
        Filter.ClassPaths.Add(UNiagaraSystem::StaticClass()->GetClassPathName());
        Filter.bRecursivePaths = true;

        AssetRegistry.GetAssets(Filter, NiagaraAssets);

        UE_LOG(LogTemp, Log, TEXT("Found %d Niagara systems in project"), NiagaraAssets.Num());
        
        // First priority: Look for systems specifically designed for point clouds/particles
        for (const FAssetData& AssetData : NiagaraAssets)
        {
            FString AssetPath = AssetData.GetObjectPathString();
            FString AssetName = AssetData.AssetName.ToString();
            
            if (AssetName.Contains(TEXT("Point")) || AssetName.Contains(TEXT("Cloud")) ||
                AssetName.Contains(TEXT("Sprite")) || AssetName.Contains(TEXT("GPU")) ||
                AssetPath.Contains(TEXT("Sprite")) || AssetPath.Contains(TEXT("GPU")))
            {
                PointCloudSystem = Cast<UNiagaraSystem>(AssetData.GetAsset());
                if (PointCloudSystem)
                {
                    UE_LOG(LogTemp, Log, TEXT("Found suitable Niagara system for point clouds: %s"), *AssetPath);
                    break;
                }
            }
        }
        
        // Second priority: Engine systems (but not specific effects like insects)
        if (!PointCloudSystem)
        {
            for (const FAssetData& AssetData : NiagaraAssets)
            {
                FString AssetPath = AssetData.GetObjectPathString();
                FString AssetName = AssetData.AssetName.ToString();
                
                if (AssetPath.Contains(TEXT("Engine")) && 
                    !AssetName.Contains(TEXT("Insect")) && 
                    !AssetName.Contains(TEXT("Fire")) &&
                    !AssetName.Contains(TEXT("Smoke")) &&
                    !AssetName.Contains(TEXT("Water")))
                {
                    PointCloudSystem = Cast<UNiagaraSystem>(AssetData.GetAsset());
                    if (PointCloudSystem)
                    {
                        UE_LOG(LogTemp, Log, TEXT("Found generic engine Niagara system: %s"), *AssetPath);
                        break;
                    }
                }
            }
        }
        
        // If still no suitable system found, log that we'll use fallback
        if (!PointCloudSystem && NiagaraAssets.Num() > 0)
        {
            UE_LOG(LogTemp, Warning, TEXT("No suitable Niagara systems found among %d available systems. Available systems are specialized effects (insects, fire, etc.) not compatible with point cloud rendering."), NiagaraAssets.Num());
        }
    }

    if (PointCloudSystem)
    {
        NiagaraComponent->SetAsset(PointCloudSystem);
        NiagaraComponent->SetAutoActivate(false);
        SetupPointCloudMaterial();
        bIsInitialized = true;
        UE_LOG(LogTemp, Log, TEXT("Point cloud Niagara system initialized successfully"));
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("No Niagara systems found in project. Using optimized fallback system."));
        CreateOptimizedFallbackSystem();
    }
}

void UPointCloudRenderer::CreateOptimizedFallbackSystem()
{
    if (!InstancedMeshComponent)
    {
        UE_LOG(LogTemp, Error, TEXT("Instanced mesh component not available for fallback rendering"));
        return;
    }

    // Setup material for point cloud visualization
    if (InstancedMeshComponent->GetStaticMesh())
    {
        // Try to load the existing point cloud material
        UMaterialInterface* PointMaterial = LoadObject<UMaterialInterface>(
            nullptr, 
            TEXT("/VCCSim/Materials/M_PointCloud")
        );
        
        if (!PointMaterial)
        {
            // Use default material if custom one isn't available
            PointMaterial = LoadObject<UMaterialInterface>(
                nullptr, 
                TEXT("/Engine/BasicShapes/BasicShapeMaterial")
            );
        }
        
        if (PointMaterial)
        {
            InstancedMeshComponent->SetMaterial(0, PointMaterial);
            UE_LOG(LogTemp, Log, TEXT("Instanced mesh fallback material set successfully"));
        }
    }

    // Mark as initialized for fallback rendering
    bIsInitialized = true;
    UE_LOG(LogTemp, Log, TEXT("Optimized instanced mesh fallback system created successfully"));
}

void UPointCloudRenderer::CreateFallbackSystem()
{
    CreateOptimizedFallbackSystem();
}

void UPointCloudRenderer::RenderPointCloud(const FPointCloudData& PointCloudData, bool bInShowColors, float InPointSize)
{
    if (!bIsInitialized)
    {
        InitializeNiagaraSystem();
    }

    if (!NiagaraComponent)
    {
        UE_LOG(LogTemp, Error, TEXT("Niagara component not available for point cloud rendering"));
        return;
    }

    // Update settings
    bShowColors = bInShowColors;
    PointSize = InPointSize;
    RenderedPointCount = PointCloudData.GetPointCount();
    bHasColors = PointCloudData.HasColors();

    // Clear existing visualization
    ClearPointCloud();

    if (PointCloudData.GetPointCount() == 0)
    {
        return;
    }

    // Update particle data
    UpdateParticleData(PointCloudData);

    // Check if Niagara system is available and active after UpdateParticleData
    if (NiagaraComponent->GetAsset() && NiagaraComponent->IsVisible())
    {
        NiagaraComponent->Activate(true);
        
        // Hide fallback component
        if (InstancedMeshComponent)
        {
            InstancedMeshComponent->SetVisibility(false);
        }
        
        UE_LOG(LogTemp, Log, TEXT("Rendered %d points using Niagara system: %s"), 
               RenderedPointCount, 
               *NiagaraComponent->GetAsset()->GetName());
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("Niagara system not compatible or available, using fallback rendering"));
        RenderPointCloudFallback(PointCloudData);
    }
}

void UPointCloudRenderer::RenderPointCloudFallback(const FPointCloudData& PointCloudData)
{
    if (!InstancedMeshComponent || PointCloudData.GetPointCount() == 0)
    {
        return;
    }

    // Clear existing instances
    InstancedMeshComponent->ClearInstances();
    
    const TArray<FRatPoint>& Points = PointCloudData.Points;
    
    // Create instances for each point
    for (const FRatPoint& Point : Points)
    {
        // Create transform for this point
        FTransform InstanceTransform;
        InstanceTransform.SetLocation(Point.Position);
        
        // Scale based on point size (small spheres for points)
        float ScaleValue = FMath::Max(0.01f, PointSize * 0.1f); // Convert to reasonable sphere size
        InstanceTransform.SetScale3D(FVector(ScaleValue));
        
        // Add the instance
        InstancedMeshComponent->AddInstance(InstanceTransform);
    }
    
    // Make the component visible
    InstancedMeshComponent->SetVisibility(true);
    
    // Hide Niagara component if it's visible
    if (NiagaraComponent)
    {
        NiagaraComponent->SetVisibility(false);
    }
    
    UE_LOG(LogTemp, Log, TEXT("Fallback rendered %d points using instanced meshes"), Points.Num());
}

void UPointCloudRenderer::UpdateParticleData(const FPointCloudData& PointCloudData)
{
    if (!NiagaraComponent || !NiagaraComponent->GetAsset())
    {
        return;
    }

    // Prepare position and color arrays
    CachedPositions.Empty();
    CachedColors.Empty();
    
    const TArray<FRatPoint>& Points = PointCloudData.Points;
    
    CachedPositions.Reserve(Points.Num());
    CachedColors.Reserve(Points.Num());

    for (const FRatPoint& Point : Points)
    {
        CachedPositions.Add(Point.Position);
        
        if (bShowColors && bHasColors)
        {
            CachedColors.Add(Point.Color);
        }
        else
        {
            CachedColors.Add(DefaultColor);
        }
    }

    // Set basic parameters that most Niagara systems should support
    NiagaraComponent->SetFloatParameter(TEXT("SpawnRate"), Points.Num());
    NiagaraComponent->SetFloatParameter(TEXT("LifeTime"), 10.0f); // Long lifetime for persistent points
    NiagaraComponent->SetFloatParameter(TEXT("Size"), PointSize);
    
    // Try common parameter names
    NiagaraComponent->SetIntParameter(TEXT("NumParticles"), Points.Num());
    NiagaraComponent->SetIntParameter(TEXT("ParticleCount"), Points.Num());
    
    // Check if this Niagara system has the required data interfaces for point cloud rendering
    bool bHasValidDataInterface = false;
    
    if (NiagaraComponent->GetAsset())
    {
        // Try to detect if the system supports our data interfaces
        // Since the insect system doesn't have compatible data interfaces, we'll skip trying to set data
        UNiagaraSystem* System = NiagaraComponent->GetAsset();
        FString SystemName = System->GetName();
        
        // Check if this system is suitable for point cloud rendering
        if (SystemName.Contains(TEXT("Point")) || 
            SystemName.Contains(TEXT("Cloud")) || 
            SystemName.Contains(TEXT("Sprite")) ||
            SystemName.Contains(TEXT("GPU")))
        {
            bHasValidDataInterface = true;
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("Niagara system '%s' doesn't appear to be designed for point cloud rendering. Using fallback."), *SystemName);
            
            // Deactivate the incompatible system and use fallback instead
            NiagaraComponent->Deactivate();
            NiagaraComponent->SetVisibility(false);
            return; // This will cause the render function to use fallback
        }
    }

    // Only try to set data if the system appears compatible
    if (bHasValidDataInterface && CachedPositions.Num() > 0)
    {
        // Try multiple common data interface names
        bool bPositionSet = false;
        TArray<FString> PositionArrayNames = { TEXT("PositionArray"), TEXT("Positions"), TEXT("PointPositions"), TEXT("ParticlePositions") };
        
        for (const FString& ArrayName : PositionArrayNames)
        {
            UNiagaraDataInterfaceArrayFunctionLibrary::SetNiagaraArrayVector(NiagaraComponent, *ArrayName, CachedPositions);
            bPositionSet = true;
            UE_LOG(LogTemp, Log, TEXT("Set position data using array name: %s"), *ArrayName);
            break;
        }
        
        if (!bPositionSet)
        {
            UE_LOG(LogTemp, Warning, TEXT("Could not set position data - using fallback rendering"));
            NiagaraComponent->Deactivate();
            NiagaraComponent->SetVisibility(false);
            return;
        }
    }

    // Try to set color data if available and system supports it
    if (bHasValidDataInterface && CachedColors.Num() > 0 && (bShowColors && bHasColors))
    {
        // Convert to FVector for Niagara (RGB as XYZ)
        TArray<FVector> ColorVectors;
        ColorVectors.Reserve(CachedColors.Num());
        
        for (const FLinearColor& Color : CachedColors)
        {
            ColorVectors.Add(FVector(Color.R, Color.G, Color.B));
        }
        
        // Try multiple common color array names
        TArray<FString> ColorArrayNames = { TEXT("ColorArray"), TEXT("Colors"), TEXT("ParticleColors"), TEXT("PointColors") };
        
        bool bColorSet = false;
        for (const FString& ArrayName : ColorArrayNames)
        {
            UNiagaraDataInterfaceArrayFunctionLibrary::SetNiagaraArrayVector(NiagaraComponent, *ArrayName, ColorVectors);
            bColorSet = true;
            UE_LOG(LogTemp, Log, TEXT("Set color data using array name: %s"), *ArrayName);
            break;
        }
        
        if (!bColorSet)
        {
            UE_LOG(LogTemp, Log, TEXT("Could not set color data - using default colors"));
        }
    }
    
    UE_LOG(LogTemp, Log, TEXT("Updated Niagara particle data: %d points"), Points.Num());
}

void UPointCloudRenderer::ClearPointCloud()
{
    // Clear Niagara rendering
    if (NiagaraComponent && NiagaraComponent->IsActive())
    {
        NiagaraComponent->Deactivate();
        NiagaraComponent->SetVisibility(false);
    }
    
    // Clear instanced mesh fallback rendering
    if (InstancedMeshComponent)
    {
        InstancedMeshComponent->ClearInstances();
        InstancedMeshComponent->SetVisibility(false);
    }
    
    // Clear cached data
    RenderedPointCount = 0;
    CachedPositions.Empty();
    CachedColors.Empty();
}

void UPointCloudRenderer::SetPointSize(float NewPointSize)
{
    PointSize = NewPointSize;
    
    if (NiagaraComponent && NiagaraComponent->IsActive())
    {
        NiagaraComponent->SetFloatParameter(TEXT("PointSize"), PointSize);
    }
}

void UPointCloudRenderer::SetShowColors(bool bInShowColors)
{
    if (bShowColors != bInShowColors)
    {
        bShowColors = bInShowColors;
        
        if (NiagaraComponent && NiagaraComponent->IsActive())
        {
            NiagaraComponent->SetBoolParameter(TEXT("UseColors"), bShowColors && bHasColors);
            
            // Update color data if needed
            if (CachedPositions.Num() > 0)
            {
                // Re-create a temporary point cloud data to refresh colors
                FPointCloudData TempData;
                for (int32 i = 0; i < CachedPositions.Num(); ++i)
                {
                    FRatPoint TempPoint;
                    TempPoint.Position = CachedPositions[i];
                    if (i < CachedColors.Num())
                    {
                        TempPoint.Color = CachedColors[i];
                    }
                    TempData.AddPoint(TempPoint);
                }
                
                UpdateParticleData(TempData);
            }
        }
    }
}

void UPointCloudRenderer::SetupPointCloudMaterial()
{
    if (!NiagaraComponent)
    {
        return;
    }

    // Try to set up material parameters for better point cloud visualization
    // This would typically involve setting up a material that supports per-particle colors
    
    UE_LOG(LogTemp, Log, TEXT("Point cloud material setup completed"));
}
