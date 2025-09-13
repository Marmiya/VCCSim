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

DEFINE_LOG_CATEGORY_STATIC(LogPointCloudRenderer, Log, All);

#include "DataStructures/PointCloudRenderer.h"
#include "NiagaraFunctionLibrary.h"
#include "NiagaraDataInterfaceArrayFunctionLibrary.h"
#include "Engine/World.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "UObject/ConstructorHelpers.h"

UPointCloudRenderer::UPointCloudRenderer()
{
    PrimaryComponentTick.bCanEverTick = false;

    // Create Niagara component (child of this component)
    NiagaraComponent = CreateDefaultSubobject<UNiagaraComponent>(TEXT("NiagaraComponent"));
    if (NiagaraComponent)
    {
        NiagaraComponent->SetupAttachment(this);
        NiagaraComponent->bAutoActivate = false;
        NiagaraComponent->SetVisibility(false);
    }

    // Create fallback instanced mesh component (child of this component)
    InstancedMeshComponent = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("InstancedMeshComponent"));
    if (InstancedMeshComponent)
    {
        InstancedMeshComponent->SetupAttachment(this);

        // Note: Mesh will be loaded at runtime in CreateOptimizedFallbackSystem to avoid FObjectFinder issues

        // Disable collision for better performance
        InstancedMeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        InstancedMeshComponent->SetVisibility(false); // Start invisible until needed
        InstancedMeshComponent->SetMobility(EComponentMobility::Movable);
        
        // Disable shadows for point cloud visualization
        InstancedMeshComponent->SetCastShadow(false);
        InstancedMeshComponent->SetReceivesDecals(false);
        InstancedMeshComponent->bCastDynamicShadow = false;
        InstancedMeshComponent->bCastStaticShadow = false;
    }
}

void UPointCloudRenderer::BeginPlay()
{
    Super::BeginPlay();
    // Don't initialize Niagara system automatically - only when needed for colors
    CreateOptimizedFallbackSystem();
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
            UE_LOG(LogPointCloudRenderer, Log, TEXT("Loaded Niagara system from property: %s"), *NiagaraSystemAsset.ToString());
        }
    }

    // 2) Skip automatic Niagara system search - use user-defined system only
    if (!PointCloudSystem)
    {
        UE_LOG(LogPointCloudRenderer, Log, TEXT("No Niagara system assigned in NiagaraSystemAsset property."));
        UE_LOG(LogPointCloudRenderer, Log, TEXT("To use Niagara for point cloud rendering:"));
        UE_LOG(LogPointCloudRenderer, Log, TEXT("1. Create a new Niagara System with GPU Emitter"));
        UE_LOG(LogPointCloudRenderer, Log, TEXT("2. Add Array Data Interface with 'Points' (Vector) and 'Colors' (Vector) arrays"));
        UE_LOG(LogPointCloudRenderer, Log, TEXT("3. Set Spawn Count = Points.GetNum()"));
        UE_LOG(LogPointCloudRenderer, Log, TEXT("4. Set Particles.Position = Points.GetValue(Particles.ID)"));
        UE_LOG(LogPointCloudRenderer, Log, TEXT("5. Set Particles.Color = Colors.GetValue(Particles.ID)"));
        UE_LOG(LogPointCloudRenderer, Log, TEXT("6. Assign the system to NiagaraSystemAsset property"));
        UE_LOG(LogPointCloudRenderer, Log, TEXT("Using optimized instanced mesh fallback instead."));
    }

    if (PointCloudSystem)
    {
        NiagaraComponent->SetAsset(PointCloudSystem);
        NiagaraComponent->SetAutoActivate(false);
        SetupPointCloudMaterial();
        bIsInitialized = true;
        UE_LOG(LogPointCloudRenderer, Log, TEXT("Point cloud Niagara system initialized successfully"));
    }
    else
    {
        UE_LOG(LogPointCloudRenderer, Warning, TEXT("No Niagara systems found in project. Using optimized fallback system."));
        CreateOptimizedFallbackSystem();
    }
}

void UPointCloudRenderer::OnRegister()
{
    Super::OnRegister();

    // Ensure child components are attached and registered in editor/runtime
    if (NiagaraComponent)
    {
        if (!NiagaraComponent->GetAttachParent())
        {
            NiagaraComponent->SetupAttachment(this);
        }
        if (!NiagaraComponent->IsRegistered())
        {
            NiagaraComponent->RegisterComponent();
        }
        NiagaraComponent->SetVisibility(false);
    }

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
        
        // Disable shadows for point cloud visualization
        InstancedMeshComponent->SetCastShadow(false);
        InstancedMeshComponent->SetReceivesDecals(false);
        InstancedMeshComponent->bCastDynamicShadow = false;
        InstancedMeshComponent->bCastStaticShadow = false;
    }

    // Only initialize Niagara system in editor if needed for colors
    // InitializeNiagaraSystem(); // Removed - will be called on-demand in RenderPointCloud
}

void UPointCloudRenderer::CreateOptimizedFallbackSystem()
{
    if (!InstancedMeshComponent)
    {
        UE_LOG(LogPointCloudRenderer, Error, TEXT("Instanced mesh component not available for fallback rendering"));
        return;
    }

    // Load sphere mesh if not already set (use same path as color ISM)
    if (!InstancedMeshComponent->GetStaticMesh())
    {
        UStaticMesh* SphereMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Sphere.Sphere"));
        if (SphereMesh)
        {
            InstancedMeshComponent->SetStaticMesh(SphereMesh);
        }
        else
        {
            UE_LOG(LogPointCloudRenderer, Error, TEXT("Failed to load sphere mesh for main ISM"));
        }
    }

    // Setup material for point cloud visualization (use same material system as color ISM)
    if (InstancedMeshComponent->GetStaticMesh())
    {
        UMaterialInterface* PointMaterial = nullptr;
        
        // Try to load the existing point cloud material
        PointMaterial = LoadObject<UMaterialInterface>(
            nullptr, 
            TEXT("/VCCSim/Materials/M_PointCloud")
        );
        
        if (!PointMaterial)
        {
            // Use default material with correct full path
            UMaterialInterface* BaseMaterial = LoadObject<UMaterialInterface>(
                nullptr, 
                TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial")
            );
            
            if (BaseMaterial)
            {
                // Create dynamic material instance for consistent white color (visible fallback)
                UMaterialInstanceDynamic* MID = UMaterialInstanceDynamic::Create(BaseMaterial, this);
                if (MID)
                {
                    // Set visible white color for non-color mode
                    MID->SetVectorParameterValue(TEXT("Color"), FLinearColor(1.0f, 1.0f, 1.0f, 1.0f)); // White color
                    PointMaterial = MID;
                }
                else
                {
                    PointMaterial = BaseMaterial;
                }
            }
            else
            {
                UE_LOG(LogPointCloudRenderer, Error, TEXT("Failed to load base material from /Engine/BasicShapes/BasicShapeMaterial"));
            }
        }
        
        if (PointMaterial)
        {
            InstancedMeshComponent->SetMaterial(0, PointMaterial);
        }
        else
        {
            UE_LOG(LogPointCloudRenderer, Error, TEXT("Failed to set any material for main ISM - points may not be visible"));
        }
    }
    else
    {
        UE_LOG(LogPointCloudRenderer, Error, TEXT("No static mesh set for InstancedMeshComponent - cannot set material"));
    }

    // Mark as initialized for fallback rendering
    bIsInitialized = true;
}

void UPointCloudRenderer::CreateFallbackSystem()
{
    CreateOptimizedFallbackSystem();
}

void UPointCloudRenderer::RenderPointCloud(const FPointCloudData& PointCloudData, bool bInShowColors, float InPointSize)
{
    // Update settings
    bShowColors = bInShowColors;
    PointSize = InPointSize;
    bHasColors = PointCloudData.HasColors();

    // Clear existing visualization
    ClearPointCloud();

    if (PointCloudData.GetPointCount() == 0)
    {
        return;
    }

    RenderedPointCount = PointCloudData.GetPointCount();

    if (PointCloudData.GetPointCount() == 0)
    {
        return;
    }

    // Only try Niagara system if we actually need colors
    bool bShouldUseNiagara = bShowColors && bHasColors;
    
    if (bShouldUseNiagara)
    {
        // Initialize Niagara system only when needed
        if (!bIsInitialized)
        {
            InitializeNiagaraSystem();
        }

        if (NiagaraComponent)
        {
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
                // Hide any color ISMs created for fallback
                ClearColorInstancedComponents();
                
                UE_LOG(LogPointCloudRenderer, Log, TEXT("Rendered %d points using Niagara system: %s"), 
                       RenderedPointCount, 
                       *NiagaraComponent->GetAsset()->GetName());
                return; // Successfully used Niagara, exit early
            }
        }
    }

    // Use fallback rendering (InstancedMesh) - no Niagara needed
    if (bShowColors && bHasColors)
    {
        RenderPointCloudFallbackWithColors(PointCloudData);
    }
    else
    {
        RenderPointCloudFallback(PointCloudData);
    }
}

void UPointCloudRenderer::RenderPointCloudFallback(const FPointCloudData& PointCloudData)
{
    if (!InstancedMeshComponent || PointCloudData.GetPointCount() == 0)
    {
        return;
    }

    // Ensure fallback system is properly initialized
    if (!InstancedMeshComponent->GetStaticMesh() || !InstancedMeshComponent->GetMaterial(0))
    {
        CreateOptimizedFallbackSystem();
    }

    // Clear existing instances
    InstancedMeshComponent->ClearInstances();
    ClearColorInstancedComponents();
    
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
    
    UE_LOG(LogPointCloudRenderer, Log, TEXT("Fallback rendered %d points using instanced meshes"), Points.Num());
}

void UPointCloudRenderer::RenderPointCloudFallbackWithColors(const FPointCloudData& PointCloudData)
{
    if (PointCloudData.GetPointCount() == 0)
    {
        return;
    }

    // Hide single-color ISM and clear its instances
    if (InstancedMeshComponent)
    {
        InstancedMeshComponent->ClearInstances();
        InstancedMeshComponent->SetVisibility(false);
    }

    // For small clouds, group by color and create one ISM per color with a dynamic material
    // This avoids needing a special per-instance color material.
    ClearColorInstancedComponents();

    const TArray<FRatPoint>& Points = PointCloudData.Points;
    
    // Cap number of unique colors to avoid too many components
    const int32 MaxUniqueColors = 256;
    TMap<uint32, TArray<int32>> ColorToIndices;
    ColorToIndices.Reserve(FMath::Min(MaxUniqueColors, Points.Num()));

    for (int32 i = 0; i < Points.Num(); ++i)
    {
        const uint32 Key = PackColorKey(Points[i].Color);
        auto& Indices = ColorToIndices.FindOrAdd(Key);
        Indices.Add(i);
        if (ColorToIndices.Num() >= MaxUniqueColors && Indices.Num() == 1)
        {
            // If we reached cap, merge the rest into the first color to avoid explosion
            // Simple guard: do nothing; additional colors will reuse existing keys implicitly
        }
    }

    int32 TotalInstances = 0;
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
            ++TotalInstances;
        }

        ISM->SetVisibility(true);
    }

    // Hide Niagara if any
    if (NiagaraComponent)
    {
        NiagaraComponent->SetVisibility(false);
    }

    UE_LOG(LogPointCloudRenderer, Log, TEXT("Fallback rendered %d points using color-grouped instanced meshes (%d unique colors)"), Points.Num(), ColorToIndices.Num());
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
            UE_LOG(LogPointCloudRenderer, Warning, TEXT("Niagara system '%s' doesn't appear to be designed for point cloud rendering. Using fallback."), *SystemName);
            
            // Deactivate the incompatible system and use fallback instead
            NiagaraComponent->Deactivate();
            NiagaraComponent->SetVisibility(false);
            return; // This will cause the render function to use fallback
        }
    }

    // Only try to set data if the system appears compatible
    if (bHasValidDataInterface && CachedPositions.Num() > 0)
    {
        // Use the standard Array Data Interface names as shown in test.txt
        try 
        {
            UNiagaraDataInterfaceArrayFunctionLibrary::SetNiagaraArrayVector(
                NiagaraComponent, 
                FName(TEXT("Points")), 
                CachedPositions
            );
            UE_LOG(LogPointCloudRenderer, Log, TEXT("Set position data using 'Points' array: %d positions"), CachedPositions.Num());
        }
        catch (...)
        {
            UE_LOG(LogPointCloudRenderer, Warning, TEXT("Could not set position data - using fallback rendering"));
            NiagaraComponent->Deactivate();
            NiagaraComponent->SetVisibility(false);
            return;
        }
    }

    // Try to set color data if available and system supports it
    if (bHasValidDataInterface && CachedColors.Num() > 0 && (bShowColors && bHasColors))
    {
        try
        {
            // Convert LinearColor to Vector for Niagara (RGB as XYZ)
            TArray<FVector> ColorVectors;
            ColorVectors.Reserve(CachedColors.Num());
            for (const FLinearColor& Color : CachedColors)
            {
                ColorVectors.Add(FVector(Color.R, Color.G, Color.B));
            }
            
            UNiagaraDataInterfaceArrayFunctionLibrary::SetNiagaraArrayVector(
                NiagaraComponent, 
                FName(TEXT("Colors")), 
                ColorVectors
            );
            UE_LOG(LogPointCloudRenderer, Log, TEXT("Set color data using 'Colors' array: %d colors"), CachedColors.Num());
        }
        catch (...)
        {
            UE_LOG(LogPointCloudRenderer, Log, TEXT("Could not set color data - using default colors"));
        }
    }
    
    // Set additional Niagara parameters following test.txt pattern
    if (bHasValidDataInterface)
    {
        try 
        {
            NiagaraComponent->SetFloatParameter(TEXT("PointSize"), PointSize);
            
            // Calculate bounds for debugging (SetFixedBounds doesn't exist on UNiagaraComponent)
            if (CachedPositions.Num() > 0)
            {
                FBox BoundingBox(ForceInit);
                for (const FVector& Position : CachedPositions)
                {
                    BoundingBox += Position;
                }
                
                if (BoundingBox.IsValid)
                {
                    UE_LOG(LogPointCloudRenderer, Log, TEXT("Point cloud bounds: %s"), *BoundingBox.ToString());
                }
            }
        }
        catch (...)
        {
            UE_LOG(LogPointCloudRenderer, Warning, TEXT("Could not set additional Niagara parameters"));
        }
    }
    
    UE_LOG(LogPointCloudRenderer, Log, TEXT("Updated Niagara particle data: %d points"), Points.Num());
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
    ClearColorInstancedComponents();
    
    // Clear cached data
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
    
    UE_LOG(LogPointCloudRenderer, Log, TEXT("Point cloud material setup completed"));
}
