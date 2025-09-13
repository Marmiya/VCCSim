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

#pragma once

#include "CoreMinimal.h"
#include "Components/SceneComponent.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "NiagaraComponent.h"
#include "NiagaraSystem.h"
#include "DataStructures/PointCloud.h"
#include "PointCloudRenderer.generated.h"

/**
 * High-performance point cloud renderer using Niagara particle system
 * Supports colored point clouds with excellent performance for large datasets
 */
UCLASS(ClassGroup=(VCCSim), meta=(BlueprintSpawnableComponent))
class VCCSIM_API UPointCloudRenderer : public USceneComponent
{
    GENERATED_BODY()

public:
    UPointCloudRenderer();

protected:
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    virtual void OnRegister() override;

public:
    /**
     * Render point cloud using particle system
     * @param PointCloudData Point cloud data to render
     * @param bShowColors Whether to display colors (if available)
     * @param PointSize Size of each point
     */
    UFUNCTION(BlueprintCallable, Category = "Point Cloud Rendering")
    void RenderPointCloud(const FPointCloudData& PointCloudData, bool bShowColors = true, float PointSize = 1.0f);

    /**
     * Clear the current point cloud visualization
     */
    UFUNCTION(BlueprintCallable, Category = "Point Cloud Rendering")
    void ClearPointCloud();

    /**
     * Update point size
     */
    UFUNCTION(BlueprintCallable, Category = "Point Cloud Rendering")
    void SetPointSize(float NewPointSize);

    /**
     * Toggle color display
     */
    UFUNCTION(BlueprintCallable, Category = "Point Cloud Rendering")
    void SetShowColors(bool bInShowColors);

    /**
     * Get current point count being rendered
     */
    UFUNCTION(BlueprintCallable, BlueprintPure, Category = "Point Cloud Rendering")
    int32 GetRenderedPointCount() const { return RenderedPointCount; }

    /**
     * Check if point cloud has colors
     */
    UFUNCTION(BlueprintCallable, BlueprintPure, Category = "Point Cloud Rendering")
    bool HasColors() const { return bHasColors; }

    /**
     * Whether a Niagara asset is set and available on the component
     */
    UFUNCTION(BlueprintCallable, BlueprintPure, Category = "Point Cloud Rendering")
    bool IsNiagaraAvailable() const { return NiagaraComponent && NiagaraComponent->GetAsset() != nullptr; }

protected:
    /**
     * Initialize Niagara system for point cloud rendering
     */
    void InitializeNiagaraSystem();

    /**
     * Update particle data with point cloud information
     */
    void UpdateParticleData(const FPointCloudData& PointCloudData);

    /**
     * Create or load the point cloud material
     */
    void SetupPointCloudMaterial();

    /**
     * Create a fallback system when Niagara is not available
     */
    void CreateFallbackSystem();
    
    /**
     * Create an optimized fallback system using instanced meshes
     */
    void CreateOptimizedFallbackSystem();

    /**
     * Fallback rendering method
     */
    void RenderPointCloudFallback(const FPointCloudData& PointCloudData);

    /** Color-aware fallback rendering (groups instances per color) */
    void RenderPointCloudFallbackWithColors(const FPointCloudData& PointCloudData);

    /** Destroy and clear any per-color instanced components */
    void ClearColorInstancedComponents();

    /** Get or create instanced component for a specific color */
    UInstancedStaticMeshComponent* GetOrCreateColorISM(const FLinearColor& Color);

protected:
    /** Optional Niagara system asset to drive rendering (set in editor). */
    UPROPERTY(EditAnywhere, Category = "Rendering")
    TSoftObjectPtr<UNiagaraSystem> NiagaraSystemAsset;

    /** If true, try known engine Niagara defaults (may spam logs if not installed). */
    UPROPERTY(EditAnywhere, Category = "Rendering")
    bool bSearchEngineNiagaraDefaults = false;

    // Niagara particle system component
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
    TObjectPtr<UNiagaraComponent> NiagaraComponent;
    
    // Fallback instanced static mesh component
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
    TObjectPtr<UInstancedStaticMeshComponent> InstancedMeshComponent;

    // Additional ISMs for color-grouped fallback rendering
    UPROPERTY(Transient)
    TMap<uint32, TObjectPtr<UInstancedStaticMeshComponent>> ColorInstancedComponents;

    // Point cloud rendering settings
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Rendering")
    float PointSize = 1.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Rendering")
    bool bShowColors = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Rendering")
    FLinearColor DefaultColor = FLinearColor(1.0f, 0.5f, 0.0f, 1.0f);

    // Runtime state
    UPROPERTY()
    int32 RenderedPointCount = 0;

    UPROPERTY()
    bool bHasColors = false;

    UPROPERTY()
    bool bIsInitialized = false;

    // Cached point cloud data for updates
    TArray<FVector> CachedPositions;
    TArray<FLinearColor> CachedColors;

private:
    /** Pack linear color (0..1) to 0xRRGGBB key */
    static uint32 PackColorKey(const FLinearColor& Color)
    {
        const uint8 R = (uint8)FMath::Clamp(FMath::RoundToInt(Color.R * 255.0f), 0, 255);
        const uint8 G = (uint8)FMath::Clamp(FMath::RoundToInt(Color.G * 255.0f), 0, 255);
        const uint8 B = (uint8)FMath::Clamp(FMath::RoundToInt(Color.B * 255.0f), 0, 255);
        return ((uint32)R << 16) | ((uint32)G << 8) | (uint32)B;
    }
};
