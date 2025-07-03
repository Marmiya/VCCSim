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
#include "Components/Image.h"
#include "Blueprint/UserWidget.h"

#include "VCCSIMDisplayWidget.generated.h"

class UMeshHandlerComponent;
class UDepthCameraComponent;
class URGBCameraComponent;
class USegmentationCameraComponent;
class UNormalCameraComponent;

UCLASS()
class VCCSIM_API UVCCSIMDisplayWidget : public UUserWidget
{
    GENERATED_BODY()

public:
    
    virtual void NativeConstruct() override;
    virtual void NativeTick(const FGeometry& MyGeometry, float InDeltaTime) override;

    void InitFromConfig(const struct FVCCSimConfig& Config);
    
    UFUNCTION()
    void SetHolder(AActor* holder){ Holder = holder; }
    
    void SetDepthContext(UTextureRenderTarget2D* DepthTexture, UDepthCameraComponent* InCamera);
    void SetRGBContext(UTextureRenderTarget2D* RGBTexture, URGBCameraComponent* InCamera);
    void SetSegContext(UTextureRenderTarget2D* SegTexture, USegmentationCameraComponent* InCamera);
    void SetNormalContext(UTextureRenderTarget2D* NormalTexture, UNormalCameraComponent* InCamera);
    
    UFUNCTION(BlueprintCallable, Category = "LitView")
    void SetLitMeshComponent(TArray<UStaticMeshComponent*> MeshComponent,
        const float& Opacity);

    UFUNCTION(BlueprintCallable, Category = "PCView")
    void SetPCViewComponent(UInstancedStaticMeshComponent* InInstancedMeshComponent
        ,const float& Opacity);
    
    UFUNCTION(BlueprintCallable, Category = "UnitView")
    void SetMeshHandler(UMeshHandlerComponent* InMeshHandler, const float& Opacity);
    
    // ID:
    // 4 -> Normal,
    // 5 -> Segmentation,
    // 6 -> RGB,
    // 7 -> Lit,
    // 8 -> PC,
    // 9 -> Unit Dynamic
    // 0 -> Depth
    UFUNCTION(BlueprintCallable, Category = "Capture")
    void RequestCapture(const int32& ID);

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ViewSaver")
    FString LogSavePath = FPaths::ProjectLogDir();
    
protected:
    // Depth camera visualization properties
    UPROPERTY(BlueprintReadWrite, meta = (BindWidget))
    TObjectPtr<UImage> DepthImageDisplay;
    UPROPERTY(EditDefaultsOnly, Category = "DepthCamera")
    TObjectPtr<UMaterialInterface> DepthVisualizationMaterial;
    UPROPERTY()
    TObjectPtr<UMaterialInstanceDynamic> DepthMaterial;
    UPROPERTY()
    TObjectPtr<UTextureRenderTarget2D> DepthRenderTarget;
    UPROPERTY()
    UDepthCameraComponent* DepthCameraComponent = nullptr;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "DepthCamera|Visualization")
    float MinDepth = 0.0f;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "DepthCamera|Visualization")
    float MaxDepth = 5000.0f;
    UPROPERTY(EditAnywhere, BlueprintReadWrite,
        Category = "DepthCamera|Visualization", meta = (ClampMin = "0.1", ClampMax = "5.0"))
    float Contrast = 1.2f;
    UPROPERTY(EditAnywhere, BlueprintReadWrite,
        Category = "DepthCamera|Visualization", meta = (ClampMin = "0.1", ClampMax = "5.0"))
    float Brightness = 1.0f;
    
    // RGB camera visualization properties
    UPROPERTY(BlueprintReadWrite, meta = (BindWidget))
    TObjectPtr<UImage> RGBImageDisplay;
    UPROPERTY(EditDefaultsOnly, Category = "RGBCamera")
    TObjectPtr<UMaterialInterface> RGBVisualizationMaterial;
    UPROPERTY()
    TObjectPtr<UMaterialInstanceDynamic> RGBMaterial;
    UPROPERTY()
    TObjectPtr<UTextureRenderTarget2D> RGBRenderTarget;
    UPROPERTY()
    URGBCameraComponent* RGBCameraComponent = nullptr;

    // Segmentation camera visualization properties
    UPROPERTY(BlueprintReadWrite, meta = (BindWidget))
    TObjectPtr<UImage> SegImageDisplay;
    UPROPERTY(EditDefaultsOnly, Category = "SegCamera")
    TObjectPtr<UMaterialInterface> SegVisualizationMaterial;
    UPROPERTY()
    TObjectPtr<UMaterialInstanceDynamic> SegMaterial;
    UPROPERTY()
    TObjectPtr<UTextureRenderTarget2D> SegRenderTarget;
    UPROPERTY()
    USegmentationCameraComponent* SegCameraComponent = nullptr;

    // Normal visualization properties
    UPROPERTY(BlueprintReadWrite, meta = (BindWidget))
    TObjectPtr<UImage> NormalImageDisplay;
    UPROPERTY(EditDefaultsOnly, Category = "NormalView")
    TObjectPtr<UMaterialInterface> NormalVisualizationMaterial;
    UPROPERTY()
    TObjectPtr<UMaterialInstanceDynamic> NormalMaterial;
    UPROPERTY()
    TObjectPtr<UTextureRenderTarget2D> NormalRenderTarget;
    UPROPERTY()
    UNormalCameraComponent* NormalCameraComponent = nullptr;
        
    // Lit image visualization properties
    UPROPERTY(BlueprintReadWrite, meta = (BindWidget))
    TObjectPtr<UImage> LitImageDisplay;
    UPROPERTY(EditDefaultsOnly, Category = "LitView")
    TObjectPtr<UMaterialInterface> LitVisualizationMaterial;
    UPROPERTY(EditDefaultsOnly, Category = "LitView")
    int32 LitRenderWidth = 960;
    UPROPERTY(EditDefaultsOnly, Category = "LitView")
    int32 LitRenderHeight = 540;
    UPROPERTY(EditDefaultsOnly, Category = "LitView")
    float LitUpdateInterval = 1.0f/30.0f;
    UPROPERTY()
    TObjectPtr<UTextureRenderTarget2D> LitRenderTarget;
    UPROPERTY()
    TObjectPtr<USceneCaptureComponent2D> LitSceneCapture;
    UPROPERTY()
    TObjectPtr<UMaterialInstanceDynamic> LitMaterial;
    
    // Point cloud visualization properties
    UPROPERTY(BlueprintReadWrite, meta = (BindWidget))
    TObjectPtr<UImage> PCImageDisplay;
    UPROPERTY(EditDefaultsOnly, Category = "PCView")
    TObjectPtr<UMaterialInterface> PCVisualizationMaterial;
    UPROPERTY(EditDefaultsOnly, Category = "PCView")
    int32 PCRenderWidth = 960;
    UPROPERTY(EditDefaultsOnly, Category = "PCView")
    int32 PCRenderHeight = 540;
    UPROPERTY(EditDefaultsOnly, Category = "PCView")
    float PCUpdateInterval = 0.1f;
    UPROPERTY()
    TObjectPtr<UTextureRenderTarget2D> PCRenderTarget;
    UPROPERTY()
    TObjectPtr<USceneCaptureComponent2D> PCSceneCapture;
    UPROPERTY()
    TObjectPtr<UMaterialInstanceDynamic> PCMaterial;
    
    // Unit visualization properties
    UPROPERTY(BlueprintReadWrite, meta = (BindWidget))
    TObjectPtr<UImage> UnitImageDisplay;
    UPROPERTY(EditDefaultsOnly, Category = "UnitView")
    TObjectPtr<UMaterialInterface> MeshVisualizationMaterial;
    UPROPERTY(EditDefaultsOnly, Category = "UnitView")
    int32 MeshRenderWidth = 960;
    UPROPERTY(EditDefaultsOnly, Category = "UnitView")
    int32 MeshRenderHeight = 540;
    UPROPERTY(EditDefaultsOnly, Category = "UnitView")
    float MeshUpdateInterval = 0.1f;
    UPROPERTY()
    TObjectPtr<UTextureRenderTarget2D> MeshRenderTarget;
    UPROPERTY()
    TObjectPtr<USceneCaptureComponent2D> MeshSceneCapture;
    UPROPERTY()
    TObjectPtr<UMaterialInstanceDynamic> MeshMaterial;
    UPROPERTY()
    TObjectPtr<UMeshHandlerComponent> MeshHandler;

private:
    void UpdateLitImage(float InDeltaTime);
    void UpdatePCImage(float InDeltaTime);
    void UpdateMeshImage(float InDeltaTime);
    
    float LitUpdateTimer = 0.0f;
    float PCUpdateTimer = 0.0f;
    float MeshUpdateTimer = 0.0f;

    TQueue<int32> CaptureQueue;
    
    // Maximum queued captures
    const int32 MaxQueuedCaptures = 15;
    int32 CurrentQueueSize = 0;
    
    UPROPERTY()
    TObjectPtr<AActor> Holder = nullptr;

    void ProcessCapture(const int32 ID);
    void SaveRenderTargetToDisk(
        UTextureRenderTarget2D* RenderTarget, const FString& FileName) const;
};