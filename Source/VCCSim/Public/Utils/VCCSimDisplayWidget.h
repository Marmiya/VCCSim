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
class URGBCameraComponent;
class UDepthCameraComponent;
class USegCameraComponent;
class UNormalCameraComponent;

UENUM(BlueprintType)
enum class EVCCSimViewType : uint8
{
    Unit = 0        UMETA(DisplayName = "Unit"), //dynamic
    RGB = 4         UMETA(DisplayName = "RGB"),
    Depth = 5       UMETA(DisplayName = "Depth"),
    Normal = 6      UMETA(DisplayName = "Normal"),
    Segmentation = 7 UMETA(DisplayName = "Segmentation"),
    Lit = 8         UMETA(DisplayName = "Lit"),
    PointCloud = 9  UMETA(DisplayName = "Point Cloud")
};

USTRUCT()
struct FVCCSimViewData
{
    GENERATED_BODY()

    UPROPERTY()
    TObjectPtr<UImage> ImageDisplay = nullptr;

    UPROPERTY()
    TObjectPtr<UMaterialInterface> VisualizationMaterial = nullptr;

    UPROPERTY()
    TObjectPtr<UMaterialInstanceDynamic> MaterialInstance = nullptr;

    UPROPERTY()
    TObjectPtr<UTextureRenderTarget2D> RenderTarget = nullptr;

    UPROPERTY()
    TObjectPtr<UObject> CameraComponent = nullptr;

    UPROPERTY()
    TObjectPtr<USceneCaptureComponent2D> SceneCapture = nullptr;

    UPROPERTY()
    TObjectPtr<UMeshHandlerComponent> MeshHandler = nullptr;

    UPROPERTY(EditAnywhere, Category = "View Settings")
    int32 RenderWidth = 960;

    UPROPERTY(EditAnywhere, Category = "View Settings")
    int32 RenderHeight = 540;

    UPROPERTY(EditAnywhere, Category = "View Settings")
    float UpdateInterval = 0.1f;

    float UpdateTimer = 0.0f;
};

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
    
    void SetCameraContext(EVCCSimViewType ViewType, UTextureRenderTarget2D* RenderTexture, UObject* CameraComponent);

    void SetDepthContext(UTextureRenderTarget2D* DepthTexture, UDepthCameraComponent* InCamera);
    void SetRGBContext(UTextureRenderTarget2D* RGBTexture, URGBCameraComponent* InCamera);
    void SetSegContext(UTextureRenderTarget2D* SegTexture, USegCameraComponent* InCamera);
    void SetNormalContext(UTextureRenderTarget2D* NormalTexture, UNormalCameraComponent* InCamera);
    
    UFUNCTION(BlueprintCallable, Category = "LitView")
    void SetLitMeshComponent(TArray<UStaticMeshComponent*> MeshComponent,
        const float& Opacity);

    UFUNCTION(BlueprintCallable, Category = "PCView")
    void SetPCViewComponent(UInstancedStaticMeshComponent* InInstancedMeshComponent
        ,const float& Opacity);
    
    UFUNCTION(BlueprintCallable, Category = "UnitView")
    void SetMeshHandler(UMeshHandlerComponent* InMeshHandler, const float& Opacity);
    
    UFUNCTION(BlueprintCallable, Category = "Capture")
    void RequestCapture(const int32& ID);

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ViewSaver")
    FString LogSavePath = FPaths::ProjectSavedDir();
    
protected:
    UPROPERTY(BlueprintReadWrite, meta = (BindWidget))
    TObjectPtr<UImage> DepthImageDisplay;
    UPROPERTY(BlueprintReadWrite, meta = (BindWidget))
    TObjectPtr<UImage> RGBImageDisplay;
    UPROPERTY(BlueprintReadWrite, meta = (BindWidget))
    TObjectPtr<UImage> SegImageDisplay;
    UPROPERTY(BlueprintReadWrite, meta = (BindWidget))
    TObjectPtr<UImage> NormalImageDisplay;
    UPROPERTY(BlueprintReadWrite, meta = (BindWidget))
    TObjectPtr<UImage> LitImageDisplay;
    UPROPERTY(BlueprintReadWrite, meta = (BindWidget))
    TObjectPtr<UImage> PCImageDisplay;
    UPROPERTY(BlueprintReadWrite, meta = (BindWidget))
    TObjectPtr<UImage> UnitImageDisplay;

    UPROPERTY(EditDefaultsOnly, Category = "ViewMaterials")
    TObjectPtr<UMaterialInterface> DepthVisualizationMaterial;
    UPROPERTY(EditDefaultsOnly, Category = "ViewMaterials")
    TObjectPtr<UMaterialInterface> RGBVisualizationMaterial;
    UPROPERTY(EditDefaultsOnly, Category = "ViewMaterials")
    TObjectPtr<UMaterialInterface> SegVisualizationMaterial;
    UPROPERTY(EditDefaultsOnly, Category = "ViewMaterials")
    TObjectPtr<UMaterialInterface> NormalVisualizationMaterial;
    UPROPERTY(EditDefaultsOnly, Category = "ViewMaterials")
    TObjectPtr<UMaterialInterface> LitVisualizationMaterial;
    UPROPERTY(EditDefaultsOnly, Category = "ViewMaterials")
    TObjectPtr<UMaterialInterface> PCVisualizationMaterial;
    UPROPERTY(EditDefaultsOnly, Category = "ViewMaterials")
    TObjectPtr<UMaterialInterface> MeshVisualizationMaterial;

    UPROPERTY()
    TMap<EVCCSimViewType, FVCCSimViewData> ViewDataMap;

private:
    void UpdateViewImage(EVCCSimViewType ViewType, float InDeltaTime);

    FVCCSimViewData* GetViewData(EVCCSimViewType ViewType);
    const FVCCSimViewData* GetViewData(EVCCSimViewType ViewType) const;

    void InitializeViewData();
    void SetupViewBindings();

    TQueue<int32> CaptureQueue;
    
    // Maximum queued captures
    const int32 MaxQueuedCaptures = 15;
    int32 CurrentQueueSize = 0;
    
    UPROPERTY()
    TObjectPtr<AActor> Holder = nullptr;

    void ProcessCapture(const int32 ID);
    void ProcessCaptureByType(EVCCSimViewType ViewType);
    void SaveRenderTargetToDisk(
        UTextureRenderTarget2D* RenderTarget, const FString& FileName, EVCCSimViewType ViewType) const;

    static EVCCSimViewType IDToViewType(int32 ID);
    static int32 ViewTypeToID(EVCCSimViewType ViewType);
};