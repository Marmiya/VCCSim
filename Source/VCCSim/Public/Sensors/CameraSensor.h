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
#include "GameFramework/Actor.h"
#include "SensorBase.h"
#include "ISensorDataProvider.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Materials/MaterialInterface.h"
#include "RHIResources.h"
#include "CameraSensor.generated.h"

class UInsMeshHolder;

struct FDCPoint
{
    FVector Location;
    FDCPoint() : Location(FVector::ZeroVector){}
};

class FRGBDCameraConfig : public FCameraConfig
{
public:
    float MaxRange = 10000.0f;
    float MinRange = 0.0f;
    bool bSaveRGB = true;
    bool bSaveDepth = true;

    FRGBDCameraConfig()
    {
        FOV = 90.0f;
        Width = 512;
        Height = 512;
    }
};

UCLASS(ClassGroup = (VCCSIM), meta = (BlueprintSpawnableComponent))
class VCCSIM_API URGBDCameraComponent : public UCameraBaseComponent, public ISensorDataProvider
{
    GENERATED_BODY()

public:
    URGBDCameraComponent();
    virtual void Configure(const FSensorConfig& Config) override final;
    void SetIgnoreLidar(UInsMeshHolder* MeshHolder);
    FString CameraName;

    UFUNCTION(BlueprintCallable, Category = "RGBDCamera")
    void CaptureRGBDScene();

    UFUNCTION(BlueprintCallable, Category = "RGBDCamera")
    void VisualizePointCloud();

    TArray<FDCPoint> GeneratePointCloud();

    // For grpc service
    void AsyncGetRGBDImageData(TFunction<void(const TArray<FLinearColor>&)> Callback);
    void AsyncGetPointCloudData(TFunction<void()> Callback);

    // ISensorDataProvider interface
    virtual UTextureRenderTarget2D* GetRenderTarget() const override { return RenderTarget; }
    virtual FIntPoint GetResolution() const override { return FIntPoint(Width, Height); }
    virtual ESensorType GetSensorType() const override { return ESensorType::RGBDCamera; }
    virtual AActor* GetOwnerActor() const override { return ParentActor; }
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RGBDCamera|Config")
    float MaxRange = 10000.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RGBDCamera|Config")
    float MinRange = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RGBDCamera|Config")
    bool bSaveRGB = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RGBDCamera|Config")
    bool bSaveDepth = true;

    TArray<FDCPoint> PointCloudData;

    const TArray<FLinearColor>& GetCombinedData() const { return CombinedData; }

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RGBDCamera|Config")
    UMaterialInterface* RGBDMaterial = Cast<UMaterialInterface>(
        StaticLoadObject(UMaterialInterface::StaticClass(), nullptr,
            TEXT("/VCCSim/Materials/M_RGBD.M_RGBD")));
        
protected:
    virtual void InitializeRenderTargets() override;

    virtual void SetCaptureComponent() const override;
    void ProcessRGBDTexture(TFunction<void()> OnComplete);
    void ProcessRGBDTextureParam(TFunction<void(const TArray<FLinearColor>&)> OnComplete);

private:
    TArray<FLinearColor> CombinedData;

    template<typename CallbackType>
    void ProcessRGBDepthTextureTemplate(CallbackType&& Callback);
};

template<typename CallbackType>
void URGBDCameraComponent::ProcessRGBDepthTextureTemplate(CallbackType&& Callback)
{
    if (!RenderTarget)
    {
        UE_LOG(LogTemp, Error, TEXT("CombinedRenderTarget is null!"));
        return;
    }

    if (CombinedData.Num() != Width * Height)
    {
        CombinedData.SetNumUninitialized(Width * Height);
    }

    struct FReadSurfaceContext
    {
        TArray<FLinearColor>* OutCombinedData;
        FIntRect Rect;
        FReadSurfaceDataFlags Flags;
    } Context { &CombinedData, FIntRect(0, 0, Width, Height),
                FReadSurfaceDataFlags(RCM_UNorm, CubeFace_MAX) };

    auto SharedCallback = MakeShared<std::decay_t<CallbackType>>(std::forward<CallbackType>(Callback));

    UTextureRenderTarget2D* RT = RenderTarget;

    ENQUEUE_RENDER_COMMAND(ReadSurfaceCommand)(
        [RT, Context, SharedCallback](FRHICommandListImmediate& RHICmdList)
        {
            if (!RT) return;

            FTextureRenderTargetResource* RTRes = RT->GetRenderTargetResource();
            if (!RTRes) return;

            TArray<FLinearColor> TempData;
            RHICmdList.ReadSurfaceData(
                RTRes->GetRenderTargetTexture(),
                Context.Rect,
                TempData,
                FReadSurfaceDataFlags()
            );

            Context.OutCombinedData->SetNumUninitialized(TempData.Num());
            for (int32 i = 0; i < TempData.Num(); ++i)
            {
                (*Context.OutCombinedData)[i] = TempData[i];
            }

            if constexpr (std::is_invocable_v<std::decay_t<CallbackType>>)
            { (*SharedCallback)(); }
            else if constexpr (std::is_invocable_v<std::decay_t<CallbackType>, const TArray<FLinearColor>&>)
            { (*SharedCallback)(*Context.OutCombinedData); }
        }
    );
}