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
#include "Engine/TextureRenderTarget2D.h"
#include "Materials/MaterialInterface.h"
#include "RHIResources.h"
#include "DepthCamera.generated.h"

class FDepthCameraConfig: public FCameraConfig
{
public:
    
    float MaxRange = 10000.0f;
    float MinRange = 0.0f;
    
    FDepthCameraConfig()
    {
        Width = 1920;
        Height = 1080;
    }
};

struct FDCPoint
{
    FVector Location;
    FDCPoint() : Location(FVector::ZeroVector){}
};

UCLASS(ClassGroup = (VCCSIM), meta = (BlueprintSpawnableComponent))
class VCCSIM_API UDepthCameraComponent : public UCameraBaseComponent
{
    GENERATED_BODY()

public:
    UDepthCameraComponent() = default;
    virtual void Configure(const FSensorConfig& Config) override final;

    UFUNCTION(BlueprintCallable, Category = "DepthCamera")
    void CaptureDepthScene();
    UFUNCTION(BlueprintCallable, Category = "DepthCamera")
    void CaptureDepthSceneDeferred();
    UFUNCTION(BlueprintCallable, Category = "DepthCamera")
    void CaptureDepthSceneAndProcess();
    virtual UTextureRenderTarget2D* GetRenderTarget() const override { return RenderTarget; }

    // For grpc server
    void AsyncGetDepthImageData(TFunction<void(const TArray<FFloat16Color>&)> Callback);
    void AsyncGetPointCloudData(TFunction<void()> Callback);
    
    UFUNCTION(BlueprintCallable, Category = "RGBDCamera")
    void VisualizePointCloud();
    TArray<FDCPoint> GeneratePointCloud();

    // UCameraBaseComponent interface
    virtual ESensorType GetSensorType() const override { return ESensorType::DepthCamera; }

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "DepthCamera|Config")
    float MaxRange = 10000.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "DepthCamera|Config")
    float MinRange = 0.0f;

protected:
    virtual void InitializeRenderTargets() override;
    virtual void SetCaptureComponent() const override;

    void ProcessDepthTexture(TFunction<void()> OnComplete);
    void ProcessDepthTextureParam(TFunction<void(const TArray<FFloat16Color>&)> OnComplete);

private:    
    TArray<FFloat16Color> DepthData;
    TArray<FDCPoint> PointCloudData;

    template<typename CallbackType>
    void ProcessDepthTextureTemplate(CallbackType&& Callback);
};

template<typename CallbackType>
void UDepthCameraComponent::ProcessDepthTextureTemplate(CallbackType&& Callback)
{
    if (!RenderTarget) { UE_LOG(LogTemp, Error, TEXT("DepthRenderTarget is null!")); return; }

    if (DepthData.Num() != Width * Height)
    {
        DepthData.SetNumUninitialized(Width * Height);
    }

    struct FReadSurfaceContext
    {
        TArray<FFloat16Color>* OutData;
        FIntRect Rect;
    } Context { &DepthData, FIntRect(0, 0, Width, Height) };

    auto SharedCallback = MakeShared<std::decay_t<CallbackType>>(std::forward<CallbackType>(Callback));

    UTextureRenderTarget2D* RT = RenderTarget;

    ENQUEUE_RENDER_COMMAND(ReadDepthSurfaceCommand)(
        [RT, Context, SharedCallback](FRHICommandListImmediate& RHICmdList)
        {
            if (!RT) return;

            FTextureRenderTargetResource* RTRes = RT->GetRenderTargetResource();
            if (!RTRes) return;

            RHICmdList.ReadSurfaceFloatData(
                RTRes->GetRenderTargetTexture(),
                Context.Rect,
                *Context.OutData,
                FReadSurfaceDataFlags()
            );

            if constexpr (std::is_invocable_v<std::decay_t<CallbackType>>)
            { (*SharedCallback)(); }
            else if constexpr (std::is_invocable_v<std::decay_t<CallbackType>, const TArray<FFloat16Color>&>)
            { (*SharedCallback)(*Context.OutData); }
        }
    );
}
