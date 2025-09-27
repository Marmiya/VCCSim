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
#include "ISensorDataProvider.h"
#include "GameFramework/Actor.h"
#include "SensorBase.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Materials/MaterialInterface.h"
#include "RHIResources.h"
#include "SegmentCamera.generated.h"

class FSegmentationCameraConfig : public FCameraConfig
{
public:
    float MaxRange = 10000.0f;

    FSegmentationCameraConfig()
    {
        Width = 512;
        Height = 512;
    }
};

UCLASS(ClassGroup = (VCCSIM), meta = (BlueprintSpawnableComponent))
class VCCSIM_API USegmentationCameraComponent : public UCameraBaseComponent, public ISensorDataProvider
{
    GENERATED_BODY()

public:
    USegmentationCameraComponent();
    virtual void Configure(const FSensorConfig& Config) override final;
    FString CameraName;
    
    UFUNCTION(BlueprintCallable, Category = "SegmentationCamera")
    void CaptureSegmentationScene();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SegmentationCamera|Config")
    UMaterialInterface* SegmentationMaterial = Cast<UMaterialInterface>(
        StaticLoadObject(UMaterialInterface::StaticClass(), nullptr,
            TEXT("/VCCSim/Materials/M_Segmentation.M_Segmentation")));
    
    virtual UTextureRenderTarget2D* GetRenderTarget() const override { return RenderTarget; }

    // For GRPC call
    void AsyncGetSegmentationImageData(TFunction<void(const TArray<FColor>&)> Callback);

    // ISensorDataProvider interface
    virtual FIntPoint GetResolution() const override { return FIntPoint(Width, Height); }
    virtual ESensorType GetSensorType() const override { return ESensorType::SegmentationCamera; }
    virtual AActor* GetOwnerActor() const override { return ParentActor; }

protected:
    virtual void InitializeRenderTargets() override;
    virtual void SetCaptureComponent() const override;
    void ProcessSegmentationTexture(TFunction<void()> OnComplete);
    void ProcessSegmentationTextureParam(TFunction<void(const TArray<FColor>&)> OnComplete);

private:
    TArray<FColor> SegmentationData;
    bool Dirty = false;
    
    template<typename CallbackType>
    void ProcessSegTextureTemplate(CallbackType&& Callback);
};

template<typename CallbackType>
void USegmentationCameraComponent::ProcessSegTextureTemplate(CallbackType&& Callback)
{
    if (!RenderTarget) { UE_LOG(LogTemp, Error, TEXT("SegmentationRenderTarget is null!")); return; }

    if (SegmentationData.Num() != Width * Height)
    {
        SegmentationData.SetNumUninitialized(Width * Height);
    }

    struct FReadSurfaceContext
    {
        TArray<FColor>* OutData;
        FIntRect Rect;
        FReadSurfaceDataFlags Flags;
    } Context { &SegmentationData, FIntRect(0, 0, Width, Height),
                FReadSurfaceDataFlags(RCM_UNorm, CubeFace_MAX) };

    auto SharedCallback = MakeShared<std::decay_t<CallbackType>>(std::forward<CallbackType>(Callback));

    UTextureRenderTarget2D* RT = RenderTarget;

    ENQUEUE_RENDER_COMMAND(ReadSurfaceCommand)(
        [RT, Context, SharedCallback](FRHICommandListImmediate& RHICmdList)
        {
            if (!RT) return;

            FTextureRenderTargetResource* RTRes = RT->GetRenderTargetResource();
            if (!RTRes) return;

            RHICmdList.ReadSurfaceData(
                RTRes->GetRenderTargetTexture(),
                Context.Rect,
                *Context.OutData,
                Context.Flags
            );
            
            for (auto& Color : *Context.OutData)
            {
                Color.A = 255;
            }

            if constexpr (std::is_invocable_v<std::decay_t<CallbackType>>)
            { (*SharedCallback)(); }
            else if constexpr (std::is_invocable_v<std::decay_t<CallbackType>, const TArray<FColor>&>)
            { (*SharedCallback)(*Context.OutData); }
        }
    );
}