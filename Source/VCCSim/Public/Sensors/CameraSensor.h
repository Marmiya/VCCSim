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
#include "CameraSensor.generated.h"


class FRGBCameraConfig : public FCameraConfig
{
public:

    FRGBCameraConfig()
    {
        FOV = 90.0f;
        Width = 512;
        Height = 512;
    }
};

class UInsMeshHolder;

UCLASS(ClassGroup = (VCCSIM), meta = (BlueprintSpawnableComponent))
class VCCSIM_API URGBCameraComponent : public UCameraBaseComponent
{
    GENERATED_BODY()

public:
    URGBCameraComponent() = default;
    virtual void Configure(const FSensorConfig& Config) override final;
    void SetIgnoreLidar(UInsMeshHolder* MeshHolder);

    FString CameraName;

    UFUNCTION(BlueprintCallable, Category = "RGBCamera")
    void CaptureRGBScene();
    UFUNCTION(BlueprintCallable, Category = "RGBCamera")
    void CaptureRGBSceneDeferred();
    
    // For grpc service
    void AsyncGetRGBImageData(TFunction<void(const TArray<FColor>&)> Callback);

    // UCameraBaseComponent interface
    virtual UTextureRenderTarget2D* GetRenderTarget() const override { return RenderTarget; }
    virtual ESensorType GetSensorType() const override { return ESensorType::RGBCamera; }

    const TArray<FColor>& GetCombinedData() const { return RGBData; }
    
protected:
    virtual void InitializeRenderTargets() override;

    virtual void SetCaptureComponent() const override;
    void ProcessRGBTexture(TFunction<void()> OnComplete);
    void ProcessRGBTextureParam(TFunction<void(const TArray<FColor>&)> OnComplete);

private:
    TArray<FColor> RGBData;

    template<typename CallbackType>
    void ProcessRGBTextureTemplate(CallbackType&& Callback);
};

template<typename CallbackType>
void URGBCameraComponent::ProcessRGBTextureTemplate(CallbackType&& Callback)
{
    if (!RenderTarget)
    {
        UE_LOG(LogTemp, Error, TEXT("CombinedRenderTarget is null!"));
        return;
    }

    if (RGBData.Num() != Width * Height)
    {
        RGBData.SetNumUninitialized(Width * Height);
    }

    struct FReadSurfaceContext
    {
        TArray<FColor>* RGBData;
        FIntRect Rect;
        FReadSurfaceDataFlags Flags;
    } Context { &RGBData, FIntRect(0, 0, Width, Height),
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
                *Context.RGBData,
                FReadSurfaceDataFlags()
            );

            if constexpr (std::is_invocable_v<std::decay_t<CallbackType>>)
            { (*SharedCallback)(); }
            else if constexpr (std::is_invocable_v<std::decay_t<CallbackType>, const TArray<FColor>&>)
            { (*SharedCallback)(*Context.RGBData); }
        }
    );
}