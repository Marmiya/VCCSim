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

class FRGBCameraConfig : public FCameraConfig
{
public:
    float MaxRange = 10000.0f;

    FRGBCameraConfig()
    {
        FOV = 90.0f;
        Width = 512;
        Height = 512;
        bOrthographic = false;
        OrthoWidth = 512.0f;
    }
};

UCLASS(ClassGroup = (VCCSIM), meta = (BlueprintSpawnableComponent))
class VCCSIM_API URGBCameraComponent : public UCameraBaseComponent, public ISensorDataProvider
{
    GENERATED_BODY()

public:
    URGBCameraComponent();
    virtual void Configure(const FSensorConfig& Config) override final;
    void SetIgnoreLidar(UInsMeshHolder* MeshHolder);
    FString CameraName;

    UFUNCTION(BlueprintCallable, Category = "RGBCamera")
    void CaptureRGBScene();

    // For grpc service
    void AsyncGetRGBImageData(TFunction<void(const TArray<FColor>&)> Callback);

    // ISensorDataProvider interface
    virtual TFuture<FSensorDataPacket> CaptureDataAsync() override;
    virtual UTextureRenderTarget2D* GetRenderTarget() const override { return RGBRenderTarget; }
    virtual FIntPoint GetResolution() const override { return FIntPoint(Width, Height); }
    virtual ESensorType GetSensorType() const override { return ESensorType::RGBCamera; }
    virtual AActor* GetOwnerActor() const override { return ParentActor; }

    // RDG interface
    virtual void ContributeToRDGPass(FSensorViewInfo& OutViewInfo) override;
    virtual int32 GetMRTSlot() const override { return 0; }
        
protected:
    virtual void InitializeRenderTargets() override;

    virtual void SetCaptureComponent() const override;
    void ProcessRGBTexture(TFunction<void()> OnComplete);
    void ProcessRGBTextureParam(TFunction<void(const TArray<FColor>&)> OnComplete);

public:
    UPROPERTY()
    UTextureRenderTarget2D* RGBRenderTarget = nullptr;

private:
    TArray<FColor> RGBData;

    template<typename CallbackType>
    void ProcessRGBTextureTemplate(CallbackType&& Callback);
};

template<typename CallbackType>
void URGBCameraComponent::ProcessRGBTextureTemplate(CallbackType&& Callback)
{
    if (!RGBRenderTarget) { UE_LOG(LogTemp, Error, TEXT("RGBRenderTarget is null!")); return; }

    if (RGBData.Num() != Width * Height)
    {
        RGBData.SetNumUninitialized(Width * Height);
    }

    struct FReadSurfaceContext
    {
        TArray<FColor>* OutData;
        FIntRect Rect;
        FReadSurfaceDataFlags Flags;
    } Context { &RGBData, FIntRect(0, 0, Width, Height),
                FReadSurfaceDataFlags(RCM_UNorm, CubeFace_MAX) };

    auto SharedCallback = MakeShared<std::decay_t<CallbackType>>(std::forward<CallbackType>(Callback));

    UTextureRenderTarget2D* RT = RGBRenderTarget;

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

            if constexpr (std::is_invocable_v<std::decay_t<CallbackType>>)
            { (*SharedCallback)(); }
            else if constexpr (std::is_invocable_v<std::decay_t<CallbackType>, const TArray<FColor>&>)
            { (*SharedCallback)(*Context.OutData); }
        }
    );
}