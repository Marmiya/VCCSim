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
#include "NormalCamera.generated.h"

class FNormalCameraConfig: public FCameraConfig
{
public:
    FNormalCameraConfig()
    {
        Width = 1920;
        Height = 1080;
    }
};

UCLASS(ClassGroup = (VCCSIM), meta = (BlueprintSpawnableComponent))
class VCCSIM_API UNormalCameraComponent : public UCameraBaseComponent, public ISensorDataProvider
{
    GENERATED_BODY()

public:
    UNormalCameraComponent();
    virtual void Configure(const FSensorConfig& Config) override final;

    UFUNCTION(BlueprintCallable, Category = "NormalCamera")
    void CaptureNormalScene();
    UFUNCTION(BlueprintCallable, Category = "NormalCamera")
    void CaptureNormalSceneAndProcess();

    // For grpc server
    void AsyncGetNormalImageData(TFunction<void(const TArray<FLinearColor>&)> Callback);

    // ISensorDataProvider interface
    virtual TFuture<FSensorDataPacket> CaptureDataAsync() override;
    virtual ESensorType GetSensorType() const override { return ESensorType::NormalCamera; }
    virtual AActor* GetOwnerActor() const override { return ParentActor; }

protected:
    virtual void InitializeRenderTargets() override;
    virtual UTextureRenderTarget2D* GetRenderTarget() const override { return NormalRenderTarget; }
    virtual void SetCaptureComponent() const override;

    void ProcessNormalTexture(TFunction<void()> OnComplete);
    void ProcessNormalTextureParam(TFunction<void(const TArray<FLinearColor>&)> OnComplete);

public:
    UPROPERTY()
    UTextureRenderTarget2D* NormalRenderTarget = nullptr;

private:    
    TArray<FLinearColor> NormalData;
    bool Dirty = false;

    template<typename CallbackType>
    void ProcessNormalTextureTemplate(CallbackType&& Callback);
};

template<typename CallbackType>
void UNormalCameraComponent::ProcessNormalTextureTemplate(CallbackType&& Callback)
{
    if (!NormalRenderTarget) { UE_LOG(LogTemp, Error, TEXT("NormalRenderTarget is null!")); return; }

    if (NormalData.Num() != Width * Height)
    {
        NormalData.SetNumUninitialized(Width * Height);
    }

    struct FReadSurfaceContext
    {
        TArray<FLinearColor>* OutData;
        FIntRect Rect;
    } Context { &NormalData, FIntRect(0, 0, Width, Height) };

    auto SharedCallback = MakeShared<std::decay_t<CallbackType>>(std::forward<CallbackType>(Callback));

    UTextureRenderTarget2D* RT = NormalRenderTarget;

    ENQUEUE_RENDER_COMMAND(ReadNormalSurfaceCommand)(
        [RT, Context, SharedCallback](FRHICommandListImmediate& RHICmdList)
        {
            if (!RT) return;

            FTextureRenderTargetResource* RTRes = RT->GetRenderTargetResource();
            if (!RTRes) return;

            RHICmdList.ReadSurfaceData(
                RTRes->GetRenderTargetTexture(),
                Context.Rect,
                *Context.OutData,
                FReadSurfaceDataFlags()
            );

            if constexpr (std::is_invocable_v<std::decay_t<CallbackType>>)
            { (*SharedCallback)(); }
            else if constexpr (std::is_invocable_v<std::decay_t<CallbackType>, const TArray<FLinearColor>&>)
            { (*SharedCallback)(*Context.OutData); }
        }
    );
}
