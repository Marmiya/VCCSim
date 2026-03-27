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
#include "RHIResources.h"
#include "RHIGPUReadback.h"
#include "BaseColorCamera.generated.h"

namespace BaseColorCameraDefaults
{
    constexpr float FOV = 90.0f;
    constexpr int32 Width = 512;
    constexpr int32 Height = 512;
}

class FBaseColorCameraConfig : public FCameraConfig
{
public:
    FBaseColorCameraConfig()
    {
        FOV   = BaseColorCameraDefaults::FOV;
        Width = BaseColorCameraDefaults::Width;
        Height = BaseColorCameraDefaults::Height;
    }
};

UCLASS(ClassGroup = (VCCSIM), meta = (BlueprintSpawnableComponent))
class VCCSIM_API UBaseColorCameraComponent : public UCameraBaseComponent
{
    GENERATED_BODY()

public:
    UBaseColorCameraComponent() = default;
    virtual void Configure(const FSensorConfig& Config) override final;

    UFUNCTION(BlueprintCallable, Category = "BaseColorCamera")
    void CaptureBaseColorScene();

    void AsyncGetBaseColorImageData(TFunction<void(const TArray<FColor>&)> Callback);

    virtual UTextureRenderTarget2D* GetRenderTarget() const override { return RenderTarget; }
    virtual ESensorType GetSensorType() const override { return ESensorType::BaseColorCamera; }

    const TArray<FColor>& GetCombinedData() const { return BaseColorData; }

protected:
    virtual void InitializeRenderTargets() override;
    virtual void SetCaptureComponent() const override;

    void ProcessBaseColorTexture(TFunction<void()> OnComplete);
    void ProcessBaseColorTextureParam(TFunction<void(const TArray<FColor>&)> OnComplete);

private:
    TArray<FColor> BaseColorData;

    template<typename CallbackType>
    void ProcessBaseColorTextureTemplate(CallbackType&& Callback);
};

template<typename CallbackType>
void UBaseColorCameraComponent::ProcessBaseColorTextureTemplate(CallbackType&& Callback)
{
    if (!RenderTarget)
    {
        UE_LOG(LogTemp, Error, TEXT("BaseColorRenderTarget is null!"));
        return;
    }

    if (BaseColorData.Num() != Width * Height)
    {
        BaseColorData.SetNumUninitialized(Width * Height);
    }

    auto SharedCallback = MakeShared<std::decay_t<CallbackType>>(std::forward<CallbackType>(Callback));

    UTextureRenderTarget2D* RT = RenderTarget;

    ENQUEUE_RENDER_COMMAND(ReadBaseColorSurfaceCommand)(
        [RT, DataPtr = &BaseColorData, CaptureWidth = Width, CaptureHeight = Height,
         SharedCallback](FRHICommandListImmediate& RHICmdList)
        {
            if (!RT) return;

            FTextureRenderTargetResource* RTRes = RT->GetRenderTargetResource();
            if (!RTRes) return;

            FRHITexture* TextureRHI = RTRes->GetRenderTargetTexture();
            if (!TextureRHI) return;

            FRHIGPUTextureReadback Readback(TEXT("BaseColorCameraReadback"));
            Readback.EnqueueCopy(RHICmdList, TextureRHI);
            RHICmdList.BlockUntilGPUIdle();

            int32 RowPitchInPixels;
            void* MappedData = Readback.Lock(RowPitchInPixels);
            if (MappedData)
            {
                const int32 SrcRowBytes = RowPitchInPixels * sizeof(FColor);
                const int32 DstRowBytes = CaptureWidth * sizeof(FColor);
                const uint8* Src = static_cast<const uint8*>(MappedData);
                uint8* Dst = reinterpret_cast<uint8*>(DataPtr->GetData());
                for (int32 Row = 0; Row < CaptureHeight; ++Row)
                {
                    FMemory::Memcpy(Dst + Row * DstRowBytes, Src + Row * SrcRowBytes, DstRowBytes);
                }
                Readback.Unlock();
            }

            if constexpr (std::is_invocable_v<std::decay_t<CallbackType>>)
            { (*SharedCallback)(); }
            else if constexpr (std::is_invocable_v<std::decay_t<CallbackType>, const TArray<FColor>&>)
            { (*SharedCallback)(*DataPtr); }
        }
    );
}
