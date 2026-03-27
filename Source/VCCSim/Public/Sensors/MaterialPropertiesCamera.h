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
#include "MaterialPropertiesCamera.generated.h"

namespace MaterialPropertiesCameraDefaults
{
    constexpr float FOV    = 90.0f;
    constexpr int32 Width  = 512;
    constexpr int32 Height = 512;
}

class FMaterialPropertiesCameraConfig : public FCameraConfig
{
public:
    FMaterialPropertiesCameraConfig()
    {
        FOV    = MaterialPropertiesCameraDefaults::FOV;
        Width  = MaterialPropertiesCameraDefaults::Width;
        Height = MaterialPropertiesCameraDefaults::Height;
    }
};

UCLASS(ClassGroup = (VCCSIM), meta = (BlueprintSpawnableComponent))
class VCCSIM_API UMaterialPropertiesCameraComponent : public UCameraBaseComponent
{
    GENERATED_BODY()

public:
    UMaterialPropertiesCameraComponent() = default;
    virtual void Configure(const FSensorConfig& Config) override final;

    UFUNCTION(BlueprintCallable, Category = "MaterialPropertiesCamera")
    void CaptureMaterialPropertiesScene();

    void AsyncGetMaterialPropertiesImageData(TFunction<void(const TArray<FColor>&)> Callback);

    virtual UTextureRenderTarget2D* GetRenderTarget() const override { return RenderTarget; }
    virtual ESensorType GetSensorType() const override { return ESensorType::MaterialPropertiesCamera; }

    const TArray<FColor>& GetCombinedData() const { return MaterialPropertiesData; }

protected:
    virtual void InitializeRenderTargets() override;
    virtual void SetCaptureComponent() const override;

    void ProcessMaterialPropertiesTexture(TFunction<void()> OnComplete);
    void ProcessMaterialPropertiesTextureParam(TFunction<void(const TArray<FColor>&)> OnComplete);

private:
    TArray<FColor> MaterialPropertiesData;

    template<typename CallbackType>
    void ProcessMaterialPropertiesTextureTemplate(CallbackType&& Callback);
};

template<typename CallbackType>
void UMaterialPropertiesCameraComponent::ProcessMaterialPropertiesTextureTemplate(CallbackType&& Callback)
{
    if (!RenderTarget)
    {
        UE_LOG(LogTemp, Error, TEXT("MaterialPropertiesRenderTarget is null!"));
        return;
    }

    if (MaterialPropertiesData.Num() != Width * Height)
    {
        MaterialPropertiesData.SetNumUninitialized(Width * Height);
    }

    auto SharedCallback = MakeShared<std::decay_t<CallbackType>>(std::forward<CallbackType>(Callback));

    UTextureRenderTarget2D* RT = RenderTarget;

    ENQUEUE_RENDER_COMMAND(ReadMaterialPropertiesSurfaceCommand)(
        [RT, DataPtr = &MaterialPropertiesData, CaptureWidth = Width, CaptureHeight = Height,
         SharedCallback](FRHICommandListImmediate& RHICmdList)
        {
            if (!RT) return;

            FTextureRenderTargetResource* RTRes = RT->GetRenderTargetResource();
            if (!RTRes) return;

            FRHITexture* TextureRHI = RTRes->GetRenderTargetTexture();
            if (!TextureRHI) return;

            FRHIGPUTextureReadback Readback(TEXT("MaterialPropertiesCameraReadback"));
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
