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
#include "DepthCamera.generated.h"

struct FDCPoint
{
    FVector Location;
    FDCPoint() : Location(FVector::ZeroVector){}
};

class FDepthCameraConfig: public FCameraConfig
{
public:
    float MaxRange = 2000.0f;
    float MinRange = .0f;
};

UCLASS(ClassGroup = (VCCSIM), meta = (BlueprintSpawnableComponent))
class VCCSIM_API UDepthCameraComponent : public UCameraBaseComponent, public ISensorDataProvider
{
    GENERATED_BODY()

public:
    UDepthCameraComponent();
    virtual void Configure(const FSensorConfig& Config) override final;

    UFUNCTION(BlueprintCallable, Category = "DepthCamera")
    void CaptureDepthScene();
    UFUNCTION(BlueprintCallable, Category = "DepthCamera")
    void CaptureDepthSceneAndProcess();
    UFUNCTION(BlueprintCallable, Category = "DepthCamera")
    void VisualizePointCloud();
    TArray<FDCPoint> GeneratePointCloud();

    // For grpc server
    void AsyncGetPointCloudData(TFunction<void()> Callback);
    void AsyncGetDepthImageData(TFunction<void(const TArray<FFloat16Color>&)> Callback);

    // ISensorDataProvider interface
    virtual TFuture<FSensorDataPacket> CaptureDataAsync() override;
    virtual ESensorType GetSensorType() const override { return ESensorType::DepthCamera; }
    virtual AActor* GetOwnerActor() const override { return ParentActor; }

    // RDG interface
    virtual void ContributeToRDGPass(FSensorViewInfo& OutViewInfo) override;
    virtual int32 GetMRTSlot() const override { return 1; }
    
protected:
    virtual void InitializeRenderTargets() override;
    virtual UTextureRenderTarget2D* GetRenderTarget() const override { return DepthRenderTarget; }
    virtual void SetCaptureComponent() const override;

    void ProcessDepthTexture(TFunction<void()> OnComplete);
    void ProcessDepthTextureParam(TFunction<void(const TArray<FFloat16Color>&)> OnComplete);

    TArray<float> GetDepthImage();

public:
    // Depth-specific properties
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "DepthCamera|Config")
    float MaxRange = 2000.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "DepthCamera|Config")
    float MinRange = 0.0f;

    UPROPERTY()
    UTextureRenderTarget2D* DepthRenderTarget = nullptr;

    TArray<FDCPoint> PointCloudData;

private:
    TArray<FFloat16Color> DepthData;
    bool Dirty = false;

    template<typename CallbackType>
    void ProcessDepthTextureTemplate(CallbackType&& Callback);
};


template<typename CallbackType>
void UDepthCameraComponent::ProcessDepthTextureTemplate(CallbackType&& Callback)
{
    if (!DepthRenderTarget) { UE_LOG(LogTemp, Error, TEXT("DepthRenderTarget is null!")); return; }

    if (DepthData.Num() != Width * Height)
    {
        DepthData.SetNumUninitialized(Width * Height);
    }

    struct FReadSurfaceContext
    {
        TArray<FFloat16Color>* OutData;
        FIntRect Rect;
        FReadSurfaceDataFlags Flags;
    } Context { &DepthData, FIntRect(0, 0, Width, Height),
                FReadSurfaceDataFlags(RCM_MinMax, CubeFace_MAX) };

    auto SharedCallback = MakeShared<std::decay_t<CallbackType>>(std::forward<CallbackType>(Callback));

    UTextureRenderTarget2D* RT = DepthRenderTarget;

    ENQUEUE_RENDER_COMMAND(ReadSurfaceCommand)(
        [RT, Context, SharedCallback](FRHICommandListImmediate& RHICmdList)
        {
            if (!RT) return;

            FTextureRenderTargetResource* RTRes = RT->GetRenderTargetResource();
            if (!RTRes) return;

            RHICmdList.ReadSurfaceFloatData(
                RTRes->GetRenderTargetTexture(),
                Context.Rect,
                *Context.OutData,
                ECubeFace::CubeFace_PosX,
                0,
                0
            );

            if constexpr (std::is_invocable_v<std::decay_t<CallbackType>>)
            { (*SharedCallback)(); }
            else if constexpr (std::is_invocable_v<std::decay_t<CallbackType>, const TArray<FFloat16Color>&>)
            { (*SharedCallback)(*Context.OutData); }
        }
    );
}