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
#include "SegmentCamera.generated.h"

class ARecorder;

class FSegmentationCameraConfig : public FCameraConfig
{
public:
    FSegmentationCameraConfig()
    {
        Width = 512;
        Height = 512;
    }
};

UCLASS(ClassGroup = (VCCSIM), meta = (BlueprintSpawnableComponent))
class VCCSIM_API USegmentationCameraComponent : public UCameraBaseComponent
{
    GENERATED_BODY()

public:
    USegmentationCameraComponent();
    void RConfigure(const FSegmentationCameraConfig& Config, ARecorder* Recorder);

    virtual ESensorType GetSensorType() const override { return ESensorType::SegmentationCamera; }
    int32 GetCameraIndex() const { return GetSensorIndex(); }

    FString CameraName;

    void ProcessSegmentationTextureAsyncRaw(TFunction<void()> OnComplete);
    void ProcessSegmentationTextureAsync(TFunction<void(const TArray<FColor>&)> OnComplete);

    UFUNCTION(BlueprintCallable, Category = "SegmentationCamera")
    void CaptureSegmentationScene();

    // For GRPC call
    void AsyncGetSegmentationImageData(TFunction<void(const TArray<FColor>&)> Callback);

protected:
    virtual void InitializeRenderTargets() override;
    virtual UTextureRenderTarget2D* GetRenderTarget() const override { return SegmentationRenderTarget; }
    virtual void SetCaptureComponent() const override;
    virtual void OnRecordTick() override;

public:
    // Segmentation-specific properties
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SegmentationCamera|Config")
    UMaterialInterface* SegmentationMaterial = Cast<UMaterialInterface>(
        StaticLoadObject(UMaterialInterface::StaticClass(), nullptr,
            TEXT("/VCCSim/Materials/M_Segmentation.M_Segmentation")));

    UPROPERTY()
    UTextureRenderTarget2D* SegmentationRenderTarget = nullptr;

private:
    void ExecuteCaptureOnGameThread();

    struct FReadSurfaceContext
    {
        TArray<FColor>* OutData;
        FTextureRenderTargetResource* RenderTarget;
        FIntRect Rect;
        FReadSurfaceDataFlags Flags;
    };

    TArray<FColor> SegmentationData;
    FCriticalSection DataLock;
    bool Dirty = false;
};