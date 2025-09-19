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

class ARecorder;
class UInsMeshHolder;

class FRGBCameraConfig : public FCameraConfig
{
public:
    FRGBCameraConfig()
    {
        FOV = 90.0f;
        Width = 512;
        Height = 512;
        bOrthographic = false;
        OrthoWidth = 512.0f;
    }
};

DECLARE_DYNAMIC_DELEGATE_TwoParams(
    FKeyPointCaptured, const FTransform&, Pose, const FString&, Name);

UCLASS(ClassGroup = (VCCSIM), meta = (BlueprintSpawnableComponent))
class VCCSIM_API URGBCameraComponent : public UCameraBaseComponent
{
    GENERATED_BODY()

public:
    URGBCameraComponent();
    void RConfigure(const FRGBCameraConfig& Config, ARecorder* Recorder);
    void SetIgnoreLidar(UInsMeshHolder* MeshHolder);

    virtual ESensorType GetSensorType() const override { return ESensorType::RGBCamera; }

    int32 GetCameraIndex() const { return GetSensorIndex(); }
    FString CameraName;

    void ProcessRGBTextureAsyncRaw(TFunction<void()> OnComplete);
    void ProcessRGBTextureAsync(TFunction<void(const TArray<FColor>&)> OnComplete);

    UFUNCTION(BlueprintCallable, Category = "RGBCamera")
    void CaptureRGBScene();

    void AsyncGetRGBImageData(TFunction<void(const TArray<FColor>&)> Callback);
        
protected:
    virtual void InitializeRenderTargets() override;
    virtual UTextureRenderTarget2D* GetRenderTarget() const override { return RGBRenderTarget; }
    virtual void SetCaptureComponent() const override;
    virtual void OnRecordTick() override;

public:

    FKeyPointCaptured OnKeyPointCaptured;

    UPROPERTY()
    UTextureRenderTarget2D* RGBRenderTarget = nullptr;

private:
    void ExecuteCaptureOnGameThread();

    struct FReadSurfaceContext
    {
        TArray<FColor>* OutData;
        FTextureRenderTargetResource* RenderTarget;
        FIntRect Rect;
        FReadSurfaceDataFlags Flags;
    };

    TArray<FColor> RGBData;
};