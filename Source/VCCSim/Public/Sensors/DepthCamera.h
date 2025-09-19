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

class ARecorder;

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
class VCCSIM_API UDepthCameraComponent : public UCameraBaseComponent
{
    GENERATED_BODY()

public:
    UDepthCameraComponent();
    void RConfigure(const FDepthCameraConfig& Config, ARecorder* Recorder);

    virtual ESensorType GetSensorType() const override { return ESensorType::DepthCamera; }
    int32 GetCameraIndex() const { return GetSensorIndex(); }

    UFUNCTION(BlueprintCallable, Category = "DepthCamera")
    void CaptureDepthScene();
    UFUNCTION(BlueprintCallable, Category = "DepthCamera")
    void OnlyCaptureDepthScene();
    UFUNCTION(BlueprintCallable, Category = "DepthCamera")
    void VisualizePointCloud();
    TArray<FDCPoint> GeneratePointCloud();

    // For grpc server
    void AsyncGetPointCloudData(TFunction<void()> Callback);
    void AsyncGetDepthImageData(TFunction<void(const TArray<FFloat16Color>&)> Callback);

protected:
    virtual void InitializeRenderTargets() override;
    virtual UTextureRenderTarget2D* GetRenderTarget() const override { return DepthRenderTarget; }
    virtual void SetCaptureComponent() const override;
    virtual void OnRecordTick() override;

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
};