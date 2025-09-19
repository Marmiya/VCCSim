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
#include "NormalCamera.generated.h"

class ARecorder;

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
class VCCSIM_API UNormalCameraComponent : public UCameraBaseComponent
{
    GENERATED_BODY()

public:
    UNormalCameraComponent();
    void RConfigure(const FNormalCameraConfig& Config, ARecorder* Recorder);

    virtual ESensorType GetSensorType() const override { return ESensorType::NormalCamera; }
    int32 GetCameraIndex() const { return GetSensorIndex(); }

    UFUNCTION(BlueprintCallable, Category = "NormalCamera")
    void CaptureScene();

    // High precision normal data access
    void AsyncGetNormalImageData(TFunction<void(const TArray<FLinearColor>&)> Callback);

protected:
    virtual void InitializeRenderTargets() override;
    virtual UTextureRenderTarget2D* GetRenderTarget() const override { return NormalRenderTarget; }
    virtual void SetCaptureComponent() const override;
    virtual void OnRecordTick() override;

    void ProcessNormalTexture(TFunction<void(const TArray<FLinearColor>&)> OnComplete);
    TArray<FLinearColor> GetNormalImage();

public:
    UPROPERTY()
    UTextureRenderTarget2D* NormalRenderTarget = nullptr;

private:
    // Store high precision normal data
    TArray<FLinearColor> NormalData;
    bool Dirty = false;
};
