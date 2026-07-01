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
#include "RGBLinearCamera.generated.h"

namespace RGBLinearCameraDefaults
{
	constexpr int32 Width = 1920;
	constexpr int32 Height = 1080;
}

class FRGBLinearCameraConfig : public FCameraConfig
{
public:
    FRGBLinearCameraConfig()
    {
        Width = RGBLinearCameraDefaults::Width;
        Height = RGBLinearCameraDefaults::Height;
    }
};

UCLASS(ClassGroup = (VCCSIM), meta = (BlueprintSpawnableComponent))
class VCCSIM_API URGBLinearCameraComponent : public UCameraBaseComponent
{
    GENERATED_BODY()

public:
    URGBLinearCameraComponent();
    virtual void Configure(const FSensorConfig& Config) override final;

    virtual UTextureRenderTarget2D* GetRenderTarget() const override { return RenderTarget; }

    void AsyncGetRGBLinearImageData(TFunction<void(const TArray<FFloat16Color>&)> Callback);

    virtual ESensorType GetSensorType() const override { return ESensorType::RGBLinearCamera; }

protected:
    virtual void InitializeRenderTargets() override;
    virtual void SetCaptureComponent() const override;
};
