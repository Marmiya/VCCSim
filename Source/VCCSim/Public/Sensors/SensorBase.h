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
#include "Components/PrimitiveComponent.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Components/SceneCaptureComponent2D.h"
#include "SensorBase.generated.h"


enum class ESensorType : uint8
{
	Lidar = 3,
	RGBCamera = 4,
	DepthCamera = 5,
	NormalCamera = 6,
	SegmentationCamera = 7
};

class VCCSIM_API FSensorConfig
{
public:
	float RecordInterval = 0.2f;
};

struct VCCSIM_API FVCCSimOdom
{
	FVector Location = FVector::ZeroVector;
	FRotator Rotation = FRotator::ZeroRotator;
	FVector LinearVelocity = FVector::ZeroVector;
	FVector AngularVelocity = FVector::ZeroVector;
};

class VCCSIM_API FCameraConfig : public FSensorConfig
{
public:
	float FOV = 90.0f;
	int32 Width = 512;
	int32 Height = 512;
	bool bOrthographic = false;
	float OrthoWidth = 512.0f;
};

UCLASS(Abstract, ClassGroup = (VCCSIM), meta = (BlueprintSpawnableComponent))
class VCCSIM_API USensorBaseComponent : public UPrimitiveComponent
{
	GENERATED_BODY()

public:
	USensorBaseComponent();

	virtual ESensorType GetSensorType() const PURE_VIRTUAL(USensorBaseComponent::GetSensorType, return ESensorType::RGBCamera;);

	UFUNCTION(BlueprintCallable, Category = "Sensor")
	virtual void SetRecordState(bool RState) { RecordState = RState; }

	virtual void Configure(const FSensorConfig& Config){};

	UFUNCTION(BlueprintCallable, Category = "Sensor")
	virtual bool IsConfigured() const { return bBPConfigured; }

	UFUNCTION(BlueprintCallable, Category = "Sensor")
	virtual int32 GetSensorIndex() const { return SensorIndex; }

protected:
	virtual void BeginPlay() override;
	virtual void OnComponentCreated() override;

public:
	// If the sensor has been configured via Editor or BP
	// If true, cpp HUD will not override the configuration
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Sensor|Config")
	bool bBPConfigured = false;

	// This index is assgined in BP or Editor to identify multiple sensors of the same type
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Sensor|Config")
	int32 SensorIndex = 0;

protected:
	UPROPERTY()
	AActor* ParentActor = nullptr;

	bool RecordState = false;
};

UCLASS(Abstract, ClassGroup = (VCCSIM), meta = (BlueprintSpawnableComponent))
class VCCSIM_API UCameraBaseComponent : public USensorBaseComponent
{
	GENERATED_BODY()

public:
	UCameraBaseComponent();

	virtual void InitializeRenderTargets() PURE_VIRTUAL(UCameraBaseComponent::InitializeRenderTargets);
	virtual void SetCaptureComponent() const;
	virtual std::pair<int32, int32> GetImageSize() const { return {Width, Height}; }
	virtual void Configure(const FSensorConfig& Config) override {};
	virtual UTextureRenderTarget2D* GetRenderTarget() const PURE_VIRTUAL(UCameraBaseComponent::GetRenderTarget, return nullptr;);

	void ComputeIntrinsics();
	FMatrix44f GetCameraIntrinsics() const { return CameraIntrinsics; }

protected:
	bool CheckComponentAndRenderTarget() const;

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera|Config")
	float FOV = 90.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera|Config")
	int32 Width = 512;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera|Config")
	int32 Height = 512;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera|Config")
	bool bOrthographic = false;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera|Config", meta = (EditCondition = "bOrthographic"))
	float OrthoWidth = 512.0f;

protected:
	UPROPERTY()
	USceneCaptureComponent2D* CaptureComponent = nullptr;

	FMatrix44f CameraIntrinsics;
};