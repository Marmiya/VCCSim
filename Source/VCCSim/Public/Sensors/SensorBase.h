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
#include "Components/SceneComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "DataStructures/RecordData.h"
#include "SensorBase.generated.h"


enum class ESensorType : uint8
{
	Lidar = 3,
	RGBCamera = 4,
	DepthCamera = 5,
	NormalCamera = 6,
	SegmentationCamera = 7,
	BaseColorCamera = 8,
	MaterialPropertiesCamera = 9
};

namespace SensorDefaults
{
	constexpr float RecordInterval = 0.2f;
}

class VCCSIM_API FSensorConfig
{
public:
	float RecordInterval = SensorDefaults::RecordInterval;
};

struct VCCSIM_API FVCCSimOdom
{
	FVector Location = FVector::ZeroVector;
	FRotator Rotation = FRotator::ZeroRotator;
	FVector LinearVelocity = FVector::ZeroVector;
	FVector AngularVelocity = FVector::ZeroVector;
};

namespace CameraDefaults
{
	constexpr float FOV = 90.0f;
	constexpr int32 Width = 512;
	constexpr int32 Height = 512;
}

class VCCSIM_API FCameraConfig : public FSensorConfig
{
public:
	float FOV = CameraDefaults::FOV;
	int32 Width = CameraDefaults::Width;
	int32 Height = CameraDefaults::Height;
};

struct VCCSIM_API FSensorDataPacket
{
	ESensorType Type;
	int32 SensorIndex;
	AActor* OwnerActor;
	TSharedPtr<FSensorData> Data;
	bool bValid;

	FSensorDataPacket()
		: Type(ESensorType::RGBCamera)
		, SensorIndex(0)
		, OwnerActor(nullptr)
		, Data(nullptr)
		, bValid(false)
	{
	}
};

UCLASS(Abstract, ClassGroup = (VCCSIM), meta = (BlueprintSpawnableComponent))
class VCCSIM_API USensorBaseComponent : public USceneComponent
{
	GENERATED_BODY()

public:
	USensorBaseComponent();

	virtual ESensorType GetSensorType() const PURE_VIRTUAL(
		USensorBaseComponent::GetSensorType, return ESensorType::RGBCamera;);

	UFUNCTION(BlueprintCallable, Category = "Sensor")
	virtual void SetRecordState(bool RState) { RecordState = RState; }

	virtual void Configure(const FSensorConfig& Config){};

	UFUNCTION(BlueprintCallable, Category = "Sensor")
	virtual bool IsConfigured() const { return bBPConfigured; }

	UFUNCTION(BlueprintCallable, Category = "Sensor")
	virtual int32 GetSensorIndex() const { return SensorIndex; }

	UFUNCTION(BlueprintCallable, Category = "Sensor")
	virtual float GetRecordInterval() const { return RecordInterval; }

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
	bool RecordState = false;
	float RecordInterval = SensorDefaults::RecordInterval;
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

	void WarmupCapture();
	void ComputeIntrinsics();
	FMatrix44f GetCameraIntrinsics() const { return CameraIntrinsics; }
	double GetLastCaptureTimestamp() const { return LastCaptureTimestamp; }

	virtual UTextureRenderTarget2D* GetRenderTarget() const PURE_VIRTUAL(UCameraBaseComponent::GetRenderTarget, return nullptr;);
	virtual FIntPoint GetResolution() const { return FIntPoint(Width, Height); }

protected:
	bool CheckComponentAndRenderTarget() const;

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera|Config")
	float FOV = CameraDefaults::FOV;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera|Config")
	int32 Width = CameraDefaults::Width;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera|Config")
	int32 Height = CameraDefaults::Height;

protected:
	UPROPERTY(Transient)
	USceneCaptureComponent2D* CaptureComponent = nullptr;
	UPROPERTY(Transient)
	UTextureRenderTarget2D* RenderTarget = nullptr;

	FMatrix44f CameraIntrinsics;
	double LastCaptureTimestamp = 0.0;
};