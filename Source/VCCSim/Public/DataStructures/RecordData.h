#pragma once

struct FSensorData
{
	double Timestamp;
	virtual ~FSensorData() = default;
};

struct FPoseData final : public FSensorData
{
	FVector Location;
	FQuat Quaternion;
};

struct FLiDARData final : public FSensorData
{
	int32 SensorIndex;
	TArray<FVector3f> Data;
};

struct FRGBCameraData final : public FSensorData
{
	int32 Width;
	int32 Height;
	int32 SensorIndex;
	TArray<FColor> RGBData;
};

struct FDepthCameraData final : public FSensorData
{
	int32 Width;
	int32 Height;
	int32 SensorIndex;
	TArray<float> DepthData;
};

struct FSegmentationCameraData final : public FSensorData
{
	int32 Width;
	int32 Height;
	int32 SensorIndex;
	TArray<FColor> Data;
};

struct FNormalCameraData final : public FSensorData
{
	int32 Width;
	int32 Height;
	int32 SensorIndex;
	TArray<FFloat16Color> Data;
};

struct FBaseColorCameraData final : public FSensorData
{
	int32 Width;
	int32 Height;
	int32 SensorIndex;
	TArray<FColor> Data;
};

struct FMaterialPropertiesCameraData final : public FSensorData
{
	int32 Width;
	int32 Height;
	int32 SensorIndex;
	TArray<FColor> Data;
};