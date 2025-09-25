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
	TArray<FVector3f> Data;
};

struct FRGBDCameraData final : public FSensorData
{
	int32 Width;
	int32 Height;
	int32 SensorIndex;
	TArray<FColor> RGBData;
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
	TArray<FColor> Data;
};