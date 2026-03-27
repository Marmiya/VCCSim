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

DEFINE_LOG_CATEGORY_STATIC(LogSensorFactory, Log, All);

#include "Sensors/SensorFactory.h"
#include "Sensors/LidarSensor.h"
#include "Sensors/RGBCamera.h"
#include "Sensors/DepthCamera.h"
#include "Sensors/BaseColorCamera.h"
#include "Sensors/MaterialPropertiesCamera.h"

USceneComponent* FSensorFactory::CreateSensor(
	ESensorType SensorType, AActor* Owner, FName Name, const FSensorConfig& Config)
{
	if (!Owner)
	{
		UE_LOG(LogSensorFactory, Error, TEXT("No owner found!"));
		return nullptr;
	}

	USceneComponent* Sensor = nullptr;

	switch (SensorType)
	{
	case ESensorType::Lidar:
		{
			const FLiDARConfig* LidarConfig =
				static_cast<const FLiDARConfig*>(&Config);
			if (!LidarConfig)
			{
				UE_LOG(LogSensorFactory, Error, TEXT("FSensorFactory::CreateSensor: "
								"Invalid config type for Lidar sensor!"));
				return nullptr;
			}
            
			if (auto LidarSensor = NewObject<ULidarComponent>(Owner, Name))
			{
				Sensor = LidarSensor;
			}
			else
			{
				UE_LOG(LogSensorFactory, Error, TEXT("FSensorFactory::CreateSensor: "
								"Failed to create Lidar sensor!"));
			}
			break;
		}
	case ESensorType::RGBCamera:
		{
			const FRGBCameraConfig* CameraConfig =
				static_cast<const FRGBCameraConfig*>(&Config);
			if (!CameraConfig)
			{
				UE_LOG(LogSensorFactory, Error, TEXT("FSensorFactory::CreateSensor: "
								"Invalid config type for RGBD Camera sensor!"));
				return nullptr;
			}

			if (auto CameraSensor = NewObject<URGBCameraComponent>(Owner, Name))
			{
				Sensor = CameraSensor;
			}
			else
			{
				UE_LOG(LogSensorFactory, Error, TEXT("FSensorFactory::CreateSensor: "
								"Failed to create RGBD Camera sensor!"));
			}

			break;
		}
	case ESensorType::DepthCamera:
		{
			const FDepthCameraConfig* CameraConfig =
				static_cast<const FDepthCameraConfig*>(&Config);
			if (!CameraConfig)
			{
				UE_LOG(LogSensorFactory, Error, TEXT("FSensorFactory::CreateSensor: "
								"Invalid config type for Depth Camera sensor!"));
				return nullptr;
			}

			if (auto CameraSensor = NewObject<UDepthCameraComponent>(Owner, Name))
			{
				Sensor = CameraSensor;
			}
			else
			{
				UE_LOG(LogSensorFactory, Error, TEXT("FSensorFactory::CreateSensor: "
								"Failed to create Depth Camera sensor!"));
			}

			break;
		}
	case ESensorType::MaterialPropertiesCamera:
		{
			const FMaterialPropertiesCameraConfig* CameraConfig =
				static_cast<const FMaterialPropertiesCameraConfig*>(&Config);
			if (!CameraConfig)
			{
				UE_LOG(LogSensorFactory, Error, TEXT("FSensorFactory::CreateSensor: "
								"Invalid config type for MaterialProperties Camera sensor!"));
				return nullptr;
			}

			if (auto CameraSensor = NewObject<UMaterialPropertiesCameraComponent>(Owner, Name))
			{
				Sensor = CameraSensor;
			}
			else
			{
				UE_LOG(LogSensorFactory, Error, TEXT("FSensorFactory::CreateSensor: "
								"Failed to create MaterialProperties Camera sensor!"));
			}

			break;
		}
	case ESensorType::BaseColorCamera:
		{
			const FBaseColorCameraConfig* CameraConfig =
				static_cast<const FBaseColorCameraConfig*>(&Config);
			if (!CameraConfig)
			{
				UE_LOG(LogSensorFactory, Error, TEXT("FSensorFactory::CreateSensor: "
								"Invalid config type for BaseColor Camera sensor!"));
				return nullptr;
			}

			if (auto CameraSensor = NewObject<UBaseColorCameraComponent>(Owner, Name))
			{
				Sensor = CameraSensor;
			}
			else
			{
				UE_LOG(LogSensorFactory, Error, TEXT("FSensorFactory::CreateSensor: "
								"Failed to create BaseColor Camera sensor!"));
			}

			break;
		}
	default:
		{
			UE_LOG(LogSensorFactory, Error, TEXT("FSensorFactory::CreateSensor: "
							"Invalid sensor type!"));
		}
	}

	if (Sensor)
	{
		Sensor->RegisterComponent();
	}

	return Sensor;
}