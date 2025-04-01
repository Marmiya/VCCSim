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

#include <map>
#include <string>

class FGrpcServerTask;
class ULidarComponent;
class UDepthCameraComponent;
class URGBCameraComponent;
class USegmentationCameraComponent;
class UFMeshManager;
struct FVCCSimConfig;

struct FRobotGrpcMaps
{
	struct FRobotComponentMaps
	{
		std::map<std::string, ULidarComponent*> RLMap;
		std::map<std::string, UDepthCameraComponent*> RDCMap;
		std::map<std::string, URGBCameraComponent*> RRGBCMap;
		std::map<std::string, USegmentationCameraComponent*> RSegMap;
	};

	struct FRobotMaps
	{
		std::map<std::string, AActor*> DroneMap;
		std::map<std::string, AActor*> CarMap;
		std::map<std::string, AActor*> FlashMap;
	};

	FRobotComponentMaps RCMaps;
	FRobotMaps RMaps;
};


extern FAsyncTask<FGrpcServerTask>* Server_Task;

void RunServer(const FVCCSimConfig& Config, AActor* Holder,
	const FRobotGrpcMaps& RGrpcMaps, UFMeshManager* MeshManager);

void ShutdownServer();
