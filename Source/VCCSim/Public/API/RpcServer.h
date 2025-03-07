// MIT License
// 
// Copyright (c) 2025 Mingyang Wang
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <map>
#include <string>

class FGrpcServerTask;
class ULidarComponent;
class UDepthCameraComponent;
class URGBCameraComponent;
class UFMeshManager;
struct FVCCSimConfig;

struct FRobotGrpcMaps
{
	struct FRobotComponentMaps
	{
		std::map<std::string, ULidarComponent*> RLMap;
		std::map<std::string, UDepthCameraComponent*> RDCMap;
		std::map<std::string, URGBCameraComponent*> RRGBCMap;
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
