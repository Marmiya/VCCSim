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

#include "API/RpcServer.h"
#include "API/GRPCCall.h"
#include "Sensors/LidarSensor.h"
#include "Sensors/DepthCamera.h"
#include "Pawns/DronePawn.h"
#include "Pawns/CarPawn.h"
#include "Pawns/FlashPawn.h"
#include "Utils/MeshHandlerComponent.h"
#include "Async/AsyncWork.h"
#include "Utils/ConfigParser.h"
#include <iostream>
#include <memory>


using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;


FAsyncTask<FGrpcServerTask>* Server_Task = nullptr;
std::atomic<bool> ShutdownRequested = false;


class FGrpcServerTask : public FNonAbandonableTask
{
public:
    FGrpcServerTask(
    	const FVCCSimConfig& Config, UMeshHandlerComponent* MeshComponent,
    	UInsMeshHolder* InstancedMeshHolder, UFMeshManager* MeshManager,
    	FRobotGrpcMaps RobotGrpcMap, ARecorder* ARecorder)
        : Config(Config), MeshComponent(MeshComponent),
		  InstancedMeshHolder(InstancedMeshHolder), MeshManager(MeshManager),
		  RGrpcMaps(RobotGrpcMap), Recorder(ARecorder) {}

    FORCEINLINE TStatId GetStatId() const
    {
        RETURN_QUICK_DECLARE_CYCLE_STAT(FGrpcServerTask, STATGROUP_ThreadPoolAsyncTasks);
    }

    void DoWork()
    {
    	grpc::ServerBuilder Builder;
    	Builder.AddListeningPort(Config.VCCSim.Server, grpc::InsecureServerCredentials());
    	
    	VCCSim::DroneService::AsyncService DroneService;
        VCCSim::CarService::AsyncService CarService;
    	VCCSim::LidarService::AsyncService LidarService;
    	VCCSim::FlashService::AsyncService FlashService;
    	VCCSim::DepthCameraService::AsyncService DepthCameraService;
    	VCCSim::RGBCameraService::AsyncService RGBCameraService;
    	VCCSim::MeshService::AsyncService MeshService;
    	VCCSim::PointCloudService::AsyncService PointCloudService;
    	VCCSim::RecordingService::AsyncService RecordingService;
    	

    	if (!RGrpcMaps.RMaps.DroneMap.empty())
		{
    		Builder.RegisterService(&DroneService);
		}
    	if (!RGrpcMaps.RMaps.CarMap.empty())
    	{
			Builder.RegisterService(&CarService);
    	}
    	if (!RGrpcMaps.RMaps.FlashMap.empty())
		{
    		Builder.RegisterService(&FlashService);
		}
    	if (!RGrpcMaps.RCMaps.RLMap.empty())
    	{
    		Builder.RegisterService(&LidarService);
    	}
    	if (!RGrpcMaps.RCMaps.RDCMap.empty())
		{
			Builder.RegisterService(&DepthCameraService);
		}
    	if (!RGrpcMaps.RCMaps.RRGBCMap.empty())
    	{
    		Builder.RegisterService(&RGBCameraService);
    		RGBIndexedCameraImageDataCall::InitializeImageModule();
    	}

    	Builder.RegisterService(&RecordingService);
        Builder.RegisterService(&MeshService);
    	Builder.RegisterService(&PointCloudService);
    	
        CompletionQueue = Builder.AddCompletionQueue();
        Server = Builder.BuildAndStart();
    	
        if (Server)
        {
            UE_LOG(LogTemp, Warning, TEXT("Asynchronous Server listening on %s"),
            	*FString(Config.VCCSim.Server.c_str()));

            // Spawn initial asynchronous calls
        	if (!RGrpcMaps.RMaps.DroneMap.empty())
        	{
        		std::map<std::string, ADronePawn*> DroneMap;
        		for (const auto& Pair : RGrpcMaps.RMaps.DroneMap)
        		{
        			DroneMap[Pair.first] = Cast<ADronePawn>(Pair.second);
        		}
        		new GetDronePoseCall(&DroneService, CompletionQueue.get(), DroneMap);
        		new SendDronePoseCall(&DroneService, CompletionQueue.get(), DroneMap);
        		new SendDronePathCall(&DroneService, CompletionQueue.get(), DroneMap);
        	}

            if (!RGrpcMaps.RMaps.CarMap.empty())
            {
                std::map<std::string, ACarPawn*> CarMap;
                for (const auto& Pair : RGrpcMaps.RMaps.CarMap)
                {
                    CarMap[Pair.first] = Cast<ACarPawn>(Pair.second);
                }
                new GetCarOdomCall(&CarService, CompletionQueue.get(), CarMap);
                new SendCarPoseCall(&CarService, CompletionQueue.get(), CarMap);
                new SendCarPathCall(&CarService, CompletionQueue.get(), CarMap);
            }

        	if (!RGrpcMaps.RMaps.FlashMap.empty())
			{
				std::map<std::string, AFlashPawn*> FlashMap;
				for (const auto& Pair : RGrpcMaps.RMaps.FlashMap)
				{
					FlashMap[Pair.first] = Cast<AFlashPawn>(Pair.second);
				}
				new GetFlashPoseCall(&FlashService, CompletionQueue.get(), FlashMap);
				new SendFlashPoseCall(&FlashService, CompletionQueue.get(), FlashMap);
				new SendFlashPathCall(&FlashService, CompletionQueue.get(), FlashMap);
				new CheckFlashReadyCall(&FlashService, CompletionQueue.get(), FlashMap);
				new MoveToNextCall(&FlashService, CompletionQueue.get(), FlashMap);
			}
        	if (!RGrpcMaps.RCMaps.RLMap.empty())
        	{
        		new LidarGetDataCall(&LidarService, CompletionQueue.get(), 
					RGrpcMaps.RCMaps.RLMap);
        		new LidarGetOdomCall(&LidarService, CompletionQueue.get(),
					RGrpcMaps.RCMaps.RLMap);
        		new LidarGetDataAndOdomCall(&LidarService, CompletionQueue.get(),
					RGrpcMaps.RCMaps.RLMap);
        	}
        	if (!RGrpcMaps.RCMaps.RDCMap.empty())
        	{
        		new DepthCameraGetImageDataCall(&DepthCameraService,
					CompletionQueue.get(), RGrpcMaps.RCMaps.RDCMap);
        		new DepthCameraGetPointDataCall(&DepthCameraService,
					CompletionQueue.get(), RGrpcMaps.RCMaps.RDCMap);
        		new DepthCameraGetOdomCall(&DepthCameraService,
					CompletionQueue.get(), RGrpcMaps.RCMaps.RDCMap);
        		new DepthCameraGetImageSizeCall(&DepthCameraService,
        			CompletionQueue.get(), RGrpcMaps.RCMaps.RDCMap);
        	}
        	if (!RGrpcMaps.RCMaps.RRGBCMap.empty())
        	{
        		new RGBCameraGetOdomCall(&RGBCameraService,
					CompletionQueue.get(), RGrpcMaps.RCMaps.RRGBCMap);
        		new RGBIndexedCameraImageDataCall(&RGBCameraService,
					CompletionQueue.get(), RGrpcMaps.RCMaps.RRGBCMap);
        		new RGBIndexedCameraImageSizeCall(&RGBCameraService,
        			CompletionQueue.get(), RGrpcMaps.RCMaps.RRGBCMap);
        	}

        	new SimRecording(&RecordingService, CompletionQueue.get(), Recorder);
        	new SendMeshCall(&MeshService, CompletionQueue.get(), MeshComponent);
        	if (Config.VCCSim.UseMeshManager)
        	{
        		new SendGlobalMeshCall(&MeshService, CompletionQueue.get(),
					MeshManager);
        		new RemoveGlobalMeshCall(&MeshService, CompletionQueue.get(),
        			MeshManager);
        	}
        	
        	new SendPointCloudWithColorCall(&PointCloudService,
        		CompletionQueue.get(), InstancedMeshHolder);

            // Start the completion queue processing loop
            CompletionQueueThread = std::thread([this]() {
                void* tag;
                bool ok;
                while (CompletionQueue->Next(&tag, &ok))
                {
                	AsyncCall* call = static_cast<AsyncCall*>(tag);
                	
                	if (ShutdownRequested.load())
                	{
                		call->Shutdown();
                	}
	                else
	                {
	                	call->Proceed(ok);
	                }
                }
            });

            // Wait until shutdown is requested
            while (!ShutdownRequested.load())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

        	// Real shutdown
        	UE_LOG(LogTemp, Warning, TEXT("DoWork() sees ShutdownRequested,"
									   " shutting down server..."));

        	if (Server)
        	{
        		Server->Shutdown();
        	}
        	if (CompletionQueue)
        	{
        		CompletionQueue->Shutdown();
        	}
        	if (CompletionQueueThread.joinable())
        	{
        		CompletionQueueThread.join();
        	}
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to start server on %s"),
            	*FString(Config.VCCSim.Server.c_str()));
        }
    }

private:
    FVCCSimConfig Config;
	UMeshHandlerComponent* MeshComponent = nullptr;
	UInsMeshHolder* InstancedMeshHolder = nullptr;
	UFMeshManager* MeshManager;
    FRobotGrpcMaps RGrpcMaps;
	ARecorder* Recorder = nullptr;
	
    std::unique_ptr<grpc::ServerCompletionQueue> CompletionQueue;
    std::unique_ptr<grpc::Server> Server;
    std::thread CompletionQueueThread;
};

void RunServer(const FVCCSimConfig& Config, AActor* Holder,
	const FRobotGrpcMaps& RGrpcMaps, UFMeshManager* MeshManager, ARecorder* Recorder)
{
	if (ShutdownRequested.load())
	{
		UE_LOG(LogTemp, Warning, TEXT("Server is already running!"
							  "Not Cleaning up before starting a new one."));
		ShutdownRequested.store(false);
	}
	
	Server_Task = new FAsyncTask<FGrpcServerTask>(
		Config,
		Holder->FindComponentByClass<UMeshHandlerComponent>(),
		Holder->FindComponentByClass<UInsMeshHolder>(),
		MeshManager,
		RGrpcMaps,
		Recorder);
	Server_Task->StartBackgroundTask();
	UE_LOG(LogTemp, Warning, TEXT("GRPC server started."));
}

void ShutdownServer()
{
	if (Server_Task)
	{		
		ShutdownRequested.store(true);
		Server_Task->EnsureCompletion();
		
		delete Server_Task;
		Server_Task = nullptr;
		
		UE_LOG(LogTemp, Warning, TEXT("GRPC Server shutdown."));
	}
	ShutdownRequested.store(false);
}