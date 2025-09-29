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
#include <grpcpp/grpcpp.h>
#include "API/VCCSim.grpc.pb.h"

class ULidarComponent;
class URGBCameraComponent;
class UDepthCameraComponent;
class USegCameraComponent;
class UNormalCameraComponent;
class UMeshHandlerComponent;
class UInsMeshHolder;
class ADronePawn;
class AFlashPawn;
class ACarPawn;

class AsyncCall
{
public:
    virtual void Proceed(bool OK) = 0;
    virtual void Shutdown() = 0;
    virtual ~AsyncCall() {}
};

template <typename RequestType, typename ResponseType,
          typename ServiceType, typename ComponentType>
class AsyncCallTemplate : public AsyncCall
{
public:
    AsyncCallTemplate(ServiceType* Service,
        grpc::ServerCompletionQueue* CompletionQueue, ComponentType* Component);

    virtual void Proceed(bool OK) override final;
    virtual void Shutdown() override final;

protected:
    virtual void PrepareNextCall() = 0;
    virtual void InitializeRequest() = 0;
    virtual void ProcessRequest() = 0;

    grpc::ServerContext Context;
    RequestType Request;
    ResponseType Response;
    grpc::ServerAsyncResponseWriter<ResponseType> Responder;
    
    enum CallStatus { CREATE, PROCESS, FINISH };
    CallStatus Status;

    ServiceType* Service;
    grpc::ServerCompletionQueue* CompletionQueue;
    ComponentType* component_;
};

template <typename RequestType, typename ResponseType,
          typename ServiceType, typename RobotComponentMap>
class AsyncCallTemplateM : public AsyncCall
{
public:
    AsyncCallTemplateM(ServiceType* Service,
        grpc::ServerCompletionQueue* CompletionQueue, const RobotComponentMap& RCMap);

    virtual void Proceed(bool OK) override final;
    virtual void Shutdown() override final;

protected:
    virtual void PrepareNextCall() = 0;
    virtual void InitializeRequest() = 0;
    virtual void ProcessRequest() = 0;

    grpc::ServerContext Context;
    RequestType Request;
    ResponseType Response;
    grpc::ServerAsyncResponseWriter<ResponseType> Responder;
    
    enum CallStatus { CREATE, PROCESS, FINISH };
    CallStatus Status;

    ServiceType* Service;
    grpc::ServerCompletionQueue* CompletionQueue;
    RobotComponentMap RCMap_;
};

// Image calls need to wait for the image to be captured
// So we need to let the async call to finish processing
template <typename RequestType, typename ResponseType,
          typename ServiceType, typename RobotComponentMap>
class AsyncCallTemplateImage : public AsyncCall
{
public:
    AsyncCallTemplateImage(ServiceType* Service,
        grpc::ServerCompletionQueue* CompletionQueue, const RobotComponentMap& RCMap);

    virtual void Proceed(bool OK) override final;
    virtual void Shutdown() override final;

protected:
    virtual void PrepareNextCall() = 0;
    virtual void InitializeRequest() = 0;
    virtual void ProcessRequest() = 0;

    grpc::ServerContext Context;
    RequestType Request;
    ResponseType Response;
    grpc::ServerAsyncResponseWriter<ResponseType> Responder;
    
    enum CallStatus { CREATE, PROCESS, FINISH };
    CallStatus Status;

    ServiceType* Service;
    grpc::ServerCompletionQueue* CompletionQueue;
    RobotComponentMap RCMap_;
};

/* -------------------------- Recording ---------------------------------- */
class ARecorder;
class SimRecording : public AsyncCallTemplate<
    VCCSim::EmptyRequest, VCCSim::Status,
    VCCSim::RecordingService::AsyncService,
    ARecorder>
{
public:
    SimRecording(
        VCCSim::RecordingService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue, ARecorder* Recorder);

protected:
    virtual void PrepareNextCall() override;
    virtual void InitializeRequest() override;
    virtual void ProcessRequest() override;
};

/* ---------------------------------LiDAR---------------------------------- */
class LidarGetDataCall : public AsyncCallTemplateM<
    VCCSim::RobotName, VCCSim::LidarData,
    VCCSim::LidarService::AsyncService,
    std::map<std::string, ULidarComponent*>>
{
public:
    LidarGetDataCall(
        VCCSim::LidarService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, ULidarComponent*>& LiDARComponentMap);

protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

class LidarGetOdomCall : public AsyncCallTemplateM<
    VCCSim::RobotName, VCCSim::Odometry,
    VCCSim::LidarService::AsyncService,
    std::map<std::string, ULidarComponent*>>
{
public:
    LidarGetOdomCall(
        VCCSim::LidarService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, ULidarComponent*>& LiDARComponentMap);

protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

class LidarGetDataAndOdomCall : public AsyncCallTemplateM<
    VCCSim::RobotName, VCCSim::LidarDataAndOdom,
    VCCSim::LidarService::AsyncService,
    std::map<std::string, ULidarComponent*>>
{
public:
    LidarGetDataAndOdomCall(
        VCCSim::LidarService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, ULidarComponent*>& LiDARComponentMap);

protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

/* --------------------------Camera Service---------------------------------- */

class CameraGetRGBDataCall : public AsyncCallTemplateImage<
    VCCSim::IndexedCamera, VCCSim::RGBData,
    VCCSim::CameraService::AsyncService,
    std::map<std::string, URGBCameraComponent*>>
{
public:
    CameraGetRGBDataCall(
        VCCSim::CameraService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, URGBCameraComponent*>& RGBComponentMap);

protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

class CameraGetDepthDataCall : public AsyncCallTemplateImage<
    VCCSim::IndexedCamera, VCCSim::DepthData,
    VCCSim::CameraService::AsyncService,
    std::map<std::string, UDepthCameraComponent*>>
{
public:
    CameraGetDepthDataCall(
        VCCSim::CameraService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, UDepthCameraComponent*>& DepthComponentMap);
protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

class CameraGetDepthPointCloudCall : public AsyncCallTemplateImage<
    VCCSim::IndexedCamera, VCCSim::TPointCloud,
    VCCSim::CameraService::AsyncService,
    std::map<std::string, UDepthCameraComponent*>>
{
public:
    CameraGetDepthPointCloudCall(
        VCCSim::CameraService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, UDepthCameraComponent*>& DepthComponentMap);
protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

class CameraGetSegmentDataCall : public AsyncCallTemplateImage<
    VCCSim::IndexedCamera, VCCSim::RGBData,
    VCCSim::CameraService::AsyncService,
    std::map<std::string, USegCameraComponent*>>
{
public:
    CameraGetSegmentDataCall(
        VCCSim::CameraService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, USegCameraComponent*>& SegComponentMap);
protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

class CameraGetNormalDataCall : public AsyncCallTemplateImage<
    VCCSim::IndexedCamera, VCCSim::NormalData,
    VCCSim::CameraService::AsyncService,
    std::map<std::string, UNormalCameraComponent*>>
{
public:
    CameraGetNormalDataCall(
        VCCSim::CameraService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, UNormalCameraComponent*>& NormalComponentMap);
protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

/* --------------------------Mesh Handler---------------------------------- */

class SendMeshCall : public AsyncCallTemplate<VCCSim::MeshData,
    VCCSim::Status, VCCSim::MeshService::AsyncService, UMeshHandlerComponent>
{
public:
    SendMeshCall(VCCSim::MeshService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue, UMeshHandlerComponent* mesh_component);
protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

class UFMeshManager;

class SendGlobalMeshCall : public AsyncCallTemplate<VCCSim::MeshData,
    VCCSim::MeshID, VCCSim::MeshService::AsyncService, UFMeshManager>
{
public:
    SendGlobalMeshCall(VCCSim::MeshService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue, UFMeshManager* MeshManager);
protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

class RemoveGlobalMeshCall : public AsyncCallTemplate<VCCSim::MeshID,
    VCCSim::Status, VCCSim::MeshService::AsyncService, UFMeshManager>
{
public:
    RemoveGlobalMeshCall(VCCSim::MeshService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue, UFMeshManager* MeshManager);
protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

/* --------------------------Misc Handler---------------------------------- */

class SendPointCloudWithColorCall : public AsyncCallTemplate<
    VCCSim::ColoredPointCloud, VCCSim::Status,
    VCCSim::PointCloudService::AsyncService, UInsMeshHolder>
{
public:
    SendPointCloudWithColorCall(VCCSim::PointCloudService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue, UInsMeshHolder* mesh_holder);

protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

/* --------------------------Drone Handler---------------------------------- */

class GetDronePoseCall final: public AsyncCallTemplateM<VCCSim::RobotName,
    VCCSim::Pose, VCCSim::DroneService::AsyncService,
    std::map<std::string, ADronePawn*>>
{
public:
    GetDronePoseCall(VCCSim::DroneService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, ADronePawn*>& rcmap);
protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

class SendDronePoseCall : public AsyncCallTemplateM<VCCSim::DronePose,
    VCCSim::Status, VCCSim::DroneService::AsyncService,
    std::map <std::string, ADronePawn*>>
{
public:
    SendDronePoseCall(VCCSim::DroneService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, ADronePawn*>& rcmap);
protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

class SendDronePathCall : public AsyncCallTemplateM<VCCSim::DronePath,
    VCCSim::Status, VCCSim::DroneService::AsyncService,
    std::map<std::string, ADronePawn*>>
{
public:
    SendDronePathCall(VCCSim::DroneService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, ADronePawn*>& rcmap);
protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};
/* --------------------------Car Handler---------------------------------- */

class GetCarOdomCall final : public AsyncCallTemplateM<VCCSim::RobotName,
    VCCSim::Odometry, VCCSim::CarService::AsyncService,
    std::map<std::string, ACarPawn*>>
{
public:
    GetCarOdomCall(VCCSim::CarService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, ACarPawn*>& rcmap);
protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

class SendCarPoseCall : public AsyncCallTemplateM<VCCSim::CarPose,
    VCCSim::Status, VCCSim::CarService::AsyncService,
    std::map<std::string, ACarPawn*>>
{
public:
    SendCarPoseCall(VCCSim::CarService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, ACarPawn*>& rcmap);
protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

class SendCarPathCall : public AsyncCallTemplateM<VCCSim::CarPath,
    VCCSim::Status, VCCSim::CarService::AsyncService,
    std::map<std::string, ACarPawn*>>
{
public:
    SendCarPathCall(VCCSim::CarService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, ACarPawn*>& rcmap);
protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

/* --------------------------Flash Handler---------------------------------- */

class GetFlashPoseCall final : public AsyncCallTemplateM<VCCSim::RobotName,
    VCCSim::Pose, VCCSim::FlashService::AsyncService,
    std::map<std::string, AFlashPawn*>>
{
public:
    GetFlashPoseCall(VCCSim::FlashService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, AFlashPawn*>& rcmap);
protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

class SendFlashPoseCall : public AsyncCallTemplateM<VCCSim::DronePose,
    VCCSim::Status, VCCSim::FlashService::AsyncService,
    std::map<std::string, AFlashPawn*>>
{
public:
    SendFlashPoseCall(VCCSim::FlashService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, AFlashPawn*>& rcmap);
protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

class SendFlashPathCall : public AsyncCallTemplateM<VCCSim::DronePath,
    VCCSim::Status, VCCSim::FlashService::AsyncService,
    std::map<std::string, AFlashPawn*>>
{
public:
    SendFlashPathCall(VCCSim::FlashService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, AFlashPawn*>& rcmap);
protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

class CheckFlashReadyCall : public AsyncCallTemplateM<VCCSim::RobotName,
    VCCSim::Status, VCCSim::FlashService::AsyncService,
    std::map<std::string, AFlashPawn*>>
{
public:
    CheckFlashReadyCall(VCCSim::FlashService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, AFlashPawn*>& rcmap);
protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};

class FlashMoveToNextCall : public AsyncCallTemplateM<VCCSim::RobotName,
    VCCSim::Status, VCCSim::FlashService::AsyncService,
    std::map<std::string, AFlashPawn*>>
{
public:
    FlashMoveToNextCall(VCCSim::FlashService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, AFlashPawn*>& rcmap);
protected:
    virtual void PrepareNextCall() override final;
    virtual void InitializeRequest() override final;
    virtual void ProcessRequest() override final;
};


/* --------------------------Template implementation----------------------- */


template <typename RequestType, typename ResponseType,
          typename ServiceType, typename ComponentType>
AsyncCallTemplate<RequestType, ResponseType,
    ServiceType, ComponentType>::AsyncCallTemplate(
    ServiceType* Service, grpc::ServerCompletionQueue* CompletionQueue, ComponentType* Component)
    : Responder(&Context), Status(CREATE), Service(Service),
      CompletionQueue(CompletionQueue), component_(Component) {}

template <typename RequestType, typename ResponseType,
          typename ServiceType, typename ComponentType>
void AsyncCallTemplate<RequestType, ResponseType, ServiceType,
    ComponentType>::Proceed(bool OK)
{
    if (Status == CREATE) {
        Status = PROCESS;
        InitializeRequest();
    }
    else if (Status == PROCESS) {
        PrepareNextCall();
        ProcessRequest();

        Status = FINISH;
        Responder.Finish(Response, grpc::Status::OK, this);
    }
    else {
        delete this;
    }
}

template <typename RequestType, typename ResponseType,
          typename ServiceType, typename ComponentType>
void AsyncCallTemplate<RequestType, ResponseType,
    ServiceType, ComponentType>::Shutdown()
{
    Status = FINISH;
    delete this;
}

template <typename RequestType, typename ResponseType, typename ServiceType,
    typename RobotComponentMap>
AsyncCallTemplateM<RequestType, ResponseType, ServiceType, RobotComponentMap>::
AsyncCallTemplateM(ServiceType* Service, grpc::ServerCompletionQueue* CompletionQueue,
    const RobotComponentMap& RCMap)
    : Responder(&Context), Status(CREATE), Service(Service),
      CompletionQueue(CompletionQueue), RCMap_(RCMap) {}

template <typename RequestType, typename ResponseType, typename ServiceType,
    typename RobotComponentMap>
void AsyncCallTemplateM<RequestType, ResponseType, ServiceType,
RobotComponentMap>::Proceed(bool OK)
{
    if (Status == CREATE) {
        Status = PROCESS;
        InitializeRequest();
    }
    else if (Status == PROCESS) {
        PrepareNextCall();
        ProcessRequest();

        Status = FINISH;
        Responder.Finish(Response, grpc::Status::OK, this);
    }
    else {
        delete this;
    }
}

template <typename RequestType, typename ResponseType, typename ServiceType,
    typename RobotComponentMap>
void AsyncCallTemplateM<RequestType, ResponseType, ServiceType,
RobotComponentMap>::Shutdown()
{
    Status = FINISH;
    delete this;
}

template <typename RequestType, typename ResponseType, typename ServiceType,
    typename RobotComponentMap>
AsyncCallTemplateImage<RequestType, ResponseType, ServiceType, RobotComponentMap>::
AsyncCallTemplateImage(ServiceType* Service, grpc::ServerCompletionQueue* CompletionQueue,
    const RobotComponentMap& RCMap)
    : Responder(&Context), Status(CREATE), Service(Service),
      CompletionQueue(CompletionQueue), RCMap_(RCMap) {}

template <typename RequestType, typename ResponseType, typename ServiceType,
    typename RobotComponentMap>
void AsyncCallTemplateImage<RequestType, ResponseType, ServiceType,
RobotComponentMap>::Proceed(bool OK)
{
    if (Status == CREATE) {
        Status = PROCESS;
        InitializeRequest();
    }
    else if (Status == PROCESS) {
        PrepareNextCall();
        ProcessRequest();
    }
    else {
        delete this;
    }
}

template <typename RequestType, typename ResponseType, typename ServiceType,
    typename RobotComponentMap>
void AsyncCallTemplateImage<RequestType, ResponseType, ServiceType,
RobotComponentMap>::Shutdown()
{
    Status = FINISH;
    delete this;
}
