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

DEFINE_LOG_CATEGORY_STATIC(LogGRPCCall, Log, All);

#include "API/GRPCCall.h"
#include "Sensors/LidarSensor.h"
#include "Sensors/CameraSensor.h"
#include "Sensors/DepthCamera.h"
#include "Sensors/SegmentCamera.h"
#include "Sensors/NormalCamera.h"
#include "Utils/MeshHandlerComponent.h"
#include "Utils/InsMeshHolder.h"
#include "Pawns/DronePawn.h"
#include "Pawns/FlashPawn.h"
#include "Pawns/CarPawn.h"
#include "Simulation/MeshManager.h"
#include "Simulation/Recorder.h"

SimRecording::SimRecording(
    VCCSim::RecordingService::AsyncService* service,
    grpc::ServerCompletionQueue* cq, ARecorder* Recorder)
        : AsyncCallTemplate(service, cq, Recorder)
{
    Proceed(true);
}

void SimRecording::PrepareNextCall()
{
    new SimRecording(service_, cq_, component_);
}

void SimRecording::InitializeRequest()
{
    service_->RequestRecording(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void SimRecording::ProcessRequest()
{
    component_->ToggleRecording();
    response_.set_status(component_->RecordState);
}

LidarGetDataCall::LidarGetDataCall(
    VCCSim::LidarService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    const std::map<std::string, ULidarComponent*>& RLMap)
    : AsyncCallTemplateM(service, cq, RLMap)
{
    Proceed(true);
}

void LidarGetDataCall::PrepareNextCall()
{
    new LidarGetDataCall(service_, cq_, RCMap_);
}

void LidarGetDataCall::InitializeRequest()
{
    service_->RequestGetLiDARData(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void LidarGetDataCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        for (const auto& Point : RCMap_[request_.name()]->GetPointCloudData())
        {
            VCCSim::Vec3f* LidarPoint = response_.add_data();
            LidarPoint->set_x(Point.X);
            LidarPoint->set_y(Point.Y);
            LidarPoint->set_z(Point.Z);
        }
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("LidarGetDataCall: "
                                      "Lidar component not found!"));
    }
}

LidarGetOdomCall::LidarGetOdomCall(
        VCCSim::LidarService::AsyncService* service,
        grpc::ServerCompletionQueue* cq,
        const std::map<std::string, ULidarComponent*>& RLMap)
    : AsyncCallTemplateM(service, cq, RLMap)
{
    Proceed(true);
}

void LidarGetOdomCall::PrepareNextCall()
{
    new LidarGetOdomCall(service_, cq_, RCMap_);
}

void LidarGetOdomCall::InitializeRequest()
{
    service_->RequestGetLiDAROdom(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void LidarGetOdomCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        const auto Lidar = RCMap_[request_.name()];
        const FVector Location = Lidar->GetComponentLocation();
        const FRotator Rotation = Lidar->GetComponentRotation();
        const FVector LinearVelocity = Lidar->GetPhysicsLinearVelocity();
        const FVector AngularVelocity = Lidar->GetPhysicsAngularVelocityInDegrees();

        VCCSim::Pose* PoseData = response_.mutable_pose();
        VCCSim::Vec3f* Position = PoseData->mutable_position();
        Position->set_x(Location.X);
        Position->set_y(Location.Y);
        Position->set_z(Location.Z);

        VCCSim::Rotation* Rot = PoseData->mutable_rotation();
        FQuat Quat = Rotation.Quaternion();
        Rot->set_x(Quat.X);
        Rot->set_y(Quat.Y);
        Rot->set_z(Quat.Z);
        Rot->set_w(Quat.W);

        VCCSim::Twist* TwistData = response_.mutable_twist();
        VCCSim::Vec3f* LinearVel = TwistData->mutable_linear();
        LinearVel->set_x(LinearVelocity.X);
        LinearVel->set_y(LinearVelocity.Y);
        LinearVel->set_z(LinearVelocity.Z);

        VCCSim::Vec3f* AngularVel = TwistData->mutable_angular();
        AngularVel->set_x(AngularVelocity.X);
        AngularVel->set_y(AngularVelocity.Y);
        AngularVel->set_z(AngularVelocity.Z);
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("LidarGetOdomCall: "
                                      "Lidar component not found!"));
    }
}

LidarGetDataAndOdomCall::LidarGetDataAndOdomCall(
    VCCSim::LidarService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    const std::map<std::string, ULidarComponent*>& rcmap)
    : AsyncCallTemplateM(service, cq, rcmap)
{
    Proceed(true);
}

void LidarGetDataAndOdomCall::PrepareNextCall()
{
    new LidarGetDataAndOdomCall(service_, cq_, RCMap_);
}

void LidarGetDataAndOdomCall::InitializeRequest()
{
    service_->RequestGetLiDARDataAndOdom(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void LidarGetDataAndOdomCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        auto DataAndOdom =
            RCMap_[request_.name()]->GetPointCloudDataAndOdom();
        const auto& Odom = DataAndOdom.Value;
        
        for (const auto& Point : DataAndOdom.Key)
        {
            VCCSim::Vec3f* LidarPoint = response_.mutable_data()->add_data();
            LidarPoint->set_x(Point.X);
            LidarPoint->set_y(Point.Y);
            LidarPoint->set_z(Point.Z);
        }

        VCCSim::Pose* PoseData = response_.mutable_odom()->mutable_pose();
        VCCSim::Vec3f* Position = PoseData->mutable_position();
        Position->set_x(Odom.Location.X);
        Position->set_y(Odom.Location.Y);
        Position->set_z(Odom.Location.Z);

        VCCSim::Rotation* Rot = PoseData->mutable_rotation();
        FQuat Quat = Odom.Rotation.Quaternion();
        Rot->set_x(Quat.X);
        Rot->set_y(Quat.Y);
        Rot->set_z(Quat.Z);
        Rot->set_w(Quat.W);

        VCCSim::Twist* TwistData = response_.mutable_odom()->mutable_twist();
        VCCSim::Vec3f* LinearVel = TwistData->mutable_linear();
        LinearVel->set_x(Odom.LinearVelocity.X);
        LinearVel->set_y(Odom.LinearVelocity.Y);
        LinearVel->set_z(Odom.LinearVelocity.Z);

        VCCSim::Vec3f* AngularVel = TwistData->mutable_angular();
        AngularVel->set_x(Odom.AngularVelocity.X);
        AngularVel->set_y(Odom.AngularVelocity.Y);
        AngularVel->set_z(Odom.AngularVelocity.Z);
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("LidarGetDataAndOdomCall: "
                                      "Lidar component not found!"));
    }
}

/* --------------------------RGB Camera---------------------------------- */

RGBCameraGetRGBDataCall::RGBCameraGetRGBDataCall(
    VCCSim::RGBCameraService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, URGBCameraComponent*>& RGBComponentMap)
        : AsyncCallTemplateImage(Service, CompletionQueue, RGBComponentMap)
{
    Proceed(true);
}

void RGBCameraGetRGBDataCall::PrepareNextCall()
{
    new RGBCameraGetRGBDataCall(service_, cq_, RCMap_);
}

void RGBCameraGetRGBDataCall::InitializeRequest()
{
    service_->RequestGetRGBData(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void RGBCameraGetRGBDataCall::ProcessRequest()
{
    std::string CameraName = request_.robot_name().name() + "^" +
                             std::to_string(request_.index());

    if (!RCMap_.contains(CameraName))
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("RGBDCameraGetRGBDataCall:"
                                      " RGBD Camera component not found!"));
        return;
    }

    auto* RGBCamera = RCMap_[CameraName];
    if (!RGBCamera)
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("RGBDCameraGetRGBDataCall:"
                                      " Invalid RGBD Camera reference!"));
        return;
    }

    RGBCamera->AsyncGetRGBImageData(
        [this, RGBCamera](const TArray<FColor>& RGBData)
    {
        AsyncTask(ENamedThreads::AnyBackgroundHiPriTask,
            [this, RGBData, RGBCamera]()
        {
            response_.set_width(RGBCamera->Width);
            response_.set_height(RGBCamera->Height);
            response_.set_timestamp(FDateTime::UtcNow().ToUnixTimestamp());
            response_.set_data(RGBData.GetData(), RGBData.Num());
            status_ = FINISH;
            responder_.Finish(response_, grpc::Status::OK, this);
        });
    });
}

RGBCameraGetCameraOdomCall::RGBCameraGetCameraOdomCall(
    VCCSim::RGBCameraService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, URGBCameraComponent*>& RGBComponentMap)
        : AsyncCallTemplateM(Service, CompletionQueue, RGBComponentMap)
{
    Proceed(true);
}

void RGBCameraGetCameraOdomCall::PrepareNextCall()
{
    new RGBCameraGetCameraOdomCall(service_, cq_, RCMap_);
}

void RGBCameraGetCameraOdomCall::InitializeRequest()
{
    service_->RequestGetCameraOdom(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void RGBCameraGetCameraOdomCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        const FVector Location = RCMap_[request_.name()]->GetComponentLocation();
        const FRotator Rotation = RCMap_[request_.name()]->GetComponentRotation();
        const FVector LinearVelocity = RCMap_[request_.name()]->GetPhysicsLinearVelocity();
        const FVector AngularVelocity = RCMap_[request_.name()]->GetPhysicsAngularVelocityInDegrees();

        VCCSim::Pose* PoseData = response_.mutable_pose();
        VCCSim::Vec3f* Position = PoseData->mutable_position();
        Position->set_x(Location.X);
        Position->set_y(Location.Y);
        Position->set_z(Location.Z);

        VCCSim::Rotation* Rot = PoseData->mutable_rotation();
        FQuat Quat = Rotation.Quaternion();
        Rot->set_x(Quat.X);
        Rot->set_y(Quat.Y);
        Rot->set_z(Quat.Z);
        Rot->set_w(Quat.W);

        VCCSim::Twist* TwistData = response_.mutable_twist();
        VCCSim::Vec3f* LinearVel = TwistData->mutable_linear();
        LinearVel->set_x(LinearVelocity.X);
        LinearVel->set_y(LinearVelocity.Y);
        LinearVel->set_z(LinearVelocity.Z);

        VCCSim::Vec3f* AngularVel = TwistData->mutable_angular();
        AngularVel->set_x(AngularVelocity.X);
        AngularVel->set_y(AngularVelocity.Y);
        AngularVel->set_z(AngularVelocity.Z);
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("RGBDCameraGetCameraOdomCall:"
                                      " RGBD Camera component not found!"));
    }
}

// --------------------------Depth Camera---------------------------------- //

DepthCameraGetDepthDataCall::DepthCameraGetDepthDataCall(
    VCCSim::DepthCameraService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, UDepthCameraComponent*>& DepthComponentMap)
        : AsyncCallTemplateImage(Service, CompletionQueue, DepthComponentMap)
{
    Proceed(true);
}

void DepthCameraGetDepthDataCall::PrepareNextCall()
{
    new DepthCameraGetDepthDataCall(service_, cq_, RCMap_);
}

void DepthCameraGetDepthDataCall::InitializeRequest()
{
    service_->RequestGetDepthCameraImageData(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void DepthCameraGetDepthDataCall::ProcessRequest()
{
    std::string CameraName = request_.robot_name().name() + "^" +
                             std::to_string(request_.index());

    if (!RCMap_.contains(CameraName))
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("RGBDCameraGetDepthDataCall:"
                                      " RGBD Camera component not found!"));
        return;
    }

    auto* RGBDCamera = RCMap_[CameraName];
    if (!RGBDCamera)
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("RGBDCameraGetDepthDataCall:"
                                      " Invalid RGBD Camera reference!"));
        return;
    }

    RGBDCamera->AsyncGetDepthImageData(
        [this, RGBDCamera](const TArray<FFloat16Color>& RGBDData)
    {
        AsyncTask(ENamedThreads::AnyBackgroundHiPriTask,
            [this, RGBDData, RGBDCamera]()
        {
            response_.set_width(RGBDCamera->Width);
            response_.set_height(RGBDCamera->Height);
            response_.set_timestamp(FDateTime::UtcNow().ToUnixTimestamp());

            // Extract depth from alpha channel
            response_.mutable_data()->Reserve(RGBDData.Num());
            for (const FFloat16Color& Pixel : RGBDData)
            {
                response_.add_data(Pixel.A);
            }

            status_ = FINISH;
            responder_.Finish(response_, grpc::Status::OK, this);
        });
    });
}

DepthCameraGetDepthPointCloudCall::DepthCameraGetDepthPointCloudCall(
    VCCSim::DepthCameraService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, UDepthCameraComponent*>& DepthComponentMap)
        : AsyncCallTemplateImage(Service, CompletionQueue, DepthComponentMap)
{
    Proceed(true);
}

void DepthCameraGetDepthPointCloudCall::PrepareNextCall()
{
    new DepthCameraGetDepthPointCloudCall(service_, cq_, RCMap_);
}

void DepthCameraGetDepthPointCloudCall::InitializeRequest()
{
    service_->RequestGetDepthPointCloud(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void DepthCameraGetDepthPointCloudCall::ProcessRequest()
{
    std::string CameraName = request_.robot_name().name() + "^" +
                             std::to_string(request_.index());

    if (!RCMap_.contains(CameraName))
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("RGBDCameraGetDepthPointCloudCall:"
                                      " RGBD Camera component not found!"));
        return;
    }

    auto* RGBDCamera = RCMap_[CameraName];
    if (!RGBDCamera)
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("RGBDCameraGetDepthPointCloudCall:"
                                      " Invalid RGBD Camera reference!"));
        return;
    }

    RGBDCamera->AsyncGetPointCloudData(
        [this, RGBDCamera]()
    {
        AsyncTask(ENamedThreads::AnyBackgroundHiPriTask,
            [this, RGBDCamera]()
        {
            const auto PointCloudData = RGBDCamera->GeneratePointCloud();
            if (PointCloudData.Num() == 0)
            {
                UE_LOG(LogGRPCCall, Warning, TEXT("RGBDCameraGetDepthPointCloudCall:"
                                              " No point cloud data available!"));
                status_ = FINISH;
                responder_.Finish(response_, grpc::Status::CANCELLED, this);
                return;
            }

            response_.set_timestamp(FDateTime::UtcNow().ToUnixTimestamp());

            for (const FDCPoint& Point : PointCloudData)
            {
                VCCSim::Vec3f* point = response_.add_data();
                point->set_x(Point.Location.X);
                point->set_y(Point.Location.Y);
                point->set_z(Point.Location.Z);
            }

            status_ = FINISH;
            responder_.Finish(response_, grpc::Status::OK, this);
        });
    });
}

DepthCameraGetCameraOdomCall::DepthCameraGetCameraOdomCall(
    VCCSim::DepthCameraService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, UDepthCameraComponent*>& DepthComponentMap)
        : AsyncCallTemplateM(Service, CompletionQueue, DepthComponentMap)
{
    Proceed(true);
}

void DepthCameraGetCameraOdomCall::PrepareNextCall()
{
    new DepthCameraGetCameraOdomCall(service_, cq_, RCMap_);
}

void DepthCameraGetCameraOdomCall::InitializeRequest()
{
    service_->RequestGetDepthCameraOdom(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void DepthCameraGetCameraOdomCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        const FVector Location = RCMap_[request_.name()]->GetComponentLocation();
        const FRotator Rotation = RCMap_[request_.name()]->GetComponentRotation();
        const FVector LinearVelocity = RCMap_[request_.name()]->GetPhysicsLinearVelocity();
        const FVector AngularVelocity = RCMap_[request_.name()]->GetPhysicsAngularVelocityInDegrees();

        VCCSim::Pose* PoseData = response_.mutable_pose();
        VCCSim::Vec3f* Position = PoseData->mutable_position();
        Position->set_x(Location.X);
        Position->set_y(Location.Y);
        Position->set_z(Location.Z);

        VCCSim::Rotation* Rot = PoseData->mutable_rotation();
        FQuat Quat = Rotation.Quaternion();
        Rot->set_x(Quat.X);
        Rot->set_y(Quat.Y);
        Rot->set_z(Quat.Z);
        Rot->set_w(Quat.W);

        VCCSim::Twist* TwistData = response_.mutable_twist();
        VCCSim::Vec3f* LinearVel = TwistData->mutable_linear();
        LinearVel->set_x(LinearVelocity.X);
        LinearVel->set_y(LinearVelocity.Y);
        LinearVel->set_z(LinearVelocity.Z);

        VCCSim::Vec3f* AngularVel = TwistData->mutable_angular();
        AngularVel->set_x(AngularVelocity.X);
        AngularVel->set_y(AngularVelocity.Y);
        AngularVel->set_z(AngularVelocity.Z);
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("RGBDCameraGetCameraOdomCall:"
                                      " RGBD Camera component not found!"));
    }
}

SegmentCameraGetOdomCall::SegmentCameraGetOdomCall(
    VCCSim::SegmentationCameraService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    const std::map<std::string, USegmentationCameraComponent*>& rscmap)
        : AsyncCallTemplateM(service, cq, rscmap)
{
    Proceed(true);
}

void SegmentCameraGetOdomCall::PrepareNextCall()
{
    new SegmentCameraGetOdomCall(service_, cq_, RCMap_);
}

void SegmentCameraGetOdomCall::InitializeRequest()
{
    service_->RequestGetSegmentationCameraOdom(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void SegmentCameraGetOdomCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        const FVector Location = RCMap_[request_.name()]->GetComponentLocation();
        const FRotator Rotation = RCMap_[request_.name()]->GetComponentRotation();
        const FVector LinearVelocity = RCMap_[request_.name()]->GetPhysicsLinearVelocity();
        const FVector AngularVelocity = RCMap_[request_.name()]->GetPhysicsAngularVelocityInDegrees();

        VCCSim::Pose* PoseData = response_.mutable_pose();
        VCCSim::Vec3f* Position = PoseData->mutable_position();
        Position->set_x(Location.X);
        Position->set_y(Location.Y);
        Position->set_z(Location.Z);

        VCCSim::Rotation* Rot = PoseData->mutable_rotation();
        FQuat Quat = Rotation.Quaternion();
        Rot->set_x(Quat.X);
        Rot->set_y(Quat.Y);
        Rot->set_z(Quat.Z);
        Rot->set_w(Quat.W);

        VCCSim::Twist* TwistData = response_.mutable_twist();
        VCCSim::Vec3f* LinearVel = TwistData->mutable_linear();
        LinearVel->set_x(LinearVelocity.X);
        LinearVel->set_y(LinearVelocity.Y);
        LinearVel->set_z(LinearVelocity.Z);

        VCCSim::Vec3f* AngularVel = TwistData->mutable_angular();
        AngularVel->set_x(AngularVelocity.X);
        AngularVel->set_y(AngularVelocity.Y);
        AngularVel->set_z(AngularVelocity.Z);
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("SegmentCameraGetOdomCall: "
                                      "Segmentation Camera component not found!"));
    }
}

SendMeshCall::SendMeshCall(VCCSim::MeshService::AsyncService* service,
                           grpc::ServerCompletionQueue* cq,
                           UMeshHandlerComponent* mesh_component)
    : AsyncCallTemplate(service, cq, mesh_component)
{
    Proceed(true);
}

void SendMeshCall::PrepareNextCall()
{
    new SendMeshCall(service_, cq_, component_);
}

void SendMeshCall::InitializeRequest()
{
    service_->RequestSendMesh(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void SendMeshCall::ProcessRequest()
{
    if (component_)
    {
        const VCCSim::Pose& Transform = request_.transform();
        FVector Location(
            Transform.position().x(),
            Transform.position().y(),
            Transform.position().z()
        );
        FQuat Quaternion(
            Transform.rotation().x(),
            Transform.rotation().y(),
            Transform.rotation().z(),
            Transform.rotation().w()
        );
        FTransform MeshTransform(Quaternion, Location);
        
        component_->UpdateMeshFromGRPC(
            reinterpret_cast<const uint8*>(request_.data().data()),
            request_.data().size(),
            MeshTransform
        );
    }
    else
    {
        UE_LOG(LogGRPCCall, Error, TEXT("SendMeshCall: "
                                    "Mesh component not found!"));
    }
    response_.set_status(true);
}

SendGlobalMeshCall::SendGlobalMeshCall(
    VCCSim::MeshService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    UFMeshManager* MeshManager)
        : AsyncCallTemplate(service, cq, MeshManager)
{
    Proceed(true);
}

void SendGlobalMeshCall::PrepareNextCall()
{
    new SendGlobalMeshCall(service_, cq_, component_);
}

void SendGlobalMeshCall::InitializeRequest()
{
    service_->RequestSendGlobalMesh(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void SendGlobalMeshCall::ProcessRequest()
{
    if (component_)
    {
        const VCCSim::Pose& Transform = request_.transform();
        FVector Location(
            Transform.position().x(),
            Transform.position().y(),
            Transform.position().z()
        );
        FQuat Quaternion(
            Transform.rotation().x(),
            Transform.rotation().y(),
            Transform.rotation().z(),
            Transform.rotation().w()
        );
        FTransform MeshTransform(Quaternion, Location);
        
        const auto ID = component_->AddGlobalMesh();
        if (component_->UpdateMesh(ID,
            reinterpret_cast<const uint8*>(request_.data().data()),
            request_.data().size(),
            MeshTransform))
        {
            response_.set_id(ID);
        }
        else
        {
            response_.set_id(-1);
        }
    }
    else
    {
        UE_LOG(LogGRPCCall, Error, TEXT("SendGlobalMeshCall: "
                                    "Mesh manager not found!"));
    }
}

RemoveGlobalMeshCall::RemoveGlobalMeshCall(
    VCCSim::MeshService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    UFMeshManager* MeshManager)
        : AsyncCallTemplate(service, cq, MeshManager)
{
    Proceed(true);
}

void RemoveGlobalMeshCall::PrepareNextCall()
{
    new RemoveGlobalMeshCall(service_, cq_, component_);
}

void RemoveGlobalMeshCall::InitializeRequest()
{
    service_->RequestRemoveGlobalMesh(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void RemoveGlobalMeshCall::ProcessRequest()
{
    if (component_)
    {
        response_.set_status(component_->RemoveGlobalMesh(request_.id()));
    }
    else
    {
        UE_LOG(LogGRPCCall, Error, TEXT("RemoveGlobalMeshCall: "
                                    "Mesh manager not found!"));
    }
}

SendPointCloudWithColorCall::SendPointCloudWithColorCall(
    VCCSim::PointCloudService::AsyncService* service,
    grpc::ServerCompletionQueue* cq, UInsMeshHolder* mesh_holder)
    : AsyncCallTemplate(service, cq, mesh_holder)
{
    Proceed(true);
}

void SendPointCloudWithColorCall::PrepareNextCall()
{
    new SendPointCloudWithColorCall(service_, cq_, component_);
}

void SendPointCloudWithColorCall::InitializeRequest()
{
    service_->RequestSendPointCloudWithColor(
        &ctx_, &request_, &responder_, cq_, cq_, this);
}

void SendPointCloudWithColorCall::ProcessRequest()
{
    if (component_)
    {
        TArray<FTransform> Transforms;
        TArray<FColor> Colors;
        for (const auto& Point : request_.data())
        {
            Transforms.Add(FTransform(
                FRotator(0, 0, 0),
                FVector(
                    Point.point().x() * 100,
                    -Point.point().y() * 100,
                    Point.point().z() * 100)
            ));
            Colors.Add(FColor(Point.color()));
        }
        component_->QueueInstanceUpdate(Transforms, Colors);
        response_.set_status(true);
    }
}

GetDronePoseCall::GetDronePoseCall(
    VCCSim::DroneService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    const std::map<std::string, ADronePawn*>& rcmap)
        : AsyncCallTemplateM(service, cq, rcmap)
{
    Proceed(true);
}

void GetDronePoseCall::PrepareNextCall()
{
    new GetDronePoseCall(service_, cq_, RCMap_);
}

void GetDronePoseCall::InitializeRequest()
{
    service_->RequestGetDronePose(
        &ctx_, &request_, &responder_, cq_, cq_, this);
}

void GetDronePoseCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        const auto Drone = RCMap_[request_.name()];
        const FVector Location = Drone->GetActorLocation();
        const FRotator Rotation = Drone->GetActorRotation();

        VCCSim::Vec3f* Position = response_.mutable_position();
        Position->set_x(Location.X);
        Position->set_y(Location.Y);
        Position->set_z(Location.Z);

        VCCSim::Rotation* Rot = response_.mutable_rotation();
        FQuat Quat = Rotation.Quaternion();
        Rot->set_x(Quat.X);
        Rot->set_y(Quat.Y);
        Rot->set_z(Quat.Z);
        Rot->set_w(Quat.W);
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("GetDroneOdomCall: "
                                      "Drone not found!"));
    }
}

SendDronePoseCall::SendDronePoseCall(
    VCCSim::DroneService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    const std::map<std::string, ADronePawn*>& rcmap)
        : AsyncCallTemplateM(service, cq, rcmap)
{
    Proceed(true);
}

void SendDronePoseCall::PrepareNextCall()
{
    new SendDronePoseCall(service_, cq_, RCMap_);
}

void SendDronePoseCall::InitializeRequest()
{
    service_->RequestSendDronePose(
        &ctx_, &request_, &responder_, cq_, cq_, this);
}

void SendDronePoseCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        if (ADronePawn* Drone = RCMap_[request_.name()])
        {
            const VCCSim::Pose& PoseData = request_.pose();
            const FVector TargetLocation(
                PoseData.position().x(),
                PoseData.position().y(),
                PoseData.position().z()
            );
            const FQuat TargetQuat(
                PoseData.rotation().x(),
                PoseData.rotation().y(),
                PoseData.rotation().z(),
                PoseData.rotation().w()
            );
            const FRotator TargetRotation = TargetQuat.Rotator();
            if (!Drone->IfCloseToTarget(TargetLocation, TargetRotation))
            {
                Drone->SetTarget(TargetLocation, TargetRotation);
            }
            response_.set_status(true);
        }
        else
        {
            UE_LOG(LogGRPCCall, Warning, TEXT("SendDronePoseCall: "
                                          "AQuadcopterDrone not found!"));
            response_.set_status(false);
        }
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("SendDronePoseCall: "
                                      "Drone not found!"));
        response_.set_status(false);
    }
}

SendDronePathCall::SendDronePathCall(
    VCCSim::DroneService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    const std::map<std::string, ADronePawn*>& rcmap)
        : AsyncCallTemplateM(service, cq, rcmap)
{
    Proceed(true);
}

void SendDronePathCall::PrepareNextCall()
{
    new SendDronePathCall(service_, cq_, RCMap_);
}

void SendDronePathCall::InitializeRequest()
{
    service_->RequestSendDronePath(
        &ctx_, &request_, &responder_, cq_, cq_, this);
}

void SendDronePathCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        if (ADronePawn* Drone = RCMap_[request_.name()])
        {
            TArray<FVector> Positions;
            TArray<FRotator> Rotations;
            for (const auto& PoseData : request_.path())
            {
                Positions.Add(FVector(
                    PoseData.position().x(),
                    PoseData.position().y(),
                    PoseData.position().z()));

                FQuat Quat(
                    PoseData.rotation().x(),
                    PoseData.rotation().y(),
                    PoseData.rotation().z(),
                    PoseData.rotation().w());
                Rotations.Add(Quat.Rotator());
            }
            Drone->SetPath(Positions, Rotations);
            response_.set_status(true);
        }
        else
        {
            UE_LOG(LogGRPCCall, Warning, TEXT("SendDronePathCall: "
                                          "AQuadcopterDrone not found!"));
            response_.set_status(false);
        }
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("SendDronePathCall: "
                                      "Drone not found!"));
        response_.set_status(false);
    }
}

GetCarOdomCall::GetCarOdomCall(
    VCCSim::CarService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    const std::map<std::string, ACarPawn*>& rcmap)
    : AsyncCallTemplateM(service, cq, rcmap)
{
    Proceed(true);
}

void GetCarOdomCall::PrepareNextCall()
{
    new GetCarOdomCall(service_, cq_, RCMap_);
}

void GetCarOdomCall::InitializeRequest()
{
    service_->RequestGetCarOdom(
        &ctx_, &request_, &responder_, cq_, cq_, this);
}

void GetCarOdomCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        const auto Car = RCMap_[request_.name()];
        const FVector Loc = Car->GetActorLocation();
        const FRotator Rot = Car->GetActorRotation();

        const FVector LinearVelocity = Car->GetPhysicsLinearVelocity();
        const FVector AngularVelocity = Car->GetPhysicsAngularVelocityInDegrees();

        VCCSim::Pose* PoseData = response_.mutable_pose();
        VCCSim::Vec3f* Position = PoseData->mutable_position();
        Position->set_x(Loc.X);
        Position->set_y(Loc.Y);
        Position->set_z(Loc.Z);

        VCCSim::Rotation* Rotation = PoseData->mutable_rotation();
        FQuat Quat = Rot.Quaternion();
        Rotation->set_x(Quat.X);
        Rotation->set_y(Quat.Y);
        Rotation->set_z(Quat.Z);
        Rotation->set_w(Quat.W);

        VCCSim::Twist* TwistData = response_.mutable_twist();
        VCCSim::Vec3f* LinearVel = TwistData->mutable_linear();
        LinearVel->set_x(LinearVelocity.X);
        LinearVel->set_y(LinearVelocity.Y);
        LinearVel->set_z(LinearVelocity.Z);

        VCCSim::Vec3f* AngularVel = TwistData->mutable_angular();
        AngularVel->set_x(AngularVelocity.X);
        AngularVel->set_y(AngularVelocity.Y);
        AngularVel->set_z(AngularVelocity.Z);
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("GetCarOdomCall: "
            "Car not found!"));
    }
}


SendCarPoseCall::SendCarPoseCall(
    VCCSim::CarService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    const std::map<std::string, ACarPawn*>& rcmap)
    : AsyncCallTemplateM(service, cq, rcmap)
{
    Proceed(true);
}

void SendCarPoseCall::PrepareNextCall()
{
    new SendCarPoseCall(service_, cq_, RCMap_);
}

void SendCarPoseCall::InitializeRequest()
{
    service_->RequestSendCarPose(
        &ctx_, &request_, &responder_, cq_, cq_, this);
}

void SendCarPoseCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        if (ACarPawn* Car = RCMap_[request_.name()])
        {
            const VCCSim::PoseYawOnly& PoseData = request_.pose();
            const FVector TargetLocation(
                PoseData.position().x(),
                PoseData.position().y(),
                PoseData.position().z()
            );
            const FRotator TargetRotation(
                0.0f,
                PoseData.yaw(),
                0.0f
            );
            Car->SetTarget(TargetLocation, TargetRotation);
            response_.set_status(true);
        }
        else
        {
            UE_LOG(LogGRPCCall, Warning, TEXT("SendCarPoseCall: "
                "Car not found!"));
            response_.set_status(false);
        }
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("SendCarPoseCall: "
            "Car not found!"));
        response_.set_status(false);
    }
}

SendCarPathCall::SendCarPathCall(
    VCCSim::CarService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    const std::map<std::string, ACarPawn*>& rcmap)
    : AsyncCallTemplateM(service, cq, rcmap)
{
    Proceed(true);
}

void SendCarPathCall::PrepareNextCall()
{
    new SendCarPathCall(service_, cq_, RCMap_);
}

void SendCarPathCall::InitializeRequest()
{
    service_->RequestSendCarPath(
        &ctx_, &request_, &responder_, cq_, cq_, this);
}

void SendCarPathCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        if (ACarPawn* Car = RCMap_[request_.name()])
        {
            TArray<FVector> Positions;
            TArray<FRotator> Rotations;
            for (const auto& PoseData : request_.path())
            {
                Positions.Add(FVector(
                    PoseData.position().x(),
                    PoseData.position().y(),
                    PoseData.position().z()));
                Rotations.Add(FRotator(0.0f, PoseData.yaw(), 0.0f)); // Only Yaw is used
            }
            Car->SetPath(Positions, Rotations);
            response_.set_status(true);
        }
        else
        {
            UE_LOG(LogGRPCCall, Warning, TEXT("SendCarPathCall: "
                "Car not found!"));
            response_.set_status(false);
        }
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("SendCarPathCall: "
            "Car not found!"));
        response_.set_status(false);
    }
}