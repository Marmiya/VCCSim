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
    VCCSim::RecordingService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue, ARecorder* Recorder)
        : AsyncCallTemplate(Service, CompletionQueue, Recorder)
{
    Proceed(true);
}

void SimRecording::PrepareNextCall()
{
    new SimRecording(Service, CompletionQueue, component_);
}

void SimRecording::InitializeRequest()
{
    Service->RequestRecording(&Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void SimRecording::ProcessRequest()
{
    component_->ToggleRecording();
    Response.set_status(component_->RecordState);
}

LidarGetDataCall::LidarGetDataCall(
    VCCSim::LidarService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, ULidarComponent*>& LiDARComponentMap)
    : AsyncCallTemplateM(Service, CompletionQueue, LiDARComponentMap)
{
    Proceed(true);
}

void LidarGetDataCall::PrepareNextCall()
{
    new LidarGetDataCall(Service, CompletionQueue, RCMap_);
}

void LidarGetDataCall::InitializeRequest()
{
    Service->RequestGetLiDARData(&Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void LidarGetDataCall::ProcessRequest()
{
    if (RCMap_.contains(Request.name()))
    {
        for (const auto& Point : RCMap_[Request.name()]->GetPointCloudData())
        {
            VCCSim::Vec3f* LidarPoint = Response.add_data();
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
        VCCSim::LidarService::AsyncService* Service,
        grpc::ServerCompletionQueue* CompletionQueue,
        const std::map<std::string, ULidarComponent*>& LiDARComponentMap)
    : AsyncCallTemplateM(Service, CompletionQueue, LiDARComponentMap)
{
    Proceed(true);
}

void LidarGetOdomCall::PrepareNextCall()
{
    new LidarGetOdomCall(Service, CompletionQueue, RCMap_);
}

void LidarGetOdomCall::InitializeRequest()
{
    Service->RequestGetLiDAROdom(&Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void LidarGetOdomCall::ProcessRequest()
{
    if (RCMap_.contains(Request.name()))
    {
        const auto Lidar = RCMap_[Request.name()];
        const FVector Location = Lidar->GetComponentLocation();
        const FRotator Rotation = Lidar->GetComponentRotation();

        FVector LinearVelocity = FVector::ZeroVector;
        FVector AngularVelocity = FVector::ZeroVector;

        if (AActor* Owner = Lidar->GetOwner())
        {
            if (UPrimitiveComponent* RootPrim = Cast<UPrimitiveComponent>(Owner->GetRootComponent()))
            {
                LinearVelocity = RootPrim->GetPhysicsLinearVelocity();
                AngularVelocity = RootPrim->GetPhysicsAngularVelocityInDegrees();
            }
        }

        VCCSim::Pose* PoseData = Response.mutable_pose();
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

        VCCSim::Twist* TwistData = Response.mutable_twist();
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
    VCCSim::LidarService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, ULidarComponent*>& LiDARComponentMap)
    : AsyncCallTemplateM(Service, CompletionQueue, LiDARComponentMap)
{
    Proceed(true);
}

void LidarGetDataAndOdomCall::PrepareNextCall()
{
    new LidarGetDataAndOdomCall(Service, CompletionQueue, RCMap_);
}

void LidarGetDataAndOdomCall::InitializeRequest()
{
    Service->RequestGetLiDARDataAndOdom(&Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void LidarGetDataAndOdomCall::ProcessRequest()
{
    if (RCMap_.contains(Request.name()))
    {
        auto DataAndOdom =
            RCMap_[Request.name()]->GetPointCloudDataAndOdom();
        const auto& Odom = DataAndOdom.Value;
        
        for (const auto& Point : DataAndOdom.Key)
        {
            VCCSim::Vec3f* LidarPoint = Response.mutable_data()->add_data();
            LidarPoint->set_x(Point.X);
            LidarPoint->set_y(Point.Y);
            LidarPoint->set_z(Point.Z);
        }

        VCCSim::Pose* PoseData = Response.mutable_odom()->mutable_pose();
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

        VCCSim::Twist* TwistData = Response.mutable_odom()->mutable_twist();
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

/* --------------------------Camera Service---------------------------------- */

CameraGetRGBDataCall::CameraGetRGBDataCall(
    VCCSim::CameraService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, URGBCameraComponent*>& RGBComponentMap)
        : AsyncCallTemplateImage(Service, CompletionQueue, RGBComponentMap)
{
    Proceed(true);
}

void CameraGetRGBDataCall::PrepareNextCall()
{
    new CameraGetRGBDataCall(Service, CompletionQueue, RCMap_);
}

void CameraGetRGBDataCall::InitializeRequest()
{
    Service->RequestGetRGBData(&Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void CameraGetRGBDataCall::ProcessRequest()
{
    std::string CameraName = Request.robot_name().name() + "^" +
                             std::to_string(Request.index());

    if (!RCMap_.contains(CameraName))
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("Cannot find rgb camera map!"));
        return;
    }

    auto* RGBCamera = RCMap_[CameraName];
    if (!RGBCamera)
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("Invalid rgb camera reference!"));
        return;
    }

    RGBCamera->AsyncGetRGBImageData(
        [this, RGBCamera](const TArray<FColor>& RGBData)
    {
        AsyncTask(ENamedThreads::AnyBackgroundHiPriTask,
            [this, RGBData, RGBCamera]()
        {
            Response.set_width(RGBCamera->Width);
            Response.set_height(RGBCamera->Height);
            Response.set_timestamp(FDateTime::UtcNow().ToUnixTimestamp());
            Response.set_data(RGBData.GetData(), RGBData.Num());
            Status = FINISH;
            Responder.Finish(Response, grpc::Status::OK, this);
        });
    });
}

CameraGetDepthDataCall::CameraGetDepthDataCall(
    VCCSim::CameraService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, UDepthCameraComponent*>& DepthComponentMap)
        : AsyncCallTemplateImage(Service, CompletionQueue, DepthComponentMap)
{
    Proceed(true);
}

void CameraGetDepthDataCall::PrepareNextCall()
{
    new CameraGetDepthDataCall(Service, CompletionQueue, RCMap_);
}

void CameraGetDepthDataCall::InitializeRequest()
{
    Service->RequestGetDepthData(&Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void CameraGetDepthDataCall::ProcessRequest()
{
    std::string CameraName = Request.robot_name().name() + "^" +
                             std::to_string(Request.index());

    if (!RCMap_.contains(CameraName))
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("Cannot find depth camera map!"));
        return;
    }

    auto* RGBDCamera = RCMap_[CameraName];
    if (!RGBDCamera)
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("Invalid depth camera reference!"));
        return;
    }

    RGBDCamera->AsyncGetDepthImageData(
        [this, RGBDCamera](const TArray<FFloat16Color>& RGBDData)
    {
        AsyncTask(ENamedThreads::AnyBackgroundHiPriTask,
            [this, RGBDData, RGBDCamera]()
        {
            Response.set_width(RGBDCamera->Width);
            Response.set_height(RGBDCamera->Height);
            Response.set_timestamp(FDateTime::UtcNow().ToUnixTimestamp());

            // Extract depth from alpha channel
            Response.mutable_data()->Reserve(RGBDData.Num());
            for (const FFloat16Color& Pixel : RGBDData)
            {
                Response.add_data(Pixel.R);
            }

            Status = FINISH;
            Responder.Finish(Response, grpc::Status::OK, this);
        });
    });
}

CameraGetDepthPointCloudCall::CameraGetDepthPointCloudCall(
    VCCSim::CameraService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, UDepthCameraComponent*>& DepthComponentMap)
        : AsyncCallTemplateImage(Service, CompletionQueue, DepthComponentMap)
{
    Proceed(true);
}

void CameraGetDepthPointCloudCall::PrepareNextCall()
{
    new CameraGetDepthPointCloudCall(Service, CompletionQueue, RCMap_);
}

void CameraGetDepthPointCloudCall::InitializeRequest()
{
    Service->RequestGetDepthPointCloud(&Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void CameraGetDepthPointCloudCall::ProcessRequest()
{
    std::string CameraName = Request.robot_name().name() + "^" +
                             std::to_string(Request.index());

    if (!RCMap_.contains(CameraName))
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("Cannot find depth camera map!"));
        return;
    }

    auto* DepthCamera = RCMap_[CameraName];
    if (!DepthCamera)
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("Invalid depth camera reference!"));
        return;
    }

    DepthCamera->AsyncGetPointCloudData(
        [this, DepthCamera]()
    {
        AsyncTask(ENamedThreads::AnyBackgroundHiPriTask,
            [this, DepthCamera]()
        {
            const auto PointCloudData = DepthCamera->GeneratePointCloud();
            if (PointCloudData.Num() == 0)
            {
                UE_LOG(LogGRPCCall, Warning, TEXT("No depth point cloud available!"));
                Status = FINISH;
                Responder.Finish(Response, grpc::Status::CANCELLED, this);
                return;
            }

            Response.set_timestamp(FDateTime::UtcNow().ToUnixTimestamp());

            for (const FDCPoint& Point : PointCloudData)
            {
                VCCSim::Vec3f* point = Response.mutable_data()->add_data();
                point->set_x(Point.Location.X);
                point->set_y(Point.Location.Y);
                point->set_z(Point.Location.Z);
            }

            Status = FINISH;
            Responder.Finish(Response, grpc::Status::OK, this);
        });
    });
}


CameraGetSegmentDataCall::CameraGetSegmentDataCall(
    VCCSim::CameraService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, USegCameraComponent*>& SegComponentMap)
    : AsyncCallTemplateImage(Service, CompletionQueue, SegComponentMap)
{
    Proceed(true);
}

void CameraGetSegmentDataCall::PrepareNextCall()
{
    new CameraGetSegmentDataCall(Service, CompletionQueue, RCMap_);
}

void CameraGetSegmentDataCall::InitializeRequest()
{
    Service->RequestGetSegData(&Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void CameraGetSegmentDataCall::ProcessRequest()
{
    const std::string CameraName = Request.robot_name().name() + "^" +
                             std::to_string(Request.index());

    if (!RCMap_.contains(CameraName))
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("Cannot find seg camera map!"));
        return;
    }

    auto* SegCamera = RCMap_[CameraName];
    if (!SegCamera)
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("Invalid seg camera reference!"));
        return;
    }

    SegCamera->AsyncGetSegmentationImageData(
        [this, SegCamera](const TArray<FColor>& SegData)
    {
        AsyncTask(ENamedThreads::AnyBackgroundHiPriTask,
            [this, SegData, SegCamera]()
        {
            Response.set_width(SegCamera->Width);
            Response.set_height(SegCamera->Height);
            Response.set_timestamp(FDateTime::UtcNow().ToUnixTimestamp());
            Response.set_data(SegData.GetData(), SegData.Num());
            Status = FINISH;
            Responder.Finish(Response, grpc::Status::OK, this);
        });
    });
}


CameraGetNormalDataCall::CameraGetNormalDataCall(
    VCCSim::CameraService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, UNormalCameraComponent*>& NormalComponentMap)
        : AsyncCallTemplateImage(Service, CompletionQueue, NormalComponentMap)
{
    Proceed(true);
}

void CameraGetNormalDataCall::PrepareNextCall()
{
    new CameraGetNormalDataCall(Service, CompletionQueue, RCMap_);
}

void CameraGetNormalDataCall::InitializeRequest()
{
    Service->RequestGetNormalData(&Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void CameraGetNormalDataCall::ProcessRequest()
{
    const std::string CameraName = Request.robot_name().name() + "^" +
                             std::to_string(Request.index());

    if (!RCMap_.contains(CameraName))
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("Cannot find normal camera map!"));
        return;
    }

    auto* NormalCamera = RCMap_[CameraName];
    if (!NormalCamera)
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("Invalid normal camera reference!"));
        return;
    }

    NormalCamera->AsyncGetNormalImageData(
        [this, NormalCamera](const TArray<FFloat16Color>& NormalData)
    {
        AsyncTask(ENamedThreads::AnyBackgroundHiPriTask,
            [this, NormalData, NormalCamera]()
        {
            Response.set_width(NormalCamera->Width);
            Response.set_height(NormalCamera->Height);
            Response.set_timestamp(FDateTime::UtcNow().ToUnixTimestamp());
                Response.mutable_data()->Reserve(NormalCamera->Width * NormalCamera->Height);

                for (int i = 0; i < NormalData.Num(); ++i)
                {
                    VCCSim::Vec3f* Point = Response.mutable_data()->Mutable(i);
                    Point->set_x(NormalData[i].R);
                    Point->set_y(NormalData[i].G);
                    Point->set_z(NormalData[i].B);
                }
            Status = FINISH;
            Responder.Finish(Response, grpc::Status::OK, this);
        });
    });
}


SendMeshCall::SendMeshCall(VCCSim::MeshService::AsyncService* Service,
                           grpc::ServerCompletionQueue* CompletionQueue,
                           UMeshHandlerComponent* mesh_component)
    : AsyncCallTemplate(Service, CompletionQueue, mesh_component)
{
    Proceed(true);
}

void SendMeshCall::PrepareNextCall()
{
    new SendMeshCall(Service, CompletionQueue, component_);
}

void SendMeshCall::InitializeRequest()
{
    Service->RequestSendMesh(&Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void SendMeshCall::ProcessRequest()
{
    if (component_)
    {
        const VCCSim::Pose& Transform = Request.transform();
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
            reinterpret_cast<const uint8*>(Request.data().data()),
            Request.data().size(),
            MeshTransform
        );
    }
    else
    {
        UE_LOG(LogGRPCCall, Error, TEXT("SendMeshCall: "
                                    "Mesh component not found!"));
    }
    Response.set_status(true);
}

SendGlobalMeshCall::SendGlobalMeshCall(
    VCCSim::MeshService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    UFMeshManager* MeshManager)
        : AsyncCallTemplate(Service, CompletionQueue, MeshManager)
{
    Proceed(true);
}

void SendGlobalMeshCall::PrepareNextCall()
{
    new SendGlobalMeshCall(Service, CompletionQueue, component_);
}

void SendGlobalMeshCall::InitializeRequest()
{
    Service->RequestSendGlobalMesh(&Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void SendGlobalMeshCall::ProcessRequest()
{
    if (component_)
    {
        const VCCSim::Pose& Transform = Request.transform();
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
            reinterpret_cast<const uint8*>(Request.data().data()),
            Request.data().size(),
            MeshTransform))
        {
            Response.set_id(ID);
        }
        else
        {
            Response.set_id(-1);
        }
    }
    else
    {
        UE_LOG(LogGRPCCall, Error, TEXT("SendGlobalMeshCall: "
                                    "Mesh manager not found!"));
    }
}

RemoveGlobalMeshCall::RemoveGlobalMeshCall(
    VCCSim::MeshService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    UFMeshManager* MeshManager)
        : AsyncCallTemplate(Service, CompletionQueue, MeshManager)
{
    Proceed(true);
}

void RemoveGlobalMeshCall::PrepareNextCall()
{
    new RemoveGlobalMeshCall(Service, CompletionQueue, component_);
}

void RemoveGlobalMeshCall::InitializeRequest()
{
    Service->RequestRemoveGlobalMesh(&Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void RemoveGlobalMeshCall::ProcessRequest()
{
    if (component_)
    {
        Response.set_status(component_->RemoveGlobalMesh(Request.id()));
    }
    else
    {
        UE_LOG(LogGRPCCall, Error, TEXT("RemoveGlobalMeshCall: "
                                    "Mesh manager not found!"));
    }
}

SendPointCloudWithColorCall::SendPointCloudWithColorCall(
    VCCSim::PointCloudService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue, UInsMeshHolder* mesh_holder)
    : AsyncCallTemplate(Service, CompletionQueue, mesh_holder)
{
    Proceed(true);
}

void SendPointCloudWithColorCall::PrepareNextCall()
{
    new SendPointCloudWithColorCall(Service, CompletionQueue, component_);
}

void SendPointCloudWithColorCall::InitializeRequest()
{
    Service->RequestSendPointCloudWithColor(
        &Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void SendPointCloudWithColorCall::ProcessRequest()
{
    if (component_)
    {
        TArray<FTransform> Transforms;
        TArray<FColor> Colors;
        for (const auto& Point : Request.data())
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
        Response.set_status(true);
    }
}

GetDronePoseCall::GetDronePoseCall(
    VCCSim::DroneService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, ADronePawn*>& rcmap)
        : AsyncCallTemplateM(Service, CompletionQueue, rcmap)
{
    Proceed(true);
}

void GetDronePoseCall::PrepareNextCall()
{
    new GetDronePoseCall(Service, CompletionQueue, RCMap_);
}

void GetDronePoseCall::InitializeRequest()
{
    Service->RequestGetDronePose(
        &Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void GetDronePoseCall::ProcessRequest()
{
    if (RCMap_.contains(Request.name()))
    {
        const auto Drone = RCMap_[Request.name()];
        const FVector Location = Drone->GetActorLocation();
        const FRotator Rotation = Drone->GetActorRotation();

        VCCSim::Vec3f* Position = Response.mutable_position();
        Position->set_x(Location.X);
        Position->set_y(Location.Y);
        Position->set_z(Location.Z);

        VCCSim::Rotation* Rot = Response.mutable_rotation();
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
    VCCSim::DroneService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, ADronePawn*>& rcmap)
        : AsyncCallTemplateM(Service, CompletionQueue, rcmap)
{
    Proceed(true);
}

void SendDronePoseCall::PrepareNextCall()
{
    new SendDronePoseCall(Service, CompletionQueue, RCMap_);
}

void SendDronePoseCall::InitializeRequest()
{
    Service->RequestSendDronePose(
        &Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void SendDronePoseCall::ProcessRequest()
{
    if (RCMap_.contains(Request.name()))
    {
        if (ADronePawn* Drone = RCMap_[Request.name()])
        {
            const VCCSim::Pose& PoseData = Request.pose();
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
            Response.set_status(true);
        }
        else
        {
            UE_LOG(LogGRPCCall, Warning, TEXT("SendDronePoseCall: "
                                          "AQuadcopterDrone not found!"));
            Response.set_status(false);
        }
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("SendDronePoseCall: "
                                      "Drone not found!"));
        Response.set_status(false);
    }
}

SendDronePathCall::SendDronePathCall(
    VCCSim::DroneService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, ADronePawn*>& rcmap)
        : AsyncCallTemplateM(Service, CompletionQueue, rcmap)
{
    Proceed(true);
}

void SendDronePathCall::PrepareNextCall()
{
    new SendDronePathCall(Service, CompletionQueue, RCMap_);
}

void SendDronePathCall::InitializeRequest()
{
    Service->RequestSendDronePath(
        &Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void SendDronePathCall::ProcessRequest()
{
    if (RCMap_.contains(Request.name()))
    {
        if (ADronePawn* Drone = RCMap_[Request.name()])
        {
            TArray<FVector> Positions;
            TArray<FRotator> Rotations;
            for (const auto& PoseData : Request.path())
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
            Response.set_status(true);
        }
        else
        {
            UE_LOG(LogGRPCCall, Warning, TEXT("SendDronePathCall: "
                                          "AQuadcopterDrone not found!"));
            Response.set_status(false);
        }
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("SendDronePathCall: "
                                      "Drone not found!"));
        Response.set_status(false);
    }
}

GetCarOdomCall::GetCarOdomCall(
    VCCSim::CarService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, ACarPawn*>& rcmap)
    : AsyncCallTemplateM(Service, CompletionQueue, rcmap)
{
    Proceed(true);
}

void GetCarOdomCall::PrepareNextCall()
{
    new GetCarOdomCall(Service, CompletionQueue, RCMap_);
}

void GetCarOdomCall::InitializeRequest()
{
    Service->RequestGetCarOdom(
        &Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void GetCarOdomCall::ProcessRequest()
{
    if (RCMap_.contains(Request.name()))
    {
        const auto Car = RCMap_[Request.name()];
        const FVector Loc = Car->GetActorLocation();
        const FRotator Rot = Car->GetActorRotation();

        const FVector LinearVelocity = Car->GetPhysicsLinearVelocity();
        const FVector AngularVelocity = Car->GetPhysicsAngularVelocityInDegrees();

        VCCSim::Pose* PoseData = Response.mutable_pose();
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

        VCCSim::Twist* TwistData = Response.mutable_twist();
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
    VCCSim::CarService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, ACarPawn*>& rcmap)
    : AsyncCallTemplateM(Service, CompletionQueue, rcmap)
{
    Proceed(true);
}

void SendCarPoseCall::PrepareNextCall()
{
    new SendCarPoseCall(Service, CompletionQueue, RCMap_);
}

void SendCarPoseCall::InitializeRequest()
{
    Service->RequestSendCarPose(
        &Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void SendCarPoseCall::ProcessRequest()
{
    if (RCMap_.contains(Request.name()))
    {
        if (ACarPawn* Car = RCMap_[Request.name()])
        {
            const VCCSim::PoseYawOnly& PoseData = Request.pose();
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
            Response.set_status(true);
        }
        else
        {
            UE_LOG(LogGRPCCall, Warning, TEXT("SendCarPoseCall: "
                "Car not found!"));
            Response.set_status(false);
        }
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("SendCarPoseCall: "
            "Car not found!"));
        Response.set_status(false);
    }
}

SendCarPathCall::SendCarPathCall(
    VCCSim::CarService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, ACarPawn*>& rcmap)
    : AsyncCallTemplateM(Service, CompletionQueue, rcmap)
{
    Proceed(true);
}

void SendCarPathCall::PrepareNextCall()
{
    new SendCarPathCall(Service, CompletionQueue, RCMap_);
}

void SendCarPathCall::InitializeRequest()
{
    Service->RequestSendCarPath(
        &Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void SendCarPathCall::ProcessRequest()
{
    if (RCMap_.contains(Request.name()))
    {
        if (ACarPawn* Car = RCMap_[Request.name()])
        {
            TArray<FVector> Positions;
            TArray<FRotator> Rotations;
            for (const auto& PoseData : Request.path())
            {
                Positions.Add(FVector(
                    PoseData.position().x(),
                    PoseData.position().y(),
                    PoseData.position().z()));
                Rotations.Add(FRotator(0.0f, PoseData.yaw(), 0.0f)); // Only Yaw is used
            }
            Car->SetPath(Positions, Rotations);
            Response.set_status(true);
        }
        else
        {
            UE_LOG(LogGRPCCall, Warning, TEXT("SendCarPathCall: "
                "Car not found!"));
            Response.set_status(false);
        }
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("SendCarPathCall: "
            "Car not found!"));
        Response.set_status(false);
    }
}

/* --------------------------Flash Handler---------------------------------- */

GetFlashPoseCall::GetFlashPoseCall(
    VCCSim::FlashService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, AFlashPawn*>& rcmap)
        : AsyncCallTemplateM(Service, CompletionQueue, rcmap)
{
    Proceed(true);
}

void GetFlashPoseCall::PrepareNextCall()
{
    new GetFlashPoseCall(Service, CompletionQueue, RCMap_);
}

void GetFlashPoseCall::InitializeRequest()
{
    Service->RequestGetFlashPose(
        &Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void GetFlashPoseCall::ProcessRequest()
{
    if (RCMap_.contains(Request.name()))
    {
        const auto Flash = RCMap_[Request.name()];
        const FVector Location = Flash->GetActorLocation();
        const FRotator Rotation = Flash->GetActorRotation();

        VCCSim::Vec3f* Position = Response.mutable_position();
        Position->set_x(Location.X);
        Position->set_y(Location.Y);
        Position->set_z(Location.Z);

        VCCSim::Rotation* Rot = Response.mutable_rotation();
        FQuat Quat = Rotation.Quaternion();
        Rot->set_x(Quat.X);
        Rot->set_y(Quat.Y);
        Rot->set_z(Quat.Z);
        Rot->set_w(Quat.W);
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("GetFlashPoseCall: "
                                      "Flash not found!"));
    }
}

SendFlashPoseCall::SendFlashPoseCall(
    VCCSim::FlashService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, AFlashPawn*>& rcmap)
        : AsyncCallTemplateM(Service, CompletionQueue, rcmap)
{
    Proceed(true);
}

void SendFlashPoseCall::PrepareNextCall()
{
    new SendFlashPoseCall(Service, CompletionQueue, RCMap_);
}

void SendFlashPoseCall::InitializeRequest()
{
    Service->RequestSendFlashPose(
        &Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void SendFlashPoseCall::ProcessRequest()
{
    if (RCMap_.contains(Request.name()))
    {
        if (AFlashPawn* Flash = RCMap_[Request.name()])
        {
            const VCCSim::Pose& PoseData = Request.pose();
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
            Flash->SetTarget(TargetLocation, TargetRotation);
            Response.set_status(true);
        }
        else
        {
            UE_LOG(LogGRPCCall, Warning, TEXT("SendFlashPoseCall: "
                                          "Flash not found!"));
            Response.set_status(false);
        }
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("SendFlashPoseCall: "
                                      "Flash not found!"));
        Response.set_status(false);
    }
}

SendFlashPathCall::SendFlashPathCall(
    VCCSim::FlashService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, AFlashPawn*>& rcmap)
        : AsyncCallTemplateM(Service, CompletionQueue, rcmap)
{
    Proceed(true);
}

void SendFlashPathCall::PrepareNextCall()
{
    new SendFlashPathCall(Service, CompletionQueue, RCMap_);
}

void SendFlashPathCall::InitializeRequest()
{
    Service->RequestSendFlashPath(
        &Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void SendFlashPathCall::ProcessRequest()
{
    if (RCMap_.contains(Request.name()))
    {
        if (AFlashPawn* Flash = RCMap_[Request.name()])
        {
            TArray<FVector> Positions;
            TArray<FRotator> Rotations;
            for (const auto& PoseData : Request.path())
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
            Flash->SetPath(Positions, Rotations);
            Response.set_status(true);
        }
        else
        {
            UE_LOG(LogGRPCCall, Warning, TEXT("SendFlashPathCall: "
                                          "Flash not found!"));
            Response.set_status(false);
        }
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("SendFlashPathCall: "
                                      "Flash not found!"));
        Response.set_status(false);
    }
}

CheckFlashReadyCall::CheckFlashReadyCall(
    VCCSim::FlashService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, AFlashPawn*>& rcmap)
        : AsyncCallTemplateM(Service, CompletionQueue, rcmap)
{
    Proceed(true);
}

void CheckFlashReadyCall::PrepareNextCall()
{
    new CheckFlashReadyCall(Service, CompletionQueue, RCMap_);
}

void CheckFlashReadyCall::InitializeRequest()
{
    Service->RequestCheckReady(
        &Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void CheckFlashReadyCall::ProcessRequest()
{
    if (RCMap_.contains(Request.name()))
    {
        if (AFlashPawn* Flash = RCMap_[Request.name()])
        {
            Response.set_status(Flash->IsReady());
        }
        else
        {
            UE_LOG(LogGRPCCall, Warning, TEXT("CheckFlashReadyCall: "
                                          "Flash not found!"));
            Response.set_status(false);
        }
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("CheckFlashReadyCall: "
                                      "Flash not found!"));
        Response.set_status(false);
    }
}

FlashMoveToNextCall::FlashMoveToNextCall(
    VCCSim::FlashService::AsyncService* Service,
    grpc::ServerCompletionQueue* CompletionQueue,
    const std::map<std::string, AFlashPawn*>& rcmap)
        : AsyncCallTemplateM(Service, CompletionQueue, rcmap)
{
    Proceed(true);
}

void FlashMoveToNextCall::PrepareNextCall()
{
    new FlashMoveToNextCall(Service, CompletionQueue, RCMap_);
}

void FlashMoveToNextCall::InitializeRequest()
{
    Service->RequestMoveToNext(
        &Context, &Request, &Responder, CompletionQueue, CompletionQueue, this);
}

void FlashMoveToNextCall::ProcessRequest()
{
    if (RCMap_.contains(Request.name()))
    {
        if (AFlashPawn* Flash = RCMap_[Request.name()])
        {
            Flash->MoveToNext();
            Response.set_status(true);
        }
        else
        {
            UE_LOG(LogGRPCCall, Warning, TEXT("FlashMoveToNextCall: "
                                          "Flash not found!"));
            Response.set_status(false);
        }
    }
    else
    {
        UE_LOG(LogGRPCCall, Warning, TEXT("FlashMoveToNextCall: "
                                      "Flash not found!"));
        Response.set_status(false);
    }
}