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

#include "API/GRPCCall.h"
#include "Sensors/LidarSensor.h"
#include "Sensors/DepthCamera.h"
#include "Sensors/CameraSensor.h"
#include "Sensors/SegmentCamera.h"
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
    std::map<std::string, ULidarComponent*> RLMap)
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
            VCCSim::Point* LidarPoint = response_.add_data();
            LidarPoint->set_x(Point.X);
            LidarPoint->set_y(Point.Y);
            LidarPoint->set_z(Point.Z);
        }
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("LidarGetDataCall: "
                                      "Lidar component not found!"));
    }
}

LidarGetOdomCall::LidarGetOdomCall(
        VCCSim::LidarService::AsyncService* service,
        grpc::ServerCompletionQueue* cq,
        std::map<std::string, ULidarComponent*> RLMap)
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
        PoseData->set_x(Location.X);
        PoseData->set_y(Location.Y);
        PoseData->set_z(Location.Z);
        PoseData->set_roll(Rotation.Roll);
        PoseData->set_pitch(Rotation.Pitch);
        PoseData->set_yaw(Rotation.Yaw);

        VCCSim::twist* TwistData = response_.mutable_twist();
        TwistData->set_linear_x(LinearVelocity.X);
        TwistData->set_linear_y(LinearVelocity.Y);
        TwistData->set_linear_z(LinearVelocity.Z);
        TwistData->set_angular_x(AngularVelocity.X);
        TwistData->set_angular_y(AngularVelocity.Y);
        TwistData->set_angular_z(AngularVelocity.Z);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("LidarGetOdomCall: "
                                      "Lidar component not found!"));
    }
}

LidarGetDataAndOdomCall::LidarGetDataAndOdomCall(
    VCCSim::LidarService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    std::map<std::string, ULidarComponent*> rcmap)
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
            VCCSim::Point* LidarPoint = response_.mutable_data()->add_data();
            LidarPoint->set_x(Point.X);
            LidarPoint->set_y(Point.Y);
            LidarPoint->set_z(Point.Z);
        }
        
        VCCSim::Pose* PoseData = response_.mutable_odom()->mutable_pose();
        PoseData->set_x(Odom.Location.X);
        PoseData->set_y(Odom.Location.Y);
        PoseData->set_z(Odom.Location.Z);
        PoseData->set_roll(Odom.Rotation.Roll);
        PoseData->set_pitch(Odom.Rotation.Pitch);
        PoseData->set_yaw(Odom.Rotation.Yaw);
        
        VCCSim::twist* TwistData = response_.mutable_odom()->mutable_twist();
        TwistData->set_linear_x(Odom.LinearVelocity.X);
        TwistData->set_linear_y(Odom.LinearVelocity.Y);
        TwistData->set_linear_z(Odom.LinearVelocity.Z);
        TwistData->set_angular_x(Odom.AngularVelocity.X);
        TwistData->set_angular_y(Odom.AngularVelocity.Y);
        TwistData->set_angular_z(Odom.AngularVelocity.Z);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("LidarGetDataAndOdomCall: "
                                      "Lidar component not found!"));
    }
}

DepthCameraGetPointDataCall::DepthCameraGetPointDataCall(
    VCCSim::DepthCameraService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    std::map<std::string, UDepthCameraComponent*> rdcmap)
        : AsyncCallTemplateImage(service, cq, rdcmap)
{
    Proceed(true);
}

void DepthCameraGetPointDataCall::PrepareNextCall()
{
    new DepthCameraGetPointDataCall(service_, cq_, RCMap_);
}

void DepthCameraGetPointDataCall::InitializeRequest()
{
    service_->RequestGetDepthCameraPointData(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void DepthCameraGetPointDataCall::ProcessRequest()
{
    if (!RCMap_.contains(request_.name()))
    {
        UE_LOG(LogTemp, Warning, TEXT("DepthCameraGetPointDataCall: "
                                      "DepthCamera component not found!"));
        return;
    }

    auto* DepthCamera = RCMap_[request_.name()];
    if (!DepthCamera)
    {
        UE_LOG(LogTemp, Warning, TEXT("DepthCameraGetPointDataCall: "
                                      "Invalid DepthCamera reference!"));
        return;
    }

    // Use the callback version to get point cloud data
    DepthCamera->AsyncGetPointCloudData(
        [this, DepthCamera]()
    {
        // Process the points in a background thread
        AsyncTask(ENamedThreads::AnyBackgroundHiPriTask,
            [this, DepthCamera]()
        {
            const auto PointCloudData = DepthCamera->GeneratePointCloud();
            if (PointCloudData.Num() == 0)
            {
                UE_LOG(LogTemp, Warning, TEXT("DepthCameraGetPointDataCall: "
                                              "No point cloud data available!"));
                status_ = FINISH;
                responder_.Finish(response_, grpc::Status::CANCELLED, this);
                return;
            }
            // Add each point to the response
            for (const FDCPoint& Point : PointCloudData)
            {
                auto* point = response_.add_data();
                point->set_x(Point.Location.X);
                point->set_y(Point.Location.Y);
                point->set_z(Point.Location.Z);
            }

            status_ = FINISH;
            responder_.Finish(response_, grpc::Status::OK, this);
        });
    });
}

DepthCameraGetImageSizeCall::DepthCameraGetImageSizeCall(
    VCCSim::DepthCameraService::AsyncService* service,
    grpc::ServerCompletionQueue* cq, std::map<std::string,
    UDepthCameraComponent*> rdcmap)
        : AsyncCallTemplateM(service, cq, rdcmap)
{
    Proceed(true);
}

void DepthCameraGetImageSizeCall::PrepareNextCall()
{
    new DepthCameraGetImageSizeCall(service_, cq_, RCMap_);
}

void DepthCameraGetImageSizeCall::InitializeRequest()
{
    service_->RequestGetDepthCameraImageSize(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void DepthCameraGetImageSizeCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        const auto& Size = RCMap_[request_.name()]->GetImageSize();
        response_.set_width(Size.first);
        response_.set_height(Size.second);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("DepthCameraGetImageSizeCall: "
                                      "DepthCamera component not found!"));
    }
}

DepthCameraGetImageDataCall::DepthCameraGetImageDataCall(
    VCCSim::DepthCameraService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    std::map<std::string, UDepthCameraComponent*> rdcmap)
        : AsyncCallTemplateImage(service, cq, rdcmap)
{
    Proceed(true);
}

void DepthCameraGetImageDataCall::PrepareNextCall()
{
    new DepthCameraGetImageDataCall(service_, cq_, RCMap_);
}

void DepthCameraGetImageDataCall::InitializeRequest()
{
    service_->RequestGetDepthCameraImageData(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void DepthCameraGetImageDataCall::ProcessRequest()
{
    if (!RCMap_.contains(request_.name()))
    {
        UE_LOG(LogTemp, Warning, TEXT("DepthCameraGetImageDataCall: "
                                      "DepthCamera component not found!"));
        return;
    }

    auto* DepthCamera = RCMap_[request_.name()];
    if (!DepthCamera)
    {
        UE_LOG(LogTemp, Warning, TEXT("DepthCameraGetImageDataCall: "
                                      "Invalid DepthCamera reference!"));
        return;
    }

    // Use the callback version to get depth image data
    DepthCamera->AsyncGetDepthImageData(
        [this, DepthCamera](const TArray<FFloat16Color>& DepthImageData)
    {
        // Process the depth data in a background thread
        AsyncTask(ENamedThreads::AnyBackgroundHiPriTask,
            [this, DepthImageData, DepthCamera]()
        {
            response_.mutable_data()->Reserve(DepthImageData.Num());
            for (const auto& DepthValue : DepthImageData)
            {
                response_.add_data(DepthValue.R.GetFloat());
            }

            status_ = FINISH;
            responder_.Finish(response_, grpc::Status::OK, this);
        });
    });
}

DepthCameraGetOdomCall::DepthCameraGetOdomCall(
    VCCSim::DepthCameraService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    std::map<std::string, UDepthCameraComponent*> rdcmap)
        : AsyncCallTemplateM(service, cq, rdcmap)
{
    Proceed(true);
}

void DepthCameraGetOdomCall::PrepareNextCall()
{
    new DepthCameraGetOdomCall(service_, cq_, RCMap_);
}

void DepthCameraGetOdomCall::InitializeRequest()
{
    service_->RequestGetDepthCameraOdom(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void DepthCameraGetOdomCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        const FVector Pose = RCMap_[request_.name()]->GetComponentLocation();
        const FRotator Rot = RCMap_[request_.name()]->GetComponentRotation();
        const FVector LinearVelocity =
            RCMap_[request_.name()]->GetPhysicsLinearVelocity();
        const FVector AngularVelocity =
            RCMap_[request_.name()]->GetPhysicsAngularVelocityInDegrees();

        VCCSim::Pose* PoseData = response_.mutable_pose();
        PoseData->set_x(Pose.X);
        PoseData->set_y(Pose.Y);
        PoseData->set_z(Pose.Z);
        PoseData->set_roll(Rot.Roll);
        PoseData->set_pitch(Rot.Pitch);
        PoseData->set_yaw(Rot.Yaw);

        VCCSim::twist* TwistData = response_.mutable_twist();
        TwistData->set_linear_x(LinearVelocity.X);
        TwistData->set_linear_y(LinearVelocity.Y);
        TwistData->set_linear_z(LinearVelocity.Z);
        TwistData->set_angular_x(AngularVelocity.X);
        TwistData->set_angular_y(AngularVelocity.Y);
        TwistData->set_angular_z(AngularVelocity.Z);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("DepthCameraGetOdomCall: "
                                      "DepthCamera component not found!"));
    }
}

RGBCameraGetOdomCall::RGBCameraGetOdomCall(
    VCCSim::RGBCameraService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    std::map<std::string, URGBCameraComponent*> rrgbcmap)
        : AsyncCallTemplateM(service, cq, rrgbcmap)
{
    Proceed(true);
}

void RGBCameraGetOdomCall::PrepareNextCall()
{
    new RGBCameraGetOdomCall(service_, cq_, RCMap_);
}

void RGBCameraGetOdomCall::InitializeRequest()
{
    service_->RequestGetRGBCameraOdom(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void RGBCameraGetOdomCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        const FVector Pose = RCMap_[request_.name()]->GetComponentLocation();
        const FRotator Rot = RCMap_[request_.name()]->GetComponentRotation();
        const FVector LinearVelocity =
            RCMap_[request_.name()]->GetPhysicsLinearVelocity();
        const FVector AngularVelocity =
            RCMap_[request_.name()]->GetPhysicsAngularVelocityInDegrees();

        VCCSim::Pose* PoseData = response_.mutable_pose();
        PoseData->set_x(Pose.X);
        PoseData->set_y(Pose.Y);
        PoseData->set_z(Pose.Z);
        PoseData->set_roll(Rot.Roll);
        PoseData->set_pitch(Rot.Pitch);
        PoseData->set_yaw(Rot.Yaw);

        VCCSim::twist* TwistData = response_.mutable_twist();
        TwistData->set_linear_x(LinearVelocity.X);
        TwistData->set_linear_y(LinearVelocity.Y);
        TwistData->set_linear_z(LinearVelocity.Z);
        TwistData->set_angular_x(AngularVelocity.X);
        TwistData->set_angular_y(AngularVelocity.Y);
        TwistData->set_angular_z(AngularVelocity.Z);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("RGBCameraGetOdomCall: "
                                    "RGB Camera component not found!"));
    }
}

RGBIndexedCameraImageSizeCall::RGBIndexedCameraImageSizeCall(
    VCCSim::RGBCameraService::AsyncService* service,
    grpc::ServerCompletionQueue* cq, std::map<std::string,
    URGBCameraComponent*> rrgbcmap)
        : AsyncCallTemplateM(service, cq, rrgbcmap)
{
    Proceed(true);
}

void RGBIndexedCameraImageSizeCall::PrepareNextCall()
{
    new RGBIndexedCameraImageSizeCall(service_, cq_, RCMap_);
}

void RGBIndexedCameraImageSizeCall::InitializeRequest()
{
    service_->RequestGetRGBIndexedCameraImageSize(
        &ctx_, &request_, &responder_, cq_, cq_, this);
}

void RGBIndexedCameraImageSizeCall::ProcessRequest()
{
    if (RCMap_.contains(request_.robot_name().name() + "^" +
                         std::to_string(request_.index())))
    {
        const auto& Size = RCMap_[request_.robot_name().name() + "^" +
                                 std::to_string(request_.index())]->GetImageSize();
        response_.set_width(Size.first);
        response_.set_height(Size.second);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("RGBIndexedCameraImageSizeCall: "
                                      "RGB Camera component not found!"));
    }
}

IImageWrapperModule* RGBIndexedCameraImageDataCall::ImageWrapperModule = nullptr;

RGBIndexedCameraImageDataCall::RGBIndexedCameraImageDataCall(
    VCCSim::RGBCameraService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    std::map<std::string, URGBCameraComponent*> rrgbcmap)
        : AsyncCallTemplateImage(service, cq, rrgbcmap)
{
    Proceed(true);
}

void RGBIndexedCameraImageDataCall::PrepareNextCall()
{
    new RGBIndexedCameraImageDataCall(service_, cq_, RCMap_);
}

void RGBIndexedCameraImageDataCall::InitializeRequest()
{
    service_->RequestGetRGBIndexedCameraImageData(
        &ctx_, &request_, &responder_, cq_, cq_, this);
}

void RGBIndexedCameraImageDataCall::ProcessRequest()
{
    std::string CameraName = request_.robot_name().name() + "^" +
                             std::to_string(request_.index());

    if (!RCMap_.contains(CameraName))
    {
        UE_LOG(LogTemp, Warning, TEXT("RGBCameraGetImageDataCall:"
                                      " RGB Camera component not found!"));
        return;
    }

    auto* RGBCamera = RCMap_[CameraName];
    if (!RGBCamera)
    {
        UE_LOG(LogTemp, Warning, TEXT("RGBCameraGetImageDataCall:"
                                      " Invalid RGB Camera reference!"));
        return;
    }

    // Get the requested format from the client
    VCCSim::Format requestedFormat = request_.format();
    
    // Use the callback version to get image data
    RGBCamera->AsyncGetRGBImageData(
        [this, RGBCamera, requestedFormat](const TArray<FColor>& ImageData)
    {
        // Process the image in a background thread
        AsyncTask(ENamedThreads::AnyBackgroundHiPriTask,
            [this, ImageData, RGBCamera, requestedFormat]()
        {
            // Set common properties
            response_.set_width(RGBCamera->Width);
            response_.set_height(RGBCamera->Height);
            response_.set_timestamp(FDateTime::UtcNow().ToUnixTimestamp());
            
            // Handle different formats
            switch (requestedFormat)
            {
                case VCCSim::Format::PNG:
                {
                    TArray<uint8> PNGData = ConvertToCompressedFormat(
                        ImageData, RGBCamera->Width, RGBCamera->Height, EImageFormat::PNG);
                    
                    response_.set_format(VCCSim::Format::PNG);
                    response_.set_data(PNGData.GetData(), PNGData.Num());
                    response_.set_is_compressed(true);
                    break;
                }
                case VCCSim::Format::JPEG:
                {
                    TArray<uint8> JPEGData = ConvertToCompressedFormat(
                        ImageData, RGBCamera->Width, RGBCamera->Height, EImageFormat::JPEG);
                    
                    response_.set_format(VCCSim::Format::JPEG);
                    response_.set_data(JPEGData.GetData(), JPEGData.Num());
                    response_.set_is_compressed(true);
                    break;
                }
                case VCCSim::Format::RAW:
                default:
                {
                    // Use RGB format for raw data (3 bytes per pixel)
                    TArray<uint8> RawRGBData = ConvertToRGB(ImageData);
                    
                    response_.set_format(VCCSim::Format::RAW);
                    response_.set_data(RawRGBData.GetData(), RawRGBData.Num());
                    response_.set_is_compressed(false);
                    response_.set_bytes_per_pixel(3);  // RGB = 3 bytes
                    response_.set_stride(RGBCamera->Width * 3);  // Width * 3 bytes per pixel
                    break;
                }
            }

            status_ = FINISH;
            responder_.Finish(response_, grpc::Status::OK, this);
        });
    });
}

SegmentCameraGetOdomCall::SegmentCameraGetOdomCall(
    VCCSim::SegmentationCameraService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    std::map<std::string, USegmentationCameraComponent*> rscmap)
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
        const FVector Pose = RCMap_[request_.name()]->GetComponentLocation();
        const FRotator Rot = RCMap_[request_.name()]->GetComponentRotation();
        const FVector LinearVelocity =
            RCMap_[request_.name()]->GetPhysicsLinearVelocity();
        const FVector AngularVelocity =
            RCMap_[request_.name()]->GetPhysicsAngularVelocityInDegrees();

        VCCSim::Pose* PoseData = response_.mutable_pose();
        PoseData->set_x(Pose.X);
        PoseData->set_y(Pose.Y);
        PoseData->set_z(Pose.Z);
        PoseData->set_roll(Rot.Roll);
        PoseData->set_pitch(Rot.Pitch);
        PoseData->set_yaw(Rot.Yaw);

        VCCSim::twist* TwistData = response_.mutable_twist();
        TwistData->set_linear_x(LinearVelocity.X);
        TwistData->set_linear_y(LinearVelocity.Y);
        TwistData->set_linear_z(LinearVelocity.Z);
        TwistData->set_angular_x(AngularVelocity.X);
        TwistData->set_angular_y(AngularVelocity.Y);
        TwistData->set_angular_z(AngularVelocity.Z);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("SegmentCameraGetOdomCall: "
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
        FTransform MeshTransform(
            FRotator(
                request_.transform().roll(),
                request_.transform().pitch(),
                request_.transform().yaw()),
            FVector(
                request_.transform().x(),
                request_.transform().y(),
                request_.transform().z())
        );
        
        component_->UpdateMeshFromGRPC(
            reinterpret_cast<const uint8*>(request_.data().data()),
            request_.data().size(),
            MeshTransform
        );
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("SendMeshCall: "
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
        FTransform MeshTransform(
            FRotator(
                request_.transform().roll(),
                request_.transform().pitch(),
                request_.transform().yaw()),
            FVector(
                request_.transform().x(),
                request_.transform().y(),
                request_.transform().z())
        );
        
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
        UE_LOG(LogTemp, Error, TEXT("SendGlobalMeshCall: "
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
        UE_LOG(LogTemp, Error, TEXT("RemoveGlobalMeshCall: "
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
    std::map<std::string, ADronePawn*> rcmap)
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
        const FVector Loc = Drone->GetActorLocation();
        const FRotator Rot = Drone->GetActorRotation();

        response_.set_x(Loc.X);
        response_.set_y(Loc.Y);
        response_.set_z(Loc.Z);
        response_.set_roll(Rot.Roll);
        response_.set_pitch(Rot.Pitch);
        response_.set_yaw(Rot.Yaw);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("GetDroneOdomCall: "
                                      "Drone not found!"));
    }
}

SendDronePoseCall::SendDronePoseCall(
    VCCSim::DroneService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    std::map<std::string, ADronePawn*> rcmap)
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
            const FVector TargetLocation(
                request_.pose().x(),
                request_.pose().y(),
                request_.pose().z()
            );
            const FRotator TargetRotation(
                request_.pose().pitch(),
                request_.pose().yaw(),
                request_.pose().roll()
            );
            if (!Drone->IfCloseToTarget(TargetLocation, TargetRotation))
            {
                Drone->SetTarget(TargetLocation, TargetRotation);
            }
            response_.set_status(true);
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("SendDronePoseCall: "
                                          "AQuadcopterDrone not found!"));
            response_.set_status(false);
        }
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("SendDronePoseCall: "
                                      "Drone not found!"));
        response_.set_status(false);
    }
}

SendDronePathCall::SendDronePathCall(
    VCCSim::DroneService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    std::map<std::string, ADronePawn*> rcmap)
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
            for (const auto& Point : request_.path())
            {
                Positions.Add(FVector(Point.x(), Point.y(), Point.z()));
                Rotations.Add(FRotator(Point.pitch(), Point.yaw(), Point.roll()));
            }
            Drone->SetPath(Positions, Rotations);
            response_.set_status(true);
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("SendDronePathCall: "
                                          "AQuadcopterDrone not found!"));
            response_.set_status(false);
        }
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("SendDronePathCall: "
                                      "Drone not found!"));
        response_.set_status(false);
    }
}

GetCarOdomCall::GetCarOdomCall(
    VCCSim::CarService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    std::map<std::string, ACarPawn*> rcmap)
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
        PoseData->set_x(Loc.X);
        PoseData->set_y(Loc.Y);
        PoseData->set_z(Loc.Z);
        PoseData->set_roll(Rot.Roll);
        PoseData->set_pitch(Rot.Pitch);
        PoseData->set_yaw(Rot.Yaw);

        VCCSim::twist* TwistData = response_.mutable_twist();
        TwistData->set_linear_x(LinearVelocity.X);
        TwistData->set_linear_y(LinearVelocity.Y);
        TwistData->set_linear_z(LinearVelocity.Z);
        TwistData->set_angular_x(AngularVelocity.X);
        TwistData->set_angular_y(AngularVelocity.Y);
        TwistData->set_angular_z(AngularVelocity.Z);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("GetCarOdomCall: "
            "Car not found!"));
    }
}


SendCarPoseCall::SendCarPoseCall(
    VCCSim::CarService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    std::map<std::string, ACarPawn*> rcmap)
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
            const FVector TargetLocation(
                request_.pose().x(),
                request_.pose().y(),
                request_.pose().z()
            );
            const FRotator TargetRotation(
                0.0f,
                request_.pose().yaw(),
                0.0f
            );
            Car->SetTarget(TargetLocation, TargetRotation);
            response_.set_status(true);
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("SendCarPoseCall: "
                "Car not found!"));
            response_.set_status(false);
        }
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("SendCarPoseCall: "
            "Car not found!"));
        response_.set_status(false);
    }
}

SendCarPathCall::SendCarPathCall(
    VCCSim::CarService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    std::map<std::string, ACarPawn*> rcmap)
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
            for (const auto& Point : request_.path())
            {
                Positions.Add(FVector(Point.x(), Point.y(), Point.z()));
                Rotations.Add(FRotator(0.0f, Point.yaw(), 0.0f)); // Only Yaw is used
            }
            Car->SetPath(Positions, Rotations);
            response_.set_status(true);
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("SendCarPathCall: "
                "Car not found!"));
            response_.set_status(false);
        }
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("SendCarPathCall: "
            "Car not found!"));
        response_.set_status(false);
    }
}

GetFlashPoseCall::GetFlashPoseCall(
    VCCSim::FlashService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    std::map<std::string, AFlashPawn*> rcmap)
        : AsyncCallTemplateM(service, cq, rcmap)
{
    Proceed(true);
}

void GetFlashPoseCall::PrepareNextCall()
{
    new GetFlashPoseCall(service_, cq_, RCMap_);
}

void GetFlashPoseCall::InitializeRequest()
{
    service_->RequestGetFlashPose(
        &ctx_, &request_, &responder_, cq_, cq_, this);
}

void GetFlashPoseCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        const auto Flash = RCMap_[request_.name()];
        const FVector Loc = Flash->GetActorLocation();
        const FRotator Rot = Flash->GetActorRotation();

        response_.set_x(Loc.X);
        response_.set_y(Loc.Y);
        response_.set_z(Loc.Z);
        response_.set_roll(Rot.Roll);
        response_.set_pitch(Rot.Pitch);
        response_.set_yaw(Rot.Yaw);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("GetFlashPoseCall: "
                                      "Flash not found!"));
    }
}

SendFlashPoseCall::SendFlashPoseCall(
    VCCSim::FlashService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    std::map<std::string, AFlashPawn*> rcmap)
        : AsyncCallTemplateM(service, cq, rcmap)
{
    Proceed(true);
}

void SendFlashPoseCall::PrepareNextCall()
{
    new SendFlashPoseCall(service_, cq_, RCMap_);
}

void SendFlashPoseCall::InitializeRequest()
{
    service_->RequestSendFlashPose(
        &ctx_, &request_, &responder_, cq_, cq_, this);
}

void SendFlashPoseCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        if (AFlashPawn* Flash = RCMap_[request_.name()])
        {
            const FVector TargetLocation(
                request_.pose().x(),
                request_.pose().y(),
                request_.pose().z()
            );
            const FRotator TargetRotation(
                request_.pose().pitch(),
                request_.pose().yaw(),
                request_.pose().roll()
            );
            Flash->SetTarget(TargetLocation, TargetRotation);
            response_.set_status(true);
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("SendFlashPoseCall: "
                                          "AFlashPawn* is invalid!"));
            response_.set_status(false);
        }
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("SendFlashPoseCall: "
                                      "Flash not found!"));
        response_.set_status(false);
    }
}

SendFlashPathCall::SendFlashPathCall(
    VCCSim::FlashService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    std::map<std::string, AFlashPawn*> rcmap)
        : AsyncCallTemplateM(service, cq, rcmap)
{
    Proceed(true);
}

void SendFlashPathCall::PrepareNextCall()
{
    new SendFlashPathCall(service_, cq_, RCMap_);
}

void SendFlashPathCall::InitializeRequest()
{
    service_->RequestSendFlashPath(
        &ctx_, &request_, &responder_, cq_, cq_, this);
}

void SendFlashPathCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        if (AFlashPawn* Flash = RCMap_[request_.name()])
        {
            TArray<FVector> Positions;
            TArray<FRotator> Rotations;
            for (const auto& Point : request_.path())
            {
                Positions.Add(FVector(Point.x(), Point.y(), Point.z()));
                Rotations.Add(FRotator(Point.pitch(), Point.yaw(), Point.roll()));
            }
            Flash->SetPath(Positions, Rotations);
            response_.set_status(true);
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("SendFlashPathCall: "
                                          "AFlashPawn* is invalid!"));
            response_.set_status(false);
        }
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("SendFlashPathCall: "
                                      "Flash not found!"));
        response_.set_status(false);
    }
}

CheckFlashReadyCall::CheckFlashReadyCall(
    VCCSim::FlashService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    std::map<std::string, AFlashPawn*> rcmap)
        : AsyncCallTemplateM(service, cq, rcmap)
{
    Proceed(true);
}

void CheckFlashReadyCall::PrepareNextCall()
{
    new CheckFlashReadyCall(service_, cq_, RCMap_);
}

void CheckFlashReadyCall::InitializeRequest()
{
    service_->RequestCheckFlashReady(
        &ctx_, &request_, &responder_, cq_, cq_, this);
}

void CheckFlashReadyCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        response_.set_status(RCMap_[request_.name()]->IsReady());
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("CheckFlashReadyCall: "
                                      "Flash not found!"));
        response_.set_status(false);
    }
}

MoveToNextCall::MoveToNextCall(
    VCCSim::FlashService::AsyncService* service,
    grpc::ServerCompletionQueue* cq,
    std::map<std::string, AFlashPawn*> rcmap)
        : AsyncCallTemplateM(service, cq, rcmap)
{
    Proceed(true);
}

void MoveToNextCall::PrepareNextCall()
{
    new MoveToNextCall(service_, cq_, RCMap_);
}

void MoveToNextCall::InitializeRequest()
{
    service_->RequestMoveToNext(
        &ctx_, &request_, &responder_, cq_, cq_, this);
}

void MoveToNextCall::ProcessRequest()
{
    if (RCMap_.contains(request_.name()))
    {
        RCMap_[request_.name()]->MoveToNext();
        response_.set_status(true);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("MoveToNextCall: "
                                      "Flash not found!"));
        response_.set_status(false);
    }
}