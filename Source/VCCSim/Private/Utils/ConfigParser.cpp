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

#include "Utils/ConfigParser.h"
#include "Sensors/LidarSensor.h"
#include "Sensors/DepthCamera.h"
#include "Sensors/CameraSensor.h"
#include "Sensors/SegmentCamera.h"
#include "Sensors/NormalCamera.h"
#include "toml++/toml.hpp"
#include <filesystem>
#include "CoreMinimal.h"

using namespace std::literals;


FVCCSimConfig ParseConfig()
{
    FVCCSimConfig Config;
    
    std::string Filename = TCHAR_TO_UTF8(*FPaths::Combine(FPaths::ProjectPluginsDir(),
        TEXT("VCCSim/Source/VCCSim/RSConfig.toml")));;
    
    if (!std::filesystem::exists(Filename))
    {
        UE_LOG(LogTemp, Warning, TEXT("ParseConfig: Using default config!"));
        // Get the user's documents directory
        FString DocPath = FPlatformProcess::UserDir();
        Filename = TCHAR_TO_UTF8(*FPaths::Combine(
            DocPath, TEXT("VCCSim"), TEXT("RSConfig.toml")));
    }

    auto Tbl = toml::parse_file(Filename);

    if (auto VCCSim = Tbl["VCCSimPresets"].as_table())
    {
        Config.VCCSim.Server =
            std::string((*VCCSim)["IP"].value_or("0.0.0.0"sv)) + ":" +
            std::to_string((*VCCSim)["Port"].value_or(50996));
        Config.VCCSim.MainCharacter = (*VCCSim)["MainCharacter"].value_or("");
        Config.VCCSim.ManualControl = (*VCCSim)["ManualControl"].value_or(true);
        Config.VCCSim.LS_StartOffset = (*VCCSim)["LS_StartOffset"].value_or(0);
        Config.VCCSim.LogSavePath = (*VCCSim)["LogSavePath"].value_or(
            TCHAR_TO_UTF8(FPlatformProcess::UserDir()));
        Config.VCCSim.DefaultDronePawn = (*VCCSim)["DefaultDronePawn"].value_or("");
        Config.VCCSim.DefaultCarPawn = (*VCCSim)["DefaultCarPawn"].value_or("");
        Config.VCCSim.DefaultFlashPawn = (*VCCSim)["DefaultFlashPawn"].value_or("");
        Config.VCCSim.BufferSize = (*VCCSim)["BufferSize"].value_or(100);
        Config.VCCSim.StartWithRecording = (*VCCSim)["StartWithRecording"].value_or(false);
        Config.VCCSim.UseMeshManager = (*VCCSim)["UseMeshManager"].value_or(false);
        Config.VCCSim.MeshMaterial = (*VCCSim)["MeshMaterial"].value_or("None"sv);
        
        if (auto staticMeshActors = (*VCCSim)["StaticMeshActor"].as_array())
        {
            for (const auto& actor : *staticMeshActors)
            {
                if (auto actorName = actor.value<std::string>())
                {
                    Config.VCCSim.StaticMeshActor.push_back(*actorName);
                }
            }
        }
        if (auto subWindows = (*VCCSim)["SubWindows"].as_array())
        {
            for (const auto& window : *subWindows)
            {
                if (auto windowName = window.value<std::string>())
                {
                    Config.VCCSim.SubWindows.push_back(*windowName);
                }
            }
        }
        if (auto subWindowsOpacities
            = (*VCCSim)["SubWindowsOpacities"].as_array())
        {
            for (const auto& opacity : *subWindowsOpacities)
            {
                if (auto op = opacity.value<double>())
                {
                    Config.VCCSim.SubWindowsOpacities.push_back(*op);
                }
            }
        }
    }

    if (auto Robots = Tbl["Robots"].as_array())
    {
        for (const auto& Robot : *Robots)
        {
            FRobot r;
            auto RobotDetails = *Robot.as_table();

            r.UETag = RobotDetails["UETag"].value_or("None"sv);
            if (r.UETag == "None")
            {
                UE_LOG(LogTemp, Error, TEXT("ParseConfig: Robot name not found!"));
                continue;
            }

            if (RobotDetails["Type"].value_or("Drone"sv) == "Drone"sv)
            {
                r.Type = EPawnType::Drone;
            }
            else if (RobotDetails["Type"].value_or("Drone"sv) == "Car"sv)
            {
                r.Type = EPawnType::Car;
            }
            else if (RobotDetails["Type"].value_or("Drone"sv) == "Flash"sv)
            {
                r.Type = EPawnType::Flash;
            }
            else
            {
                UE_LOG(LogTemp, Warning, TEXT("ParseConfig: Robot type not found!"));
                continue;
            }

            r.RecordInterval = RobotDetails["RecordInterval"].value_or(-1.0);

            if (auto ComponentConfigs = RobotDetails["ComponentConfigs"].as_table())
            {
                for (const auto& [comp_name, comp_config] : *ComponentConfigs)
                {
                    if (comp_name == "Lidar")
                    {
                        auto LiDarConfig = std::make_shared<FLiDarConfig>();
                        if (auto Table = comp_config.as_table())
                        {
                            LiDarConfig->RecordInterval =
                                (*Table)["RecordInterval"].value_or(LiDarConfig->RecordInterval);
                            LiDarConfig->NumRays =
                                (*Table)["NumRays"].value_or(LiDarConfig->NumRays);
                            LiDarConfig->NumPoints =
                                (*Table)["NumPoints"].value_or(LiDarConfig->NumPoints);
                            LiDarConfig->ScannerRangeInner =
                                (*Table)["ScannerRangeInner"].value_or(LiDarConfig->ScannerRangeInner);
                            LiDarConfig->ScannerRangeOuter =
                                (*Table)["ScannerRangeOuter"].value_or(LiDarConfig->ScannerRangeOuter);
                            LiDarConfig->ScannerAngleUp =
                                (*Table)["ScannerAngle"].value_or(LiDarConfig->ScannerAngleUp);
                            LiDarConfig->ScannerAngleDown =
                                (*Table)["ScannerAngleDown"].value_or(LiDarConfig->ScannerAngleDown);
                            LiDarConfig->bVisualizePoints =
                                (*Table)["bVisualizePoints"].value_or(LiDarConfig->bVisualizePoints);
                        }
                        r.ComponentConfigs.push_back({ESensorType::Lidar, LiDarConfig});
                    }
                    else if (comp_name == "DepthCamera")
                    {
                        auto DepthConfig = std::make_shared<FDepthCameraConfig>();
                        if (auto Table = comp_config.as_table())
                        {
                            DepthConfig->RecordInterval =
                                (*Table)["RecordInterval"].value_or(DepthConfig->RecordInterval);
                            DepthConfig->FOV = (*Table)["FOV"].value_or(DepthConfig->FOV);
                            DepthConfig->MaxRange = (*Table)["MaxRange"].value_or(DepthConfig->MaxRange);
                            DepthConfig->MinRange = (*Table)["MinRange"].value_or(DepthConfig->MinRange);
                            DepthConfig->Width = (*Table)["Width"].value_or(DepthConfig->Width);
                            DepthConfig->Height = (*Table)["Height"].value_or(DepthConfig->Height);
                            DepthConfig->bOrthographic =
                                (*Table)["bOrthographic"].value_or(DepthConfig->bOrthographic);
                            DepthConfig->OrthoWidth =
                                (*Table)["OrthoWidth"].value_or(DepthConfig->OrthoWidth);
                        }
                        r.ComponentConfigs.push_back({ESensorType::DepthCamera, DepthConfig});
                    }
                    else if (comp_name == "RGBCamera")
                    {
                        auto RGBConfig = std::make_shared<FRGBCameraConfig>();
                        if (auto Table = comp_config.as_table())
                        {
                            RGBConfig->RecordInterval =
                                (*Table)["RecordInterval"].value_or(RGBConfig->RecordInterval);
                            RGBConfig->FOV = (*Table)["FOV"].value_or(RGBConfig->FOV);
                            RGBConfig->Width = (*Table)["Width"].value_or(RGBConfig->Width);
                            RGBConfig->Height = (*Table)["Height"].value_or(RGBConfig->Height);
                            RGBConfig->bOrthographic =
                                (*Table)["bOrthographic"].value_or(RGBConfig->bOrthographic);
                            RGBConfig->OrthoWidth =
                                (*Table)["OrthoWidth"].value_or(RGBConfig->OrthoWidth);
                        }
                        r.ComponentConfigs.push_back({ESensorType::RGBCamera, RGBConfig});
                    }
                    else if (comp_name == "SegmentationCamera")
                    {
                        auto SegmentationConfig = std::make_shared<FSegmentationCameraConfig>();
                        if (auto Table = comp_config.as_table())
                        {
                            SegmentationConfig->RecordInterval =
                                (*Table)["RecordInterval"].value_or(SegmentationConfig->RecordInterval);
                            SegmentationConfig->FOV =
                                (*Table)["FOV"].value_or(SegmentationConfig->FOV);
                            SegmentationConfig->Width =
                                (*Table)["Width"].value_or(SegmentationConfig->Width);
                            SegmentationConfig->Height =
                                (*Table)["Height"].value_or(SegmentationConfig->Height);
                        }
                        r.ComponentConfigs.push_back({ESensorType::SegmentationCamera, SegmentationConfig});
                    }
                    else if (comp_name == "NormalCamera")
                    {
                        auto NormalCameraConfig = std::make_shared<FNormalCameraConfig>();
                        if (auto Table = comp_config.as_table())
                        {
                            NormalCameraConfig->RecordInterval =
                                (*Table)["RecordInterval"].value_or(NormalCameraConfig->RecordInterval);
                            NormalCameraConfig->FOV =
                                (*Table)["FOV"].value_or(NormalCameraConfig->FOV);
                            NormalCameraConfig->Width =
                                (*Table)["Width"].value_or(NormalCameraConfig->Width);
                            NormalCameraConfig->Height =
                                (*Table)["Height"].value_or(NormalCameraConfig->Height);
                        }
                        r.ComponentConfigs.push_back({ESensorType::NormalCamera, NormalCameraConfig});
                    }
                    else
                    {
                        UE_LOG(LogTemp, Warning, TEXT("ParseConfig: Component %s not found!"),
                            *FString{UTF8_TO_TCHAR(std::string(comp_name).c_str())});
                    }
                }
            }

            Config.Robots.push_back(std::move(r));
        }
    }

    return Config;
}