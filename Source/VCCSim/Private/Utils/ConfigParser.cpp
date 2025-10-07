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
#include "Sensors/CameraSensor.h"
#include "Sensors/DepthCamera.h"
#include "Sensors/SegmentCamera.h"
#include "Sensors/NormalCamera.h"
#include "toml++/toml.hpp"
#include "HAL/PlatformFilemanager.h"

DEFINE_LOG_CATEGORY_STATIC(LogConfigParser, Log, All);


namespace ConfigParserHelpers
{
	FString GetConfigFilePath()
	{
		FString ConfigPath = FPaths::Combine(FPaths::ProjectPluginsDir(), TEXT("VCCSim/Source/VCCSim/RSConfig.toml"));

		if (!FPlatformFileManager::Get().GetPlatformFile().FileExists(*ConfigPath))
		{
			UE_LOG(LogConfigParser, Warning, TEXT("ParseConfig: Using default config!"));
			ConfigPath = FPaths::Combine(FPlatformProcess::UserDir(), TEXT("VCCSim/RSConfig.toml"));
		}

		return ConfigPath;
	}

	void ParseStringArray(const toml::array* Array, TArray<FString>& OutArray)
	{
		if (!Array) return;

		for (const auto& Element : *Array)
		{
			if (auto Value = Element.value<std::string>())
			{
				OutArray.Add(FString(UTF8_TO_TCHAR(Value->c_str())));
			}
		}
	}

	void ParseFloatArray(const toml::array* Array, TArray<float>& OutArray)
	{
		if (!Array) return;

		for (const auto& Element : *Array)
		{
			if (auto Value = Element.value<double>())
			{
				OutArray.Add(static_cast<float>(*Value));
			}
		}
	}

	FString TomlStringToFString(const std::string& Str)
	{
		return FString(UTF8_TO_TCHAR(Str.c_str()));
	}

	EPawnType ParsePawnType(const std::string& TypeStr)
	{
		if (TypeStr == "Drone") return EPawnType::Drone;
		if (TypeStr == "Car") return EPawnType::Car;
		if (TypeStr == "Flash") return EPawnType::Flash;

		UE_LOG(LogConfigParser, Warning, TEXT("Unknown pawn type: %s"), *TomlStringToFString(TypeStr));
		return EPawnType::Drone;
	}

	TSharedPtr<FSensorConfig> CreateSensorConfig(const std::string& SensorType, const toml::table* ConfigTable)
	{
		if (SensorType == "Lidar")
		{
			auto Config = MakeShared<FLiDARConfig>();
			if (ConfigTable)
			{
				Config->RecordInterval = (*ConfigTable)["RecordInterval"].value_or(Config->RecordInterval);
				Config->NumRays = (*ConfigTable)["NumRays"].value_or(Config->NumRays);
				Config->NumPoints = (*ConfigTable)["NumPoints"].value_or(Config->NumPoints);
				Config->ScannerRangeInner = (*ConfigTable)["ScannerRangeInner"].value_or(Config->ScannerRangeInner);
				Config->ScannerRangeOuter = (*ConfigTable)["ScannerRangeOuter"].value_or(Config->ScannerRangeOuter);
				Config->ScannerAngleUp = (*ConfigTable)["ScannerAngle"].value_or(Config->ScannerAngleUp);
				Config->ScannerAngleDown = (*ConfigTable)["ScannerAngleDown"].value_or(Config->ScannerAngleDown);
				Config->bVisualizePoints = (*ConfigTable)["bVisualizePoints"].value_or(Config->bVisualizePoints);
			}
			return Config;
		}

		if (SensorType == "RGBCamera")
		{
			auto Config = MakeShared<FRGBCameraConfig>();
			if (ConfigTable)
			{
				Config->RecordInterval = (*ConfigTable)["RecordInterval"].value_or(Config->RecordInterval);
				Config->FOV = (*ConfigTable)["FOV"].value_or(Config->FOV);
				Config->Width = (*ConfigTable)["Width"].value_or(Config->Width);
				Config->Height = (*ConfigTable)["Height"].value_or(Config->Height);
			}
			return Config;
		}

		if (SensorType == "DepthCamera")
		{
			auto Config = MakeShared<FDepthCameraConfig>();
			if (ConfigTable)
			{
				Config->RecordInterval = (*ConfigTable)["RecordInterval"].value_or(Config->RecordInterval);
				Config->FOV = (*ConfigTable)["FOV"].value_or(Config->FOV);
				Config->Width = (*ConfigTable)["Width"].value_or(Config->Width);
				Config->Height = (*ConfigTable)["Height"].value_or(Config->Height);
				Config->MaxRange = (*ConfigTable)["MaxRange"].value_or(Config->MaxRange);
				Config->MinRange = (*ConfigTable)["MinRange"].value_or(Config->MinRange);
			}
			return Config;
		}

		if (SensorType == "SegmentationCamera")
		{
			auto Config = MakeShared<FSegmentationCameraConfig>();
			if (ConfigTable)
			{
				Config->RecordInterval = (*ConfigTable)["RecordInterval"].value_or(Config->RecordInterval);
				Config->FOV = (*ConfigTable)["FOV"].value_or(Config->FOV);
				Config->Width = (*ConfigTable)["Width"].value_or(Config->Width);
				Config->Height = (*ConfigTable)["Height"].value_or(Config->Height);
			}
			return Config;
		}

		if (SensorType == "NormalCamera")
		{
			auto Config = MakeShared<FNormalCameraConfig>();
			if (ConfigTable)
			{
				Config->RecordInterval = (*ConfigTable)["RecordInterval"].value_or(Config->RecordInterval);
				Config->FOV = (*ConfigTable)["FOV"].value_or(Config->FOV);
				Config->Width = (*ConfigTable)["Width"].value_or(Config->Width);
				Config->Height = (*ConfigTable)["Height"].value_or(Config->Height);
			}
			return Config;
		}

		UE_LOG(LogConfigParser, Warning, TEXT("Unknown sensor type: %s"), *TomlStringToFString(SensorType));
		return nullptr;
	}

	ESensorType StringToSensorType(const std::string& SensorType)
	{
		if (SensorType == "Lidar") return ESensorType::Lidar;
		if (SensorType == "RGBCamera") return ESensorType::RGBCamera;
		if (SensorType == "DepthCamera") return ESensorType::DepthCamera;
		if (SensorType == "SegmentationCamera") return ESensorType::SegmentationCamera;
		if (SensorType == "NormalCamera") return ESensorType::NormalCamera;
		return ESensorType::RGBCamera;
	}
}

FVCCSimConfig ParseConfig()
{
	using namespace ConfigParserHelpers;

	FVCCSimConfig Config;
	const FString ConfigPath = GetConfigFilePath();
	const std::string ConfigPathStd = TCHAR_TO_UTF8(*ConfigPath);

	toml::table ParsedToml;
	try
	{
		ParsedToml = toml::parse_file(ConfigPathStd);
	}
	catch (const toml::parse_error& Error)
	{
		UE_LOG(LogConfigParser, Error, TEXT("Failed to parse config file: %s"),
			*TomlStringToFString(std::string(Error.description())));
		return Config;
	}

	if (auto VCCSimTable = ParsedToml["VCCSimPresets"].as_table())
	{
		const auto& Presets = *VCCSimTable;

		const std::string IP = Presets["IP"].value_or("0.0.0.0");
		const int32 Port = Presets["Port"].value_or(50996);
		Config.VCCSim.Server = FString::Printf(TEXT("%s:%d"), *TomlStringToFString(IP), Port);

		Config.VCCSim.MainCharacter = TomlStringToFString(Presets["MainCharacter"].value_or(""));
		Config.VCCSim.ManualControl = Presets["ManualControl"].value_or(true);
		Config.VCCSim.LS_StartOffset = Presets["LS_StartOffset"].value_or(0);
		Config.VCCSim.DefaultDronePawn = TomlStringToFString(Presets["DefaultDronePawn"].value_or(""));
		Config.VCCSim.DefaultCarPawn = TomlStringToFString(Presets["DefaultCarPawn"].value_or(""));
		Config.VCCSim.DefaultFlashPawn = TomlStringToFString(Presets["DefaultFlashPawn"].value_or(""));
		Config.VCCSim.BufferSize = Presets["BufferSize"].value_or(100);
		Config.VCCSim.StartWithRecording = Presets["StartWithRecording"].value_or(false);
		Config.VCCSim.BetterVisualsRecording = Presets["BetterVisualsRecording"].value_or(false);
		Config.VCCSim.UseMeshManager = Presets["UseMeshManager"].value_or(false);
		Config.VCCSim.MeshMaterial = TomlStringToFString(Presets["MeshMaterial"].value_or("None"));

		ParseStringArray(Presets["StaticMeshActor"].as_array(), Config.VCCSim.StaticMeshActor);
		ParseStringArray(Presets["SubWindows"].as_array(), Config.VCCSim.SubWindows);
		ParseFloatArray(Presets["SubWindowsOpacities"].as_array(), Config.VCCSim.SubWindowsOpacities);
	}

	if (auto RobotsArray = ParsedToml["Robots"].as_array())
	{
		for (const auto& RobotElement : *RobotsArray)
		{
			auto RobotTable = RobotElement.as_table();
			if (!RobotTable) continue;

			const auto& Robot = *RobotTable;
			FRobot RobotConfig;

			const std::string TagStr = Robot["UETag"].value_or("None");
			if (TagStr == "None")
			{
				UE_LOG(LogConfigParser, Error, TEXT("Robot UETag not found, skipping robot"));
				continue;
			}

			RobotConfig.UETag = TomlStringToFString(TagStr);
			RobotConfig.Type = ParsePawnType(Robot["Type"].value_or("Drone"));
			RobotConfig.RecordInterval = Robot["RecordInterval"].value_or(-1.0f);

			if (auto ComponentConfigsTable = Robot["ComponentConfigs"].as_table())
			{
				for (const auto& [SensorName, SensorConfigValue] : *ComponentConfigsTable)
				{
					const std::string SensorNameStr(SensorName);
					auto SensorConfigTable = SensorConfigValue.as_table();

					if (auto SensorConfig = CreateSensorConfig(SensorNameStr, SensorConfigTable))
					{
						const ESensorType SensorType = StringToSensorType(SensorNameStr);
						RobotConfig.ComponentConfigs.Add(TPair<ESensorType,
							TSharedPtr<FSensorConfig>>(SensorType, SensorConfig));
					}
				}
			}

			Config.Robots.Add(MoveTemp(RobotConfig));
		}
	}

	return Config;
}