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
#include <string>
#include <vector>
#include <set>
#include <memory>
#include "Sensors/SensorBase.h"
#include "Pawns/PawnBase.h"

using FComponentConfig = std::pair<ESensorType, std::shared_ptr<FSensorConfig>>;

struct FRobot
{
	std::string UETag;
	EPawnType Type;
	std::vector<FComponentConfig> ComponentConfigs;
	float RecordInterval;
};

struct FVCCSimPresets
{
	std::string Server;
	std::string MainCharacter;
	std::vector<std::string> StaticMeshActor;
	bool ManualControl;
	std::vector<std::string> SubWindows{};
	std::vector<float> SubWindowsOpacities;
	int LS_StartOffset;
	bool StartWithRecording;
	bool BetterVisualsRecording;
	bool UseMeshManager;
	std::string MeshMaterial;
	std::string LogSavePath;
	std::string DefaultDronePawn;
	std::string DefaultCarPawn;
	std::string DefaultFlashPawn;
	int BufferSize;
};

struct FVCCSimConfig
{
	FVCCSimPresets VCCSim;
	std::vector<FRobot> Robots;
};

FVCCSimConfig ParseConfig();