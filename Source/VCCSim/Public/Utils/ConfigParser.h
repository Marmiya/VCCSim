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
#include "Sensors/SensorBase.h"
#include "Pawns/PawnBase.h"

using FComponentConfig = TPair<ESensorType, TSharedPtr<FSensorConfig>>;

struct FRobot
{
	FString UETag;
	EPawnType Type;
	TArray<FComponentConfig> ComponentConfigs;
	float RecordInterval;
};

struct FVCCSimPresets
{
	FString Server;
	FString MainCharacter;
	TArray<FString> StaticMeshActor;
	bool ManualControl;
	TArray<FString> SubWindows;
	TArray<float> SubWindowsOpacities;
	int32 LS_StartOffset;
	bool StartWithRecording;
	bool BetterVisualsRecording;
	bool UseMeshManager;
	FString MeshMaterial;
	FString LogSavePath;
	FString DefaultDronePawn;
	FString DefaultCarPawn;
	FString DefaultFlashPawn;
	int32 BufferSize;
};

struct FVCCSimConfig
{
	FVCCSimPresets VCCSim;
	TArray<FRobot> Robots;
};

FVCCSimConfig ParseConfig();