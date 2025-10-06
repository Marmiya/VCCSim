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

#include "Core/VCCSim.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "ShaderCore.h"

DEFINE_LOG_CATEGORY_STATIC(LogVCCSim, Log, All);

#define LOCTEXT_NAMESPACE "FVCCSimModule"

void FVCCSimModule::StartupModule()
{
    UE_LOG(LogVCCSim, Display, TEXT("VCCSim module starting up."));

    FString PluginShaderDir = FPaths::Combine(IPluginManager::Get().FindPlugin(TEXT("VCCSim"))->GetBaseDir(), TEXT("Source/VCCSim/Shaders"));
    AddShaderSourceDirectoryMapping(TEXT("/VCCSim"), PluginShaderDir);

    UE_LOG(LogVCCSim, Log, TEXT("Registered shader directory mapping: /VCCSim -> %s"), *PluginShaderDir);
}

void FVCCSimModule::ShutdownModule()
{
    // Runtime module shutdown
    UE_LOG(LogVCCSim, Display, TEXT("VCCSim module shutting down."));
}

#undef LOCTEXT_NAMESPACE
    
IMPLEMENT_MODULE(FVCCSimModule, VCCSim)