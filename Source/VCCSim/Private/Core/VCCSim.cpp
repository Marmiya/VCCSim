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

DEFINE_LOG_CATEGORY_STATIC(LogVCCSim, Log, All);

// Editor functionality moved to VCCSimEditor module

#define LOCTEXT_NAMESPACE "FVCCSimModule"

void FVCCSimModule::StartupModule()
{
    // Runtime module initialization
    UE_LOG(LogVCCSim, Display, TEXT("VCCSim module starting up."));
}

void FVCCSimModule::ShutdownModule()
{
    // Runtime module shutdown
    UE_LOG(LogVCCSim, Display, TEXT("VCCSim module shutting down."));
}

#undef LOCTEXT_NAMESPACE
    
IMPLEMENT_MODULE(FVCCSimModule, VCCSim)