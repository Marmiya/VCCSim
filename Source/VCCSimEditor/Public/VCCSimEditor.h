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
#include "Modules/ModuleInterface.h"

DECLARE_LOG_CATEGORY_EXTERN(LogVCCSimEditor, Log, All);

/**
 * VCCSimEditor Module - Editor functionality for VCCSim plugin
 * This module contains all editor-specific functionality including:
 * - VCCSim panel UI
 * - Triangle Splatting editor integration
 * - Editor tools and utilities
 */
class FVCCSimEditorModule : public IModuleInterface
{
public:
    
    /** IModuleInterface implementation */
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;
    
private:
    
    /** Register VCCSim panel tab spawner */
    void RegisterTabSpawner();
    
    /** Unregister VCCSim panel tab spawner */
    void UnregisterTabSpawner();
    
    /** Register menu extensions */
    void RegisterMenuExtensions();
    
    /** Unregister menu extensions */
    void UnregisterMenuExtensions();
};