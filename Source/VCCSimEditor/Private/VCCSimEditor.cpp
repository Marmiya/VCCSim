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

#include "VCCSimEditor.h"
#include "Editor/VCCSimPanel.h"
#include "Modules/ModuleManager.h"
#include "WorkspaceMenuStructure.h"
#include "WorkspaceMenuStructureModule.h"
#include "Framework/Docking/TabManager.h"
#include "LevelEditor.h"

DEFINE_LOG_CATEGORY(LogVCCSimEditor);

#define LOCTEXT_NAMESPACE "FVCCSimEditorModule"

void FVCCSimEditorModule::StartupModule()
{
    UE_LOG(LogVCCSimEditor, Log, TEXT("VCCSimEditor module starting up"));
    
    // Register tab spawner
    RegisterTabSpawner();
    
    // Register menu extensions
    RegisterMenuExtensions();
}

void FVCCSimEditorModule::ShutdownModule()
{
    UE_LOG(LogVCCSimEditor, Log, TEXT("VCCSimEditor module shutting down"));
    
    // Unregister menu extensions
    UnregisterMenuExtensions();
    
    // Unregister tab spawner
    UnregisterTabSpawner();
}

void FVCCSimEditorModule::RegisterTabSpawner()
{
    FLevelEditorModule& LevelEditorModule = FModuleManager::LoadModuleChecked<FLevelEditorModule>("LevelEditor");
    TSharedPtr<FTabManager> LevelEditorTabManager = LevelEditorModule.GetLevelEditorTabManager();
    
    if (LevelEditorTabManager.IsValid())
    {
        FVCCSimPanelFactory::RegisterTabSpawner(*LevelEditorTabManager);
    }
}

void FVCCSimEditorModule::UnregisterTabSpawner()
{
    FLevelEditorModule& LevelEditorModule = FModuleManager::LoadModuleChecked<FLevelEditorModule>("LevelEditor");
    TSharedPtr<FTabManager> LevelEditorTabManager = LevelEditorModule.GetLevelEditorTabManager();
    
    if (LevelEditorTabManager.IsValid())
    {
        LevelEditorTabManager->UnregisterTabSpawner(FVCCSimPanelFactory::TabId);
    }
}

void FVCCSimEditorModule::RegisterMenuExtensions()
{
    // Menu extensions can be added here if needed in the future
}

void FVCCSimEditorModule::UnregisterMenuExtensions()
{
    // Menu extension cleanup can be added here if needed in the future
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FVCCSimEditorModule, VCCSimEditor)