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
#include "ToolMenus.h"
#include "Framework/Docking/TabManager.h"
#include "LevelEditor.h"

DEFINE_LOG_CATEGORY(LogVCCSimEditor);

#define LOCTEXT_NAMESPACE "FVCCSimEditorModule"

void FVCCSimEditorModule::StartupModule()
{
    UE_LOG(LogVCCSimEditor, Display, TEXT("VCCSimEditor module starting up."));
    
    // Register tab spawner
    RegisterTabSpawner();
    
    // Register menu extensions
    RegisterMenuExtensions();
}

void FVCCSimEditorModule::ShutdownModule()
{
    UE_LOG(LogVCCSimEditor, Display, TEXT("VCCSimEditor module shutting down."));
    
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
        // TabManager is available, register immediately
        FVCCSimPanelFactory::RegisterTabSpawner(*LevelEditorTabManager);
    }
    else
    {
        // TabManager not ready yet, defer registration until it becomes available
        TabManagerChangedHandle = LevelEditorModule.OnTabManagerChanged().AddLambda([this]()
        {
            FLevelEditorModule& LevelEditorModule = FModuleManager::LoadModuleChecked<FLevelEditorModule>("LevelEditor");
            TSharedPtr<FTabManager> TabManager = LevelEditorModule.GetLevelEditorTabManager();
            if (TabManager.IsValid())
            {
                FVCCSimPanelFactory::RegisterTabSpawner(*TabManager);
                
                // Remove the delegate since we no longer need it
                LevelEditorModule.OnTabManagerChanged().Remove(TabManagerChangedHandle);
                TabManagerChangedHandle.Reset();
            }
        });
    }
}

void FVCCSimEditorModule::UnregisterTabSpawner()
{
    // Clean up any pending deferred registration delegate
    if (TabManagerChangedHandle.IsValid())
    {
        if (FModuleManager::Get().IsModuleLoaded("LevelEditor"))
        {
            FLevelEditorModule& LevelEditorModule = FModuleManager::GetModuleChecked<FLevelEditorModule>("LevelEditor");
            LevelEditorModule.OnTabManagerChanged().Remove(TabManagerChangedHandle);
        }
        TabManagerChangedHandle.Reset();
    }
    
    // Unregister the actual tab spawner
    if (FModuleManager::Get().IsModuleLoaded("LevelEditor"))
    {
        FLevelEditorModule& LevelEditorModule = FModuleManager::GetModuleChecked<FLevelEditorModule>("LevelEditor");
        TSharedPtr<FTabManager> LevelEditorTabManager = LevelEditorModule.GetLevelEditorTabManager();
        
        if (LevelEditorTabManager.IsValid())
        {
            LevelEditorTabManager->UnregisterTabSpawner(FVCCSimPanelFactory::TabId);
        }
    }
}

void FVCCSimEditorModule::RegisterMenuExtensions()
{
    // Register VCCSim panel in the Level Editor Window menu
    UToolMenus* ToolMenus = UToolMenus::Get();
    if (ToolMenus)
    {
        UToolMenu* WindowMenu = ToolMenus->ExtendMenu("LevelEditor.MainMenu.Window");
        if (WindowMenu)
        {
            FToolMenuSection& LevelEditorSection = WindowMenu->FindOrAddSection("LevelEditor");
            LevelEditorSection.AddMenuEntry(
                "VCCSimPanel",
                NSLOCTEXT("VCCSimEditor", "VCCSimPanelMenuItem", "VCCSim"),
                NSLOCTEXT("VCCSimEditor", "VCCSimPanelMenuTooltip", "Opens the VCCSim panel"),
                FSlateIcon(FAppStyle::GetAppStyleSetName(), "LevelEditor.Tabs.Viewports"),
                FUIAction(FExecuteAction::CreateLambda([]()
                {
                    FLevelEditorModule& LevelEditorModule = FModuleManager::GetModuleChecked<FLevelEditorModule>("LevelEditor");
                    TSharedPtr<FTabManager> TabManager = LevelEditorModule.GetLevelEditorTabManager();
                    if (TabManager.IsValid())
                    {
                        TabManager->TryInvokeTab(FVCCSimPanelFactory::TabId);
                    }
                }))
            );
        }
    }
}

void FVCCSimEditorModule::UnregisterMenuExtensions()
{
    // Clean up menu extensions
    UToolMenus* ToolMenus = UToolMenus::Get();
    if (ToolMenus)
    {
        ToolMenus->RemoveMenu("VCCSimPanel");
    }
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FVCCSimEditorModule, VCCSimEditor)