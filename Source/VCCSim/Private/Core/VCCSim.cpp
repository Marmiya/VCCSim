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

// Only include panel header in editor builds
#if WITH_EDITOR
#include "Core/VCCSimPanel.h"
#include "LevelEditor.h"
#endif

#define LOCTEXT_NAMESPACE "FVCCSimModule"

void FVCCSimModule::StartupModule()
{
    // This code will execute after your module is loaded into memory
    
#if WITH_EDITOR
    // Editor-only initialization
    FLevelEditorModule& LevelEditorModule =
       FModuleManager::LoadModuleChecked<FLevelEditorModule>("LevelEditor");
       
    // Register the tab spawner
    TSharedPtr<FTabManager> TabManager = LevelEditorModule.GetLevelEditorTabManager();
    if (TabManager.IsValid())
    {
       FVCCSimPanelFactory::RegisterTabSpawner(*TabManager);
    }
    
    // Register for tab manager changes
    LevelEditorTabManagerChangedHandle =
       LevelEditorModule.OnTabManagerChanged().AddLambda([this, &LevelEditorModule]()
    {
       // Get the tab manager directly when the lambda is called
       TSharedPtr<FTabManager> TabManager = LevelEditorModule.GetLevelEditorTabManager();
       if (TabManager.IsValid())
       {
         FVCCSimPanelFactory::RegisterTabSpawner(*TabManager);
       }
    });
#endif // WITH_EDITOR
    
    UE_LOG(LogTemp, Warning, TEXT("VCCSim module has started!"));
}

void FVCCSimModule::ShutdownModule()
{
#if WITH_EDITOR
    if (FModuleManager::Get().IsModuleLoaded("LevelEditor"))
    {
       FLevelEditorModule& LevelEditorModule =
          FModuleManager::GetModuleChecked<FLevelEditorModule>("LevelEditor");
        
       // Unregister delegates
       LevelEditorModule.OnTabManagerChanged().Remove(LevelEditorTabManagerChangedHandle);
        
       // Unregister the tab spawner
       TSharedPtr<FTabManager> TabManager = LevelEditorModule.GetLevelEditorTabManager();
       if (TabManager.IsValid())
       {
          TabManager->UnregisterTabSpawner(FVCCSimPanelFactory::TabId);
       }
    }
#endif // WITH_EDITOR
    
    UE_LOG(LogTemp, Warning, TEXT("VCCSim module has shut down!"));
}

#undef LOCTEXT_NAMESPACE
    
IMPLEMENT_MODULE(FVCCSimModule, VCCSim)