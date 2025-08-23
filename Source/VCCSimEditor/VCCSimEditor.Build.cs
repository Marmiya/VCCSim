// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class VCCSimEditor : ModuleRules
{
    public VCCSimEditor(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;
        
        PublicIncludePaths.AddRange(
            new string[] {
                // ... add public include paths required here ...
            }
        );
        
        PrivateIncludePaths.AddRange(
            new string[] {
                // ... add other private include paths required here ...
            }
        );
        
        // Core dependencies
        PublicDependencyModuleNames.AddRange(
            new string[]
            {
                "Core",
                "CoreUObject",
                "Engine",
                "VCCSim",  // Runtime module dependency
                
                // Editor-only modules
                "LevelEditor",
                "UnrealEd",
                "WorkspaceMenuStructure",
                "PropertyEditor",
                "EditorStyle",
                "EditorWidgets",
                "ToolMenus",
                "EditorSubsystem",
                
                // UI modules
                "Slate",
                "SlateCore",
                "UMG",
                "InputCore",
                
                // Asset management
                "AssetTools",
                "AssetRegistry",
                "ContentBrowser",
                
                // Rendering
                "RenderCore",
                "RHI"
            }
        );
        
        PrivateDependencyModuleNames.AddRange(
            new string[]
            {
                // Additional private dependencies can be added here
            }
        );
        
        DynamicallyLoadedModuleNames.AddRange(
            new string[]
            {
                // ... add any modules that your module loads dynamically here ...
            }
        );
    }
}