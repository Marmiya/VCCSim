// Copyright Epic Games, Inc. All Rights Reserved.

using System.IO;
using UnrealBuildTool;

public class VCCSim : ModuleRules
{
	public VCCSim(ReadOnlyTargetRules Target) : base(Target)
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
			
		
		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
                "ImageWrapper",
                "UMG",
                "InputCore",
                "RenderCore", "RHI",
                "ProceduralMeshComponent",
                "LevelSequence",
				"MovieScene", "EnhancedInput", 
				"ChaosVehicles",
				"LevelEditor", "UnrealEd",
				"WorkspaceMenuStructure"
				// "Serialization",
				// ... add other public dependencies that you statically link with here ...
			}
			);
			
		
		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"Slate",
				"SlateCore", "WorkspaceMenuStructure", "WorkspaceMenuStructure",
				// ... add private dependencies that you statically link with here ...	
			}
			);
		
		
		DynamicallyLoadedModuleNames.AddRange(
			new string[]
			{
				// ... add any modules that your module loads dynamically here ...
			}
			);
		
		//set grpc library for Windows
		string GrpcPath = Path.GetFullPath(Path.Combine(ModuleDirectory, "../grpc/"));
		//include the grpc header files
		string GrpcIncludePath = GrpcPath + "include/";
		PublicSystemIncludePaths.AddRange(new string[] { GrpcIncludePath });
		
		//use engine library
		AddEngineThirdPartyPrivateStaticDependencies(Target, "OpenSSL");
		AddEngineThirdPartyPrivateStaticDependencies(Target, "zlib");
		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			string LibPath = GrpcPath + "lib/";
			if (Target.Configuration == UnrealTargetConfiguration.Debug || Target.Configuration == UnrealTargetConfiguration.DebugGame)
			{
				// Link Debug libraries
				LibPath += "Debug/";
				System.Console.WriteLine("Using grpc Debug libraries");
			}
			else
			{
				// Link Release libraries
				LibPath += "Release/";
				System.Console.WriteLine("Using grpc Release libraries");
			}

			foreach (string file in Directory.GetFiles(LibPath, "*.lib"))
			{
				PublicAdditionalLibraries.Add(file);
			}
		}
		else // unsupported platform
		{
			System.Console.WriteLine("Only supported on Windows");
		}
		
		string tomlPath = Path.GetFullPath(Path.Combine(ModuleDirectory, "../tomlplusplus/"));
		string tomlIncludePath = tomlPath + "include/";
		PublicSystemIncludePaths.AddRange(new string[] { tomlIncludePath });
	}
}
