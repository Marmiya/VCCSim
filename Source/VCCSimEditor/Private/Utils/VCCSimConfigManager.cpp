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

#include "Utils/VCCSimConfigManager.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "HAL/PlatformFilemanager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

DEFINE_LOG_CATEGORY_STATIC(LogVCCSimConfigManager, Log, All);

TUniquePtr<FVCCSimConfigManager> FVCCSimConfigManager::Instance = nullptr;

// ============================================================================
// SINGLETON MANAGEMENT
// ============================================================================

FVCCSimConfigManager& FVCCSimConfigManager::Get()
{
    if (!Instance.IsValid())
    {
        Initialize();
    }
    return *Instance;
}

void FVCCSimConfigManager::Initialize()
{
    if (!Instance.IsValid())
    {
        Instance = TUniquePtr<FVCCSimConfigManager>(new FVCCSimConfigManager());
        UE_LOG(LogVCCSimConfigManager, Log, TEXT("VCCSimConfigManager initialized"));
    }
}

void FVCCSimConfigManager::Shutdown()
{
    if (Instance.IsValid())
    {
        // Save configuration before shutdown
        Instance->SavePanelConfiguration();
        Instance.Reset();
        UE_LOG(LogVCCSimConfigManager, Log, TEXT("VCCSimConfigManager shutdown"));
    }
}

// ============================================================================
// PANEL CONFIGURATION PERSISTENCE
// ============================================================================

void FVCCSimConfigManager::SavePanelConfiguration()
{
    SaveToJsonFile();
}

bool FVCCSimConfigManager::LoadPanelConfiguration()
{
    bool bLoaded = LoadFromJsonFile();
    if (bLoaded)
    {
        OnConfigLoaded.Broadcast(*this);
    }
    return bLoaded;
}

// ============================================================================
// JSON OPERATIONS
// ============================================================================

void FVCCSimConfigManager::SaveToJsonFile()
{
    const FString ConfigPath = GetConfigFilePath();

    TSharedPtr<FJsonObject> RootObject = MakeShareable(new FJsonObject);

    // Save panel states
    TSharedPtr<FJsonObject> PanelStatesJson = MakeShareable(new FJsonObject);
    PanelStatesJson->SetBoolField(TEXT("FlashPawnSection"), PanelStates.bFlashPawnSectionExpanded);
    PanelStatesJson->SetBoolField(TEXT("PathImageCaptureSection"), PanelStates.bPathImageCaptureSectionExpanded);
    PanelStatesJson->SetBoolField(TEXT("SceneAnalysisSection"), PanelStates.bSceneAnalysisSectionExpanded);
    PanelStatesJson->SetBoolField(TEXT("PointCloudSection"), PanelStates.bPointCloudSectionExpanded);
    PanelStatesJson->SetBoolField(TEXT("TexEnhancerSection"), PanelStates.bTexEnhancerSectionExpanded);
    RootObject->SetObjectField(TEXT("PanelStates"), PanelStatesJson);

    // Save TexEnhancer configuration
    TSharedPtr<FJsonObject> TexEnhancerConfigJson = MakeShareable(new FJsonObject);
    TexEnhancerConfigJson->SetStringField(TEXT("OutputDirectory"), TexEnhancerConfig.OutputDirectory);
    TexEnhancerConfigJson->SetStringField(TEXT("SceneName"), TexEnhancerConfig.SceneName);
    TexEnhancerConfigJson->SetStringField(TEXT("TexEnhancerScriptPath"), TexEnhancerConfig.TexEnhancerScriptPath);
    TexEnhancerConfigJson->SetStringField(TEXT("EstimatedMaterialsDir"), TexEnhancerConfig.EstimatedMaterialsDir);
    TexEnhancerConfigJson->SetBoolField(TEXT("IncludeNearbyMeshes"),     TexEnhancerConfig.bIncludeNearbyMeshes);
    TexEnhancerConfigJson->SetBoolField(TEXT("MergeNearbyMeshes"),       TexEnhancerConfig.bMergeNearbyMeshes);
    TexEnhancerConfigJson->SetNumberField(TEXT("NearbyRadius"),          TexEnhancerConfig.NearbyRadius);
    RootObject->SetObjectField(TEXT("TexEnhancerConfig"), TexEnhancerConfigJson);

    // Save shared target actor list
    {
        TArray<TSharedPtr<FJsonValue>> TargetActorsJson;
        for (int32 i = 0; i < TargetActorsConfig.Labels.Num(); ++i)
        {
            TSharedPtr<FJsonObject> EntryJson = MakeShareable(new FJsonObject);
            EntryJson->SetStringField(TEXT("Label"), TargetActorsConfig.Labels[i]);
            EntryJson->SetBoolField(TEXT("Enabled"),
                TargetActorsConfig.EnabledFlags.IsValidIndex(i) ? TargetActorsConfig.EnabledFlags[i] : true);
            TargetActorsJson.Add(MakeShareable(new FJsonValueObject(EntryJson)));
        }
        RootObject->SetArrayField(TEXT("TargetActors"), TargetActorsJson);
    }

    // Add metadata
    TSharedPtr<FJsonObject> Metadata = MakeShareable(new FJsonObject);
    Metadata->SetStringField(TEXT("Version"), TEXT("1.0"));
    Metadata->SetStringField(TEXT("SavedAt"), FDateTime::Now().ToString());
    RootObject->SetObjectField(TEXT("Metadata"), Metadata);

    // Write JSON to file
    FString OutputString;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutputString);
    FJsonSerializer::Serialize(RootObject.ToSharedRef(), Writer);

    // Ensure directory exists
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    const FString ConfigDir = FPaths::GetPath(ConfigPath);
    if (!PlatformFile.DirectoryExists(*ConfigDir))
    {
        PlatformFile.CreateDirectoryTree(*ConfigDir);
    }

    if (FFileHelper::SaveStringToFile(OutputString, *ConfigPath))
    {
        UE_LOG(LogVCCSimConfigManager, Log, TEXT("Panel configuration saved to: %s"), *ConfigPath);
    }
    else
    {
        UE_LOG(LogVCCSimConfigManager, Warning, TEXT("Failed to save panel configuration to: %s"), *ConfigPath);
    }
}

bool FVCCSimConfigManager::LoadFromJsonFile()
{
    const FString ConfigPath = GetConfigFilePath();

    FString JsonString;
    if (!FFileHelper::LoadFileToString(JsonString, *ConfigPath))
    {
        UE_LOG(LogVCCSimConfigManager, Log, TEXT("No configuration file found: %s"), *ConfigPath);
        return false;
    }

    TSharedPtr<FJsonObject> RootObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonString);

    if (!FJsonSerializer::Deserialize(Reader, RootObject) || !RootObject.IsValid())
    {
        UE_LOG(LogVCCSimConfigManager, Warning, TEXT("Failed to parse configuration JSON: %s"), *ConfigPath);
        return false;
    }

    // Load panel states
    const TSharedPtr<FJsonObject>* PanelStatesJson = nullptr;
    if (RootObject->TryGetObjectField(TEXT("PanelStates"), PanelStatesJson))
    {
        (*PanelStatesJson)->TryGetBoolField(TEXT("FlashPawnSection"), PanelStates.bFlashPawnSectionExpanded);
        (*PanelStatesJson)->TryGetBoolField(TEXT("PathImageCaptureSection"), PanelStates.bPathImageCaptureSectionExpanded);
        (*PanelStatesJson)->TryGetBoolField(TEXT("SceneAnalysisSection"), PanelStates.bSceneAnalysisSectionExpanded);
        (*PanelStatesJson)->TryGetBoolField(TEXT("PointCloudSection"), PanelStates.bPointCloudSectionExpanded);
        (*PanelStatesJson)->TryGetBoolField(TEXT("TexEnhancerSection"), PanelStates.bTexEnhancerSectionExpanded);
    }

    // Load TexEnhancer configuration
    const TSharedPtr<FJsonObject>* TexEnhancerConfigJson = nullptr;
    if (RootObject->TryGetObjectField(TEXT("TexEnhancerConfig"), TexEnhancerConfigJson))
    {
        (*TexEnhancerConfigJson)->TryGetStringField(TEXT("OutputDirectory"), TexEnhancerConfig.OutputDirectory);
        (*TexEnhancerConfigJson)->TryGetStringField(TEXT("SceneName"), TexEnhancerConfig.SceneName);
        (*TexEnhancerConfigJson)->TryGetStringField(TEXT("TexEnhancerScriptPath"), TexEnhancerConfig.TexEnhancerScriptPath);
        (*TexEnhancerConfigJson)->TryGetStringField(TEXT("EstimatedMaterialsDir"), TexEnhancerConfig.EstimatedMaterialsDir);
        {
            double V = 0.0;
            if ((*TexEnhancerConfigJson)->TryGetNumberField(TEXT("NearbyRadius"), V))
                TexEnhancerConfig.NearbyRadius = (float)V;
        }
        (*TexEnhancerConfigJson)->TryGetBoolField(TEXT("IncludeNearbyMeshes"), TexEnhancerConfig.bIncludeNearbyMeshes);
        (*TexEnhancerConfigJson)->TryGetBoolField(TEXT("MergeNearbyMeshes"),   TexEnhancerConfig.bMergeNearbyMeshes);
    }

    // Load shared target actor list
    {
        TargetActorsConfig.Labels.Empty();
        TargetActorsConfig.EnabledFlags.Empty();
        const TArray<TSharedPtr<FJsonValue>>* TargetActorsArr = nullptr;
        if (RootObject->TryGetArrayField(TEXT("TargetActors"), TargetActorsArr))
        {
            for (const TSharedPtr<FJsonValue>& Val : *TargetActorsArr)
            {
                const TSharedPtr<FJsonObject>* EntryJson = nullptr;
                if (!Val->TryGetObject(EntryJson)) continue;

                FString Label;
                if (!(*EntryJson)->TryGetStringField(TEXT("Label"), Label) || Label.IsEmpty())
                    continue;

                bool bEnabled = true;
                (*EntryJson)->TryGetBoolField(TEXT("Enabled"), bEnabled);

                TargetActorsConfig.Labels.Add(Label);
                TargetActorsConfig.EnabledFlags.Add(bEnabled);
            }
        }
    }

    UE_LOG(LogVCCSimConfigManager, Log, TEXT("Panel configuration loaded from: %s"), *ConfigPath);
    return true;
}

FString FVCCSimConfigManager::GetConfigFilePath() const
{
    return FPaths::Combine(
        FPaths::ProjectSavedDir(),
        TEXT("Config"),
        TEXT("VCCSimPanelState.json")
    );
}