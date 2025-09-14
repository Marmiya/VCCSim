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
    PanelStatesJson->SetBoolField(TEXT("RatSplattingSection"), PanelStates.bRatSplattingSectionExpanded);
    RootObject->SetObjectField(TEXT("PanelStates"), PanelStatesJson);

    // Save RatSplatting configuration
    TSharedPtr<FJsonObject> RatSplattingConfigJson = MakeShareable(new FJsonObject);
    RatSplattingConfigJson->SetStringField(TEXT("ImageDirectory"), RatSplattingConfig.ImageDirectory);
    RatSplattingConfigJson->SetStringField(TEXT("CameraIntrinsicsFilePath"), RatSplattingConfig.CameraIntrinsicsFilePath);
    RatSplattingConfigJson->SetStringField(TEXT("PoseFilePath"), RatSplattingConfig.PoseFilePath);
    RatSplattingConfigJson->SetStringField(TEXT("OutputDirectory"), RatSplattingConfig.OutputDirectory);
    RatSplattingConfigJson->SetStringField(TEXT("ColmapDatasetPath"), RatSplattingConfig.ColmapDatasetPath);

    const FString MeshPath = RatSplattingConfig.SelectedMesh.IsValid() ?
        RatSplattingConfig.SelectedMesh->GetPathName() : TEXT("");
    RatSplattingConfigJson->SetStringField(TEXT("SelectedMeshPath"), MeshPath);

    RootObject->SetObjectField(TEXT("RatSplattingConfig"), RatSplattingConfigJson);

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
        (*PanelStatesJson)->TryGetBoolField(TEXT("RatSplattingSection"), PanelStates.bRatSplattingSectionExpanded);
    }

    // Load RatSplatting configuration
    const TSharedPtr<FJsonObject>* RatSplattingConfigJson = nullptr;
    if (RootObject->TryGetObjectField(TEXT("RatSplattingConfig"), RatSplattingConfigJson))
    {
        (*RatSplattingConfigJson)->TryGetStringField(TEXT("ImageDirectory"), RatSplattingConfig.ImageDirectory);
        (*RatSplattingConfigJson)->TryGetStringField(TEXT("CameraIntrinsicsFilePath"), RatSplattingConfig.CameraIntrinsicsFilePath);
        (*RatSplattingConfigJson)->TryGetStringField(TEXT("PoseFilePath"), RatSplattingConfig.PoseFilePath);
        (*RatSplattingConfigJson)->TryGetStringField(TEXT("OutputDirectory"), RatSplattingConfig.OutputDirectory);
        (*RatSplattingConfigJson)->TryGetStringField(TEXT("ColmapDatasetPath"), RatSplattingConfig.ColmapDatasetPath);

        FString SelectedMeshPath;
        if ((*RatSplattingConfigJson)->TryGetStringField(TEXT("SelectedMeshPath"), SelectedMeshPath) &&
            !SelectedMeshPath.IsEmpty())
        {
            if (UStaticMesh* LoadedMesh = LoadObject<UStaticMesh>(nullptr, *SelectedMeshPath))
            {
                RatSplattingConfig.SelectedMesh = LoadedMesh;
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