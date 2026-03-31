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
    PanelStatesJson->SetBoolField(TEXT("RatSplattingSection"), PanelStates.bRatSplattingSectionExpanded);
    PanelStatesJson->SetBoolField(TEXT("TexEnhancerSection"), PanelStates.bTexEnhancerSectionExpanded);
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

    // Save TexEnhancer configuration
    TSharedPtr<FJsonObject> TexEnhancerConfigJson = MakeShareable(new FJsonObject);
    TexEnhancerConfigJson->SetStringField(TEXT("OutputDirectory"), TexEnhancerConfig.OutputDirectory);
    TexEnhancerConfigJson->SetStringField(TEXT("SceneName"), TexEnhancerConfig.SceneName);
    TexEnhancerConfigJson->SetStringField(TEXT("TexEnhancerScriptPath"), TexEnhancerConfig.TexEnhancerScriptPath);
    TexEnhancerConfigJson->SetStringField(TEXT("EstimatedMaterialsDir"), TexEnhancerConfig.EstimatedMaterialsDir);
    {
        TArray<TSharedPtr<FJsonValue>> GTLabelsJson;
        for (const FString& Label : TexEnhancerConfig.GTActorLabels)
            GTLabelsJson.Add(MakeShareable(new FJsonValueString(Label)));
        TexEnhancerConfigJson->SetArrayField(TEXT("GTActorLabels"), GTLabelsJson);
    }
    TexEnhancerConfigJson->SetStringField(TEXT("NanobananaResultDir"),    TexEnhancerConfig.NanobananaResultDir);
    TexEnhancerConfigJson->SetStringField(TEXT("NanobananaPosesFile"),    TexEnhancerConfig.NanobananaPosesFile);
    TexEnhancerConfigJson->SetStringField(TEXT("NanobananaManifestFile"), TexEnhancerConfig.NanobananaManifestFile);
    TexEnhancerConfigJson->SetNumberField(TEXT("NanobananaHFOV"),         TexEnhancerConfig.NanobananaHFOV);
    TexEnhancerConfigJson->SetNumberField(TEXT("NanobananaImageWidth"),   TexEnhancerConfig.NanobananaImageWidth);
    TexEnhancerConfigJson->SetNumberField(TEXT("NanobananaImageHeight"),  TexEnhancerConfig.NanobananaImageHeight);
    TexEnhancerConfigJson->SetNumberField(TEXT("NanobananaRaysPerClass"), TexEnhancerConfig.NanobananaRaysPerClass);
    RootObject->SetObjectField(TEXT("TexEnhancerConfig"), TexEnhancerConfigJson);

    // Save PathImageCapture configuration
    TSharedPtr<FJsonObject> PathImageCaptureConfigJson = MakeShareable(new FJsonObject);
    {
        TArray<TSharedPtr<FJsonValue>> OrbitLabelsJson;
        for (const FString& Label : PathImageCaptureConfig.OrbitActorLabels)
            OrbitLabelsJson.Add(MakeShareable(new FJsonValueString(Label)));
        PathImageCaptureConfigJson->SetArrayField(TEXT("OrbitActorLabels"), OrbitLabelsJson);
    }
    RootObject->SetObjectField(TEXT("PathImageCaptureConfig"), PathImageCaptureConfigJson);

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
        (*PanelStatesJson)->TryGetBoolField(TEXT("TexEnhancerSection"), PanelStates.bTexEnhancerSectionExpanded);
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

    // Load TexEnhancer configuration
    const TSharedPtr<FJsonObject>* TexEnhancerConfigJson = nullptr;
    if (RootObject->TryGetObjectField(TEXT("TexEnhancerConfig"), TexEnhancerConfigJson))
    {
        (*TexEnhancerConfigJson)->TryGetStringField(TEXT("OutputDirectory"), TexEnhancerConfig.OutputDirectory);
        (*TexEnhancerConfigJson)->TryGetStringField(TEXT("SceneName"), TexEnhancerConfig.SceneName);
        (*TexEnhancerConfigJson)->TryGetStringField(TEXT("TexEnhancerScriptPath"), TexEnhancerConfig.TexEnhancerScriptPath);
        (*TexEnhancerConfigJson)->TryGetStringField(TEXT("EstimatedMaterialsDir"), TexEnhancerConfig.EstimatedMaterialsDir);

        TexEnhancerConfig.GTActorLabels.Empty();
        const TArray<TSharedPtr<FJsonValue>>* GTLabelsArr = nullptr;
        if ((*TexEnhancerConfigJson)->TryGetArrayField(TEXT("GTActorLabels"), GTLabelsArr))
        {
            for (const TSharedPtr<FJsonValue>& Val : *GTLabelsArr)
            {
                FString Label;
                if (Val->TryGetString(Label))
                    TexEnhancerConfig.GTActorLabels.Add(Label);
            }
        }
        (*TexEnhancerConfigJson)->TryGetStringField(TEXT("NanobananaResultDir"),    TexEnhancerConfig.NanobananaResultDir);
        (*TexEnhancerConfigJson)->TryGetStringField(TEXT("NanobananaPosesFile"),    TexEnhancerConfig.NanobananaPosesFile);
        (*TexEnhancerConfigJson)->TryGetStringField(TEXT("NanobananaManifestFile"), TexEnhancerConfig.NanobananaManifestFile);
        {
            double V = 0.0;
            if ((*TexEnhancerConfigJson)->TryGetNumberField(TEXT("NanobananaHFOV"), V))        TexEnhancerConfig.NanobananaHFOV        = (float)V;
            if ((*TexEnhancerConfigJson)->TryGetNumberField(TEXT("NanobananaImageWidth"), V))  TexEnhancerConfig.NanobananaImageWidth  = (int32)V;
            if ((*TexEnhancerConfigJson)->TryGetNumberField(TEXT("NanobananaImageHeight"), V)) TexEnhancerConfig.NanobananaImageHeight = (int32)V;
            if ((*TexEnhancerConfigJson)->TryGetNumberField(TEXT("NanobananaRaysPerClass"), V))TexEnhancerConfig.NanobananaRaysPerClass = (int32)V;
        }
    }

    // Load PathImageCapture configuration
    const TSharedPtr<FJsonObject>* PathImageCaptureConfigJson = nullptr;
    if (RootObject->TryGetObjectField(TEXT("PathImageCaptureConfig"), PathImageCaptureConfigJson))
    {
        PathImageCaptureConfig.OrbitActorLabels.Empty();
        const TArray<TSharedPtr<FJsonValue>>* OrbitLabelsArr = nullptr;
        if ((*PathImageCaptureConfigJson)->TryGetArrayField(TEXT("OrbitActorLabels"), OrbitLabelsArr))
        {
            for (const TSharedPtr<FJsonValue>& Val : *OrbitLabelsArr)
            {
                FString Label;
                if (Val->TryGetString(Label))
                    PathImageCaptureConfig.OrbitActorLabels.Add(Label);
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