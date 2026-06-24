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
    TexEnhancerConfigJson->SetNumberField(TEXT("GTTextureResolution"),   TexEnhancerConfig.GTTextureResolution);
    TexEnhancerConfigJson->SetNumberField(TEXT("DayCycleSpeed"),         TexEnhancerConfig.DayCycleSpeed);
    TexEnhancerConfigJson->SetBoolField(TEXT("OutputImages"),            TexEnhancerConfig.bOutputImages);
    TexEnhancerConfigJson->SetBoolField(TEXT("OutputMesh"),              TexEnhancerConfig.bOutputMesh);

    {
        auto MakeFloatArrayJson = [](const TArray<float>& Values)
        {
            TArray<TSharedPtr<FJsonValue>> Arr;
            for (float V : Values)
                Arr.Add(MakeShareable(new FJsonValueNumber(V)));
            return Arr;
        };
        TexEnhancerConfigJson->SetArrayField(TEXT("LightingElevation"), MakeFloatArrayJson(TexEnhancerConfig.LightingElevation));
        TexEnhancerConfigJson->SetArrayField(TEXT("LightingAzimuth"),   MakeFloatArrayJson(TexEnhancerConfig.LightingAzimuth));

        TArray<TSharedPtr<FJsonValue>> SelectedArr;
        for (bool bSel : TexEnhancerConfig.LightingSelected)
            SelectedArr.Add(MakeShareable(new FJsonValueBoolean(bSel)));
        TexEnhancerConfigJson->SetArrayField(TEXT("LightingSelected"), SelectedArr);
    }

    TexEnhancerConfigJson->SetNumberField(TEXT("SunCalcLatitude"),  TexEnhancerConfig.SunCalcLatitude);
    TexEnhancerConfigJson->SetNumberField(TEXT("SunCalcLongitude"), TexEnhancerConfig.SunCalcLongitude);
    TexEnhancerConfigJson->SetNumberField(TEXT("SunCalcTimeZone"),  TexEnhancerConfig.SunCalcTimeZone);
    TexEnhancerConfigJson->SetNumberField(TEXT("SunCalcYear"),      TexEnhancerConfig.SunCalcYear);
    TexEnhancerConfigJson->SetNumberField(TEXT("SunCalcMonth"),     TexEnhancerConfig.SunCalcMonth);
    TexEnhancerConfigJson->SetNumberField(TEXT("SunCalcDay"),       TexEnhancerConfig.SunCalcDay);
    TexEnhancerConfigJson->SetNumberField(TEXT("SunCalcHour"),      TexEnhancerConfig.SunCalcHour);
    TexEnhancerConfigJson->SetNumberField(TEXT("SunCalcMinute"),    TexEnhancerConfig.SunCalcMinute);
    TexEnhancerConfigJson->SetNumberField(TEXT("SunCalcFillSlot"),  TexEnhancerConfig.SunCalcFillSlot);
    RootObject->SetObjectField(TEXT("TexEnhancerConfig"), TexEnhancerConfigJson);

    // Save PathImageCapture parameters
    {
        TSharedPtr<FJsonObject> PathCaptureJson = MakeShareable(new FJsonObject);
        PathCaptureJson->SetNumberField(TEXT("Margin"),              PathImageCaptureConfig.Margin);
        PathCaptureJson->SetNumberField(TEXT("StartHeight"),         PathImageCaptureConfig.StartHeight);
        PathCaptureJson->SetNumberField(TEXT("CameraHFOV"),          PathImageCaptureConfig.CameraHFOV);
        PathCaptureJson->SetNumberField(TEXT("HOverlap"),            PathImageCaptureConfig.HOverlap);
        PathCaptureJson->SetNumberField(TEXT("VOverlap"),            PathImageCaptureConfig.VOverlap);
        PathCaptureJson->SetNumberField(TEXT("SurveyHOverlap"),      PathImageCaptureConfig.SurveyHOverlap);
        PathCaptureJson->SetNumberField(TEXT("NadirAltitude"),       PathImageCaptureConfig.NadirAltitude);
        PathCaptureJson->SetNumberField(TEXT("NadirTiltAngle"),      PathImageCaptureConfig.NadirTiltAngle);
        PathCaptureJson->SetBoolField(TEXT("IncludeOblique"),        PathImageCaptureConfig.bIncludeOblique);
        PathCaptureJson->SetNumberField(TEXT("NumObliqueRings"),     PathImageCaptureConfig.NumObliqueRings);
        PathCaptureJson->SetBoolField(TEXT("SideOrbit"),             PathImageCaptureConfig.bSideOrbit);
        PathCaptureJson->SetNumberField(TEXT("CaptureTickInterval"), PathImageCaptureConfig.CaptureTickInterval);
        RootObject->SetObjectField(TEXT("PathImageCaptureConfig"), PathCaptureJson);
    }

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

        TSharedPtr<FJsonObject> BoundsJson = MakeShareable(new FJsonObject);
        BoundsJson->SetNumberField(TEXT("MinX"), TargetActorsConfig.BoundsMin.X);
        BoundsJson->SetNumberField(TEXT("MinY"), TargetActorsConfig.BoundsMin.Y);
        BoundsJson->SetNumberField(TEXT("MinZ"), TargetActorsConfig.BoundsMin.Z);
        BoundsJson->SetNumberField(TEXT("MaxX"), TargetActorsConfig.BoundsMax.X);
        BoundsJson->SetNumberField(TEXT("MaxY"), TargetActorsConfig.BoundsMax.Y);
        BoundsJson->SetNumberField(TEXT("MaxZ"), TargetActorsConfig.BoundsMax.Z);
        BoundsJson->SetNumberField(TEXT("MinBuildingHeight"), TargetActorsConfig.MinBuildingHeight);
        BoundsJson->SetNumberField(TEXT("MinBuildingFootprint"), TargetActorsConfig.MinBuildingFootprint);
        BoundsJson->SetNumberField(TEXT("ConnectGap"), TargetActorsConfig.ConnectGap);
        RootObject->SetObjectField(TEXT("BoundsSelect"), BoundsJson);
    }

    // Skip the disk write when nothing changed (the panel autosave timer calls
    // this periodically); compare BEFORE the SavedAt timestamp is added.
    FString ContentString;
    {
        TSharedRef<TJsonWriter<>> ContentWriter = TJsonWriterFactory<>::Create(&ContentString);
        FJsonSerializer::Serialize(RootObject.ToSharedRef(), ContentWriter);
    }
    if (ContentString == LastSavedContent)
    {
        return;
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
        LastSavedContent = MoveTemp(ContentString);
        UE_LOG(LogVCCSimConfigManager, Verbose, TEXT("Panel configuration saved to: %s"), *ConfigPath);
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
            if ((*TexEnhancerConfigJson)->TryGetNumberField(TEXT("GTTextureResolution"), V))  TexEnhancerConfig.GTTextureResolution = (int32)V;
            if ((*TexEnhancerConfigJson)->TryGetNumberField(TEXT("DayCycleSpeed"), V))        TexEnhancerConfig.DayCycleSpeed       = (float)V;
            if ((*TexEnhancerConfigJson)->TryGetNumberField(TEXT("SunCalcLatitude"), V))      TexEnhancerConfig.SunCalcLatitude     = (float)V;
            if ((*TexEnhancerConfigJson)->TryGetNumberField(TEXT("SunCalcLongitude"), V))     TexEnhancerConfig.SunCalcLongitude    = (float)V;
            if ((*TexEnhancerConfigJson)->TryGetNumberField(TEXT("SunCalcTimeZone"), V))      TexEnhancerConfig.SunCalcTimeZone     = (float)V;
            if ((*TexEnhancerConfigJson)->TryGetNumberField(TEXT("SunCalcYear"), V))          TexEnhancerConfig.SunCalcYear         = (int32)V;
            if ((*TexEnhancerConfigJson)->TryGetNumberField(TEXT("SunCalcMonth"), V))         TexEnhancerConfig.SunCalcMonth        = (int32)V;
            if ((*TexEnhancerConfigJson)->TryGetNumberField(TEXT("SunCalcDay"), V))           TexEnhancerConfig.SunCalcDay          = (int32)V;
            if ((*TexEnhancerConfigJson)->TryGetNumberField(TEXT("SunCalcHour"), V))          TexEnhancerConfig.SunCalcHour         = (int32)V;
            if ((*TexEnhancerConfigJson)->TryGetNumberField(TEXT("SunCalcMinute"), V))        TexEnhancerConfig.SunCalcMinute       = (int32)V;
            if ((*TexEnhancerConfigJson)->TryGetNumberField(TEXT("SunCalcFillSlot"), V))      TexEnhancerConfig.SunCalcFillSlot     = (int32)V;
        }
        (*TexEnhancerConfigJson)->TryGetBoolField(TEXT("OutputImages"), TexEnhancerConfig.bOutputImages);
        (*TexEnhancerConfigJson)->TryGetBoolField(TEXT("OutputMesh"),   TexEnhancerConfig.bOutputMesh);

        {
            auto LoadFloatArray = [&TexEnhancerConfigJson](const TCHAR* Field, TArray<float>& Out)
            {
                const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
                if (!(*TexEnhancerConfigJson)->TryGetArrayField(Field, Arr)) return;
                Out.Empty();
                for (const TSharedPtr<FJsonValue>& Val : *Arr)
                {
                    double V = 0.0;
                    if (Val->TryGetNumber(V))
                        Out.Add((float)V);
                }
            };
            LoadFloatArray(TEXT("LightingElevation"), TexEnhancerConfig.LightingElevation);
            LoadFloatArray(TEXT("LightingAzimuth"),   TexEnhancerConfig.LightingAzimuth);

            const TArray<TSharedPtr<FJsonValue>>* SelectedArr = nullptr;
            if ((*TexEnhancerConfigJson)->TryGetArrayField(TEXT("LightingSelected"), SelectedArr))
            {
                TexEnhancerConfig.LightingSelected.Empty();
                for (const TSharedPtr<FJsonValue>& Val : *SelectedArr)
                {
                    bool bSel = false;
                    if (Val->TryGetBool(bSel))
                        TexEnhancerConfig.LightingSelected.Add(bSel);
                }
            }
        }
    }

    // Load PathImageCapture parameters
    const TSharedPtr<FJsonObject>* PathCaptureJson = nullptr;
    if (RootObject->TryGetObjectField(TEXT("PathImageCaptureConfig"), PathCaptureJson))
    {
        double V = 0.0;
        if ((*PathCaptureJson)->TryGetNumberField(TEXT("Margin"), V))              PathImageCaptureConfig.Margin              = (float)V;
        if ((*PathCaptureJson)->TryGetNumberField(TEXT("StartHeight"), V))         PathImageCaptureConfig.StartHeight         = (float)V;
        if ((*PathCaptureJson)->TryGetNumberField(TEXT("CameraHFOV"), V))          PathImageCaptureConfig.CameraHFOV          = (float)V;
        if ((*PathCaptureJson)->TryGetNumberField(TEXT("HOverlap"), V))            PathImageCaptureConfig.HOverlap            = (float)V;
        if ((*PathCaptureJson)->TryGetNumberField(TEXT("VOverlap"), V))            PathImageCaptureConfig.VOverlap            = (float)V;
        if ((*PathCaptureJson)->TryGetNumberField(TEXT("SurveyHOverlap"), V))      PathImageCaptureConfig.SurveyHOverlap      = (float)V;
        if ((*PathCaptureJson)->TryGetNumberField(TEXT("NadirAltitude"), V))       PathImageCaptureConfig.NadirAltitude       = (float)V;
        if ((*PathCaptureJson)->TryGetNumberField(TEXT("NadirTiltAngle"), V))      PathImageCaptureConfig.NadirTiltAngle      = (float)V;
        if ((*PathCaptureJson)->TryGetNumberField(TEXT("NumObliqueRings"), V))     PathImageCaptureConfig.NumObliqueRings     = (int32)V;
        if ((*PathCaptureJson)->TryGetNumberField(TEXT("CaptureTickInterval"), V)) PathImageCaptureConfig.CaptureTickInterval = (float)V;
        (*PathCaptureJson)->TryGetBoolField(TEXT("IncludeOblique"), PathImageCaptureConfig.bIncludeOblique);
        (*PathCaptureJson)->TryGetBoolField(TEXT("SideOrbit"), PathImageCaptureConfig.bSideOrbit);
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

        const TSharedPtr<FJsonObject>* BoundsJson = nullptr;
        if (RootObject->TryGetObjectField(TEXT("BoundsSelect"), BoundsJson))
        {
            (*BoundsJson)->TryGetNumberField(TEXT("MinX"), TargetActorsConfig.BoundsMin.X);
            (*BoundsJson)->TryGetNumberField(TEXT("MinY"), TargetActorsConfig.BoundsMin.Y);
            (*BoundsJson)->TryGetNumberField(TEXT("MinZ"), TargetActorsConfig.BoundsMin.Z);
            (*BoundsJson)->TryGetNumberField(TEXT("MaxX"), TargetActorsConfig.BoundsMax.X);
            (*BoundsJson)->TryGetNumberField(TEXT("MaxY"), TargetActorsConfig.BoundsMax.Y);
            (*BoundsJson)->TryGetNumberField(TEXT("MaxZ"), TargetActorsConfig.BoundsMax.Z);
            (*BoundsJson)->TryGetNumberField(TEXT("MinBuildingHeight"), TargetActorsConfig.MinBuildingHeight);
            (*BoundsJson)->TryGetNumberField(TEXT("MinBuildingFootprint"), TargetActorsConfig.MinBuildingFootprint);
            (*BoundsJson)->TryGetNumberField(TEXT("ConnectGap"), TargetActorsConfig.ConnectGap);
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