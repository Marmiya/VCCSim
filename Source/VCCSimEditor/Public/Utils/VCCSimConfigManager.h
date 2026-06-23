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

/**
 * Centralized configuration manager for VCCSim panels
 * Handles JSON-based persistence of UI states and panel configurations
 * Avoids circular dependencies between panels and main panel
 */
class VCCSIMEDITOR_API FVCCSimConfigManager
{
public:
    // Singleton access
    static FVCCSimConfigManager& Get();
    static void Initialize();
    static void Shutdown();

    // ============================================================================
    // PANEL STATE PERSISTENCE
    // ============================================================================

    /** Save all panel states and configurations to JSON */
    void SavePanelConfiguration();

    /** Load panel states and configurations from JSON */
    bool LoadPanelConfiguration();

    // ============================================================================
    // PANEL STATE ACCESS
    // ============================================================================

    // UI section expanded states
    struct FPanelStates
    {
        bool bFlashPawnSectionExpanded = false;
        bool bPathImageCaptureSectionExpanded = false;
        bool bSceneAnalysisSectionExpanded = false;
        bool bPointCloudSectionExpanded = false;
        bool bTexEnhancerSectionExpanded = false;
    };

    // TexEnhancer data generation & evaluation configuration
    struct FTexEnhancerConfig
    {
        FString OutputDirectory;
        FString SceneName = TEXT("Scene_A");
        FString TexEnhancerScriptPath;
        FString EstimatedMaterialsDir;

        int32   GTTextureResolution  = 2048;
        float   DayCycleSpeed        = 10.f;

        bool    bOutputImages = true;
        bool    bOutputMesh   = true;

        TArray<float> SetAElevation;
        TArray<float> SetAAzimuth;
        TArray<float> SetBElevation;
        TArray<float> SetBAzimuth;

        float SunCalcLatitude  = 22.52933f;
        float SunCalcLongitude = 113.94092f;
        float SunCalcTimeZone  = 8.0f;
        int32 SunCalcYear      = 2026;
        int32 SunCalcMonth     = 3;
        int32 SunCalcDay       = 20;
        int32 SunCalcHour      = 10;
        int32 SunCalcMinute    = 0;
        int32 SunCalcFillSlot  = 1;
    };

    // Shared target actor list (path generation + GT material export) + bounds-select state
    struct FTargetActorsConfig
    {
        TArray<FString> Labels;
        TArray<bool>    EnabledFlags;
        FVector BoundsMin = FVector(-100000.0);
        FVector BoundsMax = FVector(100000.0);
        float   MinBuildingHeight = 300.0f;
        float   MinBuildingFootprint = 300.0f;
        float   ConnectGap = 15.0f;
    };

    // PathImageCapture orbit & capture parameters
    struct FPathImageCaptureConfig
    {
        float Margin              = 500.f;
        float StartHeight         = 200.f;
        float CameraHFOV          = 90.f;
        float HOverlap            = 0.6f;
        float VOverlap            = 0.6f;
        float SurveyHOverlap      = 0.7f;
        float NadirAltitude       = 500.f;
        float NadirTiltAngle      = 45.f;
        bool  bIncludeOblique     = false;
        int32 NumObliqueRings     = 2;
        bool  bSideOrbit          = false;
        float CaptureTickInterval = 0.2f;
    };

    // Getters and setters for panel states
    const FPanelStates& GetPanelStates() const { return PanelStates; }
    void SetPanelStates(const FPanelStates& InStates) { PanelStates = InStates; }

    // Getters and setters for TexEnhancer config
    const FTexEnhancerConfig& GetTexEnhancerConfig() const { return TexEnhancerConfig; }
    void SetTexEnhancerConfig(const FTexEnhancerConfig& InConfig) { TexEnhancerConfig = InConfig; }

    // Getters and setters for the shared target actor list
    const FTargetActorsConfig& GetTargetActorsConfig() const { return TargetActorsConfig; }
    void SetTargetActorsConfig(const FTargetActorsConfig& InConfig) { TargetActorsConfig = InConfig; }

    // Getters and setters for PathImageCapture parameters
    const FPathImageCaptureConfig& GetPathImageCaptureConfig() const { return PathImageCaptureConfig; }
    void SetPathImageCaptureConfig(const FPathImageCaptureConfig& InConfig) { PathImageCaptureConfig = InConfig; }

    // Individual state setters
    void SetFlashPawnSectionExpanded(bool bExpanded) { PanelStates.bFlashPawnSectionExpanded = bExpanded; }
    void SetPathImageCaptureSectionExpanded(bool bExpanded) { PanelStates.bPathImageCaptureSectionExpanded = bExpanded; }
    void SetSceneAnalysisSectionExpanded(bool bExpanded) { PanelStates.bSceneAnalysisSectionExpanded = bExpanded; }
    void SetPointCloudSectionExpanded(bool bExpanded) { PanelStates.bPointCloudSectionExpanded = bExpanded; }
    void SetTexEnhancerSectionExpanded(bool bExpanded) { PanelStates.bTexEnhancerSectionExpanded = bExpanded; }

    // ============================================================================
    // EVENTS AND CALLBACKS
    // ============================================================================

    DECLARE_MULTICAST_DELEGATE_OneParam(FOnConfigLoaded, const FVCCSimConfigManager&);
    FOnConfigLoaded OnConfigLoaded;

public:
    ~FVCCSimConfigManager() = default;

private:
    FVCCSimConfigManager() = default;

    // Singleton instance
    static TUniquePtr<FVCCSimConfigManager> Instance;

    // Configuration data
    FPanelStates PanelStates;
    FTexEnhancerConfig TexEnhancerConfig;
    FTargetActorsConfig TargetActorsConfig;
    FPathImageCaptureConfig PathImageCaptureConfig;

    // Serialized content (without metadata) of the last successful save,
    // used to skip redundant autosave disk writes.
    FString LastSavedContent;

    // ============================================================================
    // INTERNAL JSON OPERATIONS
    // ============================================================================

    void SaveToJsonFile();
    bool LoadFromJsonFile();
    FString GetConfigFilePath() const;
};