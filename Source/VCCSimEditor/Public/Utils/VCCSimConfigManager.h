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
        TArray<FString> GTActorLabels;
        bool    bIncludeNearbyMeshes = false;
        bool    bMergeNearbyMeshes   = false;
        float   NearbyRadius         = 500.f;
    };

    // PathImageCapture configuration
    struct FPathImageCaptureConfig
    {
        TArray<FString> OrbitActorLabels;
    };

    // Getters and setters for panel states
    const FPanelStates& GetPanelStates() const { return PanelStates; }
    void SetPanelStates(const FPanelStates& InStates) { PanelStates = InStates; }

    // Getters and setters for TexEnhancer config
    const FTexEnhancerConfig& GetTexEnhancerConfig() const { return TexEnhancerConfig; }
    void SetTexEnhancerConfig(const FTexEnhancerConfig& InConfig) { TexEnhancerConfig = InConfig; }

    // Getters and setters for PathImageCapture config
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
    FPathImageCaptureConfig PathImageCaptureConfig;

    // ============================================================================
    // INTERNAL JSON OPERATIONS
    // ============================================================================

    void SaveToJsonFile();
    bool LoadFromJsonFile();
    FString GetConfigFilePath() const;
};