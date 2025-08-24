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
#include "Widgets/SCompoundWidget.h"
#include "VCCSimDataStructures.h"

class SVCCSimPanel;
class AFlashPawn;

/**
 * Delegate declarations for module communication
 */
DECLARE_DELEGATE_OneParam(FOnConfigurationChanged, bool /* bIsValid */);
DECLARE_DELEGATE_OneParam(FOnStatusChanged, const FString& /* StatusMessage */);
DECLARE_DELEGATE_OneParam(FOnProgressChanged, float /* Progress */);

/**
 * Base interface for VCCSim panel modules
 * Each functional area (Camera, Pose, PointCloud, etc.) will implement this interface
 */
class VCCSIMEDITOR_API IVCCSimModule
{
public:
    virtual ~IVCCSimModule() = default;

    /**
     * Initialize the module with parent panel reference
     */
    virtual void Initialize(TWeakPtr<SVCCSimPanel> InParentPanel) = 0;

    /**
     * Shutdown and cleanup the module
     */
    virtual void Shutdown() = 0;

    /**
     * Update module state (called periodically)
     */
    virtual void Tick(float DeltaTime) {}

    /**
     * Check if the module is valid and ready to use
     */
    virtual bool IsValid() const = 0;

    /**
     * Get the current status message for display
     */
    virtual FString GetStatusMessage() const = 0;

    /**
     * Handle selection changes from the editor
     */
    virtual void OnSelectionChanged(UObject* SelectedObject) {}

    /**
     * Get the module's display name
     */
    virtual FText GetDisplayName() const = 0;

    /**
     * Check if the module has any pending operations
     */
    virtual bool HasPendingOperations() const { return false; }

protected:
    TWeakPtr<SVCCSimPanel> ParentPanel;
};

/**
 * Widget interface for VCCSim panel widgets
 */
class VCCSIMEDITOR_API SVCCSimModuleWidget : public SCompoundWidget
{
public:
    SLATE_BEGIN_ARGS(SVCCSimModuleWidget) {}
    SLATE_END_ARGS()

    virtual ~SVCCSimModuleWidget() = default;

    /**
     * Initialize widget with module reference
     */
    virtual void Initialize(TSharedPtr<IVCCSimModule> InModule) { Module = InModule; }

    /**
     * Refresh widget content
     */
    virtual void RefreshContent() {}

    /**
     * Handle configuration changes
     */
    virtual void OnConfigurationChanged() {}

protected:
    TSharedPtr<IVCCSimModule> Module;
    
    /**
     * Create standard property row
     */
    TSharedRef<SWidget> CreatePropertyRow(const FString& Label, TSharedRef<SWidget> Content);
    
    /**
     * Create section header
     */
    TSharedRef<SWidget> CreateSectionHeader(const FString& Title);
    
    /**
     * Create separator line
     */
    TSharedRef<SWidget> CreateSeparator();
};

/**
 * Camera module interface
 */
class VCCSIMEDITOR_API IVCCSimCameraModule : public IVCCSimModule
{
public:
    /**
     * Get current camera configuration
     */
    virtual const FCameraConfiguration& GetCameraConfiguration() const = 0;

    /**
     * Set camera configuration
     */
    virtual void SetCameraConfiguration(const FCameraConfiguration& InConfig) = 0;

    /**
     * Update camera components based on selected FlashPawn
     */
    virtual void UpdateCameraComponents(TWeakObjectPtr<AFlashPawn> FlashPawn) = 0;

    /**
     * Get camera status information for UI display
     */
    virtual TArray<FCameraStatusInfo> GetCameraStatusInfo() const = 0;

    /**
     * Configuration change delegate
     */
    FOnConfigurationChanged OnCameraConfigChanged;
};

/**
 * Pose module interface
 */
class VCCSIMEDITOR_API IVCCSimPoseModule : public IVCCSimModule
{
public:
    /**
     * Get current pose configuration
     */
    virtual const FPoseConfiguration& GetPoseConfiguration() const = 0;

    /**
     * Set pose configuration
     */
    virtual void SetPoseConfiguration(const FPoseConfiguration& InConfig) = 0;

    /**
     * Get limited region configuration
     */
    virtual const FLimitedRegionConfiguration& GetLimitedRegionConfiguration() const = 0;

    /**
     * Set limited region configuration
     */
    virtual void SetLimitedRegionConfiguration(const FLimitedRegionConfiguration& InConfig) = 0;

    /**
     * Generate poses around target
     */
    virtual bool GeneratePoses(TWeakObjectPtr<AActor> TargetActor) = 0;

    /**
     * Load poses from file
     */
    virtual bool LoadPoses(const FString& FilePath) = 0;

    /**
     * Save current poses to file
     */
    virtual bool SavePoses(const FString& FilePath) = 0;

    /**
     * Configuration change delegates
     */
    FOnConfigurationChanged OnPoseConfigChanged;
    FOnConfigurationChanged OnLimitedRegionConfigChanged;
};

/**
 * Point Cloud module interface
 */
class VCCSIMEDITOR_API IVCCSimPointCloudModule : public IVCCSimModule
{
public:
    /**
     * Load point cloud from file
     */
    virtual bool LoadPointCloud(const FString& FilePath) = 0;

    /**
     * Get point cloud information
     */
    virtual const FPointCloudInfo& GetPointCloudInfo() const = 0;

    /**
     * Toggle point cloud visualization
     */
    virtual void ToggleVisualization() = 0;

    /**
     * Toggle normal lines visualization
     */
    virtual void ToggleNormalLines(bool bShow) = 0;

    /**
     * Check if point cloud is loaded
     */
    virtual bool IsPointCloudLoaded() const = 0;

    /**
     * Check if visualization is active
     */
    virtual bool IsVisualizationActive() const = 0;

    /**
     * Progress change delegate for loading operations
     */
    FOnProgressChanged OnLoadingProgressChanged;
};

/**
 * Triangle Splatting module interface
 */
class VCCSIMEDITOR_API IVCCSimTriangleSplattingModule : public IVCCSimModule
{
public:
    /**
     * Get current Triangle Splatting configuration
     */
    virtual const FTriangleSplattingConfiguration& GetConfiguration() const = 0;

    /**
     * Set Triangle Splatting configuration
     */
    virtual void SetConfiguration(const FTriangleSplattingConfiguration& InConfig) = 0;

    /**
     * Start training process
     */
    virtual bool StartTraining() = 0;

    /**
     * Stop training process
     */
    virtual void StopTraining() = 0;

    /**
     * Check if training is in progress
     */
    virtual bool IsTrainingInProgress() const = 0;

    /**
     * Get training progress (0.0 to 1.0)
     */
    virtual float GetTrainingProgress() const = 0;

    /**
     * Export to COLMAP format
     */
    virtual bool ExportToColmap() = 0;

    /**
     * Test coordinate transformation
     */
    virtual bool TestTransformation() = 0;

    /**
     * Configuration and progress change delegates
     */
    FOnConfigurationChanged OnConfigurationChanged;
    FOnProgressChanged OnTrainingProgressChanged;
};

/**
 * Scene Analysis module interface
 */
class VCCSIMEDITOR_API IVCCSimSceneAnalysisModule : public IVCCSimModule
{
public:
    /**
     * Get current scene analysis configuration
     */
    virtual const FSceneAnalysisConfiguration& GetConfiguration() const = 0;

    /**
     * Set scene analysis configuration
     */
    virtual void SetConfiguration(const FSceneAnalysisConfiguration& InConfig) = 0;

    /**
     * Toggle safe zone visualization
     */
    virtual void ToggleSafeZoneVisualization() = 0;

    /**
     * Toggle coverage visualization
     */
    virtual void ToggleCoverageVisualization() = 0;

    /**
     * Toggle complexity visualization
     */
    virtual void ToggleComplexityVisualization() = 0;

    /**
     * Check visualization states
     */
    virtual bool IsSafeZoneVisualized() const = 0;
    virtual bool IsCoverageVisualized() const = 0;
    virtual bool IsComplexityVisualized() const = 0;

    /**
     * Configuration change delegate
     */
    FOnConfigurationChanged OnAnalysisConfigChanged;
};