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

#include "Editor/VCCSimPanel.h"
#include "Editor/Panels/VCCSimPanelPointCloud.h"
#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Editor/Panels/VCCSimPanelPathImageCapture.h"
#include "Engine/Selection.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Text/STextBlock.h"
#include "Misc/DateTime.h"
#include "HAL/FileManager.h"
#include "Misc/Paths.h"
#include "Simulation/SceneAnalysisManager.h"
#include "Misc/FileHelper.h"
#include "Utils/TriangleSplattingManager.h"
#include "DrawDebugHelpers.h"

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

SVCCSimPanel::~SVCCSimPanel()
{
    // Unregister from selection events
    if (GEditor && GEditor->GetSelectedActors())
    {
        GEditor->GetSelectedActors()->SelectionChangedEvent.RemoveAll(this);
    }
    
    // Clean up PathImageCapture manager
    if (PathImageCaptureManager.IsValid())
    {
        PathImageCaptureManager->Cleanup();
        PathImageCaptureManager.Reset();
    }

    // Clean up scene analysis visualizations
    if (SceneAnalysisManager.IsValid())
    {
        SceneAnalysisManager->InterfaceClearSafeZoneVisualization();
        SceneAnalysisManager->InterfaceClearCoverageVisualization();
        SceneAnalysisManager->InterfaceClearComplexityVisualization();
    }

    // Clean up Point Cloud manager
    if (PointCloudManager.IsValid())
    {
        PointCloudManager->Cleanup();
        PointCloudManager.Reset();
    }
    
    // Clean up Selection manager
    if (SelectionManager.IsValid())
    {
        SelectionManager->Cleanup();
        SelectionManager.Reset();
    }
    
    // Clean up Triangle Splatting resources
    if (GSTrainingManager.IsValid() && GSTrainingManager->IsTrainingInProgress())
    {
        GSTrainingManager->StopTraining();
    }
    
    // Clear Triangle Splatting timer
    if (GEditor && GSStatusUpdateTimerHandle.IsValid())
    {
        GEditor->GetTimerManager()->ClearTimer(GSStatusUpdateTimerHandle);
        GSStatusUpdateTimerHandle.Invalidate();
    }
}

void SVCCSimPanel::OnUseLimitedToggleChanged(ECheckBoxState NewState)
{
    bUseLimited = (NewState == ECheckBoxState::Checked);
}

void SVCCSimPanel::OnSelectionChanged(UObject* Object)
{
    // Delegate to Selection Manager
    if (SelectionManager.IsValid())
    {
        USelection* Selection = GEditor->GetSelectedActors();
        if (Selection && Selection->Num() > 0)
        {
            AActor* Actor = Cast<AActor>(Selection->GetSelectedObject(0));
            if (Actor)
            {
                SelectionManager->HandleActorSelection(Actor);
            }
        }
        return;
    }
}