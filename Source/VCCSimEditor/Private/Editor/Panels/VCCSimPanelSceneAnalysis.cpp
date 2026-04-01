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

#include "Editor/Panels/VCCSimPanelSceneAnalysis.h"
#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Simulation/SceneAnalysisManager.h"
#include "Engine/World.h"
#include "Editor.h"
#include "EngineUtils.h"

FVCCSimPanelSceneAnalysis::FVCCSimPanelSceneAnalysis()
{
    SafeDistanceValue = SafeDistance;
    SafeHeightValue = SafeHeight;
    LimitedMinXValue = LimitedMinX;
    LimitedMaxXValue = LimitedMaxX;
    LimitedMinYValue = LimitedMinY;
    LimitedMaxYValue = LimitedMaxY;
    LimitedMinZValue = LimitedMinZ;
    LimitedMaxZValue = LimitedMaxZ;
}

FVCCSimPanelSceneAnalysis::~FVCCSimPanelSceneAnalysis()
{
}

void FVCCSimPanelSceneAnalysis::Initialize(TSharedPtr<FVCCSimPanelSelection> InSelectionManager)
{
    SelectionManager = InSelectionManager;
    InitializeSceneAnalysisManager();
}

void FVCCSimPanelSceneAnalysis::Cleanup()
{
    SceneAnalysisManager.Reset();
    SelectionManager.Reset();
}

void FVCCSimPanelSceneAnalysis::InitializeSceneAnalysisManager()
{
    if (UWorld* World = GEditor->GetEditorWorldContext().World())
    {
        for (TActorIterator<ASceneAnalysisManager> It(World); It; ++It)
        {
            SceneAnalysisManager = *It;
            if (SceneAnalysisManager.IsValid())
            {
                SceneAnalysisManager->Initialize(World,
                    FPaths::ProjectSavedDir() / TEXT("VCCSimCaptures"));
                SceneAnalysisManager->InterfaceInitializeSafeZoneVisualization();
                SceneAnalysisManager->InterfaceInitializeCoverageVisualization();
                SceneAnalysisManager->InterfaceInitializeComplexityVisualization();
                break;
            }
            break;
        }
    }
}

void FVCCSimPanelSceneAnalysis::UpdateVisualization()
{
}

void FVCCSimPanelSceneAnalysis::OnUseLimitedToggleChanged(ECheckBoxState NewState)
{
    bUseLimited = (NewState == ECheckBoxState::Checked);
}

FReply FVCCSimPanelSceneAnalysis::OnToggleSafeZoneVisualizationClicked()
{
    if (!SceneAnalysisManager.IsValid())
    {
        return FReply::Handled();
    }

    bSafeZoneVisualized = !bSafeZoneVisualized;
    SceneAnalysisManager->InterfaceVisualizeSafeZone(bSafeZoneVisualized);

    VisualizeSafeZoneButton->SetButtonStyle(bSafeZoneVisualized ?
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") :
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));

    return FReply::Handled();
}

FReply FVCCSimPanelSceneAnalysis::OnToggleCoverageVisualizationClicked()
{
    if (!SceneAnalysisManager.IsValid())
    {
        return FReply::Handled();
    }

    bCoverageVisualized = !bCoverageVisualized;
    SceneAnalysisManager->InterfaceVisualizeCoverage(bCoverageVisualized);

    VisualizeCoverageButton->SetButtonStyle(bCoverageVisualized ?
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") :
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));

    return FReply::Handled();
}

FReply FVCCSimPanelSceneAnalysis::OnToggleComplexityVisualizationClicked()
{
    if (!SceneAnalysisManager.IsValid())
    {
        return FReply::Handled();
    }

    bComplexityVisualized = !bComplexityVisualized;
    SceneAnalysisManager->InterfaceVisualizeComplexity(bComplexityVisualized);

    VisualizeComplexityButton->SetButtonStyle(bComplexityVisualized ?
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") :
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));

    return FReply::Handled();
}
