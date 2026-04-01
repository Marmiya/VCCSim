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
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program. If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "CoreMinimal.h"
#include "Widgets/SCompoundWidget.h"
#include "Widgets/SWidget.h"
#include "Utils/PointCloudManager.h"

class VCCSIMEDITOR_API FVCCSimPanelPointCloud
{
public:
    FVCCSimPanelPointCloud();
    ~FVCCSimPanelPointCloud();

    void Initialize();
    void Cleanup();
    TSharedRef<SWidget> CreatePointCloudPanel();
    void UpdateVisualization();

    int32 GetPointCloudCount() const { return PointCloudMgr ? PointCloudMgr->GetPointCount() : 0; }
    bool IsPointCloudLoaded() const { return PointCloudMgr && PointCloudMgr->IsLoaded(); }
    bool IsPointCloudVisualized() const { return PointCloudMgr && PointCloudMgr->IsVisualized(); }

    bool IsPointCloudSectionExpanded() const { return bPointCloudSectionExpanded; }
    void SetPointCloudSectionExpanded(bool bExpanded) { bPointCloudSectionExpanded = bExpanded; }

private:
    TSharedRef<SWidget> CreatePointCloudButtons();
    TSharedRef<SWidget> CreatePointCloudNormalControls();

    FReply OnLoadPointCloudClicked();
    FReply OnTogglePointCloudVisualizationClicked();
    FReply OnClearPointCloudClicked();
    void OnShowNormalsChanged(ECheckBoxState NewState);
    void OnShowColorsChanged(ECheckBoxState NewState);

    TSharedPtr<FPointCloudManager> PointCloudMgr;

    TSharedPtr<SButton> LoadPointCloudButton;
    TSharedPtr<SButton> VisualizePointCloudButton;
    TSharedPtr<SCheckBox> ShowNormalsCheckBox;
    TSharedPtr<SCheckBox> ShowColorsCheckBox;
    TSharedPtr<STextBlock> PointCloudStatusText;
    TSharedPtr<STextBlock> PointCloudColorStatusText;
    TSharedPtr<STextBlock> PointCloudNormalStatusText;

    bool bShowNormals = false;
    bool bShowColors = false;
    bool bPointCloudSectionExpanded = false;
};
