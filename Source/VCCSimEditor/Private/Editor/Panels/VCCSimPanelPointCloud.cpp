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

DEFINE_LOG_CATEGORY_STATIC(LogPointCloud, Log, All);

#include "Editor/Panels/VCCSimPanelPointCloud.h"
#include "Engine/World.h"
#include "Engine/Engine.h"
#include "DesktopPlatformModule.h"
#include "Framework/Application/SlateApplication.h"

FVCCSimPanelPointCloud::FVCCSimPanelPointCloud()
{
    PointCloudMgr = MakeShared<FPointCloudManager>();
}

FVCCSimPanelPointCloud::~FVCCSimPanelPointCloud()
{
    Cleanup();
}

void FVCCSimPanelPointCloud::Initialize()
{
    UE_LOG(LogPointCloud, Log, TEXT("VCCSimPanelPointCloud initialized"));
}

void FVCCSimPanelPointCloud::Cleanup()
{
    if (PointCloudMgr)
    {
        PointCloudMgr->ClearVisualization();
    }

    LoadPointCloudButton.Reset();
    VisualizePointCloudButton.Reset();
    ShowNormalsCheckBox.Reset();
    ShowColorsCheckBox.Reset();
    PointCloudStatusText.Reset();
    PointCloudColorStatusText.Reset();
    PointCloudNormalStatusText.Reset();
}

void FVCCSimPanelPointCloud::UpdateVisualization()
{
    if (PointCloudMgr && PointCloudMgr->IsLoaded() && PointCloudMgr->IsVisualized())
    {
        if (UWorld* World = GEditor->GetEditorWorldContext().World())
        {
            PointCloudMgr->ShowVisualization(World, bShowColors, bShowNormals);
        }
    }
}

FReply FVCCSimPanelPointCloud::OnLoadPointCloudClicked()
{
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (!DesktopPlatform)
    {
        return FReply::Handled();
    }

    TArray<FString> OpenedFiles;
    const bool bOpened = DesktopPlatform->OpenFileDialog(
        FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr),
        TEXT("Select PLY Point Cloud File"),
        TEXT(""),
        TEXT(""),
        TEXT("PLY Files (*.ply)|*.ply"),
        EFileDialogFlags::None,
        OpenedFiles
    );

    if (bOpened && OpenedFiles.Num() > 0)
    {
        PointCloudMgr->LoadFromFile(OpenedFiles[0]);
    }

    return FReply::Handled();
}

FReply FVCCSimPanelPointCloud::OnTogglePointCloudVisualizationClicked()
{
    if (!PointCloudMgr->IsLoaded())
    {
        return FReply::Handled();
    }

    if (PointCloudMgr->IsVisualized())
    {
        PointCloudMgr->ClearVisualization();
    }
    else
    {
        if (UWorld* World = GEditor->GetEditorWorldContext().World())
        {
            PointCloudMgr->ShowVisualization(World, bShowColors, bShowNormals);
        }
    }

    return FReply::Handled();
}

FReply FVCCSimPanelPointCloud::OnClearPointCloudClicked()
{
    if (!PointCloudMgr->IsLoaded())
    {
        return FReply::Handled();
    }

    PointCloudMgr->ClearVisualization();
    PointCloudMgr->ClearData();
    bShowNormals = false;
    bShowColors = false;

    return FReply::Handled();
}

void FVCCSimPanelPointCloud::OnShowNormalsChanged(ECheckBoxState NewState)
{
    bShowNormals = (NewState == ECheckBoxState::Checked);
    if (PointCloudMgr->IsVisualized())
    {
        PointCloudMgr->UpdateNormalVisibility(bShowNormals && PointCloudMgr->HasNormals());
    }
}

void FVCCSimPanelPointCloud::OnShowColorsChanged(ECheckBoxState NewState)
{
    bShowColors = (NewState == ECheckBoxState::Checked);
    if (PointCloudMgr->IsVisualized())
    {
        UpdateVisualization();
    }
}
