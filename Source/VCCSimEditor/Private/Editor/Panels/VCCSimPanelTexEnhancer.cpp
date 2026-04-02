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

#include "Editor/Panels/VCCSimPanelTexEnhancer.h"
#include "Utils/VCCSimSunPositionHelper.h"
#include "Utils/VCCSimUIHelpers.h"
#include "Utils/VCCSimConfigManager.h"
#include "Utils/LightingManager.h"
#include "Utils/GTMaterialExporter.h"
#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/StaticMesh.h"
#include "Engine/Texture.h"
#include "Engine/Texture2D.h"
#include "Materials/MaterialInterface.h"
#include "Selection.h"
#include "MeshDescription.h"
#include "Modules/ModuleManager.h"

#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "HAL/PlatformFilemanager.h"
#include "HAL/PlatformProcess.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#include "Framework/Application/SlateApplication.h"
#include "Editor.h"
#include "TimerManager.h"
#include "Async/Async.h"

DEFINE_LOG_CATEGORY_STATIC(LogTexEnhancerPanel, Log, All);

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

FVCCSimPanelTexEnhancer::FVCCSimPanelTexEnhancer()
{
    GTMaterialExporter = MakeShared<FGTMaterialExporter>();
    NanobananaManager  = MakeShared<FNanobananaManager>();

    for (int32 i = 0; i < MaxLightingEntries; ++i)
    {
        SetAElevationValue[i] = SetAElevation[i];
        SetAAzimuthValue[i]   = SetAAzimuth[i];
        SetBElevationValue[i] = SetBElevation[i];
        SetBAzimuthValue[i]   = SetBAzimuth[i];
    }
    SunCalcLatValue      = SunCalcLatitude;
    SunCalcLonValue      = SunCalcLongitude;
    SunCalcTZValue       = SunCalcTimeZone;
    SunCalcYearValue     = SunCalcYear;
    SunCalcMonthValue    = SunCalcMonth;
    SunCalcDayValue      = SunCalcDay;
    SunCalcHourValue     = SunCalcHour;
    SunCalcMinuteValue   = SunCalcMinute;
    SunCalcFillSlotValue = SunCalcFillSlot;
    GTTexResValue        = GTTextureResolution;
    DayCycleSpeedValue   = DayCycleSpeed;

    NanobananaHFOVValue         = NanobananaHFOV;
    NanobananaImageWidthValue   = NanobananaImageWidth;
    NanobananaImageHeightValue  = NanobananaImageHeight;
    NanobananaOverlayAlphaValue = NanobananaOverlayAlpha;
}

FVCCSimPanelTexEnhancer::~FVCCSimPanelTexEnhancer()
{
    Cleanup();
}

// ============================================================================
// INITIALIZATION
// ============================================================================

void FVCCSimPanelTexEnhancer::Initialize()
{
    LightingManager = MakeShared<FLightingManager>(GEditor->GetEditorWorldContext().World());
    LoadPaths();
    UE_LOG(LogTexEnhancerPanel, Log, TEXT("TexEnhancer panel initialized"));
}

void FVCCSimPanelTexEnhancer::Cleanup()
{
    LightingManager.Reset();
    if (GEditor)
    {
        GEditor->GetTimerManager()->ClearTimer(StatusTimerHandle);
    }
    bNanobananaInProgress = false;

    if (PipelineProcHandle.IsValid())
    {
        FPlatformProcess::TerminateProc(PipelineProcHandle, false);
        FPlatformProcess::CloseProc(PipelineProcHandle);
    }

    SavePaths();
    UE_LOG(LogTexEnhancerPanel, Log, TEXT("TexEnhancer panel cleaned up"));
}

void FVCCSimPanelTexEnhancer::SetSelectionManager(TSharedPtr<FVCCSimPanelSelection> InSelectionManager)
{
    SelectionManager = InSelectionManager;
}

void FVCCSimPanelTexEnhancer::LoadFromConfigManager()
{
    const auto& Config = FVCCSimConfigManager::Get().GetTexEnhancerConfig();

    if (!Config.OutputDirectory.IsEmpty())
    {
        OutputDirectory = Config.OutputDirectory;
        if (OutputDirTextBox.IsValid())
        {
            OutputDirTextBox->SetText(FText::FromString(OutputDirectory));
        }
    }
    if (!Config.SceneName.IsEmpty())
    {
        SceneName = Config.SceneName;
        if (SceneNameTextBox.IsValid())
        {
            SceneNameTextBox->SetText(FText::FromString(SceneName));
        }
    }
    if (!Config.TexEnhancerScriptPath.IsEmpty())
    {
        TexEnhancerScriptPath = Config.TexEnhancerScriptPath;
        if (TexEnhancerScriptTextBox.IsValid())
        {
            TexEnhancerScriptTextBox->SetText(FText::FromString(TexEnhancerScriptPath));
        }
    }
    if (!Config.EstimatedMaterialsDir.IsEmpty())
    {
        EstimatedMaterialsDir = Config.EstimatedMaterialsDir;
        if (EstimatedMaterialsDirTextBox.IsValid())
        {
            EstimatedMaterialsDirTextBox->SetText(FText::FromString(EstimatedMaterialsDir));
        }
    }

    if (!Config.GTActorLabels.IsEmpty())
    {
        GTActorListItems.Empty();
        for (const FString& Label : Config.GTActorLabels)
            GTActorListItems.Add(MakeShareable(new FString(Label)));
        if (GTActorListView.IsValid())
            GTActorListView->RequestListRefresh();
    }

    if (!Config.NanobananaResultDir.IsEmpty())
    {
        NanobananaResultDir = Config.NanobananaResultDir;
        if (NanobananaResultDirTextBox.IsValid())
            NanobananaResultDirTextBox->SetText(FText::FromString(NanobananaResultDir));
    }
    if (!Config.NanobananaPosesFile.IsEmpty())
    {
        NanobananaPosesFile = Config.NanobananaPosesFile;
        if (NanobananaPosesFileTextBox.IsValid())
            NanobananaPosesFileTextBox->SetText(FText::FromString(NanobananaPosesFile));
    }
    if (!Config.NanobananaManifestFile.IsEmpty())
    {
        NanobananaManifestFile = Config.NanobananaManifestFile;
        if (NanobananaManifestFileTextBox.IsValid())
            NanobananaManifestFileTextBox->SetText(FText::FromString(NanobananaManifestFile));
    }
    NanobananaHFOV         = Config.NanobananaHFOV;
    NanobananaImageWidth   = Config.NanobananaImageWidth;
    NanobananaImageHeight  = Config.NanobananaImageHeight;
    NanobananaOverlayAlpha = Config.NanobananaOverlayAlpha;
    NanobananaHFOVValue         = NanobananaHFOV;
    NanobananaImageWidthValue   = NanobananaImageWidth;
    NanobananaImageHeightValue  = NanobananaImageHeight;
    NanobananaOverlayAlphaValue = NanobananaOverlayAlpha;
}

// ============================================================================
// SECTION 1: DATASET CONFIGURATION
// ============================================================================

FReply FVCCSimPanelTexEnhancer::OnBrowseOutputDirClicked()
{
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (!DesktopPlatform) return FReply::Handled();

    FString SelectedDir;
    void* ParentWindowHandle = const_cast<void*>(FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr));

    if (DesktopPlatform->OpenDirectoryDialog(ParentWindowHandle, TEXT("Select Output Directory"), OutputDirectory, SelectedDir))
    {
        OutputDirectory = SelectedDir;
        if (OutputDirTextBox.IsValid())
        {
            OutputDirTextBox->SetText(FText::FromString(OutputDirectory));
        }
        SavePaths();
    }

    return FReply::Handled();
}

// ============================================================================
// SECTION 2: LIGHTING SCHEDULE
// ============================================================================

FReply FVCCSimPanelTexEnhancer::OnApplySetALightingClicked(int32 Index)
{
    if (Index < 0 || Index >= NumLightingSetA || !LightingManager.IsValid()) return FReply::Handled();
    LightingManager->ApplyLightingCondition(SetAElevation[Index], SetAAzimuth[Index]);
    if (LightingStatusTextBlock.IsValid())
        LightingStatusTextBlock->SetText(FText::FromString(FString::Printf(TEXT("Set-A%d Applied: Elev=%.1f\u00B0 Az=%.1f\u00B0"), Index + 1, SetAElevation[Index], SetAAzimuth[Index])));
    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnApplySetBLightingClicked(int32 Index)
{
    if (Index < 0 || Index >= NumLightingSetB || !LightingManager.IsValid()) return FReply::Handled();
    LightingManager->ApplyLightingCondition(SetBElevation[Index], SetBAzimuth[Index]);
    if (LightingStatusTextBlock.IsValid())
        LightingStatusTextBlock->SetText(FText::FromString(FString::Printf(TEXT("Set-B%d Applied: Elev=%.1f\u00B0 Az=%.1f\u00B0"), Index + 1, SetBElevation[Index], SetBAzimuth[Index])));
    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnCalculateSunPositionClicked()
{
    if (!LightingManager.IsValid()) return FReply::Handled();

    FVCCSimSunPositionHelper::FSunParams Params;
    Params.Latitude  = SunCalcLatitude;
    Params.Longitude = SunCalcLongitude;
    Params.TimeZone  = SunCalcTimeZone;
    Params.Year      = SunCalcYear;
    Params.Month     = SunCalcMonth;
    Params.Day       = SunCalcDay;
    Params.Hour      = SunCalcHour;
    Params.Minute    = SunCalcMinute;

    TPair<float, float> SunPos = LightingManager->CalculateAndApplySunPosition(Params);
    SunCalcElevation = SunPos.Key;
    SunCalcAzimuth = SunPos.Value;
    
    if (LightingStatusTextBlock.IsValid())
        LightingStatusTextBlock->SetText(FText::FromString(FString::Printf(TEXT("Calculated & Applied: Elev=%.1f\u00B0 Az=%.1f\u00B0"), SunCalcElevation, SunCalcAzimuth)));

    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnFillSetAFromSunPositionClicked()
{
    int32 SlotIdx = FMath::Clamp(SunCalcFillSlot - 1, 0, NumLightingSetA - 1);
    SetAElevation[SlotIdx]      = SunCalcElevation;
    SetAAzimuth[SlotIdx]        = SunCalcAzimuth;
    SetAElevationValue[SlotIdx] = SunCalcElevation;
    SetAAzimuthValue[SlotIdx]   = SunCalcAzimuth;

    if (SetAElevationSpinBox[SlotIdx].IsValid())
    {
        SetAElevationSpinBox[SlotIdx]->Invalidate(EInvalidateWidgetReason::Paint);
    }
    if (SetAAzimuthSpinBox[SlotIdx].IsValid())
    {
        SetAAzimuthSpinBox[SlotIdx]->Invalidate(EInvalidateWidgetReason::Paint);
    }

    UE_LOG(LogTexEnhancerPanel, Log,
        TEXT("Sun position filled into Set-A slot %d: Elevation=%.1f°  Azimuth=%.1f°"),
        SunCalcFillSlot, SunCalcElevation, SunCalcAzimuth);

    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnFillSetBFromSunPositionClicked()
{
    int32 SlotIdx = FMath::Clamp(SunCalcFillSlot - 1, 0, NumLightingSetB - 1);
    SetBElevation[SlotIdx]      = SunCalcElevation;
    SetBAzimuth[SlotIdx]        = SunCalcAzimuth;
    SetBElevationValue[SlotIdx] = SunCalcElevation;
    SetBAzimuthValue[SlotIdx]   = SunCalcAzimuth;

    if (SetBElevationSpinBox[SlotIdx].IsValid())
    {
        SetBElevationSpinBox[SlotIdx]->Invalidate(EInvalidateWidgetReason::Paint);
    }
    if (SetBAzimuthSpinBox[SlotIdx].IsValid())
    {
        SetBAzimuthSpinBox[SlotIdx]->Invalidate(EInvalidateWidgetReason::Paint);
    }

    UE_LOG(LogTexEnhancerPanel, Log,
        TEXT("Sun position filled into Set-B slot %d: Elevation=%.1f°  Azimuth=%.1f°"),
        SunCalcFillSlot, SunCalcElevation, SunCalcAzimuth);

    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnToggleDayCycleClicked()
{
    if (!LightingManager.IsValid()) return FReply::Handled();

    bDayCycleActive = !bDayCycleActive;
    
    FVCCSimSunPositionHelper::FSunParams Params;
    Params.Latitude  = SunCalcLatitude;
    Params.Longitude = SunCalcLongitude;
    Params.TimeZone  = SunCalcTimeZone;
    Params.Year      = SunCalcYear;
    Params.Month     = SunCalcMonth;
    Params.Day       = SunCalcDay;

    LightingManager->ToggleDayCycle(bDayCycleActive, Params, DayCycleSpeed);

    return FReply::Handled();
}

// ============================================================================
// SECTION 4: GT MATERIAL EXPORT
// ============================================================================

FReply FVCCSimPanelTexEnhancer::OnAddSelectedActorsClicked()
{
    USelection* Sel = GEditor ? GEditor->GetSelectedActors() : nullptr;
    if (!Sel) return FReply::Handled();

    bool bAdded = false;
    for (int32 i = 0; i < Sel->Num(); ++i)
    {
        AStaticMeshActor* SMA = Cast<AStaticMeshActor>(Sel->GetSelectedObject(i));
        if (!SMA) continue;

        const FString Label = SMA->GetActorLabel();
        bool bDuplicate = GTActorListItems.ContainsByPredicate(
            [&Label](const TSharedPtr<FString>& P) { return P.IsValid() && *P == Label; });

        if (!bDuplicate)
        {
            GTActorListItems.Add(MakeShareable(new FString(Label)));
            bAdded = true;
        }
    }

    if (bAdded && GTActorListView.IsValid())
        GTActorListView->RequestListRefresh();
    if (bAdded)
        SavePaths();

    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnRemoveFromGTListClicked()
{
    if (!GTActorListView.IsValid()) return FReply::Handled();

    TArray<TSharedPtr<FString>> Selected = GTActorListView->GetSelectedItems();
    if (Selected.IsEmpty()) return FReply::Handled();

    for (const TSharedPtr<FString>& Item : Selected)
    {
        if (!Item.IsValid()) continue;
        const FString ItemStr = *Item;
        GTActorListItems.RemoveAll([&ItemStr](const TSharedPtr<FString>& P)
        {
            return P.IsValid() && *P == ItemStr;
        });
    }

    GTActorListView->ClearSelection();
    GTActorListView->RequestListRefresh();
    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnExportGTMaterialsClicked()
{
    if (bGTExportInProgress)
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("GT material export already in progress."), true);
        return FReply::Handled();
    }
    if (OutputDirectory.IsEmpty())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Output directory is not set."), true);
        return FReply::Handled();
    }
    if (GTActorListItems.IsEmpty())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("GT actor list is empty. Select actors in viewport and click '+ Add Selected'."), true);
        return FReply::Handled();
    }
    if (!GEditor || !GEditor->GetEditorWorldContext().World())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("No editor world available."), true);
        return FReply::Handled();
    }

    bGTExportInProgress = true;
    FVCCSimUIHelpers::ShowNotification(TEXT("Starting GT material export..."));

    FSimpleDelegate OnComplete = FSimpleDelegate::CreateLambda([this]()
    {
        bGTExportInProgress = false;
    });

    TArray<FString> ActorLabels;
    for (const auto& Item : GTActorListItems)
    {
        if(Item.IsValid()) ActorLabels.Add(*Item);
    }
    
    const FString Timestamp = FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S"));
    const FString BaseDir = OutputDirectory / TEXT("gt_materials") / Timestamp;

    GTMaterialExporter->ExportMaterials(
        ActorLabels,
        GEditor->GetEditorWorldContext().World(),
        BaseDir,
        SceneName,
        GTTextureResolution,
        OnComplete
    );
    
    return FReply::Handled();
}

// ============================================================================
// SECTION 6: TEXENHANCER PIPELINE
// ============================================================================

FReply FVCCSimPanelTexEnhancer::OnBrowseScriptClicked()
{
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (!DesktopPlatform) return FReply::Handled();

    TArray<FString> SelectedFiles;
    void* ParentWindowHandle = const_cast<void*>(FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr));

    if (DesktopPlatform->OpenFileDialog(
        ParentWindowHandle,
        TEXT("Select TexEnhancer Python Script"),
        FPaths::GetPath(TexEnhancerScriptPath),
        TEXT(""),
        TEXT("Python Scripts (*.py)|*.py"),
        EFileDialogFlags::None,
        SelectedFiles))
    {
        if (SelectedFiles.Num() > 0)
        {
            TexEnhancerScriptPath = SelectedFiles[0];
            if (TexEnhancerScriptTextBox.IsValid())
            {
                TexEnhancerScriptTextBox->SetText(FText::FromString(TexEnhancerScriptPath));
            }
            SavePaths();
        }
    }

    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnRunTexEnhancerClicked()
{
    if (TexEnhancerScriptPath.IsEmpty() || !FPaths::FileExists(TexEnhancerScriptPath))
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("TexEnhancer script not found. Please browse to the script."), true);
        return FReply::Handled();
    }

    if (OutputDirectory.IsEmpty())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Output directory is not set."), true);
        return FReply::Handled();
    }

    if (bPipelineInProgress)
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Pipeline is already running."), true);
        return FReply::Handled();
    }

    FString CameraInfoDir  = FPaths::Combine(GetSetACaptureDir(), TEXT("config"));
    FString GTMaterialsPath  = GetGTMaterialsPath();
    FString PipelineOutDir   = FPaths::Combine(OutputDirectory, TEXT("estimated"));

    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.DirectoryExists(*PipelineOutDir))
    {
        PlatformFile.CreateDirectoryTree(*PipelineOutDir);
    }

    FString Args = FString::Printf(
        TEXT("\"%s\" --camera_info_dir \"%s\" --image_dir \"%s\" --output_dir \"%s\" --gt_materials \"%s\""),
        *TexEnhancerScriptPath,
        *CameraInfoDir,
        *FPaths::Combine(GetSetACaptureDir(), TEXT("images")),
        *PipelineOutDir,
        *GTMaterialsPath
    );

    uint32 ProcessId = 0;
    PipelineProcHandle = FPlatformProcess::CreateProc(
        TEXT("python"),
        *Args,
        false, false, false,
        &ProcessId, 0, nullptr, nullptr
    );

    if (PipelineProcHandle.IsValid())
    {
        bPipelineInProgress = true;
        UE_LOG(LogTexEnhancerPanel, Log, TEXT("TexEnhancer pipeline started..."));

        GEditor->GetTimerManager()->SetTimer(StatusTimerHandle, [this]()
        {
            PollPipelineProcess();
        }, 2.f, true);
    }
    else
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Failed to launch TexEnhancer. Check that Python is in PATH."), true);
    }

    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnStopTexEnhancerClicked()
{
    if (PipelineProcHandle.IsValid())
    {
        FPlatformProcess::TerminateProc(PipelineProcHandle, false);
        FPlatformProcess::CloseProc(PipelineProcHandle);
    }

    if (GEditor)
    {
        GEditor->GetTimerManager()->ClearTimer(StatusTimerHandle);
    }

    bPipelineInProgress = false;
    UE_LOG(LogTexEnhancerPanel, Log, TEXT("TexEnhancer pipeline stopped by user."));

    return FReply::Handled();
}

void FVCCSimPanelTexEnhancer::PollPipelineProcess()
{
    if (!PipelineProcHandle.IsValid()) return;

    if (!FPlatformProcess::IsProcRunning(PipelineProcHandle))
    {
        int32 ReturnCode = 0;
        FPlatformProcess::GetProcReturnCode(PipelineProcHandle, &ReturnCode);
        FPlatformProcess::CloseProc(PipelineProcHandle);

        if (GEditor)
        {
            GEditor->GetTimerManager()->ClearTimer(StatusTimerHandle);
        }

        bPipelineInProgress = false;

        FString Msg = ReturnCode == 0
            ? TEXT("TexEnhancer pipeline completed successfully.")
            : FString::Printf(TEXT("TexEnhancer pipeline exited with code %d."), ReturnCode);

        FVCCSimUIHelpers::ShowNotification(Msg, ReturnCode != 0);
        UE_LOG(LogTexEnhancerPanel, Log, TEXT("%s"), *Msg);
    }
}

// ============================================================================
// SECTION 7: EVALUATION
// ============================================================================

FReply FVCCSimPanelTexEnhancer::OnBrowseEstimatedDirClicked()
{
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (!DesktopPlatform) return FReply::Handled();

    FString SelectedDir;
    void* ParentWindowHandle = const_cast<void*>(FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr));

    if (DesktopPlatform->OpenDirectoryDialog(
        ParentWindowHandle,
        TEXT("Select Estimated Materials Directory"),
        EstimatedMaterialsDir,
        SelectedDir))
    {
        EstimatedMaterialsDir = SelectedDir;
        if (EstimatedMaterialsDirTextBox.IsValid())
        {
            EstimatedMaterialsDirTextBox->SetText(FText::FromString(EstimatedMaterialsDir));
        }
        SavePaths();
    }

    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnRunEvaluationClicked()
{
    FString GTPath = GetGTMaterialsPath();
    if (!FPaths::FileExists(GTPath))
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("GT materials file not found. Run GT Export first."), true);
        return FReply::Handled();
    }

    if (EstimatedMaterialsDir.IsEmpty())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Estimated materials directory is not set."), true);
        return FReply::Handled();
    }

    RunBRDFEvaluation();
    return FReply::Handled();
}

void FVCCSimPanelTexEnhancer::RunBRDFEvaluation()
{
    FString GTPath = GetGTMaterialsPath();

    FString GTJsonString;
    if (!FFileHelper::LoadFileToString(GTJsonString, *GTPath))
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Failed to load GT materials JSON."), true);
        return;
    }

    TSharedPtr<FJsonObject> GTRoot;
    TSharedRef<TJsonReader<>> GTReader = TJsonReaderFactory<>::Create(GTJsonString);
    if (!FJsonSerializer::Deserialize(GTReader, GTRoot) || !GTRoot.IsValid())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Failed to parse GT materials JSON."), true);
        return;
    }

    FString EstJsonPath = FPaths::Combine(EstimatedMaterialsDir, TEXT("estimated_materials.json"));
    FString EstJsonString;
    if (!FFileHelper::LoadFileToString(EstJsonString, *EstJsonPath))
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("estimated_materials.json not found in estimated dir."), true);
        return;
    }

    TSharedPtr<FJsonObject> EstRoot;
    TSharedRef<TJsonReader<>> EstReader = TJsonReaderFactory<>::Create(EstJsonString);
    if (!FJsonSerializer::Deserialize(EstReader, EstRoot) || !EstRoot.IsValid())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Failed to parse estimated materials JSON."), true);
        return;
    }

    const TArray<TSharedPtr<FJsonValue>>* GTActors = nullptr;
    const TArray<TSharedPtr<FJsonValue>>* EstActors = nullptr;

    if (!GTRoot->TryGetArrayField(TEXT("actors"), GTActors) ||
        !EstRoot->TryGetArrayField(TEXT("actors"), EstActors))
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Invalid JSON format in materials files."), true);
        return;
    }

    TMap<FString, TSharedPtr<FJsonObject>> GTMap;
    for (auto& V : *GTActors)
    {
        if (V->Type == EJson::Object)
        {
            TSharedPtr<FJsonObject> Obj = V->AsObject();
            FString ActorName;
            Obj->TryGetStringField(TEXT("name"), ActorName);
            GTMap.Add(ActorName, Obj);
        }
    }

    float TotalRoughnessDiff = 0.f;
    float TotalMetallicDiff  = 0.f;
    int32 TotalSlots         = 0;

    for (auto& V : *EstActors)
    {
        if (V->Type != EJson::Object) continue;
        TSharedPtr<FJsonObject> EstActor = V->AsObject();

        FString ActorName;
        EstActor->TryGetStringField(TEXT("name"), ActorName);

        TSharedPtr<FJsonObject>* GTActorPtr = GTMap.Find(ActorName);
        if (!GTActorPtr) continue;

        const TArray<TSharedPtr<FJsonValue>>* EstMats = nullptr;
        const TArray<TSharedPtr<FJsonValue>>* GTMats  = nullptr;

        if (!EstActor->TryGetArrayField(TEXT("materials"), EstMats)) continue;
        if (!(*GTActorPtr)->TryGetArrayField(TEXT("materials"), GTMats)) continue;

        for (int32 i = 0; i < FMath::Min(EstMats->Num(), GTMats->Num()); ++i)
        {
            TSharedPtr<FJsonObject> EstMat = (*EstMats)[i]->AsObject();
            TSharedPtr<FJsonObject> GTMat  = (*GTMats)[i]->AsObject();

            double EstR = 0.0, GTR = 0.0, EstM = 0.0, GTM = 0.0;
            EstMat->TryGetNumberField(TEXT("roughness"), EstR);
            GTMat->TryGetNumberField(TEXT("roughness"), GTR);
            EstMat->TryGetNumberField(TEXT("metallic"),  EstM);
            GTMat->TryGetNumberField(TEXT("metallic"),  GTM);

            TotalRoughnessDiff += FMath::Abs((float)(EstR - GTR));
            TotalMetallicDiff  += FMath::Abs((float)(EstM - GTM));
            ++TotalSlots;
        }
    }

    FString Results;
    if (TotalSlots > 0)
    {
        float MAERoughness = TotalRoughnessDiff / TotalSlots;
        float MAEMetallic  = TotalMetallicDiff  / TotalSlots;
        Results = FString::Printf(
            TEXT("BRDF Evaluation Results\n"
                 "──────────────────────\n"
                 "Compared slots:       %d\n"
                 "Roughness MAE:        %.4f\n"
                 "Metallic  MAE:        %.4f\n"),
            TotalSlots, MAERoughness, MAEMetallic);
    }
    else
    {
        Results = TEXT("No matching actors found between GT and estimated materials.");
    }

    if (EvalResultsTextBlock.IsValid())
    {
        EvalResultsTextBlock->SetText(FText::FromString(Results));
    }
    UE_LOG(LogTexEnhancerPanel, Log, TEXT("BRDF evaluation complete."));

    FString EvalDir = GetEvaluationOutputDir();
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.DirectoryExists(*EvalDir))
    {
        PlatformFile.CreateDirectoryTree(*EvalDir);
    }

    FString CSVPath = FPaths::Combine(EvalDir, TEXT("brdf_accuracy.csv"));
    FString CSVContent = TEXT("metric,value\n");
    if (TotalSlots > 0)
    {
        CSVContent += FString::Printf(TEXT("roughness_mae,%.6f\n"), TotalRoughnessDiff / TotalSlots);
        CSVContent += FString::Printf(TEXT("metallic_mae,%.6f\n"),  TotalMetallicDiff  / TotalSlots);
        CSVContent += FString::Printf(TEXT("compared_slots,%d\n"),  TotalSlots);
    }
    FFileHelper::SaveStringToFile(CSVContent, *CSVPath);

    UE_LOG(LogTexEnhancerPanel, Log, TEXT("Evaluation complete. CSV saved: %s"), *CSVPath);
    FVCCSimUIHelpers::ShowNotification(TEXT("BRDF evaluation complete. Results saved to evaluation directory."), false);
}

// ============================================================================
// UTILITIES
// ============================================================================

FString FVCCSimPanelTexEnhancer::GetSetACaptureDir() const
{
    return FPaths::Combine(OutputDirectory, SceneName, TEXT("capture_setA"));
}

FString FVCCSimPanelTexEnhancer::GetSetBCaptureDir() const
{
    return FPaths::Combine(OutputDirectory, SceneName, TEXT("capture_setB"));
}

FString FVCCSimPanelTexEnhancer::GetGTMaterialsPath() const
{
    return FPaths::Combine(OutputDirectory, TEXT("gt_materials"), TEXT("manifest.json"));
}

FString FVCCSimPanelTexEnhancer::GetEvaluationOutputDir() const
{
    return FPaths::Combine(OutputDirectory, SceneName, TEXT("evaluation"));
}

// ============================================================================
// SECTION 5: NANOBANANA PROJECTION
// ============================================================================

FReply FVCCSimPanelTexEnhancer::OnBrowseNanobananaResultDirClicked()
{
    IDesktopPlatform* DP = FDesktopPlatformModule::Get();
    if (!DP) return FReply::Handled();
    FString Sel;
    void* Handle = const_cast<void*>(FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr));
    if (DP->OpenDirectoryDialog(Handle, TEXT("Select Nanobanana Result Directory"), NanobananaResultDir, Sel))
    {
        NanobananaResultDir = Sel;
        if (NanobananaResultDirTextBox.IsValid())
            NanobananaResultDirTextBox->SetText(FText::FromString(Sel));
        SavePaths();
    }
    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnBrowseNanobananaPosesFileClicked()
{
    IDesktopPlatform* DP = FDesktopPlatformModule::Get();
    if (!DP) return FReply::Handled();
    TArray<FString> Files;
    void* Handle = const_cast<void*>(FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr));
    if (DP->OpenFileDialog(Handle, TEXT("Select Poses File"), FPaths::GetPath(NanobananaPosesFile),
        TEXT(""), TEXT("Text Files (*.txt)|*.txt|All Files (*.*)|*.*"), EFileDialogFlags::None, Files) && Files.Num() > 0)
    {
        NanobananaPosesFile = Files[0];
        if (NanobananaPosesFileTextBox.IsValid())
            NanobananaPosesFileTextBox->SetText(FText::FromString(Files[0]));
        SavePaths();
    }
    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnBrowseNanobananaManifestFileClicked()
{
    IDesktopPlatform* DP = FDesktopPlatformModule::Get();
    if (!DP) return FReply::Handled();
    TArray<FString> Files;
    void* Handle = const_cast<void*>(FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr));
    if (DP->OpenFileDialog(Handle, TEXT("Select manifest.json"), FPaths::GetPath(NanobananaManifestFile),
        TEXT(""), TEXT("JSON Files (*.json)|*.json|All Files (*.*)|*.*"), EFileDialogFlags::None, Files) && Files.Num() > 0)
    {
        NanobananaManifestFile = Files[0];
        if (NanobananaManifestFileTextBox.IsValid())
            NanobananaManifestFileTextBox->SetText(FText::FromString(Files[0]));
        SavePaths();
    }
    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnRunNanobananaProjectionClicked()
{
    if (bNanobananaInProgress)
    {
        UE_LOG(LogTexEnhancerPanel, Warning, TEXT("Nanobanana projection already in progress."));
        return FReply::Handled();
    }
    if (NanobananaResultDir.IsEmpty() || !FPaths::DirectoryExists(NanobananaResultDir))
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Nanobanana result directory not set or missing."), true);
        return FReply::Handled();
    }
    if (NanobananaPosesFile.IsEmpty() || !FPaths::FileExists(NanobananaPosesFile))
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Poses file not set or missing."), true);
        return FReply::Handled();
    }
    if (NanobananaManifestFile.IsEmpty() || !FPaths::FileExists(NanobananaManifestFile))
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("manifest.json not set or missing."), true);
        return FReply::Handled();
    }

    NanobananaManager = MakeShared<FNanobananaManager>();

    FNanobananaManager::FProjectionParams Params;
    Params.ResultDir         = NanobananaResultDir;
    Params.PosesFile         = NanobananaPosesFile;
    Params.ManifestFile      = NanobananaManifestFile;
    Params.HFOV              = NanobananaHFOV;
    Params.ImageWidth        = NanobananaImageWidth;
    Params.ImageHeight       = NanobananaImageHeight;
    Params.OverlayAlpha      = NanobananaOverlayAlpha;
    Params.World             = GEditor->GetEditorWorldContext().World();
    Params.SceneName         = SceneName;
    Params.TextureResolution = GTTextureResolution;

    FOnNanobananaProgress OnProgressDelegate = FOnNanobananaProgress::CreateLambda(
        [this](const FString& Status, int32 Processed, int32 Total)
    {
        if (Total > 0)
        {
            UE_LOG(LogTexEnhancerPanel, Log, TEXT("%s %d / %d"), *Status, Processed, Total);
        }
        else
        {
            UE_LOG(LogTexEnhancerPanel, Log, TEXT("%s"), *Status);
        }
    });

    FOnNanobananaComplete OnCompleteDelegate = FOnNanobananaComplete::CreateLambda(
        [this](const FString& FinalStatus)
    {
        bNanobananaInProgress = false;
        UE_LOG(LogTexEnhancerPanel, Log, TEXT("%s"), *FinalStatus);
        FVCCSimUIHelpers::ShowNotification(TEXT("Nanobanana projection complete!"), false);
    });

    bNanobananaInProgress = true;
    NanobananaManager->RunProjection(Params, OnProgressDelegate, OnCompleteDelegate);

    return FReply::Handled();
}

// ============================================================================
// PATH PERSISTENCE
// ============================================================================

void FVCCSimPanelTexEnhancer::SavePaths()
{
    FVCCSimConfigManager::FTexEnhancerConfig Config;
    Config.OutputDirectory       = OutputDirectory;
    Config.SceneName             = SceneName;
    Config.TexEnhancerScriptPath = TexEnhancerScriptPath;
    Config.EstimatedMaterialsDir = EstimatedMaterialsDir;
    for (const TSharedPtr<FString>& Label : GTActorListItems)
    {
        if (Label.IsValid())
            Config.GTActorLabels.Add(*Label);
    }
    Config.NanobananaResultDir    = NanobananaResultDir;
    Config.NanobananaPosesFile    = NanobananaPosesFile;
    Config.NanobananaManifestFile = NanobananaManifestFile;
    Config.NanobananaHFOV         = NanobananaHFOV;
    Config.NanobananaImageWidth   = NanobananaImageWidth;
    Config.NanobananaImageHeight  = NanobananaImageHeight;
    Config.NanobananaOverlayAlpha = NanobananaOverlayAlpha;
    FVCCSimConfigManager::Get().SetTexEnhancerConfig(Config);
}

void FVCCSimPanelTexEnhancer::LoadPaths()
{
    const auto& Config = FVCCSimConfigManager::Get().GetTexEnhancerConfig();
    if (!Config.OutputDirectory.IsEmpty())       OutputDirectory       = Config.OutputDirectory;
    if (!Config.SceneName.IsEmpty())             SceneName             = Config.SceneName;
    if (!Config.TexEnhancerScriptPath.IsEmpty()) TexEnhancerScriptPath = Config.TexEnhancerScriptPath;
    if (!Config.EstimatedMaterialsDir.IsEmpty()) EstimatedMaterialsDir = Config.EstimatedMaterialsDir;

    GTActorListItems.Empty();
    for (const FString& Label : Config.GTActorLabels)
        GTActorListItems.Add(MakeShareable(new FString(Label)));
    if (GTActorListView.IsValid())
        GTActorListView->RequestListRefresh();

    if (!Config.NanobananaResultDir.IsEmpty())    NanobananaResultDir    = Config.NanobananaResultDir;
    if (!Config.NanobananaPosesFile.IsEmpty())    NanobananaPosesFile    = Config.NanobananaPosesFile;
    if (!Config.NanobananaManifestFile.IsEmpty()) NanobananaManifestFile = Config.NanobananaManifestFile;
    NanobananaHFOV         = Config.NanobananaHFOV;
    NanobananaImageWidth   = Config.NanobananaImageWidth;
    NanobananaImageHeight  = Config.NanobananaImageHeight;
    NanobananaOverlayAlpha = Config.NanobananaOverlayAlpha;

    NanobananaHFOVValue         = NanobananaHFOV;
    NanobananaImageWidthValue   = NanobananaImageWidth;
    NanobananaImageHeightValue  = NanobananaImageHeight;
    NanobananaOverlayAlphaValue = NanobananaOverlayAlpha;
}