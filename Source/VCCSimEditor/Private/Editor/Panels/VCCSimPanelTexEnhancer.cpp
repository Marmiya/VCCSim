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
#include "Utils/ConfigParser.h"
#include "Utils/LightingManager.h"
#include "Utils/GTMaterialExporter.h"
#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Editor/Panels/VCCSimPanelPathImageCapture.h"
#include "HAL/FileManager.h"

#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "HAL/PlatformFilemanager.h"
#include "HAL/PlatformProcess.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/PlatformFileManager.h"
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
    LightingManager = MakeShared<FLightingManager>();
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

void FVCCSimPanelTexEnhancer::SetPathImageCaptureManager(
    TSharedPtr<FVCCSimPanelPathImageCapture> InPathImageCaptureManager)
{
    PathImageCaptureManager = InPathImageCaptureManager;
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

    bOutputImages = Config.bOutputImages;
    bOutputMesh   = Config.bOutputMesh;

    LoadParamsFromConfig();
}

void FVCCSimPanelTexEnhancer::SaveToConfigManager()
{
    SavePaths();
}

void FVCCSimPanelTexEnhancer::LoadParamsFromConfig()
{
    const auto& Config = FVCCSimConfigManager::Get().GetTexEnhancerConfig();

    GTTextureResolution = Config.GTTextureResolution;
    DayCycleSpeed       = Config.DayCycleSpeed;

    for (int32 i = 0; i < MaxLightingEntries; ++i)
    {
        if (Config.SetAElevation.IsValidIndex(i)) SetAElevation[i] = Config.SetAElevation[i];
        if (Config.SetAAzimuth.IsValidIndex(i))   SetAAzimuth[i]   = Config.SetAAzimuth[i];
        if (Config.SetBElevation.IsValidIndex(i)) SetBElevation[i] = Config.SetBElevation[i];
        if (Config.SetBAzimuth.IsValidIndex(i))   SetBAzimuth[i]   = Config.SetBAzimuth[i];

        SetAElevationValue[i] = SetAElevation[i];
        SetAAzimuthValue[i]   = SetAAzimuth[i];
        SetBElevationValue[i] = SetBElevation[i];
        SetBAzimuthValue[i]   = SetBAzimuth[i];
    }

    SunCalcLatitude  = Config.SunCalcLatitude;
    SunCalcLongitude = Config.SunCalcLongitude;
    SunCalcTimeZone  = Config.SunCalcTimeZone;
    SunCalcYear      = Config.SunCalcYear;
    SunCalcMonth     = Config.SunCalcMonth;
    SunCalcDay       = Config.SunCalcDay;
    SunCalcHour      = Config.SunCalcHour;
    SunCalcMinute    = Config.SunCalcMinute;
    SunCalcFillSlot  = Config.SunCalcFillSlot;

    GTTexResValue        = GTTextureResolution;
    DayCycleSpeedValue   = DayCycleSpeed;
    SunCalcLatValue      = SunCalcLatitude;
    SunCalcLonValue      = SunCalcLongitude;
    SunCalcTZValue       = SunCalcTimeZone;
    SunCalcYearValue     = SunCalcYear;
    SunCalcMonthValue    = SunCalcMonth;
    SunCalcDayValue      = SunCalcDay;
    SunCalcHourValue     = SunCalcHour;
    SunCalcMinuteValue   = SunCalcMinute;
    SunCalcFillSlotValue = SunCalcFillSlot;
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
    UE_LOG(LogTexEnhancerPanel, Log, TEXT("Set-A%d applied: Elev=%.1f Az=%.1f"),
        Index + 1, SetAElevation[Index], SetAAzimuth[Index]);
    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnApplySetBLightingClicked(int32 Index)
{
    if (Index < 0 || Index >= NumLightingSetB || !LightingManager.IsValid()) return FReply::Handled();
    LightingManager->ApplyLightingCondition(SetBElevation[Index], SetBAzimuth[Index]);
    UE_LOG(LogTexEnhancerPanel, Log, TEXT("Set-B%d applied: Elev=%.1f Az=%.1f"),
        Index + 1, SetBElevation[Index], SetBAzimuth[Index]);
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

    UE_LOG(LogTexEnhancerPanel, Log, TEXT("Sun position calculated & applied: Elev=%.1f Az=%.1f"),
        SunCalcElevation, SunCalcAzimuth);

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

FString FVCCSimPanelTexEnhancer::FindLatestCaptureDirectory() const
{
    const FString Root = GetDatasetCapturesRoot();
    TArray<FString> CaptureDirs;
    IFileManager::Get().FindFiles(CaptureDirs, *(Root / TEXT("capture_*")), false, true);
    if (CaptureDirs.IsEmpty())
    {
        return FString();
    }
    CaptureDirs.Sort();
    return Root / CaptureDirs.Last();
}

FReply FVCCSimPanelTexEnhancer::OnExportGTMaterialsClicked()
{
    const FString CaptureDir = FindLatestCaptureDirectory();
    if (CaptureDir.IsEmpty())
    {
        FVCCSimUIHelpers::ShowNotification(
            TEXT("No capture found. Run 'Capture Dataset' first; GT materials are stored inside the capture directory."), true);
        return FReply::Handled();
    }
    StartGTMaterialExport(CaptureDir / TEXT("gt_materials"));
    return FReply::Handled();
}

bool FVCCSimPanelTexEnhancer::StartGTMaterialExport(const FString& BaseDir)
{
    if (bGTExportInProgress)
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("GT material export already in progress."), true);
        return false;
    }
    if (OutputDirectory.IsEmpty())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Output directory is not set."), true);
        return false;
    }
    TSharedPtr<FVCCSimPanelSelection> Sel = SelectionManager.Pin();
    if (!Sel.IsValid())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Object Selection panel is not available."), true);
        return false;
    }
    const TArray<FString> ActorLabels = Sel->GetEnabledTargetActorLabels();
    if (ActorLabels.IsEmpty())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("No enabled target actors. Add and check actors in the Object Selection panel."), true);
        return false;
    }
    UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
    if (!World)
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("No editor world available."), true);
        return false;
    }
    if (Sel->IsGTExportInProgress())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("GT material export already in progress."), true);
        return false;
    }

    // GT materials are lighting-independent: if a previous capture of this scene exported
    // the same target set / transforms / materials, copy it instead of re-running the
    // (slow) glTF export.
    const FString Signature = FGTMaterialExporter::ComputeSignature(
        World, ActorLabels, SceneName, GTTextureResolution);

    {
        const FString CapturesRoot = GetDatasetCapturesRoot();
        const FString CurrentCaptureDir = FPaths::GetCleanFilename(FPaths::GetPath(BaseDir));
        const FString Reusable = FGTMaterialExporter::FindReusableExport(
            CapturesRoot, CurrentCaptureDir, Signature);
        if (!Reusable.IsEmpty())
        {
            if (FPlatformFileManager::Get().GetPlatformFile().CopyDirectoryTree(*BaseDir, *Reusable, true))
            {
                UE_LOG(LogTexEnhancerPanel, Log,
                    TEXT("Reused GT materials from %s (signature match); skipped export"), *Reusable);
                FVCCSimUIHelpers::ShowNotification(
                    TEXT("GT materials reused from a previous capture (unchanged)."));
                return true;
            }
            UE_LOG(LogTexEnhancerPanel, Warning,
                TEXT("Failed to copy reusable GT materials from %s; exporting fresh"), *Reusable);
        }
    }

    bGTExportInProgress = true;
    FVCCSimUIHelpers::ShowNotification(TEXT("Starting GT material export..."));

    FSimpleDelegate OnComplete = FSimpleDelegate::CreateLambda([this]()
    {
        bGTExportInProgress = false;
    });

    Sel->RunGTMeshExport(BaseDir, SceneName, GTTextureResolution, Signature, OnComplete);
    return true;
}

// ============================================================================
// SECTION 3: DATASET CAPTURE
// ============================================================================

FString FVCCSimPanelTexEnhancer::GetDatasetCapturesRoot() const
{
    return FPaths::Combine(OutputDirectory, SceneName, TEXT("captures"));
}

FString FVCCSimPanelTexEnhancer::MakeNextCaptureDirectory() const
{
    const FString Root = GetDatasetCapturesRoot();
    const FString Timestamp = FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S"));
    IFileManager& FileManager = IFileManager::Get();

    FString Candidate = Root / FString::Printf(TEXT("capture_%s"), *Timestamp);
    for (int32 Suffix = 2; FileManager.DirectoryExists(*Candidate) && Suffix < 100; ++Suffix)
    {
        Candidate = Root / FString::Printf(TEXT("capture_%s_%d"), *Timestamp, Suffix);
    }
    return FileManager.DirectoryExists(*Candidate) ? FString() : Candidate;
}

FReply FVCCSimPanelTexEnhancer::OnCaptureDatasetClicked()
{
    if (bDatasetCaptureInProgress)
    {
        if (TSharedPtr<FVCCSimPanelPathImageCapture> PathCapture = PathImageCaptureManager.Pin())
        {
            PathCapture->StopAutoCapture();
        }
        return FReply::Handled();
    }
    if (OutputDirectory.IsEmpty())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Output directory is not set."), true);
        return FReply::Handled();
    }
    if (!bOutputImages && !bOutputMesh)
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Enable Photos and/or Mesh output first."), true);
        return FReply::Handled();
    }

    // Mesh-only run: skip the image capture session entirely and export gt_materials into a fresh
    // capture directory (reuse/copy from a sibling capture still applies via the signature check).
    if (!bOutputImages)
    {
        const FString MeshOnlyDir = MakeNextCaptureDirectory();
        if (MeshOnlyDir.IsEmpty())
        {
            FVCCSimUIHelpers::ShowNotification(TEXT("Could not allocate a capture directory."), true);
            return FReply::Handled();
        }
        if (!StartGTMaterialExport(MeshOnlyDir / TEXT("gt_materials")))
        {
            UE_LOG(LogTexEnhancerPanel, Warning, TEXT("Mesh-only gt_materials export could not start"));
        }
        return FReply::Handled();
    }

    TSharedPtr<FVCCSimPanelPathImageCapture> PathCapture = PathImageCaptureManager.Pin();
    if (!PathCapture.IsValid())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("PathImageCapture panel is not available."), true);
        return FReply::Handled();
    }

    FString Reason;
    if (!PathCapture->CanRunDatasetCapture(Reason))
    {
        FVCCSimUIHelpers::ShowNotification(Reason, true);
        return FReply::Handled();
    }

    const FString CaptureDir = MakeNextCaptureDirectory();
    if (CaptureDir.IsEmpty())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Could not allocate a capture directory."), true);
        return FReply::Handled();
    }

    TWeakPtr<FVCCSimPanelTexEnhancer> WeakSelf = AsShared();
    const bool bStarted = PathCapture->StartCaptureSession(
        CaptureDir,
        true,
        FOnCaptureSessionComplete::CreateLambda(
            [WeakSelf, CaptureDir](bool bSuccess)
            {
                if (TSharedPtr<FVCCSimPanelTexEnhancer> Pinned = WeakSelf.Pin())
                {
                    Pinned->OnDatasetCaptureFinished(bSuccess, CaptureDir);
                }
            }));

    if (!bStarted)
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Failed to start dataset capture."), true);
        return FReply::Handled();
    }

    bDatasetCaptureInProgress = true;
    UE_LOG(LogTexEnhancerPanel, Log, TEXT("Dataset capture started: %s"), *CaptureDir);
    return FReply::Handled();
}

void FVCCSimPanelTexEnhancer::OnDatasetCaptureFinished(bool bSuccess, FString CaptureDirectory)
{
    bDatasetCaptureInProgress = false;

    if (!bSuccess)
    {
        if (IFileManager::Get().DeleteDirectory(*CaptureDirectory, false, true))
        {
            UE_LOG(LogTexEnhancerPanel, Warning,
                TEXT("Dataset capture cancelled or failed; partial directory removed: %s"), *CaptureDirectory);
        }
        else
        {
            UE_LOG(LogTexEnhancerPanel, Warning,
                TEXT("Dataset capture cancelled or failed; could not remove partial directory: %s"), *CaptureDirectory);
        }
        FVCCSimUIHelpers::ShowNotification(TEXT("Dataset capture did not complete."), true);
        return;
    }

    UE_LOG(LogTexEnhancerPanel, Log, TEXT("Dataset capture complete: %s"), *CaptureDirectory);
    FVCCSimUIHelpers::ShowNotification(
        FString::Printf(TEXT("Dataset capture complete: %s"), *CaptureDirectory), false);

    if (!bOutputMesh)
    {
        UE_LOG(LogTexEnhancerPanel, Log, TEXT("Mesh output disabled, gt_materials export skipped"));
        return;
    }

    TSharedPtr<FVCCSimPanelSelection> Sel = SelectionManager.Pin();
    if (!Sel.IsValid() || !Sel->HasEnabledTargetActors())
    {
        UE_LOG(LogTexEnhancerPanel, Log, TEXT("No enabled target actors, gt_materials export skipped"));
        return;
    }

    if (!StartGTMaterialExport(CaptureDirectory / TEXT("gt_materials")))
    {
        UE_LOG(LogTexEnhancerPanel, Warning, TEXT("gt_materials export could not start"));
    }
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

    UE_LOG(LogTexEnhancerPanel, Log, TEXT("%s"), *Results);

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
    const FString CaptureDir = FindLatestCaptureDirectory();
    return CaptureDir.IsEmpty()
        ? FString()
        : CaptureDir / TEXT("gt_materials") / TEXT("manifest.json");
}

FString FVCCSimPanelTexEnhancer::GetEvaluationOutputDir() const
{
    return FPaths::Combine(OutputDirectory, SceneName, TEXT("evaluation"));
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

    Config.GTTextureResolution = GTTextureResolution;
    Config.DayCycleSpeed       = DayCycleSpeed;
    Config.bOutputImages       = bOutputImages;
    Config.bOutputMesh         = bOutputMesh;

    Config.SetAElevation.Append(SetAElevation, MaxLightingEntries);
    Config.SetAAzimuth.Append(SetAAzimuth, MaxLightingEntries);
    Config.SetBElevation.Append(SetBElevation, MaxLightingEntries);
    Config.SetBAzimuth.Append(SetBAzimuth, MaxLightingEntries);

    Config.SunCalcLatitude  = SunCalcLatitude;
    Config.SunCalcLongitude = SunCalcLongitude;
    Config.SunCalcTimeZone  = SunCalcTimeZone;
    Config.SunCalcYear      = SunCalcYear;
    Config.SunCalcMonth     = SunCalcMonth;
    Config.SunCalcDay       = SunCalcDay;
    Config.SunCalcHour      = SunCalcHour;
    Config.SunCalcMinute    = SunCalcMinute;
    Config.SunCalcFillSlot  = SunCalcFillSlot;

    FVCCSimConfigManager::Get().SetTexEnhancerConfig(Config);
}

void FVCCSimPanelTexEnhancer::LoadPaths()
{
    const auto& Config = FVCCSimConfigManager::Get().GetTexEnhancerConfig();
    if (!Config.OutputDirectory.IsEmpty())       OutputDirectory       = Config.OutputDirectory;
    if (OutputDirectory.IsEmpty())               OutputDirectory       = GetVCCSimOutputRoot() / TEXT("TexEnhancer");
    if (!Config.SceneName.IsEmpty())             SceneName             = Config.SceneName;
    if (!Config.TexEnhancerScriptPath.IsEmpty()) TexEnhancerScriptPath = Config.TexEnhancerScriptPath;
    if (!Config.EstimatedMaterialsDir.IsEmpty()) EstimatedMaterialsDir = Config.EstimatedMaterialsDir;
    bOutputImages = Config.bOutputImages;
    bOutputMesh   = Config.bOutputMesh;

    LoadParamsFromConfig();
}