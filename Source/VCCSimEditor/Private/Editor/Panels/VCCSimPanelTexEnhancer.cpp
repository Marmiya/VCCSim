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
#include "Utils/CaptureReuseManifest.h"
#include "Utils/CaptureSessionCheckpoint.h"
#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Editor/Panels/VCCSimPanelPathImageCapture.h"
#include "Pawns/FlashPawn.h"
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
    for (int32 i = 0; i < NumLightingConditions; ++i)
    {
        LightingElevationValue[i] = LightingElevation[i];
        LightingAzimuthValue[i]   = LightingAzimuth[i];
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
    bUseCaptureReuse = Config.bUseCaptureReuse;

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

    for (int32 i = 0; i < NumLightingConditions; ++i)
    {
        if (Config.LightingElevation.IsValidIndex(i)) LightingElevation[i] = Config.LightingElevation[i];
        if (Config.LightingAzimuth.IsValidIndex(i))   LightingAzimuth[i]   = Config.LightingAzimuth[i];
        if (Config.LightingSelected.IsValidIndex(i))  bLightingSelected[i] = Config.LightingSelected[i];

        LightingElevationValue[i] = LightingElevation[i];
        LightingAzimuthValue[i]   = LightingAzimuth[i];
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

FReply FVCCSimPanelTexEnhancer::OnApplyLightingClicked(int32 Index)
{
    if (Index < 0 || Index >= NumLightingConditions || !LightingManager.IsValid()) return FReply::Handled();
    LightingManager->ApplyLightingCondition(LightingElevation[Index], LightingAzimuth[Index]);
    UE_LOG(LogTexEnhancerPanel, Log, TEXT("Lighting condition %d applied: Elev=%.1f Az=%.1f"),
        Index + 1, LightingElevation[Index], LightingAzimuth[Index]);
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

FReply FVCCSimPanelTexEnhancer::OnFillFromSunPositionClicked()
{
    int32 SlotIdx = FMath::Clamp(SunCalcFillSlot - 1, 0, NumLightingConditions - 1);
    LightingElevation[SlotIdx]      = SunCalcElevation;
    LightingAzimuth[SlotIdx]        = SunCalcAzimuth;
    LightingElevationValue[SlotIdx] = SunCalcElevation;
    LightingAzimuthValue[SlotIdx]   = SunCalcAzimuth;

    if (LightingElevationSpinBox[SlotIdx].IsValid())
    {
        LightingElevationSpinBox[SlotIdx]->Invalidate(EInvalidateWidgetReason::Paint);
    }
    if (LightingAzimuthSpinBox[SlotIdx].IsValid())
    {
        LightingAzimuthSpinBox[SlotIdx]->Invalidate(EInvalidateWidgetReason::Paint);
    }

    UE_LOG(LogTexEnhancerPanel, Log,
        TEXT("Sun position filled into slot %d: Elevation=%.1f°  Azimuth=%.1f°"),
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

    // gt_materials reuse is decided upstream via captures/reuse.json (DecideAndStartCapture):
    // when reusable, the export is skipped and an owner reference is recorded instead. Reaching
    // here means this capture is the owner, so always run the full export.
    const FString Signature = FGTMaterialExporter::ComputeSignature(
        World, ActorLabels, SceneName, GTTextureResolution);

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
    // capture directory. gt_materials reuse is recorded in captures/reuse.json (owner reference).
    if (!bOutputImages)
    {
        const FString MeshOnlyDir = MakeNextCaptureDirectory();
        if (MeshOnlyDir.IsEmpty())
        {
            FVCCSimUIHelpers::ShowNotification(TEXT("Could not allocate a capture directory."), true);
            return FReply::Handled();
        }

        UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
        TSharedPtr<FVCCSimPanelSelection> Sel = SelectionManager.Pin();
        FString GtMatKey;
        if (Sel.IsValid() && World)
        {
            const TArray<FString> Labels = Sel->GetEnabledTargetActorLabels();
            if (Labels.Num() > 0)
            {
                GtMatKey = FGTMaterialExporter::ComputeSignature(World, Labels, SceneName, GTTextureResolution);
            }
        }

        FCaptureReuseManifest Manifest = FCaptureReuseManifest::Load(GetDatasetCapturesRoot());
        FCaptureReuseEntry Entry;
        Entry.SceneKey = FGTMaterialExporter::ComputeSceneSignature(World);
        Entry.GtMaterialsKey = GtMatKey;
        Entry.GtMaterialsOwner = (bUseCaptureReuse && !GtMatKey.IsEmpty())
            ? Manifest.FindGtMaterialsOwner(GtMatKey) : FString();

        if (!Entry.GtMaterialsOwner.IsEmpty())
        {
            UE_LOG(LogTexEnhancerPanel, Log,
                TEXT("Mesh-only: gt_materials reused from %s (manifest reference)"), *Entry.GtMaterialsOwner);
        }
        else if (!StartGTMaterialExport(MeshOnlyDir / TEXT("gt_materials")))
        {
            UE_LOG(LogTexEnhancerPanel, Warning, TEXT("Mesh-only gt_materials export could not start"));
        }

        Manifest.AddOrUpdate(FPaths::GetCleanFilename(MeshOnlyDir), Entry);
        Manifest.Save();
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

    LightingCaptureQueue.Reset();
    for (int32 i = 0; i < NumLightingConditions; ++i)
    {
        if (bLightingSelected[i]) LightingCaptureQueue.Add(i);
    }
    if (LightingCaptureQueue.Num() > 0)
    {
        BatchCaptureTimestamp = FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S"));
        bBatchCapture = true;
        bDatasetCaptureInProgress = true;

        // Resume checkpoint: record every planned lighting window up front so an interrupted run
        // (Stop or editor crash) can be continued from <captures>/capture_session.json.
        {
            UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
            ActiveCheckpoint = FCaptureSessionCheckpoint();
            ActiveCheckpoint.CapturesRoot       = GetDatasetCapturesRoot();
            ActiveCheckpoint.BatchTimestamp     = BatchCaptureTimestamp;
            ActiveCheckpoint.PoseKey            = PathCapture->ComputePathPoseKey();
            ActiveCheckpoint.SceneKey           = FGTMaterialExporter::ComputeSceneSignature(World);
            ActiveCheckpoint.bOutputMesh        = bOutputMesh;
            ActiveCheckpoint.GTTextureResolution= GTTextureResolution;
            ActiveCheckpoint.bUseCaptureReuse   = bUseCaptureReuse;
            ActiveCheckpoint.SceneName          = SceneName;
            if (TSharedPtr<FVCCSimPanelSelection> Sel = SelectionManager.Pin())
            {
                ActiveCheckpoint.TargetLabels = Sel->GetEnabledTargetActorLabels();
                if (AFlashPawn* Pawn = Sel->GetSelectedFlashPawn().Get())
                {
                    Pawn->GetCurrentPath(ActiveCheckpoint.PathPositions, ActiveCheckpoint.PathRotations);
                }
            }
            for (int32 Slot : LightingCaptureQueue)
            {
                FCaptureWindow W;
                W.Slot      = Slot;
                W.Elevation = LightingElevation[Slot];
                W.Azimuth   = LightingAzimuth[Slot];
                W.DirName   = FString::Printf(TEXT("capture_%s_L%d"), *BatchCaptureTimestamp, Slot + 1);
                ActiveCheckpoint.Windows.Add(W);
            }
            ActiveCheckpoint.Save();
        }

        UE_LOG(LogTexEnhancerPanel, Log,
            TEXT("Batch dataset capture: %d selected lighting conditions"), LightingCaptureQueue.Num());
        StartNextBatchCapture();
        return FReply::Handled();
    }

    const FString CaptureDir = MakeNextCaptureDirectory();
    if (CaptureDir.IsEmpty())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Could not allocate a capture directory."), true);
        return FReply::Handled();
    }

    // Resume checkpoint for the single (no-lighting) capture: one window, Slot -1 (no lighting to
    // re-apply on resume — it captures under whatever lighting is currently in the level).
    {
        UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
        ActiveCheckpoint = FCaptureSessionCheckpoint();
        ActiveCheckpoint.CapturesRoot       = GetDatasetCapturesRoot();
        ActiveCheckpoint.BatchTimestamp.Empty();
        ActiveCheckpoint.PoseKey            = PathCapture->ComputePathPoseKey();
        ActiveCheckpoint.SceneKey           = FGTMaterialExporter::ComputeSceneSignature(World);
        ActiveCheckpoint.bOutputMesh        = bOutputMesh;
        ActiveCheckpoint.GTTextureResolution= GTTextureResolution;
        ActiveCheckpoint.bUseCaptureReuse   = bUseCaptureReuse;
        ActiveCheckpoint.SceneName          = SceneName;
        if (TSharedPtr<FVCCSimPanelSelection> Sel = SelectionManager.Pin())
        {
            ActiveCheckpoint.TargetLabels = Sel->GetEnabledTargetActorLabels();
            if (AFlashPawn* Pawn = Sel->GetSelectedFlashPawn().Get())
            {
                Pawn->GetCurrentPath(ActiveCheckpoint.PathPositions, ActiveCheckpoint.PathRotations);
            }
        }
        FCaptureWindow W;
        W.Slot    = -1;
        W.DirName = FPaths::GetCleanFilename(CaptureDir);
        ActiveCheckpoint.Windows.Add(W);
        ActiveCheckpoint.Save();
    }

    bDatasetCaptureInProgress = true;

    if (!DecideAndStartCapture(CaptureDir))
    {
        bDatasetCaptureInProgress = false;
        FCaptureSessionCheckpoint::Clear(GetDatasetCapturesRoot());
        FVCCSimUIHelpers::ShowNotification(TEXT("Failed to start dataset capture."), true);
        return FReply::Handled();
    }

    return FReply::Handled();
}

bool FVCCSimPanelTexEnhancer::DecideAndStartCapture(const FString& CaptureDir)
{
    TSharedPtr<FVCCSimPanelPathImageCapture> PathCapture = PathImageCaptureManager.Pin();
    if (!PathCapture.IsValid())
    {
        return false;
    }

    UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
    TSharedPtr<FVCCSimPanelSelection> Sel = SelectionManager.Pin();

    const FString PoseKey  = PathCapture->ComputePathPoseKey();
    const FString SceneKey = FGTMaterialExporter::ComputeSceneSignature(World);
    FString GtMatKey;
    if (bOutputMesh && Sel.IsValid() && World)
    {
        const TArray<FString> Labels = Sel->GetEnabledTargetActorLabels();
        if (Labels.Num() > 0)
        {
            GtMatKey = FGTMaterialExporter::ComputeSignature(World, Labels, SceneName, GTTextureResolution);
        }
    }

    FCaptureReuseManifest Manifest = FCaptureReuseManifest::Load(GetDatasetCapturesRoot());
    const FString ViewGtOwner = bUseCaptureReuse ? Manifest.FindViewGtOwner(PoseKey, SceneKey) : FString();
    const FString GtMatOwner  = (bUseCaptureReuse && !GtMatKey.IsEmpty())
        ? Manifest.FindGtMaterialsOwner(GtMatKey) : FString();

    PendingCaptureName = FPaths::GetCleanFilename(CaptureDir);
    PendingReuseEntry = FCaptureReuseEntry();
    PendingReuseEntry.PoseKey          = PoseKey;
    PendingReuseEntry.SceneKey         = SceneKey;
    PendingReuseEntry.GtMaterialsKey   = GtMatKey;
    PendingReuseEntry.ViewGtOwner      = ViewGtOwner;
    PendingReuseEntry.GtMaterialsOwner = GtMatOwner;

    const bool bRgbOnly = !ViewGtOwner.IsEmpty();

    TWeakPtr<FVCCSimPanelTexEnhancer> WeakSelf = AsShared();
    const bool bStarted = PathCapture->StartCaptureSession(
        CaptureDir,
        /*bDatasetChannelsOnly*/ true,
        bRgbOnly,
        FOnCaptureSessionComplete::CreateLambda(
            [WeakSelf, CaptureDir](bool bSuccess)
            {
                if (TSharedPtr<FVCCSimPanelTexEnhancer> Pinned = WeakSelf.Pin())
                {
                    Pinned->OnDatasetCaptureFinished(bSuccess, CaptureDir);
                }
            }));

    if (bStarted)
    {
        // Record the resolved channel mode for this window so resume scans it with the right channel
        // set (RGB-only windows expect only RGB files; full windows expect the GT channels too).
        ActiveCheckpoint.SetWindowRgbOnly(FPaths::GetCleanFilename(CaptureDir), bRgbOnly);
        if (!ActiveCheckpoint.CapturesRoot.IsEmpty())
        {
            ActiveCheckpoint.Save();
        }

        if (bRgbOnly)
        {
            UE_LOG(LogTexEnhancerPanel, Log,
                TEXT("Capture %s: RGB-only (GT image channels reused from %s)"),
                *PendingCaptureName, *ViewGtOwner);
        }
        else
        {
            UE_LOG(LogTexEnhancerPanel, Log,
                TEXT("Capture %s: full GT capture (owner)"), *PendingCaptureName);
        }
    }
    return bStarted;
}

void FVCCSimPanelTexEnhancer::StartNextBatchCapture()
{
    if (LightingCaptureQueue.Num() == 0)
    {
        bBatchCapture = false;
        bDatasetCaptureInProgress = false;
        return;
    }

    const int32 Slot = LightingCaptureQueue[0];
    LightingCaptureQueue.RemoveAt(0);

    // Apply the lighting recorded for this slot in the active checkpoint, so a resumed run reproduces
    // the exact lighting the window was started with even if the panel's slot values changed since.
    // During a fresh run the checkpoint mirrors the panel, so this is equivalent. Falls back to panel.
    float Elev = LightingElevation[Slot];
    float Az   = LightingAzimuth[Slot];
    for (const FCaptureWindow& W : ActiveCheckpoint.Windows)
    {
        if (W.Slot == Slot) { Elev = W.Elevation; Az = W.Azimuth; break; }
    }
    if (LightingManager.IsValid())
    {
        LightingManager->ApplyLightingCondition(Elev, Az);
    }

    const FString CaptureDir = GetDatasetCapturesRoot()
        / FString::Printf(TEXT("capture_%s_L%d"), *BatchCaptureTimestamp, Slot + 1);

    if (!DecideAndStartCapture(CaptureDir))
    {
        bBatchCapture = false;
        bDatasetCaptureInProgress = false;
        LightingCaptureQueue.Reset();
        FVCCSimUIHelpers::ShowNotification(TEXT("Failed to start dataset capture."), true);
        return;
    }

    UE_LOG(LogTexEnhancerPanel, Log,
        TEXT("Batch capture (lighting %d, Elev=%.1f Az=%.1f): %s"),
        Slot + 1, LightingElevation[Slot], LightingAzimuth[Slot], *CaptureDir);
}

void FVCCSimPanelTexEnhancer::OnDatasetCaptureFinished(bool bSuccess, FString CaptureDirectory)
{
    if (!bSuccess)
    {
        // Keep the partial output AND the resume checkpoint on disk so the run can be continued via the
        // Resume button (this is the whole point — a large dataset interrupted by Stop or a crash must
        // not lose what it already captured). Only the live (in-memory) run state is reset here.
        UE_LOG(LogTexEnhancerPanel, Warning,
            TEXT("Dataset capture stopped or failed; partial output kept for resume: %s"), *CaptureDirectory);
        LightingCaptureQueue.Reset();
        bBatchCapture = false;
        bDatasetCaptureInProgress = false;
        return;
    }

    UE_LOG(LogTexEnhancerPanel, Log, TEXT("Dataset capture complete: %s"), *CaptureDirectory);
    FVCCSimUIHelpers::ShowNotification(
        FString::Printf(TEXT("Dataset capture complete: %s"), *CaptureDirectory), false);

    TSharedPtr<FVCCSimPanelSelection> Sel = SelectionManager.Pin();
    if (!PendingReuseEntry.GtMaterialsOwner.IsEmpty())
    {
        UE_LOG(LogTexEnhancerPanel, Log,
            TEXT("gt_materials reused from %s (manifest reference); export skipped"),
            *PendingReuseEntry.GtMaterialsOwner);
    }
    else if (!bOutputMesh)
    {
        UE_LOG(LogTexEnhancerPanel, Log, TEXT("Mesh output disabled, gt_materials export skipped"));
    }
    else if (!Sel.IsValid() || !Sel->HasEnabledTargetActors())
    {
        UE_LOG(LogTexEnhancerPanel, Log, TEXT("No enabled target actors, gt_materials export skipped"));
    }
    else if (!StartGTMaterialExport(CaptureDirectory / TEXT("gt_materials")))
    {
        UE_LOG(LogTexEnhancerPanel, Warning, TEXT("gt_materials export could not start"));
    }

    // Record this capture's reuse relationship (owner refs) for the Python resolve step.
    if (!PendingCaptureName.IsEmpty()
        && FPaths::GetCleanFilename(CaptureDirectory) == PendingCaptureName)
    {
        FCaptureReuseManifest Manifest = FCaptureReuseManifest::Load(GetDatasetCapturesRoot());
        Manifest.AddOrUpdate(PendingCaptureName, PendingReuseEntry);
        Manifest.Save();
    }

    if (bBatchCapture)
    {
        if (LightingCaptureQueue.Num() > 0)
        {
            StartNextBatchCapture();
            return;   // more windows to go — keep the resume checkpoint
        }
        bBatchCapture = false;
        FVCCSimUIHelpers::ShowNotification(
            TEXT("Batch dataset capture complete (all selected lighting)."), false);
    }

    // Whole run finished successfully — drop the resume checkpoint so the Resume button goes inactive.
    FCaptureSessionCheckpoint::Clear(GetDatasetCapturesRoot());
    ActiveCheckpoint = FCaptureSessionCheckpoint();
    bDatasetCaptureInProgress = false;
}

bool FVCCSimPanelTexEnhancer::HasResumableCapture() const
{
    return !bDatasetCaptureInProgress
        && FCaptureSessionCheckpoint::Exists(GetDatasetCapturesRoot());
}

FReply FVCCSimPanelTexEnhancer::OnResumeCaptureClicked()
{
    if (bDatasetCaptureInProgress)
    {
        return FReply::Handled();
    }

    const FString Root = GetDatasetCapturesRoot();
    FCaptureSessionCheckpoint Cp = FCaptureSessionCheckpoint::Load(Root);
    if (!Cp.IsValid())
    {
        UE_LOG(LogTexEnhancerPanel, Warning, TEXT("Resume: no resumable capture found in %s"), *Root);
        return FReply::Handled();
    }

    TSharedPtr<FVCCSimPanelPathImageCapture> PathCapture = PathImageCaptureManager.Pin();
    if (!PathCapture.IsValid())
    {
        UE_LOG(LogTexEnhancerPanel, Warning, TEXT("Resume: PathImageCapture panel is not available."));
        return FReply::Handled();
    }

    TSharedPtr<FVCCSimPanelSelection> Sel = SelectionManager.Pin();
    AFlashPawn* Pawn = Sel.IsValid() ? Sel->GetSelectedFlashPawn().Get() : nullptr;
    if (!Pawn)
    {
        UE_LOG(LogTexEnhancerPanel, Warning, TEXT("Resume: no FlashPawn selected to drive the capture."));
        return FReply::Handled();
    }

    // Restore the recorded path onto the FlashPawn so resume is self-contained even after a crash where
    // the level (and the pawn's in-memory path) was never saved. The path is the capture target; with it
    // restored, the pose key matches by construction. (Scene changes are tolerated — only the pose path
    // is validated; if it still does not match after restore, refuse rather than mix mismatched images.)
    if (Cp.PathPositions.Num() > 0 && Cp.PathPositions.Num() == Cp.PathRotations.Num())
    {
        Pawn->SetPathPanel(Cp.PathPositions, Cp.PathRotations);
        Pawn->MoveTo(0);
        UE_LOG(LogTexEnhancerPanel, Log,
            TEXT("Resume: restored FlashPawn path from checkpoint (%d poses)."), Cp.PathPositions.Num());
    }

    const FString CurPoseKey = PathCapture->ComputePathPoseKey();
    if (!Cp.PoseKey.IsEmpty() && CurPoseKey != Cp.PoseKey)
    {
        UE_LOG(LogTexEnhancerPanel, Warning,
            TEXT("Resume refused: the FlashPawn path differs from the interrupted capture and could not "
                 "be restored. Load or regenerate the original path, then Resume."));
        return FReply::Handled();
    }

    // Log (do not refuse) if the enabled target actors differ from the recorded task — gt_materials
    // export uses the current selection, so this is worth surfacing.
    if (Sel.IsValid())
    {
        TArray<FString> CurLabels = Sel->GetEnabledTargetActorLabels();
        CurLabels.Sort();
        TArray<FString> SavedLabels = Cp.TargetLabels;
        SavedLabels.Sort();
        if (CurLabels != SavedLabels)
        {
            UE_LOG(LogTexEnhancerPanel, Warning,
                TEXT("Resume: enabled target actors differ from the interrupted capture; "
                     "gt_materials will use the CURRENT selection."));
        }
    }

    // Restore run-wide settings from the checkpoint.
    ActiveCheckpoint      = Cp;
    BatchCaptureTimestamp = Cp.BatchTimestamp;
    bOutputMesh           = Cp.bOutputMesh;
    bUseCaptureReuse      = Cp.bUseCaptureReuse;
    bOutputImages         = true;
    if (Cp.GTTextureResolution > 0) GTTextureResolution = Cp.GTTextureResolution;

    // Build the work list: skip windows already fully present on disk; enqueue the rest. A run is
    // either batch (Slot >= 0 windows) or single (one Slot == -1 window) — never both.
    LightingCaptureQueue.Reset();
    bool bHasSingle = false;
    FString SingleDir;
    for (const FCaptureWindow& W : Cp.Windows)
    {
        const FString Dir = Root / W.DirName;
        if (PathCapture->IsCaptureWindowComplete(Dir, W.bRgbOnly))
        {
            continue;
        }
        if (W.Slot >= 0) LightingCaptureQueue.Add(W.Slot);
        else { bHasSingle = true; SingleDir = Dir; }
    }

    if (LightingCaptureQueue.Num() == 0 && !bHasSingle)
    {
        UE_LOG(LogTexEnhancerPanel, Log,
            TEXT("Resume: every window is already complete; clearing checkpoint."));
        FCaptureSessionCheckpoint::Clear(Root);
        ActiveCheckpoint = FCaptureSessionCheckpoint();
        return FReply::Handled();
    }

    bDatasetCaptureInProgress = true;
    if (LightingCaptureQueue.Num() > 0)
    {
        bBatchCapture = true;
        UE_LOG(LogTexEnhancerPanel, Log,
            TEXT("Resuming dataset capture: %d lighting window(s) remaining."), LightingCaptureQueue.Num());
        StartNextBatchCapture();
    }
    else
    {
        bBatchCapture = false;
        UE_LOG(LogTexEnhancerPanel, Log, TEXT("Resuming single dataset capture: %s"), *SingleDir);
        if (!DecideAndStartCapture(SingleDir))
        {
            bDatasetCaptureInProgress = false;
            UE_LOG(LogTexEnhancerPanel, Warning, TEXT("Resume: failed to start capture."));
        }
    }
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

    const FString CaptureDir = FindLatestCaptureDirectory();
    if (CaptureDir.IsEmpty())
    {
        FVCCSimUIHelpers::ShowNotification(
            TEXT("No capture found. Run 'Capture Dataset' first."), true);
        return FReply::Handled();
    }

    FString CameraInfoDir  = CaptureDir;
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
        *CaptureDir,
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
    Config.bUseCaptureReuse    = bUseCaptureReuse;

    Config.LightingElevation.Append(LightingElevation, NumLightingConditions);
    Config.LightingAzimuth.Append(LightingAzimuth, NumLightingConditions);
    Config.LightingSelected.Append(bLightingSelected, NumLightingConditions);

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
    bUseCaptureReuse = Config.bUseCaptureReuse;

    LoadParamsFromConfig();
}