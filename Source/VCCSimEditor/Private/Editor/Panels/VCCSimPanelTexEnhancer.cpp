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
#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Pawns/FlashPawn.h"
#include "Utils/VCCSimDataConverter.h"
#include "DataStructures/CameraData.h"

#include "Engine/DirectionalLight.h"
#include "Components/DirectionalLightComponent.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/StaticMesh.h"
#include "Engine/Texture.h"
#include "Engine/Texture2D.h"
#include "Materials/MaterialInstance.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Kismet/GameplayStatics.h"
#include "EngineUtils.h"
#include "Selection.h"

#include "MeshDescription.h"
#include "StaticMeshAttributes.h"

#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Modules/ModuleManager.h"

#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "HAL/PlatformFilemanager.h"
#include "HAL/PlatformProcess.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Framework/Notifications/NotificationManager.h"
#include "Widgets/Notifications/SNotificationList.h"
#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#include "Framework/Application/SlateApplication.h"
#include "Editor.h"
#include "TimerManager.h"

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
    SphereRadiusValue  = SphereRadius;
    SphereRingsValue   = SphereRings;
    PosesPerRingValue  = PosesPerRing;
    NadirAltitudeValue = NadirAltitude;
    FrontOverlapValue  = FrontOverlap;
    SideOverlapValue   = SideOverlap;

    CaptureFOVValue    = CaptureFOVDegrees;
    CaptureWidthValue  = CaptureWidth;
    CaptureHeightValue = CaptureHeight;

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
    LoadPaths();
    UE_LOG(LogTexEnhancerPanel, Log, TEXT("TexEnhancer panel initialized"));
}

void FVCCSimPanelTexEnhancer::Cleanup()
{
    if (GEditor)
    {
        GEditor->GetTimerManager()->ClearTimer(StatusTimerHandle);
        GEditor->GetTimerManager()->ClearTimer(DayCycleTimerHandle);
    }
    bDayCycleActive = false;

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

void FVCCSimPanelTexEnhancer::ApplyLightingCondition(float ElevationDeg, float AzimuthDeg, bool bMarkDirty)
{
    if (!GEditor || !GEditor->GetEditorWorldContext().World())
    {
        UpdateStatus(TEXT("Error: No editor world available"));
        return;
    }

    UWorld* World = GEditor->GetEditorWorldContext().World();

    ADirectionalLight* DirectionalLight = nullptr;
    for (TActorIterator<ADirectionalLight> It(World); It; ++It)
    {
        ADirectionalLight* Candidate = *It;
        if (!Candidate) continue;

        UDirectionalLightComponent* LightComp = Candidate->GetComponent();
        if (LightComp && LightComp->bAtmosphereSunLight)
        {
            DirectionalLight = Candidate;
            break;
        }

        if (!DirectionalLight)
        {
            DirectionalLight = Candidate;
        }
    }

    if (!DirectionalLight)
    {
        UpdateStatus(TEXT("Warning: No Directional Light found in scene"));
        return;
    }

    if (bMarkDirty) DirectionalLight->Modify();
    FRotator NewRotation(-ElevationDeg, AzimuthDeg, 0.f);
    DirectionalLight->SetActorRotation(NewRotation);
    GEditor->RedrawAllViewports();

    FString Msg = FString::Printf(TEXT("Elev=%.1f\u00B0  Az=%.1f\u00B0  applied"), ElevationDeg, AzimuthDeg);
    if (LightingStatusTextBlock.IsValid())
        LightingStatusTextBlock->SetText(FText::FromString(Msg));
    UE_LOG(LogTexEnhancerPanel, Log, TEXT("Lighting applied: Elevation=%.1f Az=%.1f"), ElevationDeg, AzimuthDeg);
}

FReply FVCCSimPanelTexEnhancer::OnApplySetALightingClicked(int32 Index)
{
    if (Index < 0 || Index >= NumLightingSetA) return FReply::Handled();
    ApplyLightingCondition(SetAElevation[Index], SetAAzimuth[Index]);
    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnApplySetBLightingClicked(int32 Index)
{
    if (Index < 0 || Index >= NumLightingSetB) return FReply::Handled();
    ApplyLightingCondition(SetBElevation[Index], SetBAzimuth[Index]);
    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnCalculateSunPositionClicked()
{
    FVCCSimSunPositionHelper::FSunParams Params;
    Params.Latitude  = SunCalcLatitude;
    Params.Longitude = SunCalcLongitude;
    Params.TimeZone  = SunCalcTimeZone;
    Params.Year      = SunCalcYear;
    Params.Month     = SunCalcMonth;
    Params.Day       = SunCalcDay;
    Params.Hour      = SunCalcHour;
    Params.Minute    = SunCalcMinute;

    bool bAboveHorizon = FVCCSimSunPositionHelper::Calculate(Params, SunCalcElevation, SunCalcAzimuth);

    FString ResultStr = FString::Printf(
        TEXT("Elevation: %.1f°   Azimuth: %.1f°%s"),
        SunCalcElevation, SunCalcAzimuth,
        bAboveHorizon ? TEXT("") : TEXT("  ⚠ below horizon"));

    if (SunCalcResultTextBlock.IsValid())
    {
        SunCalcResultTextBlock->SetText(FText::FromString(ResultStr));
    }

    if (bAboveHorizon)
    {
        ApplyLightingCondition(SunCalcElevation, SunCalcAzimuth);
    }
    else
    {
        UpdateStatus(FString::Printf(TEXT("Sun is below the horizon at the specified time (Elevation=%.1f°)"), SunCalcElevation));
    }

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
    if (!GEditor) return FReply::Handled();

    bDayCycleActive = !bDayCycleActive;

    if (bDayCycleActive)
    {
        DayCycleSimMinute = 0.f;
        FTimerDelegate Del = FTimerDelegate::CreateRaw(this, &FVCCSimPanelTexEnhancer::TickDayCycle);
        GEditor->GetTimerManager()->SetTimer(DayCycleTimerHandle, Del, 0.1f, true);
    }
    else
    {
        GEditor->GetTimerManager()->ClearTimer(DayCycleTimerHandle);
    }

    return FReply::Handled();
}

void FVCCSimPanelTexEnhancer::TickDayCycle()
{
    const float MinutesPerTick = 1440.f * 0.1f / FMath::Max(DayCycleSpeed, 1.f);
    DayCycleSimMinute += MinutesPerTick;
    if (DayCycleSimMinute >= 1440.f) DayCycleSimMinute -= 1440.f;

    const int32 SimH = FMath::FloorToInt(DayCycleSimMinute / 60.f) % 24;
    const int32 SimM = FMath::FloorToInt(DayCycleSimMinute) % 60;

    FVCCSimSunPositionHelper::FSunParams Params;
    Params.Latitude  = SunCalcLatitude;
    Params.Longitude = SunCalcLongitude;
    Params.TimeZone  = SunCalcTimeZone;
    Params.Year      = SunCalcYear;
    Params.Month     = SunCalcMonth;
    Params.Day       = SunCalcDay;
    Params.Hour      = SimH;
    Params.Minute    = SimM;

    float Elev = 0.f, Az = 0.f;
    FVCCSimSunPositionHelper::Calculate(Params, Elev, Az);

    ApplyLightingCondition(Elev, Az, false);

    if (SunCalcResultTextBlock.IsValid())
    {
        SunCalcResultTextBlock->SetText(FText::FromString(
            FString::Printf(TEXT("%02d:%02d  Elev=%.1f°  Az=%.1f°"), SimH, SimM, Elev, Az)));
    }
}

// ============================================================================
// SECTION 3: CAPTURE PROTOCOL
// ============================================================================

FReply FVCCSimPanelTexEnhancer::OnCheckCoverageClicked()
{
    if (OutputDirectory.IsEmpty())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Please set an output directory first."), true);
        return FReply::Handled();
    }

    int32 TotalPoses = SphereRings * PosesPerRing;
    FString Msg = FString::Printf(
        TEXT("Coverage estimate: %d semi-spherical poses (%d rings × %d/ring)  |  Nadir grid at %.0f cm altitude"),
        TotalPoses, SphereRings, PosesPerRing, NadirAltitude);
    UpdateStatus(Msg);
    FVCCSimUIHelpers::ShowNotification(Msg, false);

    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnStartCaptureSetAClicked()
{
    ExecuteCapturePipeline(false);
    return FReply::Handled();
}

FReply FVCCSimPanelTexEnhancer::OnStartCaptureSetBClicked()
{
    ExecuteCapturePipeline(true);
    return FReply::Handled();
}

void FVCCSimPanelTexEnhancer::ExecuteCapturePipeline(bool bIsSetB)
{
    if (OutputDirectory.IsEmpty())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Output directory is not set."), true);
        return;
    }

    if (!SelectionManager.IsValid() || !SelectionManager.Pin()->GetSelectedFlashPawn().IsValid())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Please select a FlashPawn in the Selection panel."), true);
        return;
    }

    FString CaptureDir = bIsSetB ? GetSetBCaptureDir() : GetSetACaptureDir();

    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.DirectoryExists(*CaptureDir))
    {
        PlatformFile.CreateDirectoryTree(*CaptureDir);
    }

    bCaptureInProgress = true;
    FString SetLabel = bIsSetB ? TEXT("Set-B (Evaluation)") : TEXT("Set-A (Estimation)");
    FString Msg = FString::Printf(TEXT("Starting capture: %s  →  %s"), *SetLabel, *CaptureDir);
    UpdateStatus(Msg);
    UE_LOG(LogTexEnhancerPanel, Log, TEXT("%s"), *Msg);

    if (bIsSetB)
    {
        bSetBLocked = true;
        UpdateStatus(TEXT("Set-B captured and locked. This set is reserved for evaluation only."));
        FVCCSimUIHelpers::ShowNotification(TEXT("Set-B locked. Run evaluation after TexEnhancer completes."), false);
    }

    GenerateCameraInfoFromFlashPawn(FPaths::Combine(CaptureDir, TEXT("images")));

    bCaptureInProgress = false;
}

void FVCCSimPanelTexEnhancer::GenerateCameraInfoFromFlashPawn(const FString& ImageDir)
{
    if (!SelectionManager.IsValid()) return;

    TWeakObjectPtr<AFlashPawn> FlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    if (!FlashPawn.IsValid()) return;

    TArray<FVector> Positions;
    TArray<FRotator> Rotations;
    FlashPawn->GetCurrentPath(Positions, Rotations);

    if (Positions.Num() == 0)
    {
        UpdateStatus(TEXT("Warning: FlashPawn has no path poses — camera_info.json not generated."));
        return;
    }

    FCameraIntrinsics Intrinsics = FVCCSimDataConverter::ConvertCameraParamsWithFocalLength(
        CaptureFOVDegrees, CaptureWidth, CaptureHeight);

    FMatrix TWorld = FMatrix::Identity;
    TWorld.M[0][0] = 0.f; TWorld.M[0][1] = 1.f;
    TWorld.M[1][0] = 1.f; TWorld.M[1][1] = 0.f;
    TWorld.M[2][2] = 1.f;

    FMatrix TCam = FMatrix::Identity;
    TCam.M[0][0] = 0.f;  TCam.M[0][1] = 0.f;  TCam.M[0][2] = 1.f;
    TCam.M[1][0] = 1.f;  TCam.M[1][1] = 0.f;  TCam.M[1][2] = 0.f;
    TCam.M[2][0] = 0.f;  TCam.M[2][1] = -1.f; TCam.M[2][2] = 0.f;

    TArray<FCameraInfo> CameraInfos;
    CameraInfos.Reserve(Positions.Num());

    for (int32 i = 0; i < Positions.Num(); ++i)
    {
        FCameraInfo Info;
        Info.UID = i;

        FMatrix R_c2w_ue = FQuatRotationMatrix::Make(Rotations[i].Quaternion());
        FMatrix R_c2w_ts = TWorld.GetTransposed() * R_c2w_ue.GetTransposed() * TCam;
        Info.Rotation = R_c2w_ts.GetTransposed().ToQuat();
        Info.Position = FVCCSimDataConverter::ConvertLocation(Positions[i]);

        Info.FOVDegrees = CaptureFOVDegrees;
        Info.Width      = CaptureWidth;
        Info.Height     = CaptureHeight;
        Info.FocalX     = Intrinsics.FocalX;
        Info.FocalY     = Intrinsics.FocalY;
        Info.CenterX    = Intrinsics.CenterX;
        Info.CenterY    = Intrinsics.CenterY;

        FString ImageName = FVCCSimDataConverter::GenerateImageFileName(i);
        Info.ImageName = ImageName;
        Info.ImagePath = FPaths::Combine(ImageDir, ImageName);

        CameraInfos.Add(Info);
    }

    FString ConfigDir = FPaths::Combine(FPaths::GetPath(ImageDir), TEXT("config"));
    FPaths::NormalizeDirectoryName(ConfigDir);

    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.DirectoryExists(*ConfigDir))
    {
        PlatformFile.CreateDirectoryTree(*ConfigDir);
    }

    if (FVCCSimDataConverter::SaveCameraInfo(CameraInfos, ConfigDir))
    {
        FString Msg = FString::Printf(TEXT("camera_info.json saved: %d poses  →  %s"), Positions.Num(), *ConfigDir);
        UpdateStatus(Msg);
        UE_LOG(LogTexEnhancerPanel, Log, TEXT("%s"), *Msg);
    }
    else
    {
        FString Msg = FString::Printf(TEXT("Failed to save camera_info.json to: %s"), *ConfigDir);
        UpdateStatus(Msg);
        UE_LOG(LogTexEnhancerPanel, Warning, TEXT("%s"), *Msg);
    }
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

    ExportGTMaterialsFromScene();
    return FReply::Handled();
}

void FVCCSimPanelTexEnhancer::ExportGTMaterialsFromScene()
{
    if (!GEditor || !GEditor->GetEditorWorldContext().World())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("No editor world available."), true);
        return;
    }

    UWorld* World = GEditor->GetEditorWorldContext().World();

    TMap<FString, AStaticMeshActor*> LabelMap;
    for (TActorIterator<AStaticMeshActor> It(World); It; ++It)
    {
        if (AStaticMeshActor* A = *It)
            LabelMap.Add(A->GetActorLabel(), A);
    }

    const FString BaseDir = OutputDirectory / TEXT("gt_materials");
    FPlatformFileManager::Get().GetPlatformFile().CreateDirectoryTree(*BaseDir);

    TSharedPtr<FJsonObject> RootJson    = MakeShareable(new FJsonObject);
    TArray<TSharedPtr<FJsonValue>> ActorArray;
    int32 ExportedCount = 0;

    for (const TSharedPtr<FString>& LabelPtr : GTActorListItems)
    {
        if (!LabelPtr.IsValid()) continue;

        AStaticMeshActor** Found = LabelMap.Find(*LabelPtr);
        if (!Found)
        {
            UE_LOG(LogTexEnhancerPanel, Warning, TEXT("GT Export: actor '%s' not found in world"), **LabelPtr);
            continue;
        }

        TSharedPtr<FJsonObject> ActorJson = MakeShareable(new FJsonObject);
        ExportSingleActorGT(*Found, BaseDir, ActorJson);
        ActorArray.Add(MakeShareable(new FJsonValueObject(ActorJson)));
        ++ExportedCount;
    }

    TSharedPtr<FJsonObject> MetaJson = MakeShareable(new FJsonObject);
    MetaJson->SetStringField(TEXT("scene_name"),        SceneName);
    MetaJson->SetStringField(TEXT("exported_at"),       FDateTime::Now().ToString());
    MetaJson->SetNumberField(TEXT("actor_count"),       ExportedCount);
    MetaJson->SetNumberField(TEXT("texture_resolution"), GTTextureResolution);
    RootJson->SetObjectField(TEXT("metadata"), MetaJson);
    RootJson->SetArrayField(TEXT("actors"),   ActorArray);

    FString JsonStr;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&JsonStr);
    FJsonSerializer::Serialize(RootJson.ToSharedRef(), Writer);

    if (!FFileHelper::SaveStringToFile(JsonStr, *(BaseDir / TEXT("manifest.json"))))
    {
        UE_LOG(LogTexEnhancerPanel, Warning, TEXT("GT Export: failed to write manifest.json → %s"), *BaseDir);
    }

    const FString Msg = FString::Printf(TEXT("GT export done: %d actors → %s"), ExportedCount, *BaseDir);
    UpdateStatus(Msg);
    FVCCSimUIHelpers::ShowNotification(Msg, false);
    UE_LOG(LogTexEnhancerPanel, Log, TEXT("%s"), *Msg);
}

void FVCCSimPanelTexEnhancer::ExportSingleActorGT(
    AStaticMeshActor* Actor, const FString& BaseDir, TSharedPtr<FJsonObject> ActorJson)
{
    const FString Label = Actor->GetActorLabel();

    FString SafeLabel = Label;
    for (TCHAR& Ch : SafeLabel)
    {
        if (Ch == TEXT(' ')  || Ch == TEXT('/') || Ch == TEXT('\\') ||
            Ch == TEXT(':')  || Ch == TEXT('*') || Ch == TEXT('?')  ||
            Ch == TEXT('"')  || Ch == TEXT('<')  || Ch == TEXT('>')  || Ch == TEXT('|'))
        {
            Ch = TEXT('_');
        }
    }

    const FString ActorDir = BaseDir / SafeLabel;
    FPlatformFileManager::Get().GetPlatformFile().CreateDirectoryTree(*ActorDir);

    UStaticMeshComponent* MeshComp = Actor->GetStaticMeshComponent();
    UStaticMesh*           SM      = MeshComp ? MeshComp->GetStaticMesh() : nullptr;

    FString MeshFile;
    if (SM)
    {
        const FString ObjPath = ActorDir / TEXT("mesh.obj");
        if (ExportMeshAsOBJ(SM, ObjPath))
            MeshFile = SafeLabel / TEXT("mesh.obj");
        else
            UE_LOG(LogTexEnhancerPanel, Warning, TEXT("GT Export: mesh export failed for '%s'"), *Label);
    }

    ActorJson->SetStringField(TEXT("label"),     Label);
    ActorJson->SetStringField(TEXT("mesh_file"), MeshFile);

    TArray<TSharedPtr<FJsonValue>> SlotArray;
    if (MeshComp)
    {
        for (int32 SlotIdx = 0; SlotIdx < MeshComp->GetNumMaterials(); ++SlotIdx)
        {
            UMaterialInterface* Mat = MeshComp->GetMaterial(SlotIdx);
            if (!Mat) continue;

            TSharedPtr<FJsonObject> SlotJson = MakeShareable(new FJsonObject);
            SlotJson->SetNumberField(TEXT("slot"),          SlotIdx);
            SlotJson->SetStringField(TEXT("material_name"), Mat->GetName());

            ExportMaterialSlotTextures(Mat, SlotIdx, ActorDir, SlotJson);
            SlotArray.Add(MakeShareable(new FJsonValueObject(SlotJson)));
        }
    }
    ActorJson->SetArrayField(TEXT("slots"), SlotArray);
}

bool FVCCSimPanelTexEnhancer::ExportMeshAsOBJ(UStaticMesh* SM, const FString& ObjPath)
{
    SM->ConditionalPostLoad();

    const FMeshDescription* MD = SM->GetMeshDescription(0);
    if (!MD)
    {
        UE_LOG(LogTexEnhancerPanel, Warning, TEXT("No MeshDescription for LOD0 on '%s'"), *SM->GetName());
        return false;
    }

    FStaticMeshConstAttributes Attrs(*MD);
    TVertexAttributesConstRef<FVector3f>    Positions = Attrs.GetVertexPositions();
    TVertexInstanceAttributesConstRef<FVector2f> UVs  = Attrs.GetVertexInstanceUVs();

    FString Obj;
    Obj.Reserve(MD->Vertices().Num() * 32 + MD->VertexInstances().Num() * 20 + MD->Triangles().Num() * 24);

    for (const FVertexID VID : MD->Vertices().GetElementIDs())
    {
        const FVector3f& P = Positions[VID];
        Obj += FString::Printf(TEXT("v %f %f %f\n"), P.X, P.Y, P.Z);
    }

    for (const FVertexInstanceID IID : MD->VertexInstances().GetElementIDs())
    {
        const FVector2f UV = UVs.Get(IID, 0);
        Obj += FString::Printf(TEXT("vt %f %f\n"), UV.X, 1.f - UV.Y);
    }

    for (const FTriangleID TID : MD->Triangles().GetElementIDs())
    {
        TArrayView<const FVertexInstanceID> Insts = MD->GetTriangleVertexInstances(TID);
        Obj += TEXT("f");
        for (const FVertexInstanceID IID : Insts)
        {
            const int32 VI = MD->GetVertexInstanceVertex(IID).GetValue() + 1;
            const int32 UI = IID.GetValue() + 1;
            Obj += FString::Printf(TEXT(" %d/%d"), VI, UI);
        }
        Obj += TEXT("\n");
    }

    return FFileHelper::SaveStringToFile(Obj, *ObjPath,
        FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), FILEWRITE_EvenIfReadOnly);
}

bool FVCCSimPanelTexEnhancer::ExportTextureAsPNG(UTexture2D* Tex, const FString& PngPath, int32 Channel)
{
    if (!Tex) return false;

    FTextureSource& Source = Tex->Source;
    if (!Source.IsValid()) return false;

    const int32              W      = Source.GetSizeX();
    const int32              H      = Source.GetSizeY();
    const ETextureSourceFormat Fmt  = Source.GetFormat();

    TArray64<uint8> Raw;
    if (!Source.GetMipData(Raw, 0)) return false;

    TArray<FColor> Pixels;
    Pixels.SetNumUninitialized(W * H);

    auto WritePixel = [&](int32 i, uint8 R, uint8 G, uint8 B, uint8 A)
    {
        if (Channel == 0)      Pixels[i] = FColor(R, R, R, 255);
        else if (Channel == 1) Pixels[i] = FColor(G, G, G, 255);
        else if (Channel == 2) Pixels[i] = FColor(B, B, B, 255);
        else                   Pixels[i] = FColor(R, G, B, A);
    };

    if (Fmt == TSF_BGRA8)
    {
        for (int32 i = 0; i < W * H; ++i)
            WritePixel(i, Raw[i*4+2], Raw[i*4+1], Raw[i*4+0], Raw[i*4+3]);
    }
    else
    {
        UE_LOG(LogTexEnhancerPanel, Warning,
            TEXT("ExportTextureAsPNG: unsupported source format %d on '%s' — skipping"),
            (int32)Fmt, *Tex->GetName());
        return false;
    }

    IImageWrapperModule& IWM = FModuleManager::LoadModuleChecked<IImageWrapperModule>("ImageWrapper");
    TSharedPtr<IImageWrapper> Wrapper = IWM.CreateImageWrapper(EImageFormat::PNG);
    if (!Wrapper.IsValid()) return false;

    Wrapper->SetRaw(Pixels.GetData(), Pixels.Num() * sizeof(FColor), W, H, ERGBFormat::BGRA, 8);
    TArray64<uint8> PngData = Wrapper->GetCompressed();
    if (PngData.IsEmpty()) return false;

    return FFileHelper::SaveArrayToFile(PngData, *PngPath);
}

bool FVCCSimPanelTexEnhancer::ExportSolidColorPNG(float Value, int32 Resolution, const FString& PngPath)
{
    const uint8 Val = (uint8)FMath::Clamp(FMath::RoundToInt(Value * 255.f), 0, 255);

    TArray<FColor> Pixels;
    Pixels.Init(FColor(Val, Val, Val, 255), Resolution * Resolution);

    IImageWrapperModule& IWM = FModuleManager::LoadModuleChecked<IImageWrapperModule>("ImageWrapper");
    TSharedPtr<IImageWrapper> Wrapper = IWM.CreateImageWrapper(EImageFormat::PNG);
    if (!Wrapper.IsValid()) return false;

    Wrapper->SetRaw(Pixels.GetData(), Pixels.Num() * sizeof(FColor),
        Resolution, Resolution, ERGBFormat::BGRA, 8);
    TArray64<uint8> PngData = Wrapper->GetCompressed();
    if (PngData.IsEmpty()) return false;

    return FFileHelper::SaveArrayToFile(PngData, *PngPath);
}

bool FVCCSimPanelTexEnhancer::ExportMaterialSlotTextures(
    UMaterialInterface* Mat, int32 SlotIdx, const FString& ActorDir, TSharedPtr<FJsonObject> SlotJson)
{
    if (!Mat) return false;

    auto TryGetTex2D = [&](const FString& ParamName) -> UTexture2D*
    {
        const FName ParamFName(*ParamName);
        FHashedMaterialParameterInfo ParamInfo(ParamFName);
        UTexture* T = nullptr;
        if (Mat->GetTextureParameterValue(ParamInfo, T) && T)
            return Cast<UTexture2D>(T);
        return nullptr;
    };

    auto GetScalar = [&](const FString& ParamName, float Default) -> float
    {
        float V = Default;
        const FName ParamFName(*ParamName);
        FHashedMaterialParameterInfo ParamInfo(ParamFName);
        Mat->GetScalarParameterValue(ParamInfo, V);
        return V;
    };

    static const TArray<FString> RoughnessNames = {
        TEXT("Roughness"), TEXT("RoughnessMap"), TEXT("T_Roughness"), TEXT("MetallicRoughnessTexture") };
    static const TArray<FString> MetallicNames = {
        TEXT("Metallic"), TEXT("MetallicMap"), TEXT("T_Metallic"), TEXT("MetallicRoughnessTexture") };

    auto ExportPBRChannel = [&](
        const TArray<FString>& DedicatedNames,
        const FString& ORMParamName, int32 ORMChannel,
        const FString& ScalarParamName, float ScalarDefault,
        const FString& SlotPrefix)
    {
        const FString PngName    = FString::Printf(TEXT("%s_s%d.png"), *SlotPrefix, SlotIdx);
        const FString PngPath    = ActorDir / PngName;
        const FString SourceKey  = SlotPrefix + TEXT("_source");
        const FString TexKey     = SlotPrefix + TEXT("_tex");
        const FString ScalarKey  = SlotPrefix + TEXT("_scalar");

        if (UTexture2D* ORM = TryGetTex2D(ORMParamName))
        {
            if (ExportTextureAsPNG(ORM, PngPath, ORMChannel))
            {
                const FString ORMSource = FString::Printf(TEXT("orm_%s_channel"),
                    ORMChannel == 1 ? TEXT("g") : TEXT("b"));
                SlotJson->SetStringField(SourceKey, ORMSource);
                SlotJson->SetStringField(TexKey,    PngName);
                return;
            }
        }

        for (const FString& Name : DedicatedNames)
        {
            if (UTexture2D* T = TryGetTex2D(Name))
            {
                if (ExportTextureAsPNG(T, PngPath, -1))
                {
                    SlotJson->SetStringField(SourceKey, TEXT("texture"));
                    SlotJson->SetStringField(TexKey,    PngName);
                    return;
                }
            }
        }

        const float Scalar = GetScalar(ScalarParamName, ScalarDefault);
        ExportSolidColorPNG(Scalar, GTTextureResolution, PngPath);
        SlotJson->SetStringField(SourceKey, TEXT("scalar"));
        SlotJson->SetStringField(TexKey,    PngName);
        SlotJson->SetNumberField(ScalarKey, Scalar);
    };

    ExportPBRChannel(RoughnessNames, TEXT("ORM"), 1, TEXT("Roughness"), 1.f, TEXT("roughness"));
    ExportPBRChannel(MetallicNames,  TEXT("ORM"), 2, TEXT("Metallic"),  0.f, TEXT("metallic"));

    return true;
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
        UpdateStatus(TEXT("TexEnhancer pipeline started..."));

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
    UpdateStatus(TEXT("TexEnhancer pipeline stopped by user."));

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

        UpdateStatus(Msg);
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
    UpdateStatus(TEXT("BRDF evaluation complete."));

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

void FVCCSimPanelTexEnhancer::UpdateStatus(const FString& Message)
{
    StatusMessage = Message;
    if (StatusTextBlock.IsValid())
    {
        StatusTextBlock->SetText(FText::FromString(Message));
    }
}

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
// PATH PERSISTENCE
// ============================================================================

void FVCCSimPanelTexEnhancer::SavePaths()
{
    FVCCSimConfigManager::FTexEnhancerConfig Config;
    Config.OutputDirectory      = OutputDirectory;
    Config.SceneName            = SceneName;
    Config.TexEnhancerScriptPath = TexEnhancerScriptPath;
    Config.EstimatedMaterialsDir = EstimatedMaterialsDir;
    FVCCSimConfigManager::Get().SetTexEnhancerConfig(Config);
}

void FVCCSimPanelTexEnhancer::LoadPaths()
{
    const auto& Config = FVCCSimConfigManager::Get().GetTexEnhancerConfig();
    if (!Config.OutputDirectory.IsEmpty())      OutputDirectory      = Config.OutputDirectory;
    if (!Config.SceneName.IsEmpty())            SceneName            = Config.SceneName;
    if (!Config.TexEnhancerScriptPath.IsEmpty()) TexEnhancerScriptPath = Config.TexEnhancerScriptPath;
    if (!Config.EstimatedMaterialsDir.IsEmpty()) EstimatedMaterialsDir = Config.EstimatedMaterialsDir;
}

FString FVCCSimPanelTexEnhancer::GetPathConfigFilePath() const
{
    return FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("Config"), TEXT("VCCSimTexEnhancer.json"));
}
