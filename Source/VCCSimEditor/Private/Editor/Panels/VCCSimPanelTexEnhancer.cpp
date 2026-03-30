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
#include "Utils/ColmapManager.h"
#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Pawns/FlashPawn.h"
#include "Engine/DirectionalLight.h"
#include "Components/DirectionalLightComponent.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/StaticMesh.h"
#include "Engine/Texture.h"
#include "Engine/Texture2D.h"
#include "Materials/MaterialInterface.h"
#include "EngineUtils.h"
#include "Selection.h"
#include "MeshDescription.h"
#include "StaticMeshAttributes.h"
#include "Utils/VCCSimDataConverter.h"

#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
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

    if (!Config.GTActorLabels.IsEmpty())
    {
        GTActorListItems.Empty();
        for (const FString& Label : Config.GTActorLabels)
            GTActorListItems.Add(MakeShareable(new FString(Label)));
        if (GTActorListView.IsValid())
            GTActorListView->RequestListRefresh();
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
        UE_LOG(LogTexEnhancerPanel, Error, TEXT("Error: No editor world available"));
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
        UE_LOG(LogTexEnhancerPanel, Warning, TEXT("Warning: No Directional Light found in scene"));
        return;
    }

    if (bMarkDirty) DirectionalLight->Modify();
    FRotator NewRotation(-ElevationDeg, AzimuthDeg - 180.f, 0.f);
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

    ApplyLightingCondition(SunCalcElevation, SunCalcAzimuth);

    if (!bAboveHorizon)
    {
        UE_LOG(LogTexEnhancerPanel, Log, TEXT("Night: Sun %.1f below horizon"), -SunCalcElevation);
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

    const FString Timestamp = FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S"));
    const FString BaseDir = OutputDirectory / TEXT("gt_materials") / Timestamp;
    FPlatformFileManager::Get().GetPlatformFile().CreateDirectoryTree(*BaseDir);

    ExportMergedGTMaterials(BaseDir);
}

// ============================================================================
// GT EXPORT — internal POD structures (file-local)
// ============================================================================

struct FGTRawTex
{
    TArray64<uint8> Bytes;
    int32  W        = 0;
    int32  H        = 0;
    int32  Ch       = -1;
    float  Fallback = 0.f;
};

struct FGTMeshRaw
{
    FString    Label;
    FTransform WorldTransform;
    int32      ActorTileOffset = 0;

    TArray<FVector3f> LocalVertPos;
    TArray<int32>     InstVertIdx;
    TArray<FVector2f> InstUV0;
    TArray<int32>     TriInstFlat;
    TArray<int32>     TriSlotFlat;

    struct FSlotRaw
    {
        FString   MatName;
        FGTRawTex Rough, Metal, Color;
        int32     TileIdx = 0;
    };
    TArray<FSlotRaw> Slots;
};

struct FGTActorBuilt
{
    TArray<FVector>   WorldVerts;
    TArray<FVector2f> AtlasUVs;
    TArray<int32>     FaceVerts;
    TArray<int32>     FaceUVs;
};

// ── Game-thread helpers (UE object access) ─────────────────────────────────

static void GT_ExtractRawTex(UTexture2D* Tex, int32 Ch, FGTRawTex& Out)
{
    if (!Tex) return;
    FTextureSource& S = Tex->Source;
    if (!S.IsValid() || S.GetFormat() != TSF_BGRA8) return;
    Out.W = S.GetSizeX();
    Out.H = S.GetSizeY();
    Out.Ch = Ch;
    S.GetMipData(Out.Bytes, 0);
}

static void GT_CollectMatChannel(UMaterialInterface* Mat, bool bRough, FGTRawTex& Out)
{
    Out.Fallback = bRough ? 1.f : 0.f;
    if (!Mat) return;

    auto Get = [&](const FString& N) -> UTexture2D*
    {
        UTexture* T = nullptr;
        Mat->GetTextureParameterValue(FHashedMaterialParameterInfo(FName(*N)), T);
        return T ? Cast<UTexture2D>(T) : nullptr;
    };

    if (UTexture2D* ORM = Get(TEXT("ORM")))
    {
        GT_ExtractRawTex(ORM, bRough ? 1 : 2, Out);
        if (Out.Bytes.Num() > 0) return;
    }

    static const TArray<FString> RN = { TEXT("Roughness"), TEXT("RoughnessMap"), TEXT("T_Roughness"), TEXT("MetallicRoughnessTexture") };
    static const TArray<FString> MN = { TEXT("Metallic"),  TEXT("MetallicMap"),  TEXT("T_Metallic"),  TEXT("MetallicRoughnessTexture") };
    for (const FString& N : (bRough ? RN : MN))
    {
        if (UTexture2D* T = Get(N)) { GT_ExtractRawTex(T, -1, Out); if (Out.Bytes.Num() > 0) return; }
    }

    float V = Out.Fallback;
    Mat->GetScalarParameterValue(
        FHashedMaterialParameterInfo(FName(bRough ? TEXT("Roughness") : TEXT("Metallic"))), V);
    Out.Fallback = V;
}

static void GT_CollectBaseColor(UMaterialInterface* Mat, FGTRawTex& Out)
{
    Out.Fallback = 1.f;
    if (!Mat) return;

    auto Get = [&](const FString& N) -> UTexture2D*
    {
        UTexture* T = nullptr;
        Mat->GetTextureParameterValue(FHashedMaterialParameterInfo(FName(*N)), T);
        return T ? Cast<UTexture2D>(T) : nullptr;
    };

    static const TArray<FString> TexNames = {
        TEXT("BaseColor"), TEXT("Base Color"), TEXT("BaseColorMap"),
        TEXT("Albedo"), TEXT("AlbedoMap"), TEXT("DiffuseColor"),
        TEXT("Diffuse"), TEXT("T_BaseColor"), TEXT("Color")
    };
    for (const FString& N : TexNames)
    {
        if (UTexture2D* T = Get(N)) { GT_ExtractRawTex(T, -1, Out); if (Out.Bytes.Num() > 0) return; }
    }

    FLinearColor Vec = FLinearColor::White;
    for (const FString& N : { FString(TEXT("BaseColor")), FString(TEXT("Base Color")), FString(TEXT("Color")) })
    {
        if (Mat->GetVectorParameterValue(FHashedMaterialParameterInfo(FName(*N)), Vec))
        {
            const FColor C = Vec.ToFColor(true);
            Out.Bytes.SetNumUninitialized(4);
            Out.Bytes[0] = C.B; Out.Bytes[1] = C.G; Out.Bytes[2] = C.R; Out.Bytes[3] = C.A;
            Out.W = Out.H = 1; Out.Ch = -1;
            return;
        }
    }
}

// ── Background-safe helpers ────────────────────────────────────────────────

static TArray<FColor> BG_SampleFromRaw(const FGTRawTex& R, int32 TargetSize)
{
    TArray<FColor> Out;
    if (R.Bytes.IsEmpty())
    {
        const uint8 V = (uint8)FMath::Clamp(FMath::RoundToInt(R.Fallback * 255.f), 0, 255);
        Out.Init(FColor(V, V, V, 255), TargetSize * TargetSize);
        return Out;
    }

    const int32 N = R.W * R.H;
    TArray<FColor> Src;
    Src.SetNumUninitialized(N);
    for (int32 i = 0; i < N; ++i)
    {
        const uint8 B = R.Bytes[i*4], G = R.Bytes[i*4+1], Rv = R.Bytes[i*4+2], A = R.Bytes[i*4+3];
        if      (R.Ch == 0) Src[i] = FColor(Rv, Rv, Rv, 255);
        else if (R.Ch == 1) Src[i] = FColor(G,  G,  G,  255);
        else if (R.Ch == 2) Src[i] = FColor(B,  B,  B,  255);
        else                Src[i] = FColor(Rv, G,  B,  A);
    }

    if (R.W == TargetSize && R.H == TargetSize) return Src;

    Out.SetNumUninitialized(TargetSize * TargetSize);
    for (int32 Dy = 0; Dy < TargetSize; ++Dy)
    for (int32 Dx = 0; Dx < TargetSize; ++Dx)
    {
        const int32 Sx = FMath::Clamp(Dx * R.W / TargetSize, 0, R.W - 1);
        const int32 Sy = FMath::Clamp(Dy * R.H / TargetSize, 0, R.H - 1);
        Out[Dy * TargetSize + Dx] = Src[Sy * R.W + Sx];
    }
    return Out;
}

static FString BG_BuildOBJContent(const TArray<FGTActorBuilt>& Built)
{
    FString V = TEXT("mtllib merged_mesh.mtl\nusemtl merged_material\n");
    FString UV_str, F_str;
    for (const FGTActorBuilt& A : Built)
    {
        for (const FVector& P : A.WorldVerts)
            V += FString::Printf(TEXT("v %f %f %f\n"), P.X, P.Y, P.Z);
        for (const FVector2f& UV : A.AtlasUVs)
            UV_str += FString::Printf(TEXT("vt %f %f\n"), UV.X, UV.Y);
        for (int32 fi = 0; fi < A.FaceVerts.Num(); fi += 3)
            F_str += FString::Printf(TEXT("f %d/%d %d/%d %d/%d\n"),
                A.FaceVerts[fi],   A.FaceUVs[fi],
                A.FaceVerts[fi+2], A.FaceUVs[fi+2],
                A.FaceVerts[fi+1], A.FaceUVs[fi+1]);
    }
    return V + UV_str + F_str;
}

// ── ExportMergedGTMaterials ────────────────────────────────────────────────

void FVCCSimPanelTexEnhancer::ExportMergedGTMaterials(const FString& BaseDir)
{
    UWorld* World = GEditor->GetEditorWorldContext().World();

    TMap<FString, AStaticMeshActor*> LabelMap;
    for (TActorIterator<AStaticMeshActor> It(World); It; ++It)
        if (AStaticMeshActor* A = *It) LabelMap.Add(A->GetActorLabel(), A);

    TArray<AStaticMeshActor*> Actors;
    TArray<int32>             SlotCounts;
    TArray<FString>           Labels;

    for (const TSharedPtr<FString>& LabelPtr : GTActorListItems)
    {
        if (!LabelPtr.IsValid()) continue;
        AStaticMeshActor** Found = LabelMap.Find(*LabelPtr);
        if (!Found)
        {
            UE_LOG(LogTexEnhancerPanel, Warning, TEXT("GT Export: actor '%s' not found"), **LabelPtr);
            continue;
        }
        UStaticMeshComponent* MC = (*Found)->GetStaticMeshComponent();
        const int32 NS = MC ? MC->GetNumMaterials() : 0;
        if (NS == 0) continue;
        Actors.Add(*Found); SlotCounts.Add(NS); Labels.Add(*LabelPtr);
    }

    if (Actors.IsEmpty()) { FVCCSimUIHelpers::ShowNotification(TEXT("No valid actors to export."), true); return; }

    int32 TotalTiles = 0;
    TArray<int32> ActorTileOffsets;
    for (int32 i = 0; i < Actors.Num(); ++i) { ActorTileOffsets.Add(TotalTiles); TotalTiles += SlotCounts[i]; }

    const int32 AtlasCols = FMath::Max(1, FMath::CeilToInt(FMath::Sqrt((float)TotalTiles)));
    const int32 AtlasRows = FMath::Max(1, FMath::CeilToInt((float)TotalTiles / AtlasCols));

    // ── Game thread: minimal UObject access, copy to plain arrays ──────────

    TArray<FGTMeshRaw> RawMeshes;
    RawMeshes.Reserve(Actors.Num());

    TSharedPtr<FJsonObject> RootJson = MakeShareable(new FJsonObject);
    TArray<TSharedPtr<FJsonValue>> ActorArray;

    for (int32 ai = 0; ai < Actors.Num(); ++ai)
    {
        UStaticMeshComponent* MC = Actors[ai]->GetStaticMeshComponent();
        UStaticMesh* SM = MC->GetStaticMesh();
        if (!SM) continue;
        SM->ConditionalPostLoad();

        const FMeshDescription* MD = SM->GetMeshDescription(0);
        if (!MD) continue;

        FGTMeshRaw Raw;
        Raw.Label           = Labels[ai];
        Raw.WorldTransform  = Actors[ai]->GetActorTransform();
        Raw.ActorTileOffset = ActorTileOffsets[ai];

        FStaticMeshConstAttributes Attrs(*MD);
        TVertexAttributesConstRef<FVector3f>         Positions = Attrs.GetVertexPositions();
        TVertexInstanceAttributesConstRef<FVector2f> UVs       = Attrs.GetVertexInstanceUVs();

        // Build dense vertex ID → local index via TArray (cache-friendly, avoids TMap)
        int32 MaxVID = -1;
        for (const FVertexID VID : MD->Vertices().GetElementIDs())
            MaxVID = FMath::Max(MaxVID, VID.GetValue());

        TArray<int32> VIDToLocal;
        VIDToLocal.Init(-1, MaxVID + 1);
        Raw.LocalVertPos.Reserve(MD->Vertices().Num());
        for (const FVertexID VID : MD->Vertices().GetElementIDs())
        {
            VIDToLocal[VID.GetValue()] = Raw.LocalVertPos.Num();
            Raw.LocalVertPos.Add(Positions[VID]);
        }

        // Build dense instance ID → local index
        int32 MaxIID = -1;
        for (const FVertexInstanceID IID : MD->VertexInstances().GetElementIDs())
            MaxIID = FMath::Max(MaxIID, IID.GetValue());

        TArray<int32> IIDToLocal;
        IIDToLocal.Init(-1, MaxIID + 1);
        Raw.InstVertIdx.Reserve(MD->VertexInstances().Num());
        Raw.InstUV0.Reserve(MD->VertexInstances().Num());
        for (const FVertexInstanceID IID : MD->VertexInstances().GetElementIDs())
        {
            IIDToLocal[IID.GetValue()] = Raw.InstVertIdx.Num();
            Raw.InstVertIdx.Add(VIDToLocal[MD->GetVertexInstanceVertex(IID).GetValue()]);
            Raw.InstUV0.Add(UVs.Get(IID, 0));
        }

        // Polygon group → slot (small TMap, typically < 10 entries)
        TMap<int32, int32> GroupToSlot;
        { int32 Idx = 0; for (const FPolygonGroupID GID : MD->PolygonGroups().GetElementIDs()) GroupToSlot.Add(GID.GetValue(), Idx++); }

        // Copy triangle topology as plain int arrays
        const int32 NumTris = MD->Triangles().Num();
        Raw.TriInstFlat.Reserve(NumTris * 3);
        Raw.TriSlotFlat.Reserve(NumTris);
        for (const FTriangleID TID : MD->Triangles().GetElementIDs())
        {
            Raw.TriSlotFlat.Add(GroupToSlot.FindRef(MD->GetTrianglePolygonGroup(TID).GetValue()));
            for (const FVertexInstanceID IID : MD->GetTriangleVertexInstances(TID))
                Raw.TriInstFlat.Add(IIDToLocal[IID.GetValue()]);
        }

        // Collect texture raw data per slot (GetMipData must be game thread)
        TSharedPtr<FJsonObject> ActorJson = MakeShareable(new FJsonObject);
        ActorJson->SetStringField(TEXT("label"),     Labels[ai]);
        ActorJson->SetStringField(TEXT("mesh_file"), TEXT("merged_mesh.obj"));
        TArray<TSharedPtr<FJsonValue>> SlotArray;

        for (int32 si = 0; si < SlotCounts[ai]; ++si)
        {
            UMaterialInterface* Mat = MC->GetMaterial(si);
            const int32 TileIdx = ActorTileOffsets[ai] + si;

            FGTMeshRaw::FSlotRaw Slot;
            Slot.MatName = Mat ? Mat->GetName() : TEXT("");
            Slot.TileIdx = TileIdx;
            GT_CollectMatChannel(Mat, true,  Slot.Rough);
            GT_CollectMatChannel(Mat, false, Slot.Metal);
            GT_CollectBaseColor (Mat,        Slot.Color);

            TSharedPtr<FJsonObject> SlotJson = MakeShareable(new FJsonObject);
            SlotJson->SetNumberField(TEXT("slot"),          si);
            SlotJson->SetStringField(TEXT("material_name"), Slot.MatName);
            SlotJson->SetNumberField(TEXT("atlas_tile"),    TileIdx);
            SlotArray.Add(MakeShareable(new FJsonValueObject(SlotJson)));
            Raw.Slots.Add(MoveTemp(Slot));
        }
        ActorJson->SetArrayField(TEXT("slots"), SlotArray);
        ActorArray.Add(MakeShareable(new FJsonValueObject(ActorJson)));
        RawMeshes.Add(MoveTemp(Raw));
    }

    if (RawMeshes.IsEmpty()) { FVCCSimUIHelpers::ShowNotification(TEXT("No valid mesh data to export."), true); return; }

    TSharedPtr<FJsonObject> MetaJson = MakeShareable(new FJsonObject);
    MetaJson->SetStringField(TEXT("scene_name"),         SceneName);
    MetaJson->SetStringField(TEXT("exported_at"),        FDateTime::Now().ToString());
    MetaJson->SetNumberField(TEXT("actor_count"),        RawMeshes.Num());
    MetaJson->SetNumberField(TEXT("texture_resolution"), GTTextureResolution);
    MetaJson->SetNumberField(TEXT("atlas_cols"),         AtlasCols);
    MetaJson->SetNumberField(TEXT("atlas_rows"),         AtlasRows);
    MetaJson->SetStringField(TEXT("basecolor_atlas"),    TEXT("basecolor_atlas.png"));
    MetaJson->SetStringField(TEXT("roughness_atlas"),    TEXT("roughness_atlas.png"));
    MetaJson->SetStringField(TEXT("metallic_atlas"),     TEXT("metallic_atlas.png"));
    MetaJson->SetStringField(TEXT("mesh_file"),          TEXT("merged_mesh.obj"));
    RootJson->SetObjectField(TEXT("metadata"), MetaJson);
    RootJson->SetArrayField(TEXT("actors"),    ActorArray);

    FString JsonStr;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&JsonStr);
    FJsonSerializer::Serialize(RootJson.ToSharedRef(), Writer);

    // ── Dispatch ALL heavy computation and I/O to background thread ────────

    bGTExportInProgress = true;

    const int32 TexRes = GTTextureResolution;
    TWeakPtr<FVCCSimPanelTexEnhancer> WeakSelf = AsShared();

    Async(EAsyncExecution::Thread,
        [RawMeshes = MoveTemp(RawMeshes),
         JsonStr   = MoveTemp(JsonStr),
         BaseDir, AtlasCols, AtlasRows, TotalTiles, TexRes, WeakSelf]()
    {
        // ── Step 1: Build geometry and UV data ─────────────────────────────
        TArray<FGTActorBuilt> BuiltActors;
        BuiltActors.Reserve(RawMeshes.Num());

        int32 GlobalVertBase = 1;
        int32 GlobalUVBase   = 1;

        for (const FGTMeshRaw& Raw : RawMeshes)
        {
            const int32 NumVerts = Raw.LocalVertPos.Num();
            const int32 NumInsts = Raw.InstUV0.Num();
            const int32 NumTris  = Raw.TriSlotFlat.Num();

            FGTActorBuilt Built;

            // Transform vertices to right-handed world coords
            Built.WorldVerts.SetNumUninitialized(NumVerts);
            for (int32 vi = 0; vi < NumVerts; ++vi)
                Built.WorldVerts[vi] = FVCCSimDataConverter::ConvertLocation(
                    Raw.WorldTransform.TransformPosition(FVector(Raw.LocalVertPos[vi])));

            // Determine atlas tile per instance from triangle assignments
            TArray<int32> InstTile;
            InstTile.Init(-1, NumInsts);
            for (int32 ti = 0; ti < NumTris; ++ti)
            {
                const int32 TileIdx = Raw.ActorTileOffset + Raw.TriSlotFlat[ti];
                for (int32 k = 0; k < 3; ++k)
                {
                    const int32 IIdx = Raw.TriInstFlat[ti * 3 + k];
                    if (IIdx >= 0 && IIdx < NumInsts && InstTile[IIdx] < 0)
                        InstTile[IIdx] = TileIdx;
                }
            }

            // Compute atlas UV for each instance
            // PNG: Y=0=top, Row=0 is at top of image
            // OBJ: V=0=bottom, V=1=top
            // UE UV: V=0=top (DirectX convention)
            // Correct formula: V = (1 - srcV) / AtlasRows + (AtlasRows - Row - 1) / AtlasRows
            Built.AtlasUVs.SetNumUninitialized(NumInsts);
            for (int32 ii = 0; ii < NumInsts; ++ii)
            {
                const int32 Tile = (InstTile[ii] >= 0) ? InstTile[ii] : Raw.ActorTileOffset;
                const int32 Col  = Tile % AtlasCols;
                const int32 Row  = Tile / AtlasCols;
                const FVector2f SrcUV = Raw.InstUV0[ii];
                const float U = SrcUV.X / AtlasCols + (float)Col / AtlasCols;
                const float V = (1.f - SrcUV.Y) / AtlasRows + (float)(AtlasRows - Row - 1) / AtlasRows;
                Built.AtlasUVs[ii] = FVector2f(U, V);
            }

            // Build 1-based global face index arrays
            Built.FaceVerts.SetNumUninitialized(NumTris * 3);
            Built.FaceUVs.SetNumUninitialized(NumTris * 3);
            for (int32 ti = 0; ti < NumTris; ++ti)
            {
                for (int32 k = 0; k < 3; ++k)
                {
                    const int32 IIdx = Raw.TriInstFlat[ti * 3 + k];
                    Built.FaceVerts[ti * 3 + k] = GlobalVertBase + Raw.InstVertIdx[IIdx];
                    Built.FaceUVs  [ti * 3 + k] = GlobalUVBase   + IIdx;
                }
            }

            GlobalVertBase += NumVerts;
            GlobalUVBase   += NumInsts;

            BuiltActors.Add(MoveTemp(Built));
        }

        // ── Step 2: Sample atlas tiles ─────────────────────────────────────
        TArray<TArray<FColor>> RoughTiles, MetalTiles, ColorTiles;
        RoughTiles.SetNum(TotalTiles);
        MetalTiles.SetNum(TotalTiles);
        ColorTiles.SetNum(TotalTiles);

        for (const FGTMeshRaw& Raw : RawMeshes)
            for (const FGTMeshRaw::FSlotRaw& S : Raw.Slots)
            {
                RoughTiles[S.TileIdx] = BG_SampleFromRaw(S.Rough, TexRes);
                MetalTiles[S.TileIdx] = BG_SampleFromRaw(S.Metal, TexRes);
                ColorTiles[S.TileIdx] = BG_SampleFromRaw(S.Color, TexRes);
            }

        // ── Step 3: Write all output files ─────────────────────────────────
        const FString ObjContent = BG_BuildOBJContent(BuiltActors);

        const bool bObjOk   = FFileHelper::SaveStringToFile(ObjContent, *(BaseDir / TEXT("merged_mesh.obj")),
            FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), FILEWRITE_EvenIfReadOnly);
        const bool bMtlOk   = FFileHelper::SaveStringToFile(
            TEXT("newmtl merged_material\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n")
            TEXT("map_Kd basecolor_atlas.png\nmap_Pr roughness_atlas.png\nmap_Pm metallic_atlas.png\n"),
            *(BaseDir / TEXT("merged_mesh.mtl")));
        const bool bColorOk = FVCCSimPanelTexEnhancer::WriteAtlasPNG(ColorTiles, TexRes, AtlasCols, AtlasRows, BaseDir / TEXT("basecolor_atlas.png"));
        const bool bRoughOk = FVCCSimPanelTexEnhancer::WriteAtlasPNG(RoughTiles, TexRes, AtlasCols, AtlasRows, BaseDir / TEXT("roughness_atlas.png"));
        const bool bMetalOk = FVCCSimPanelTexEnhancer::WriteAtlasPNG(MetalTiles, TexRes, AtlasCols, AtlasRows, BaseDir / TEXT("metallic_atlas.png"));
        const bool bJsonOk  = FFileHelper::SaveStringToFile(JsonStr, *(BaseDir / TEXT("manifest.json")));

        if (!bObjOk)   UE_LOG(LogTexEnhancerPanel, Warning, TEXT("GT Export: failed merged_mesh.obj"));
        if (!bMtlOk)   UE_LOG(LogTexEnhancerPanel, Warning, TEXT("GT Export: failed merged_mesh.mtl"));
        if (!bColorOk) UE_LOG(LogTexEnhancerPanel, Warning, TEXT("GT Export: failed basecolor_atlas.png"));
        if (!bRoughOk) UE_LOG(LogTexEnhancerPanel, Warning, TEXT("GT Export: failed roughness_atlas.png"));
        if (!bMetalOk) UE_LOG(LogTexEnhancerPanel, Warning, TEXT("GT Export: failed metallic_atlas.png"));
        if (!bJsonOk)  UE_LOG(LogTexEnhancerPanel, Warning, TEXT("GT Export: failed manifest.json"));

        const bool bSuccess = bObjOk && bMtlOk && bColorOk && bRoughOk && bMetalOk && bJsonOk;
        const FString Msg = bSuccess
            ? FString::Printf(TEXT("GT export done: %d actors, %d tiles (%dx%d atlas) -> %s"),
                RawMeshes.Num(), TotalTiles, AtlasCols, AtlasRows, *BaseDir)
            : FString::Printf(TEXT("GT export completed with errors -> %s"), *BaseDir);

        UE_LOG(LogTexEnhancerPanel, Log, TEXT("%s"), *Msg);

        AsyncTask(ENamedThreads::GameThread, [WeakSelf, Msg, bSuccess]()
        {
            TSharedPtr<FVCCSimPanelTexEnhancer> Panel = WeakSelf.Pin();
            if (!Panel.IsValid()) return;
            Panel->bGTExportInProgress = false;
            FVCCSimUIHelpers::ShowNotification(Msg, !bSuccess);
        });
    });
}

bool FVCCSimPanelTexEnhancer::WriteMTLFile(const FString& MtlPath)
{
    FString Mtl;
    Mtl += TEXT("newmtl merged_material\n");
    Mtl += TEXT("Ka 1.0 1.0 1.0\n");
    Mtl += TEXT("Kd 1.0 1.0 1.0\n");
    Mtl += TEXT("Ks 0.0 0.0 0.0\n");
    Mtl += TEXT("map_Kd basecolor_atlas.png\n");
    Mtl += TEXT("map_Pr roughness_atlas.png\n");
    Mtl += TEXT("map_Pm metallic_atlas.png\n");
    return FFileHelper::SaveStringToFile(Mtl, *MtlPath);
}

bool FVCCSimPanelTexEnhancer::WriteAtlasPNG(
    const TArray<TArray<FColor>>& Tiles,
    int32 TileSize, int32 Cols, int32 Rows,
    const FString& PngPath)
{
    const int32 W = TileSize * Cols;
    const int32 H = TileSize * Rows;
    TArray<FColor> Atlas;
    Atlas.SetNumZeroed(W * H);

    for (int32 TileIdx = 0; TileIdx < Tiles.Num(); ++TileIdx)
    {
        const int32 Col = TileIdx % Cols;
        const int32 Row = TileIdx / Cols;
        const TArray<FColor>& Tile = Tiles[TileIdx];
        for (int32 Py = 0; Py < TileSize; ++Py)
        {
            for (int32 Px = 0; Px < TileSize; ++Px)
            {
                const int32 SrcIdx = Py * TileSize + Px;
                const int32 DstIdx = (Row * TileSize + Py) * W + (Col * TileSize + Px);
                if (SrcIdx < Tile.Num())
                    Atlas[DstIdx] = Tile[SrcIdx];
            }
        }
    }

    IImageWrapperModule& IWM = FModuleManager::LoadModuleChecked<IImageWrapperModule>("ImageWrapper");
    TSharedPtr<IImageWrapper> Wrapper = IWM.CreateImageWrapper(EImageFormat::PNG);
    if (!Wrapper.IsValid()) return false;

    Wrapper->SetRaw(Atlas.GetData(), Atlas.Num() * sizeof(FColor), W, H, ERGBFormat::BGRA, 8);
    TArray64<uint8> PngData = Wrapper->GetCompressed();
    if (PngData.IsEmpty()) return false;

    return FFileHelper::SaveArrayToFile(PngData, *PngPath);
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
}

