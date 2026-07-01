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

DEFINE_LOG_CATEGORY_STATIC(LogPathImageCapture, Log, All);

#include "Editor/Panels/VCCSimPanelPathImageCapture.h"
#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Utils/VCCSimConfigManager.h"
#include "Utils/ConfigParser.h"
#include "Utils/PathGenerator.h"
#include "Utils/ImageCaptureService.h"
#include "Utils/SkyHDRICapture.h"
#include "Utils/LightingManager.h"
#include "Utils/GTMaterialExporter.h"
#include "Utils/VCCSimSunPositionHelper.h"
#include "Utils/VCCSimUIHelpers.h"
#include "Pawns/FlashPawn.h"
#include "Pawns/SimLookAtPath.h"
#include "Sensors/RGBCamera.h"
#include "Sensors/DepthCamera.h"
#include "Components/PrimitiveComponent.h"
#include "Engine/DirectionalLight.h"
#include "Components/DirectionalLightComponent.h"
#include "Engine/SkyLight.h"
#include "Components/SkyLightComponent.h"
#include "Engine/ExponentialHeightFog.h"
#include "Components/ExponentialHeightFogComponent.h"
#include "Styling/AppStyle.h"
#include "Styling/CoreStyle.h"
#include "Utils/TrajectoryViewer.h"
#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#include "EngineUtils.h"
#include "LevelEditorViewport.h"
#include "Async/Async.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Misc/SecureHash.h"
#include "HAL/FileManager.h"
#include "Framework/Application/SlateApplication.h"
#include "Editor.h"

static bool EnsureGameView()
{
    for (FLevelEditorViewportClient* Client : GEditor->GetLevelViewportClients())
    {
        if (Client && Client->IsPerspective())
        {
            if (!Client->IsInGameView())
            {
                Client->SetGameView(true);
                return true;
            }
            return false;
        }
    }
    return false;
}

static void RestoreGameView(bool bWasChangedByUs)
{
    if (!bWasChangedByUs) return;
    for (FLevelEditorViewportClient* Client : GEditor->GetLevelViewportClients())
    {
        if (Client && Client->IsPerspective())
        {
            Client->SetGameView(false);
            return;
        }
    }
}

// ============================================================================
// CAPTURE-TIME METADATA WRITERS (intrinsics.json + lighting.json)
// ============================================================================
//
// These sit next to poses.txt so preprocess/ue_to_colmap.py can build the
// COLMAP-style cameras.json without any Python-side intrinsics guessing.

namespace
{
    bool WriteJsonObjectToFile(const TSharedRef<FJsonObject>& Obj, const FString& FilePath)
    {
        FString OutString;
        TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutString);
        if (!FJsonSerializer::Serialize(Obj, Writer))
        {
            return false;
        }
        return FFileHelper::SaveStringToFile(OutString, *FilePath);
    }

    void WriteIntrinsicsJsonFromFlashPawn(AFlashPawn* Pawn, const FString& OutDir)
    {
        if (!Pawn)
        {
            return;
        }

        TArray<URGBCameraComponent*> Cameras;
        Pawn->GetComponents<URGBCameraComponent>(Cameras);
        if (Cameras.Num() == 0)
        {
            UE_LOG(LogPathImageCapture, Warning,
                TEXT("WriteIntrinsicsJson: no RGB camera on FlashPawn; skip"));
            return;
        }

        URGBCameraComponent* Camera = Cameras[0];
        const float FOV = Camera->FOV;
        const std::pair<int32, int32> Size = Camera->GetImageSize();
        const int32 Width  = Size.first;
        const int32 Height = Size.second;

        // UE FOVAngle is the horizontal FOV. Pinhole with square pixels → fx = fy.
        const float FxFy = (static_cast<float>(Width) * 0.5f)
                         / FMath::Tan(FMath::DegreesToRadians(FOV) * 0.5f);
        const float Cx = static_cast<float>(Width)  * 0.5f;
        const float Cy = static_cast<float>(Height) * 0.5f;

        TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
        Root->SetStringField(TEXT("model"),  TEXT("PINHOLE"));
        Root->SetNumberField(TEXT("width"),  Width);
        Root->SetNumberField(TEXT("height"), Height);
        Root->SetNumberField(TEXT("fx"),     FxFy);
        Root->SetNumberField(TEXT("fy"),     FxFy);
        Root->SetNumberField(TEXT("cx"),     Cx);
        Root->SetNumberField(TEXT("cy"),     Cy);
        Root->SetNumberField(TEXT("fov_h_deg"), FOV);

        if (Cameras.Num() > 1)
        {
            TArray<TSharedPtr<FJsonValue>> ExtraCams;
            for (int32 i = 1; i < Cameras.Num(); ++i)
            {
                URGBCameraComponent* Extra = Cameras[i];
                const std::pair<int32, int32> ESize = Extra->GetImageSize();
                TSharedRef<FJsonObject> ECam = MakeShared<FJsonObject>();
                ECam->SetNumberField(TEXT("index"),  Extra->GetSensorIndex());
                ECam->SetNumberField(TEXT("fov_h_deg"), Extra->FOV);
                ECam->SetNumberField(TEXT("width"),  ESize.first);
                ECam->SetNumberField(TEXT("height"), ESize.second);
                ExtraCams.Add(MakeShared<FJsonValueObject>(ECam));
            }
            Root->SetArrayField(TEXT("extra_cameras"), ExtraCams);
        }

        const FString OutPath = OutDir / TEXT("intrinsics.json");
        if (!WriteJsonObjectToFile(Root, OutPath))
        {
            UE_LOG(LogPathImageCapture, Warning,
                TEXT("Failed to save intrinsics to %s"), *OutPath);
            return;
        }
        UE_LOG(LogPathImageCapture, Log,
            TEXT("Wrote intrinsics.json (fx=fy=%.3f cx=%.1f cy=%.1f W=%d H=%d FOV=%.2f)"),
            FxFy, Cx, Cy, Width, Height, FOV);
    }

    void WriteLightingJsonFromWorld(UWorld* World, const FString& OutDir)
    {
        if (!World)
        {
            return;
        }

        ADirectionalLight* Best = nullptr;
        float BestIntensity = -1.0f;
        for (TActorIterator<ADirectionalLight> It(World); It; ++It)
        {
            ADirectionalLight* Light = *It;
            if (!Light) continue;
            UDirectionalLightComponent* Comp = Cast<UDirectionalLightComponent>(
                Light->GetLightComponent());
            if (!Comp || !Comp->bAffectsWorld) continue;
            const float Intensity = Comp->Intensity;
            if (Intensity > BestIntensity)
            {
                Best = Light;
                BestIntensity = Intensity;
            }
        }

        if (!Best)
        {
            UE_LOG(LogPathImageCapture, Warning,
                TEXT("WriteLightingJson: no ADirectionalLight in world; skip"));
            return;
        }

        UDirectionalLightComponent* Comp = Cast<UDirectionalLightComponent>(
            Best->GetLightComponent());
        const FRotator UeRot = Best->GetActorRotation();
        const FVector ForwardUe = UeRot.Vector();  // UE LH (+X fwd, +Y right, +Z up)

        // UE → RH world (+X right, +Y fwd, +Z up): swap X↔Y.
        // Sun direction points toward the sun, so negate the light forward.
        const FVector ForwardRh(ForwardUe.Y, ForwardUe.X, ForwardUe.Z);
        FVector SunDir = -ForwardRh;
        if (!SunDir.Normalize())
        {
            SunDir = FVector(0.0f, 0.0f, 1.0f);
        }

        TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
        Root->SetStringField(TEXT("coord_frame"),
            TEXT("RH world, +X east/right, +Y north/fwd, +Z up"));

        TArray<TSharedPtr<FJsonValue>> SunDirArr;
        SunDirArr.Add(MakeShared<FJsonValueNumber>(SunDir.X));
        SunDirArr.Add(MakeShared<FJsonValueNumber>(SunDir.Y));
        SunDirArr.Add(MakeShared<FJsonValueNumber>(SunDir.Z));
        Root->SetArrayField(TEXT("sun_dir_world"), SunDirArr);

        TSharedRef<FJsonObject> UeRotJson = MakeShared<FJsonObject>();
        UeRotJson->SetNumberField(TEXT("pitch"), UeRot.Pitch);
        UeRotJson->SetNumberField(TEXT("yaw"),   UeRot.Yaw);
        UeRotJson->SetNumberField(TEXT("roll"),  UeRot.Roll);
        Root->SetObjectField(TEXT("ue_directional_light_rotation"), UeRotJson);

        Root->SetStringField(TEXT("actor_label"), Best->GetActorLabel());
        if (Comp)
        {
            Root->SetNumberField(TEXT("sun_intensity"), Comp->Intensity);
            const FLinearColor Color = Comp->GetLightColor();
            TArray<TSharedPtr<FJsonValue>> ColorArr;
            ColorArr.Add(MakeShared<FJsonValueNumber>(Color.R));
            ColorArr.Add(MakeShared<FJsonValueNumber>(Color.G));
            ColorArr.Add(MakeShared<FJsonValueNumber>(Color.B));
            Root->SetArrayField(TEXT("light_color_linear"), ColorArr);
            Root->SetBoolField(TEXT("atmosphere_sun_light"), Comp->IsUsedAsAtmosphereSunLight());
            Root->SetStringField(TEXT("sun_intensity_unit"), TEXT("lux"));
            Root->SetNumberField(TEXT("sun_angular_diameter_deg"), Comp->LightSourceAngle);
        }

        Root->SetStringField(TEXT("exposure_mode"), TEXT("manual"));
        Root->SetNumberField(TEXT("exposure_value"), 1.0);
        Root->SetStringField(TEXT("tone_curve"), TEXT("disabled"));

        ASkyLight* SkyLight = nullptr;
        for (TActorIterator<ASkyLight> It(World); It; ++It)
        {
            if (*It) { SkyLight = *It; break; }
        }
        if (SkyLight)
        {
            if (USkyLightComponent* SkyComp = SkyLight->GetLightComponent())
            {
                Root->SetNumberField(TEXT("sky_intensity_scale"), SkyComp->Intensity);
            }
        }
        Root->SetStringField(TEXT("sky_hdri"), TEXT("sky.exr"));
        Root->SetStringField(TEXT("sky_equirect_convention"),
            TEXT("RH world +Z up; v=0->+Z (north pole), v=H->-Z (south pole); "
                 "u: yaw=0->+Y (north/fwd), increasing toward +X (east/right)"));

        AExponentialHeightFog* Fog = nullptr;
        for (TActorIterator<AExponentialHeightFog> It(World); It; ++It)
        {
            if (*It) { Fog = *It; break; }
        }
        UExponentialHeightFogComponent* FogComp = Fog ? Fog->GetComponent() : nullptr;
        if (FogComp && FogComp->FogDensity > 0.0f)
        {
            TSharedRef<FJsonObject> FogJson = MakeShared<FJsonObject>();
            FogJson->SetNumberField(TEXT("density"), FogComp->FogDensity);
            FogJson->SetNumberField(TEXT("height_falloff"), FogComp->FogHeightFalloff);
            FogJson->SetNumberField(TEXT("fog_height"), Fog->GetActorLocation().Z);
            FogJson->SetNumberField(TEXT("start_distance"), FogComp->StartDistance);
            Root->SetObjectField(TEXT("height_fog"), FogJson);
        }
        else
        {
            Root->SetStringField(TEXT("height_fog"), TEXT("off"));
        }

        Root->SetStringField(TEXT("utc_captured"), FDateTime::UtcNow().ToIso8601());

        const FString OutPath = OutDir / TEXT("lighting.json");
        if (!WriteJsonObjectToFile(Root, OutPath))
        {
            UE_LOG(LogPathImageCapture, Warning,
                TEXT("Failed to save lighting to %s"), *OutPath);
            return;
        }
        UE_LOG(LogPathImageCapture, Log,
            TEXT("Wrote lighting.json (sun_dir_world=(%.3f, %.3f, %.3f) from '%s')"),
            SunDir.X, SunDir.Y, SunDir.Z, *Best->GetActorLabel());
    }
}

static void ApplyCameraLocalTransform(
    AFlashPawn* Pawn,
    TArray<FVector>& Positions,
    TArray<FRotator>& Rotations)
{
    TArray<URGBCameraComponent*> Cameras;
    Pawn->GetComponents<URGBCameraComponent>(Cameras);
    if (Cameras.Num() == 0) return;
    const FVector LocalPos = Cameras[0]->GetRelativeLocation();
    const FQuat   LocalRot = Cameras[0]->GetRelativeRotation().Quaternion();
    for (int32 i = 0; i < Positions.Num(); ++i)
    {
        const FQuat ActorRot = Rotations[i].Quaternion();
        Positions[i] = Positions[i] + ActorRot.RotateVector(LocalPos);
        Rotations[i] = (ActorRot * LocalRot).Rotator();
    }
}

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

FVCCSimPanelPathImageCapture::FVCCSimPanelPathImageCapture()
{
    PathGenerator = MakeShared<FPathGenerator>();
    // ImageCaptureService is initialized in SetSelectionManager, as it depends on it.

    OrbitMarginValue      = OrbitMargin;
    OrbitStartHeightValue = OrbitStartHeight;
    OrbitCameraHFOVValue  = OrbitCameraHFOV;
    OrbitHOverlapValue    = OrbitHOverlap;
    OrbitVOverlapValue    = OrbitVOverlap;
    OrbitNadirAltValue    = OrbitNadirAlt;
    OrbitNadirTiltValue   = OrbitNadirTiltAngle;
    OrbitObliqueRingsValue = OrbitObliqueRings;
    CaptureTickIntervalValue = CaptureTickInterval;
    PoseWarmupFramesValue = PoseWarmupFrames;

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
}

FVCCSimPanelPathImageCapture::~FVCCSimPanelPathImageCapture()
{
    Cleanup();
}

// ============================================================================
// INITIALIZATION
// ============================================================================

void FVCCSimPanelPathImageCapture::Initialize()
{
    LightingManager = MakeShared<FLightingManager>();
}

void FVCCSimPanelPathImageCapture::Cleanup()
{
    LightingManager.Reset();

    // Clear timer if active
    if (GEditor && bAutoCaptureInProgress)
    {
        GEditor->GetTimerManager()->ClearTimer(AutoCaptureTimerHandle);
        bAutoCaptureInProgress = false;
    }

    // Clean up path visualization
    if (GEditor && GEditor->GetEditorWorldContext().World())
    {
        UWorld* World = GEditor->GetEditorWorldContext().World();
        
        // Clean up any PathVisualization actors
        for (TActorIterator<AActor> ActorIterator(World); ActorIterator; ++ActorIterator)
        {
            AActor* Actor = *ActorIterator;
            if (Actor && (Actor->GetActorLabel().Contains(TEXT("PathVisualization")) || 
                         Actor->Tags.Contains(FName("VCCSimPathViz"))))
            {
                World->DestroyActor(Actor);
            }
        }
    }
    
    if (PathVisualizationActor.IsValid() && GEditor)
    {
        if (UWorld* World = GEditor->GetEditorWorldContext().World())
        {
            World->DestroyActor(PathVisualizationActor.Get());
        }
        PathVisualizationActor.Reset();
    }
}

void FVCCSimPanelPathImageCapture::SetSelectionManager(TSharedPtr<FVCCSimPanelSelection> InSelectionManager)
{
    SelectionManager = InSelectionManager;
    ImageCaptureService = MakeShared<FImageCaptureService>(InSelectionManager);
}

void FVCCSimPanelPathImageCapture::LoadFromConfigManager()
{
    const auto& Config = FVCCSimConfigManager::Get().GetPathImageCaptureConfig();

    OrbitMargin         = Config.Margin;
    OrbitStartHeight    = Config.StartHeight;
    OrbitCameraHFOV     = Config.CameraHFOV;
    OrbitHOverlap       = Config.HOverlap;
    OrbitVOverlap       = Config.VOverlap;
    OrbitSurveyOverlap  = Config.SurveyHOverlap;
    OrbitNadirAlt       = Config.NadirAltitude;
    OrbitNadirTiltAngle = Config.NadirTiltAngle;
    bOrbitIncludeOblique = Config.bIncludeOblique;
    OrbitObliqueRings   = Config.NumObliqueRings;
    bOrbitSideOrbit     = Config.bSideOrbit;
    CaptureTickInterval = Config.CaptureTickInterval;
    PoseWarmupFrames    = Config.PoseWarmupFrames;

    OrbitMarginValue         = OrbitMargin;
    OrbitStartHeightValue    = OrbitStartHeight;
    OrbitCameraHFOVValue     = OrbitCameraHFOV;
    OrbitHOverlapValue       = OrbitHOverlap;
    OrbitVOverlapValue       = OrbitVOverlap;
    OrbitSurveyOverlapValue  = OrbitSurveyOverlap;
    OrbitNadirAltValue       = OrbitNadirAlt;
    OrbitNadirTiltValue      = OrbitNadirTiltAngle;
    OrbitObliqueRingsValue   = OrbitObliqueRings;
    CaptureTickIntervalValue = CaptureTickInterval;
    PoseWarmupFramesValue    = PoseWarmupFrames;

    if (!Config.OutputDirectory.IsEmpty())
    {
        OutputDirectory = Config.OutputDirectory;
    }
    if (OutputDirectory.IsEmpty())
    {
        OutputDirectory = GetVCCSimOutputRoot() / TEXT("Dataset");
    }
    if (OutputDirTextBox.IsValid())
    {
        OutputDirTextBox->SetText(FText::FromString(OutputDirectory));
    }
    if (!Config.SceneName.IsEmpty())
    {
        SceneName = Config.SceneName;
        if (SceneNameTextBox.IsValid())
        {
            SceneNameTextBox->SetText(FText::FromString(SceneName));
        }
    }

    GTTextureResolution = Config.GTTextureResolution;
    bOutputImages       = Config.bOutputImages;
    bOutputMesh         = Config.bOutputMesh;
    bUseCaptureReuse    = Config.bUseCaptureReuse;

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

void FVCCSimPanelPathImageCapture::SaveToConfigManager() const
{
    FVCCSimConfigManager::FPathImageCaptureConfig Config;
    Config.Margin              = OrbitMargin;
    Config.StartHeight         = OrbitStartHeight;
    Config.CameraHFOV          = OrbitCameraHFOV;
    Config.HOverlap            = OrbitHOverlap;
    Config.VOverlap            = OrbitVOverlap;
    Config.SurveyHOverlap      = OrbitSurveyOverlap;
    Config.NadirAltitude       = OrbitNadirAlt;
    Config.NadirTiltAngle      = OrbitNadirTiltAngle;
    Config.bIncludeOblique     = bOrbitIncludeOblique;
    Config.NumObliqueRings     = OrbitObliqueRings;
    Config.bSideOrbit          = bOrbitSideOrbit;
    Config.CaptureTickInterval = CaptureTickInterval;
    Config.PoseWarmupFrames    = PoseWarmupFrames;

    Config.OutputDirectory = OutputDirectory;
    Config.SceneName       = SceneName;

    Config.GTTextureResolution = GTTextureResolution;
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

    FVCCSimConfigManager::Get().SetPathImageCaptureConfig(Config);
}

// ============================================================================
// POSE GENERATION AND MANAGEMENT
// ============================================================================

FReply FVCCSimPanelPathImageCapture::OnGeneratePosesClicked()
{
    if (bGenerationInProgress) return FReply::Handled();

    bGenerationInProgress = true;
    HidePathVisualization();
    bPathVisualized = false;
    bPathNeedsUpdate = false;

    GeneratePosesAroundTarget();
    return FReply::Handled();
}

void FVCCSimPanelPathImageCapture::GeneratePosesAroundTarget()
{
    auto FailCleanup = [this](const TCHAR* Msg)
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("%s"), Msg);
        bGenerationInProgress = false;
    };

    if (!SelectionManager.IsValid() || !SelectionManager.Pin()->GetSelectedFlashPawn().IsValid())
        return FailCleanup(TEXT("No FlashPawn selected"));

    const TArray<FString> TargetLabels = SelectionManager.Pin()->GetEnabledTargetActorLabels();
    if (TargetLabels.IsEmpty())
        return FailCleanup(TEXT("No enabled target actors in the Object Selection list"));

    UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
    if (!World)
        return FailCleanup(TEXT("Editor world is not available"));

    FPathGenerator::FConformalOrbitParams Params;

    TMap<FString, AActor*> LabelMap;
    for (TActorIterator<AActor> It(World); It; ++It)
    {
        if (AActor* A = *It)
            LabelMap.Add(A->GetActorLabel(), A);
    }

    // Resolve enabled labels to actors; FPathGenerator::DetectBuildings then drops ground and
    // clusters the rest by oriented-box adjacency so each building is orbited as one unit.
    TArray<AActor*> EnabledActors;
    EnabledActors.Reserve(TargetLabels.Num());
    for (const FString& Label : TargetLabels)
        if (AActor** Found = LabelMap.Find(Label))
            if (*Found) EnabledActors.Add(*Found);

    if (EnabledActors.Num() == 0)
        return FailCleanup(TEXT("No target actors selected"));

    // Buildings (geometric subset) get facade orbits; the whole target set gets the region survey.
    FPathGenerator::FBuildingDetectParams BParams;
    if (TSharedPtr<FVCCSimPanelSelection> SM = SelectionManager.Pin())
    {
        BParams.MinBuildingHeight = SM->GetMinBuildingHeight();
        BParams.MinBuildingFootprint = SM->GetMinBuildingFootprint();
        BParams.ConnectGap = SM->GetConnectGap();
        BParams.ForcedGroundActors = SM->GetForcedGroundActors(World);
    }
    Params.Buildings = FPathGenerator::DetectBuildingsCached(World, EnabledActors, BParams);
    Params.SurveyTargets = EnabledActors;

    Params.World = World;
    Params.Margin = OrbitMargin;
    Params.StartHeight = OrbitStartHeight;
    Params.CameraHFOV = OrbitCameraHFOV;
    Params.HOverlap = OrbitHOverlap;
    Params.VOverlap = OrbitVOverlap;
    Params.SurveyHOverlap = OrbitSurveyOverlap;
    Params.bIncludeOblique = bOrbitIncludeOblique;
    Params.NadirAltitude = OrbitNadirAlt;
    Params.NadirTiltAngle = OrbitNadirTiltAngle;
    Params.NumObliqueRings = OrbitObliqueRings;
    Params.bSideOrbit = bOrbitSideOrbit;

    TArray<URGBCameraComponent*> RGBCameras;
    SelectionManager.Pin()->GetSelectedFlashPawn()->GetComponents<URGBCameraComponent>(RGBCameras);
    if (RGBCameras.Num() > 0)
    {
        const std::pair<int32, int32> Size = RGBCameras[0]->GetImageSize();
        Params.CameraResolution = FIntPoint(Size.first, Size.second);
    }

    TWeakObjectPtr<AFlashPawn> FlashPawnWeak = SelectionManager.Pin()->GetSelectedFlashPawn();
    TWeakObjectPtr<AVCCSimLookAtPath> LookAtWeak  = SelectionManager.Pin()->GetSelectedLookAtPath();
    
    PathGenerator->GenerateConformalOrbit(Params, FPathGenerator::FOnPathGenerated::CreateLambda(
        [this, FlashPawnWeak, LookAtWeak](const FPathGenerator::FGeneratedPath& Path)
        {
            bGenerationInProgress = false;
            if (!FlashPawnWeak.IsValid())
            {
                UE_LOG(LogPathImageCapture, Warning, TEXT("FlashPawn became invalid after path generation."));
                return;
            }

            FlashPawnWeak->Modify();
            FlashPawnWeak->SetPathPanel(Path.Positions, Path.Rotations);
            FlashPawnWeak->MoveTo(0);

            // The capture path lives in the FlashPawn (PendingPositions). The LookAtPath
            // spline is NOT consumed by capture (GetSamplePoses has no callers), and writing
            // one spline control point per pose makes the editor spline visualizer tank the
            // framerate on large paths. So keep the spline EMPTY and use "Show Path" (instanced
            // meshes, one draw call per role) for the preview — only the look-at target is set.
            if (AVCCSimLookAtPath* LookAt = LookAtWeak.Get())
            {
                if (LookAt->Spline)
                {
                    LookAt->Modify();
                    LookAt->Spline->Modify();
                    LookAt->Spline->ClearSplinePoints(true);
                }
                // Free (per-pose) orientation instead of looking at a single centre point —
                // the capture uses the FlashPawn's per-pose orbit rotations, so a look-at
                // target is unused; default to FreeOrientation for any manual look-at use.
                LookAt->OrientationMode = EOrientationMode::FreeOrientation;
            }

            UE_LOG(LogPathImageCapture, Log, TEXT("Conformal orbit generated with %d poses."), Path.Positions.Num());
        }
    ));
}

FReply FVCCSimPanelPathImageCapture::OnLoadPoseClicked()
{
    LoadPredefinedPose();
    return FReply::Handled();
}

FReply FVCCSimPanelPathImageCapture::OnSavePoseClicked()
{
    SaveGeneratedPose();
    return FReply::Handled();
}

void FVCCSimPanelPathImageCapture::LoadPredefinedPose()
{
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    if (SelectionManager.IsValid())
    {
        SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    }
    
    if (!SelectedFlashPawn.IsValid())
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("No FlashPawn selected"));
        return;
    }
    
    // Open file dialog to select pose file
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (DesktopPlatform)
    {
        TArray<FString> OpenFilenames;
        FString ExtensionStr = TEXT("Pose Files (*.txt)|*.txt");
        
        bool bOpened = DesktopPlatform->OpenFileDialog(
            FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr),
            TEXT("Load Pose File"),
            FPaths::ProjectSavedDir(),
            TEXT(""),
            *ExtensionStr,
            EFileDialogFlags::None,
            OpenFilenames
        );
        
        if (bOpened && OpenFilenames.Num() > 0)
        {
            FString SelectedFile = OpenFilenames[0];
            
            // Read file content
            TArray<FString> FileLines;
            if (FFileHelper::LoadFileToStringArray(FileLines, *SelectedFile))
            {
                TArray<FVector> Positions;
                TArray<FRotator> Rotations;
                
                for (const FString& Line : FileLines)
                {
                    if (Line.IsEmpty() || Line.StartsWith(TEXT("#")))
                    {
                        continue;
                    }

                    TArray<FString> Values;
                    Line.ParseIntoArray(Values, TEXT(" "), true);

                    if (Values.Num() == 8)
                    {
                        float X = FCString::Atof(*Values[1]);
                        float Y = FCString::Atof(*Values[2]);
                        float Z = FCString::Atof(*Values[3]);
                        float Qx = FCString::Atof(*Values[4]);
                        float Qy = FCString::Atof(*Values[5]);
                        float Qz = FCString::Atof(*Values[6]);
                        float Qw = FCString::Atof(*Values[7]);

                        Positions.Add(FVector(X, Y, Z));

                        FQuat Quaternion(Qx, Qy, Qz, Qw);
                        Quaternion.Normalize();
                        FRotator Rotation = Quaternion.Rotator();
                        Rotations.Add(Rotation);
                    }
                    else
                    {
                        UE_LOG(LogPathImageCapture, Warning, TEXT("Invalid pose line format (expected 8 values): %s"), *Line);
                    }
                }
                
                if (Positions.Num() > 0 && Positions.Num() == Rotations.Num())
                {
                    // Set the path on the FlashPawn
                    SelectedFlashPawn->Modify();
                    SelectedFlashPawn->SetPathPanel(Positions, Rotations);
                    
                    // Clean up any existing visualization
                    HidePathVisualization();
                    
                    // Allow path visualization after loading
                    bPathVisualized = false;
                    bPathNeedsUpdate = false;
                    
                    UE_LOG(LogPathImageCapture, Log, TEXT("Successfully loaded %d "
                                              "poses from file"), Positions.Num());
                }
                else
                {
                    UE_LOG(LogPathImageCapture, Warning, TEXT("Failed to parse pose file: "
                                                  "Invalid format or empty file"));
                }
            }
            else
            {
                UE_LOG(LogPathImageCapture, Warning, TEXT("Failed to load file"));
            }
        }
    }
}

void FVCCSimPanelPathImageCapture::SaveGeneratedPose()
{
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    if (SelectionManager.IsValid())
    {
        SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    }
    
    if (!SelectedFlashPawn.IsValid())
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("No FlashPawn selected"));
        return;
    }
    
    // Check if there are poses to save
    int32 PoseCount = SelectedFlashPawn->GetPoseCount();
    if (PoseCount <= 0)
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("No poses to save"));
        return;
    }
    
    // Open file dialog to select save location
    IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
    if (DesktopPlatform)
    {
        TArray<FString> SaveFilenames;
        FString ExtensionStr = TEXT("Pose Files (*.txt)|*.txt");
        
        bool bSaved = DesktopPlatform->SaveFileDialog(
            FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr),
            TEXT("Save Pose File"),
            FPaths::ProjectSavedDir(),
            TEXT("poses.txt"),
            *ExtensionStr,
            EFileDialogFlags::None,
            SaveFilenames
        );
        
        if (bSaved && SaveFilenames.Num() > 0)
        {
            FString SelectedFile = SaveFilenames[0];
            
            // Ensure the file has .txt extension
            if (!SelectedFile.EndsWith(TEXT(".txt")))
            {
                SelectedFile += TEXT(".txt");
            }
            
            TArray<FVector> Positions;
            TArray<FRotator> Rotations;
            SelectedFlashPawn->GetCurrentPath(Positions, Rotations);
            ApplyCameraLocalTransform(SelectedFlashPawn.Get(), Positions, Rotations);
            WritePosesToFile(Positions, Rotations, SelectedFile);

            const FString OutDir = FPaths::GetPath(SelectedFile);
            WriteIntrinsicsJsonFromFlashPawn(SelectedFlashPawn.Get(), OutDir);
            WriteLightingJsonFromWorld(
                GEditor ? GEditor->GetEditorWorldContext().World() : nullptr,
                OutDir);
        }
    }
}

void FVCCSimPanelPathImageCapture::WritePosesToFile(
    const TArray<FVector>& Positions,
    const TArray<FRotator>& Rotations,
    const FString& FilePath)
{
    FString FileContent;
    FileContent += TEXT("# UE coordinate system poses (left-handed, cm)\n");
    FileContent += TEXT("# Coordinate axes: +X forward, +Y right, +Z up\n");
    FileContent += TEXT("# Format: Timestamp X Y Z Qx Qy Qz Qw\n");
    FileContent += TEXT("# Quaternion order: [x, y, z, w] (UE format, scalar last)\n");
    FileContent += TEXT("# Timestamp: Sequential pseudo timestamps for pose ordering\n");

    for (int32 i = 0; i < Positions.Num(); ++i)
    {
        const FQuat Quat = Rotations[i].Quaternion();
        FileContent += FString::Printf(
            TEXT("%.1f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n"),
            static_cast<double>(i),
            Positions[i].X, Positions[i].Y, Positions[i].Z,
            Quat.X, Quat.Y, Quat.Z, Quat.W
        );
    }

    if (!FFileHelper::SaveStringToFile(FileContent, *FilePath))
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("Failed to save poses to %s"), *FilePath);
    }
}

// ============================================================================
// PATH VISUALIZATION
// ============================================================================

FReply FVCCSimPanelPathImageCapture::OnTogglePathVisualizationClicked()
{
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    if (SelectionManager.IsValid())
    {
        SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    }
    
    if (!SelectedFlashPawn.IsValid())
    {
        return FReply::Handled();
    }
    
    // Toggle the visualization state
    bPathVisualized = !bPathVisualized;

    if (bPathVisualized)
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("Showing path visualization..."));
        ShowPathVisualization();
    }
    else
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("Hiding path visualization..."));
        HidePathVisualization();
    }

    VisualizePathButton->SetButtonStyle(bPathVisualized ? 
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") : 
        &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
    
    return FReply::Handled();
}

void FVCCSimPanelPathImageCapture::UpdatePathVisualization()
{
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    if (SelectionManager.IsValid())
    {
        SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    }
    
    if (!SelectedFlashPawn.IsValid()) return;
    
    const TArray<FVector> Positions = SelectedFlashPawn->PendingPositions;
    const TArray<FRotator> Rotations = SelectedFlashPawn->PendingRotations;

    if (Positions.Num() == 0 || Positions.Num() != Rotations.Num())
    {
        bPathVisualized = false;
        return;
    }

    // Destroy any existing preview first: GenerateVisibleElements spawns a fresh "PathVisualization"
    // actor (with several instanced-mesh components + dynamic materials) every call, so re-showing or
    // regenerating without this leaves the old ones stacked in the world until the next Hide.
    HidePathVisualization();

    PathVisualizationActor = UTrajectoryViewer::GenerateVisibleElements(
        GEditor->GetEditorWorldContext().World(),
        Positions,
        Rotations,
        5.f,     // Path width
        15.0f,   // Cone size
        75.0f    // Cone length
    );
        
    if (!PathVisualizationActor.IsValid())
    {
        bPathVisualized = false;
        return;
    }

    PathVisualizationActor->Tags.Add(FName("NotSMActor"));
    bPathNeedsUpdate = false;
}

void FVCCSimPanelPathImageCapture::ShowPathVisualization()
{
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    if (SelectionManager.IsValid())
    {
        SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    }
    
    if (SelectedFlashPawn.IsValid())
    {
        UpdatePathVisualization();
    }
}

void FVCCSimPanelPathImageCapture::HidePathVisualization()
{
    if (GEditor && GEditor->GetEditorWorldContext().World())
    {
        UWorld* World = GEditor->GetEditorWorldContext().World();
        FlushPersistentDebugLines(World);
        
        // Clean up any PathVisualization actors in the world
        for (TActorIterator<AActor> ActorIterator(World); ActorIterator; ++ActorIterator)
        {
            AActor* Actor = *ActorIterator;
            if (Actor && (Actor->GetActorLabel().Contains(TEXT("PathVisualization")) || 
                         Actor->Tags.Contains(FName("VCCSimPathViz"))))
            {
                World->DestroyActor(Actor);
            }
        }
    }
    
    if (PathVisualizationActor.IsValid() && GEditor)
    {
        if (UWorld* World = GEditor->GetEditorWorldContext().World())
        {
            World->DestroyActor(PathVisualizationActor.Get());
        }
        PathVisualizationActor.Reset();
    }
}

// ============================================================================
// IMAGE CAPTURE OPERATIONS
// ============================================================================

FReply FVCCSimPanelPathImageCapture::OnCaptureImagesClicked()
{
    const bool bChanged = EnsureGameView();
    CaptureImageFromCurrentPose();
    RestoreGameView(bChanged);
    return FReply::Handled();
}

void FVCCSimPanelPathImageCapture::CaptureImageFromCurrentPose()
{
    if (!ImageCaptureService.IsValid())
    {
        UE_LOG(LogPathImageCapture, Error, TEXT("ImageCaptureService is not valid."));
        return;
    }

    // Create a directory for saving images if it doesn't exist yet
    if (SaveDirectory.IsEmpty())
    {
        SaveDirectory = GetVCCSimOutputRoot() / TEXT("VCCSimCaptures") / GetTimestampedFilename();
        IFileManager::Get().MakeDirectory(*SaveDirectory, true);
    }

    bool bAnyCaptured = false;
    int32 PoseIndex = -1;
    if (SelectionManager.IsValid() && SelectionManager.Pin()->GetSelectedFlashPawn().IsValid())
    {
        PoseIndex = SelectionManager.Pin()->GetSelectedFlashPawn()->GetCurrentIndex();
    }

    // Direct-viewport RGB capture renders throwaway frames before reading back so the viewport's
    // temporal occlusion culling / Lumen / streaming converge to the jumped-to pose; drive that count
    // from the panel's Warmup setting.
    ImageCaptureService->SetViewportWarmupFrames(PoseWarmupFrames);
    ImageCaptureService->CaptureImageFromCurrentPose(
        PoseIndex, SaveDirectory, bAnyCaptured, bSessionRgbOnly);
}

bool FVCCSimPanelPathImageCapture::CanRunDatasetCapture(FString& OutReason) const
{
    if (bAutoCaptureInProgress)
    {
        OutReason = TEXT("A capture is already in progress.");
        return false;
    }

    TSharedPtr<FVCCSimPanelSelection> Sel = SelectionManager.Pin();
    if (!Sel.IsValid() || !Sel->GetSelectedFlashPawn().IsValid())
    {
        OutReason = TEXT("No FlashPawn selected.");
        return false;
    }
    if (Sel->GetSelectedFlashPawn()->GetPoseCount() <= 0)
    {
        OutReason = TEXT("FlashPawn has no path. Generate or load poses first.");
        return false;
    }
    if (!Sel->HasAnyActiveCamera())
    {
        OutReason = TEXT("No active camera selected in Object Selection.");
        return false;
    }
    return true;
}

FString FVCCSimPanelPathImageCapture::ComputePathPoseKey() const
{
    if (!SelectionManager.IsValid()) return FString();
    TWeakObjectPtr<AFlashPawn> Pawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    if (!Pawn.IsValid() || Pawn->GetPoseCount() <= 0) return FString();

    TArray<FVector> Positions;
    TArray<FRotator> Rotations;
    Pawn->GetCurrentPath(Positions, Rotations);
    ApplyCameraLocalTransform(Pawn.Get(), Positions, Rotations);

    FString Canon = FString::Printf(TEXT("n=%d"), Positions.Num());
    for (int32 i = 0; i < Positions.Num(); ++i)
    {
        const FVector& P = Positions[i];
        const FRotator& R = Rotations.IsValidIndex(i) ? Rotations[i] : FRotator::ZeroRotator;
        Canon += FString::Printf(TEXT(";%.3f,%.3f,%.3f,%.3f,%.3f,%.3f"),
            P.X, P.Y, P.Z, R.Pitch, R.Yaw, R.Roll);
    }
    return FMD5::HashAnsiString(*Canon);
}

bool FVCCSimPanelPathImageCapture::StartCaptureSession(
    const FString& TargetDirectory,
    bool bRgbOnly,
    FOnCaptureSessionComplete OnComplete)
{
    if (bAutoCaptureInProgress)
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("Capture session already in progress"));
        return false;
    }

    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    if (SelectionManager.IsValid())
    {
        SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    }

    if (!SelectedFlashPawn.IsValid())
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("No FlashPawn selected"));
        return false;
    }
    if (SelectedFlashPawn->GetPoseCount() <= 0)
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("FlashPawn has no path to capture"));
        return false;
    }
    if (!ImageCaptureService.IsValid())
    {
        UE_LOG(LogPathImageCapture, Error, TEXT("ImageCaptureService is not valid."));
        return false;
    }

    SaveDirectory = TargetDirectory;
    IFileManager::Get().MakeDirectory(*SaveDirectory, true);

    TArray<FVector> Positions;
    TArray<FRotator> Rotations;
    SelectedFlashPawn->GetCurrentPath(Positions, Rotations);
    ApplyCameraLocalTransform(SelectedFlashPawn.Get(), Positions, Rotations);
    WritePosesToFile(Positions, Rotations, SaveDirectory / TEXT("poses.txt"));
    WriteIntrinsicsJsonFromFlashPawn(SelectedFlashPawn.Get(), SaveDirectory);
    UWorld* CaptureWorld = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
    WriteLightingJsonFromWorld(CaptureWorld, SaveDirectory);

    const FString SkyExrPath = SaveDirectory / TEXT("sky.exr");
    if (CaptureWorld && IFileManager::Get().FileSize(*SkyExrPath) <= 0)
    {
        FSkyHDRICapture::CaptureSkyEquirect(CaptureWorld, SkyExrPath);
    }

    bSessionRgbOnly = bRgbOnly;
    SessionCompleteDelegate = MoveTemp(OnComplete);
    bDrainingCaptureJobs = false;
    bSessionCancelled = false;
    PoseWarmupRemaining = PoseWarmupFrames;
    bGameViewChangedForCapture = EnsureGameView();
    if (ImageCaptureService.IsValid())
    {
        ImageCaptureService->BeginViewportCaptureSession(SelectedFlashPawn.Get());
    }
    bAutoCaptureInProgress = true;

    // Resume support: skip poses already fully captured on disk. A pose counts as done only when every
    // expected channel file for it exists and is non-empty. Force re-capture of the highest existing
    // pose — a previous run may have crashed mid-write, leaving that pose's last file truncated. Start
    // the session at the first missing pose so a large interrupted capture continues instead of redoing.
    int32 ResumeStartIndex = 0;
    {
        const int32 PoseCount = SelectedFlashPawn->GetPoseCount();
        SessionCompleted = ImageCaptureService->ComputeCompletedPoses(
            SelectedFlashPawn.Get(), SaveDirectory, PoseCount, bSessionRgbOnly);

        int32 HighestDone = INDEX_NONE, NumDone = 0;
        for (int32 i = 0; i < SessionCompleted.Num(); ++i)
        {
            if (SessionCompleted[i]) { ++NumDone; HighestDone = i; }
        }
        if (HighestDone != INDEX_NONE)
        {
            SessionCompleted[HighestDone] = false;   // re-capture the last present pose (truncation guard)
            --NumDone;
        }
        while (ResumeStartIndex < SessionCompleted.Num() && SessionCompleted[ResumeStartIndex])
        {
            ++ResumeStartIndex;
        }

        if (NumDone > 0 || ResumeStartIndex > 0)
        {
            UE_LOG(LogPathImageCapture, Log,
                TEXT("Resuming capture in %s: %d/%d poses already present, starting at pose %d (re-capturing pose %d)"),
                *SaveDirectory, NumDone, PoseCount, ResumeStartIndex, HighestDone);
        }
    }

    SelectedFlashPawn->MoveTo(ResumeStartIndex);

    // Set up a timer to check if the FlashPawn is ready for capture
    GEditor->GetTimerManager()->SetTimer(
        AutoCaptureTimerHandle,
        FTimerDelegate::CreateLambda([this]() { TickCaptureSession(); }),
        CaptureTickInterval,
        true
    );

    return true;
}

void FVCCSimPanelPathImageCapture::TickCaptureSession()
{
    TWeakObjectPtr<AFlashPawn> SelectedFlashPawn;
    if (SelectionManager.IsValid())
    {
        SelectedFlashPawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    }

    if (!bAutoCaptureInProgress || !SelectedFlashPawn.IsValid())
    {
        UE_LOG(LogPathImageCapture, Warning,
            TEXT("Capture session aborted: FlashPawn is no longer valid"));
        FinishCaptureSession(false);
        return;
    }

    if (bDrainingCaptureJobs)
    {
        if (ImageCaptureService->GetInFlightCount() == 0)
        {
            FinishCaptureSession(!bSessionCancelled);
        }
        return;
    }

    // Check if the FlashPawn is ready to capture
    if (SelectedFlashPawn->IsReady())
    {
        // Resume: a pose already present on disk (from an earlier interrupted run) is skipped entirely —
        // no warm-up, no capture — just advance. Common case jumps straight past it via MoveTo at start;
        // this also covers any interior gap. Last-pose handling mirrors the capture branch below.
        const int32 CurIdx = SelectedFlashPawn->GetCurrentIndex();
        if (SessionCompleted.IsValidIndex(CurIdx) && SessionCompleted[CurIdx])
        {
            const bool bWasLastPose = CurIdx == SelectedFlashPawn->GetPoseCount() - 1;
            SelectedFlashPawn->MoveToNext();
            PoseWarmupRemaining = PoseWarmupFrames;
            if (bWasLastPose)
            {
                bDrainingCaptureJobs = true;
            }
            return;
        }

        // Per-pose warm-up: after the camera jumps to a pose, the SceneCapture channels (BaseColor /
        // Normal / Depth / MaterialProperties) need a few throwaway renders for temporal occlusion
        // culling / Lumen / exposure history to converge — otherwise they capture stale or incomplete.
        // Direct-viewport RGB does its own warm-up draws inside CaptureRGBFromViewport; both use the same
        // PoseWarmupFrames count.
        if (PoseWarmupRemaining > 0)
        {
            TArray<UCameraBaseComponent*> Cameras;
            SelectedFlashPawn->GetComponents<UCameraBaseComponent>(Cameras);
            for (UCameraBaseComponent* Camera : Cameras)
            {
                if (Camera)
                {
                    Camera->WarmupCapture();
                }
            }
            --PoseWarmupRemaining;
            return;
        }

        CaptureImageFromCurrentPose();

        const int32 DoneIdx = SelectedFlashPawn->GetCurrentIndex();
        const int32 Total = SelectedFlashPawn->GetPoseCount();
        const bool bWasLastPose = DoneIdx == Total - 1;

        // Periodic progress to the Output Log. The actual resumable progress is the image files on disk
        // (the resume scan reads them), so this is purely visibility — no separate progress file needed.
        if (bWasLastPose || (DoneIdx % 25) == 0)
        {
            UE_LOG(LogPathImageCapture, Log, TEXT("Capture progress: pose %d/%d"), DoneIdx + 1, Total);
        }

        SelectedFlashPawn->MoveToNext();
        PoseWarmupRemaining = PoseWarmupFrames;   // re-arm warm-up for the next pose

        if (bWasLastPose)
        {
            bDrainingCaptureJobs = true;
        }
    }
    else
    {
        SelectedFlashPawn->MoveForward();
    }
}

void FVCCSimPanelPathImageCapture::FinishCaptureSession(bool bSuccess)
{
    if (bSuccess)
    {
        UE_LOG(LogPathImageCapture, Log, TEXT("Capture session complete: %s"), *SaveDirectory);
    }
    else
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("Capture session ended early: %s"), *SaveDirectory);
    }

    if (GEditor)
    {
        GEditor->GetTimerManager()->ClearTimer(AutoCaptureTimerHandle);
    }
    bAutoCaptureInProgress = false;
    bDrainingCaptureJobs = false;
    bSessionCancelled = false;
    PoseWarmupRemaining = 0;
    SessionCompleted.Reset();
    bSessionRgbOnly = false;
    SaveDirectory.Empty();
    if (ImageCaptureService.IsValid())
    {
        ImageCaptureService->EndViewportCaptureSession();
    }
    RestoreGameView(bGameViewChangedForCapture);
    bGameViewChangedForCapture = false;

    if (AutoCaptureButton.IsValid())
    {
        AutoCaptureButton->SetButtonStyle(&FAppStyle::Get().
            GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
    }

    FOnCaptureSessionComplete Delegate = MoveTemp(SessionCompleteDelegate);
    SessionCompleteDelegate.Unbind();
    Delegate.ExecuteIfBound(bSuccess);
}

void FVCCSimPanelPathImageCapture::StartAutoCapture()
{
    const FString TargetDirectory =
        GetVCCSimOutputRoot() / TEXT("VCCSimCaptures") / GetTimestampedFilename();
    StartCaptureSession(TargetDirectory, false, FOnCaptureSessionComplete());
}

void FVCCSimPanelPathImageCapture::StopAutoCapture()
{
    if (!bAutoCaptureInProgress || bSessionCancelled)
    {
        return;
    }

    bSessionCancelled = true;
    bDrainingCaptureJobs = true;
    UE_LOG(LogPathImageCapture, Log,
        TEXT("Auto-capture stop requested; waiting for pending writes"));
}

bool FVCCSimPanelPathImageCapture::IsCaptureWindowComplete(const FString& Dir, bool bRgbOnly) const
{
    if (!ImageCaptureService.IsValid() || !SelectionManager.IsValid())
    {
        return false;
    }
    TWeakObjectPtr<AFlashPawn> Pawn = SelectionManager.Pin()->GetSelectedFlashPawn();
    if (!Pawn.IsValid())
    {
        return false;
    }
    const int32 PoseCount = Pawn->GetPoseCount();
    if (PoseCount <= 0)
    {
        return false;
    }
    const TArray<bool> Done = ImageCaptureService->ComputeCompletedPoses(
        Pawn.Get(), Dir, PoseCount, bRgbOnly);
    if (Done.Num() != PoseCount)
    {
        return false;
    }
    for (bool b : Done)
    {
        if (!b) return false;
    }
    return true;
}

// ============================================================================
// DATASET CONFIGURATION
// ============================================================================

FReply FVCCSimPanelPathImageCapture::OnBrowseOutputDirClicked()
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
        SaveToConfigManager();
    }

    return FReply::Handled();
}

// ============================================================================
// LIGHTING SCHEDULE
// ============================================================================

FReply FVCCSimPanelPathImageCapture::OnApplyLightingClicked(int32 Index)
{
    if (Index < 0 || Index >= NumLightingConditions || !LightingManager.IsValid()) return FReply::Handled();
    LightingManager->ApplyLightingCondition(LightingElevation[Index], LightingAzimuth[Index]);
    UE_LOG(LogPathImageCapture, Log, TEXT("Lighting condition %d applied: Elev=%.1f Az=%.1f"),
        Index + 1, LightingElevation[Index], LightingAzimuth[Index]);
    return FReply::Handled();
}

FReply FVCCSimPanelPathImageCapture::OnCalculateSunPositionClicked()
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

    UE_LOG(LogPathImageCapture, Log, TEXT("Sun position calculated & applied: Elev=%.1f Az=%.1f"),
        SunCalcElevation, SunCalcAzimuth);

    return FReply::Handled();
}

FReply FVCCSimPanelPathImageCapture::OnFillFromSunPositionClicked()
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

    UE_LOG(LogPathImageCapture, Log,
        TEXT("Sun position filled into slot %d: Elevation=%.1f°  Azimuth=%.1f°"),
        SunCalcFillSlot, SunCalcElevation, SunCalcAzimuth);

    return FReply::Handled();
}

// ============================================================================
// DATASET CAPTURE
// ============================================================================

FString FVCCSimPanelPathImageCapture::GetDatasetCapturesRoot() const
{
    return FPaths::Combine(OutputDirectory, SceneName, TEXT("captures"));
}

FString FVCCSimPanelPathImageCapture::MakeNextCaptureDirectory() const
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

FReply FVCCSimPanelPathImageCapture::OnCaptureDatasetClicked()
{
    if (bDatasetCaptureInProgress)
    {
        StopAutoCapture();
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
            UE_LOG(LogPathImageCapture, Log,
                TEXT("Mesh-only: gt_materials reused from %s (manifest reference)"), *Entry.GtMaterialsOwner);
        }
        else if (!StartGTMaterialExport(MeshOnlyDir / TEXT("gt_materials")))
        {
            UE_LOG(LogPathImageCapture, Warning, TEXT("Mesh-only gt_materials export could not start"));
        }

        Manifest.AddOrUpdate(FPaths::GetCleanFilename(MeshOnlyDir), Entry);
        Manifest.Save();
        return FReply::Handled();
    }

    FString Reason;
    if (!CanRunDatasetCapture(Reason))
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
            ActiveCheckpoint.PoseKey            = ComputePathPoseKey();
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

        UE_LOG(LogPathImageCapture, Log,
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
        ActiveCheckpoint.PoseKey            = ComputePathPoseKey();
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

bool FVCCSimPanelPathImageCapture::DecideAndStartCapture(const FString& CaptureDir)
{
    UWorld* World = GEditor ? GEditor->GetEditorWorldContext().World() : nullptr;
    TSharedPtr<FVCCSimPanelSelection> Sel = SelectionManager.Pin();

    const FString PoseKey  = ComputePathPoseKey();
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

    const bool bStarted = StartCaptureSession(
        CaptureDir,
        bRgbOnly,
        FOnCaptureSessionComplete::CreateLambda(
            [this, CaptureDir](bool bSuccess)
            {
                OnDatasetCaptureFinished(bSuccess, CaptureDir);
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
            UE_LOG(LogPathImageCapture, Log,
                TEXT("Capture %s: RGB-only (GT image channels reused from %s)"),
                *PendingCaptureName, *ViewGtOwner);
        }
        else
        {
            UE_LOG(LogPathImageCapture, Log,
                TEXT("Capture %s: full GT capture (owner)"), *PendingCaptureName);
        }
    }
    return bStarted;
}

void FVCCSimPanelPathImageCapture::StartNextBatchCapture()
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

    UE_LOG(LogPathImageCapture, Log,
        TEXT("Batch capture (lighting %d, Elev=%.1f Az=%.1f): %s"),
        Slot + 1, LightingElevation[Slot], LightingAzimuth[Slot], *CaptureDir);
}

void FVCCSimPanelPathImageCapture::OnDatasetCaptureFinished(bool bSuccess, FString CaptureDirectory)
{
    if (!bSuccess)
    {
        // Keep the partial output AND the resume checkpoint on disk so the run can be continued via the
        // Resume button (this is the whole point — a large dataset interrupted by Stop or a crash must
        // not lose what it already captured). Only the live (in-memory) run state is reset here.
        UE_LOG(LogPathImageCapture, Warning,
            TEXT("Dataset capture stopped or failed; partial output kept for resume: %s"), *CaptureDirectory);
        LightingCaptureQueue.Reset();
        bBatchCapture = false;
        bDatasetCaptureInProgress = false;
        return;
    }

    UE_LOG(LogPathImageCapture, Log, TEXT("Dataset capture complete: %s"), *CaptureDirectory);
    FVCCSimUIHelpers::ShowNotification(
        FString::Printf(TEXT("Dataset capture complete: %s"), *CaptureDirectory), false);

    TSharedPtr<FVCCSimPanelSelection> Sel = SelectionManager.Pin();
    if (!PendingReuseEntry.GtMaterialsOwner.IsEmpty())
    {
        UE_LOG(LogPathImageCapture, Log,
            TEXT("gt_materials reused from %s (manifest reference); export skipped"),
            *PendingReuseEntry.GtMaterialsOwner);
    }
    else if (!bOutputMesh)
    {
        UE_LOG(LogPathImageCapture, Log, TEXT("Mesh output disabled, gt_materials export skipped"));
    }
    else if (!Sel.IsValid() || !Sel->HasEnabledTargetActors())
    {
        UE_LOG(LogPathImageCapture, Log, TEXT("No enabled target actors, gt_materials export skipped"));
    }
    else if (!StartGTMaterialExport(CaptureDirectory / TEXT("gt_materials")))
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("gt_materials export could not start"));
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

bool FVCCSimPanelPathImageCapture::HasResumableCapture() const
{
    return !bDatasetCaptureInProgress
        && FCaptureSessionCheckpoint::Exists(GetDatasetCapturesRoot());
}

FReply FVCCSimPanelPathImageCapture::OnResumeCaptureClicked()
{
    if (bDatasetCaptureInProgress)
    {
        return FReply::Handled();
    }

    const FString Root = GetDatasetCapturesRoot();
    FCaptureSessionCheckpoint Cp = FCaptureSessionCheckpoint::Load(Root);
    if (!Cp.IsValid())
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("Resume: no resumable capture found in %s"), *Root);
        return FReply::Handled();
    }

    TSharedPtr<FVCCSimPanelSelection> Sel = SelectionManager.Pin();
    AFlashPawn* Pawn = Sel.IsValid() ? Sel->GetSelectedFlashPawn().Get() : nullptr;
    if (!Pawn)
    {
        UE_LOG(LogPathImageCapture, Warning, TEXT("Resume: no FlashPawn selected to drive the capture."));
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
        UE_LOG(LogPathImageCapture, Log,
            TEXT("Resume: restored FlashPawn path from checkpoint (%d poses)."), Cp.PathPositions.Num());
    }

    const FString CurPoseKey = ComputePathPoseKey();
    if (!Cp.PoseKey.IsEmpty() && CurPoseKey != Cp.PoseKey)
    {
        UE_LOG(LogPathImageCapture, Warning,
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
            UE_LOG(LogPathImageCapture, Warning,
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
        if (IsCaptureWindowComplete(Dir, W.bRgbOnly))
        {
            continue;
        }
        if (W.Slot >= 0) LightingCaptureQueue.Add(W.Slot);
        else { bHasSingle = true; SingleDir = Dir; }
    }

    if (LightingCaptureQueue.Num() == 0 && !bHasSingle)
    {
        UE_LOG(LogPathImageCapture, Log,
            TEXT("Resume: every window is already complete; clearing checkpoint."));
        FCaptureSessionCheckpoint::Clear(Root);
        ActiveCheckpoint = FCaptureSessionCheckpoint();
        return FReply::Handled();
    }

    bDatasetCaptureInProgress = true;
    if (LightingCaptureQueue.Num() > 0)
    {
        bBatchCapture = true;
        UE_LOG(LogPathImageCapture, Log,
            TEXT("Resuming dataset capture: %d lighting window(s) remaining."), LightingCaptureQueue.Num());
        StartNextBatchCapture();
    }
    else
    {
        bBatchCapture = false;
        UE_LOG(LogPathImageCapture, Log, TEXT("Resuming single dataset capture: %s"), *SingleDir);
        if (!DecideAndStartCapture(SingleDir))
        {
            bDatasetCaptureInProgress = false;
            UE_LOG(LogPathImageCapture, Warning, TEXT("Resume: failed to start capture."));
        }
    }
    return FReply::Handled();
}

bool FVCCSimPanelPathImageCapture::StartGTMaterialExport(const FString& BaseDir)
{
    if (GTMaterialExporter.IsValid() && GTMaterialExporter->IsExportInProgress())
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

    // gt_materials reuse is decided upstream via captures/reuse.json (DecideAndStartCapture):
    // when reusable, the export is skipped and an owner reference is recorded instead. Reaching
    // here means this capture is the owner, so always run the full export.
    const FString Signature = FGTMaterialExporter::ComputeSignature(
        World, ActorLabels, SceneName, GTTextureResolution);

    if (!GTMaterialExporter.IsValid())
    {
        GTMaterialExporter = MakeShared<FGTMaterialExporter>();
    }

    FVCCSimUIHelpers::ShowNotification(TEXT("Starting GT material export..."));
    GTMaterialExporter->RunExport(ActorLabels, TArray<FGTFoliageExportEntry>(), World, BaseDir,
        SceneName, GTTextureResolution, Signature, FSimpleDelegate());
    return true;
}

// ============================================================================
// UTILITIES
// ============================================================================

FString FVCCSimPanelPathImageCapture::GetTimestampedFilename()
{
    FDateTime Now = FDateTime::Now();
    return FString::Printf(TEXT("%04d-%02d-%02d_%02d-%02d-%02d"),
        Now.GetYear(), Now.GetMonth(), Now.GetDay(),
        Now.GetHour(), Now.GetMinute(), Now.GetSecond());
}

