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

#include "Utils/NanobananaManager.h"
#include "Utils/VCCSimDataConverter.h"
#include "IO/GLTFUtils.h"
#include "Exporters/GLTFExporter.h"
#include "Options/GLTFExportOptions.h"
#include "Utils/VCCSimUIHelpers.h"

// Engine
#include "Engine/SceneCapture2D.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Components/StaticMeshComponent.h"
#include "EngineUtils.h"
#include "RenderingThread.h"
#include "TextureResource.h"

// Materials
#include "Materials/Material.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Materials/MaterialInstanceConstant.h"
#include "Materials/MaterialExpressionVectorParameter.h"
#include "Materials/MaterialExpressionScalarParameter.h"
#include "Materials/MaterialExpressionTextureSampleParameter2D.h"
#include "Materials/MaterialExpressionLinearInterpolate.h"

// Editor / Asset Tools
#include "AssetToolsModule.h"
#include "IAssetTools.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "Factories/MaterialFactoryNew.h"
#include "Factories/MaterialInstanceConstantFactoryNew.h"
#include "Factories/TextureFactory.h"
#include "Misc/PackageName.h"
#include "UObject/SavePackage.h"

// JSON / Files
#include "HAL/FileManager.h"
#include "HAL/PlatformFileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/ScopedSlowTask.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "IImageWrapperModule.h"
#include "IImageWrapper.h"
#include "Framework/Application/SlateApplication.h"
#include "ShaderCompiler.h"

DEFINE_LOG_CATEGORY_STATIC(LogNanobananaManager, Log, All);

namespace
{
    struct FNBClassDef { const TCHAR* Name; uint8 R, G, B; };
    static const FNBClassDef NBClasses[] = {
        { TEXT("glass"),          0, 200, 255 },
        { TEXT("polished_metal"), 220, 220,   0 },
        { TEXT("matte_metal"),    128, 128, 128 },
        { TEXT("concrete"),       200, 140,  80 },
        { TEXT("brick"),          200,  50,  30 },
        { TEXT("painted_wall"),   240, 180, 240 },
        { TEXT("asphalt"),         40,  40,  40 },
        { TEXT("vegetation"),      30, 180,  50 },
    };
    static const int32 NBClassCount = UE_ARRAY_COUNT(NBClasses);
    static const int32 NBTolerance  = 40;
}

FNanobananaManager::FNanobananaManager()  {}
FNanobananaManager::~FNanobananaManager() {}

bool FNanobananaManager::RunProjection(
    const FProjectionParams& InParams,
    FOnNanobananaProgress InOnProgress,
    FOnNanobananaComplete InOnComplete)
{
    if (bIsInProgress)
    {
        UE_LOG(LogNanobananaManager, Warning, TEXT("Projection is already in progress."));
        return false;
    }
    bIsInProgress = true;
    Params        = InParams;
    OnProgress    = InOnProgress;
    OnComplete    = InOnComplete;
    OutputDir     = FPaths::GetPath(Params.ManifestFile);
    RunOnGameThread();
    return true;
}

void FNanobananaManager::Cancel()
{
    UE_LOG(LogNanobananaManager, Warning, TEXT("Cancel() called – use the progress dialog cancel button to abort."));
}

// ============================================================================
// GLTF Slot PNG Info Loading (Phase A / Phase B data source)
// ============================================================================

bool FNanobananaManager::LoadSlotPngInfos(
    const FString&        ActorDir,
    TArray<FSlotPngInfo>& OutSlots,
    int32&                OutGltfPrimCount,
    int32&                OutSkippedNoPng)
{
    OutGltfPrimCount = 0;
    OutSkippedNoPng  = 0;
    OutSlots.Reset();

    FGLTFMeshData GltfData;
    if (!FGLTFUtils::LoadMeshData(ActorDir / TEXT("mesh.gltf"), GltfData)) return false;
    if (GltfData.PrimToMaterial.IsEmpty()) return false;

    OutGltfPrimCount = GltfData.PrimToMaterial.Num();

    for (int32 PrimIdx = 0; PrimIdx < GltfData.PrimToMaterial.Num(); ++PrimIdx)
    {
        const int32 MatIdx = GltfData.PrimToMaterial[PrimIdx];
        FSlotPngInfo Info;
        if (MatIdx >= 0 && MatIdx < GltfData.Materials.Num())
        {
            const FString& Uri = GltfData.Materials[MatIdx].BaseColor;
            if (!Uri.IsEmpty())
            {
                Info.PngPath = ActorDir / Uri;
                Info.PngUri  = Uri;
            }
        }
        if (Info.PngPath.IsEmpty()) OutSkippedNoPng++;
        OutSlots.Add(MoveTemp(Info));
    }

    return !OutSlots.IsEmpty();
}

// ============================================================================
// Phase A: M_SlotID material (unlit, SlotColor vector parameter)
// ============================================================================

UMaterial* FNanobananaManager::GetOrCreateSlotIDMaterial()
{
    UMaterial* Mat = NewObject<UMaterial>(
        GetTransientPackage(), NAME_None, RF_Transient);

    Mat->SetShadingModel(MSM_Unlit);
    Mat->BlendMode = BLEND_Opaque;

    UMaterialExpressionVectorParameter* VecParam =
        NewObject<UMaterialExpressionVectorParameter>(Mat);
    VecParam->ParameterName = TEXT("SlotColor");
    VecParam->DefaultValue  = FLinearColor(1.f, 1.f, 0.f, 1.f);
    Mat->GetExpressionCollection().AddExpression(VecParam);
    Mat->GetEditorOnlyData()->EmissiveColor.Expression = VecParam;

    Mat->PreEditChange(nullptr);
    Mat->PostEditChange();

    Mat->ForceRecompileForRendering();

    if (GShaderCompilingManager)
        GShaderCompilingManager->FinishAllCompilation();
    FlushRenderingCommands();

    return Mat;
}

// ============================================================================
// Phase B: M_LabeledOverlay material (Lerp(BaseColorTex, ClassColor, OverlayAlpha))
// ============================================================================

UMaterial* FNanobananaManager::GetOrCreateLabeledOverlayMaterial()
{
    const FString PackageName = TEXT("/VCCSim/TexEnhancer/M_LabeledOverlay");
    const FString AssetPath   = PackageName + TEXT(".M_LabeledOverlay");

    if (UMaterial* Existing = LoadObject<UMaterial>(nullptr, *AssetPath))
        return Existing;

    UPackage* Package = CreatePackage(*PackageName);
    Package->FullyLoad();

    UMaterialFactoryNew* Factory = NewObject<UMaterialFactoryNew>();
    UMaterial* Mat = CastChecked<UMaterial>(
        Factory->FactoryCreateNew(UMaterial::StaticClass(), Package, TEXT("M_LabeledOverlay"),
            RF_Public | RF_Standalone | RF_Transactional, nullptr, GWarn));
    if (!Mat) return nullptr;

    UMaterialExpressionTextureSampleParameter2D* TexParam =
        NewObject<UMaterialExpressionTextureSampleParameter2D>(Mat);
    TexParam->ParameterName           = TEXT("BaseColorTex");
    TexParam->MaterialExpressionEditorX = -500;
    TexParam->MaterialExpressionEditorY = 0;
    Mat->GetExpressionCollection().AddExpression(TexParam);

    UMaterialExpressionVectorParameter* ColorParam =
        NewObject<UMaterialExpressionVectorParameter>(Mat);
    ColorParam->ParameterName           = TEXT("ClassColor");
    ColorParam->DefaultValue            = FLinearColor(1.f, 0.f, 0.f, 1.f);
    ColorParam->MaterialExpressionEditorX = -500;
    ColorParam->MaterialExpressionEditorY = 200;
    Mat->GetExpressionCollection().AddExpression(ColorParam);

    UMaterialExpressionScalarParameter* AlphaParam =
        NewObject<UMaterialExpressionScalarParameter>(Mat);
    AlphaParam->ParameterName           = TEXT("OverlayAlpha");
    AlphaParam->DefaultValue            = 0.4f;
    AlphaParam->MaterialExpressionEditorX = -500;
    AlphaParam->MaterialExpressionEditorY = 400;
    Mat->GetExpressionCollection().AddExpression(AlphaParam);

    UMaterialExpressionLinearInterpolate* Lerp =
        NewObject<UMaterialExpressionLinearInterpolate>(Mat);
    Lerp->MaterialExpressionEditorX = -150;
    Lerp->A.Expression              = TexParam;
    Lerp->B.Expression              = ColorParam;
    Lerp->Alpha.Expression          = AlphaParam;
    Mat->GetExpressionCollection().AddExpression(Lerp);

    Mat->GetEditorOnlyData()->BaseColor.Expression = Lerp;

    Mat->PreEditChange(nullptr);
    Mat->PostEditChange();
    Package->MarkPackageDirty();
    FAssetRegistryModule::AssetCreated(Mat);

    const FString FilePath = FPackageName::LongPackageNameToFilename(
        PackageName, FPackageName::GetAssetPackageExtension());
    FSavePackageArgs SaveArgs;
    SaveArgs.TopLevelFlags = RF_Public | RF_Standalone;
    UPackage::SavePackage(Package, Mat, *FilePath, SaveArgs);

    return Mat;
}

// ============================================================================
// Phase B: Import PNG as UTexture2D editor asset
// ============================================================================

UTexture2D* FNanobananaManager::ImportPngAsTexture(
    const FString& PngPath,
    const FString& PackagePath,
    const FString& AssetName)
{
    const FString FullPackageName = PackagePath / AssetName;
    const FString FullAssetPath   = FullPackageName + TEXT(".") + AssetName;

    if (UTexture2D* Existing = FindObject<UTexture2D>(nullptr, *FullAssetPath))
    {
        if (!Existing->IsRooted())
            Existing->ConditionalBeginDestroy();
        CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
    }

    UPackage* Package = CreatePackage(*FullPackageName);
    Package->FullyLoad();

    bool bCanceled = false;
    UTextureFactory* TexFactory = NewObject<UTextureFactory>();
    TexFactory->SuppressImportOverwriteDialog();

    UTexture2D* Tex = CastChecked<UTexture2D>(
        TexFactory->ImportObject(UTexture2D::StaticClass(), Package, *AssetName,
            RF_Public | RF_Standalone, PngPath, nullptr, bCanceled),
        ECastCheckedType::NullAllowed);
    if (!Tex) return nullptr;

    Tex->SRGB                = true;
    Tex->CompressionSettings = TC_Default;
    Tex->PostEditChange();
    Package->MarkPackageDirty();
    FAssetRegistryModule::AssetCreated(Tex);

    const FString FilePath = FPackageName::LongPackageNameToFilename(
        FullPackageName, FPackageName::GetAssetPackageExtension());
    FSavePackageArgs SaveArgs;
    SaveArgs.TopLevelFlags = RF_Public | RF_Standalone;
    UPackage::SavePackage(Package, Tex, *FilePath, SaveArgs);

    return Tex;
}

// ============================================================================
// Phase B: Create UMaterialInstanceConstant for a labeled slot
// ============================================================================

UMaterialInstanceConstant* FNanobananaManager::CreateLabeledMIC(
    UMaterial*     ParentMat,
    UTexture2D*    BaseTex,
    uint8 CR, uint8 CG, uint8 CB,
    float          OverlayAlpha,
    const FString& PackagePath,
    const FString& AssetName)
{
    const FString FullPackageName = PackagePath / AssetName;
    const FString FullAssetPath   = FullPackageName + TEXT(".") + AssetName;

    if (UMaterialInstanceConstant* Existing = FindObject<UMaterialInstanceConstant>(nullptr, *FullAssetPath))
    {
        if (!Existing->IsRooted())
            Existing->ConditionalBeginDestroy();
        CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
    }

    IAssetTools& AssetTools =
        FModuleManager::LoadModuleChecked<FAssetToolsModule>("AssetTools").Get();

    UMaterialInstanceConstantFactoryNew* MICFactory =
        NewObject<UMaterialInstanceConstantFactoryNew>();
    MICFactory->InitialParent = ParentMat;

    UMaterialInstanceConstant* MIC = Cast<UMaterialInstanceConstant>(
        AssetTools.CreateAsset(AssetName, PackagePath,
            UMaterialInstanceConstant::StaticClass(), MICFactory));
    if (!MIC) return nullptr;

    if (BaseTex)
    {
        FTextureParameterValue TPV;
        TPV.ParameterInfo.Name = TEXT("BaseColorTex");
        TPV.ParameterValue     = BaseTex;
        MIC->TextureParameterValues.Add(TPV);
    }

    {
        FVectorParameterValue VPV;
        VPV.ParameterInfo.Name = TEXT("ClassColor");
        VPV.ParameterValue     = FLinearColor(CR / 255.f, CG / 255.f, CB / 255.f, 1.f);
        MIC->VectorParameterValues.Add(VPV);
    }

    {
        FScalarParameterValue SPV;
        SPV.ParameterInfo.Name = TEXT("OverlayAlpha");
        SPV.ParameterValue     = OverlayAlpha;
        MIC->ScalarParameterValues.Add(SPV);
    }

    MIC->PostEditChange();
    return MIC;
}

// ============================================================================
// Main synchronous pipeline
// ============================================================================

void FNanobananaManager::RunOnGameThread()
{
    check(IsInGameThread());

    OnProgress.ExecuteIfBound(TEXT("Loading manifest..."), 0, 0);

    // -------------------------------------------------------------------------
    // Parse manifest → collect actor labels and directories
    // -------------------------------------------------------------------------
    FString ManifestStr;
    if (!FFileHelper::LoadFileToString(ManifestStr, *Params.ManifestFile))
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Cannot load manifest.json"), true);
        bIsInProgress = false;
        OnComplete.ExecuteIfBound(TEXT("Error: Cannot load manifest.json"));
        return;
    }

    TSharedPtr<FJsonObject> ManifestRoot;
    {
        TSharedRef<TJsonReader<>> MR = TJsonReaderFactory<>::Create(ManifestStr);
        if (!FJsonSerializer::Deserialize(MR, ManifestRoot) || !ManifestRoot.IsValid())
        {
            bIsInProgress = false;
            OnComplete.ExecuteIfBound(TEXT("Error: Failed to parse manifest.json"));
            return;
        }
    }

    TArray<FString> ActorLabels;
    TArray<FString> ActorDirs;
    const TArray<TSharedPtr<FJsonValue>>* ActorsArr = nullptr;
    if (ManifestRoot->TryGetArrayField(TEXT("actors"), ActorsArr))
    {
        for (const auto& V : *ActorsArr)
        {
            FString L;
            if (V->AsObject().IsValid()) V->AsObject()->TryGetStringField(TEXT("label"), L);
            if (!L.IsEmpty()) { ActorLabels.Add(L); ActorDirs.Add(OutputDir / L); }
        }
    }
    else
    {
        FString SingleLabel;
        if (ManifestRoot->TryGetStringField(TEXT("label"), SingleLabel) && !SingleLabel.IsEmpty())
        {
            ActorLabels.Add(SingleLabel);
            ActorDirs.Add(OutputDir);
        }
    }

    if (ActorLabels.IsEmpty())
    {
        bIsInProgress = false;
        OnComplete.ExecuteIfBound(TEXT("No actors found in manifest."));
        return;
    }

    if (!Params.World)
    {
        bIsInProgress = false;
        OnComplete.ExecuteIfBound(TEXT("Error: World is null. Pass a valid UWorld* in FProjectionParams."));
        return;
    }

    // -------------------------------------------------------------------------
    // Load slot PNG infos per actor from mesh.gltf (no vertex data needed)
    // -------------------------------------------------------------------------
    TArray<FActorData> Actors;
    for (int32 li = 0; li < ActorLabels.Num(); ++li)
    {
        FActorData AD;
        AD.Label = ActorLabels[li];
        AD.Dir   = ActorDirs[li];

        int32 GltfPrimCount = 0, SkippedNoPng = 0;
        if (!LoadSlotPngInfos(AD.Dir, AD.Slots, GltfPrimCount, SkippedNoPng))
        {
            UE_LOG(LogNanobananaManager, Warning,
                TEXT("Actor '%s': no mesh.gltf found in '%s'"), *AD.Label, *AD.Dir);
            continue;
        }

        UE_LOG(LogNanobananaManager, Log,
            TEXT("Actor '%s': GLTF primitives=%d, loaded slots=%d (%d slot(s) had no baseColorTexture)"),
            *AD.Label, GltfPrimCount, AD.Slots.Num(), SkippedNoPng);

        Actors.Add(MoveTemp(AD));
    }

    if (Actors.IsEmpty())
    {
        bIsInProgress = false;
        OnComplete.ExecuteIfBound(TEXT("No GLTF data loaded for any actor."));
        return;
    }

    // -------------------------------------------------------------------------
    // Load poses (UE LH cm)
    // -------------------------------------------------------------------------
    TArray<FString> PoseLines;
    FFileHelper::LoadFileToStringArray(PoseLines, *Params.PosesFile);
    TArray<FVCCSimPoseData> Poses;
    for (const FString& Line : PoseLines)
    {
        if (Line.IsEmpty() || Line.StartsWith(TEXT("#"))) continue;
        FVCCSimPoseData P = FVCCSimDataConverter::ParsePoseLine(Line);
        if (!P.Location.IsZero() || !P.Quaternion.IsIdentity()) Poses.Add(P);
    }

    // -------------------------------------------------------------------------
    // Find nanobanana mask images
    // -------------------------------------------------------------------------
    TArray<FString> PngFiles;
    IFileManager::Get().FindFilesRecursive(
        PngFiles, *Params.ResultDir, TEXT("*_nanobanana_mask.png"), true, false);
    PngFiles.Sort();

    if (PngFiles.Num() != Poses.Num())
    {
        UE_LOG(LogNanobananaManager, Warning,
            TEXT("Count mismatch: %d mask image(s) vs %d pose(s). Will use the first %d pair(s)."),
            PngFiles.Num(), Poses.Num(), FMath::Min(PngFiles.Num(), Poses.Num()));
    }

    const int32 NumImages = FMath::Min(Poses.Num(), PngFiles.Num());
    if (NumImages == 0)
    {
        bIsInProgress = false;
        OnComplete.ExecuteIfBound(TEXT("No images or poses found."));
        return;
    }

    UE_LOG(LogNanobananaManager, Log,
        TEXT("Processing %d images, %d actors"), NumImages, Actors.Num());

    // =========================================================================
    // PHASE A: SceneCapture2D ID Pass
    // =========================================================================

    UMaterial* SlotIDMat = GetOrCreateSlotIDMaterial();
    if (!SlotIDMat)
    {
        bIsInProgress = false;
        OnComplete.ExecuteIfBound(TEXT("Error: Failed to create M_SlotID material."));
        return;
    }

    IImageWrapperModule& IWM =
        FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));

    int32 RTWidth  = Params.ImageWidth;
    int32 RTHeight = Params.ImageHeight;
    {
        TArray<uint8> ProbBytes;
        if (FFileHelper::LoadFileToArray(ProbBytes, *PngFiles[0]))
        {
            TSharedPtr<IImageWrapper> WrapProbe = IWM.CreateImageWrapper(EImageFormat::PNG);
            if (WrapProbe.IsValid() && WrapProbe->SetCompressed(ProbBytes.GetData(), ProbBytes.Num()))
            {
                RTWidth  = WrapProbe->GetWidth();
                RTHeight = WrapProbe->GetHeight();
                if (RTWidth != Params.ImageWidth || RTHeight != Params.ImageHeight)
                {
                    UE_LOG(LogNanobananaManager, Log,
                        TEXT("Mask image size %dx%d detected; render target adapted (configured was %dx%d)."),
                        RTWidth, RTHeight, Params.ImageWidth, Params.ImageHeight);
                }
            }
        }
    }

    // Per-actor, per-slot vote table.  Votes[ai][slot][class]
    TArray<TArray<TArray<int32>>> Votes;
    Votes.SetNum(Actors.Num());

    // Build actor→world map once
    TMap<FString, AStaticMeshActor*> LabelMap;
    for (TActorIterator<AStaticMeshActor> It(Params.World); It; ++It)
        if (AStaticMeshActor* A = *It) LabelMap.Add(A->GetActorLabel(), A);

    // Spawn reusable SceneCapture2D
    ASceneCapture2D* CaptureActor = Params.World->SpawnActor<ASceneCapture2D>();
    USceneCaptureComponent2D* Comp = CaptureActor->GetCaptureComponent2D();

    UTextureRenderTarget2D* RT = NewObject<UTextureRenderTarget2D>(CaptureActor);
    RT->RenderTargetFormat = RTF_RGBA32f;
    RT->ClearColor         = FLinearColor(0.f, 0.f, 0.f, 0.f);
    RT->TargetGamma        = 1.0f;
    RT->InitAutoFormat(RTWidth, RTHeight);
    Comp->TextureTarget         = RT;
    Comp->CaptureSource         = SCS_FinalColorHDR;
    Comp->bCaptureEveryFrame    = false;
    Comp->bCaptureOnMovement    = false;
    Comp->FOVAngle              = Params.HFOV;

    // Disable all post-processing so unlit base color passes through unchanged
    Comp->ShowFlags.SetBloom(false);
    Comp->ShowFlags.SetEyeAdaptation(false);
    Comp->ShowFlags.SetTonemapper(false);
    Comp->ShowFlags.SetAmbientOcclusion(false);
    Comp->ShowFlags.SetAtmosphere(false);
    Comp->ShowFlags.SetFog(false);
    Comp->ShowFlags.SetMotionBlur(false);
    Comp->ShowFlags.SetLensFlares(false);
    Comp->ShowFlags.SetContactShadows(false);
    Comp->ShowFlags.SetDepthOfField(false);
    Comp->ShowFlags.SetScreenSpaceReflections(false);
    Comp->ShowFlags.SetAntiAliasing(false);
    Comp->ShowFlags.SetTemporalAA(false);

    Comp->PostProcessSettings.bOverride_AutoExposureMethod        = true;
    Comp->PostProcessSettings.AutoExposureMethod                  = AEM_Manual;
    Comp->PostProcessSettings.bOverride_AutoExposureBias          = true;
    Comp->PostProcessSettings.AutoExposureBias                    = 0.f;
    Comp->PostProcessSettings.bOverride_AutoExposureMinBrightness = true;
    Comp->PostProcessSettings.AutoExposureMinBrightness           = 1.f;
    Comp->PostProcessSettings.bOverride_AutoExposureMaxBrightness = true;
    Comp->PostProcessSettings.AutoExposureMaxBrightness           = 1.f;
    Comp->PostProcessSettings.bOverride_ToneCurveAmount           = true;
    Comp->PostProcessSettings.ToneCurveAmount                     = 0.f;

    Comp->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_UseShowOnlyList;

    // Per-actor ID pass
    for (int32 ai = 0; ai < Actors.Num(); ++ai)
    {
        const FActorData& Actor = Actors[ai];
        AStaticMeshActor** FoundActor = LabelMap.Find(Actor.Label);
        if (!FoundActor)
        {
            UE_LOG(LogNanobananaManager, Warning,
                TEXT("Actor '%s': not found in world, skipping ID pass"), *Actor.Label);
            Votes[ai].SetNum(Actor.Slots.Num());
            for (auto& V : Votes[ai]) V.Init(0, NBClassCount);
            continue;
        }

        AStaticMeshActor* TargetActor = *FoundActor;
        UStaticMeshComponent* MC      = TargetActor->GetStaticMeshComponent();
        const int32 NumSlots          = MC ? MC->GetNumMaterials() : 0;

        // Allocate vote table
        Votes[ai].SetNum(NumSlots);
        for (auto& V : Votes[ai]) V.Init(0, NBClassCount);

        if (NumSlots == 0)
        {
            UE_LOG(LogNanobananaManager, Warning,
                TEXT("Actor '%s': zero material slots, skipping"), *Actor.Label);
            continue;
        }

        TArray<UMaterialInterface*> OriginalMaterials;
        TArray<UMaterialInstanceDynamic*> SlotIDMIDs;
        OriginalMaterials.SetNum(NumSlots);
        SlotIDMIDs.SetNum(NumSlots);

        for (int32 s = 0; s < NumSlots; ++s)
        {
            OriginalMaterials[s] = MC->GetMaterial(s);
            const int32 Enc = s + 1;
            UMaterialInstanceDynamic* MID =
                UMaterialInstanceDynamic::Create(SlotIDMat, Params.World);
            MID->SetVectorParameterValue(TEXT("SlotColor"), FLinearColor(
                float(Enc), 1.f, 0.f, 1.f));
            SlotIDMIDs[s] = MID;
            MC->SetMaterial(s, MID);
        }

        MC->UnregisterComponent();
        MC->RegisterComponent();
        FlushRenderingCommands();

        Comp->ShowOnlyActors.Reset();
        Comp->ShowOnlyActors.Add(TargetActor);

        FScopedSlowTask SlowTask(NumImages,
            FText::Format(NSLOCTEXT("NanobananaManager", "IDPass", "ID pass: {0} ({1} frames)"),
                FText::FromString(Actor.Label), FText::AsNumber(NumImages)));
        SlowTask.MakeDialog(true);

        for (int32 ImgIdx = 0; ImgIdx < NumImages; ++ImgIdx)
        {
            SlowTask.EnterProgressFrame(1.f,
                FText::Format(NSLOCTEXT("NanobananaManager", "Frame", "Frame {0}/{1}"),
                    FText::AsNumber(ImgIdx + 1), FText::AsNumber(NumImages)));

            if (SlowTask.ShouldCancel()) break;

            // Set camera transform from pose
            const FVCCSimPoseData& Pose = Poses[ImgIdx];
            CaptureActor->SetActorLocationAndRotation(
                Pose.Location, Pose.Quaternion.Rotator());

            Comp->CaptureScene();
            FlushRenderingCommands();

            TArray<FLinearColor> IDPixels;
            RT->GameThread_GetRenderTargetResource()->ReadLinearColorPixels(IDPixels);

            if (FSlateApplication::IsInitialized())
                FSlateApplication::Get().PumpMessages();

            // Load nanobanana mask
            TArray<uint8> FileBytes;
            if (!FFileHelper::LoadFileToArray(FileBytes, *PngFiles[ImgIdx])) continue;
            TSharedPtr<IImageWrapper> Wrap = IWM.CreateImageWrapper(EImageFormat::PNG);
            if (!Wrap.IsValid() || !Wrap->SetCompressed(FileBytes.GetData(), FileBytes.Num())) continue;
            TArray<uint8> MaskRaw;
            if (!Wrap->GetRaw(ERGBFormat::BGRA, 8, MaskRaw)) continue;
            const int32 MW = Wrap->GetWidth(), MH = Wrap->GetHeight();
            if (MaskRaw.Num() != MW * MH * 4) continue;
            if (MW != RTWidth || MH != RTHeight)
            {
                UE_LOG(LogNanobananaManager, Warning,
                    TEXT("Frame %d: mask size %dx%d != render target %dx%d, skipping"),
                    ImgIdx, MW, MH, RTWidth, RTHeight);
                continue;
            }

            const int32 NumPixels = MW * MH;
            if (IDPixels.Num() < NumPixels) continue;

            for (int32 p = 0; p < NumPixels; ++p)
            {
                const float RefG = IDPixels[p].G;
                if (RefG < 1e-6f) continue;
                const int32 Decoded = FMath::RoundToInt(IDPixels[p].R / RefG);
                if (Decoded < 1 || Decoded > NumSlots) continue;
                const int32 SlotIdx = Decoded - 1;

                const int32 MaskOff = p * 4;
                const uint8 Bv = MaskRaw[MaskOff];
                const uint8 Gv = MaskRaw[MaskOff + 1];
                const uint8 Rv = MaskRaw[MaskOff + 2];

                int32 BestClass = -1, BestDist = NBTolerance + 1;
                for (int32 c = 0; c < NBClassCount; ++c)
                {
                    const int32 D = FMath::Abs((int32)Rv - NBClasses[c].R)
                                  + FMath::Abs((int32)Gv - NBClasses[c].G)
                                  + FMath::Abs((int32)Bv - NBClasses[c].B);
                    if (D < BestDist) { BestDist = D; BestClass = c; }
                }
                if (BestClass >= 0) { ++Votes[ai][SlotIdx][BestClass]; }
            }

        }

        for (int32 s = 0; s < NumSlots; ++s)
            MC->SetMaterial(s, OriginalMaterials[s]);
        MC->UnregisterComponent();
        MC->RegisterComponent();
    }

    Comp->ShowOnlyActors.Reset();
    CaptureActor->Destroy();

    // =========================================================================
    // PHASE B & C: Create labeled materials, labeled actor, and GLTF export
    // =========================================================================

    UMaterial* LabeledOverlayMat = GetOrCreateLabeledOverlayMaterial();
    if (!LabeledOverlayMat)
    {
        bIsInProgress = false;
        OnComplete.ExecuteIfBound(TEXT("Error: Failed to create M_LabeledOverlay material."));
        return;
    }

    int32 TotalLabeledSlots   = 0;
    int32 TotalActorsExported = 0;

    TSharedPtr<FJsonObject> ResultRoot     = MakeShareable(new FJsonObject);
    TArray<TSharedPtr<FJsonValue>> ActorResultsJson;

    int32 TotalPhaseBCWork = 0;
    for (const FActorData& AD : Actors)
    {
        AStaticMeshActor** FA = LabelMap.Find(AD.Label);
        if (!FA) continue;
        UStaticMeshComponent* TMC = (*FA)->GetStaticMeshComponent();
        TotalPhaseBCWork += (TMC ? TMC->GetNumMaterials() : 0) + 1;
    }

    FScopedSlowTask PhaseBCTask(float(FMath::Max(1, TotalPhaseBCWork)),
        NSLOCTEXT("NanobananaManager", "PhaseBC", "Creating labeled materials..."));
    PhaseBCTask.MakeDialog(false);

    for (int32 ai = 0; ai < Actors.Num(); ++ai)
    {
        const FActorData& Actor = Actors[ai];
        AStaticMeshActor** FoundActor = LabelMap.Find(Actor.Label);
        if (!FoundActor) continue;

        AStaticMeshActor*    TargetActor = *FoundActor;
        UStaticMeshComponent* MC         = TargetActor->GetStaticMeshComponent();
        const int32 NumSlots             = MC ? MC->GetNumMaterials() : 0;

        // Summary counters for logging
        int32 SlotWithGeom     = Actor.Slots.Num();
        int32 SlotWithPng      = 0;
        int32 SlotLabeled      = 0;
        int32 SkipZeroVotes    = 0;
        int32 SkipNoPng        = 0;
        int32 SkipPngMissing   = 0;

        const FString SafeLabel     = Actor.Label.Replace(TEXT(" "), TEXT("_"));
        const FString AssetPkgBase  =
            FString::Printf(TEXT("/VCCSim/TexEnhancer/%s"), *SafeLabel);

        const FString AssetDiskDir = FPackageName::LongPackageNameToFilename(AssetPkgBase);
        if (IFileManager::Get().DirectoryExists(*AssetDiskDir))
            IFileManager::Get().DeleteDirectory(*AssetDiskDir, false, true);
        IFileManager::Get().MakeDirectory(*AssetDiskDir, true);

        TSharedPtr<FJsonObject> ActorJson = MakeShareable(new FJsonObject);
        ActorJson->SetStringField(TEXT("label"), Actor.Label);
        TArray<TSharedPtr<FJsonValue>> PrimJson;

        TArray<UMaterialInterface*> LabeledMaterials;
        LabeledMaterials.SetNum(NumSlots);
        for (int32 s = 0; s < NumSlots; ++s)
            LabeledMaterials[s] = MC->GetMaterial(s);  // default = original

        for (int32 pi = 0; pi < NumSlots; ++pi)
        {
            PhaseBCTask.EnterProgressFrame(1.f, FText::Format(
                NSLOCTEXT("NanobananaManager", "PhaseBCSlot", "'{0}': slot {1} / {2}"),
                FText::FromString(Actor.Label),
                FText::AsNumber(pi + 1),
                FText::AsNumber(NumSlots)));

            TSharedPtr<FJsonObject> PJ = MakeShareable(new FJsonObject);
            PJ->SetNumberField(TEXT("primitive"), pi);

            const bool bHasVotes = pi < Votes[ai].Num();
            int32 BestClass = -1, BestVotes = 0;
            if (bHasVotes)
            {
                const TArray<int32>& PVotes = Votes[ai][pi];
                for (int32 c = 0; c < NBClassCount; ++c)
                    if (PVotes[c] > BestVotes) { BestVotes = PVotes[c]; BestClass = c; }
            }

            const bool bHasSlotInfo  = pi < Actor.Slots.Num();
            const FString PngPath    = bHasSlotInfo ? Actor.Slots[pi].PngPath : FString{};
            const bool bPngEmpty     = PngPath.IsEmpty();
            const bool bPngMissing   = !bPngEmpty && !IFileManager::Get().FileExists(*PngPath);

            if (BestVotes == 0)
            {
                PJ->SetStringField(TEXT("class"), TEXT(""));
                PJ->SetNumberField(TEXT("votes"), 0);
                PJ->SetStringField(TEXT("skip_reason"), TEXT("zero_votes"));
                ++SkipZeroVotes;
            }
            else if (bPngEmpty)
            {
                PJ->SetStringField(TEXT("class"), FString(NBClasses[BestClass].Name));
                PJ->SetNumberField(TEXT("votes"), BestVotes);
                PJ->SetStringField(TEXT("skip_reason"), TEXT("no_basecolor_texture"));
                ++SkipNoPng;
            }
            else if (bPngMissing)
            {
                UE_LOG(LogNanobananaManager, Warning,
                    TEXT("Actor '%s' slot %d: PNG not found '%s'"),
                    *Actor.Label, pi, *PngPath);
                PJ->SetStringField(TEXT("class"), FString(NBClasses[BestClass].Name));
                PJ->SetNumberField(TEXT("votes"), BestVotes);
                PJ->SetStringField(TEXT("skip_reason"), TEXT("png_not_found"));
                ++SkipPngMissing;
            }
            else
            {
                // Success path: create labeled MIC
                ++SlotWithPng;

                const TArray<int32>& PVotes = Votes[ai][pi];
                int32 TotalVotes = 0;
                for (int32 c = 0; c < NBClassCount; ++c)
                    TotalVotes += PVotes[c];

                TSharedPtr<FJsonObject> DistJson = MakeShareable(new FJsonObject);
                for (int32 c = 0; c < NBClassCount; ++c)
                    if (PVotes[c] > 0)
                        DistJson->SetNumberField(FString(NBClasses[c].Name), PVotes[c]);
                PJ->SetObjectField(TEXT("vote_distribution"), DistJson);
                PJ->SetStringField(TEXT("class"), FString(NBClasses[BestClass].Name));
                PJ->SetNumberField(TEXT("votes"), BestVotes);

                const FString TexName = FString::Printf(TEXT("Tex_slot%d_base"), pi);
                const FString MICName = FString::Printf(TEXT("MI_slot%d_%s"), pi,
                    NBClasses[BestClass].Name);

                UTexture2D* BaseTex =
                    ImportPngAsTexture(PngPath, AssetPkgBase, TexName);

                UMaterialInstanceConstant* MIC = CreateLabeledMIC(
                    LabeledOverlayMat, BaseTex,
                    NBClasses[BestClass].R, NBClasses[BestClass].G, NBClasses[BestClass].B,
                    Params.OverlayAlpha, AssetPkgBase, MICName);

                if (MIC) LabeledMaterials[pi] = MIC;

                PJ->SetStringField(TEXT("mic_asset"), AssetPkgBase / MICName);
                ++SlotLabeled;
                ++TotalLabeledSlots;
            }

            PrimJson.Add(MakeShareable(new FJsonValueObject(PJ)));
        }

        UE_LOG(LogNanobananaManager, Log,
            TEXT("Actor '%s' summary: total_slots=%d, with_geometry=%d, with_png=%d, labeled=%d "
                 "| skipped: zero_votes=%d, no_png=%d, png_missing=%d"),
            *Actor.Label, NumSlots, SlotWithGeom, SlotWithPng, SlotLabeled,
            SkipZeroVotes, SkipNoPng, SkipPngMissing);

        ActorJson->SetArrayField(TEXT("primitives"), PrimJson);

        // All outputs go into Actor.Dir/labeled/ — wipe it clean first so successive
        // runs never accumulate stale files and never write anything outside this folder.
        const FString LabeledDir = Actor.Dir / TEXT("labeled");
        if (IFileManager::Get().DirectoryExists(*LabeledDir))
            IFileManager::Get().DeleteDirectory(*LabeledDir, false, true);
        IFileManager::Get().MakeDirectory(*LabeledDir, true);

        // Save labeled StaticMesh asset
        {
            UStaticMesh* OrigMesh = TargetActor->GetStaticMeshComponent()->GetStaticMesh();
            const FString MeshAssetName = FString::Printf(TEXT("%s_labeled"), *SafeLabel);
            const FString MeshPkgName   = AssetPkgBase / MeshAssetName;

            UPackage* MeshPackage = CreatePackage(*MeshPkgName);
            UStaticMesh* LabeledMesh = DuplicateObject<UStaticMesh>(
                OrigMesh, MeshPackage, *MeshAssetName);
            LabeledMesh->SetFlags(RF_Public | RF_Standalone);

            TArray<FStaticMaterial>& StaticMats = LabeledMesh->GetStaticMaterials();
            for (int32 s = 0; s < NumSlots && s < StaticMats.Num(); ++s)
                StaticMats[s].MaterialInterface = LabeledMaterials[s];

            LabeledMesh->PostEditChange();
            MeshPackage->MarkPackageDirty();
            FAssetRegistryModule::AssetCreated(LabeledMesh);

            const FString MeshFilePath = FPackageName::LongPackageNameToFilename(
                MeshPkgName, FPackageName::GetAssetPackageExtension());
            FSavePackageArgs MeshSaveArgs;
            MeshSaveArgs.TopLevelFlags = RF_Public | RF_Standalone;
            UPackage::SavePackage(MeshPackage, LabeledMesh, *MeshFilePath, MeshSaveArgs);

            ActorJson->SetStringField(TEXT("labeled_mesh"), MeshPkgName);
            UE_LOG(LogNanobananaManager, Log,
                TEXT("Actor '%s' -> labeled mesh: %s"), *Actor.Label, *MeshPkgName);
        }

        // Spawn temporary actor for GLTF export, then destroy
        FActorSpawnParameters SpawnParams;
        SpawnParams.SpawnCollisionHandlingOverride =
            ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
        AStaticMeshActor* LabeledActor = Params.World->SpawnActor<AStaticMeshActor>(
            TargetActor->GetActorLocation(), TargetActor->GetActorRotation(), SpawnParams);
        if (LabeledActor)
        {
            LabeledActor->SetActorScale3D(TargetActor->GetActorScale3D());
            LabeledActor->GetStaticMeshComponent()->SetStaticMesh(
                TargetActor->GetStaticMeshComponent()->GetStaticMesh());
            for (int32 s = 0; s < NumSlots; ++s)
                LabeledActor->GetStaticMeshComponent()->SetMaterial(s, LabeledMaterials[s]);

            PhaseBCTask.EnterProgressFrame(1.f, FText::Format(
                NSLOCTEXT("NanobananaManager", "PhaseBCExport", "Exporting '{0}'..."),
                FText::FromString(Actor.Label)));

            const FString LabeledGltf = LabeledDir / TEXT("mesh.gltf");

            UGLTFExportOptions* ExportOptions = NewObject<UGLTFExportOptions>();
            ExportOptions->ExportUniformScale        = 0.01f;
            ExportOptions->BakeMaterialInputs        = EGLTFMaterialBakeMode::UseMeshData;
            ExportOptions->DefaultMaterialBakeSize   = FGLTFMaterialBakeSize{
                Params.TextureResolution, Params.TextureResolution, false };
            ExportOptions->TextureImageFormat        = EGLTFTextureImageFormat::PNG;
            ExportOptions->bAdjustNormalmaps         = true;
            ExportOptions->bExportLights             = false;
            ExportOptions->bExportCameras            = false;
            ExportOptions->bExportAnimationSequences = false;
            ExportOptions->bExportLevelSequences     = false;
            ExportOptions->DefaultLevelOfDetail      = 0;

            TSet<AActor*> ExportActors = { LabeledActor };
            FGLTFExportMessages ExportMessages;
            const bool bExportOk = UGLTFExporter::ExportToGLTF(
                Params.World, LabeledGltf, ExportOptions, ExportActors, ExportMessages);

            for (const FString& W : ExportMessages.Warnings)
                UE_LOG(LogNanobananaManager, Warning,
                    TEXT("GLTF '%s': %s"), *Actor.Label, *W);
            for (const FString& E : ExportMessages.Errors)
                UE_LOG(LogNanobananaManager, Error,
                    TEXT("GLTF '%s': %s"), *Actor.Label, *E);

            if (bExportOk)
            {
                ActorJson->SetStringField(TEXT("labeled_gltf"), LabeledGltf);
                ++TotalActorsExported;
                UE_LOG(LogNanobananaManager, Log,
                    TEXT("Actor '%s' -> labeled GLTF: %s"), *Actor.Label, *LabeledGltf);
            }
            else
            {
                UE_LOG(LogNanobananaManager, Warning,
                    TEXT("Actor '%s': GLTF export failed"), *Actor.Label);
            }

            LabeledActor->Destroy();
        }

        // Write per-actor result JSON into labeled/ — never into the input directory.
        ActorJson->SetStringField(TEXT("labeled_dir"), LabeledDir);
        FString ActorResultStr;
        TSharedRef<TJsonWriter<>> ActorWriter = TJsonWriterFactory<>::Create(&ActorResultStr);
        FJsonSerializer::Serialize(ActorJson.ToSharedRef(), ActorWriter);
        FFileHelper::SaveStringToFile(ActorResultStr,
            *(LabeledDir / TEXT("label_assignment.json")));

        ActorResultsJson.Add(MakeShareable(new FJsonValueObject(ActorJson)));
    }

    UE_LOG(LogNanobananaManager, Log,
        TEXT("Nanobanana complete: %d slots labeled, %d actors exported"),
        TotalLabeledSlots, TotalActorsExported);

    const FString FinalStatus = FString::Printf(
        TEXT("Done: %d slots labeled across %d actors"),
        TotalLabeledSlots, TotalActorsExported);

    bIsInProgress = false;
    OnComplete.ExecuteIfBound(FinalStatus);
}
