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

#include "Utils/VCCSIMDisplayWidget.h"
#include "Utils/ConfigParser.h"
#include "Utils/MeshHandlerComponent.h"
#include "Utils/InsMeshHolder.h"
#include "Utils/ImageProcesser.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Sensors/RGBCamera.h"
#include "Sensors/SegmentationCamera.h"
#include "Sensors/NormalCamera.h"
#include "EngineUtils.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/World.h"

#include "Windows/WindowsHWrapper.h" // Deal with UpdateResourceW error
#include "HAL/FileManagerGeneric.h"
#include "Sensors/DepthCamera.h"

DEFINE_LOG_CATEGORY_STATIC(LogVCCSimDisplayWidget, Log, All);

void UVCCSIMDisplayWidget::NativeConstruct()
{
    Super::NativeConstruct();

    InitializeViewData();
    SetupViewBindings();
}

void UVCCSIMDisplayWidget::InitializeViewData()
{
    ViewDataMap.Empty();

    ViewDataMap.Add(EVCCSimViewType::RGB, FVCCSimViewData());
    ViewDataMap.Add(EVCCSimViewType::Depth, FVCCSimViewData());
    ViewDataMap.Add(EVCCSimViewType::Normal, FVCCSimViewData());
    ViewDataMap.Add(EVCCSimViewType::Segmentation, FVCCSimViewData());
    ViewDataMap.Add(EVCCSimViewType::Lit, FVCCSimViewData());
    ViewDataMap.Add(EVCCSimViewType::PointCloud, FVCCSimViewData());
    ViewDataMap.Add(EVCCSimViewType::Unit, FVCCSimViewData());

    // Initialize render settings for scene capture views
    if (FVCCSimViewData* LitData = GetViewData(EVCCSimViewType::Lit))
    {
        LitData->UpdateInterval = 1.0f/30.0f;
    }

    if (FVCCSimViewData* PCData = GetViewData(EVCCSimViewType::PointCloud))
    {
        PCData->UpdateInterval = 0.1f;
    }

    if (FVCCSimViewData* UnitData = GetViewData(EVCCSimViewType::Unit))
    {
        UnitData->UpdateInterval = 0.1f;
    }
}

void UVCCSIMDisplayWidget::SetupViewBindings()
{
    for (auto& [ViewType, ViewData] : ViewDataMap)
    {
        UImage* ImageDisplay = nullptr;
        UMaterialInterface* Material = nullptr;

        switch (ViewType)
        {
        case EVCCSimViewType::RGB:
            ImageDisplay = RGBImageDisplay;
            Material = RGBVisualizationMaterial;
            break;
        case EVCCSimViewType::Depth:
            ImageDisplay = DepthImageDisplay;
            Material = DepthVisualizationMaterial;
            break;
        case EVCCSimViewType::Normal:
            ImageDisplay = NormalImageDisplay;
            Material = NormalVisualizationMaterial;
            break;
        case EVCCSimViewType::Segmentation:
            ImageDisplay = SegImageDisplay;
            Material = SegVisualizationMaterial;
            break;
        case EVCCSimViewType::Lit:
            ImageDisplay = LitImageDisplay;
            Material = LitVisualizationMaterial;
            break;
        case EVCCSimViewType::PointCloud:
            ImageDisplay = PCImageDisplay;
            Material = PCVisualizationMaterial;
            break;
        case EVCCSimViewType::Unit:
            ImageDisplay = UnitImageDisplay;
            Material = MeshVisualizationMaterial;
            break;
        }

        ViewData.ImageDisplay = ImageDisplay;
        ViewData.VisualizationMaterial = Material;

        if (ViewData.VisualizationMaterial && ViewData.ImageDisplay)
        {
            ViewData.MaterialInstance =
                UMaterialInstanceDynamic::Create(ViewData.VisualizationMaterial, this);
            ViewData.ImageDisplay->SetBrushFromMaterial(ViewData.MaterialInstance);
        }
    }
}

void UVCCSIMDisplayWidget::NativeTick(const FGeometry& MyGeometry, float InDeltaTime)
{
    Super::NativeTick(MyGeometry, InDeltaTime);

    if (LitImageDisplay && LitImageDisplay->IsVisible())
    {
        UpdateViewImage(EVCCSimViewType::Lit, InDeltaTime);
    }

    if (PCImageDisplay && PCImageDisplay->IsVisible())
    {
        UpdateViewImage(EVCCSimViewType::PointCloud, InDeltaTime);
    }

    if (UnitImageDisplay && UnitImageDisplay->IsVisible())
    {
        UpdateViewImage(EVCCSimViewType::Unit, InDeltaTime);
    }

    for (int i = 0; i < 6; ++i)
    {
        int32 ID;
        if (CaptureQueue.Dequeue(ID))
        {
            CurrentQueueSize--;
            EVCCSimViewType ViewType = IDToViewType(ID);

            switch (ID)
            {
            case 4: // RGB
            case 5: // Depth
            case 6: // Normal
            case 7: // Segmentation
                ProcessCapture(ID);
                break;
            case 8: // Lit
                UpdateViewImage(EVCCSimViewType::Lit, 0.0f);
                ProcessCapture(ID);
                break;
            case 9: // Point Cloud
                UpdateViewImage(EVCCSimViewType::PointCloud, 0.0f);
                ProcessCapture(ID);
                break;
            case 0: // Unit
                UpdateViewImage(EVCCSimViewType::Unit, 0.0f);
                ProcessCapture(ID);
                break;
            default:
                break;
            }
        }
    }
}

FVCCSimViewData* UVCCSIMDisplayWidget::GetViewData(EVCCSimViewType ViewType)
{
    return ViewDataMap.Find(ViewType);
}

const FVCCSimViewData* UVCCSIMDisplayWidget::GetViewData(EVCCSimViewType ViewType) const
{
    return ViewDataMap.Find(ViewType);
}

EVCCSimViewType UVCCSIMDisplayWidget::IDToViewType(int32 ID)
{
    switch (ID)
    {
    case 4: return EVCCSimViewType::RGB;
    case 5: return EVCCSimViewType::Depth;
    case 6: return EVCCSimViewType::Normal;
    case 7: return EVCCSimViewType::Segmentation;
    case 8: return EVCCSimViewType::Lit;
    case 9: return EVCCSimViewType::PointCloud;
    case 0: return EVCCSimViewType::Unit;
    default: return EVCCSimViewType::RGB;
    }
}

int32 UVCCSIMDisplayWidget::ViewTypeToID(EVCCSimViewType ViewType)
{
    switch (ViewType)
    {
    case EVCCSimViewType::RGB: return 4;
    case EVCCSimViewType::Depth: return 5;
    case EVCCSimViewType::Normal: return 6;
    case EVCCSimViewType::Segmentation: return 7;
    case EVCCSimViewType::Lit: return 8;
    case EVCCSimViewType::PointCloud: return 9;
    case EVCCSimViewType::Unit: return 0;
    default: return 4;
    }
}

void UVCCSIMDisplayWidget::InitFromConfig(const struct FVCCSimConfig& Config)
{
    auto SubWindows = Config.VCCSim.SubWindows;
    auto SubWindowsOpacities = Config.VCCSim.SubWindowsOpacities;

    for (int32 i = 0; i < SubWindows.Num(); ++i)
    {
        if (SubWindows[i] == "Lit")
        {
            if (Config.VCCSim.StaticMeshActor.Num() > 0)
            {
                TArray<UStaticMeshComponent*> MeshComponents;
                for (const FString& mesh : Config.VCCSim.StaticMeshActor)
                {
                    auto ActorName = *mesh;
                    for (TActorIterator<AStaticMeshActor> It(GetWorld()); It; ++It)
                    {
                        if (It->ActorHasTag(ActorName))
                        {
                            MeshComponents.Add(It->GetStaticMeshComponent());
                        }
                    }
                    for (TActorIterator<AStaticMeshActor> It(GetWorld()); It; ++It)
                    {
                        if (It->GetName().Contains(ActorName))
                        {
                            MeshComponents.Add(It->GetStaticMeshComponent());
                        }
                    }
                }
                if (MeshComponents.Num() > 0)
                {
                    SetLitMeshComponent(MeshComponents, SubWindowsOpacities[i]);
                }
            }
            else
            {
                UE_LOG(LogVCCSimDisplayWidget, Warning, TEXT("StaticMeshActor not set"));
            }
        }
        else if (SubWindows[i] == "PointCloud")
        {
            if (UInsMeshHolder* InstancedMeshHolder = NewObject<UInsMeshHolder>(Holder))
            {
                InstancedMeshHolder->SetWorldTransform(FTransform::Identity);
                InstancedMeshHolder->RegisterComponent();
                InstancedMeshHolder->CreateStaticMeshes();

                SetPCViewComponent(InstancedMeshHolder->GetInstancedMeshComponentColor(),
                    SubWindowsOpacities[i]);
            }
            else
            {
                UE_LOG(LogVCCSimDisplayWidget, Error, TEXT("Failed to create InstancedMeshHolder"));
            }
        }
        else if (SubWindows[i] == "Unit")
        {
            if (UMeshHandlerComponent* MeshHandlerComponent =
                NewObject<UMeshHandlerComponent>(Holder))
            {
                MeshHandlerComponent->SetWorldTransform(FTransform::Identity);
                MeshHandlerComponent->RegisterComponent();

                SetMeshHandler(MeshHandlerComponent, SubWindowsOpacities[i]);
            }
            else
            {
                UE_LOG(LogVCCSimDisplayWidget, Error, TEXT("Failed to set MeshHandler"));
            }
        }
        else if (SubWindows[i] == "Depth")
        {
            if (DepthImageDisplay)
            {
                DepthImageDisplay->SetOpacity(SubWindowsOpacities[i]);
            }
            else
            {
                UE_LOG(LogVCCSimDisplayWidget, Error, TEXT("DepthImageDisplay not set"));
            }
        }
        else if (SubWindows[i] == "RGB")
        {
            if (RGBImageDisplay)
            {
                RGBImageDisplay->SetOpacity(SubWindowsOpacities[i]);
            }
            else
            {
                UE_LOG(LogVCCSimDisplayWidget, Error, TEXT("RGBImageDisplay not set"));
            }
        }
        else if (SubWindows[i] == "Segmentation")
        {
            if (SegImageDisplay)
            {
                SegImageDisplay->SetOpacity(SubWindowsOpacities[i]);
            }
            else
            {
                UE_LOG(LogVCCSimDisplayWidget, Error, TEXT("SegImageDisplay not set"));
            }
        }
        else if (SubWindows[i] == "Normal")
        {
            if (NormalImageDisplay)
            {
                NormalImageDisplay->SetOpacity(SubWindowsOpacities[i]);
            }
            else
            {
                UE_LOG(LogVCCSimDisplayWidget, Error, TEXT("NormalImageDisplay not set"));
            }
        }
        else
        {
            UE_LOG(LogVCCSimDisplayWidget, Error, TEXT("Unknown SubWindow: %s"),
                *SubWindows[i]);
        }
    }
}

void UVCCSIMDisplayWidget::SetCameraContext(EVCCSimViewType ViewType,
    UTextureRenderTarget2D* RenderTexture, UObject* CameraComponent)
{
    FVCCSimViewData* ViewData = GetViewData(ViewType);
    if (!ViewData)
    {
        UE_LOG(LogVCCSimDisplayWidget, Error, TEXT("Invalid ViewType for SetCameraContext"));
        return;
    }

    if (ViewData->MaterialInstance && RenderTexture)
    {
        ViewData->RenderTarget = RenderTexture;
        ViewData->CameraComponent = CameraComponent;

        const TCHAR* TextureParam = nullptr;
        switch (ViewType)
        {
        case EVCCSimViewType::Depth:
            TextureParam = TEXT("DepthTexture");
            break;
        case EVCCSimViewType::RGB:
            TextureParam = TEXT("RGBTexture");
            break;
        case EVCCSimViewType::Segmentation:
            TextureParam = TEXT("SegTexture");
            break;
        case EVCCSimViewType::Normal:
            TextureParam = TEXT("NormalTexture");
            break;
        default:
            UE_LOG(LogVCCSimDisplayWidget, Warning, TEXT("Unknown texture parameter for ViewType"));
            return;
        }

        ViewData->MaterialInstance->SetTextureParameterValue(TextureParam, ViewData->RenderTarget);
    }
    else
    {
        UE_LOG(LogVCCSimDisplayWidget, Warning,
            TEXT("SetCameraContext failed - Material: %s, Texture: %s"),
            ViewData->MaterialInstance ? TEXT("valid") : TEXT("null"),
            RenderTexture ? TEXT("valid") : TEXT("null"));
    }
}

void UVCCSIMDisplayWidget::SetDepthContext(UTextureRenderTarget2D* DepthTexture, UDepthCameraComponent* InCamera)
{
    SetCameraContext(EVCCSimViewType::Depth, DepthTexture, InCamera);
}

void UVCCSIMDisplayWidget::SetRGBContext(UTextureRenderTarget2D* RGBTexture, URGBCameraComponent* InCamera)
{
    SetCameraContext(EVCCSimViewType::RGB, RGBTexture, InCamera);
}

void UVCCSIMDisplayWidget::SetSegContext(UTextureRenderTarget2D* SegTexture, USegCameraComponent* InCamera)
{
    SetCameraContext(EVCCSimViewType::Segmentation, SegTexture, InCamera);
}

void UVCCSIMDisplayWidget::SetNormalContext(UTextureRenderTarget2D* NormalTexture, UNormalCameraComponent* InCamera)
{
    SetCameraContext(EVCCSimViewType::Normal, NormalTexture, InCamera);
}

void UVCCSIMDisplayWidget::SetLitMeshComponent(
    TArray<UStaticMeshComponent*> MeshComponent, const float& Opacity)
{
    FVCCSimViewData* LitData = GetViewData(EVCCSimViewType::Lit);
    if (!LitData || !LitImageDisplay || !GetWorld())
    {
        UE_LOG(LogVCCSimDisplayWidget, Error, TEXT("LitImageDisplay or World not valid"));
        return;
    }

    LitData->RenderTarget = NewObject<UTextureRenderTarget2D>(this);
    LitData->RenderTarget->InitCustomFormat(LitData->RenderWidth, LitData->RenderHeight, PF_B8G8R8A8, true);
    LitData->RenderTarget->bAutoGenerateMips = false;
    LitData->RenderTarget->UpdateResource();

    LitData->SceneCapture = NewObject<USceneCaptureComponent2D>(MeshComponent[0]);
    LitData->SceneCapture->RegisterComponent();

    LitData->SceneCapture->bCaptureEveryFrame = false;
    LitData->SceneCapture->bCaptureOnMovement = true;
    LitData->SceneCapture->TextureTarget = LitData->RenderTarget;
    LitData->SceneCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;

    FEngineShowFlags& ShowFlags = LitData->SceneCapture->ShowFlags;
    ShowFlags.SetAtmosphere(false);
    ShowFlags.SetFog(false);
    ShowFlags.SetBloom(false);
    ShowFlags.SetAmbientOcclusion(true);
    ShowFlags.SetAntiAliasing(true);
    ShowFlags.SetDynamicShadows(true);
    ShowFlags.SetTemporalAA(false);
    ShowFlags.SetMotionBlur(false);
    ShowFlags.SetGlobalIllumination(false);
    ShowFlags.SetReflectionEnvironment(false);
    ShowFlags.SetDecals(false);

    LitData->SceneCapture->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_UseShowOnlyList;
    for (UStaticMeshComponent* Mesh : MeshComponent)
    {
        LitData->SceneCapture->ShowOnlyComponents.Add(Mesh);
    }

    if (LitData->MaterialInstance)
    {
        LitData->MaterialInstance->SetTextureParameterValue(TEXT("MeshTexture"), LitData->RenderTarget);
        LitData->MaterialInstance->SetScalarParameterValue(TEXT("Opacity"), Opacity);
    }
    else
    {
        UE_LOG(LogVCCSimDisplayWidget, Error, TEXT("LitVisualizationMaterial not set"));
    }
}

void UVCCSIMDisplayWidget::SetPCViewComponent(
    UInstancedStaticMeshComponent* InInstancedMeshComponent, const float& Opacity)
{
    FVCCSimViewData* PCData = GetViewData(EVCCSimViewType::PointCloud);
    if (!PCData || !PCImageDisplay || !GetWorld())
    {
        UE_LOG(LogVCCSimDisplayWidget, Error, TEXT("PCImageDisplay or World not valid"));
        return;
    }

    PCData->RenderTarget = NewObject<UTextureRenderTarget2D>(this);
    PCData->RenderTarget->InitCustomFormat(PCData->RenderWidth, PCData->RenderHeight, PF_B8G8R8A8, true);
    PCData->RenderTarget->bAutoGenerateMips = false;
    PCData->RenderTarget->UpdateResource();

    if (!Holder)
    {
        UE_LOG(LogVCCSimDisplayWidget, Error, TEXT("Holder not valid"));
        return;
    }

    PCData->SceneCapture = NewObject<USceneCaptureComponent2D>(Holder);
    PCData->SceneCapture->RegisterComponent();
    PCData->SceneCapture->bCaptureOnMovement = true;
    PCData->SceneCapture->TextureTarget = PCData->RenderTarget;
    PCData->SceneCapture->bCaptureEveryFrame = false;
    PCData->SceneCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;

    FEngineShowFlags& ShowFlags = PCData->SceneCapture->ShowFlags;
    ShowFlags.SetAtmosphere(false);
    ShowFlags.SetFog(false);
    ShowFlags.SetBloom(false);
    ShowFlags.SetAmbientOcclusion(true);
    ShowFlags.SetAntiAliasing(true);
    ShowFlags.SetDynamicShadows(true);
    ShowFlags.SetTemporalAA(false);
    ShowFlags.SetMotionBlur(false);
    ShowFlags.SetGlobalIllumination(false);
    ShowFlags.SetReflectionEnvironment(false);
    ShowFlags.SetDecals(false);

    PCData->SceneCapture->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_UseShowOnlyList;
    PCData->SceneCapture->ShowOnlyComponents.Add(InInstancedMeshComponent);

    if (PCData->MaterialInstance)
    {
        PCData->MaterialInstance->SetTextureParameterValue(TEXT("MeshTexture"), PCData->RenderTarget);
        PCData->MaterialInstance->SetScalarParameterValue(TEXT("Opacity"), Opacity);
    }
    else
    {
        UE_LOG(LogVCCSimDisplayWidget, Error, TEXT("PCVisualizationMaterial not set"));
    }
}

void UVCCSIMDisplayWidget::SetMeshHandler(UMeshHandlerComponent* InMeshHandler, const float& Opacity)
{
    FVCCSimViewData* UnitData = GetViewData(EVCCSimViewType::Unit);
    if (!UnitData || !UnitImageDisplay || !GetWorld())
    {
        UE_LOG(LogVCCSimDisplayWidget, Error, TEXT("UnitImageDisplay or World not valid"));
        return;
    }

    UnitData->MeshHandler = InMeshHandler;
    UnitData->RenderTarget = NewObject<UTextureRenderTarget2D>(this);
    UnitData->RenderTarget->InitCustomFormat(UnitData->RenderWidth, UnitData->RenderHeight, PF_B8G8R8A8, true);
    UnitData->RenderTarget->bAutoGenerateMips = false;
    UnitData->RenderTarget->UpdateResource();

    UnitData->SceneCapture = NewObject<USceneCaptureComponent2D>(InMeshHandler);
    UnitData->SceneCapture->RegisterComponent();
    UnitData->SceneCapture->bCaptureEveryFrame = false;
    UnitData->SceneCapture->bCaptureOnMovement = true;
    UnitData->SceneCapture->TextureTarget = UnitData->RenderTarget;
    UnitData->SceneCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;

    FEngineShowFlags& ShowFlags = UnitData->SceneCapture->ShowFlags;
    ShowFlags.SetAtmosphere(false);
    ShowFlags.SetFog(false);
    ShowFlags.SetLighting(true);
    ShowFlags.SetPostProcessing(true);
    ShowFlags.SetBloom(false);
    ShowFlags.SetAmbientOcclusion(true);
    ShowFlags.SetAntiAliasing(true);
    ShowFlags.SetDynamicShadows(true);
    ShowFlags.SetTemporalAA(false);
    ShowFlags.SetMotionBlur(false);
    ShowFlags.SetGlobalIllumination(false);
    ShowFlags.SetReflectionEnvironment(false);
    ShowFlags.SetDecals(false);

    UnitData->SceneCapture->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_UseShowOnlyList;
    UnitData->SceneCapture->ShowOnlyComponents.Add(InMeshHandler->GetMeshComponent());

    if (UnitData->MaterialInstance)
    {
        UnitData->MaterialInstance->SetTextureParameterValue(TEXT("MeshTexture"), UnitData->RenderTarget);
        UnitData->MaterialInstance->SetScalarParameterValue(TEXT("Opacity"), Opacity);
    }
    else
    {
        UE_LOG(LogVCCSimDisplayWidget, Error, TEXT("MeshVisualizationMaterial not set"));
    }
}

void UVCCSIMDisplayWidget::RequestCapture(const int32& ID)
{
    if (CurrentQueueSize >= MaxQueuedCaptures)
    {
        UE_LOG(LogVCCSimDisplayWidget, Warning, TEXT("Too many pending captures. Skipping."));
        return;
    }

    CaptureQueue.Enqueue(ID);
    ++CurrentQueueSize;
}

void UVCCSIMDisplayWidget::UpdateViewImage(EVCCSimViewType ViewType, float InDeltaTime)
{
    FVCCSimViewData* ViewData = GetViewData(ViewType);
    if (!ViewData || !ViewData->SceneCapture || !ViewData->RenderTarget || !GetWorld())
    {
        return;
    }

    ViewData->UpdateTimer += InDeltaTime;
    if (ViewData->UpdateTimer < ViewData->UpdateInterval)
    {
        return;
    }
    ViewData->UpdateTimer = 0.0f;

    APlayerController* PlayerController = GetWorld()->GetFirstPlayerController();
    if (!PlayerController || !PlayerController->PlayerCameraManager)
    {
        return;
    }

    FVector ViewLocation = PlayerController->PlayerCameraManager->GetCameraLocation();
    FRotator ViewRotation = PlayerController->PlayerCameraManager->GetCameraRotation();
    ViewData->SceneCapture->SetWorldLocation(ViewLocation);
    ViewData->SceneCapture->SetWorldRotation(ViewRotation);

    const float ViewportFOV = PlayerController->PlayerCameraManager->GetFOVAngle();
    ViewData->SceneCapture->FOVAngle = ViewportFOV;

    ViewData->SceneCapture->CaptureSceneDeferred();
}

void UVCCSIMDisplayWidget::ProcessCapture(const int32 ID)
{
    EVCCSimViewType ViewType = IDToViewType(ID);
    ProcessCaptureByType(ViewType);
}

void UVCCSIMDisplayWidget::ProcessCaptureByType(EVCCSimViewType ViewType)
{
    FVCCSimViewData* ViewData = GetViewData(ViewType);
    if (!ViewData || !ViewData->RenderTarget)
    {
        UE_LOG(LogVCCSimDisplayWidget, Warning,
            TEXT("ProcessCaptureByType: ViewData or RenderTarget not set for ViewType"));
        return;
    }

    FString BaseDir = LogSavePath + "/LiveCaptures/";
    FFileManagerGeneric FileManager;
    if (!FileManager.MakeDirectory(*BaseDir, true))
    {
        UE_LOG(LogVCCSimDisplayWidget, Error, TEXT("Failed to create capture directory."));
        return;
    }

    FString Timestamp = FDateTime::Now().ToString();

    switch (ViewType)
    {
    case EVCCSimViewType::RGB:
    {
        auto* Camera = Cast<URGBCameraComponent>(ViewData->CameraComponent);
        if (!Camera) return;
        FString FilePath = BaseDir + TEXT("RGBCapture_") + Timestamp + TEXT(".png");
        FIntPoint Size(Camera->Width, Camera->Height);
        Camera->AsyncGetRGBImageData([FilePath, Size](const TArray<FColor>& ImageData)
        {
            (new FAutoDeleteAsyncTask<FAsyncImageSaveTask>(ImageData, Size, FilePath))
                ->StartBackgroundTask();
        });
        return;
    }
    case EVCCSimViewType::Depth:
    {
        auto* Camera = Cast<UDepthCameraComponent>(ViewData->CameraComponent);
        if (!Camera) return;
        FString FilePath = BaseDir + TEXT("DepthCapture_") + Timestamp + TEXT(".png");
        FIntPoint Size(Camera->Width, Camera->Height);
        Camera->AsyncGetDepthImageData([FilePath, Size](const TArray<FFloat16Color>& ImageData)
        {
            TArray<float> DepthPixelsFloat;
            DepthPixelsFloat.SetNum(ImageData.Num());
            for (int32 i = 0; i < ImageData.Num(); ++i)
            {
                DepthPixelsFloat[i] = ImageData[i].R.GetFloat();
            }
            (new FAutoDeleteAsyncTask<FAsyncDepthSaveTask>(DepthPixelsFloat, Size, FilePath))
                ->StartBackgroundTask();
        });
        return;
    }
    case EVCCSimViewType::Normal:
    {
        auto* Camera = Cast<UNormalCameraComponent>(ViewData->CameraComponent);
        if (!Camera) return;
        FString FilePath = BaseDir + TEXT("NormalCapture_") + Timestamp + TEXT(".exr");
        FIntPoint Size(Camera->Width, Camera->Height);
        Camera->AsyncGetNormalImageData([FilePath, Size](const TArray<FFloat16Color>& ImageData)
        {
            (new FAutoDeleteAsyncTask<FAsyncNormalEXRSaveTask>(ImageData, Size, FilePath))
                ->StartBackgroundTask();
        });
        return;
    }
    case EVCCSimViewType::Segmentation:
    {
        auto* Camera = Cast<USegCameraComponent>(ViewData->CameraComponent);
        if (!Camera) return;
        FString FilePath = BaseDir + TEXT("SegmentationCapture_") + Timestamp + TEXT(".png");
        FIntPoint Size(Camera->Width, Camera->Height);
        Camera->AsyncGetSegmentationImageData([FilePath, Size](const TArray<FColor>& ImageData)
        {
            (new FAutoDeleteAsyncTask<FAsyncImageSaveTask>(ImageData, Size, FilePath))
                ->StartBackgroundTask();
        });
        return;
    }
    case EVCCSimViewType::Lit:
        SaveRenderTargetToDisk(ViewData->RenderTarget, TEXT("LitCapture"));
        break;
    case EVCCSimViewType::PointCloud:
        SaveRenderTargetToDisk(ViewData->RenderTarget, TEXT("PCCapture"));
        break;
    case EVCCSimViewType::Unit:
        SaveRenderTargetToDisk(ViewData->RenderTarget, TEXT("UnitCapture"));
        break;
    default:
        UE_LOG(LogVCCSimDisplayWidget, Warning, TEXT("Invalid ViewType: %d"), (int32)ViewType);
        break;
    }
}

void UVCCSIMDisplayWidget::SaveRenderTargetToDisk(
    UTextureRenderTarget2D* RenderTarget, const FString& FileName) const
{
    if (!RenderTarget)
    {
        UE_LOG(LogVCCSimDisplayWidget, Error, TEXT("RenderTarget is null. Cannot save to disk."));
        return;
    }

    FTextureRenderTargetResource* RTResource =
        RenderTarget->GameThread_GetRenderTargetResource();
    if (!RTResource)
    {
        UE_LOG(LogVCCSimDisplayWidget, Error,
            TEXT("RenderTargetResource is null. Cannot save to disk."));
        return;
    }

    FIntPoint Size = RTResource->GetSizeXY();
    FString FilePath = LogSavePath + "/LiveCaptures/" + FileName + "_" +
        FDateTime::Now().ToString() + ".png";

    FFileManagerGeneric FileManager;
    if (!FileManager.MakeDirectory(*FPaths::GetPath(FilePath), true))
    {
        UE_LOG(LogVCCSimDisplayWidget, Error, TEXT("Failed to create directory for render target."));
        return;
    }

    TArray<FColor> Pixels;
    Pixels.SetNum(Size.X * Size.Y);
    if (!RTResource->ReadPixels(Pixels))
    {
        UE_LOG(LogVCCSimDisplayWidget, Error, TEXT("Failed to read pixels from RenderTarget."));
        return;
    }
    for (FColor& Color : Pixels)
    {
        Color.A = 255;
    }

    (new FAutoDeleteAsyncTask<FAsyncImageSaveTask>(Pixels, Size, FilePath))->StartBackgroundTask();
}