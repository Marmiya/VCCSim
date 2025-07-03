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
#include "Sensors/CameraSensor.h"
#include "Sensors/DepthCamera.h"
#include "Sensors/SegmentCamera.h"
#include "Sensors/NormalCamera.h"
#include "EngineUtils.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/World.h"

#include "Windows/WindowsHWrapper.h" // Deal with UpdateResourceW error
#include "HAL/FileManagerGeneric.h"


void UVCCSIMDisplayWidget::NativeConstruct()
{
    Super::NativeConstruct();
    
    if (DepthVisualizationMaterial && DepthImageDisplay)
    {
        DepthMaterial = UMaterialInstanceDynamic::Create(
            DepthVisualizationMaterial, this);
        DepthImageDisplay->SetBrushFromMaterial(DepthMaterial);

        DepthMaterial->SetScalarParameterValue("MinDepth", MinDepth);
        DepthMaterial->SetScalarParameterValue("MaxDepth", MaxDepth);
        DepthMaterial->SetScalarParameterValue("Contrast", Contrast);
        DepthMaterial->SetScalarParameterValue("Brightness", Brightness);
    }
    if (RGBVisualizationMaterial && RGBImageDisplay)
    {
        RGBMaterial = UMaterialInstanceDynamic::Create(
            RGBVisualizationMaterial, this);
        RGBImageDisplay->SetBrushFromMaterial(RGBMaterial);
    }
    if (SegVisualizationMaterial && SegImageDisplay)
    {
        SegMaterial = UMaterialInstanceDynamic::Create(
            SegVisualizationMaterial, this);
        SegImageDisplay->SetBrushFromMaterial(SegMaterial);
    }
    if (NormalVisualizationMaterial && NormalImageDisplay)
    {
        NormalMaterial = UMaterialInstanceDynamic::Create(
            NormalVisualizationMaterial, this);
        NormalImageDisplay->SetBrushFromMaterial(NormalMaterial);
    }
}

void UVCCSIMDisplayWidget::NativeTick(const FGeometry& MyGeometry, float InDeltaTime)
{
    Super::NativeTick(MyGeometry, InDeltaTime);

    if (LitImageDisplay->IsVisible())
    {
        UpdateLitImage(InDeltaTime);
    }

    if (PCImageDisplay->IsVisible())
    {
        UpdatePCImage(InDeltaTime);
    }
    
    if (UnitImageDisplay->IsVisible())
    {
        UpdateMeshImage(InDeltaTime);
    }

    for (int i = 0; i < 6; ++i)
    {
        int32 ID;
        if (CaptureQueue.Dequeue(ID))
        {
            CurrentQueueSize--;
            switch (ID)
            {
            case 5:
                ProcessCapture(5);
                break;
            case 6:
                ProcessCapture(6);
                break;
            case 7:
                UpdateLitImage(InDeltaTime);
                ProcessCapture(7);
                break;
            case 8:
                UpdatePCImage(InDeltaTime);
                ProcessCapture(8);
                break;
            case 9:
                UpdateMeshImage(InDeltaTime);
                ProcessCapture(9);
                break;
            case 0:
                ProcessCapture(0);
                break;
            default:
                break;
            }
        }
    }
}

void UVCCSIMDisplayWidget::InitFromConfig(const struct FVCCSimConfig& Config)
{
    auto SubWindows = Config.VCCSim.SubWindows;
    auto SubWindowsOpacities = Config.VCCSim.SubWindowsOpacities;
    
    for (int i = 0; i < SubWindows.size(); ++i)
    {
        if (SubWindows[i] == "Lit")
        {
            if (!Config.VCCSim.StaticMeshActor.empty())
            {
                TArray<UStaticMeshComponent*> MeshComponents;
                for (const std::string& mesh : Config.VCCSim.StaticMeshActor)
                {
                    auto ActorName = mesh.c_str();
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
                UE_LOG(LogTemp, Warning, TEXT("StaticMeshActor not set"));
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
                UE_LOG(LogTemp, Error, TEXT("Failed to create InstancedMeshHolder"));
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
                UE_LOG(LogTemp, Error, TEXT("Failed to set MeshHandler"));
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
                UE_LOG(LogTemp, Error, TEXT("DepthImageDisplay not set"));
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
                UE_LOG(LogTemp, Error, TEXT("RGBImageDisplay not set"));
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
                UE_LOG(LogTemp, Error, TEXT("SegImageDisplay not set"));
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
                UE_LOG(LogTemp, Error, TEXT("NormalImageDisplay not set"));
            }
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("Unknown SubWindow: %s"), 
                *FString(SubWindows[i].c_str()));
        }
    }
}

void UVCCSIMDisplayWidget::SetDepthContext(
    UTextureRenderTarget2D* DepthTexture, UDepthCameraComponent* InCamera)
{    
    if (DepthMaterial && DepthTexture)
    {
        DepthRenderTarget = DepthTexture;
        DepthMaterial->SetTextureParameterValue(TEXT("DepthTexture"), DepthRenderTarget);
        DepthCameraComponent = InCamera;
    }
    else
    {
        UE_LOG(LogTemp, Warning, 
            TEXT("SetDepthTexture failed - Material: %s, Texture: %s"),
            DepthMaterial ? TEXT("valid") : TEXT("null"),
            DepthTexture ? TEXT("valid") : TEXT("null"));
    }
}

void UVCCSIMDisplayWidget::SetRGBContext(
    UTextureRenderTarget2D* RGBTexture, URGBCameraComponent* InCamera)
{
    if (RGBMaterial && RGBTexture)
    {
        RGBRenderTarget = RGBTexture;
        RGBMaterial->SetTextureParameterValue(TEXT("RGBTexture"), RGBRenderTarget);
        RGBCameraComponent = InCamera;
    }
    else
    {
        UE_LOG(LogTemp, Warning, 
            TEXT("SetRGBTexture failed - Material: %s, Texture: %s"),
            RGBMaterial ? TEXT("valid") : TEXT("null"),
            RGBTexture ? TEXT("valid") : TEXT("null"));
    }
}

void UVCCSIMDisplayWidget::SetSegContext(
    UTextureRenderTarget2D* SegTexture, USegmentationCameraComponent* InCamera)
{
    if (SegMaterial && SegTexture)
    {
        SegRenderTarget = SegTexture;
        SegMaterial->SetTextureParameterValue(TEXT("SegTexture"), SegRenderTarget);
        SegCameraComponent = InCamera;
    }
    else
    {
        UE_LOG(LogTemp, Warning, 
            TEXT("SetSegTexture failed - Material: %s, Texture: %s"),
            SegMaterial ? TEXT("valid") : TEXT("null"),
            SegTexture ? TEXT("valid") : TEXT("null"));
    }
}

void UVCCSIMDisplayWidget::SetNormalContext(
    UTextureRenderTarget2D* NormalTexture, UNormalCameraComponent* InCamera)
{
    if (NormalMaterial && NormalTexture)
    {
        NormalRenderTarget = NormalTexture;
        NormalMaterial->SetTextureParameterValue(TEXT("NormalTexture"), NormalRenderTarget);
        NormalCameraComponent = InCamera;
    }
    else
    {
        UE_LOG(LogTemp, Warning, 
            TEXT("SetNormalTexture failed - Material: %s, Texture: %s"),
            NormalMaterial ? TEXT("valid") : TEXT("null"),
            NormalTexture ? TEXT("valid") : TEXT("null"));
    }
}

void UVCCSIMDisplayWidget::SetLitMeshComponent(
    TArray<UStaticMeshComponent*> MeshComponent, const float& Opacity)
{
    if (!LitImageDisplay || !GetWorld())
    {
        UE_LOG(LogTemp, Error, TEXT("LitImageDisplay or World not valid"));
        return;
    }

    LitRenderTarget = NewObject<UTextureRenderTarget2D>(this);
    LitRenderTarget->InitCustomFormat(LitRenderWidth, LitRenderHeight,
        PF_B8G8R8A8, true);
    LitRenderTarget->bAutoGenerateMips = false;
    LitRenderTarget->UpdateResource();

    LitSceneCapture = NewObject<USceneCaptureComponent2D>(MeshComponent[0]);
    LitSceneCapture->RegisterComponent();
    
    LitSceneCapture->bCaptureEveryFrame = false;
    LitSceneCapture->bCaptureOnMovement = true;
    LitSceneCapture->TextureTarget = LitRenderTarget;
    LitSceneCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;

    FEngineShowFlags& ShowFlags = LitSceneCapture->ShowFlags;
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

    LitSceneCapture->PrimitiveRenderMode =
        ESceneCapturePrimitiveRenderMode::PRM_UseShowOnlyList;
    for (UStaticMeshComponent* Mesh : MeshComponent)
    {
        LitSceneCapture->ShowOnlyComponents.Add(Mesh);
    }
    
    if (LitVisualizationMaterial)
    {
        LitMaterial = UMaterialInstanceDynamic::Create(LitVisualizationMaterial, this);
        LitImageDisplay->SetBrushFromMaterial(LitMaterial);
        
        if (LitMaterial)
        {
            LitMaterial->SetTextureParameterValue(TEXT("MeshTexture"), LitRenderTarget);
            LitMaterial->SetScalarParameterValue(TEXT("Opacity"), Opacity);
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to create LitMaterial"));
        }
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("LitVisualizationMaterial not set"));
    }
}

void UVCCSIMDisplayWidget::SetPCViewComponent(
    UInstancedStaticMeshComponent* InInstancedMeshComponent, const float& Opacity)
{
    if (!PCImageDisplay || !GetWorld())
    {
        UE_LOG(LogTemp, Error, TEXT("PCImageDisplay or World not valid"));
        return;
    }

    PCRenderTarget = NewObject<UTextureRenderTarget2D>(this);
    PCRenderTarget->InitCustomFormat(PCRenderWidth, PCRenderHeight,
        PF_B8G8R8A8, true);
    PCRenderTarget->bAutoGenerateMips = false;
    PCRenderTarget->UpdateResource();

    if (!Holder)
    {
        UE_LOG(LogTemp, Error, TEXT("Holder not valid"));
        return;
    }
    PCSceneCapture = NewObject<USceneCaptureComponent2D>(Holder);
    PCSceneCapture->RegisterComponent();

    PCSceneCapture->bCaptureOnMovement = true;
    PCSceneCapture->TextureTarget = PCRenderTarget;
    PCSceneCapture->bCaptureEveryFrame = false;
    PCSceneCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;

    FEngineShowFlags& ShowFlags = PCSceneCapture->ShowFlags;
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

    PCSceneCapture->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_UseShowOnlyList;
    PCSceneCapture->ShowOnlyComponents.Add(InInstancedMeshComponent);
    
    if (PCVisualizationMaterial)
    {
        PCMaterial = UMaterialInstanceDynamic::Create(PCVisualizationMaterial, this);
        PCImageDisplay->SetBrushFromMaterial(PCMaterial);
        
        if (PCMaterial)
        {
            PCMaterial->SetTextureParameterValue(TEXT("MeshTexture"), PCRenderTarget);
            PCMaterial->SetScalarParameterValue(TEXT("Opacity"), Opacity);
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to create PCMaterial"));
        }
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("PCVisualizationMaterial not set"));
    }
}

void UVCCSIMDisplayWidget::SetMeshHandler(UMeshHandlerComponent* InMeshHandler,
    const float& Opacity)
{
    MeshHandler = InMeshHandler;
    if (!UnitImageDisplay || !GetWorld())
    {
        UE_LOG(LogTemp, Error, TEXT("UnitImageDisplay or World not valid"));
        return;
    }

    MeshRenderTarget = NewObject<UTextureRenderTarget2D>(this);
    MeshRenderTarget->InitCustomFormat(MeshRenderWidth, MeshRenderHeight,
        PF_B8G8R8A8, true);
    MeshRenderTarget->bAutoGenerateMips = false;
    MeshRenderTarget->UpdateResource();

    MeshSceneCapture = NewObject<USceneCaptureComponent2D>(MeshHandler);
    MeshSceneCapture->RegisterComponent();  
    
    MeshSceneCapture->bCaptureEveryFrame = false;
    MeshSceneCapture->bCaptureOnMovement = true;
    MeshSceneCapture->TextureTarget = MeshRenderTarget;
    MeshSceneCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
    
    FEngineShowFlags& ShowFlags = MeshSceneCapture->ShowFlags;
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

    // Set up primitive showing settings
    MeshSceneCapture->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_UseShowOnlyList;
    MeshSceneCapture->ShowOnlyComponents.Add(MeshHandler->GetMeshComponent());

    // Create and setup material for UnitImageDisplay
    if (MeshVisualizationMaterial)
    {
        MeshMaterial = UMaterialInstanceDynamic::Create(MeshVisualizationMaterial, this);
        UnitImageDisplay->SetBrushFromMaterial(MeshMaterial);
        
        if (MeshMaterial)
        {
            MeshMaterial->SetTextureParameterValue(TEXT("MeshTexture"), MeshRenderTarget);
            MeshMaterial->SetScalarParameterValue(TEXT("Opacity"), Opacity);
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to create MeshMaterial"));
        }
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("MeshVisualizationMaterial not set"));
    }
}

void UVCCSIMDisplayWidget::RequestCapture(const int& ID)
{    
    if (CurrentQueueSize >= MaxQueuedCaptures)
    {
        UE_LOG(LogTemp, Warning, TEXT("Too many pending captures. Skipping."));
        return;
    }

    CaptureQueue.Enqueue(ID);
    ++CurrentQueueSize;
}

void UVCCSIMDisplayWidget::UpdateLitImage(float InDeltaTime)
{
    LitUpdateTimer += InDeltaTime;

    if (LitUpdateTimer >= LitUpdateInterval)
    {
        LitUpdateTimer = 0.0f;
    }
    else
    {
        return;
    }
    
    if (!LitSceneCapture || !LitRenderTarget || !GetWorld())
    {
        UE_LOG(LogTemp, Error, TEXT("Required components not set. "
                                    "LitSceneCapture: %s, LitRenderTarget: %s"),
               LitSceneCapture ? TEXT("Valid") : TEXT("Invalid"),
               LitRenderTarget ? TEXT("Valid") : TEXT("Invalid"));
        return;
    }

    // Get the actual viewport camera transform
    APlayerController* PlayerController = GetWorld()->GetFirstPlayerController();
    if (!PlayerController)
    {
        UE_LOG(LogTemp, Error, TEXT("PlayerController not found"));
        return;
    }

    if (PlayerController->PlayerCameraManager)
    {
        FVector ViewLocation = PlayerController->PlayerCameraManager->GetCameraLocation();
        FRotator ViewRotation = PlayerController->PlayerCameraManager->GetCameraRotation();
        LitSceneCapture->SetWorldLocation(ViewLocation);
        LitSceneCapture->SetWorldRotation(ViewRotation);
    }

    const float ViewportFOV = PlayerController->PlayerCameraManager
        ? PlayerController->PlayerCameraManager->GetFOVAngle()
        : 90.0f;
    LitSceneCapture->FOVAngle = ViewportFOV;
    
    LitSceneCapture->CaptureScene();
}

void UVCCSIMDisplayWidget::UpdatePCImage(float InDeltaTime)
{
    PCUpdateTimer += InDeltaTime;
    if (PCUpdateTimer >= PCUpdateInterval)
    {
        PCUpdateTimer = 0.0f;
    }
    else
    {
        return;
    }
    
    if (!PCSceneCapture || !PCRenderTarget || !GetWorld())
    {
        UE_LOG(LogTemp, Error, TEXT("Required components not set. "
                                    "PCSceneCapture: %s, PCRenderTarget: %s"),
               PCSceneCapture ? TEXT("Valid") : TEXT("Invalid"),
               PCRenderTarget ? TEXT("Valid") : TEXT("Invalid"));
        return;
    }

    // Get the actual viewport camera transform
    APlayerController* PlayerController = GetWorld()->GetFirstPlayerController();
    if (!PlayerController)
    {
        UE_LOG(LogTemp, Error, TEXT("PlayerController not found"));
        return;
    }

    if (PlayerController->PlayerCameraManager)
    {
        FVector ViewLocation = PlayerController->PlayerCameraManager->GetCameraLocation();
        FRotator ViewRotation = PlayerController->PlayerCameraManager->GetCameraRotation();
        PCSceneCapture->SetWorldLocation(ViewLocation);
        PCSceneCapture->SetWorldRotation(ViewRotation);
    }

    const float ViewportFOV = PlayerController->PlayerCameraManager
        ? PlayerController->PlayerCameraManager->GetFOVAngle()
        : 90.0f;
    PCSceneCapture->FOVAngle = ViewportFOV;
    
    PCSceneCapture->CaptureScene();
}

void UVCCSIMDisplayWidget::UpdateMeshImage(float InDeltaTime)
{
    MeshUpdateTimer += InDeltaTime;
    if (MeshUpdateTimer >= MeshUpdateInterval)
    {
        MeshUpdateTimer = 0.0f;
    }
    else
    {
        return;
    }    
    
    if (!MeshHandler || !MeshSceneCapture || !MeshRenderTarget || !GetWorld())
    {
        UE_LOG(LogTemp, Error, TEXT("Required components not set. "
                                    "MeshHandler: %s, SceneCapture: %s, RenderTarget: %s"), 
               MeshHandler ? TEXT("Valid") : TEXT("Invalid"),
               MeshSceneCapture ? TEXT("Valid") : TEXT("Invalid"),
               MeshRenderTarget ? TEXT("Valid") : TEXT("Invalid"));
        return;
    }

    // Get the actual viewport camera transform
    APlayerController* PlayerController = GetWorld()->GetFirstPlayerController();
    if (!PlayerController)
    {
        UE_LOG(LogTemp, Error, TEXT("PlayerController not found"));
        return;
    }

    FVector ViewLocation;
    FRotator ViewRotation;
    PlayerController->GetPlayerViewPoint(ViewLocation, ViewRotation);
    
    // Update scene capture to match the actual view
    MeshSceneCapture->SetWorldLocation(ViewLocation);
    MeshSceneCapture->SetWorldRotation(ViewRotation);

    // Get the FOV from the viewport
    const float ViewportFOV = PlayerController->PlayerCameraManager 
        ? PlayerController->PlayerCameraManager->GetFOVAngle()
        : 90.0f;
    
    MeshSceneCapture->FOVAngle = ViewportFOV;

    MeshSceneCapture->CaptureScene();
}

void UVCCSIMDisplayWidget::ProcessCapture(const int32 ID)
{
    // Update and capture the image
    switch (ID)
    {
    case 5:
        if (SegRenderTarget)
        {
            SegCameraComponent->CaptureSegmentationScene();
            SaveRenderTargetToDisk(SegRenderTarget, "SegmentationCapture");
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("UVCCSIMDisplayWidget::ProcessCapture: "
                                          "SegRenderTarget not set"));
        }
        break;
    case 6:
        if (RGBRenderTarget)
        {
            RGBCameraComponent->CaptureRGBScene();
            SaveRenderTargetToDisk(RGBRenderTarget, "RGBCapture");
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("UVCCSIMDisplayWidget::ProcessCapture: "
                                          "RGBRenderTarget not set"));
        }
        break;
    case 7:
        if (LitSceneCapture && LitRenderTarget)
        {
            SaveRenderTargetToDisk(LitRenderTarget, "LitCapture");
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("UVCCSIMDisplayWidget::ProcessCapture: "
                                          "LitSceneCapture or LitRenderTarget not set"));
        }
        break;
    case 8:
        if (PCSceneCapture && PCRenderTarget)
        {
            SaveRenderTargetToDisk(PCRenderTarget, "PCCapture");
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("UVCCSIMDisplayWidget::ProcessCapture: "
                                          "PCSceneCapture or PCRenderTarget not set"));
        }
        break;
    case 9:
        if (MeshSceneCapture && MeshRenderTarget)
        {
            SaveRenderTargetToDisk(MeshRenderTarget, "MeshCapture");
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("UVCCSIMDisplayWidget::ProcessCapture: "
                                          "MeshSceneCapture or MeshRenderTarget not set"));
        }
        break;
    case 0:
        if (DepthRenderTarget)
        {
            DepthCameraComponent->CaptureDepthScene();
            SaveRenderTargetToDisk(DepthRenderTarget, "DepthCapture");
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("UVCCSIMDisplayWidget::ProcessCapture: "
                                          "DepthRenderTarget not set"));
        }
        break;
    default:
        UE_LOG(LogTemp, Warning, TEXT("Invalid ID: %d"), ID);
    }
}

void UVCCSIMDisplayWidget::SaveRenderTargetToDisk(
    UTextureRenderTarget2D* RenderTarget, const FString& FileName) const
{
    if (!RenderTarget)
    {
        UE_LOG(LogTemp, Error, TEXT("RenderTarget is null. Cannot save to disk."));
        return;
    }

    FTextureRenderTargetResource* RTResource =
        RenderTarget->GameThread_GetRenderTargetResource();
    if (!RTResource)
    {
        UE_LOG(LogTemp, Error,
            TEXT("RenderTargetResource is null. Cannot save to disk."));
        return;
    }

    FIntPoint Size = RTResource->GetSizeXY();
    TArray<FColor> Pixels;
    Pixels.SetNum(Size.X * Size.Y);

    bool bReadPixels = RTResource->ReadPixels(Pixels);
    if (!bReadPixels)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to read pixels from RenderTarget."));
        return;
    }

    if (FileName == TEXT("SegmentationCapture"))
    {
        for (FColor& Pixel : Pixels)
        {
            Pixel.A = 255; // Set alpha to 255 for segmentation
        }
    }

    auto CurTime = FDateTime::Now();
    FString FilePath = LogSavePath + "/LiveCaptures/" + FileName + "_" + CurTime.ToString() + ".png";

    // Create directory synchronously as it's typically quick
    FFileManagerGeneric FileManager;
    if (!FileManager.MakeDirectory(*FPaths::GetPath(FilePath), true))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to create directory for render target."));
        return;
    }

    // Start async task to save the image
    (new FAutoDeleteAsyncTask<FAsyncImageSaveTask>(
        Pixels, Size, FilePath))->StartBackgroundTask();
}