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

#include "Sensors/NormalCamera.h"
#include "Simulation/Recorder.h"
#include "RenderingThread.h"
#include "Async/AsyncWork.h"
#include "Windows/WindowsHWrapper.h"
#include "RHI.h"

UNormalCameraComponent::UNormalCameraComponent()
    : FOV(90.0f)
    , Width(1920)
    , Height(1080)
    , TimeSinceLastCapture(0.0f)
{
    PrimaryComponentTick.bCanEverTick = true;
}

void UNormalCameraComponent::RConfigure(
    const FNormalCameraConfig& Config, ARecorder* Recorder)
{  
    FOV = Config.FOV;
    Width = Config.Width;
    Height = Config.Height;
    
    InitializeRenderTargets();
    SetCaptureComponent();

    if (Config.RecordInterval > 0)
    {
        ParentActor = GetOwner();
        RecorderPtr = Recorder;
        RecordInterval = Config.RecordInterval;
        RecordState = Recorder->RecordState;
        
        Recorder->OnRecordStateChanged.AddDynamic(this,
            &UNormalCameraComponent::SetRecordState);
        SetComponentTickEnabled(true);
        bRecorded = true;
    }
    else
    {
        SetComponentTickEnabled(false);
    }
    bBPConfigured = true;
}

void UNormalCameraComponent::SetCaptureComponent() const
{
    if (CaptureComponent)
    {
        CaptureComponent->ProjectionType = ECameraProjectionMode::Perspective;
        CaptureComponent->FOVAngle = FOV;
        CaptureComponent->CaptureSource = ESceneCaptureSource::SCS_Normal;
        CaptureComponent->bCaptureEveryFrame = false;
        CaptureComponent->bCaptureOnMovement = false;
    }
    else 
    {
        UE_LOG(LogTemp, Error, TEXT("Capture component not initialized!"));
    }
}

void UNormalCameraComponent::BeginPlay()
{
    Super::BeginPlay();
    InitializeRenderTargets();
    SetCaptureComponent();
    
    SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
    SetCollisionResponseToAllChannels(ECR_Ignore);
    SetSimulatePhysics(true);
}

void UNormalCameraComponent::OnComponentCreated()
{
    Super::OnComponentCreated();
    
    // Initialize capture component
    CaptureComponent = NewObject<USceneCaptureComponent2D>(this);
    CaptureComponent->AttachToComponent(this,
        FAttachmentTransformRules::SnapToTargetIncludingScale);
    
    SetCaptureComponent();
}

void UNormalCameraComponent::TickComponent(float DeltaTime, ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    if (bRecorded && RecordState)
    {
        TimeSinceLastCapture += DeltaTime;
        if (TimeSinceLastCapture >= RecordInterval)
        {
            TimeSinceLastCapture = 0.0f;
            CaptureScene();
            
            if (RecorderPtr)
            {
                FNormalCameraData NormalCameraData;
                NormalCameraData.Timestamp = FPlatformTime::Seconds();
                NormalCameraData.SensorIndex = CameraIndex;
                NormalCameraData.Width = Width;
                NormalCameraData.Height = Height;
                while(!Dirty)
                {
                    FPlatformProcess::Sleep(0.01f);
                }
                NormalCameraData.Data = GetNormalImage();
                Dirty = false;
                RecorderPtr->SubmitNormalData(ParentActor, MoveTemp(NormalCameraData));
            }
        }
    }
}

void UNormalCameraComponent::InitializeRenderTargets()
{
    // Use floating point format for high precision normals
    NormalRenderTarget = NewObject<UTextureRenderTarget2D>(this);
    NormalRenderTarget->InitCustomFormat(Width, Height,
        PF_A32B32G32R32F, true);  // 32-bit float per channel
    NormalRenderTarget->UpdateResource();
    
    if (CaptureComponent)
    {
        CaptureComponent->TextureTarget = NormalRenderTarget;
    }
}

bool UNormalCameraComponent::CheckComponentAndRenderTarget() const
{
    if (!CaptureComponent || !NormalRenderTarget)
    {
        UE_LOG(LogTemp, Error, TEXT("Capture component or "
                                    "render target not initialized!"));
        return true;
    }
    return false;
}

void UNormalCameraComponent::CaptureScene()
{
    if (CheckComponentAndRenderTarget())
    {
        UE_LOG(LogTemp, Error, TEXT("UNormalCameraComponent: "
                                    "Capture component or render target not initialized!"));
        return;
    }
    
    CaptureComponent->CaptureScene();
    
    ProcessNormalTexture([this](const TArray<FLinearColor>& ImageData)
        {
            NormalData = ImageData;
            Dirty = true;
        });
}

void UNormalCameraComponent::ProcessNormalTexture(TFunction<void(const TArray<FLinearColor>&)> OnComplete)
{
    // Get the render target resource
    FTextureRenderTargetResource* RenderTargetResource = 
        NormalRenderTarget->GameThread_GetRenderTargetResource();
    
    if (!RenderTargetResource)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to get render target resource!"));
        return;
    }

    // Prepare normal data array
    TArray<FLinearColor>* NormalDataPtr = new TArray<FLinearColor>();
    NormalDataPtr->Empty(Width * Height);
    NormalDataPtr->SetNumUninitialized(Width * Height);

    // Define context for reading surface data
    struct FReadSurfaceContext
    {
        TArray<FLinearColor>* OutData;
        FTextureRenderTargetResource* RenderTarget;
        FIntRect Rect;
    };

    FReadSurfaceContext Context = {
        NormalDataPtr,
        RenderTargetResource,
        FIntRect(0, 0, Width, Height)
    };

    auto SharedCallback = MakeShared<TFunction<void(const TArray<FLinearColor>&)>>(OnComplete);
    
    // Submit the render thread command
    ENQUEUE_RENDER_COMMAND(ReadNormalSurfaceCommand)(
        [Context, SharedCallback](FRHICommandListImmediate& RHICmdList)
    {
        // Read the float data directly into FLinearColor array
        RHICmdList.ReadSurfaceData(
            Context.RenderTarget->GetRenderTargetTexture(),
            Context.Rect,
            *Context.OutData,
            FReadSurfaceDataFlags()
        );
        
        // Call the callback with the data and then clean up
        (*SharedCallback)(*Context.OutData);
        delete Context.OutData;
    });
}

TArray<FLinearColor> UNormalCameraComponent::GetNormalImage()
{
    if (NormalData.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("GetNormalImage: No normal data available!"));
        return TArray<FLinearColor>();
    }
    
    return NormalData;
}

void UNormalCameraComponent::AsyncGetNormalImageData(
    TFunction<void(const TArray<FLinearColor>&)> Callback)
{
    AsyncTask(ENamedThreads::GameThread, [this, Callback = MoveTemp(Callback)]() {
        if (CheckComponentAndRenderTarget())
        {
            Callback({});
            return;
        }
        
        CaptureComponent->CaptureScene();
        ProcessNormalTexture(Callback);
    });
}