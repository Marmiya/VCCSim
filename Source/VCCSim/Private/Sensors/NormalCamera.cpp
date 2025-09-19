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

DEFINE_LOG_CATEGORY_STATIC(LogNormalCamera, Log, All);

#include "Sensors/NormalCamera.h"
#include "Simulation/Recorder.h"
#include "RenderingThread.h"
#include "Async/AsyncWork.h"
#include "Windows/WindowsHWrapper.h"
#include "RHI.h"

UNormalCameraComponent::UNormalCameraComponent()
{
}

void UNormalCameraComponent::RConfigure(
    const FNormalCameraConfig& Config, ARecorder* Recorder)
{
    FOV = Config.FOV;
    Width = Config.Width;
    Height = Config.Height;

    ComputeIntrinsics();
    SetCaptureComponent();
    InitializeRenderTargets();

    if (Config.RecordInterval > 0)
    {
        RecordInterval = Config.RecordInterval;
        SetupRecorder(Recorder);
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



void UNormalCameraComponent::OnRecordTick()
{
    CaptureScene();

    if (RecorderPtr)
    {
        FNormalCameraData NormalCameraData;
        NormalCameraData.Timestamp = FPlatformTime::Seconds();
        NormalCameraData.SensorIndex = GetSensorIndex();
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

void UNormalCameraComponent::SetCaptureComponent() const
{
    Super::SetCaptureComponent();

    if (CaptureComponent)
    {
        CaptureComponent->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives;
        CaptureComponent->CaptureSource = ESceneCaptureSource::SCS_Normal;
    }
}

void UNormalCameraComponent::InitializeRenderTargets()
{
    NormalRenderTarget = NewObject<UTextureRenderTarget2D>(this);
    NormalRenderTarget->InitCustomFormat(Width, Height,
        PF_A32B32G32R32F, true);
    NormalRenderTarget->UpdateResource();
}


void UNormalCameraComponent::CaptureScene()
{
    if (!CheckComponentAndRenderTarget())
    {
        return;
    }
    
    CaptureComponent->CaptureScene();
    
    ProcessNormalTexture([this](const TArray<FLinearColor>& ImageData)
        {
            NormalData = ImageData;
            Dirty = true;
        });
}

void UNormalCameraComponent::ProcessNormalTexture(
    TFunction<void(const TArray<FLinearColor>&)> OnComplete)
{
    // Get the render target resource
    FTextureRenderTargetResource* RenderTargetResource = 
        NormalRenderTarget->GameThread_GetRenderTargetResource();
    
    if (!RenderTargetResource)
    {
        UE_LOG(LogNormalCamera, Error, TEXT("Failed to get render target resource!"));
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

    auto SharedCallback =
        MakeShared<TFunction<void(const TArray<FLinearColor>&)>>(OnComplete);
    
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
        UE_LOG(LogNormalCamera, Warning, TEXT("GetNormalImage: No normal data available!"));
        return TArray<FLinearColor>();
    }
    
    return NormalData;
}

void UNormalCameraComponent::AsyncGetNormalImageData(
    TFunction<void(const TArray<FLinearColor>&)> Callback)
{
    AsyncTask(ENamedThreads::GameThread,
        [this, Callback = MoveTemp(Callback)]() {
        if (!CheckComponentAndRenderTarget())
        {
            Callback({});
            return;
        }
        
        CaptureComponent->CaptureScene();
        ProcessNormalTexture(Callback);
    });
}