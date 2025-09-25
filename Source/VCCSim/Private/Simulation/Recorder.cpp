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

#include "Simulation/Recorder.h"
#include "Sensors/CameraSensor.h"
#include "Sensors/NormalCamera.h"
#include "Sensors/SegmentCamera.h"
#include "Utils/AsyncFileWriter.h"
#include "Engine/World.h"
#include "Misc/Paths.h"
#include "TimerManager.h"
#include "HAL/PlatformFilemanager.h"
#include "RenderingThread.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphDefinitions.h"
#include "RenderGraphUtils.h"
#include "RHICommandList.h"
#include "RHI.h"
#include "RHIGPUReadback.h"
#include "RendererInterface.h"

DEFINE_LOG_CATEGORY_STATIC(LogRecorder, Log, All);

ARecorder::ARecorder()
{
    PrimaryActorTick.bCanEverTick = false;

    // Create timestamp-based directory like RuntimeLogs
    FDateTime Now = FDateTime::Now();
    FString Timestamp = FString::Printf(TEXT("%04d%02d%02d_%02d%02d%02d"),
        Now.GetYear(), Now.GetMonth(), Now.GetDay(),
        Now.GetHour(), Now.GetMinute(), Now.GetSecond());

    RecordingPath = FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("RuntimeLogs"), Timestamp);
}

ARecorder::~ARecorder()
{
    StopRecording();
    ShutdownAsyncWriter();
}

void ARecorder::InitializeAsyncWriter()
{
    if (FileWriter.IsValid())
    {
        return;
    }

    FileWriter = MakeUnique<FAsyncFileWriter>(RecordingPath);
    UE_LOG(LogRecorder, Log, TEXT("Async file writer initialized"));
}

void ARecorder::ShutdownAsyncWriter()
{
    if (FileWriter.IsValid())
    {
        FileWriter->Flush();
        FileWriter.Reset();
        UE_LOG(LogRecorder, Log, TEXT("Async file writer shutdown"));
    }
}

void ARecorder::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    StopRecording();
    Super::EndPlay(EndPlayReason);
}

void ARecorder::StartRecording()
{
    if (bIsRecording)
    {
        UE_LOG(LogRecorder, Warning, TEXT("Recording already active"));
        return;
    }

    if (!GetWorld())
    {
        UE_LOG(LogRecorder, Error, TEXT("Cannot start recording - no valid world"));
        return;
    }

    InitializeAsyncWriter();

    float ClampedInterval = FMath::Max(RecordingInterval, 0.001f);
    GetWorld()->GetTimerManager().SetTimer(
        RecordingTimerHandle,
        [this]()
        {
            TickRecording();
        },
        ClampedInterval,
        true
    );

    bIsRecording = true;
    RecordState = true;
}

void ARecorder::StopRecording()
{
    if (!bIsRecording)
    {
        return;
    }

    // Clear the timer
    if (GetWorld() && RecordingTimerHandle.IsValid())
    {
        GetWorld()->GetTimerManager().ClearTimer(RecordingTimerHandle);
        RecordingTimerHandle.Invalidate();
    }

    // Shutdown file writer
    ShutdownAsyncWriter();

    bIsRecording = false;
    RecordState = false;

    UE_LOG(LogRecorder, Log, TEXT("Recording stopped"));
}

void ARecorder::ToggleRecording()
{
    if (bIsRecording)
    {
        StopRecording();
    }
    else
    {
        StartRecording();
    }
}

void ARecorder::TickRecording()
{
    if (!bIsRecording)
    {
        UE_LOG(LogRecorder, Warning, TEXT("TickRecording: Recording not active"));
        return;
    }

    SensorRegistry.CleanupInvalidEntries();
    CollectSensorDataConcurrently();
}

void ARecorder::CreateActorDirectories(const FString& ActorName, TSet<ESensorType>&& SensorTypes)
{
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();

    // Create actor directory
    FString ActorDir = FPaths::Combine(RecordingPath, ActorName);
    if (PlatformFile.CreateDirectoryTree(*ActorDir))
    {
        UE_LOG(LogRecorder, Log, TEXT("Created actor directory: %s"), *ActorDir);
    }

    // Create sensor directories
    for (const ESensorType SensorType : SensorTypes)
    {
        FString SensorDir;
        switch (SensorType)
        {
            case ESensorType::RGBDCamera:
                SensorDir = FPaths::Combine(ActorDir, TEXT("RGB"));
                PlatformFile.CreateDirectoryTree(*SensorDir);
                SensorDir = FPaths::Combine(ActorDir, TEXT("Depth"));
                PlatformFile.CreateDirectoryTree(*SensorDir);
                break;
            case ESensorType::NormalCamera:
                SensorDir = FPaths::Combine(ActorDir, TEXT("Normal"));
                break;
            case ESensorType::SegmentationCamera:
                SensorDir = FPaths::Combine(ActorDir, TEXT("Segmentation"));
                break;
            case ESensorType::Lidar:
                SensorDir = FPaths::Combine(ActorDir, TEXT("Lidar"));
                break;
            default:
                continue; // Do nothing for other types
        }
        PlatformFile.CreateDirectoryTree(*SensorDir);
    }
}

void ARecorder::CollectSensorDataConcurrently()
{
    TArray<FSensorRegistryEntry> AllSensorEntries = SensorRegistry.GetAllSensors();
    if (AllSensorEntries.Num() == 0)
    {
        UE_LOG(LogRecorder, Warning, TEXT("No sensors registered for recording"));
        return;
    }

    TArray<ISensorDataProvider*> Sensors;
    for (const FSensorRegistryEntry& Entry : AllSensorEntries)
    {
        if (Entry.IsValid())
        {
            if (ISensorDataProvider* Provider = Entry.GetProvider())
            {
                Sensors.Add(Provider);
            }
        }
    }

    AsyncTask(ENamedThreads::GameThread, [this, Sensors]()
    {
        for (ISensorDataProvider* Sensor : Sensors)
        {
            if (auto* CameraBase = Cast<UCameraBaseComponent>(Sensor))
            {
                switch (Sensor->GetSensorType())
                {
                    case ESensorType::RGBDCamera:
                        if (auto* RGBDCamera = Cast<URGBDCameraComponent>(CameraBase))
                        {
                            RGBDCamera->CaptureRGBDScene();
                        }
                        break;
                    case ESensorType::NormalCamera:
                        if (auto* NormalCamera = Cast<UNormalCameraComponent>(CameraBase))
                        {
                            NormalCamera->CaptureNormalScene();
                        }
                        break;
                    case ESensorType::SegmentationCamera:
                        if (auto* SegCamera = Cast<USegmentationCameraComponent>(CameraBase))
                        {
                            SegCamera->CaptureSegmentationScene();
                        }
                        break;
                    default:
                        UE_LOG(LogRecorder, Warning, TEXT("Unsupported sensor type for MRT: %d"),
                            static_cast<int32>(Sensor->GetSensorType()));
                        break;
                }
            }
        }

        // After scene captures, execute MRT render command
        ENQUEUE_RENDER_COMMAND(MRTSensorCapture)(
            [this, Sensors](FRHICommandListImmediate& RHICmdList)
            {
                FRDGBuilder GraphBuilder(RHICmdList);
                TArray<FRDGTextureRef> MRTTextures;

                for (ISensorDataProvider* Sensor : Sensors)
                {
                    FIntPoint Resolution = Sensors[0]->GetResolution();
                    if (Sensor->GetSensorType() == ESensorType::RGBDCamera)
                    {
                        MRTTextures.Add(GraphBuilder.CreateTexture(
                        FRDGTextureDesc::Create2D(Resolution, PF_A32B32G32R32F, FClearValueBinding::Black,
                        TexCreate_RenderTargetable | TexCreate_UAV), TEXT("MRT_Normal")));
                    }
                    else if (Sensor->GetSensorType() == ESensorType::SegmentationCamera)
                    {
                        MRTTextures.Add(GraphBuilder.CreateTexture(
                        FRDGTextureDesc::Create2D(Resolution, PF_B8G8R8A8, FClearValueBinding::Black,
                        TexCreate_RenderTargetable | TexCreate_UAV), TEXT("MRT_Segment")));
                    }
                    else if (Sensor->GetSensorType() == ESensorType::NormalCamera)
                    {
                        MRTTextures.Add(GraphBuilder.CreateTexture(
                        FRDGTextureDesc::Create2D(Resolution, PF_B8G8R8A8, FClearValueBinding::Black,
                        TexCreate_RenderTargetable | TexCreate_UAV), TEXT("MRT_Normal")));
                    }
                }

                // Copy from sensor render targets to MRT textures
                AddMRTRenderPass(GraphBuilder, Sensors, MRTTextures);

                GraphBuilder.Execute();
            }
        );
    });
}

void ARecorder::AddMRTRenderPass(FRDGBuilder& GraphBuilder,
        const TArray<ISensorDataProvider*>& Sensors, TArray<FRDGTextureRef> MRTTextures)
{
    // Copy from sensor render targets to MRT textures
    for (int i = 0; i < Sensors.Num(); ++i)
    {
        ISensorDataProvider* Sensor = Sensors[i];
        UTextureRenderTarget2D* SensorRT = Sensor->GetRenderTarget();
        FRDGTextureRef TargetTexture = MRTTextures.IsValidIndex(i) ? MRTTextures[i] : nullptr;
        if (!TargetTexture) continue;
        // Register the sensor's render target as external texture
        FTextureRenderTargetResource* RTResource = SensorRT->GetRenderTargetResource();
        if (RTResource && RTResource->GetRenderTargetTexture())
        {
            FRDGTextureRef SensorTexture = GraphBuilder.RegisterExternalTexture(
                CreateRenderTarget(RTResource->GetRenderTargetTexture(), TEXT("SensorRT")));

            // Add copy pass from sensor RT to MRT texture
            AddCopyTexturePass(GraphBuilder, SensorTexture, TargetTexture, FRHICopyTextureInfo());
        }
    }

    // Extract data from all textures after copying
    double Timestamp = FPlatformTime::Seconds();

    for (int i = 0; i < Sensors.Num(); ++i)
    {
        FIntPoint Resolution = Sensors[i]->GetResolution();
        ISensorDataProvider* Sensor = Sensors[i];
        FRDGTextureRef TargetTexture = MRTTextures[i];

        FRHIGPUTextureReadback* Readback = new FRHIGPUTextureReadback(TEXT("MRTSensorReadback"));

        AddEnqueueCopyPass(GraphBuilder, Readback, TargetTexture, FResolveRect());

        // Store context for later processing after graph execution
        TSharedPtr<FRHIGPUTextureReadback> SharedReadback(Readback);

        // Process readback after render graph executes
        TWeakObjectPtr WeakThis(this);
        Async(EAsyncExecution::TaskGraph, [WeakThis, Sensor, SharedReadback, Resolution, Timestamp]()
        {
            // Wait for readback completion with timeout
            int32 TimeoutCounter = 0;
            const int32 MaxTimeout = 5000; // 5 seconds
            while (!SharedReadback->IsReady() && TimeoutCounter < MaxTimeout)
            {
                FPlatformProcess::Sleep(0.001f); // 1ms
                TimeoutCounter++;
            }

            if (TimeoutCounter >= MaxTimeout)
            {
                UE_LOG(LogRecorder, Error, TEXT("Readback timeout for sensor type %d"),
                       static_cast<int32>(Sensor->GetSensorType()));
                return;
            }

            // Get pixel data with validation
            int32 RowPitchInPixels = 0;
            const void* PixelData = SharedReadback->Lock(RowPitchInPixels);

            if (!PixelData)
            {
                UE_LOG(LogRecorder, Error, TEXT("Failed to lock readback data for sensor type %d"),
                       static_cast<int32>(Sensor->GetSensorType()));
                return;
            }

            int32 Width = Resolution.X;
            int32 Height = Resolution.Y;
            int32 NumPixels = Width * Height;

            // Validate dimensions
            if (Width <= 0 || Height <= 0 || NumPixels <= 0)
            {
                UE_LOG(LogRecorder, Error, TEXT("Invalid resolution %dx%d for sensor type %d"),
                       Width, Height, static_cast<int32>(Sensor->GetSensorType()));
                SharedReadback->Unlock();
                return;
            }

            FSensorDataPacket Packet;
            Packet.Type = Sensor->GetSensorType();
            if (auto* SensorBase = Cast<USensorBaseComponent>(Sensor))
            {
                Packet.SensorIndex = SensorBase->GetSensorIndex();
            }
            else
            {
                Packet.SensorIndex = 0;
            }
            Packet.OwnerActor = Sensor->GetOwnerActor();
            Packet.bValid = true;

            switch (Sensor->GetSensorType())
            {
                case ESensorType::NormalCamera:
                {
                    auto NormalData = MakeShared<FNormalCameraData>();
                    NormalData->Timestamp = Timestamp;
                    NormalData->Width = Width;
                    NormalData->Height = Height;
                    NormalData->SensorIndex = Packet.SensorIndex;
                    NormalData->Data.SetNumUninitialized(Width * Height);
                    const FColor* SourceColors = static_cast<const FColor*>(PixelData);
                    for (int32 i = 0; i < NumPixels; i++)
                    {
                        NormalData->Data[i] = SourceColors[i];
                    }
                    Packet.Data = NormalData;
                    break;
                }
                case ESensorType::SegmentationCamera:
                {
                    auto SegData = MakeShared<FSegmentationCameraData>();
                    SegData->Timestamp = Timestamp;
                    SegData->Width = Width;
                    SegData->Height = Height;
                    SegData->SensorIndex = Packet.SensorIndex;
                    SegData->Data.SetNumUninitialized(NumPixels);
                    const FColor* SourceColors = static_cast<const FColor*>(PixelData);
                    for (int32 i = 0; i < NumPixels; i++)
                    {
                        SegData->Data[i] = SourceColors[i];
                    }
                    Packet.Data = SegData;
                    break;
                }
                case ESensorType::RGBDCamera:
                {
                    auto RGBDData = MakeShared<FRGBDCameraData>();
                    RGBDData->Timestamp = Timestamp;
                    RGBDData->Width = Width;
                    RGBDData->Height = Height;
                    RGBDData->SensorIndex = Packet.SensorIndex;
                    URGBDCameraComponent* RGBDCamera = Cast<URGBDCameraComponent>(Sensor);
                    if (RGBDCamera->bSaveRGB)
                    {
                        RGBDData->RGBData.SetNumUninitialized(NumPixels);
                    }
                    if (RGBDCamera->bSaveDepth)
                    {
                        RGBDData->DepthData.SetNumUninitialized(NumPixels);
                    }
                    const FLinearColor* SourceColors = static_cast<const FLinearColor*>(PixelData);
                    for (int32 i = 0; i < NumPixels; i++)
                    {
                        if (RGBDCamera->bSaveRGB)
                        {
                            FColor EncodedColor = SourceColors[i].ToFColor(true);
                            RGBDData->RGBData[i] = EncodedColor;
                        }
                        if (RGBDCamera->bSaveDepth)
                        {
                            RGBDData->DepthData[i] = SourceColors[i].A;
                        }
                    }
                    Packet.Data = RGBDData;
                    break;
                }
                default:
                    Packet.bValid = false;
                    break;
            }

            if (Packet.bValid && WeakThis.Get()->FileWriter.IsValid())
            {
                WeakThis.Get()->FileWriter->WriteDataAsync(Packet);
            }

            SharedReadback->Unlock();
        });
    }
}