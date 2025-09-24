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

    // Create complete recording directory structure immediately
    CreateRecordingDirectoryStructure();
}

void ARecorder::CreateRecordingDirectoryStructure()
{
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();

    // Create base recording directory
    if (PlatformFile.CreateDirectoryTree(*RecordingPath))
    {
        UE_LOG(LogRecorder, Log, TEXT("Created base recording directory: %s"), *RecordingPath);
    }
    else
    {
        UE_LOG(LogRecorder, Error, TEXT("Failed to create base recording directory: %s"), *RecordingPath);
        return;
    }

    // Pre-create standard sensor directories for common sensor types
    TArray<FString> SensorDirectories = {
        TEXT("RGB"),
        TEXT("Depth"),
        TEXT("Normal"),
        TEXT("Segmentation"),
        TEXT("Lidar")
    };

    // Create directories for potential drone actors (we don't know actor names yet)
    // These will be created dynamically when sensors are registered
    UE_LOG(LogRecorder, Log, TEXT("Recording directory structure ready at: %s"), *RecordingPath);
}

ARecorder::~ARecorder()
{
    StopRecording();
    ShutdownAsyncWriter();
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

    // Initialize file writer
    InitializeAsyncWriter();

    // Start the recording timer
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

    UE_LOG(LogRecorder, Log, TEXT("Recording started at path: %s with interval: %.4f seconds"),
           *RecordingPath, ClampedInterval);
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

void ARecorder::CreateActorDirectories(const FString& ActorName, TSet<ESensorType>&& SensorTypes)
{
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();

    // Create actor directory
    FString ActorDir = FPaths::Combine(RecordingPath, ActorName);
    if (PlatformFile.CreateDirectoryTree(*ActorDir))
    {
        UE_LOG(LogRecorder, Log, TEXT("Created actor directory: %s"), *ActorDir);
    }

    // Create only the sensor directories that will be recorded
    if (SensorTypes.Find(ESensorType::RGBCamera))
    {
        FString RGBDir = FPaths::Combine(ActorDir, TEXT("RGB"));
        PlatformFile.CreateDirectoryTree(*RGBDir);
    }
    if (SensorTypes.Find(ESensorType::DepthCamera))
    {
        FString DepthDir = FPaths::Combine(ActorDir, TEXT("Depth"));
        PlatformFile.CreateDirectoryTree(*DepthDir);
    }
    if (SensorTypes.Find(ESensorType::NormalCamera))
    {
        FString NormalDir = FPaths::Combine(ActorDir, TEXT("Normal"));
        PlatformFile.CreateDirectoryTree(*NormalDir);
    }
    if (SensorTypes.Find(ESensorType::SegmentationCamera))
    {
        FString SegDir = FPaths::Combine(ActorDir, TEXT("Segmentation"));
        PlatformFile.CreateDirectoryTree(*SegDir);
    }
    if (SensorTypes.Find(ESensorType::Lidar))
    {
        FString LidarDir = FPaths::Combine(ActorDir, TEXT("Lidar"));
        PlatformFile.CreateDirectoryTree(*LidarDir);
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

void ARecorder::CollectSensorDataConcurrently()
{
    TArray<FSensorRegistryEntry> AllSensors = SensorRegistry.GetAllSensors();
    if (AllSensors.Num() == 0)
    {
        UE_LOG(LogRecorder, Warning, TEXT("No sensors registered for recording"));
        return;
    }

    // Group sensors by actor for spatial coherence and RDG batching
    TMap<AActor*, TArray<ISensorDataProvider*>> SensorsByActor;
    for (const FSensorRegistryEntry& Entry : AllSensors)
    {
        if (Entry.IsValid())
        {
            if (ISensorDataProvider* Provider = Entry.GetProvider())
            {
                SensorsByActor.FindOrAdd(Provider->GetOwnerActor()).Add(Provider);
            }
        }
    }

    // Process each actor's sensors
    for (auto& [Actor, Sensors] : SensorsByActor)
    {
        if (Sensors.Num() == 0)
        {
            continue;
        }

        // Use MRT rendering for multiple sensors
        if (Sensors.Num() >= 2)
        {
            CollectSensorDataMRT(Actor, Sensors);
        }
        else
        {
            CollectSensorDataIndividual(Sensors);
        }
    }
}

void ARecorder::CollectSensorDataMRT(AActor* Actor, const TArray<ISensorDataProvider*>& Sensors)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(ARecorder::CollectSensorDataMRT);

    // First, trigger scene capture for each sensor on game thread
    AsyncTask(ENamedThreads::GameThread, [this, Actor, Sensors]()
    {
        // Capture scene for each sensor to populate their render targets
        for (ISensorDataProvider* Sensor : Sensors)
        {
            if (auto* CameraBase = Cast<UCameraBaseComponent>(Sensor))
            {
                // Trigger specific sensor captures based on type
                switch (Sensor->GetSensorType())
                {
                    case ESensorType::RGBCamera:
                        if (auto* RGBCamera = Cast<URGBCameraComponent>(CameraBase))
                        {
                            RGBCamera->CaptureRGBScene();
                        }
                        break;
                    case ESensorType::DepthCamera:
                        if (auto* DepthCamera = Cast<UDepthCameraComponent>(CameraBase))
                        {
                            DepthCamera->CaptureDepthScene();
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
                }
            }
        }

        // After scene captures, execute MRT render command
        ENQUEUE_RENDER_COMMAND(MRTSensorCapture)(
            [this, Actor, Sensors](FRHICommandListImmediate& RHICmdList)
            {
                FRDGBuilder GraphBuilder(RHICmdList);

                // Get common resolution from first sensor
                FIntPoint Resolution(512, 512);
                if (Sensors.Num() > 0 && Sensors[0])
                {
                    if (auto* CameraBase = Cast<UCameraBaseComponent>(Sensors[0]))
                    {
                        Resolution = FIntPoint(CameraBase->Width, CameraBase->Height);
                    }
                }

                // Create MRT textures for different sensor types
                FRDGTextureRef RGBTexture = GraphBuilder.CreateTexture(
                    FRDGTextureDesc::Create2D(Resolution, PF_B8G8R8A8, FClearValueBinding::Black,
                    TexCreate_RenderTargetable | TexCreate_UAV), TEXT("MRT_RGB"));

                FRDGTextureRef DepthTexture = GraphBuilder.CreateTexture(
                    FRDGTextureDesc::Create2D(Resolution, PF_R32_FLOAT, FClearValueBinding::Black,
                    TexCreate_RenderTargetable | TexCreate_UAV), TEXT("MRT_Depth"));

                FRDGTextureRef NormalTexture = GraphBuilder.CreateTexture(
                    FRDGTextureDesc::Create2D(Resolution, PF_A32B32G32R32F, FClearValueBinding::Black,
                    TexCreate_RenderTargetable | TexCreate_UAV), TEXT("MRT_Normal"));

                FRDGTextureRef SegmentTexture = GraphBuilder.CreateTexture(
                    FRDGTextureDesc::Create2D(Resolution, PF_B8G8R8A8, FClearValueBinding::Black,
                    TexCreate_RenderTargetable | TexCreate_UAV), TEXT("MRT_Segment"));

                // Copy from sensor render targets to MRT textures
                AddMRTRenderPass(GraphBuilder, Actor, Sensors, RGBTexture, DepthTexture, NormalTexture, SegmentTexture);

                GraphBuilder.Execute();
            }
        );
    });
}

void ARecorder::AddMRTRenderPass(FRDGBuilder& GraphBuilder, AActor* Actor,
    const TArray<ISensorDataProvider*>& Sensors,
    FRDGTextureRef RGBTexture, FRDGTextureRef DepthTexture,
    FRDGTextureRef NormalTexture, FRDGTextureRef SegmentTexture)
{
    // Copy from sensor render targets to MRT textures
    for (ISensorDataProvider* Sensor : Sensors)
    {
        UTextureRenderTarget2D* SensorRT = nullptr;
        FRDGTextureRef TargetTexture = nullptr;

        // Get specific sensor render target based on type
        switch (Sensor->GetSensorType())
        {
            case ESensorType::RGBCamera:
                if (auto* RGBCamera = Cast<URGBCameraComponent>(Sensor))
                {
                    SensorRT = RGBCamera->RGBRenderTarget;
                    TargetTexture = RGBTexture;
                }
                break;
            case ESensorType::DepthCamera:
                if (auto* DepthCamera = Cast<UDepthCameraComponent>(Sensor))
                {
                    SensorRT = DepthCamera->DepthRenderTarget;
                    TargetTexture = DepthTexture;
                }
                break;
            case ESensorType::NormalCamera:
                if (auto* NormalCamera = Cast<UNormalCameraComponent>(Sensor))
                {
                    SensorRT = NormalCamera->NormalRenderTarget;
                    TargetTexture = NormalTexture;
                }
                break;
            case ESensorType::SegmentationCamera:
                if (auto* SegCamera = Cast<USegmentationCameraComponent>(Sensor))
                {
                    SensorRT = SegCamera->SegmentationRenderTarget;
                    TargetTexture = SegmentTexture;
                }
                break;
            default:
                continue;
        }

        if (SensorRT && TargetTexture)
        {
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
    }

    UE_LOG(LogRecorder, Log, TEXT("MRT textures populated from sensor render targets for %d sensors"), Sensors.Num());

    // Extract data from all textures after copying
    ExtractMRTData(GraphBuilder, Actor, Sensors, RGBTexture, DepthTexture, NormalTexture, SegmentTexture);
}

void ARecorder::RenderSceneToMRT(FRHICommandList& RHICmdList, AActor* Actor,
    const TArray<ISensorDataProvider*>& Sensors,
    FRDGTextureRef RGBTexture, FRDGTextureRef DepthTexture,
    FRDGTextureRef NormalTexture, FRDGTextureRef SegmentTexture)
{
    if (Sensors.Num() == 0) return;

    // Get view info from first sensor (camera)
    FSensorViewInfo ViewInfo;
    Sensors[0]->ContributeToRDGPass(ViewInfo);

    UE_LOG(LogRecorder, Log, TEXT("MRT rendering for %d sensors at resolution %dx%d"),
           Sensors.Num(), ViewInfo.Resolution.X, ViewInfo.Resolution.Y);

    // Clear each texture with distinctive test patterns for MRT verification
    // This simulates actual sensor data with different colors per sensor type

    // RGB Camera (Red pattern) - Slot 0
    FLinearColor RGBColor(0.8f, 0.2f, 0.2f, 1.0f);

    // Depth Camera (Gray pattern) - Slot 1
    FLinearColor DepthColor(0.5f, 0.5f, 0.5f, 1.0f);

    // Normal Camera (Blue pattern) - Slot 2
    FLinearColor NormalColor(0.5f, 0.5f, 1.0f, 1.0f);

    // Segmentation Camera (Yellow pattern) - Slot 3
    FLinearColor SegmentColor(1.0f, 1.0f, 0.0f, 1.0f);

    // In a full implementation, these would be rendered from scene geometry
    // For now, we use test patterns to verify MRT functionality

    UE_LOG(LogRecorder, Log, TEXT("MRT rendering completed with test patterns"));
}

void ARecorder::ExtractMRTData(FRDGBuilder& GraphBuilder, AActor* Actor,
    const TArray<ISensorDataProvider*>& Sensors,
    FRDGTextureRef RGBTexture, FRDGTextureRef DepthTexture,
    FRDGTextureRef NormalTexture, FRDGTextureRef SegmentTexture)
{
    double Timestamp = FPlatformTime::Seconds();

    // Get resolution from texture descriptor (safe before graph execution)
    FIntPoint Resolution = RGBTexture->Desc.Extent;

    for (ISensorDataProvider* Sensor : Sensors)
    {
        int32 MRTSlot = Sensor->GetMRTSlot();
        FRDGTextureRef SourceTexture = nullptr;

        switch (MRTSlot)
        {
            case 0: SourceTexture = RGBTexture; break;
            case 1: SourceTexture = DepthTexture; break;
            case 2: SourceTexture = NormalTexture; break;
            case 3: SourceTexture = SegmentTexture; break;
            default: continue;
        }

        if (!SourceTexture) continue;

        UE_LOG(LogRecorder, Log, TEXT("Setting up texture readback for sensor type %d, MRT slot %d, resolution %dx%d"),
               static_cast<int32>(Sensor->GetSensorType()), MRTSlot, Resolution.X, Resolution.Y);

        // Create readback buffer for UE 5.6
        FRHIGPUTextureReadback* Readback = new FRHIGPUTextureReadback(TEXT("MRTSensorReadback"));

        // Use standard UE 5.6 enqueue copy pass
        AddEnqueueCopyPass(GraphBuilder, Readback, SourceTexture, FResolveRect());

        // Store context for later processing after graph execution
        TSharedPtr<FRHIGPUTextureReadback> SharedReadback(Readback);

        // Process readback after render graph executes
        ENQUEUE_RENDER_COMMAND(ProcessMRTReadback)(
            [this, Sensor, SharedReadback, Resolution, Timestamp](FRHICommandListImmediate& RHICmdList)
            {
                Async(EAsyncExecution::TaskGraph, [this, Sensor, SharedReadback, Resolution, Timestamp]()
                {
                    UE_LOG(LogRecorder, Log, TEXT("Starting async readback wait for sensor type %d"),
                           static_cast<int32>(Sensor->GetSensorType()));

                    // Wait for readback completion
                    while (!SharedReadback->IsReady())
                    {
                        FPlatformProcess::Sleep(0.001f); // 1ms
                    }

                    UE_LOG(LogRecorder, Log, TEXT("Readback ready for sensor type %d"),
                           static_cast<int32>(Sensor->GetSensorType()));

                    // Get pixel data
                    int32 RowPitchInPixels = 0;
                    const void* PixelData = SharedReadback->Lock(RowPitchInPixels);

                    if (PixelData)
                    {
                        int32 Width = Resolution.X;
                        int32 Height = Resolution.Y;
                        int32 NumPixels = Width * Height;

                        // Process data based on sensor type
                        if (Sensor->GetSensorType() == ESensorType::DepthCamera)
                        {
                            // Depth camera uses float format - read as FFloat16Color
                            TArray<FFloat16Color> DepthColorData;
                            DepthColorData.SetNumUninitialized(NumPixels);

                            const FFloat16Color* SourceFloats = static_cast<const FFloat16Color*>(PixelData);
                            for (int32 i = 0; i < NumPixels; i++)
                            {
                                DepthColorData[i] = SourceFloats[i];
                            }

                            // Process depth data separately
                            ProcessDepthSensorData(Sensor, Resolution, Timestamp, MoveTemp(DepthColorData));
                        }
                        else
                        {
                            // Other sensors use regular color format
                            TArray<FColor> ColorData;
                            ColorData.SetNumUninitialized(NumPixels);

                            // Copy pixel data (assuming BGRA format)
                            const uint8* SourceBytes = static_cast<const uint8*>(PixelData);
                            for (int32 i = 0; i < NumPixels; i++)
                            {
                                ColorData[i] = FColor(
                                    SourceBytes[i * 4 + 2], // R
                                    SourceBytes[i * 4 + 1], // G
                                    SourceBytes[i * 4 + 0], // B
                                    SourceBytes[i * 4 + 3]  // A
                                );
                            }

                            // Process the sensor data
                            ProcessSensorPixelData(Sensor, Resolution, Timestamp, MoveTemp(ColorData));
                        }
                    }
                    else
                    {
                        UE_LOG(LogRecorder, Warning, TEXT("Failed to lock readback data for sensor type %d"),
                               static_cast<int32>(Sensor->GetSensorType()));
                    }

                    SharedReadback->Unlock();
                });
            }
        );
    }
}

void ARecorder::ProcessSensorPixelData(ISensorDataProvider* Sensor, FIntPoint Resolution,
    double Timestamp, TArray<FColor>&& PixelData)
{
    UE_LOG(LogRecorder, Log, TEXT("ProcessSensorPixelData called for sensor type %d with %d pixels"),
           Sensor ? static_cast<int32>(Sensor->GetSensorType()) : -1, PixelData.Num());

    if (!Sensor || PixelData.Num() == 0)
    {
        UE_LOG(LogRecorder, Warning, TEXT("ProcessSensorPixelData: Invalid sensor or empty pixel data"));
        return;
    }

    // Create data packet based on sensor type
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
    Packet.Timestamp = Timestamp;
    Packet.bFromRDGBatch = true;
    Packet.bValid = true;

    int32 Width = Resolution.X;
    int32 Height = Resolution.Y;

    switch (Sensor->GetSensorType())
    {
        case ESensorType::RGBCamera:
        {
            auto RGBData = MakeShared<FRGBCameraData>();
            RGBData->Timestamp = Timestamp;
            RGBData->Width = Width;
            RGBData->Height = Height;
            RGBData->Data = MoveTemp(PixelData);

            Packet.Data = RGBData;
            break;
        }
        case ESensorType::DepthCamera:
        {
            auto DepthData = MakeShared<FDepthCameraData>();
            DepthData->Timestamp = Timestamp;
            DepthData->Width = Width;
            DepthData->Height = Height;
            DepthData->Data.SetNumUninitialized(Width * Height);

            // Convert FColor to depth data (R channel contains depth)
            for (int32 i = 0; i < PixelData.Num(); i++)
            {
                float DepthValue = PixelData[i].R / 255.0f; // Normalize to 0-1 range
                DepthData->Data[i] = DepthValue; // Data is TArray<float>, not FFloat16Color
            }

            Packet.Data = DepthData;
            break;
        }
        case ESensorType::NormalCamera:
        {
            auto NormalData = MakeShared<FNormalCameraData>();
            NormalData->Timestamp = Timestamp;
            NormalData->Width = Width;
            NormalData->Height = Height;
            NormalData->Data.SetNumUninitialized(Width * Height);

            // Convert FColor to FLinearColor for normal data
            for (int32 i = 0; i < PixelData.Num(); i++)
            {
                const FColor& Pixel = PixelData[i];
                // Convert from [0,255] back to [-1,1] normal space
                float R = (Pixel.R / 255.0f) * 2.0f - 1.0f;
                float G = (Pixel.G / 255.0f) * 2.0f - 1.0f;
                float B = (Pixel.B / 255.0f) * 2.0f - 1.0f;
                NormalData->Data[i] = FLinearColor(R, G, B, 1.0f);
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
            SegData->Data = MoveTemp(PixelData);

            Packet.Data = SegData;
            break;
        }
        default:
            Packet.bValid = false;
            break;
    }

    // Save data if valid
    if (Packet.bValid && FileWriter.IsValid())
    {
        FileWriter->WriteDataAsync(Packet);
        UE_LOG(LogRecorder, Log, TEXT("MRT data saved for sensor type %d, resolution %dx%d"),
               static_cast<int32>(Sensor->GetSensorType()), Width, Height);
    }
}

void ARecorder::ProcessDepthSensorData(ISensorDataProvider* Sensor, FIntPoint Resolution,
    double Timestamp, TArray<FFloat16Color>&& DepthPixelData)
{
    UE_LOG(LogRecorder, Log, TEXT("ProcessDepthSensorData called for sensor type %d with %d depth pixels"),
           Sensor ? static_cast<int32>(Sensor->GetSensorType()) : -1, DepthPixelData.Num());

    if (!Sensor || DepthPixelData.Num() == 0)
    {
        UE_LOG(LogRecorder, Warning, TEXT("ProcessDepthSensorData: Invalid sensor or empty depth data"));
        return;
    }

    // Create depth data packet
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
    Packet.Timestamp = Timestamp;
    Packet.bFromRDGBatch = true;
    Packet.bValid = true;

    int32 Width = Resolution.X;
    int32 Height = Resolution.Y;

    // Create depth camera data with correct float processing
    auto DepthData = MakeShared<FDepthCameraData>();
    DepthData->Timestamp = Timestamp;
    DepthData->Width = Width;
    DepthData->Height = Height;
    DepthData->Data.SetNumUninitialized(Width * Height);

    // Convert FFloat16Color to float array (like the original depth camera does)
    for (int32 i = 0; i < DepthPixelData.Num(); i++)
    {
        DepthData->Data[i] = DepthPixelData[i].R.GetFloat(); // Extract depth from R channel as float
    }

    Packet.Data = DepthData;

    // Save data if valid
    if (Packet.bValid && FileWriter.IsValid())
    {
        FileWriter->WriteDataAsync(Packet);
        UE_LOG(LogRecorder, Log, TEXT("MRT depth data saved for sensor, resolution %dx%d"),
               Width, Height);
    }
}

void ARecorder::CollectSensorDataIndividual(const TArray<ISensorDataProvider*>& Sensors)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(ARecorder::CollectSensorDataIndividual);

    TArray<TFuture<FSensorDataPacket>> ConcurrentTasks;
    ConcurrentTasks.Reserve(Sensors.Num());

    for (ISensorDataProvider* Provider : Sensors)
    {
        TFuture<FSensorDataPacket> Future = Provider->CaptureDataAsync();
        ConcurrentTasks.Add(MoveTemp(Future));
    }

    // Stream processing: handle each sensor result as soon as it's ready
    for (auto& Future : ConcurrentTasks)
    {
        Future.Then([this](TFuture<FSensorDataPacket> CompletedFuture)
        {
            FSensorDataPacket Result = CompletedFuture.Get();
            if (Result.bValid && Result.Data.IsValid())
            {
                // Process immediately on background thread
                Async(EAsyncExecution::TaskGraph, [this, Result = MoveTemp(Result)]()
                {
                    if (FileWriter.IsValid())
                    {
                        FileWriter->WriteDataAsync(Result);
                    }
                });
            }
        });
    }
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