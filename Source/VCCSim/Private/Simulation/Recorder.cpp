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
#include "Sensors/DepthCamera.h"

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
    SensorsToReadThisFrame.Empty();
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
    LastSensorCaptureTimes.Empty();
    SensorsToReadThisFrame.Empty();

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
    CollectSensorData();
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
            case ESensorType::RGBCamera:
                SensorDir = FPaths::Combine(ActorDir, TEXT("RGB"));
                PlatformFile.CreateDirectoryTree(*SensorDir);
                break;
            case ESensorType::DepthCamera:
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

void ARecorder::CollectSensorData()
{
    TArray<FSensorRegistryEntry> AllSensorEntries = SensorRegistry.GetAllSensors();
    if (AllSensorEntries.Num() == 0)
    {
        UE_LOG(LogRecorder, Warning, TEXT("No sensors registered for recording"));
        return;
    }

    TArray<USensorBaseComponent*> Sensors;
    for (const FSensorRegistryEntry& Entry : AllSensorEntries)
    {
        if (Entry.IsValid())
        {
            if (USensorBaseComponent* SensorBase = Entry.GetSensorComponent())
            {
                Sensors.Add(SensorBase);
            }
        }
    }

    TArray<USensorBaseComponent*> SensorsToRead = SensorsToReadThisFrame.Array();
    SensorsToReadThisFrame.Empty();
    
    AsyncTask(ENamedThreads::AnyThread,
        [this, SensorsToRead]()
        {
            if (SensorsToReadThisFrame.Num() > 0)
            {
                ENQUEUE_RENDER_COMMAND(DirectSensorCapture)(
                    [this, SensorsToRead](FRHICommandListImmediate& RHICmdList)
                    {
                        FRDGBuilder GraphBuilder(RHICmdList);
                        ProcessSensorResults(GraphBuilder, SensorsToRead);
                        GraphBuilder.Execute();
                    }
                );
            }
        });
    

    double CurrentTime = FPlatformTime::Seconds();
    for (USensorBaseComponent* Sensor : Sensors)
    {
        if (auto* CameraBase = Cast<UCameraBaseComponent>(Sensor))
        {
            if (ShouldCaptureSensor(Sensor, CurrentTime))
            {
                switch (Sensor->GetSensorType())
                {
                    case ESensorType::RGBCamera:
                        if (auto* RGBCamera = Cast<URGBCameraComponent>(CameraBase))
                        {
                            RGBCamera->CaptureRGBSceneDeferred();
                        }
                        break;
                    case ESensorType::DepthCamera:
                        if (auto* DepthCamera = Cast<UDepthCameraComponent>(CameraBase))
                        {
                            DepthCamera->CaptureDepthSceneDeferred();
                        }
                        break;
                    case ESensorType::NormalCamera:
                        if (auto* NormalCamera = Cast<UNormalCameraComponent>(CameraBase))
                        {
                            NormalCamera->CaptureNormalSceneDeferred();
                        }
                        break;
                    case ESensorType::SegmentationCamera:
                        if (auto* SegCamera = Cast<USegCameraComponent>(CameraBase))
                        {
                            SegCamera->CaptureSegmentationSceneDeferred();
                        }
                        break;
                    default:
                        UE_LOG(LogRecorder, Warning, TEXT("Unsupported sensor type: %d"),
                            static_cast<int32>(Sensor->GetSensorType()));
                        break;
                }
            }
        }
    }
}

void ARecorder::ProcessCameraResult(FRDGBuilder& GraphBuilder, USensorBaseComponent* Sensor)
{
    auto* CameraBase = Cast<UCameraBaseComponent>(Sensor);
    FTextureRenderTargetResource* RTResource =
        CameraBase->GetRenderTarget()->GetRenderTargetResource();
    if (!RTResource || !RTResource->GetRenderTargetTexture())
    {
        UE_LOG(LogRecorder, Error,
            TEXT("Invalid render target resource for sensor type %d"),
            static_cast<int32>(Sensor->GetSensorType()));
    }

    FRDGTextureRef SensorTexture = GraphBuilder.RegisterExternalTexture(
        CreateRenderTarget(RTResource->GetRenderTargetTexture(), TEXT("SensorRT")));

    FRHIGPUTextureReadback* Readback = new FRHIGPUTextureReadback(TEXT("DirectSensorReadback"));

    AddEnqueueCopyPass(GraphBuilder, Readback, SensorTexture, FResolveRect());

    TSharedPtr<FRHIGPUTextureReadback> SharedReadback(Readback);

    double CaptureTimestamp = CameraBase->GetLastCaptureTimestamp();

    Async(EAsyncExecution::TaskGraph,
          [CameraBase, SharedReadback, CaptureTimestamp, this]()
          {
              int32 TimeoutCounter = 0;
              constexpr int32 MaxTimeout = 1000;
              while (!SharedReadback->IsReady() && TimeoutCounter < MaxTimeout)
              {
                  FPlatformProcess::Sleep(0.001f);
                  TimeoutCounter++;
              }
              if (TimeoutCounter >= MaxTimeout)
              {
                  UE_LOG(LogRecorder, Error, TEXT("Readback timeout for sensor type %d"),
                     static_cast<int32>(CameraBase->GetSensorType()));
                  return;
              }

              int32 RowPitchInPixels = 0;
              const void* PixelData = SharedReadback->Lock(RowPitchInPixels);
              if (!PixelData)
              {
                  UE_LOG(LogRecorder, Error,
                      TEXT("Failed to lock readback data for sensor type %d"),
                      static_cast<int32>(CameraBase->GetSensorType()));
                  return;
              }

              const FIntPoint Resolution = CameraBase->GetResolution();
              const int32 NumPixels = Resolution.X * Resolution.Y;

              FSensorDataPacket Packet;
              Packet.Type = CameraBase->GetSensorType();
              Packet.SensorIndex = CameraBase->GetSensorIndex();
              Packet.OwnerActor = CameraBase->GetOwnerActor();
              Packet.bValid = true;

              switch (Packet.Type)
              {
              case ESensorType::NormalCamera:
                  {
                      auto NormalData = MakeShared<FNormalCameraData>();
                      NormalData->Timestamp = CaptureTimestamp;
                      NormalData->Width = Resolution.X;
                      NormalData->Height = Resolution.Y;
                      NormalData->SensorIndex = Packet.SensorIndex;
                      NormalData->Data.SetNumUninitialized(NumPixels);
                      const FFloat16Color* SourceColors = static_cast<const FFloat16Color*>(PixelData);
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
                      SegData->Timestamp = CaptureTimestamp;
                      SegData->Width = Resolution.X;
                      SegData->Height = Resolution.Y;
                      SegData->SensorIndex = Packet.SensorIndex;
                      SegData->Data.SetNumUninitialized(NumPixels);
                      const FColor* SourceColors = static_cast<const FColor*>(PixelData);
                      for (int32 i = 0; i < NumPixels; i++)
                      {
                          SegData->Data[i] = SourceColors[i];
                          SegData->Data[i].A = 255;
                      }
                      Packet.Data = SegData;
                      break;
                  }
              case ESensorType::RGBCamera:
                  {
                      auto RGBData = MakeShared<FRGBCameraData>();
                      RGBData->Timestamp = CaptureTimestamp;
                      RGBData->Width = Resolution.X;
                      RGBData->Height = Resolution.Y;
                      RGBData->SensorIndex = Packet.SensorIndex;
                      RGBData->RGBData.SetNumUninitialized(NumPixels);
                      
                      const FColor* SourceColors = static_cast<const FColor*>(PixelData);
                      for (int32 i = 0; i < NumPixels; i++)
                      {
                          RGBData->RGBData[i] = SourceColors[i];
                          
                      }
                      Packet.Data = RGBData;
                      break;
                  }
              case ESensorType::DepthCamera:
                  {
                      auto DepthData = MakeShared<FDepthCameraData>();
                      DepthData->Timestamp = CaptureTimestamp;
                      DepthData->Width = Resolution.X;
                      DepthData->Height = Resolution.Y;
                      DepthData->SensorIndex = Packet.SensorIndex;
                      DepthData->DepthData.SetNumUninitialized(NumPixels);
                      
                      const FFloat16Color* SourceColors = static_cast<const FFloat16Color*>(PixelData);
                      for (int32 i = 0; i < NumPixels; i++)
                      {
                          DepthData->DepthData[i] = SourceColors[i].R.GetFloat();
                      }
                      Packet.Data = DepthData;
                      break;
                  }
              default:
                  Packet.bValid = false;
                  break;
              }

              if (Packet.bValid)
              {
                  this->FileWriter->WriteData(MoveTemp(Packet));
              }
              SharedReadback->Unlock();
          });
}

void ARecorder::ProcessSensorResults(
    FRDGBuilder& GraphBuilder, const TArray<USensorBaseComponent*>& Sensors)
{
    for (USensorBaseComponent* Sensor : Sensors)
    {
        ESensorType SensorType = Sensor->GetSensorType();
        // Camera sensor
        if (SensorType == ESensorType::NormalCamera ||
            SensorType == ESensorType::SegmentationCamera ||
            SensorType == ESensorType::RGBCamera ||
            SensorType == ESensorType::DepthCamera)
        {
            ProcessCameraResult(GraphBuilder, Sensor);
        }
        // Non-camera sensor (e.g., LiDAR)
        else
        {
            
        }
    }
}

bool ARecorder::ShouldCaptureSensor(USensorBaseComponent* Sensor, double CurrentTime)
{
    double SensorInterval = Sensor->GetRecordInterval();

    if (!LastSensorCaptureTimes.Contains(Sensor))
    {
        LastSensorCaptureTimes.Add(Sensor, CurrentTime);
        SensorsToReadThisFrame.Add(Sensor);
        return true;
    }

    double LastCaptureTime = LastSensorCaptureTimes[Sensor];
    double TimeSinceLastCapture = CurrentTime - LastCaptureTime;

    if (TimeSinceLastCapture >= SensorInterval)
    {
        LastSensorCaptureTimes[Sensor] = CurrentTime;
        SensorsToReadThisFrame.Add(Sensor);
        return true;
    }

    return false;
}