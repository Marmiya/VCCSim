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
#include "Sensors/DepthCamera.h"
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
#include "SceneRenderTargetParameters.h"
#include "PixelShaderUtils.h"
#include "ScreenRendering.h"
#include "SystemTextures.h"

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

    SetupSensorProperties();
    GroupCamerasByPose();

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
    if (CameraViewGroups.Num() == 0)
    {
        UE_LOG(LogRecorder, Warning, TEXT("No camera view groups available for recording"));
        return;
    }

    TArray<FSensorRegistryEntry> AllSensorEntries = SensorRegistry.GetAllSensors();
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
            if (SensorsToRead.Num() > 0)
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
    
    for (FCameraViewGroup& ViewGroup : CameraViewGroups)
    {
        TArray<USensorBaseComponent*> CamerasNeedingUpdate;

        for (auto& SensorAndState : ViewGroup.Cameras)
        {
            SensorAndState.Value = ShouldCaptureSensor(SensorAndState.Key, CurrentTime);
        }
    }

    RenderViewGroupsRDG();
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

    double CaptureTimestamp = LastSensorCaptureTimes[Sensor];

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

              if (Packet.bValid && this->FileWriter.IsValid())
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
    double SensorInterval = SensorIntervals[Sensor];

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

void ARecorder::SetupSensorProperties()
{
    for (const FSensorRegistryEntry& Entry : SensorRegistry.GetAllSensors())
    {
        if (Entry.IsValid())
        {
            if (USensorBaseComponent* Sensor = Entry.GetSensorComponent())
            {
                SensorIntervals.Add(Sensor, Sensor->GetRecordInterval());
            }
            else
            {
                UE_LOG(LogRecorder, Warning, TEXT("Invalid sensor component in registry entry"));
            }
        }
    }
}

void ARecorder::GroupCamerasByPose()
{
    CameraViewGroups.Empty();

    TArray<FSensorRegistryEntry> AllSensorEntries = SensorRegistry.GetAllSensors();
    TArray<UCameraBaseComponent*> AllCameras;

    for (const FSensorRegistryEntry& Entry : AllSensorEntries)
    {
        if (Entry.IsValid())
        {
            if (UCameraBaseComponent* Camera = Cast<UCameraBaseComponent>(Entry.GetSensorComponent()))
            {
                AllCameras.Add(Camera);
            }
        }
    }

    TArray<bool> Grouped;
    Grouped.SetNumZeroed(AllCameras.Num());

    int32 ViewIndex = 0;
    for (int32 i = 0; i < AllCameras.Num(); ++i)
    {
        if (Grouped[i])
            continue;

        UCameraBaseComponent* BaseCamera = AllCameras[i];
        FCameraViewGroup NewGroup;
        NewGroup.ViewIndex = ViewIndex++;
        NewGroup.Cameras.FindOrAdd(BaseCamera) = true;
        Grouped[i] = true;

        for (int32 j = i + 1; j < AllCameras.Num(); ++j)
        {
            if (Grouped[j])
                continue;

            UCameraBaseComponent* OtherCamera = AllCameras[j];
            if (ArePosesSimilar(BaseCamera, OtherCamera))
            {
                NewGroup.Cameras.FindOrAdd(OtherCamera) = true;
                Grouped[j] = true;
            }
        }

        CameraViewGroups.Add(NewGroup);

        UE_LOG(LogRecorder, Log, TEXT("View Group %d: %d"),
            NewGroup.ViewIndex, NewGroup.Cameras.Num());
    }
}

bool ARecorder::ArePosesSimilar(const UCameraBaseComponent* CamA, const UCameraBaseComponent* CamB) const
{
    if (CamA->GetResolution() != CamB->GetResolution())
        return false;

    if (!FMath::IsNearlyEqual(CamA->FOV, CamB->FOV, 0.1f))
        return false;

    FVector PosA = CamA->GetComponentLocation();
    FVector PosB = CamB->GetComponentLocation();
    float Distance = FVector::Dist(PosA, PosB);
    if (Distance > PositionThreshold)
        return false;

    FRotator RotA = CamA->GetComponentRotation();
    FRotator RotB = CamB->GetComponentRotation();
    float AngularDiff = FMath::Abs(FRotator::NormalizeAxis(RotA.Pitch - RotB.Pitch)) +
                        FMath::Abs(FRotator::NormalizeAxis(RotA.Yaw - RotB.Yaw)) +
                        FMath::Abs(FRotator::NormalizeAxis(RotA.Roll - RotB.Roll));
    if (AngularDiff > RotationThreshold * 3.0f)
        return false;

    return true;
}

class FDepthPS : public FGlobalShader
{
public:
    DECLARE_GLOBAL_SHADER(FDepthPS);
    SHADER_USE_PARAMETER_STRUCT(FDepthPS, FGlobalShader);

    BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
        SHADER_PARAMETER_RDG_TEXTURE(Texture2D, SceneDepthTexture)
        SHADER_PARAMETER_SAMPLER(SamplerState, SceneDepthSampler)
        RENDER_TARGET_BINDING_SLOTS()
    END_SHADER_PARAMETER_STRUCT()

    static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
    {
        return true;
    }
};

IMPLEMENT_GLOBAL_SHADER(FDepthPS, "/VCCSim/DepthCapture.usf", "MainPS", SF_Pixel);


void ARecorder::RenderViewGroupsRDG()
{
    if (!GetWorld() || CameraViewGroups.Num() == 0)
    {
        return;
    }

    TArray<TArray<UCameraBaseComponent*>> UpdatedCamerasPerGroup;

    for (const auto& CameraViewGroup : CameraViewGroups)
    {
        TArray<UCameraBaseComponent*> CamerasToUpdate;
        for (const auto& CamPair : CameraViewGroup.Cameras)
        {
            if (CamPair.Value)
            {
                if (UCameraBaseComponent* CameraComp = Cast<UCameraBaseComponent>(CamPair.Key))
                {
                    CamerasToUpdate.Add(CameraComp);
                }
            }
        }
        UpdatedCamerasPerGroup.Add(MoveTemp(CamerasToUpdate));
    }

    const ERHIFeatureLevel::Type FeatureLevel = GMaxRHIFeatureLevel;

    ENQUEUE_RENDER_COMMAND(MRTMultiViewCapture)(
        [UpdatedCamerasPerGroup, FeatureLevel](FRHICommandListImmediate& RHICmdList)
        {
            FRDGBuilder GraphBuilder(RHICmdList, RDG_EVENT_NAME("VCCSimMRTCapture"));

            FRDGSystemTextures::Create(GraphBuilder);

            const FSceneTextures* SceneTextures = nullptr;
            for (const auto& UpdatedCameras : UpdatedCamerasPerGroup)
            {
                if (UpdatedCameras.Num() == 0)
                {
                    continue;
                }

                for (int32 i = 0; i < UpdatedCameras.Num(); ++i)
                {
                    if (UpdatedCameras[i]->GetSensorType() == ESensorType::DepthCamera)
                    {
                        FString TextureName = FString::Printf(TEXT("DepthCameraRT_%d"), UpdatedCameras[i]->GetSensorIndex());
                        FRDGTextureRef OutputTexture = GraphBuilder.RegisterExternalTexture(
                            CreateRenderTarget(UpdatedCameras[i]->GetRenderTarget()->
                                GetRenderTargetResource()->GetRenderTargetTexture(), *TextureName));

                        auto* PassParameters = GraphBuilder.AllocParameters<FDepthPS::FParameters>();

                        PassParameters->SceneDepthTexture = GSystemTextures.GetBlackDummy(GraphBuilder);
                        PassParameters->SceneDepthSampler = TStaticSamplerState<SF_Point>::GetRHI();

                        PassParameters->RenderTargets[0] =
                            FRenderTargetBinding(OutputTexture, ERenderTargetLoadAction::EClear);

                        TShaderMapRef<FDepthPS> PixelShader(GetGlobalShaderMap(FeatureLevel));

                        const FIntPoint Extent = OutputTexture->Desc.Extent;
                        const FIntRect ViewRect(0, 0, Extent.X, Extent.Y);

                        FPixelShaderUtils::AddFullscreenPass(
                            GraphBuilder,
                            GetGlobalShaderMap(FeatureLevel),
                            RDG_EVENT_NAME("VCCSimDepthCapture"),
                            PixelShader,
                            PassParameters,
                            ViewRect);
                    }
                }
            }

            GraphBuilder.Execute();
        });
}

