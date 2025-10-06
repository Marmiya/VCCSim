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
#include "Engine/TextureRenderTarget2D.h"
#include "RenderGraphBuilder.h"
#include "PixelShaderUtils.h"
#include "SceneRenderBuilderInterface.h"
#include "SceneRendering.h"
#include "LegacyScreenPercentageDriver.h"

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
            case ESensorType::SegmentCamera:
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
              case ESensorType::SegmentCamera:
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
            SensorType == ESensorType::SegmentCamera ||
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
        NewGroup.OwnerActor = BaseCamera->GetOwner();
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
        SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
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

class FRGBPS : public FGlobalShader
{
public:
    DECLARE_GLOBAL_SHADER(FRGBPS);
    SHADER_USE_PARAMETER_STRUCT(FRGBPS, FGlobalShader);

    BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
        SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
        SHADER_PARAMETER_RDG_TEXTURE(Texture2D, SceneColorTexture)
        SHADER_PARAMETER_SAMPLER(SamplerState, SceneColorSampler)
        RENDER_TARGET_BINDING_SLOTS()
    END_SHADER_PARAMETER_STRUCT()

    static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
    {
        return true;
    }
};

IMPLEMENT_GLOBAL_SHADER(FRGBPS, "/VCCSim/RGBCapture.usf", "MainPS", SF_Pixel);

class FNormalPS : public FGlobalShader
{
public:
    DECLARE_GLOBAL_SHADER(FNormalPS);
    SHADER_USE_PARAMETER_STRUCT(FNormalPS, FGlobalShader);

    BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
        SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
        SHADER_PARAMETER_RDG_TEXTURE(Texture2D, InputGBufferA)
        RENDER_TARGET_BINDING_SLOTS()
    END_SHADER_PARAMETER_STRUCT()

    static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
    {
        return true;
    }
};

IMPLEMENT_GLOBAL_SHADER(FNormalPS, "/VCCSim/NormalCapture.usf", "MainPS", SF_Pixel);

class FSegmentationPS : public FGlobalShader
{
public:
    DECLARE_GLOBAL_SHADER(FSegmentationPS);
    SHADER_USE_PARAMETER_STRUCT(FSegmentationPS, FGlobalShader);

    BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
        SHADER_PARAMETER_RDG_TEXTURE_SRV(Texture2D<uint2>, CustomStencilTexture)
        RENDER_TARGET_BINDING_SLOTS()
    END_SHADER_PARAMETER_STRUCT()

    static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
    {
        return true;
    }
};

IMPLEMENT_GLOBAL_SHADER(FSegmentationPS, "/VCCSim/SegmentationCapture.usf", "MainPS", SF_Pixel);


void ARecorder::RenderViewGroupsRDG()
{
    if (!GetWorld() || CameraViewGroups.Num() == 0)
    {
        return;
    }

    UWorld* World = GetWorld();
    if (!World || !World->Scene)
    {
        return;
    }

    FSceneInterface* Scene = World->Scene;

    struct FCameraGroupData
    {
        TArray<UCameraBaseComponent*> Cameras;
        FVector ViewOrigin;
        FMatrix ViewRotationMatrix;
        FMatrix ProjectionMatrix;
        FIntPoint Resolution;
        AActor* OwnerActor;
    };

    TArray<FCameraGroupData> GroupDataArray;

    for (const auto& CameraViewGroup : CameraViewGroups)
    {
        FCameraGroupData GroupData;
        GroupData.OwnerActor = CameraViewGroup.OwnerActor;

        for (const auto& CamPair : CameraViewGroup.Cameras)
        {
            if (CamPair.Value && CamPair.Key)
            {
                GroupData.Cameras.Add(CamPair.Key);
            }
        }

        if (GroupData.Cameras.Num() == 0)
        {
            continue;
        }

        UCameraBaseComponent* RepresentativeCamera = GroupData.Cameras[0];

        GroupData.ViewOrigin = RepresentativeCamera->GetComponentLocation();
        GroupData.ViewRotationMatrix = FInverseRotationMatrix(RepresentativeCamera->GetComponentRotation()) * FMatrix(
            FPlane(0, 0, 1, 0),
            FPlane(1, 0, 0, 0),
            FPlane(0, 1, 0, 0),
            FPlane(0, 0, 0, 1));

        const float FOVRad = FMath::DegreesToRadians(RepresentativeCamera->FOV);
        const float HalfFOV = FOVRad * 0.5f;
        GroupData.Resolution = RepresentativeCamera->GetResolution();
        const float AspectRatio = static_cast<float>(GroupData.Resolution.X) / GroupData.Resolution.Y;

        GroupData.ProjectionMatrix = FReversedZPerspectiveMatrix(
            HalfFOV,
            AspectRatio,
            1.0f,
            GNearClippingPlane);

        GroupDataArray.Add(MoveTemp(GroupData));
    }

    TUniquePtr<ISceneRenderBuilder> SceneRenderBuilder = ISceneRenderBuilder::Create(Scene);
    const ERHIFeatureLevel::Type FeatureLevel = Scene->GetFeatureLevel();

    for (const FCameraGroupData& GroupData : GroupDataArray)
    {
        UTextureRenderTarget2D* TempRenderTarget = NewObject<UTextureRenderTarget2D>();
        TempRenderTarget->InitAutoFormat(GroupData.Resolution.X, GroupData.Resolution.Y);
        TempRenderTarget->UpdateResourceImmediate(true);
        FTextureRenderTargetResource* TempRTResource = TempRenderTarget->GameThread_GetRenderTargetResource();

        FSceneViewFamilyContext* ViewFamily = new FSceneViewFamilyContext(
            FSceneViewFamily::ConstructionValues(
                TempRTResource,
                Scene,
                FEngineShowFlags(ESFIM_Game))
            .SetTime(FGameTime::GetTimeSinceAppStart())
            .SetRealtimeUpdate(true));

        ViewFamily->EngineShowFlags.ScreenPercentage = false;
        ViewFamily->EngineShowFlags.SetRendering(true);
        ViewFamily->EngineShowFlags.SetPostProcessing(true);
        ViewFamily->SetScreenPercentageInterface(new FLegacyScreenPercentageDriver(
            *ViewFamily, 1.0f));

        FSceneViewInitOptions ViewInitOptions;
        ViewInitOptions.ViewFamily = ViewFamily;
        ViewInitOptions.SetViewRectangle(FIntRect(0, 0, GroupData.Resolution.X, GroupData.Resolution.Y));
        ViewInitOptions.ViewOrigin = GroupData.ViewOrigin;
        ViewInitOptions.ViewRotationMatrix = GroupData.ViewRotationMatrix;
        ViewInitOptions.ProjectionMatrix = GroupData.ProjectionMatrix;
        ViewInitOptions.BackgroundColor = FLinearColor::Black;
        ViewInitOptions.OverrideFarClippingPlaneDistance = -1.0f;

        FSceneView* View = new FSceneView(ViewInitOptions);
        if (GroupData.OwnerActor)
        {
            for (UActorComponent* Component : GroupData.OwnerActor->GetComponents())
            {
                if (UPrimitiveComponent* PrimComp = Cast<UPrimitiveComponent>(Component))
                {
                    View->HiddenPrimitives.Add(PrimComp->GetPrimitiveSceneId());
                }
            }
        }
        ViewFamily->Views.Add(View);

        FSceneRenderer* SceneRenderer = SceneRenderBuilder->CreateSceneRenderer(ViewFamily);

        SceneRenderBuilder->AddRenderer(SceneRenderer, TEXT("VCCSimMultiSensorCapture"),
            [GroupData, FeatureLevel](FRDGBuilder& GraphBuilder, const FSceneRenderFunctionInputs& Inputs)
            {
                if (!Inputs.Renderer)
                {
                    return false;
                }

                Inputs.Renderer->Render(GraphBuilder, Inputs.SceneUpdateInputs);

                const FSceneTextures& SceneTextures = Inputs.Renderer->GetActiveSceneTextures();

                for (UCameraBaseComponent* Camera : GroupData.Cameras)
                {
                    FTextureRenderTargetResource* RTResource = Camera->GetRenderTarget()->GetRenderTargetResource();

                    ESensorType SensorType = Camera->GetSensorType();
                    FString TextureName = FString::Printf(TEXT("CameraRT_%d"), Camera->GetSensorIndex());
                    FRDGTextureRef OutputTexture = GraphBuilder.RegisterExternalTexture(
                        CreateRenderTarget(RTResource->GetRenderTargetTexture(), *TextureName));

                    const FIntPoint Extent = OutputTexture->Desc.Extent;
                    const FIntRect ViewRect(0, 0, Extent.X, Extent.Y);

                    if (SensorType == ESensorType::DepthCamera)
                    {
                        auto* PassParameters = GraphBuilder.AllocParameters<FDepthPS::FParameters>();
                        PassParameters->View = Inputs.Renderer->Views[0].ViewUniformBuffer;
                        PassParameters->SceneDepthTexture = SceneTextures.Depth.Resolve;
                        PassParameters->SceneDepthSampler = TStaticSamplerState<SF_Point, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
                        PassParameters->RenderTargets[0] = FRenderTargetBinding(OutputTexture, ERenderTargetLoadAction::EClear);

                        TShaderMapRef<FDepthPS> PixelShader(GetGlobalShaderMap(FeatureLevel));
                        FPixelShaderUtils::AddFullscreenPass(GraphBuilder, GetGlobalShaderMap(FeatureLevel),
                            RDG_EVENT_NAME("VCCSimDepthCapture"), PixelShader, PassParameters, ViewRect);
                    }
                    else if (SensorType == ESensorType::RGBCamera)
                    {
                        auto* PassParameters = GraphBuilder.AllocParameters<FRGBPS::FParameters>();
                        PassParameters->View = Inputs.Renderer->Views[0].ViewUniformBuffer;
                        PassParameters->SceneColorTexture = SceneTextures.Color.Resolve;
                        PassParameters->SceneColorSampler = TStaticSamplerState<SF_Bilinear, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
                        PassParameters->RenderTargets[0] = FRenderTargetBinding(OutputTexture, ERenderTargetLoadAction::EClear);

                        TShaderMapRef<FRGBPS> PixelShader(GetGlobalShaderMap(FeatureLevel));
                        FPixelShaderUtils::AddFullscreenPass(GraphBuilder, GetGlobalShaderMap(FeatureLevel),
                            RDG_EVENT_NAME("VCCSimRGBCapture"), PixelShader, PassParameters, ViewRect);
                    }
                    else if (SensorType == ESensorType::NormalCamera)
                    {
                        auto* PassParameters = GraphBuilder.AllocParameters<FNormalPS::FParameters>();
                        PassParameters->View = Inputs.Renderer->Views[0].ViewUniformBuffer;
                        PassParameters->InputGBufferA = SceneTextures.GBufferA;
                        PassParameters->RenderTargets[0] = FRenderTargetBinding(OutputTexture, ERenderTargetLoadAction::EClear);

                        TShaderMapRef<FNormalPS> PixelShader(GetGlobalShaderMap(FeatureLevel));
                        FPixelShaderUtils::AddFullscreenPass(GraphBuilder, GetGlobalShaderMap(FeatureLevel),
                            RDG_EVENT_NAME("VCCSimNormalCapture"), PixelShader, PassParameters, ViewRect);
                    }
                    else if (SensorType == ESensorType::SegmentCamera)
                    {
                        auto* PassParameters = GraphBuilder.AllocParameters<FSegmentationPS::FParameters>();
                        PassParameters->CustomStencilTexture = SceneTextures.CustomDepth.Stencil;
                        PassParameters->RenderTargets[0] = FRenderTargetBinding(OutputTexture, ERenderTargetLoadAction::EClear);

                        TShaderMapRef<FSegmentationPS> PixelShader(GetGlobalShaderMap(FeatureLevel));
                        FPixelShaderUtils::AddFullscreenPass(GraphBuilder, GetGlobalShaderMap(FeatureLevel),
                            RDG_EVENT_NAME("VCCSimSegmentationCapture"), PixelShader, PassParameters, ViewRect);
                    }
                }

                return true;
            });
    }

    SceneRenderBuilder->Execute();
}