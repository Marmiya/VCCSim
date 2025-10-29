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
#include "Simulation/ActorRegistry.h"
#include "Engine/TextureRenderTarget2D.h"
#include "RenderGraphBuilder.h"
#include "PixelShaderUtils.h"
#include "RenderGraphUtils.h"
#include "SceneRenderBuilderInterface.h"
#include "SceneRendering.h"
#include "LegacyScreenPercentageDriver.h"
#include "Sensors/LidarSensor.h"

DEFINE_LOG_CATEGORY_STATIC(LogRecorder, Log, All);

class FDepthPS : public FGlobalShader
{
public:
    DECLARE_GLOBAL_SHADER(FDepthPS);
    SHADER_USE_PARAMETER_STRUCT(FDepthPS, FGlobalShader);

    BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
        SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
        SHADER_PARAMETER_RDG_TEXTURE(Texture2D, InputTexture)
        SHADER_PARAMETER_SAMPLER(SamplerState, InputSampler)
        RENDER_TARGET_BINDING_SLOTS()
    END_SHADER_PARAMETER_STRUCT()
};

IMPLEMENT_GLOBAL_SHADER(FDepthPS, "/VCCSim/DepthCapture.usf", "MainPS", SF_Pixel);

class FRGBPS : public FGlobalShader
{
public:
    DECLARE_GLOBAL_SHADER(FRGBPS);
    SHADER_USE_PARAMETER_STRUCT(FRGBPS, FGlobalShader);

    BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
        SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
        SHADER_PARAMETER_RDG_TEXTURE(Texture2D, InputTexture)
        SHADER_PARAMETER_SAMPLER(SamplerState, InputSampler)
        RENDER_TARGET_BINDING_SLOTS()
    END_SHADER_PARAMETER_STRUCT()
};

IMPLEMENT_GLOBAL_SHADER(FRGBPS, "/VCCSim/RGBCapture.usf", "MainPS", SF_Pixel);

class FNormalPS : public FGlobalShader
{
public:
    DECLARE_GLOBAL_SHADER(FNormalPS);
    SHADER_USE_PARAMETER_STRUCT(FNormalPS, FGlobalShader);

    BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
        SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
        SHADER_PARAMETER_RDG_TEXTURE(Texture2D, InputTexture)
        RENDER_TARGET_BINDING_SLOTS()
    END_SHADER_PARAMETER_STRUCT()
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
};

IMPLEMENT_GLOBAL_SHADER(FSegmentationPS, "/VCCSim/SegmentationCapture.usf", "MainPS", SF_Pixel);

BEGIN_SHADER_PARAMETER_STRUCT(FEnqueueCopyTexturePass, )
    RDG_TEXTURE_ACCESS(Texture, ERHIAccess::CopySrc)
END_SHADER_PARAMETER_STRUCT()

ARecorder::ARecorder()
{
    PrimaryActorTick.bCanEverTick = false;

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
    InitializeReadbackWorker();

    SetupSensorProperties();
    GroupCamerasByPose();

    GetWorld()->GetTimerManager().SetTimer(
        RecordingTimerHandle,
        [this]()
        {
            TickRecording();
        },
        RecorderInterval,
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

    if (GetWorld() && RecordingTimerHandle.IsValid())
    {
        GetWorld()->GetTimerManager().ClearTimer(RecordingTimerHandle);
        RecordingTimerHandle.Invalidate();
    }

    ShutdownReadbackWorker();
    ShutdownAsyncWriter();

    bIsRecording = false;
    RecordState = false;
    LastSensorCaptureTimes.Empty();
    ActorRegistry.UnregisterAllActors();

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
    CollectData();
}

void ARecorder::CreateActorDirectories(const FString& ActorName, TSet<ESensorType>&& SensorTypes)
{
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();

    FString ActorDir = FPaths::Combine(RecordingPath, ActorName);
    if (PlatformFile.CreateDirectoryTree(*ActorDir))
    {
        UE_LOG(LogRecorder, Log, TEXT("Created actor directory: %s"), *ActorDir);
    }

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
                UE_LOG(LogRecorder, Warning, TEXT("Unknown sensor type for directory creation"));
                continue;
        }
        PlatformFile.CreateDirectoryTree(*SensorDir);
    }
}

void ARecorder::CollectData()
{
    if (CameraViewGroups.Num() == 0)
    {
        return;
    }

    double CurrentTime = FPlatformTime::Seconds();

    CheckRPCTimeouts(CurrentTime);

    // Step 1: Determine which cameras need capture this frame

    for (FCameraViewGroup& ViewGroup : CameraViewGroups)
    {
        for (auto& SensorAndState : ViewGroup.Cameras)
        {
            SensorAndState.Value = ShouldCaptureSensor(SensorAndState.Key, CurrentTime);
        }
    }

    // Step 2: Render current frame and setup readbacks
    RenderViewGroupsRDG(CurrentTime);

    // Step 3: Process LiDAR sensors (no GPU dependency)
    for (const auto& SensorEntry : SensorRegistry.GetAllSensors())
    {
        USensorBaseComponent* Sensor = SensorEntry.GetSensorComponent();
        if (Sensor && Sensor->IsA<ULidarComponent>() && ShouldCaptureSensor(Sensor, CurrentTime))
        {
            SampleLiDARData(Sensor);
        }
    }

    // Step 4: Record actor poses
    RecordActorPoses(std::move(CurrentTime));
}

void ARecorder::RenderViewGroupsRDG(const double& CaptureTime)
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
        GroupData.ViewRotationMatrix = FInverseRotationMatrix(
            RepresentativeCamera->GetComponentRotation()) *
                FMatrix(
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
            [this, GroupData, FeatureLevel, CaptureTime](FRDGBuilder& GraphBuilder, const FSceneRenderFunctionInputs& Inputs)
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

                    bool bValidSensorType = false;

                    if (SensorType == ESensorType::DepthCamera)
                    {
                        auto* PassParameters = GraphBuilder.AllocParameters<FDepthPS::FParameters>();
                        PassParameters->View = Inputs.Renderer->Views[0].ViewUniformBuffer;
                        PassParameters->InputTexture = SceneTextures.Depth.Resolve;
                        PassParameters->InputSampler = TStaticSamplerState<SF_Point, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
                        PassParameters->RenderTargets[0] = FRenderTargetBinding(OutputTexture, ERenderTargetLoadAction::EClear);

                        TShaderMapRef<FDepthPS> PixelShader(GetGlobalShaderMap(FeatureLevel));
                        FPixelShaderUtils::AddFullscreenPass(GraphBuilder, GetGlobalShaderMap(FeatureLevel),
                            RDG_EVENT_NAME("VCCSimDepthCapture"), PixelShader, PassParameters, ViewRect);
                        bValidSensorType = true;
                    }
                    else if (SensorType == ESensorType::RGBCamera)
                    {
                        auto* PassParameters = GraphBuilder.AllocParameters<FRGBPS::FParameters>();
                        PassParameters->View = Inputs.Renderer->Views[0].ViewUniformBuffer;
                        PassParameters->InputTexture = SceneTextures.Color.Resolve;
                        PassParameters->InputSampler = TStaticSamplerState<SF_Bilinear, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
                        PassParameters->RenderTargets[0] = FRenderTargetBinding(OutputTexture, ERenderTargetLoadAction::EClear);

                        TShaderMapRef<FRGBPS> PixelShader(GetGlobalShaderMap(FeatureLevel));
                        FPixelShaderUtils::AddFullscreenPass(GraphBuilder, GetGlobalShaderMap(FeatureLevel),
                            RDG_EVENT_NAME("VCCSimRGBCapture"), PixelShader, PassParameters, ViewRect);
                        bValidSensorType = true;
                    }
                    else if (SensorType == ESensorType::NormalCamera)
                    {
                        auto* PassParameters = GraphBuilder.AllocParameters<FNormalPS::FParameters>();
                        PassParameters->View = Inputs.Renderer->Views[0].ViewUniformBuffer;
                        PassParameters->InputTexture = SceneTextures.GBufferA;
                        PassParameters->RenderTargets[0] = FRenderTargetBinding(OutputTexture, ERenderTargetLoadAction::EClear);

                        TShaderMapRef<FNormalPS> PixelShader(GetGlobalShaderMap(FeatureLevel));
                        FPixelShaderUtils::AddFullscreenPass(GraphBuilder, GetGlobalShaderMap(FeatureLevel),
                            RDG_EVENT_NAME("VCCSimNormalCapture"), PixelShader, PassParameters, ViewRect);
                        bValidSensorType = true;
                    }
                    else if (SensorType == ESensorType::SegmentationCamera)
                    {
                        const FRDGTextureSRVRef CustomStencilTexture = SceneTextures.CustomDepth.Stencil;

                        if (CustomStencilTexture)
                        {
                            auto* PassParameters = GraphBuilder.AllocParameters<FSegmentationPS::FParameters>();
                            PassParameters->CustomStencilTexture = CustomStencilTexture;
                            PassParameters->RenderTargets[0] = FRenderTargetBinding(OutputTexture, ERenderTargetLoadAction::EClear);

                            TShaderMapRef<FSegmentationPS> PixelShader(GetGlobalShaderMap(FeatureLevel));
                            FPixelShaderUtils::AddFullscreenPass(GraphBuilder, GetGlobalShaderMap(FeatureLevel),
                                RDG_EVENT_NAME("VCCSimSegmentationCapture"), PixelShader, PassParameters, ViewRect);
                        }
                        else
                        {
                            AddClearRenderTargetPass(GraphBuilder, OutputTexture, FLinearColor::Black);
                        }

                        bValidSensorType = true;
                    }
                    else
                    {
                        UE_LOG(LogRecorder, Error, TEXT("Unsupported sensor type for camera %d"), Camera->GetSensorIndex());
                    }

                    if (bValidSensorType)
                    {
                        TSharedPtr<FRHIGPUTextureReadback> Readback = MakeShared<FRHIGPUTextureReadback>(
                            *FString::Printf(TEXT("CameraReadback_%d"), Camera->GetSensorIndex()));

                        FEnqueueCopyTexturePass* PassParameters = GraphBuilder.AllocParameters<FEnqueueCopyTexturePass>();
                        PassParameters->Texture = OutputTexture;

                        GraphBuilder.AddPass(
                            RDG_EVENT_NAME("EnqueueCopyAndEnqueue(%s)", OutputTexture->Name),
                            PassParameters,
                            ERDGPassFlags::Readback,
                            [this, Readback, OutputTexture, CaptureTime, Camera](FRDGAsyncTask, FRHICommandList& RHICmdList)
                            {
                                Readback->EnqueueCopy(RHICmdList, OutputTexture->GetRHI(), FResolveRect());

                                FPendingReadback PendingData;
                                PendingData.Readback = Readback;
                                PendingData.CaptureTimestamp = CaptureTime;
                                PendingData.Camera = Camera;

                                this->PendingReadbacks.Enqueue(MoveTemp(PendingData));
                            });
                    }
                }

                return true;
            });
    }

    SceneRenderBuilder->Execute();
}

void ARecorder::ProcessPendingReadback(const FPendingReadback& PendingData)
{
    UCameraBaseComponent* Camera = PendingData.Camera.Get();
    if (!Camera)
    {
        return;
    }

    TSharedPtr<FRHIGPUTextureReadback> Readback = PendingData.Readback;

    const double CaptureTime = PendingData.CaptureTimestamp;

    Async(EAsyncExecution::TaskGraph,
    [Camera, Readback, CaptureTime, this]()
    {
        int32 Attempts = 0;
        while (!Readback->IsReady())
        {
            FPlatformProcess::YieldThread();
            Attempts++;
            if (Attempts > 5000000)
            {
                UE_LOG(LogRecorder, Error, TEXT("Readback timeout for sensor %d"), Camera->GetSensorIndex());
                return;
            }
        }

        int32 RowPitchInPixels = 0;
        const void* PixelData = Readback->Lock(RowPitchInPixels);
        if (!PixelData)
        {
            UE_LOG(LogRecorder, Error, TEXT("Failed to lock readback for sensor %d"), Camera->GetSensorIndex());
            return;
        }

        const FIntPoint Resolution = Camera->GetResolution();
        const int32 NumPixels = Resolution.X * Resolution.Y;
        const bool bHasPadding = (RowPitchInPixels != Resolution.X);

        FSensorDataPacket Packet;
        Packet.Type = Camera->GetSensorType();
        Packet.SensorIndex = Camera->GetSensorIndex();
        Packet.OwnerActor = Camera->GetOwner();
        Packet.bValid = true;

        switch (Packet.Type)
        {
        case ESensorType::NormalCamera:
            {
                auto NormalData = MakeShared<FNormalCameraData>();
                NormalData->Timestamp = CaptureTime;
                NormalData->Width = Resolution.X;
                NormalData->Height = Resolution.Y;
                NormalData->SensorIndex = Packet.SensorIndex;
                NormalData->Data.SetNumUninitialized(NumPixels);
                const FFloat16Color* SourceColors = static_cast<const FFloat16Color*>(PixelData);

                if (bHasPadding)
                {
                    for (int32 Row = 0; Row < Resolution.Y; ++Row)
                    {
                        const FFloat16Color* RowSrc = SourceColors + Row * RowPitchInPixels;
                        FFloat16Color* RowDst = NormalData->Data.GetData() + Row * Resolution.X;
                        FMemory::Memcpy(RowDst, RowSrc, Resolution.X * sizeof(FFloat16Color));
                    }
                }
                else
                {
                    FMemory::Memcpy(NormalData->Data.GetData(), SourceColors, NumPixels * sizeof(FFloat16Color));
                }
                Packet.Data = NormalData;
                break;
            }
        case ESensorType::SegmentationCamera:
            {
                auto SegData = MakeShared<FSegmentationCameraData>();
                SegData->Timestamp = CaptureTime;
                SegData->Width = Resolution.X;
                SegData->Height = Resolution.Y;
                SegData->SensorIndex = Packet.SensorIndex;
                SegData->Data.SetNumUninitialized(NumPixels);
                const FColor* SourceColors = static_cast<const FColor*>(PixelData);

                if (bHasPadding)
                {
                    for (int32 Row = 0; Row < Resolution.Y; ++Row)
                    {
                        const FColor* RowSrc = SourceColors + Row * RowPitchInPixels;
                        FColor* RowDst = SegData->Data.GetData() + Row * Resolution.X;
                        FMemory::Memcpy(RowDst, RowSrc, Resolution.X * sizeof(FColor));
                    }
                }
                else
                {
                    FMemory::Memcpy(SegData->Data.GetData(), SourceColors, NumPixels * sizeof(FColor));
                }
                Packet.Data = SegData;
                break;
            }
        case ESensorType::RGBCamera:
            {
                auto RGBData = MakeShared<FRGBCameraData>();
                RGBData->Timestamp = CaptureTime;
                RGBData->Width = Resolution.X;
                RGBData->Height = Resolution.Y;
                RGBData->SensorIndex = Packet.SensorIndex;
                RGBData->RGBData.SetNumUninitialized(NumPixels);
                const FColor* SourceColors = static_cast<const FColor*>(PixelData);

                if (bHasPadding)
                {
                    for (int32 Row = 0; Row < Resolution.Y; ++Row)
                    {
                        const FColor* RowSrc = SourceColors + Row * RowPitchInPixels;
                        FColor* RowDst = RGBData->RGBData.GetData() + Row * Resolution.X;
                        FMemory::Memcpy(RowDst, RowSrc, Resolution.X * sizeof(FColor));
                    }
                }
                else
                {
                    FMemory::Memcpy(RGBData->RGBData.GetData(), SourceColors, NumPixels * sizeof(FColor));
                }
                Packet.Data = RGBData;
                break;
            }
        case ESensorType::DepthCamera:
            {
                auto DepthData = MakeShared<FDepthCameraData>();
                DepthData->Timestamp = CaptureTime;
                DepthData->Width = Resolution.X;
                DepthData->Height = Resolution.Y;
                DepthData->SensorIndex = Packet.SensorIndex;
                DepthData->DepthData.SetNumUninitialized(NumPixels);
                const FFloat16Color* SourceColors = static_cast<const FFloat16Color*>(PixelData);

                if (bHasPadding)
                {
                    for (int32 Row = 0; Row < Resolution.Y; ++Row)
                    {
                        const FFloat16Color* RowSrc = SourceColors + Row * RowPitchInPixels;
                        for (int32 Col = 0; Col < Resolution.X; ++Col)
                        {
                            DepthData->DepthData[Row * Resolution.X + Col] = RowSrc[Col].R.GetFloat();
                        }
                    }
                }
                else
                {
                    for (int32 i = 0; i < NumPixels; i++)
                    {
                        DepthData->DepthData[i] = SourceColors[i].R.GetFloat();
                    }
                }
                Packet.Data = DepthData;
                break;
            }
        default:
            Packet.bValid = false;
            break;
        }

        Readback->Unlock();

        if (Packet.bValid)
        {
            bool bHasRPCCallback = false;
            {
                FScopeLock Lock(&this->RPCRequestLock);
                bHasRPCCallback = this->PendingRPCCallbacks.Contains(Camera);
            }

            if (bHasRPCCallback)
            {
                this->TriggerRPCCallbacks(Camera, Packet);
            }

            if (this->bIsRecording && this->FileWriter.IsValid())
            {
                this->FileWriter->WriteData(FSensorDataPacket(Packet));
            }
        }
    });
}

void ARecorder::SampleLiDARData(USensorBaseComponent* Sensor)
{
    ULidarComponent* Lidar = Cast<ULidarComponent>(Sensor);
    if (!Lidar)
    {
        UE_LOG(LogRecorder, Error, TEXT("Failed to cast sensor to ULidarComponent"));
        return;
    }

    double CaptureTimestamp = LastSensorCaptureTimes[Sensor];

    Async(EAsyncExecution::ThreadPool,
        [Lidar, CaptureTimestamp, this]()
        {
            TArray<FVector3f> PointCloudData = Lidar->GetPointCloudData();

            FSensorDataPacket Packet;
            Packet.Type = ESensorType::Lidar;
            Packet.SensorIndex = Lidar->GetSensorIndex();
            Packet.OwnerActor = Lidar->GetOwner();
            Packet.bValid = true;

            auto LidarData = MakeShared<FLiDARData>();
            LidarData->Timestamp = CaptureTimestamp;
            LidarData->SensorIndex = Packet.SensorIndex;
            LidarData->Data = MoveTemp(PointCloudData);
            Packet.Data = LidarData;

            if (this->FileWriter.IsValid())
            {
                this->FileWriter->WriteData(MoveTemp(Packet));
            }
        });
}

bool ARecorder::ShouldCaptureSensor(USensorBaseComponent* Sensor, double CurrentTime)
{
    bool bConsumeForceRequest = false;
    {
        FScopeLock Lock(&RPCRequestLock);
        if (ForceCaptureThisFrame.Contains(Sensor))
        {
            ForceCaptureThisFrame.Remove(Sensor);
            bConsumeForceRequest = true;
            LastSensorCaptureTimes.Add(Sensor, CurrentTime);
        }
    }

    if (bConsumeForceRequest)
    {
        return true;
    }

    if (!RecordState)
    {
        return false;
    }

    double* IntervalPtr = SensorIntervals.Find(Sensor);
    double SensorInterval = IntervalPtr ? *IntervalPtr : Sensor->GetRecordInterval();
    if (!IntervalPtr)
    {
        SensorIntervals.Add(Sensor, SensorInterval);
    }

    if (!LastSensorCaptureTimes.Contains(Sensor))
    {
        LastSensorCaptureTimes.Add(Sensor, CurrentTime);
        return true;
    }

    const double LastCaptureTime = LastSensorCaptureTimes[Sensor];
    const double TimeSinceLastCapture = CurrentTime - LastCaptureTime;

    if (TimeSinceLastCapture >= SensorInterval)
    {
        LastSensorCaptureTimes[Sensor] = CurrentTime;
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

void ARecorder::EnsureOnDemandCaptureSetup()
{
    
    if (bIsRecording)
    {
        return;
    }
    UE_LOG(LogRecorder, Log, TEXT("ARecorder::EnsureOnDemandCaptureSetup()"));
    InitializeReadbackWorker();

    SensorRegistry.CleanupInvalidEntries();
    SetupSensorProperties();
    GroupCamerasByPose();
}

void ARecorder::ScheduleManualCaptureTick()
{
    UE_LOG(LogRecorder, Log, TEXT("ARecorder::ScheduleManualCaptureTick()"));
    bool bExpected = false;
    if (!bManualCaptureTickScheduled.compare_exchange_strong(bExpected, true))
    {
        return;
    }

    TWeakObjectPtr<ARecorder> WeakThis(this);
    AsyncTask(ENamedThreads::GameThread, [WeakThis]()
    {
        if (ARecorder* Recorder = WeakThis.Get())
        {
            Recorder->EnsureOnDemandCaptureSetup();
            Recorder->CollectData();
            Recorder->bManualCaptureTickScheduled.store(false);
        }
    });
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

void ARecorder::InitializeAsyncWriter()
{
    if (FileWriter.IsValid())
    {
        return;
    }

    FileWriter = MakeUnique<FAsyncFileWriter>(RecordingPath);
    UE_LOG(LogRecorder, Log, TEXT("Async file writer initialized"));
}

void ARecorder::RecordActorPoses(double&& CurrentTime)
{
    ActorRegistry.CleanupInvalidEntries();

    TArray<FActorRegistryEntry> AllActors = ActorRegistry.GetAllActors();

    for (FActorRegistryEntry& Entry : AllActors)
    {
        if (!Entry.IsValid())
        {
            continue;
        }

        AActor* Actor = Entry.GetActor();
        if (!Actor)
        {
            continue;
        }

        FActorRegistryEntry* EntryPtr = ActorRegistry.FindEntry(Actor);
        if (!EntryPtr)
        {
            continue;
        }

        if (EntryPtr->LastRecordTime > 0.0 &&
            (CurrentTime - EntryPtr->LastRecordTime) < EntryPtr->RecordInterval)
        {
            continue;
        }

        EntryPtr->LastRecordTime = CurrentTime;

        FVector Location = Actor->GetActorLocation();
        FQuat Rotation = Actor->GetActorQuat();

        FString PoseLine = FString::Printf(TEXT("%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n"),
            CurrentTime,
            Location.X, Location.Y, Location.Z,
            Rotation.X, Rotation.Y, Rotation.Z, Rotation.W);

        FString PoseFilePath = EntryPtr->PoseFilePath;

        Async(EAsyncExecution::ThreadPool,
            [PoseLine, PoseFilePath]()
            {
                FFileHelper::SaveStringToFile(PoseLine, *PoseFilePath,
                    FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM,
                    &IFileManager::Get(), FILEWRITE_Append);
            });
    }
}

void ARecorder::InitializeReadbackWorker()
{
    if (ReadbackWorker.IsValid())
    {
        return;
    }

    bReadbackWorkerShouldStop.store(false);
    ReadbackWorker = MakeUnique<FReadbackWorker>(this);
    ReadbackThread.Reset(FRunnableThread::Create(ReadbackWorker.Get(),
        TEXT("RecorderReadbackWorker"), 0, TPri_Highest));
    UE_LOG(LogRecorder, Log, TEXT("Readback worker initialized"));
}

void ARecorder::ShutdownReadbackWorker()
{
    bReadbackWorkerShouldStop.store(true);
    if (ReadbackWorker.IsValid())
    {
        ReadbackWorker->Stop();
    }
    if (ReadbackThread.IsValid())
    {
        ReadbackThread->WaitForCompletion();
        ReadbackThread.Reset();
    }
    ReadbackWorker.Reset();
    UE_LOG(LogRecorder, Log, TEXT("Readback worker shutdown"));
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

uint32 ARecorder::FReadbackWorker::Run()
{
    while (!bShouldStop.load())
    {
        FPendingReadback PendingData;
        while (Owner->PendingReadbacks.Dequeue(PendingData))
        {
            if (!Owner->bReadbackWorkerShouldStop.load())
            {
                Owner->ProcessPendingReadback(PendingData);
            }
        }

        FPlatformProcess::YieldThread();
    }
    return 0;
}

void ARecorder::SubmitCameraRequest(
    UCameraBaseComponent* Camera,
    TFunction<void(const FSensorDataPacket&)> OnSuccess,
    TFunction<void(const FString&)> OnError)
{
    if (!Camera)
    {
        OnError(TEXT("Invalid camera pointer"));
        return;
    }

    {
        FScopeLock Lock(&RPCRequestLock);

        static constexpr int32 MaxPendingCallbacksPerCamera = 50;
        TArray<FRPCRequestCallback>& Callbacks = PendingRPCCallbacks.FindOrAdd(Camera);

        if (Callbacks.Num() >= MaxPendingCallbacksPerCamera)
        {
            OnError(TEXT("RPC request queue full for this camera"));
            UE_LOG(LogRecorder, Warning, TEXT("RPC queue full for camera %d"), Camera->GetSensorIndex());
            return;
        }

        FRPCRequestCallback Callback;
        Callback.OnSuccess = OnSuccess;
        Callback.OnError = OnError;
        Callback.RequestTime = FPlatformTime::Seconds();
        Callbacks.Add(MoveTemp(Callback));
        ForceCaptureThisFrame.Add(Camera);
    }

    if (!bIsRecording)
    {
        ScheduleManualCaptureTick();
    }
}

void ARecorder::CheckRPCTimeouts(double CurrentTime)
{
    FScopeLock Lock(&RPCRequestLock);

    TArray<TWeakObjectPtr<USensorBaseComponent>> ToRemove;

    for (auto& Pair : PendingRPCCallbacks)
    {
        TArray<FRPCRequestCallback>& Callbacks = Pair.Value;

        Callbacks.RemoveAll([CurrentTime](const FRPCRequestCallback& CB)
        {
            if (CurrentTime - CB.RequestTime > CB.TimeoutDuration)
            {
                CB.OnError(TEXT("Request timeout"));
                UE_LOG(LogRecorder, Warning, TEXT("RPC request timeout"));
                return true;
            }
            return false;
        });

        if (Callbacks.Num() == 0)
        {
            ToRemove.Add(Pair.Key);
        }
    }

    for (TWeakObjectPtr Sensor : ToRemove)
    {
        PendingRPCCallbacks.Remove(Sensor);
    }
}

void ARecorder::TriggerRPCCallbacks(UCameraBaseComponent* Camera, const FSensorDataPacket& Packet)
{
    TArray<FRPCRequestCallback> CallbacksCopy;

    {
        FScopeLock Lock(&RPCRequestLock);

        if (TArray<FRPCRequestCallback>* Callbacks = PendingRPCCallbacks.Find(Camera))
        {
            CallbacksCopy = *Callbacks;
            PendingRPCCallbacks.Remove(Camera);
        }
    }

    for (const FRPCRequestCallback& CB : CallbacksCopy)
    {
        CB.OnSuccess(Packet);
    }

    UE_LOG(LogRecorder, Log, TEXT("Triggered %d RPC callbacks for camera %d"),
        CallbacksCopy.Num(), Camera->GetSensorIndex());
}
