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

#pragma once

#include "CoreMinimal.h"
#include "SensorRegistry.h"
#include "Utils/AsyncFileWriter.h"
#include "Sensors/ISensorDataProvider.h"
#include "Async/Async.h"
#include "RenderGraphDefinitions.h"
#include "RenderGraphPass.h"
#include "RHICommandList.h"
#include "Recorder.generated.h"

// MRT pass parameters for sensor data capture
BEGIN_SHADER_PARAMETER_STRUCT(FMRTPassParameters, )
    RENDER_TARGET_BINDING_SLOTS()
END_SHADER_PARAMETER_STRUCT()


UCLASS()
class VCCSIM_API ARecorder : public AActor
{
    GENERATED_BODY()

public:
    ARecorder();
	~ARecorder();
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

    UPROPERTY(EditAnywhere, Category = "Recording")
    FString RecordingPath;

    UPROPERTY(EditAnywhere, Category = "Recording")
    float RecordingInterval = 0.033f;

    void StartRecording();
    void StopRecording();
    void ToggleRecording();
    bool IsRecording() const { return bIsRecording; }

    void CreateActorDirectories(const FString& ActorName, TSet<ESensorType>&& SensorTypes);

    bool RecordState = false;

    // Sensor registry for direct access from VCCHUD
    FSensorRegistry SensorRegistry;

private:
    bool bIsRecording = false;
    FTimerHandle RecordingTimerHandle;
    TUniquePtr<FAsyncFileWriter> FileWriter;

    void CreateRecordingDirectoryStructure();
    void TickRecording();
    void CollectSensorDataConcurrently();
    void CollectSensorDataMRT(AActor* Actor, const TArray<ISensorDataProvider*>& Sensors);
    void AddMRTRenderPass(FRDGBuilder& GraphBuilder, AActor* Actor,
        const TArray<ISensorDataProvider*>& Sensors,
        FRDGTextureRef RGBTexture, FRDGTextureRef DepthTexture,
        FRDGTextureRef NormalTexture, FRDGTextureRef SegmentTexture);
    void RenderSceneToMRT(FRHICommandList& RHICmdList, AActor* Actor,
        const TArray<ISensorDataProvider*>& Sensors,
        FRDGTextureRef RGBTexture, FRDGTextureRef DepthTexture,
        FRDGTextureRef NormalTexture, FRDGTextureRef SegmentTexture);
    void ExtractMRTData(FRDGBuilder& GraphBuilder, AActor* Actor,
        const TArray<ISensorDataProvider*>& Sensors,
        FRDGTextureRef RGBTexture, FRDGTextureRef DepthTexture,
        FRDGTextureRef NormalTexture, FRDGTextureRef SegmentTexture);
    void ProcessSensorPixelData(ISensorDataProvider* Sensor, FIntPoint Resolution,
        double Timestamp, TArray<FColor>&& PixelData);
    void ProcessDepthSensorData(ISensorDataProvider* Sensor, FIntPoint Resolution,
        double Timestamp, TArray<FFloat16Color>&& DepthData);
    void CollectSensorDataIndividual(const TArray<ISensorDataProvider*>& Sensors);
    void InitializeAsyncWriter();
    void ShutdownAsyncWriter();
};