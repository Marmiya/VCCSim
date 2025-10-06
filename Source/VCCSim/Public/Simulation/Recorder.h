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
#include "Async/Async.h"
#include "RHICommandList.h"
#include "RenderGraphDefinitions.h"
#include "Utils/AsyncFileWriter.h"
#include "Recorder.generated.h"

struct FCameraViewGroup
{
    TMap<UCameraBaseComponent*, bool> Cameras;
    int32 ViewIndex;
    AActor* OwnerActor;

    FCameraViewGroup()
        : ViewIndex(0)
        , OwnerActor(nullptr)
    {}
};

struct FRDGViewResources
{
    FRDGTextureRef OutputTexture;
    TSharedPtr<FRHIGPUTextureReadback> Readback;
    double CaptureTimestamp;

    FRDGViewResources()
        : OutputTexture(nullptr)
        , Readback(nullptr)
        , CaptureTimestamp(0.0)
    {}
};

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
    float RecordingInterval = 1.f/120.f;
    bool RecordState = false;
    FSensorRegistry SensorRegistry;

    void StartRecording();
    void StopRecording();
    void ToggleRecording();
    bool IsRecording() const { return bIsRecording; }

    void CreateActorDirectories(const FString& ActorName, TSet<ESensorType>&& SensorTypes);

private:
    bool bIsRecording = false;
    FTimerHandle RecordingTimerHandle;
    TUniquePtr<FAsyncFileWriter> FileWriter;
    UPROPERTY()
    TMap<USensorBaseComponent*, double> LastSensorCaptureTimes;
    UPROPERTY()
    TMap<USensorBaseComponent*, double> SensorIntervals;
    UPROPERTY()
    TSet<USensorBaseComponent*> SensorsToReadThisFrame;

    TArray<FCameraViewGroup> CameraViewGroups;
    TMap<int32, FRDGViewResources> ViewResourcesMap;

    static constexpr float PositionThreshold = 5.0f;
    static constexpr float RotationThreshold = 2.0f;

    void TickRecording();
    void CollectSensorData();
    void ProcessCameraResult(FRDGBuilder& GraphBuilder, USensorBaseComponent* Sensor);
    void ProcessSensorResults(FRDGBuilder& GraphBuilder, const TArray<USensorBaseComponent*>& Sensors);
    bool ShouldCaptureSensor(USensorBaseComponent* Sensor, double CurrentTime);

    void SetupSensorProperties();
    void GroupCamerasByPose();
    bool ArePosesSimilar(const UCameraBaseComponent* CamA, const UCameraBaseComponent* CamB) const;
    
    void RenderViewGroupsRDG();

    void InitializeAsyncWriter();
    void ShutdownAsyncWriter();
};