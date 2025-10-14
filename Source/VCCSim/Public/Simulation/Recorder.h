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
#include "ActorRegistry.h"
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

    // Recorder will attempt to record at this interval (in seconds).
    // Sensors and actors have their own intervals!
    UPROPERTY(EditAnywhere, Category = "Recording")
    float RecorderInterval = 1.f/120.f;
    bool RecordState = false;
    FSensorRegistry SensorRegistry;
    FActorRegistry ActorRegistry;

    void StartRecording();
    void StopRecording();
    void ToggleRecording();
    bool IsRecording() const { return bIsRecording; }

    void CreateActorDirectories(const FString& ActorName, TSet<ESensorType>&& SensorTypes);

    void SubmitCameraRequest(
        UCameraBaseComponent* Camera,
        TFunction<void(const FSensorDataPacket&)> OnSuccess,
        TFunction<void(const FString&)> OnError);

private:
    bool bIsRecording = false;
    FTimerHandle RecordingTimerHandle;
    TUniquePtr<FAsyncFileWriter> FileWriter;
    UPROPERTY()
    TMap<USensorBaseComponent*, double> LastSensorCaptureTimes;
    UPROPERTY()
    TMap<USensorBaseComponent*, double> SensorIntervals;

    TArray<FCameraViewGroup> CameraViewGroups;

    struct FPendingReadback
    {
        TSharedPtr<FRHIGPUTextureReadback> Readback;
        double CaptureTimestamp;
        TWeakObjectPtr<UCameraBaseComponent> Camera;
    };
    TQueue<FPendingReadback, EQueueMode::Mpsc> PendingReadbacks;

    struct FRPCRequestCallback
    {
        TFunction<void(const FSensorDataPacket&)> OnSuccess;
        TFunction<void(const FString&)> OnError;
        double RequestTime;
        double TimeoutDuration = 5.0;

        FRPCRequestCallback() : RequestTime(0.0) {}
    };

    FCriticalSection RPCRequestLock;
    TMap<TWeakObjectPtr<USensorBaseComponent>, TArray<FRPCRequestCallback>> PendingRPCCallbacks;
    TSet<TWeakObjectPtr<USensorBaseComponent>> ForceCaptureThisFrame;

    void CheckRPCTimeouts(double CurrentTime);
    void TriggerRPCCallbacks(UCameraBaseComponent* Camera, const FSensorDataPacket& Packet);

    class FReadbackWorker : public FRunnable
    {
    public:
        FReadbackWorker(ARecorder* InOwner) : Owner(InOwner) {}
        virtual uint32 Run() override;
        virtual void Stop() override { bShouldStop.store(true); }
    private:
        ARecorder* Owner;
        std::atomic<bool> bShouldStop{false};
    };
    TUniquePtr<FReadbackWorker> ReadbackWorker;
    TUniquePtr<FRunnableThread> ReadbackThread;
    std::atomic<bool> bReadbackWorkerShouldStop{false};

    static constexpr float PositionThreshold = 5.0f;
    static constexpr float RotationThreshold = 2.0f;

    void TickRecording();
    void CollectData();

    void ProcessPendingReadback(const FPendingReadback& PendingData);
    void SampleLiDARData(USensorBaseComponent* Sensor);

    void RenderViewGroupsRDG(const double& CaptureTime);
    void RecordActorPoses(double&& CurrentTime);

    bool ShouldCaptureSensor(USensorBaseComponent* Sensor, double CurrentTime);
    void SetupSensorProperties();
    void GroupCamerasByPose();
    bool ArePosesSimilar(const UCameraBaseComponent* CamA, const UCameraBaseComponent* CamB) const;

    void InitializeAsyncWriter();
    void ShutdownAsyncWriter();
    void InitializeReadbackWorker();
    void ShutdownReadbackWorker();
};