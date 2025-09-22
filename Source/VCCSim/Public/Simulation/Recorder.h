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
#include "Recorder.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FRecordStateChanged, bool, RecordState);

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

    UPROPERTY(BlueprintAssignable)
    FRecordStateChanged OnRecordStateChanged;

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
    void InitializeAsyncWriter();
    void ShutdownAsyncWriter();
};