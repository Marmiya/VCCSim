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
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program. If not, see <https://www.gnu.org/licenses/>.
*/
#pragma once

#include "CoreMinimal.h"
#include "Engine/TimerHandle.h"

class UWorld;

// Delegate to report progress
DECLARE_DELEGATE_ThreeParams(FOnNanobananaProgress, const FString& /* Status */, int32 /* Processed */, int32 /* Total */);
// Delegate for completion
DECLARE_DELEGATE_OneParam(FOnNanobananaComplete, const FString& /* FinalStatus */);

/**
 * Manages the Nanobanana projection process, which involves:
 * - Reading pose and segmentation image data.
 * - Generating and casting rays into the scene.
 * - Voting for material classes based on ray hits.
 * - Writing out the final class assignments for each actor.
 */
class VCCSIMEDITOR_API FNanobananaManager : public TSharedFromThis<FNanobananaManager>
{
public:
    explicit FNanobananaManager(UWorld* InWorld);
    ~FNanobananaManager();

    struct FProjectionParams
    {
        FString ResultDir;
        FString PosesFile;
        FString ManifestFile;
        float   HFOV = 90.f;
        int32   ImageWidth = 1920;
        int32   ImageHeight = 1080;
        int32   RaysPerClass = 80;
    };

    /**
     * Starts the asynchronous projection process.
     * @param InParams The input parameters for the projection.
     * @param InOnProgress Delegate called periodically with progress updates.
     * @param InOnComplete Delegate called when the entire process is finished.
     * @return True if the process started successfully, false otherwise.
     */
    bool RunProjection(
        const FProjectionParams& InParams,
        FOnNanobananaProgress InOnProgress,
        FOnNanobananaComplete InOnComplete
    );

    bool IsInProgress() const { return bIsInProgress; }

private:
    struct FNanobananaRay
    {
        FVector Origin;
        FVector Direction;
        FString ClassName;
    };

    // Main logic steps
    void StartAsyncDataLoading();
    void TickRayCasting();
    void FinalizeProjection();

    // World context for ray casting
    TWeakObjectPtr<UWorld> WorldPtr;
    
    // State variables
    bool bIsInProgress = false;
    FProjectionParams Params;
    FTimerHandle TimerHandle;

    // Data for the projection process
    TArray<FNanobananaRay>              PendingRays;
    int32                               ProcessedRayCount = 0;
    int32                               TotalRayCount = 0;
    TMap<FString, TMap<FString, int32>> Votes;
    TArray<FString>                     ManifestActors;
    FString                             OutputDir;

    // Delegates for callbacks
    FOnNanobananaProgress OnProgress;
    FOnNanobananaComplete OnComplete;
};
