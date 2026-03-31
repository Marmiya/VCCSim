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

class UPrimitiveComponent;

/**
 * Generates complex camera paths for scene capture.
 */
class VCCSIMEDITOR_API FPathGenerator
{
public:
    /** Parameters for generating a conformal orbit path. */
    struct FConformalOrbitParams
    {
        FBox TargetBounds;
        TArray<UPrimitiveComponent*> TargetPrimitives;
        UWorld* World;

        float Margin = 500.0f;
        float StartHeight = 200.0f;
        float CameraHFOV = 90.0f;
        FIntPoint CameraResolution = FIntPoint(1920, 1080);
        float HOverlap = 0.6f;
        float VOverlap = 0.6f;

        bool bIncludeNadir = true;
        float NadirAltitude = 500.0f;
        float NadirTiltAngle = 45.0f;
    };

    /** The output of a path generation operation. */
    struct FGeneratedPath
    {
        TArray<FVector> Positions;
        TArray<FRotator> Rotations;
    };

    /** Delegate called upon completion of asynchronous path generation. */
    DECLARE_DELEGATE_OneParam(FOnPathGenerated, const FGeneratedPath&);

    /**
     * Asynchronously generates a conformal orbit path around the specified target.
     * Performs ray-casting on the game thread, then heavy geometry processing on a background thread,
     * and finally calls the completion delegate back on the game thread.
     *
     * @param Params The parameters for path generation.
     * @param OnComplete The delegate to call when path generation is finished.
     */
    void GenerateConformalOrbit(const FConformalOrbitParams& Params, FOnPathGenerated OnComplete);
};
