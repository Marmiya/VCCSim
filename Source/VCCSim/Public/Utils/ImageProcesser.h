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
#include "Async/AsyncWork.h"
#include "Sensors/DepthCamera.h"

class FAsyncImageSaveTask : public FNonAbandonableTask
{
public:
    FAsyncImageSaveTask(
       const TArray<FColor>& InPixels,
       FIntPoint InSize, const FString& InFilePath)
       : Pixels(InPixels)
       , Size(InSize)
       , FilePath(InFilePath)
    {}

    void DoWork();

    FORCEINLINE TStatId GetStatId() const
    {
       RETURN_QUICK_DECLARE_CYCLE_STAT(FAsyncImageSaveTask,
          STATGROUP_ThreadPoolAsyncTasks);
    }

private:
    TArray<FColor> Pixels;
    FIntPoint Size;
    FString FilePath;
};

class FAsyncDepth16SaveTask : public FNonAbandonableTask
{
public:
    FAsyncDepth16SaveTask(
        const TArray<FFloat16Color>& InDepthPixels,
        FIntPoint InSize, 
        const FString& InFilePath,
        float InDepthScale = 1.0f)
        : DepthPixels(InDepthPixels)
        , Size(InSize)
        , FilePath(InFilePath)
        , DepthScale(InDepthScale)
    {}

    void DoWork();

    FORCEINLINE TStatId GetStatId() const
    {
        RETURN_QUICK_DECLARE_CYCLE_STAT(FAsyncDepth16SaveTask,
           STATGROUP_ThreadPoolAsyncTasks);
    }

private:
    TArray<FFloat16Color> DepthPixels;
    FIntPoint Size;
    FString FilePath;
    float DepthScale;
};

class FAsyncPLYSaveTask : public FNonAbandonableTask
{
public:
    FAsyncPLYSaveTask(const TArray<FDCPoint>& InPointCloud, const FString& InFilePath)
        : PointCloud(InPointCloud)
        , FilePath(InFilePath)
    {
    }

    void DoWork();
    
    FORCEINLINE TStatId GetStatId() const
    {
        RETURN_QUICK_DECLARE_CYCLE_STAT(FAsyncPLYSaveTask,
            STATGROUP_ThreadPoolAsyncTasks);
    }

private:
    TArray<FDCPoint> PointCloud;
    FString FilePath;
};

class FAsyncNormalEXRSaveTask : public FNonAbandonableTask
{
public:
    FAsyncNormalEXRSaveTask(
        const TArray<FLinearColor>& InNormalPixels,
        FIntPoint InSize, 
        const FString& InFilePath)
        : NormalPixels(InNormalPixels)
        , Size(InSize)
        , FilePath(InFilePath)
    {}

    void DoWork();

    FORCEINLINE TStatId GetStatId() const
    {
        RETURN_QUICK_DECLARE_CYCLE_STAT(FAsyncNormalEXRSaveTask,
           STATGROUP_ThreadPoolAsyncTasks);
    }

private:
    TArray<FLinearColor> NormalPixels;
    FIntPoint Size;
    FString FilePath;
};