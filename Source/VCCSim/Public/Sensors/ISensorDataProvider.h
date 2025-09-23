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
#include "SensorBase.h"
#include "Async/Async.h"
#include "DataStructures/RecordData.h"
#include "UObject/Interface.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphDefinitions.h"
#include "ISensorDataProvider.generated.h"

class ISensorDataProvider;

struct VCCSIM_API FSensorViewInfo
{
    ESensorType SensorType;
    int32 MRTSlot;
    FMatrix ViewMatrix;
    FMatrix ProjectionMatrix;
    ESceneCaptureSource CaptureSource;
    FIntPoint Resolution;
    ISensorDataProvider* Provider;

    FSensorViewInfo()
        : SensorType(ESensorType::RGBCamera)
        , MRTSlot(0)
        , ViewMatrix(FMatrix::Identity)
        , ProjectionMatrix(FMatrix::Identity)
        , CaptureSource(ESceneCaptureSource::SCS_FinalColorLDR)
        , Resolution(FIntPoint(512, 512))
        , Provider(nullptr)
    {
    }
};

struct VCCSIM_API FSensorDataPacket
{
    ESensorType Type;
    int32 SensorIndex;
    AActor* OwnerActor;
    double Timestamp;
    TSharedPtr<FSensorData> Data;
    bool bValid;
    bool bFromRDGBatch;

    FSensorDataPacket()
        : Type(ESensorType::RGBCamera)
        , SensorIndex(0)
        , OwnerActor(nullptr)
        , Timestamp(0.0)
        , Data(nullptr)
        , bValid(false)
        , bFromRDGBatch(false)
    {
    }
};

UINTERFACE(MinimalAPI)
class USensorDataProvider : public UInterface
{
    GENERATED_BODY()
};

class VCCSIM_API ISensorDataProvider
{
    GENERATED_BODY()

public:
    virtual TFuture<FSensorDataPacket> CaptureDataAsync() = 0;
    virtual ESensorType GetSensorType() const = 0;
    virtual AActor* GetOwnerActor() const = 0;

    virtual void ContributeToRDGPass(FSensorViewInfo& OutViewInfo) = 0;
    virtual int32 GetMRTSlot() const = 0;
};