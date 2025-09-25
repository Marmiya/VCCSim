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
#include "RenderGraphDefinitions.h"
#include "ISensorDataProvider.generated.h"


struct VCCSIM_API FSensorDataPacket
{
    ESensorType Type;
    int32 SensorIndex;
    AActor* OwnerActor;
    TSharedPtr<FSensorData> Data;
    bool bValid;

    FSensorDataPacket()
        : Type(ESensorType::RGBDCamera)
        , SensorIndex(0)
        , OwnerActor(nullptr)
        , Data(nullptr)
        , bValid(false)
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
    virtual ESensorType GetSensorType() const = 0;
    virtual AActor* GetOwnerActor() const = 0;
    virtual FIntPoint GetResolution() const = 0;
    virtual UTextureRenderTarget2D* GetRenderTarget() const PURE_VIRTUAL(UCameraBaseComponent::GetRenderTarget, return nullptr;);
};