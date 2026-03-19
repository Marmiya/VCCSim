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

#include "Utils/VCCSimSunPositionHelper.h"
#include "SunPosition.h"

DEFINE_LOG_CATEGORY_STATIC(LogSunPositionHelper, Log, All);

bool FVCCSimSunPositionHelper::Calculate(const FSunParams& Params, float& OutElevation, float& OutAzimuth)
{
    FSunPositionData SunData;
    USunPositionFunctionLibrary::GetSunPosition(
        Params.Latitude,
        Params.Longitude,
        Params.TimeZone,
        Params.bDaylightSaving,
        Params.Year,
        Params.Month,
        Params.Day,
        Params.Hour,
        Params.Minute,
        Params.Second,
        SunData
    );

    OutElevation = SunData.CorrectedElevation;
    OutAzimuth   = SunData.Azimuth;

    UE_LOG(LogSunPositionHelper, Log,
        TEXT("Sun: Lat=%.2f Lon=%.2f %d-%02d-%02d %02d:%02d UTC%+.1f → Elev=%.2f° Az=%.2f°"),
        Params.Latitude, Params.Longitude,
        Params.Year, Params.Month, Params.Day,
        Params.Hour, Params.Minute,
        Params.TimeZone,
        OutElevation, OutAzimuth);

    return OutElevation > 0.f;
}
