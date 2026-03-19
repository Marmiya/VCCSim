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

class VCCSIMEDITOR_API FVCCSimSunPositionHelper
{
public:
    struct FSunParams
    {
        float Latitude       = 22.55f;
        float Longitude      = 114.05f;
        float TimeZone       = 8.0f;
        bool  bDaylightSaving = false;
        int32 Year           = 2025;
        int32 Month          = 6;
        int32 Day            = 21;
        int32 Hour           = 10;
        int32 Minute         = 0;
        int32 Second         = 0;
    };

    /**
     * Compute sun elevation and azimuth for the given geographic location and UTC time.
     * Uses UE's built-in SunPosition plugin ephemeris algorithm.
     * @param Params      Geographic and time inputs
     * @param OutElevation Atmospheric-corrected sun elevation in degrees (0 = horizon, 90 = zenith)
     * @param OutAzimuth  Sun azimuth in degrees (0 = North, clockwise)
     * @return True if the sun is above the horizon (Elevation > 0)
     */
    static bool Calculate(const FSunParams& Params, float& OutElevation, float& OutAzimuth);
};
