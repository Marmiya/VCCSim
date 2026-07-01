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

class UWorld;

/**
 * Renders the scene's sky (SkyAtmosphere + VolumetricCloud, sun disk excluded, fog disabled) into an
 * equirectangular linear-HDR panorama and saves it as an OpenEXR. Used as the renderer's split-sum IBL
 * environment light, captured once per lighting window alongside lighting.json.
 *
 * The panorama is authored in the RH-world convention recorded in lighting.json's
 * sky_equirect_convention field (+Z up; v=0 -> +Z zenith; u: yaw=0 -> +Y, increasing toward +X).
 */
class VCCSIMEDITOR_API FSkyHDRICapture
{
public:
    static bool CaptureSkyEquirect(
        UWorld* World,
        const FString& OutExrPath,
        int32 EquirectWidth = 2048,
        int32 FaceSize = 1024);
};
