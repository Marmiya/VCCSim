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
#include "DataStructures/Mesh.h"
#include "PathPlanner.generated.h"

UCLASS(BlueprintType, Blueprintable)
class VCCSIM_API UPathPlanner : public UObject
{
	GENERATED_BODY()
	
public:
	static void SemiSphericalPath(
		const TArray<FMeshInfo>& InMesh,
		const float Radius,
		const int32 NumPoses,
		const float& SafeHeight,
		TArray<FVector>& OutPositions,
		TArray<FRotator>& OutRotations
	);
	
	static void VisDebugPath(
		const TArray<FVector>& InPositions,
		const TArray<FRotator>& InRotations,
		const float Duration = 5.0f,
		const FColor& Color = FColor::Red
	);
};