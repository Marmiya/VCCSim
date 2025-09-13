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

// grpc
struct FCompactVertex {
	float X, Y, Z;
};

struct FMeshHeader {
	uint32 Magic;
	uint32 Version;
	uint32 VertexCount;
	uint32 IndexCount;
	uint32 Flags;
};

// simulation
struct FMeshInfo
{
	int32 MeshID;
	FString MeshName;
	UStaticMesh* Mesh;
	FTransform Transform;
	FBoxSphereBounds Bounds;
	int32 NumTriangles;
	int32 NumVertices;
	TArray<FVector> VertexPositions;
	TArray<int32> Indices;
	bool bIsVisible;
};

struct FUnifiedGridCell
{
	// Coverage data
	int32 TotalPoints;
	int32 VisiblePoints;
	float Coverage;
    
	// Complexity data
	float CurvatureScore;
	float EdgeDensityScore;
	float AngleVariationScore;
	float ComplexityScore; 
    
	FUnifiedGridCell() 
			: TotalPoints(0)
			, VisiblePoints(0)
			, Coverage(0.0f)
			, CurvatureScore(0.0f)
			, EdgeDensityScore(0.0f)
			, AngleVariationScore(0.0f)
			, ComplexityScore(0.0f)
	{}
};