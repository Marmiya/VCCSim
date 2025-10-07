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

DEFINE_LOG_CATEGORY_STATIC(LogLidarSensor, Log, All);

#include "Sensors/LidarSensor.h"
#include "Utils/InsMeshHolder.h"
#include "DrawDebugHelpers.h"
#include "Async/Async.h"
#include "Misc/FileHelper.h"
#include "Engine/World.h"
#include "Async/ParallelFor.h"
#include "HAL/CriticalSection.h"

void ULidarComponent::OnComponentCreated()
{
	Super::OnComponentCreated();
	InitSensor();
}

ULidarComponent::ULidarComponent()
{
    MeshHolder = nullptr;
	QueryParams.bTraceComplex = true;
	QueryParams.bReturnPhysicalMaterial = false;
	QueryParams.bReturnFaceIndex = false;
}

void ULidarComponent::Configure(const FSensorConfig& Config)
{
	if (!bBPConfigured)
	{
		const auto& LidarConfig = static_cast<const FLiDARConfig&>(Config);
		NumPoints = LidarConfig.NumPoints;
		NumRays = LidarConfig.NumRays;
		ScannerRangeInner = LidarConfig.ScannerRangeInner;
		ScannerRangeOuter = LidarConfig.ScannerRangeOuter;
		ScannerAngleUp = LidarConfig.ScannerAngleUp;
		ScannerAngleDown = LidarConfig.ScannerAngleDown;
		bVisualizePoints = LidarConfig.bVisualizePoints;

		InitSensor();
	}

	if (MeshHolder)
	{
		OnPointCloudUpdated.AddLambda([this](const TArray<FVector>& Points)
		{
			TArray<FTransform> HitTransforms;
			HitTransforms.Reserve(Points.Num());
			for (const FVector& Point : Points)
			{
				HitTransforms.Add(FTransform(Point));
			}
			MeshHolder->ClearAndAddNewInstances(HitTransforms);
		});
	}
}

void ULidarComponent::BeginPlay()
{
    Super::BeginPlay();
}

void ULidarComponent::InitSensor()
{
    LocalStartPoints.Empty();
	LocalEndPoints.Empty();

	const int32 PointsPerLine = NumPoints / NumRays;
	const double LayerAngleStep = 360.0 / PointsPerLine;

	for (int32 i = 0; i < NumRays; i++)
	{
	    // Map i to [0, PI] for base distribution
	    const double Theta = static_cast<double>(i) / (NumRays - 1) * PI;
	    
	    double SinValue = FMath::Sin(Theta - PI/2);
	    double t = (1.0 + SinValue) / 2.0;
	    
	    // Apply power function and horizon bias
	    t = FMath::Pow(t, 1.0); // Higher values make distribution more extreme
	    t = FMath::Lerp(t, 1.0 - t, 0.1); // Higher values concentrate more rays near horizon
	    
	    // Map t to vertical angle range with non-linear distribution
	    double CurrentLineAngle = FMath::Lerp(ScannerAngleUp, -ScannerAngleDown, t);
	    
	    // Ground coverage compensation
	    // Adjust range based on angle to maintain more even ground coverage
	    const double CurrentLineRad = FMath::DegreesToRadians(CurrentLineAngle);
	    const double CosAngle = FMath::Cos(CurrentLineRad);
	    
	    // Adjust ranges to compensate for ground projection
	    const double AdjustedInnerRange = ScannerRangeInner / FMath::Max(CosAngle, 0.1);
	    const double AdjustedOuterRange = ScannerRangeOuter / FMath::Max(CosAngle, 0.1);
	    
	    // Calculate vertical offsets
	    const double InnerZOffset = AdjustedInnerRange * FMath::Tan(CurrentLineRad);
	    const double OuterZOffset = AdjustedOuterRange * FMath::Tan(CurrentLineRad);

	    // Generate points for this vertical angle
	    double CurrentLayerAngle = 0.0;
	    for (int32 j = 0; j < PointsPerLine; j++)
	    {
	        double LayerRad = FMath::DegreesToRadians(CurrentLayerAngle);
	        
	        FVector2D Direction(
	            FMath::Cos(LayerRad),
	            FMath::Sin(LayerRad)
	        );

	        FVector CurrentStart(
	            Direction.X * AdjustedInnerRange,
	            Direction.Y * AdjustedInnerRange,
	            InnerZOffset
	        );
	        
	        FVector CurrentEnd(
	            Direction.X * AdjustedOuterRange,
	            Direction.Y * AdjustedOuterRange,
	            OuterZOffset
	        );

	        LocalStartPoints.Add(CurrentStart);
	        LocalEndPoints.Add(CurrentEnd);

	        CurrentLayerAngle += LayerAngleStep;
	    }
	}

    ActualNumPoints = LocalStartPoints.Num();

	PointPool.Empty(ActualNumPoints);
	PointPool.SetNum(ActualNumPoints);

	// Ensure ChunkSize is within valid range
	ChunkSize = FMath::Clamp(ChunkSize, 32, 1024);
    
	// Calculate number of chunks needed
	NumChunks = FMath::DivideAndRoundUp(ActualNumPoints, ChunkSize);
    
	// Pre-calculate chunk boundaries
	ChunkStartIndices.SetNum(NumChunks);
	ChunkEndIndices.SetNum(NumChunks);
    
	for (int32 i = 0; i < NumChunks; ++i)
	{
		ChunkStartIndices[i] = i * ChunkSize;
		ChunkEndIndices[i] = FMath::Min((i + 1) * ChunkSize, ActualNumPoints);
	}
}

TArray<FVector3f> ULidarComponent::PerformLineTraces(FVCCSimOdom* Odom)
{
    if (!GetWorld())
    {
	    UE_LOG(LogLidarSensor, Error, TEXT("No world found!"));
    	return{};
    }

	const FVector ComponentLocation = GetComponentLocation();
    const FRotator ComponentRotation = GetComponentRotation();

	if (Odom)
	{
		Odom->Location = ComponentLocation;
		Odom->Rotation = ComponentRotation;

		if (AActor* Owner = GetOwner())
		{
			if (UPrimitiveComponent* RootPrim = Cast<UPrimitiveComponent>(Owner->GetRootComponent()))
			{
				Odom->LinearVelocity = RootPrim->GetPhysicsLinearVelocity();
				Odom->AngularVelocity = RootPrim->GetPhysicsAngularVelocityInDegrees();
			}
		}
	}

	const FTransform WorldTransform = GetComponentTransform();

	TArray<TArray<FVector3f>> ChunkResults;
	ChunkResults.SetNum(NumChunks);

	ParallelFor(NumChunks, [&](int32 ChunkIndex)
	{
		ProcessChunk(ChunkIndex, WorldTransform);

		TArray<FVector3f>& ChunkPoints = ChunkResults[ChunkIndex];
		ChunkPoints.Reserve(ChunkSize);

		const int32 StartIdx = ChunkStartIndices[ChunkIndex];
		const int32 EndIdx = ChunkEndIndices[ChunkIndex];

		for (int32 Index = StartIdx; Index < EndIdx; ++Index)
		{
			if (PointPool[Index].bHit)
			{
				ChunkPoints.Add(FVector3f(PointPool[Index].Position));
			}
		}
	});

	TArray<FVector3f> ValidPoints;
	ValidPoints.Reserve(ActualNumPoints);
	for (const TArray<FVector3f>& ChunkPoints : ChunkResults)
	{
		ValidPoints.Append(ChunkPoints);
	}

	return ValidPoints;
}

void ULidarComponent::VisualizePointCloud()
{
	if (!GetWorld())
	{
		UE_LOG(LogLidarSensor, Warning,
			TEXT("No world found!, cannot visualize point cloud"));
	}

	if (!MeshHolder)
	{
		UE_LOG(LogLidarSensor, Warning,
			TEXT("MeshHolder not set, cannot visualize point cloud"));
	}
	else
	{
		MeshHolder->ClearAndAddNewInstances(GetHitTransforms());
	}
}

TArray<FVector3f> ULidarComponent::GetPointCloudData()
{
	const auto ans = PerformLineTraces();

	if (bVisualizePoints && OnPointCloudUpdated.IsBound())
	{
		TArray<FVector> Points;
		Points.Reserve(ans.Num());
		for (const FVector3f& Point : ans)
		{
			Points.Add(FVector(Point));
		}
		AsyncTask(ENamedThreads::GameThread, [this, Points = MoveTemp(Points)]()
		{
			OnPointCloudUpdated.Broadcast(Points);
		});
	}

	return ans;
}

TPair<TArray<FVector3f>, FVCCSimOdom> ULidarComponent::GetPointCloudDataAndOdom()
{
	FVCCSimOdom Pose;
	const auto ans = PerformLineTraces(&Pose);

	if (bVisualizePoints && OnPointCloudUpdated.IsBound())
	{
		TArray<FVector> Points;
		Points.Reserve(ans.Num());
		for (const FVector3f& Point : ans)
		{
			Points.Add(FVector(Point));
		}

		AsyncTask(ENamedThreads::GameThread, [this, Points = MoveTemp(Points)]()
		{
			OnPointCloudUpdated.Broadcast(Points);
		});
	}

	return {ans, Pose};
}

void ULidarComponent::ProcessChunk(int32 ChunkIndex, const FTransform& WorldTransform)
{
	check(ChunkIndex >= 0 && ChunkIndex < NumChunks);

	const int32 StartIndex = ChunkStartIndices[ChunkIndex];
	const int32 EndIndex = ChunkEndIndices[ChunkIndex];

	for (int32 Index = StartIndex; Index < EndIndex; ++Index)
	{
		const FVector WorldStart = WorldTransform.TransformPosition(LocalStartPoints[Index]);
		const FVector WorldEnd = WorldTransform.TransformPosition(LocalEndPoints[Index]);

		FHitResult HitResult;
		const bool bHit = GetWorld()->LineTraceSingleByChannel(
			HitResult,
			WorldStart,
			WorldEnd,
			ECC_Visibility,
			QueryParams
		);

		FLiDARPoint& Point = PointPool[Index];
		Point.bHit = bHit;
		Point.Position = bHit ? HitResult.Location : WorldEnd;
	}
}

TArray<FTransform> ULidarComponent::GetHitTransforms() const
{
	TArray<FTransform> HitTransforms;
	HitTransforms.Reserve(ActualNumPoints);
	Algo::TransformIf(
		PointPool,
		HitTransforms,
		[](const FLiDARPoint& Point) { return Point.bHit; },
		[](const FLiDARPoint& Point) { return FTransform(Point.Position); }
	);
	return HitTransforms;
}