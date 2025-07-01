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

#include "Sensors/LidarSensor.h"
#include "DataType/PointCloud.h"
#include "Simulation/Recorder.h"
#include "DrawDebugHelpers.h"
#include "Async/Async.h"
#include "Misc/FileHelper.h"
#include "Engine/World.h"
#include "Async/ParallelFor.h"
#include "HAL/CriticalSection.h"


ULidarComponent::ULidarComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
	
    MeshHolder = nullptr;

	QueryParams.bTraceComplex = true;
	QueryParams.bReturnPhysicalMaterial = false;
	QueryParams.bReturnFaceIndex = false;
}

void ULidarComponent::RConfigure(
	const FLiDarConfig& Config, ARecorder* Recorder)
{
	NumPoints = Config.NumPoints;
	NumRays = Config.NumRays;
	ScannerRangeInner = Config.ScannerRangeInner;
	ScannerRangeOuter = Config.ScannerRangeOuter;
	ScannerAngleUp = Config.ScannerAngleUp;
	ScannerAngleDown = Config.ScannerAngleDown;
	bVisualizePoints = Config.bVisualizePoints;
	
	if (Config.RecordInterval > 0)
	{
		ParentActor = GetOwner();
		RecorderPtr = Recorder;
		RecordInterval = Config.RecordInterval;
		RecordState = Recorder->RecordState;
		Recorder->OnRecordStateChanged.AddDynamic(this,
			&ULidarComponent::SetRecordState);
		SetComponentTickEnabled(true);
		bRecorded = true;
	}
	else
	{
		SetComponentTickEnabled(false);
	}
}

void ULidarComponent::BeginPlay()
{
    Super::BeginPlay();

	// Enable collision for physics simulation
	SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);

	// Set collision responses to ignore all channels
	SetCollisionResponseToAllChannels(ECR_Ignore);

	// Enable physics simulation
	SetSimulatePhysics(true);
}

void ULidarComponent::FirstCall()
{
	InitSensor();

	LastLocation = GetComponentLocation();
	LastRotation = GetComponentRotation();
	UpdateCachedPoints(LastLocation, LastRotation);
}

void ULidarComponent::OnComponentCreated()
{
    Super::OnComponentCreated();
	InitSensor();
}

void ULidarComponent::TickComponent(
	float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);


	if (bRecorded && RecordState)
	{
		TimeSinceLastCapture += DeltaTime;
		if (TimeSinceLastCapture >= RecordInterval)
		{
			FLidarData LidarData;
			LidarData.Timestamp = FPlatformTime::Seconds();
			LidarData.Data = PerformLineTraces(nullptr);
			TimeSinceLastCapture = 0.0f;
			if (RecorderPtr)
			{
				RecorderPtr->SubmitLidarData(ParentActor, MoveTemp(LidarData));
			}
		}
	}
}

TArray<FVector3f> ULidarComponent::PerformLineTraces(FVCCSimOdom* Odom)
{
    if (!GetWorld())
    {
	    UE_LOG(LogTemp, Error, TEXT("No world found!"));
    	return{};
    }

	const FVector ComponentLocation = GetComponentLocation();
    const FRotator ComponentRotation = GetComponentRotation();

	if (Odom)
	{
		Odom->Location = ComponentLocation;
		Odom->Rotation = ComponentRotation;
		Odom->LinearVelocity = GetPhysicsLinearVelocity();
		Odom->AngularVelocity = GetPhysicsAngularVelocityInDegrees();
	}

	// Check if we need to update cached points
	if (ShouldUpdateCache(ComponentLocation, ComponentRotation))
	{
		UpdateCachedPoints(ComponentLocation, ComponentRotation);
		LastLocation = ComponentLocation;
		LastRotation = ComponentRotation;
	}

	// Perform chunked parallel processing
	ParallelFor(NumChunks, [&](int32 ChunkIndex)
	{
		ProcessChunk(ChunkIndex);
	});

	// Collect valid points
	TArray<FVector3f> ValidPoints;
	
	ValidPoints.Reserve(ActualNumPoints);
	FCriticalSection CriticalSection;  // For thread-safe array access

	// Process chunks in parallel
	ParallelFor(NumChunks, [&](int32 ChunkIndex)
	{
		// Create local array for this chunk's valid points
		TArray<FVector3f> ChunkValidPoints;
		ChunkValidPoints.Reserve(ChunkSize);

		const int32 StartIdx = ChunkStartIndices[ChunkIndex];
		const int32 EndIdx = ChunkEndIndices[ChunkIndex];

		// Collect hit points for this chunk
		for (int32 Index = StartIdx; Index < EndIdx; ++Index)
		{
			if (PointPool[Index].bHit)
			{
				ChunkValidPoints.Add({
					static_cast<float>(PointPool[Index].Position.X),
					static_cast<float>(PointPool[Index].Position.Y),
					static_cast<float>(PointPool[Index].Position.Z)
				});
			}
		}

		// Add chunk results to main array
		if (ChunkValidPoints.Num() > 0)
		{
			FScopeLock Lock(&CriticalSection);
			ValidPoints.Append(ChunkValidPoints);
		}
	});
	
	return ValidPoints;
}

void ULidarComponent::VisualizePointCloud()
{
	if (!GetWorld())
	{
		UE_LOG(LogTemp, Warning,
			TEXT("No world found!, cannot visualize point cloud"));
	}

	if (!MeshHolder)
	{
		UE_LOG(LogTemp, Warning,
			TEXT("MeshHolder not set, cannot visualize point cloud"));
	}
	else
	{
		MeshHolder->ClearAndAddNewInstances(GetHitTransforms());
	}
}

void ULidarComponent::InitSensor()
{
    LocalStartPoints.Empty();
	LocalEndPoints.Empty();

	const int32 PointsPerLine = NumPoints / NumRays;
	const double LayerAngleStep = 360.0 / PointsPerLine;

	// Parameters to control density distribution
	const double DensityPower = 1.0;  // Higher values make distribution more extreme
	const double HorizonBias = 0.1;   // Higher values concentrate more rays near horizon

	for (int32 i = 0; i < NumRays; i++)
	{
	    // Map i to [0, PI] for base distribution
	    double theta = static_cast<double>(i) / (NumRays - 1) * PI;
	    
	    double sinValue = FMath::Sin(theta - PI/2);
	    double t = (1.0 + sinValue) / 2.0;
	    
	    // Apply power function and horizon bias
	    t = FMath::Pow(t, DensityPower);
	    t = FMath::Lerp(t, 1.0 - t, HorizonBias);
	    
	    // Map t to vertical angle range with non-linear distribution
	    double CurrentLineAngle = FMath::Lerp(ScannerAngleUp, -ScannerAngleDown, t);
	    
	    // Ground coverage compensation
	    // Adjust range based on angle to maintain more even ground coverage
	    double CurrentLineRad = FMath::DegreesToRadians(CurrentLineAngle);
	    double cosAngle = FMath::Cos(CurrentLineRad);
	    
	    // Adjust ranges to compensate for ground projection
	    double adjustedInnerRange = ScannerRangeInner / FMath::Max(cosAngle, 0.1);
	    double adjustedOuterRange = ScannerRangeOuter / FMath::Max(cosAngle, 0.1);
	    
	    // Calculate vertical offsets
	    double InnerZOffset = adjustedInnerRange * FMath::Tan(CurrentLineRad);
	    double OuterZOffset = adjustedOuterRange * FMath::Tan(CurrentLineRad);

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
	            Direction.X * adjustedInnerRange,
	            Direction.Y * adjustedInnerRange,
	            InnerZOffset
	        );
	        
	        FVector CurrentEnd(
	            Direction.X * adjustedOuterRange,
	            Direction.Y * adjustedOuterRange,
	            OuterZOffset
	        );

	        LocalStartPoints.Add(CurrentStart);
	        LocalEndPoints.Add(CurrentEnd);

	        CurrentLayerAngle += LayerAngleStep;
	    }
	}

    // Update total rays
    ActualNumPoints = LocalStartPoints.Num();
	
	// Pre-allocate arrays
	CachedStartPoints.Reserve(ActualNumPoints);
	CachedEndPoints.Reserve(ActualNumPoints);
	CachedStartPoints.SetNum(ActualNumPoints);
	CachedEndPoints.SetNum(ActualNumPoints);
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

TArray<FVector3f> ULidarComponent::GetPointCloudData()
{
	const auto ans = PerformLineTraces();
	
	if (bVisualizePoints)
	{
		// Start a new task to visualize the point cloud
		AsyncTask(ENamedThreads::GameThread, [this]()
			{
				VisualizePointCloud();
			});
	}

	return ans;
}

TPair<TArray<FVector3f>, FVCCSimOdom> ULidarComponent::GetPointCloudDataAndOdom()
{
	FVCCSimOdom Pose;
	const auto ans = PerformLineTraces(&Pose);

	if (bVisualizePoints)
	{
		// Start a new task to visualize the point cloud
		AsyncTask(ENamedThreads::GameThread, [this]()
			{
				VisualizePointCloud();
			});
	}
	
	return {ans, Pose};
}

void ULidarComponent::ProcessChunk(int32 ChunkIndex)
{
	check(ChunkIndex >= 0 && ChunkIndex < NumChunks);
    
	// Get chunk boundaries
	const int32 StartIndex = ChunkStartIndices[ChunkIndex];
	const int32 EndIndex = ChunkEndIndices[ChunkIndex];
    
	// Calculate cache-line aligned processing groups
	constexpr int32 PointsPerCacheLine = CACHE_LINE_SIZE / sizeof(FVector);
    
	// Process points in this chunk with cache-line alignment
	for (int32 Index = StartIndex; Index < EndIndex; Index += PointsPerCacheLine)
	{
		// Process a cache-line sized group of points
		const int32 GroupEnd = FMath::Min(Index + PointsPerCacheLine, EndIndex);
        
		for (int32 GroupIndex = Index; GroupIndex < GroupEnd; ++GroupIndex)
		{
			FHitResult HitResult;
			const bool bHit = GetWorld()->LineTraceSingleByChannel(
				HitResult,
				CachedStartPoints[GroupIndex],
				CachedEndPoints[GroupIndex],
				ECC_Visibility,
				QueryParams
			);

			FLiDARPoint& Point = PointPool[GroupIndex];
			Point.bHit = bHit;
			Point.Position = bHit ? HitResult.Location : CachedEndPoints[GroupIndex];
		}
	}
}

void ULidarComponent::UpdateCachedPoints(const FVector& NewLocation, const FRotator& NewRotation)
{
	for (int32 i = 0; i < ActualNumPoints; ++i)
	{
		CachedStartPoints[i] = NewRotation.RotateVector(LocalStartPoints[i]) + NewLocation;
		CachedEndPoints[i] = NewRotation.RotateVector(LocalEndPoints[i]) + NewLocation;
	}
}

bool ULidarComponent::ShouldUpdateCache(const FVector& NewLocation,
	const FRotator& NewRotation) const
{
	// Check if movement exceeds threshold
	float LocationDiff = FVector::Distance(LastLocation, NewLocation);
	if (LocationDiff > UpdateThresholdDistance)
	{
		return true;
	}

	// Check if rotation exceeds threshold
	FRotator RotationDiff = (NewRotation - LastRotation).GetNormalized();
	if (FMath::Abs(RotationDiff.Yaw) > UpdateThresholdAngle ||
		FMath::Abs(RotationDiff.Pitch) > UpdateThresholdAngle ||
		FMath::Abs(RotationDiff.Roll) > UpdateThresholdAngle)
	{
		return true;
	}

	return false;
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