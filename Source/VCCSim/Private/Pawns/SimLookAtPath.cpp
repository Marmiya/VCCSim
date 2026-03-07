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

#include "Pawns/SimLookAtPath.h"
#include "Pawns/FlashPawn.h"
#include "Sensors/RGBCamera.h"
#include "Sensors/DepthCamera.h"
#include "Sensors/SegmentationCamera.h"
#include "Sensors/NormalCamera.h"
#include "Utils/ImageProcesser.h"
#include "Components/SplineComponent.h"
#include "Components/StaticMeshComponent.h"
#include "UObject/ConstructorHelpers.h"
#include "Kismet/GameplayStatics.h"
#include "HAL/FileManager.h"
#include "Async/AsyncWork.h"

DEFINE_LOG_CATEGORY_STATIC(LogSimLookAtPath, Log, All);

AVCCSimLookAtPath::AVCCSimLookAtPath()
{
	PrimaryActorTick.bCanEverTick = false;

	Spline = CreateDefaultSubobject<USplineComponent>(TEXT("Spline"));
	RootComponent = Spline;

	TargetPoint = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("TargetPoint"));
	TargetPoint->SetupAttachment(RootComponent);

	static ConstructorHelpers::FObjectFinder<UStaticMesh> SphereMesh(
		TEXT("/Engine/BasicShapes/Sphere.Sphere"));
	if (SphereMesh.Succeeded())
	{
		TargetPoint->SetStaticMesh(SphereMesh.Object);
		TargetPoint->SetWorldScale3D(FVector(0.3f));
	}
	TargetPoint->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	TargetPoint->SetRelativeLocation(FVector(0.f, 0.f, 300.f));

	SamplingMode = ELookAtSamplingMode::ControlPoints;
	NumDivisions = 50;
	PathLength = 0.f;
	NumSamplePoints = 0;
	SaveDirectoryBase = FPaths::ProjectSavedDir() / TEXT("VCCSimLookAtCaptures");

	JobNum = MakeShared<std::atomic<int32>>(0);
}

void AVCCSimLookAtPath::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);

	DiscoverTraceIgnores();

	PathLength = Spline->GetSplineLength();
	NumSamplePoints = GetNumSamplePoints();
}

void AVCCSimLookAtPath::DiscoverTraceIgnores()
{
	TArray<AActor*> FoundActors;
	UGameplayStatics::GetAllActorsWithTag(GetWorld(), "IgnoreTrace", FoundActors);
	for (AActor* Actor : FoundActors)
	{
		ExcludedInTrace.AddUnique(Actor);
	}
}

void AVCCSimLookAtPath::SnapAllPointsToGround()
{
	for (int32 i = 0; i < Spline->GetNumberOfSplinePoints(); i++)
	{
		FVector Location = Spline->GetLocationAtSplinePoint(i, ESplineCoordinateSpace::World);
		FHitResult HitResult;
		FVector Start = Location + FVector(0, 0, 150);
		FVector End   = Location - FVector(0, 0, 10000);
		FCollisionQueryParams TraceParams;
		TraceParams.AddIgnoredActors(ExcludedInTrace);
		if (GetWorld()->LineTraceSingleByChannel(HitResult, Start, End,
			ECC_Visibility, TraceParams))
		{
			Spline->SetLocationAtSplinePoint(i, HitResult.Location,
				ESplineCoordinateSpace::World);
		}
	}
	Spline->UpdateSpline();
}

void AVCCSimLookAtPath::FlattenAllPointsToSameHeight()
{
	const int32 NumPoints = Spline->GetNumberOfSplinePoints();
	if (NumPoints == 0) return;

	const float TargetZ =
		Spline->GetLocationAtSplinePoint(0, ESplineCoordinateSpace::World).Z;

	for (int32 i = 1; i < NumPoints; i++)
	{
		FVector Location = Spline->GetLocationAtSplinePoint(i, ESplineCoordinateSpace::World);
		Location.Z = TargetZ;
		Spline->SetLocationAtSplinePoint(i, Location, ESplineCoordinateSpace::World);
	}
	Spline->UpdateSpline();
	PathLength = Spline->GetSplineLength();
}

void AVCCSimLookAtPath::GetSamplePoses(
	TArray<FVector>& OutPositions, TArray<FRotator>& OutRotations) const
{
	OutPositions.Empty();
	OutRotations.Empty();

	if (!Spline || !TargetPoint) return;

	const FVector TargetLocation = TargetPoint->GetComponentLocation();

	if (SamplingMode == ELookAtSamplingMode::ControlPoints)
	{
		const int32 NumPoints = Spline->GetNumberOfSplinePoints();
		OutPositions.Reserve(NumPoints);
		OutRotations.Reserve(NumPoints);

		for (int32 i = 0; i < NumPoints; i++)
		{
			const FVector Position =
				Spline->GetLocationAtSplinePoint(i, ESplineCoordinateSpace::World);
			const FVector Direction = (TargetLocation - Position).GetSafeNormal();
			OutPositions.Add(Position);
			OutRotations.Add(Direction.Rotation());
		}
	}
	else
	{
		OutPositions.Reserve(NumDivisions);
		OutRotations.Reserve(NumDivisions);

		const float SplineLen = Spline->GetSplineLength();
		for (int32 i = 0; i < NumDivisions; i++)
		{
			const float Distance = SplineLen * static_cast<float>(i) /
				static_cast<float>(NumDivisions - 1);
			const FVector Position =
				Spline->GetLocationAtDistanceAlongSpline(Distance, ESplineCoordinateSpace::World);
			const FVector Direction = (TargetLocation - Position).GetSafeNormal();
			OutPositions.Add(Position);
			OutRotations.Add(Direction.Rotation());
		}
	}
}

int32 AVCCSimLookAtPath::GetNumSamplePoints() const
{
	if (SamplingMode == ELookAtSamplingMode::ControlPoints)
		return Spline ? Spline->GetNumberOfSplinePoints() : 0;
	return NumDivisions;
}

void AVCCSimLookAtPath::StartCapture()
{
	if (!FlashPawnRef)
	{
		UE_LOG(LogSimLookAtPath, Warning, TEXT("FlashPawnRef is not set"));
		return;
	}

	TArray<FVector> Positions;
	TArray<FRotator> Rotations;
	GetSamplePoses(Positions, Rotations);

	if (Positions.IsEmpty())
	{
		UE_LOG(LogSimLookAtPath, Warning, TEXT("No sample poses generated"));
		return;
	}

	const FString Timestamp = FDateTime::Now().ToString(TEXT("%Y-%m-%d_%H-%M-%S"));
	ActiveSaveDirectory = SaveDirectoryBase / Timestamp;
	IFileManager::Get().MakeDirectory(*ActiveSaveDirectory, true);

	if (bCaptureRGB)
	{
		TArray<URGBCameraComponent*> Cams;
		FlashPawnRef->GetComponents(Cams);
		for (URGBCameraComponent* Cam : Cams)
		{
			if (!Cam->GetRenderTarget())
			{
				FRGBCameraConfig Config;
				Config.FOV = Cam->FOV;
				Config.Width = Cam->Width;
				Config.Height = Cam->Height;
				Cam->Configure(Config);
			}
		}
	}
	if (bCaptureDepth)
	{
		TArray<UDepthCameraComponent*> Cams;
		FlashPawnRef->GetComponents(Cams);
		for (UDepthCameraComponent* Cam : Cams)
		{
			if (!Cam->GetRenderTarget())
			{
				FDepthCameraConfig Config;
				Config.FOV = Cam->FOV;
				Config.Width = Cam->Width;
				Config.Height = Cam->Height;
				Cam->Configure(Config);
			}
		}
	}
	if (bCaptureSeg)
	{
		TArray<USegCameraComponent*> Cams;
		FlashPawnRef->GetComponents(Cams);
		for (USegCameraComponent* Cam : Cams)
		{
			if (!Cam->GetRenderTarget())
			{
				FSegmentationCameraConfig Config;
				Config.FOV = Cam->FOV;
				Config.Width = Cam->Width;
				Config.Height = Cam->Height;
				Cam->Configure(Config);
			}
		}
	}
	if (bCaptureNormal)
	{
		TArray<UNormalCameraComponent*> Cams;
		FlashPawnRef->GetComponents(Cams);
		for (UNormalCameraComponent* Cam : Cams)
		{
			if (!Cam->GetRenderTarget())
			{
				FNormalCameraConfig Config;
				Config.FOV = Cam->FOV;
				Config.Width = Cam->Width;
				Config.Height = Cam->Height;
				Cam->Configure(Config);
			}
		}
	}

	FlashPawnRef->SetPathPanel(Positions, Rotations);
	FlashPawnRef->MoveTo(0);

	bCaptureInProgress = true;
	*JobNum = 0;

	GetWorldTimerManager().SetTimer(
		CaptureTimerHandle,
		this,
		&AVCCSimLookAtPath::ProcessCaptureTick,
		0.2f,
		true
	);

	UE_LOG(LogSimLookAtPath, Log, TEXT("Capture started: %d poses → %s"),
		Positions.Num(), *ActiveSaveDirectory);
}

void AVCCSimLookAtPath::StopCapture()
{
	bCaptureInProgress = false;
	GetWorldTimerManager().ClearTimer(CaptureTimerHandle);
	ActiveSaveDirectory.Empty();
	UE_LOG(LogSimLookAtPath, Log, TEXT("Capture stopped"));
}

void AVCCSimLookAtPath::ProcessCaptureTick()
{
	if (!bCaptureInProgress || !FlashPawnRef)
	{
		StopCapture();
		return;
	}

	if (FlashPawnRef->IsReady())
	{
		const int32 PoseIndex = FlashPawnRef->GetCurrentIndex();
		CaptureCurrentPose();
		FlashPawnRef->MoveToNext();

		if (PoseIndex >= FlashPawnRef->GetPoseCount() - 1)
		{
			StopCapture();
		}
	}
	else if (*JobNum == 0)
	{
		FlashPawnRef->MoveForward();
	}
}

void AVCCSimLookAtPath::CaptureCurrentPose()
{
	if (!FlashPawnRef) return;

	const int32 PoseIndex = FlashPawnRef->GetCurrentIndex();
	bool bAnyCaptured = false;

	if (bCaptureRGB)   SaveRGB(PoseIndex, bAnyCaptured);
	if (bCaptureDepth) SaveDepth(PoseIndex, bAnyCaptured);
	if (bCaptureSeg)   SaveSeg(PoseIndex, bAnyCaptured);
	if (bCaptureNormal) SaveNormal(PoseIndex, bAnyCaptured);
}

void AVCCSimLookAtPath::SaveRGB(int32 PoseIndex, bool& bAnyCaptured)
{
	TArray<URGBCameraComponent*> Cameras;
	FlashPawnRef->GetComponents<URGBCameraComponent>(Cameras);
	*JobNum += Cameras.Num();

	for (int32 i = 0; i < Cameras.Num(); ++i)
	{
		URGBCameraComponent* Camera = Cameras[i];
		if (!Camera)
		{
			*JobNum -= 1;
			continue;
		}

		if (!Camera->IsActive()) Camera->SetActive(true);

		int32 CamIdx = Camera->GetSensorIndex();
		if (CamIdx < 0) CamIdx = i;

		FString Filename = ActiveSaveDirectory / FString::Printf(
			TEXT("RGB_Cam%02d_Pose%03d.png"), CamIdx, PoseIndex);

		const FIntPoint Size = {Camera->GetImageSize().first, Camera->GetImageSize().second};

		Camera->AsyncGetRGBImageData(
			[Filename, Size, JobNum = this->JobNum](const TArray<FColor>& ImageData)
			{
				(new FAutoDeleteAsyncTask<FAsyncImageSaveTask>(ImageData, Size, Filename))
					->StartBackgroundTask();
				*JobNum -= 1;
			});

		bAnyCaptured = true;
	}
}

void AVCCSimLookAtPath::SaveDepth(int32 PoseIndex, bool& bAnyCaptured)
{
	TArray<UDepthCameraComponent*> Cameras;
	FlashPawnRef->GetComponents<UDepthCameraComponent>(Cameras);
	*JobNum += Cameras.Num();

	for (int32 i = 0; i < Cameras.Num(); ++i)
	{
		UDepthCameraComponent* Camera = Cameras[i];
		if (!Camera)
		{
			*JobNum -= 1;
			continue;
		}

		if (!Camera->IsActive()) Camera->SetActive(true);

		int32 CamIdx = Camera->GetSensorIndex();
		if (CamIdx < 0) CamIdx = i;

		FString Filename = ActiveSaveDirectory / FString::Printf(
			TEXT("Depth16_Cam%02d_Pose%03d.png"), CamIdx, PoseIndex);

		const FIntPoint Size = {Camera->GetImageSize().first, Camera->GetImageSize().second};

		Camera->AsyncGetDepthImageData(
			[Filename, Size, JobNum = this->JobNum](const TArray<FFloat16Color>& ImageData)
			{
				TArray<float> DepthValues;
				DepthValues.SetNum(ImageData.Num());
				for (int32 idx = 0; idx < ImageData.Num(); ++idx)
				{
					DepthValues[idx] = ImageData[idx].R;
				}
				(new FAutoDeleteAsyncTask<FAsyncDepthSaveTask>(DepthValues, Size, Filename))
					->StartBackgroundTask();
				*JobNum -= 1;
			});

		bAnyCaptured = true;
	}
}

void AVCCSimLookAtPath::SaveSeg(int32 PoseIndex, bool& bAnyCaptured)
{
	TArray<USegCameraComponent*> Cameras;
	FlashPawnRef->GetComponents<USegCameraComponent>(Cameras);
	*JobNum += Cameras.Num();

	for (int32 i = 0; i < Cameras.Num(); ++i)
	{
		USegCameraComponent* Camera = Cameras[i];
		if (!Camera)
		{
			*JobNum -= 1;
			continue;
		}

		if (!Camera->IsActive()) Camera->SetActive(true);

		int32 CamIdx = Camera->GetSensorIndex();
		if (CamIdx < 0) CamIdx = i;

		FString Filename = ActiveSaveDirectory / FString::Printf(
			TEXT("Seg_Cam%02d_Pose%03d.png"), CamIdx, PoseIndex);

		const FIntPoint Size = {Camera->GetImageSize().first, Camera->GetImageSize().second};

		Camera->AsyncGetSegmentationImageData(
			[Filename, Size, JobNum = this->JobNum](const TArray<FColor>& ImageData)
			{
				(new FAutoDeleteAsyncTask<FAsyncImageSaveTask>(ImageData, Size, Filename))
					->StartBackgroundTask();
				*JobNum -= 1;
			});

		bAnyCaptured = true;
	}
}

void AVCCSimLookAtPath::SaveNormal(int32 PoseIndex, bool& bAnyCaptured)
{
	TArray<UNormalCameraComponent*> Cameras;
	FlashPawnRef->GetComponents<UNormalCameraComponent>(Cameras);
	*JobNum += Cameras.Num();

	for (int32 i = 0; i < Cameras.Num(); ++i)
	{
		UNormalCameraComponent* Camera = Cameras[i];
		if (!Camera)
		{
			*JobNum -= 1;
			continue;
		}

		if (!Camera->IsActive()) Camera->SetActive(true);

		int32 CamIdx = Camera->GetSensorIndex();
		if (CamIdx < 0) CamIdx = i;

		FString Filename = ActiveSaveDirectory / FString::Printf(
			TEXT("Normal_Cam%02d_Pose%03d.exr"), CamIdx, PoseIndex);

		const FIntPoint Size = {Camera->GetImageSize().first, Camera->GetImageSize().second};

		Camera->AsyncGetNormalImageData(
			[Filename, Size, JobNum = this->JobNum](const TArray<FFloat16Color>& NormalData)
			{
				(new FAutoDeleteAsyncTask<FAsyncNormalEXRSaveTask>(NormalData, Size, Filename))
					->StartBackgroundTask();
				*JobNum -= 1;
			});

		bAnyCaptured = true;
	}
}
