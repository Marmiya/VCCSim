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

DEFINE_LOG_CATEGORY_STATIC(LogCameraSensor, Log, All);


#include "Sensors/SensorBase.h"
#include "Engine/World.h"
#include "GameFramework/Actor.h"

USensorBaseComponent::USensorBaseComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.bStartWithTickEnabled = true;
}

void USensorBaseComponent::BeginPlay()
{
	Super::BeginPlay();
	ParentActor = GetOwner();
	SetComponentTickEnabled(false);
	SetCollisionEnabled(ECollisionEnabled::NoCollision);
	SetCollisionResponseToAllChannels(ECR_Ignore);
}

void USensorBaseComponent::OnComponentCreated()
{
	Super::OnComponentCreated();
}

UCameraBaseComponent::UCameraBaseComponent()
{
	CaptureComponent = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("CaptureComponent"));
	if (CaptureComponent)
	{
		CaptureComponent->SetupAttachment(this);
		CaptureComponent->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives;
		CaptureComponent->bCaptureEveryFrame = false;
		CaptureComponent->bCaptureOnMovement = false;
	}
}

void UCameraBaseComponent::SetCaptureComponent() const
{
	if (!CaptureComponent) return;

	CaptureComponent->FOVAngle = FOV;
	CaptureComponent->ProjectionType = bOrthographic ? ECameraProjectionMode::Orthographic : ECameraProjectionMode::Perspective;
	CaptureComponent->OrthoWidth = OrthoWidth;

	if (UTextureRenderTarget2D* RenderTarget = GetRenderTarget())
	{
		CaptureComponent->TextureTarget = RenderTarget;
	}
	else 
	{
		UE_LOG(LogCameraSensor, Error, TEXT("GetRenderTarget() returned null!"));
	}
}

bool UCameraBaseComponent::CheckComponentAndRenderTarget() const
{
	return CaptureComponent && GetRenderTarget();
}

void UCameraBaseComponent::ComputeIntrinsics()
{
	if (bOrthographic)
	{
		const float Scale = 2.0f / OrthoWidth;
		CameraIntrinsics = FMatrix44f(
			FPlane4f(Scale, 0.0f, 0.0f, 0.0f),
			FPlane4f(0.0f, Scale, 0.0f, 0.0f),
			FPlane4f(0.0f, 0.0f, 1.0f, 0.0f),
			FPlane4f(0.0f, 0.0f, 0.0f, 1.0f)
		);
	}
	else
	{
		const float HalfFOVRad = FMath::DegreesToRadians(FOV / 2.0f);
		const float FocalLengthX = Width / (2.0f * FMath::Tan(HalfFOVRad));
		const float FocalLengthY = Height / (2.0f * FMath::Tan(HalfFOVRad));
		const float CenterX = Width / 2.0f;
		const float CenterY = Height / 2.0f;

		CameraIntrinsics = FMatrix44f(
			FPlane4f(FocalLengthX, 0.0f, CenterX, 0.0f),
			FPlane4f(0.0f, FocalLengthY, CenterY, 0.0f),
			FPlane4f(0.0f, 0.0f, 1.0f, 0.0f),
			FPlane4f(0.0f, 0.0f, 0.0f, 1.0f)
		);
	}
}