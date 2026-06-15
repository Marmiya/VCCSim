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
#include "Engine/TextureRenderTarget2D.h"
#include "GameFramework/Actor.h"
#include "RenderingThread.h"
#include "RHIGPUReadback.h"
#include <atomic>

USensorBaseComponent::USensorBaseComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.bStartWithTickEnabled = true;
}

void USensorBaseComponent::BeginPlay()
{
	Super::BeginPlay();
	SetComponentTickEnabled(false);
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
	else 
	{
		UE_LOG(LogCameraSensor, Error, TEXT("Failed to create CaptureComponent!"));
	}
}

void UCameraBaseComponent::SetCaptureComponent() const
{
	if (!CaptureComponent) return;

	CaptureComponent->FOVAngle = FOV;
	CaptureComponent->ProjectionType = ECameraProjectionMode::Perspective;

	if (RenderTarget)
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
	return CaptureComponent && RenderTarget;
}

void UCameraBaseComponent::WarmupCapture()
{
	InitializeRenderTargets();
	SetCaptureComponent();
	if (CaptureComponent)
	{
		CaptureComponent->CaptureScene();
	}
}

void UCameraBaseComponent::ComputeIntrinsics()
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

// ============================================================================
// Async GPU readback: non-blocking EnqueueCopy + polled Lock, so dataset capture
// never stalls the render thread on BlockUntilGPUIdle. The pending list is touched
// only inside render commands (serialized), so no extra locking is needed.
// ============================================================================

struct UCameraBaseComponent::FReadbackState
{
	struct FEntry
	{
		TUniquePtr<FRHIGPUTextureReadback> Readback;
		TUniqueFunction<void(const void*, int32)> OnReady;
	};
	TArray<FEntry> Pending;            // render-thread only
	std::atomic<int32> InFlight{ 0 };  // ++ on the game thread, -- on the render thread
};

UCameraBaseComponent::~UCameraBaseComponent()
{
	if (PollTickerHandle.IsValid())
	{
		FTSTicker::GetCoreTicker().RemoveTicker(PollTickerHandle);
		PollTickerHandle.Reset();
	}
}

void UCameraBaseComponent::EnqueueReadback(
	TUniqueFunction<void(const void* MappedData, int32 RowPitchInPixels)> OnReady)
{
	if (!RenderTarget)
	{
		return;
	}
	if (!ReadbackState.IsValid())
	{
		ReadbackState = MakeShared<FReadbackState, ESPMode::ThreadSafe>();
	}
	ReadbackState->InFlight.fetch_add(1);

	UTextureRenderTarget2D* RT = RenderTarget;
	TSharedPtr<FReadbackState, ESPMode::ThreadSafe> State = ReadbackState;
	ENQUEUE_RENDER_COMMAND(VCCSimEnqueueReadback)(
		[RT, State, OnReady = MoveTemp(OnReady)](FRHICommandListImmediate& RHICmdList) mutable
		{
			FTextureRenderTargetResource* Res = RT->GetRenderTargetResource();
			FRHITexture* Tex = Res ? Res->GetRenderTargetTexture() : nullptr;
			if (!Tex)
			{
				State->InFlight.fetch_sub(1);
				return;
			}
			FReadbackState::FEntry Entry;
			Entry.Readback = MakeUnique<FRHIGPUTextureReadback>(TEXT("VCCSimReadback"));
			Entry.Readback->EnqueueCopy(RHICmdList, Tex);
			Entry.OnReady = MoveTemp(OnReady);
			State->Pending.Add(MoveTemp(Entry));
		});

	EnsurePollTicker();
}

void UCameraBaseComponent::EnsurePollTicker()
{
	if (PollTickerHandle.IsValid())
	{
		return;
	}
	TWeakObjectPtr<UCameraBaseComponent> WeakThis(this);
	PollTickerHandle = FTSTicker::GetCoreTicker().AddTicker(
		FTickerDelegate::CreateLambda([WeakThis](float) -> bool
		{
			UCameraBaseComponent* Self = WeakThis.Get();
			if (!Self)
			{
				return false;
			}
			Self->PollReadbacks();
			if (Self->ReadbackState.IsValid() && Self->ReadbackState->InFlight.load() > 0)
			{
				return true;
			}
			Self->PollTickerHandle.Reset();
			return false;
		}), 0.0f);
}

void UCameraBaseComponent::PollReadbacks()
{
	if (!ReadbackState.IsValid())
	{
		return;
	}
	TSharedPtr<FReadbackState, ESPMode::ThreadSafe> State = ReadbackState;
	ENQUEUE_RENDER_COMMAND(VCCSimPollReadbacks)(
		[State](FRHICommandListImmediate& RHICmdList)
		{
			for (int32 i = 0; i < State->Pending.Num(); )
			{
				FReadbackState::FEntry& Entry = State->Pending[i];
				if (Entry.Readback->IsReady())
				{
					int32 RowPitchInPixels = 0;
					void* Mapped = Entry.Readback->Lock(RowPitchInPixels);
					if (Mapped && Entry.OnReady)
					{
						Entry.OnReady(Mapped, RowPitchInPixels);
					}
					Entry.Readback->Unlock();
					State->Pending.RemoveAtSwap(i);
					State->InFlight.fetch_sub(1);
				}
				else
				{
					++i;
				}
			}
		});
}