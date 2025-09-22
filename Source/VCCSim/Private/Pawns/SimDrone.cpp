// /*
// * Copyright (C) 2025 Visual Computing Research Center, Shenzhen University
//  *
//  * This program is free software: you can redistribute it and/or modify
//  * it under the terms of the GNU Affero General Public License as published
//  * by the Free Software Foundation, either version 3 of the License, or
//  * (at your option) any later version.
//  *
//  * This program is distributed in the hope that it will be useful,
//  * but WITHOUT ANY WARRANTY; without even the implied warranty of
//  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  * GNU Affero General Public License for more details.
//  *
//  * You should have received a copy of the GNU Affero General Public License
//  * along with this program.  If not, see <https://www.gnu.org/licenses/>.
// */
//
// #include "Pawns/SimDrone.h"
// #include "Components/BoxComponent.h"
// #include "EnhancedInputComponent.h"
// #include "Components/SplineComponent.h"
// #include "EnhancedInputSubsystems.h"
// #include "InputMappingContext.h"
// #include "InputAction.h"
// #include "Kismet/GameplayStatics.h"
// #include "PostProcess/PostProcessMaterialInputs.h"
//
// ASimDrone::ASimDrone()
// {
//
//     PrimaryActorTick.bCanEverTick = true;
//
//     // Create root component
//     DroneRoot = CreateDefaultSubobject<UBoxComponent>(TEXT("DroneRoot"));
//     RootComponent = DroneRoot;
//     
//     // Create drone mesh
//     DroneMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("DroneMesh"));
//     DroneMesh->SetupAttachment(DroneRoot);
//     DroneMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
//     
//     // Create rotor meshes directly attached to the drone root
//     RotorMesh1 = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("RotorMesh1"));
//     RotorMesh2 = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("RotorMesh2"));
//     RotorMesh3 = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("RotorMesh3"));
//     RotorMesh4 = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("RotorMesh4"));
//     
//     RotorMesh1->SetupAttachment(DroneRoot);
//     RotorMesh2->SetupAttachment(DroneRoot);
//     RotorMesh3->SetupAttachment(DroneRoot);
//     RotorMesh4->SetupAttachment(DroneRoot);
//
//     RotorMesh1->SetCollisionEnabled(ECollisionEnabled::NoCollision);
//     RotorMesh2->SetCollisionEnabled(ECollisionEnabled::NoCollision);
//     RotorMesh3->SetCollisionEnabled(ECollisionEnabled::NoCollision);
//     RotorMesh4->SetCollisionEnabled(ECollisionEnabled::NoCollision);
//     
//     // Hide from scene capture by default
//     DroneMesh->bHiddenInSceneCapture = true;
//     RotorMesh1->bHiddenInSceneCapture = true;
//     RotorMesh2->bHiddenInSceneCapture = true;
//     RotorMesh3->bHiddenInSceneCapture = true;
//     RotorMesh4->bHiddenInSceneCapture = true;
// }
//
// void ASimDrone::BeginPlay()
// {
//     Super::BeginPlay();
//     AddMapContext();
// }
//
// void ASimDrone::Tick(float DeltaTime)
// {
//     Super::Tick(DeltaTime);
//     if (IfAutoMove)
//     {
//         AutoMove(DeltaTime);
//         CalculateDistance();
//         ASimDrone::FollowThePathAndSteer(DeltaTime);
//     }
//     else
//     {
//         if (bRecorded && RecordState)
//         {
//             TimeSinceLastCapture += DeltaTime;
//             if (TimeSinceLastCapture >= RecordInterval)
//             {
//                 TimeSinceLastCapture = 0.0f;
//                 FPoseData PoseData;
//                 PoseData.Location = GetActorLocation();
//                 PoseData.Rotation = GetActorRotation();
//                 PoseData.Timestamp = FPlatformTime::Seconds();
//                 Recorder->SubmitPoseData(this, MoveTemp(PoseData));
//             }
//         }
//         
//         if (bUseTarget)
//         {
//             MoveToTarget(DeltaTime);
//         }
//         else
//         {
//             UE_LOG(LogTemp, Log, TEXT("Moving by input "));
//             // Apply manual input controls
//             FVector CurrentLocation = GetActorLocation();
//             FRotator CurrentRotation = GetActorRotation();
//             
//             // Handle Pitch & Roll
//             // Negative when moving forward, positive when moving backward
//             float TargetPitch = -MovementInput.Y * MaxPitchAngle;
//             
//             // Negative when moving left, positive when moving right
//             float TargetRoll = MovementInput.X * MaxRollAngle;  
//         
//             // Smooth interpolation for more natural tilting (simulates physical inertia)
//             CurrentRotation.Pitch = FMath::FInterpTo(CurrentRotation.Pitch, TargetPitch, DeltaTime, TiltInterpSpeed);
//             CurrentRotation.Roll = FMath::FInterpTo(CurrentRotation.Roll, TargetRoll, DeltaTime, TiltInterpSpeed);
//         
//             // Handle vertical movement
//             CurrentLocation.Z += VerticalInput * MovementSpeed * DeltaTime;
//             
//             // Handle horizontal movement with rotation-based direction
//             FRotator YawRotation(0.0f, CurrentRotation.Yaw, 0.0f);
//             FVector ForwardDirection = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::X);
//             FVector RightDirection = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::Y);
//             
//             // physical simulation 
//             if (bEnablePhysical == true)
//             {
//                 // if there is input,TargetSpeed will be set to MovementSpeed; else be zero.
//                 float TargetSpeed = (MovementInput.Size() > 0) ? MovementSpeed : 0.0f;
//                 CurrentMoveSpeed = FMath::FInterpTo(CurrentMoveSpeed, TargetSpeed, DeltaTime, AccelerationSpeed);
//                 if (MovementInput.Size() > 0)
//                 {
//                     LastMoveDirection = (ForwardDirection * MovementInput.Y + RightDirection * MovementInput.X).GetSafeNormal();
//                 }
//                 CurrentLocation += LastMoveDirection * CurrentMoveSpeed * DeltaTime;
//             }
//             else  
//             {
//                 // moving without physical
//                 CurrentLocation += (ForwardDirection * MovementInput.Y + RightDirection * MovementInput.X) 
//                                   * MovementSpeed * DeltaTime;
//             }
//             // Handle yaw rotation
//             CurrentRotation.Yaw += YawInput * RotationSpeed * DeltaTime;
//             
//             // Apply the new position and rotation
//             SetActorLocationAndRotation(CurrentLocation, CurrentRotation);
//         }
//     }
//     // Always rotate the rotors
//     RotateRotors(DeltaTime);
// }
//
// void ASimDrone::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
// {
//     Super::SetupPlayerInputComponent(PlayerInputComponent);
//
//     EnhancedInputComponent = CastChecked<UEnhancedInputComponent>(PlayerInputComponent);
//
//     if (EnhancedInputComponent)
//     {
//         EnhancedInputComponent->BindAction(ThrottleAction, ETriggerEvent::Triggered,
//             this, &ASimDrone::HandleThrottleInput);
//         EnhancedInputComponent->BindAction(ThrottleAction, ETriggerEvent::Completed,
//             this, &ASimDrone::HandleThrottleReleased);
//         
//         EnhancedInputComponent->BindAction(MovementAction, ETriggerEvent::Triggered,
//             this, &ASimDrone::HandleMovementInput);
//         EnhancedInputComponent->BindAction(MovementAction, ETriggerEvent::Completed,
//             this, &ASimDrone::HandleMovementReleased);
//         
//         EnhancedInputComponent->BindAction(YawAction, ETriggerEvent::Triggered,
//             this, &ASimDrone::HandleYawInput);
//         EnhancedInputComponent->BindAction(YawAction, ETriggerEvent::Completed,
//             this, &ASimDrone::HandleYawReleased);
//
//         EnhancedInputComponent->BindAction(LookAction, ETriggerEvent::Triggered,
//             this, &ASimDrone::HandleLookInput);
//     }
// }
//
// bool ASimDrone::IfCloseToTarget(FVector Location, FRotator Rotation) const
// {
//     // Calculate position error
//     const FVector CurrentLocation = GetActorLocation();
//     const FRotator CurrentRotation = GetActorRotation();
//     const FVector PositionError = Location - CurrentLocation;
//     
//     // Position check
//     bool PositionOK = PositionError.SizeSquared() < FMath::Square(PositionThreshold);
//     
//     // Rotation checks
//     bool YawOK = FMath::Abs(FMath::FindDeltaAngleDegrees(
//                 Rotation.Yaw, CurrentRotation.Yaw)) < RotationThreshold;
//     
//     bool PitchOK = FMath::Abs(FMath::FindDeltaAngleDegrees(
//                  Rotation.Pitch, CurrentRotation.Pitch)) < RotationThreshold;
//     
//     bool RollOK = FMath::Abs(FMath::FindDeltaAngleDegrees(
//                Rotation.Roll, CurrentRotation.Roll)) < RotationThreshold;
//     
//     return PositionOK && YawOK && PitchOK && RollOK;
// }
//
// void ASimDrone::FollowThePathAndSteer(float DeltaTime)
// {
//     if (Path)
//     {
//         const auto NewLocation = Path->Spline->GetLocationAtDistanceAlongSpline(
//             CourseDistance, ESplineCoordinateSpace::World);
//         
//         const auto NextRotation = Path->Spline->GetRotationAtDistanceAlongSpline(
//             CourseDistance, ESplineCoordinateSpace::World);
//         
//         const auto NewRotation = FMath::RInterpTo(GetActorRotation(), NextRotation,
//             DeltaTime, 8.f);
//         if (bEnablePhysical)
//         {
//             
//             FVector CurrentLocation = GetActorLocation();
//             FVector Delta = NewLocation - CurrentLocation;
//             float HorizontalDistance = FVector(Delta.X, Delta.Y, 0.f).Size();
//             
//             float Speed = HorizontalDistance / DeltaTime;
//             
//             const float MaxHorizontalSpeed = 120.f; 
//             
//             float CalculatedPitch = -FMath::Clamp((Speed / MaxHorizontalSpeed) * MaxPitchAngle, 0.f, MaxPitchAngle);
//             
//             SetActorLocationAndRotation(NewLocation,
//                 FRotator(CalculatedPitch, NewRotation.Yaw, NewRotation.Roll));
//         }
//         else
//         {
//             SetActorLocationAndRotation(NewLocation,
//                 FRotator(0, NewRotation.Yaw, NewRotation.Roll));
//         }
//     }
// }
//
//
// void ASimDrone::HandleThrottleInput(const FInputActionValue& Value)
// {
//     VerticalInput = Value.Get<float>();
// }
//
// void ASimDrone::HandleThrottleReleased(const FInputActionValue& Value)
// {
//     VerticalInput = 0.0f;
// }
//
// void ASimDrone::HandleMovementInput(const FInputActionValue& Value)
// {
//     MovementInput = Value.Get<FVector2D>();
// }
//
// void ASimDrone::HandleMovementReleased(const FInputActionValue& Value)
// {
//     MovementInput = FVector2D::ZeroVector;
// }
//
// void ASimDrone::HandleYawInput(const FInputActionValue& Value)
// {
//     YawInput = Value.Get<float>();
// }
//
// void ASimDrone::HandleYawReleased(const FInputActionValue& Value)
// {
//     YawInput = 0.0f;
// }
//
// void ASimDrone::HandleLookInput(const FInputActionValue& Value)
// {
//     const FVector2D LookAxisVector = Value.Get<FVector2D>();
//     
//     if (APlayerController* PlayerController = Cast<APlayerController>(Controller))
//     {
//         // Add yaw input to the camera spring arm
//         AddControllerYawInput(LookAxisVector.X);
//         // Add pitch input to the camera spring arm
//         AddControllerPitchInput(LookAxisVector.Y);
//     }
// }
//
// void ASimDrone::MoveToTarget(float DeltaTime)
// {
//     if (!bUseTarget)
//     {
//         return;
//     }
//
//     // Get current state
//     FVector CurrentLocation = GetActorLocation();
//     FRotator CurrentRotation = GetActorRotation();
//     
//     // Calculate direction to target
//     FVector Direction = TargetLocation - CurrentLocation;
//     float DistanceToTarget = Direction.Size();
//     
//     // Check if we've already reached the position target
//     if (DistanceToTarget <= PositionThreshold)
//     {
//         // We've reached the target position, now just handle rotation
//         FRotator RotationDelta = TargetRotation - CurrentRotation;
//         RotationDelta.Normalize();
//         
//         // Check if we need to rotate
//         if (!RotationDelta.IsNearlyZero(RotationThreshold))
//         {
//             // Calculate the rotation step for this frame
//             float RotationStep = RotationSpeed * DeltaTime;
//             
//             // Limit the rotation step to not overshoot the target
//             FRotator StepRotation = RotationDelta;
//             if (FMath::Abs(RotationDelta.Yaw) > RotationStep)
//             {
//                 StepRotation.Yaw = RotationStep * FMath::Sign(RotationDelta.Yaw);
//             }
//             if (FMath::Abs(RotationDelta.Pitch) > RotationStep)
//             {
//                 StepRotation.Pitch = RotationStep * FMath::Sign(RotationDelta.Pitch);
//             }
//             if (FMath::Abs(RotationDelta.Roll) > RotationStep)
//             {
//                 StepRotation.Roll = RotationStep * FMath::Sign(RotationDelta.Roll);
//             }
//             
//             // Apply the rotation
//             CurrentRotation += StepRotation;
//             SetActorRotation(CurrentRotation);
//         }
//         else if (IfCloseToTarget(TargetLocation, TargetRotation))
//         {
//             // We've reached both position and rotation targets
//             bUseTarget = false;
//         }
//     }
//     else
//     {
//         // Calculate the movement step for this frame
//         float MovementStep = MovementSpeed * DeltaTime;
//         
//         // Normalize the direction and scale by movement step
//         Direction.Normalize();
//         FVector Movement = Direction * MovementStep;
//         
//         // Ensure we don't overshoot the target
//         if (Movement.Size() > DistanceToTarget)
//         {
//             Movement = Direction * DistanceToTarget;
//         }
//         
//         // Apply the movement
//         CurrentLocation += Movement;
//         SetActorLocation(CurrentLocation);
//         
//         // Also gradually rotate toward target yaw during movement
//         float YawDelta = FMath::FindDeltaAngleDegrees(CurrentRotation.Yaw, TargetRotation.Yaw);
//         if (FMath::Abs(YawDelta) > RotationThreshold)
//         {
//             float YawStep = RotationSpeed * DeltaTime;
//             if (FMath::Abs(YawDelta) > YawStep)
//             {
//                 CurrentRotation.Yaw += YawStep * FMath::Sign(YawDelta);
//             }
//             else
//             {
//                 CurrentRotation.Yaw = TargetRotation.Yaw;
//             }
//             
//             SetActorRotation(CurrentRotation);
//         }
//     }
// }
//
// void ASimDrone::RotateRotors(float DeltaTime)
// {
//     const float RotationAmount = RotorRotationSpeed * DeltaTime;
//     const FRotator ClockwiseRotation(0.0f, RotationAmount, 0.0f);
//     const FRotator CounterClockwiseRotation(0.0f, -RotationAmount, 0.0f);
//     
//     // Rotors 1 and 3 rotate clockwise, 2 and 4 counterclockwise
//     RotorMesh1->AddLocalRotation(ClockwiseRotation);
//     RotorMesh3->AddLocalRotation(ClockwiseRotation);
//     RotorMesh2->AddLocalRotation(CounterClockwiseRotation);
//     RotorMesh4->AddLocalRotation(CounterClockwiseRotation);
//
//     // TODO: Changing the rotation speed of each rotor to accommodate the different speeds and attitudes of the drone
//     // 
//     // FVector CurrentVelocity = GetVelocity();  
//     // float ForwardSpeed = CurrentMoveSpeed;
//     // float YawRotation = GetActorRotation().Yaw;  
//     // UE_LOG(LogTemp, Display, TEXT("%f %f"), YawRotation,ForwardSpeed);
//     // 
//     // float RotorSpeedFactor = 2000.0f;  
//     //
//     // 
//     // 
//     // float RotorSpeed1 = (ForwardSpeed - RotorSpeedFactor) * (YawRotation > 0 ? 1.0f : -1.0f);  // Rotor 1 顺时针
//     // float RotorSpeed2 = (ForwardSpeed - RotorSpeedFactor) * (YawRotation > 0 ? -1.0f : 1.0f);  // Rotor 2 逆时针
//     // float RotorSpeed3 = (ForwardSpeed + RotorSpeedFactor) * (YawRotation > 0 ? 1.0f : -1.0f);  // Rotor 3 顺时针
//     // float RotorSpeed4 = (ForwardSpeed + RotorSpeedFactor) * (YawRotation > 0 ? -1.0f : 1.0f);  // Rotor 4 逆时针
//     //
//     // 
//     // const float RotationAmount1 = RotorSpeed1 * DeltaTime;
//     // const float RotationAmount2 = RotorSpeed2 * DeltaTime;
//     // const float RotationAmount3 = RotorSpeed3 * DeltaTime;
//     // const float RotationAmount4 = RotorSpeed4 * DeltaTime;
//     //
//     // 
//     // RotorMesh1->AddLocalRotation(FRotator(0.0f, RotationAmount1, 0.0f));
//     // RotorMesh2->AddLocalRotation(FRotator(0.0f, RotationAmount2, 0.0f));
//     // RotorMesh3->AddLocalRotation(FRotator(0.0f, RotationAmount3, 0.0f));
//     // RotorMesh4->AddLocalRotation(FRotator(0.0f, RotationAmount4, 0.0f));
// }