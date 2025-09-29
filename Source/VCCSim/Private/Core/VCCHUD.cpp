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

#include "Core/VCCHUD.h"
#include "Core/MenuWidgets.h"
#include "API/RpcServer.h"
#include "Sensors/LidarSensor.h"
#include "Sensors/CameraSensor.h"
#include "Sensors/SegmentCamera.h"
#include "Sensors/NormalCamera.h"
#include "Simulation/Recorder.h"
#include "Simulation/MeshManager.h"
#include "Simulation/SceneAnalysisManager.h"
#include "Utils/ConfigParser.h"
#include "Utils/InsMeshHolder.h"
#include "Utils/VCCSIMDisplayWidget.h"
#include "EnhancedInputComponent.h"
#include "LevelSequencePlayer.h"
#include "LevelSequenceActor.h"
#include "Kismet/GameplayStatics.h"
#include "Sensors/DepthCamera.h"
#include "Simulation/SemanticAnalyzer.h"

DEFINE_LOG_CATEGORY_STATIC(LogVCCHUD, Log, All);

void AVCCHUD::BeginPlay()
{
    Super::BeginPlay();
    
    FVCCSimConfig Config = ParseConfig();
    
    // SceneAnalysisManager = Cast<ASceneAnalysisManager>(UGameplayStatics::
    //     GetActorOfClass(GetWorld(), ASceneAnalysisManager::StaticClass()));
    
    SetupRecorder(Config);
    SetupWidgetsAndLS(Config);
    auto RCMaps = SetupActors(Config);
    
    if (Config.VCCSim.UseMeshManager)
    {
        MeshManager = NewObject<UFMeshManager>(Holder);
        MeshManager->RConfigure(Config);
    }
    
    RunServer(Config, Holder, RCMaps, MeshManager, Recorder);
    
    if (Config.VCCSim.StartWithRecording)
    {
        UE_LOG(LogVCCHUD, Log, TEXT("Starting recording automatically"));
        Recorder->StartRecording();
    }
}

void AVCCHUD::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    UE_LOG(LogVCCHUD, Warning, TEXT("VCCHUD::EndPlay called, reason: %d "
                                    "- Stopping recording"), (int32)EndPlayReason);

    // Clean up Recorder FIRST before calling Super::EndPlay
    if (Recorder)
    {
        UE_LOG(LogVCCHUD, Log, TEXT("Stopping Recorder in EndPlay"));
        Recorder->StopRecording();
    }

    Super::EndPlay(EndPlayReason);

    if (APlayerController* PC = GetOwningPlayerController())
    {
        // Clean up Enhanced Input Subsystem
        if (UEnhancedInputLocalPlayerSubsystem* Subsystem =
            ULocalPlayer::GetSubsystem<UEnhancedInputLocalPlayerSubsystem>(PC->GetLocalPlayer()))
        {
            Subsystem->RemoveMappingContext(DefaultMappingContext);
        }

        // Clean up InputComponent bindings
        if (UEnhancedInputComponent* EnhancedInputComponent =
            Cast<UEnhancedInputComponent>(PC->InputComponent))
        {
            EnhancedInputComponent->ClearBindingsForObject(this);
        }
    }

    if (CurrentPauseMenu)
    {
        CurrentPauseMenu->RemoveFromParent();
    }

    if (WidgetInstance)
    {
        WidgetInstance->RemoveFromParent();
    }

    ShutdownServer();
}

void AVCCHUD::SetupEnhancedInput()
{
    if (!DefaultMappingContext || !PauseAction) return;
    APlayerController* PC = GetOwningPlayerController();
    if (!PC) return;

    // Get the local player subsystem
    if (UEnhancedInputLocalPlayerSubsystem* Subsystem =
        ULocalPlayer::GetSubsystem<UEnhancedInputLocalPlayerSubsystem>(PC->GetLocalPlayer()))
    {
        Subsystem->AddMappingContext(DefaultMappingContext, 1);
    }

    if (UEnhancedInputComponent* EnhancedInputComponent =
        Cast<UEnhancedInputComponent>(PC->InputComponent))
    {
        // Bind the pause action
        EnhancedInputComponent->BindAction(PauseAction, ETriggerEvent::Triggered,
            this, &AVCCHUD::OnPauseActionTriggered);
        if (ToggleRecordingAction)
        {
            EnhancedInputComponent->BindAction(ToggleRecordingAction, ETriggerEvent::Started,
                this, &AVCCHUD::OnToggleRecordingTriggered);
        }
    }

    PC->SetInputMode(FInputModeGameOnly());
    PC->bShowMouseCursor = false;
}

void AVCCHUD::OnPauseActionTriggered()
{    
    if (CurrentPauseMenu)
    {
        CurrentPauseMenu->RemoveFromParent();
        CurrentPauseMenu = nullptr;
    }

    if (PauseWidgetClass)
    {
        CurrentPauseMenu = CreateWidget<UPauseMenuWidget>(
            GetOwningPlayerController(), PauseWidgetClass);
        CurrentPauseMenu->AddToViewport();
                
        if (APlayerController* PC = GetOwningPlayerController())
        {
            PC->SetInputMode(FInputModeUIOnly());
            PC->SetShowMouseCursor(true);
            UGameplayStatics::SetGamePaused(GetWorld(), true);
        }
    }
    else
    {
        UE_LOG(LogVCCHUD, Warning, TEXT("PauseWidgetClass not set"));
    }
}

void AVCCHUD::OnToggleRecordingTriggered()
{
    if (Recorder)
    {
        if (Recorder->IsRecording())
        {
            Recorder->StopRecording();
        }
        else
        {
            Recorder->StartRecording();
        }
    }
    else
    {
        UE_LOG(LogVCCHUD, Warning, TEXT("Recorder not found!"));
    }
}

void AVCCHUD::SetupRecorder(FVCCSimConfig& Config)
{
    if (!Recorder || !IsValid(Recorder))
    {
        Recorder = GetWorld()->SpawnActor<ARecorder>(ARecorder::StaticClass(), FTransform::Identity);
        UE_LOG(LogVCCHUD, Log, TEXT("Created new Recorder instance"));
    }
    else
    {
        UE_LOG(LogVCCHUD, Log, TEXT("Reusing existing Recorder instance"));
    }

    Recorder->RecordState = Config.VCCSim.StartWithRecording;
}

void AVCCHUD::SetupWidgetsAndLS(const FVCCSimConfig& Config)
{
    if (!WidgetClass)
    {
        static ConstructorHelpers::FClassFinder<UVCCSIMDisplayWidget>
            HUDWidgetClass(TEXT("WidgetBlueprint'/VCCSim/HUD/BP_VCCSIMDisplayWidget_VF'"));
        WidgetClass = HUDWidgetClass.Succeeded() ? HUDWidgetClass.Class : nullptr;
    }

    WidgetInstance = CreateWidget<UVCCSIMDisplayWidget>(GetWorld(), WidgetClass);
    if (WidgetInstance)
    {
        WidgetInstance->AddToViewport();
    }
    else
    {
        UE_LOG(LogVCCHUD, Error, TEXT("Failed to create widget instance"));
    }

    SetupEnhancedInput();
    
    WidgetInstance->LogSavePath = Recorder->RecordingPath;

    TArray<AActor*> LevelSequenceActors;
    UGameplayStatics::GetAllActorsOfClass(GetWorld(), ALevelSequenceActor::StaticClass(), 
                                          LevelSequenceActors);

    for (AActor* Actor : LevelSequenceActors)
    {
        ALevelSequenceActor* LevelSequenceActor = Cast<ALevelSequenceActor>(Actor);
        if (LevelSequenceActor && LevelSequenceActor->ActorHasTag(FName(TEXT("MainShowOff"))))
        {
            if (ULevelSequencePlayer* SequencePlayer = LevelSequenceActor->GetSequencePlayer())
            {
                FFrameTime NewStartTime = Config.VCCSim.LS_StartOffset;
                FMovieSceneSequencePlaybackParams PlaybackParams(NewStartTime,
                    EUpdatePositionMethod::Play);
                SequencePlayer->SetPlaybackPosition(PlaybackParams);
            }
            break;
        }
    }

    Holder = GetWorld()->SpawnActor<AActor>(AActor::StaticClass(), FTransform::Identity);
    WidgetInstance->SetHolder(Holder);
    WidgetInstance->InitFromConfig(Config);
}

void AVCCHUD::SetupMainCharacter(const FVCCSimConfig& Config, TArray<AActor*> FoundPawns)
{
    FRobot MainRobotConfig;
    
    MainCharacter = Cast<APawn>(FindPawnInTagAndName(Config.VCCSim.MainCharacter, FoundPawns));

    if (Config.Robots.Num() == 1 && !MainCharacter)
    {
        MainCharacter = Cast<APawn>(FindPawnInTagAndName(Config.Robots[0].UETag, FoundPawns));
        MainRobotConfig = Config.Robots[0];
    }
    else
    {
        for (const FRobot& Robot : Config.Robots)
        {
            if (Robot.UETag == Config.VCCSim.MainCharacter)
            {
                MainRobotConfig = Robot;
                break;
            }
        }
    }
    
    if (!MainCharacter)
    {
        UE_LOG(LogVCCHUD, Warning, TEXT("SetupMainCharacter: "
                                      "MainCharacter not found!"));
        return;
    }
    else
    {
        if (SceneAnalysisManager)
        {
            SceneAnalysisManager->SemanticAnalyzer->CenterCharacter = MainCharacter;
        }
    }

    if (const auto SetManualControlFuc =
        MainCharacter->FindFunction(FName(TEXT("SetManualControl"))))
    {
        MainCharacter->ProcessEvent(SetManualControlFuc, (void*)&Config.VCCSim.ManualControl);
    }
    else
    {
        UE_LOG(LogVCCHUD, Warning, TEXT("AVCCHUD: SetManualControl function not found!"));
    }

    // Set the camera as the view target
    APlayerController* PlayerController = GetWorld()->GetFirstPlayerController();
    if (PlayerController)
    {
        PlayerController->SetViewTarget(MainCharacter);
    }
    else
    {
        UE_LOG(LogVCCHUD, Error, TEXT("PlayerController not found!"));
    }

    if (PlayerController && MainCharacter)
    {
        PlayerController->Possess(MainCharacter);
        if (auto Func = MainCharacter->FindFunction(FName(TEXT("AddMapContext"))))
        {
            MainCharacter->ProcessEvent(Func, nullptr);
        }
        else
        {
            UE_LOG(LogVCCHUD, Warning, TEXT("AddMapContext not found!"));
        }
    }
    else
    {
        UE_LOG(LogVCCHUD, Error, TEXT("Failed to possess MainCharacter!"));
    }
    
    for (const auto& Component : MainRobotConfig.ComponentConfigs)
    {
        if (Component.Get<0>() == ESensorType::RGBCamera)
        {
            if (URGBCameraComponent* RGBCameraComponent =
                MainCharacter->FindComponentByClass<URGBCameraComponent>())
            {
                WidgetInstance->SetRGBContext(
                    RGBCameraComponent->GetRenderTarget(),
                    RGBCameraComponent);
            }
            else
            {
                UE_LOG(LogVCCHUD, Warning, TEXT("RGBDCamera component not found!"));
            }
        }
        if (Component.Get<0>() == ESensorType::DepthCamera)
        {
            if (UDepthCameraComponent* DepthCameraComponent =
                MainCharacter->FindComponentByClass<UDepthCameraComponent>())
            {
                WidgetInstance->SetDepthContext(
                    DepthCameraComponent->GetRenderTarget(), 
                    DepthCameraComponent);
            }
            else
            {
                UE_LOG(LogVCCHUD, Warning, TEXT("DepthCamera component not found!"));
            }
        }
        if (Component.Get<0>() == ESensorType::SegmentationCamera)
        {
            if (USegCameraComponent* SegCameraComponent =
                MainCharacter->FindComponentByClass<USegCameraComponent>())
            {
                WidgetInstance->SetSegContext(
                    SegCameraComponent->GetRenderTarget(),
                    SegCameraComponent);
            }
            else
            {
                UE_LOG(LogVCCHUD, Warning, TEXT("AVCCHUD: "
                                              "RGBCamera component not found!"));
            }
        }
        if (Component.Get<0>() == ESensorType::NormalCamera)
        {
            if (UNormalCameraComponent* NormalCameraComponent =
                MainCharacter->FindComponentByClass<UNormalCameraComponent>())
            {
                WidgetInstance->SetNormalContext(
                    NormalCameraComponent->GetRenderTarget(),
                    NormalCameraComponent);
            }
            else
            {
                UE_LOG(LogVCCHUD, Warning, TEXT("AVCCHUD: "
                                              "NormalCamera component not found!"));
            }
        }
    }
}

FRobotGrpcMaps AVCCHUD::SetupActors(const FVCCSimConfig& Config)
{
    TArray<AActor*> FoundPawns;
    UGameplayStatics::GetAllActorsOfClass(GetWorld(), APawn::StaticClass(), FoundPawns);

    FRobotGrpcMaps RGrpcMaps;
    
    for (const FRobot& Robot : Config.Robots)
    {
        APawn* RobotPawn = Cast<APawn>(FindPawnInTagAndName(Robot.UETag, FoundPawns));
            
        if (!RobotPawn)
        {
            UE_LOG(LogVCCHUD, Warning, TEXT("Robot %s not found! Creating a new one"), *Robot.UETag);
            RobotPawn = CreatePawn(Config, Robot);
            if (!RobotPawn)
            {
                UE_LOG(LogVCCHUD, Error, TEXT("Failed to create Robot %s"), *Robot.UETag);
                continue;
            }
        }

        const FString RobotTagKey = Robot.UETag;
        const std::string RobotTagStd = TCHAR_TO_UTF8(*RobotTagKey);

        if (Robot.Type == EPawnType::Drone)
        {
            RGrpcMaps.RMaps.DroneMap[RobotTagStd] = RobotPawn;
        }
        else if (Robot.Type == EPawnType::Car)
        {
            RGrpcMaps.RMaps.CarMap[RobotTagStd] = RobotPawn;
        }
        else if (Robot.Type == EPawnType::Flash)
        {
            RGrpcMaps.RMaps.FlashMap[RobotTagStd] = RobotPawn;
        }
        else
        {
            UE_LOG(LogVCCHUD, Error, TEXT("AVCCHUD::SetupActors:"
                                        "Unknown pawn type!"));
        }

        if (Robot.RecordInterval > 0)
        {
            if (auto Func = RobotPawn->FindFunction(FName(TEXT("SetRecorder"))))
            {
                RobotPawn->ProcessEvent(Func, &Recorder);
            }
            if (auto Func = RobotPawn->FindFunction(FName(TEXT("SetRecordInterval"))))
            {
                auto RecordInterval = Robot.RecordInterval;
                RobotPawn->ProcessEvent(Func, &RecordInterval);
            }
        }
        
        TSet<ESensorType> SensorTypes;
        
        for (const auto& Component : Robot.ComponentConfigs)
        {
            const FSensorConfig* SensorConfig = Component.Get<1>().Get();
            TArray<UObject*> Objects;

            if (Component.Get<0>() == ESensorType::Lidar)
            {
                if (ULidarComponent* LidarComponent = RobotPawn->FindComponentByClass<ULidarComponent>())
                {
                    LidarComponent->Configure(*SensorConfig);
                    LidarComponent->MeshHolder = Holder->FindComponentByClass<UInsMeshHolder>();
                    RGrpcMaps.RCMaps.LiDARMap[RobotTagStd] = LidarComponent;
                    Objects.Add(LidarComponent);
                }
                else
                {
                    UE_LOG(LogVCCHUD, Warning,
                        TEXT("LidarComponent not found on robot %s"), *Robot.UETag);
                }
            }
            else if (Component.Get<0>() == ESensorType::RGBCamera)
            {
                TArray<URGBCameraComponent*> RGBCameras;
                RobotPawn->GetComponents<URGBCameraComponent>(RGBCameras);
                if (RGBCameras.Num() == 0)
                {
                    UE_LOG(LogVCCHUD, Warning,
                        TEXT("No RGBCameraComponent found on robot %s"), *Robot.UETag);
                    continue;
                }

                for (auto* RGBCam : RGBCameras)
                {
                    RGBCam->Configure(*SensorConfig);
                    FString CameraKey = FString::Printf(TEXT("%s^%d"),
                        *Robot.UETag, RGBCam->GetSensorIndex());
                    RGrpcMaps.RCMaps.RGBMap[TCHAR_TO_UTF8(*CameraKey)] = RGBCam;
                    RGBCam->CameraName = CameraKey;
                    Objects.Add(RGBCam);
                }
            }
            else if (Component.Get<0>() == ESensorType::DepthCamera)
            {
                TArray<UDepthCameraComponent*> DepthCameras;
                RobotPawn->GetComponents<UDepthCameraComponent>(DepthCameras);
                if (DepthCameras.Num() == 0)
                {
                    UE_LOG(LogVCCHUD, Warning,
                        TEXT("No UDepthCameraComponent found on robot %s"), *Robot.UETag);
                    continue;
                }

                for (auto* DepthCam : DepthCameras)
                {
                    DepthCam->Configure(*SensorConfig);
                    FString CameraKey = FString::Printf(TEXT("%s^%d"),
                        *Robot.UETag, DepthCam->GetSensorIndex());
                    RGrpcMaps.RCMaps.DepthMap[TCHAR_TO_UTF8(*CameraKey)] = DepthCam;
                    Objects.Add(DepthCam);
                }
            }
            else if (Component.Get<0>() == ESensorType::SegmentationCamera)
            {
                TArray<USegCameraComponent*> SegmentationCameras;
                RobotPawn->GetComponents<USegCameraComponent>(SegmentationCameras);
                if (SegmentationCameras.Num() == 0)
                {
                    UE_LOG(LogVCCHUD, Warning,
                        TEXT("No SegmentationCameraComponent found on robot %s"), *Robot.UETag);
                    continue;
                }

                for (auto* SegCam : SegmentationCameras)
                {
                    SegCam->Configure(*SensorConfig);
                    FString CameraKey = FString::Printf(TEXT("%s^%d"),
                        *Robot.UETag, SegCam->GetSensorIndex());
                    RGrpcMaps.RCMaps.SegMap[TCHAR_TO_UTF8(*CameraKey)] = SegCam;
                    Objects.Add(SegCam);
                }
            }
            else if (Component.Get<0>() == ESensorType::NormalCamera)
            {
                TArray<UNormalCameraComponent*> NormalCameras;
                RobotPawn->GetComponents<UNormalCameraComponent>(NormalCameras);
                if (NormalCameras.Num() == 0)
                {
                    UE_LOG(LogVCCHUD, Warning,
                        TEXT("No NormalCameraComponent found on robot %s"), *Robot.UETag);
                    continue;
                }

                for (auto* NormalCam : NormalCameras)
                {
                    NormalCam->Configure(*SensorConfig);
                    FString CameraKey = FString::Printf(TEXT("%s^%d"),
                        *Robot.UETag, NormalCam->GetSensorIndex());
                    RGrpcMaps.RCMaps.NormalMap[TCHAR_TO_UTF8(*CameraKey)] = NormalCam;
                    Objects.Add(NormalCam);
                }
            }
            else
            {
                UE_LOG(LogVCCHUD, Warning, TEXT("Unknown sensor %d"),
                    static_cast<int32>(Component.Get<0>()));
            }

            if (SensorConfig && SensorConfig->RecordInterval > 0.0f && Recorder)
            {
                SensorTypes.Add(Component.Get<0>());
                for (UObject* Obj : Objects)
                {
                    Recorder->SensorRegistry.RegisterSensor(Obj);
                }
            }
        }

        if (SensorTypes.Find(ESensorType::Lidar) && SensorTypes.Find(ESensorType::RGBCamera))
        {
            TArray<URGBCameraComponent*> RGBCameras;
            RobotPawn->GetComponents<URGBCameraComponent>(RGBCameras);
            ULidarComponent* LidarComponent = RobotPawn->FindComponentByClass<ULidarComponent>();

            for (auto* RGBCam : RGBCameras)
            {
                RGBCam->SetIgnoreLidar(LidarComponent->MeshHolder);
            }
        }
        Recorder->CreateActorDirectories(RobotPawn->GetName(), std::move(SensorTypes));
    }

    SetupMainCharacter(Config, FoundPawns);
    
    return RGrpcMaps;
}

APawn* AVCCHUD::CreatePawn(const FVCCSimConfig& Config, const FRobot& Robot)
{
    UWorld* World = GetWorld();
    if (!World)
    {
        UE_LOG(LogVCCHUD, Error, TEXT("Failed to get World!"));
        return nullptr;
    }
    
    FActorSpawnParameters SpawnParams;
    SpawnParams.SpawnCollisionHandlingOverride =
       ESpawnActorCollisionHandlingMethod::AdjustIfPossibleButAlwaysSpawn;

    APawn* RobotPawn = nullptr;
    UClass* PawnClass = nullptr;
    if (Robot.Type == EPawnType::Drone)
    {
        PawnClass = LoadClass<APawn>(nullptr, *Config.VCCSim.DefaultDronePawn);
    }
    else if (Robot.Type == EPawnType::Car)
    {
        PawnClass = LoadClass<APawn>(nullptr, *Config.VCCSim.DefaultCarPawn);
    }
    else if (Robot.Type == EPawnType::Flash)
    {
        PawnClass = LoadClass<APawn>(nullptr, *Config.VCCSim.DefaultFlashPawn);
    }
    else
    {
        UE_LOG(LogVCCHUD, Error, TEXT("Unknown pawn type!"));
        return nullptr;
    }
    
    if (PawnClass)
    {
        RobotPawn = World->SpawnActor<APawn>(PawnClass, 
            FVector{0, 0, 10}, FRotator::ZeroRotator, SpawnParams);
        if (!RobotPawn)
        {
            UE_LOG(LogVCCHUD, Error, TEXT("Failed to create Pawn %s"), *Robot.UETag);
            return nullptr;
        }
    }
    else
    {
        UE_LOG(LogVCCHUD, Error, TEXT("Failed to load Drone Blueprint class!"));
        return nullptr;
    }
    
    RobotPawn->Tags.Add(FName(*Robot.UETag));
    return RobotPawn;
}

AActor* AVCCHUD::FindPawnInTagAndName(const FString& Target, TArray<AActor*> FoundPawns)
{
    UGameplayStatics::GetAllActorsOfClass(GetWorld(), APawn::StaticClass(), FoundPawns);

    AActor* Ans = nullptr;

    for (AActor* Actor : FoundPawns)
    {
        if (Actor->ActorHasTag(FName(*Target)))
        {
            Ans = Actor;
            break;
        }
    }
    if (!Ans)
    {
        for (AActor* Actor : FoundPawns)
        {
            if (Actor->GetName().Contains(Target))
            {
                Ans = Actor;
                break;
            }
        }
    }
    return Ans;
}