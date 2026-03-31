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

#include "Utils/NanobananaManager.h"
#include "Utils/VCCSimDataConverter.h"
#include "Utils/VCCSimUIHelpers.h"
#include "Engine/World.h"
#include "TimerManager.h"
#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "IImageWrapperModule.h"
#include "IImageWrapper.h"
#include "Async/Async.h"
#include "Engine/Engine.h"

DEFINE_LOG_CATEGORY_STATIC(LogNanobananaManager, Log, All);

namespace
{
    // Color definitions for material classes
    struct FNBClassDef { const TCHAR* Name; uint8 R, G, B; };
    static const FNBClassDef NBClasses[] = {
        { TEXT("concrete"),     128, 128, 128 }, { TEXT("brick"),        180,  80,  50 },
        { TEXT("glass"),        100, 180, 220 }, { TEXT("metal"),        180, 180, 200 },
        { TEXT("wood"),         140,  90,  40 }, { TEXT("vegetation"),    60, 140,  60 },
        { TEXT("painted_wall"), 220, 210, 200 }, { TEXT("asphalt"),       60,  60,  70 },
    };
    static const int32 NBClassCount = UE_ARRAY_COUNT(NBClasses);
    static const int32 NBTolerance  = 40;
}

FNanobananaManager::FNanobananaManager(UWorld* InWorld) : WorldPtr(InWorld) {}

FNanobananaManager::~FNanobananaManager()
{
    if (WorldPtr.IsValid() && GEngine)
    {
        WorldPtr->GetTimerManager().ClearTimer(TimerHandle);
    }
}

bool FNanobananaManager::RunProjection(
    const FProjectionParams& InParams,
    FOnNanobananaProgress InOnProgress,
    FOnNanobananaComplete InOnComplete)
{
    if (bIsInProgress)
    {
        UE_LOG(LogNanobananaManager, Warning, TEXT("Projection is already in progress."));
        return false;
    }
    if (!WorldPtr.IsValid())
    {
        UE_LOG(LogNanobananaManager, Error, TEXT("World is not valid. Cannot start projection."));
        return false;
    }
    
    // Reset state and store params/delegates
    bIsInProgress = true;
    Params = InParams;
    OnProgress = InOnProgress;
    OnComplete = InOnComplete;
    
    ProcessedRayCount = 0;
    TotalRayCount = 0;
    Votes.Empty();
    ManifestActors.Empty();
    PendingRays.Empty();
    OutputDir.Empty();
    
    StartAsyncDataLoading();
    
    return true;
}

void FNanobananaManager::StartAsyncDataLoading()
{
    OnProgress.ExecuteIfBound(TEXT("Loading data..."), 0, 0);

    FString ManifestStr;
    if (!FFileHelper::LoadFileToString(ManifestStr, *Params.ManifestFile))
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Cannot load manifest.json"), true);
        bIsInProgress = false;
        OnComplete.ExecuteIfBound(TEXT("Error: Cannot load manifest.json"));
        return;
    }

    TSharedPtr<FJsonObject> ManifestRoot;
    TSharedRef<TJsonReader<>> MReader = TJsonReaderFactory<>::Create(ManifestStr);
    if (!FJsonSerializer::Deserialize(MReader, ManifestRoot) || !ManifestRoot.IsValid())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("Failed to parse manifest.json"), true);
        bIsInProgress = false;
        OnComplete.ExecuteIfBound(TEXT("Error: Failed to parse manifest.json"));
        return;
    }

    const TArray<TSharedPtr<FJsonValue>>* ActorsArr = nullptr;
    if (ManifestRoot->TryGetArrayField(TEXT("actors"), ActorsArr))
    {
        for (const TSharedPtr<FJsonValue>& Val : *ActorsArr)
        {
            if (Val->Type != EJson::Object) continue;
            FString Label;
            Val->AsObject()->TryGetStringField(TEXT("label"), Label);
            if (!Label.IsEmpty()) ManifestActors.Add(Label);
        }
    }

    if (ManifestActors.IsEmpty())
    {
        FVCCSimUIHelpers::ShowNotification(TEXT("No actors found in manifest."), true);
        bIsInProgress = false;
        OnComplete.ExecuteIfBound(TEXT("Finished: No actors in manifest."));
        return;
    }

    for (const FString& Label : ManifestActors)
    {
        Votes.Add(Label, TMap<FString, int32>());
    }

    OutputDir = FPaths::GetPath(Params.ManifestFile);
    TWeakPtr<FNanobananaManager> WeakSelf = AsShared();

    // Heavy file I/O and processing on a background thread
    Async(EAsyncExecution::Thread, [this, WeakSelf]()
    {
        TArray<FString> PoseLines;
        FFileHelper::LoadFileToStringArray(PoseLines, *Params.PosesFile);

        TArray<FVCCSimPoseData> Poses;
        for (const FString& Line : PoseLines)
        {
            if (Line.IsEmpty() || Line.StartsWith(TEXT("#"))) continue;
            FVCCSimPoseData P = FVCCSimDataConverter::ParsePoseLine(Line);
            if (!P.Location.IsZero() || !P.Quaternion.IsIdentity()) Poses.Add(P);
        }

        TArray<FString> PngFiles;
        IFileManager::Get().FindFilesRecursive(PngFiles, *Params.ResultDir, TEXT("*.png"), true, false);
        PngFiles.Sort();

        const int32 NumImages = FMath::Min(Poses.Num(), PngFiles.Num());
        const float HFOVRad = FMath::DegreesToRadians(Params.HFOV);
        const float Fx = (Params.ImageWidth * 0.5f) / FMath::Tan(HFOVRad * 0.5f);

        TArray<FNanobananaRay> AllRays;
        AllRays.Reserve(NumImages * NBClassCount * Params.RaysPerClass);

        IImageWrapperModule& IWM = FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));

        for (int32 ImgIdx = 0; ImgIdx < NumImages; ++ImgIdx)
        {
            TArray<uint8> FileBytes;
            if (!FFileHelper::LoadFileToArray(FileBytes, *PngFiles[ImgIdx])) continue;

            TSharedPtr<IImageWrapper> Wrapper = IWM.CreateImageWrapper(EImageFormat::PNG);
            if (!Wrapper.IsValid() || !Wrapper->SetCompressed(FileBytes.GetData(), FileBytes.Num())) continue;
            
            TArray<uint8> Raw;
            if (!Wrapper->GetRaw(ERGBFormat::BGRA, 8, Raw)) continue;

            const int32 W = Wrapper->GetWidth(), H = Wrapper->GetHeight();
            if (W == 0 || H == 0 || Raw.Num() != W * H * 4) continue;
            
            TArray<TArray<FIntPoint>> PixelsByClass;
            PixelsByClass.SetNum(NBClassCount);
            for (int32 y = 0; y < H; ++y)
            {
                for (int32 x = 0; x < W; ++x)
                {
                    const int32 Idx = (y * W + x) * 4;
                    const uint8 Bv = Raw[Idx], Gv = Raw[Idx+1], Rv = Raw[Idx+2];
                    int32 BestClass = -1, BestDist = NBTolerance + 1;
                    for (int32 c = 0; c < NBClassCount; ++c)
                    {
                        const int32 Dist = FMath::Abs((int32)Rv - NBClasses[c].R) + FMath::Abs((int32)Gv - NBClasses[c].G) + FMath::Abs((int32)Bv - NBClasses[c].B);
                        if (Dist < BestDist) { BestDist = Dist; BestClass = c; }
                    }
                    if (BestClass >= 0) PixelsByClass[BestClass].Add(FIntPoint(x,y));
                }
            }
            
            const FVCCSimPoseData& Pose = Poses[ImgIdx];
            for (int32 c = 0; c < NBClassCount; ++c)
            {
                const TArray<FIntPoint>& Pixels = PixelsByClass[c];
                if (Pixels.IsEmpty()) continue;

                const int32 Total = Pixels.Num();
                const int32 N = FMath::Min(Total, Params.RaysPerClass);
                const int32 Stride = FMath::Max(1, Total / N);

                for (int32 s = 0; s < N; ++s)
                {
                    const FIntPoint& Px = Pixels[FMath::Min(s * Stride, Total - 1)];
                    const float u = Px.X + 0.5f, v = Px.Y + 0.5f;
                    const FVector DirCam(1.f, (u - Params.ImageWidth * 0.5f) / Fx, -(v - Params.ImageHeight * 0.5f) / Fx);
                    FNanobananaRay Ray;
                    Ray.Origin = Pose.Location;
                    Ray.Direction = Pose.Quaternion.RotateVector(DirCam).GetSafeNormal();
                    Ray.ClassName = FString(NBClasses[c].Name);
                    AllRays.Add(MoveTemp(Ray));
                }
            }
        }
        
        UE_LOG(LogNanobananaManager, Log, TEXT("Generated %d rays from %d images"), AllRays.Num(), NumImages);

        // Switch back to game thread to start the ray casting timer
        AsyncTask(ENamedThreads::GameThread, [WeakSelf, AllRays = MoveTemp(AllRays)]() mutable {
            TSharedPtr<FNanobananaManager> StrongSelf = WeakSelf.Pin();
            if (!StrongSelf.IsValid()) return;
            
            StrongSelf->PendingRays = MoveTemp(AllRays);
            StrongSelf->TotalRayCount = StrongSelf->PendingRays.Num();

            if (StrongSelf->TotalRayCount == 0)
            {
                StrongSelf->FinalizeProjection();
                return;
            }

            StrongSelf->OnProgress.ExecuteIfBound(FString::Printf(TEXT("Casting %d rays..."), StrongSelf->TotalRayCount), 0, StrongSelf->TotalRayCount);
            
            FTimerDelegate Delegate = FTimerDelegate::CreateSP(StrongSelf.Get(), &FNanobananaManager::TickRayCasting);
            StrongSelf->WorldPtr->GetTimerManager().SetTimer(StrongSelf->TimerHandle, Delegate, 0.05f, true);
        });
    });
}

void FNanobananaManager::TickRayCasting()
{
    if (!WorldPtr.IsValid())
    {
        UE_LOG(LogNanobananaManager, Error, TEXT("World became invalid during ray casting."));
        WorldPtr->GetTimerManager().ClearTimer(TimerHandle);
        FinalizeProjection();
        return;
    }

    static const int32 BatchSize = 200;
    const int32 End = FMath::Min(ProcessedRayCount + BatchSize, TotalRayCount);

    FCollisionQueryParams QParams;
    QParams.bTraceComplex = false;

    for (int32 i = ProcessedRayCount; i < End; ++i)
    {
        const FNanobananaRay& Ray = PendingRays[i];
        FHitResult Hit;
        if (WorldPtr->LineTraceSingleByChannel(Hit, Ray.Origin, Ray.Origin + Ray.Direction * 100000.f, ECC_Visibility, QParams))
        {
            if (AActor* HitActor = Hit.GetActor())
            {
                if (TMap<FString, int32>* ActorVotes = Votes.Find(HitActor->GetActorLabel()))
                {
                    ++ActorVotes->FindOrAdd(Ray.ClassName, 0);
                }
            }
        }
    }

    ProcessedRayCount = End;
    OnProgress.ExecuteIfBound(FString::Printf(TEXT("Casting rays...")), ProcessedRayCount, TotalRayCount);

    if (ProcessedRayCount >= TotalRayCount)
    {
        WorldPtr->GetTimerManager().ClearTimer(TimerHandle);
        FinalizeProjection();
    }
}

void FNanobananaManager::FinalizeProjection()
{
    const int32 TotalActors = ManifestActors.Num();
    int32 LabeledActors = 0, NotHitActors = 0, MultiClassConflicts = 0;

    TSharedPtr<FJsonObject> RootJson = MakeShareable(new FJsonObject);
    TArray<TSharedPtr<FJsonValue>> ActorResults;

    for (const FString& Label : ManifestActors)
    {
        TMap<FString, int32>* VoteMap = Votes.Find(Label);
        TSharedPtr<FJsonObject> ActorJson = MakeShareable(new FJsonObject);
        ActorJson->SetStringField(TEXT("label"), Label);

        if (!VoteMap || VoteMap->IsEmpty())
        {
            ++NotHitActors;
            ActorJson->SetStringField(TEXT("class"), TEXT(""));
            ActorJson->SetNumberField(TEXT("votes"), 0);
            UE_LOG(LogNanobananaManager, Warning, TEXT("Actor %-30s -> NOT HIT (0 rays)"), *Label);
        }
        else
        {
            ++LabeledActors;
            FString BestClass;
            int32 BestVotes = 0;
            for (const auto& Pair : *VoteMap)
            {
                if (Pair.Value > BestVotes) { BestVotes = Pair.Value; BestClass = Pair.Key; }
            }

            TSharedPtr<FJsonObject> DistJson = MakeShareable(new FJsonObject);
            FString DistStr;
            bool bConflict = false;
            for (const auto& Pair : *VoteMap)
            {
                DistJson->SetNumberField(Pair.Key, Pair.Value);
                if (Pair.Key != BestClass && Pair.Value * 10 > BestVotes) bConflict = true;
                DistStr += FString::Printf(TEXT("%s:%d, "), *Pair.Key, Pair.Value);
            }
            if (bConflict) ++MultiClassConflicts;

            ActorJson->SetStringField(TEXT("class"), BestClass);
            ActorJson->SetNumberField(TEXT("votes"), BestVotes);
            ActorJson->SetObjectField(TEXT("vote_distribution"), DistJson);
            UE_LOG(LogNanobananaManager, Log, TEXT("Actor %-30s -> %-15s (%d votes | %s)"), *Label, *BestClass, BestVotes, *DistStr);
        }
        ActorResults.Add(MakeShareable(new FJsonValueObject(ActorJson)));
    }

    UE_LOG(LogNanobananaManager, Log, TEXT("--- Nanobanana Final Stats ---"));
    UE_LOG(LogNanobananaManager, Log, TEXT("Total actors in manifest : %d"), TotalActors);
    UE_LOG(LogNanobananaManager, Log, TEXT("Labeled (>=1 vote)       : %d"), LabeledActors);
    UE_LOG(LogNanobananaManager, Log, TEXT("Not hit                  : %d"), NotHitActors);
    UE_LOG(LogNanobananaManager, Log, TEXT("Multi-class conflict >10%%: %d"), MultiClassConflicts);

    TSharedPtr<FJsonObject> StatsJson = MakeShareable(new FJsonObject);
    StatsJson->SetNumberField(TEXT("total_actors"), TotalActors);
    StatsJson->SetNumberField(TEXT("labeled"), LabeledActors);
    StatsJson->SetNumberField(TEXT("not_hit"), NotHitActors);
    StatsJson->SetNumberField(TEXT("multi_class_conflict"), MultiClassConflicts);

    RootJson->SetArrayField(TEXT("actors"), ActorResults);
    RootJson->SetObjectField(TEXT("stats"), StatsJson);

    FString JsonStr;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&JsonStr);
    FJsonSerializer::Serialize(RootJson.ToSharedRef(), Writer);

    const FString OutPath = OutputDir / TEXT("label_assignment.json");
    if(FFileHelper::SaveStringToFile(JsonStr, *OutPath))
    {
        UE_LOG(LogNanobananaManager, Log, TEXT("Saved results to -> %s"), *OutPath);
    }
    else
    {
        UE_LOG(LogNanobananaManager, Error, TEXT("Failed to save results to -> %s"), *OutPath);
    }
    
    bIsInProgress = false;
    const FString FinalStatus = FString::Printf(TEXT("Done: %d/%d labeled, %d not hit"), LabeledActors, TotalActors, NotHitActors);
    OnComplete.ExecuteIfBound(FinalStatus);
}
