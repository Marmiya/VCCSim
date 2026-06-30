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

#include "Utils/CaptureSessionCheckpoint.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/FileManager.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonWriter.h"

DEFINE_LOG_CATEGORY_STATIC(LogCaptureSessionCheckpoint, Log, All);

namespace
{
    const TCHAR* CheckpointFileName = TEXT("capture_session.json");
}

FCaptureSessionCheckpoint FCaptureSessionCheckpoint::Load(const FString& InCapturesRoot)
{
    FCaptureSessionCheckpoint Checkpoint;
    Checkpoint.CapturesRoot = InCapturesRoot;

    FString JsonStr;
    if (!FFileHelper::LoadFileToString(JsonStr, *(InCapturesRoot / CheckpointFileName)))
    {
        return Checkpoint;   // missing → empty (!IsValid)
    }

    TSharedPtr<FJsonObject> Root;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonStr);
    if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
    {
        UE_LOG(LogCaptureSessionCheckpoint, Warning,
            TEXT("capture_session.json is unparseable; treating as no resumable capture"));
        return Checkpoint;
    }

    Root->TryGetStringField(TEXT("batch_timestamp"), Checkpoint.BatchTimestamp);
    Root->TryGetStringField(TEXT("pose_key"), Checkpoint.PoseKey);
    Root->TryGetStringField(TEXT("scene_key"), Checkpoint.SceneKey);
    Root->TryGetBoolField(TEXT("output_mesh"), Checkpoint.bOutputMesh);
    double TexRes = 0.0;
    if (Root->TryGetNumberField(TEXT("gt_texture_resolution"), TexRes)) Checkpoint.GTTextureResolution = (int32)TexRes;
    Root->TryGetBoolField(TEXT("use_capture_reuse"), Checkpoint.bUseCaptureReuse);
    Root->TryGetStringField(TEXT("scene_name"), Checkpoint.SceneName);

    const TArray<TSharedPtr<FJsonValue>>* Labels = nullptr;
    if (Root->TryGetArrayField(TEXT("target_labels"), Labels) && Labels)
    {
        for (const TSharedPtr<FJsonValue>& V : *Labels)
        {
            FString L;
            if (V.IsValid() && V->TryGetString(L) && !L.IsEmpty()) Checkpoint.TargetLabels.Add(L);
        }
    }

    // Path: a flat number array, 6 values per pose (x,y,z,pitch,yaw,roll).
    const TArray<TSharedPtr<FJsonValue>>* Path = nullptr;
    if (Root->TryGetArrayField(TEXT("path"), Path) && Path)
    {
        const int32 Count = Path->Num() / 6;
        Checkpoint.PathPositions.Reserve(Count);
        Checkpoint.PathRotations.Reserve(Count);
        for (int32 i = 0; i + 5 < Path->Num(); i += 6)
        {
            const double X  = (*Path)[i + 0]->AsNumber();
            const double Y  = (*Path)[i + 1]->AsNumber();
            const double Z  = (*Path)[i + 2]->AsNumber();
            const double Pi = (*Path)[i + 3]->AsNumber();
            const double Yw = (*Path)[i + 4]->AsNumber();
            const double Rl = (*Path)[i + 5]->AsNumber();
            Checkpoint.PathPositions.Add(FVector(X, Y, Z));
            Checkpoint.PathRotations.Add(FRotator(Pi, Yw, Rl));
        }
    }

    const TArray<TSharedPtr<FJsonValue>>* Windows = nullptr;
    if (Root->TryGetArrayField(TEXT("windows"), Windows) && Windows)
    {
        for (const TSharedPtr<FJsonValue>& V : *Windows)
        {
            const TSharedPtr<FJsonObject> Obj = V.IsValid() ? V->AsObject() : nullptr;
            if (!Obj.IsValid()) continue;

            FCaptureWindow W;
            double SlotN = -1.0;
            if (Obj->TryGetNumberField(TEXT("slot"), SlotN)) W.Slot = (int32)SlotN;
            double Elev = 0.0, Az = 0.0;
            if (Obj->TryGetNumberField(TEXT("elevation"), Elev)) W.Elevation = (float)Elev;
            if (Obj->TryGetNumberField(TEXT("azimuth"), Az))     W.Azimuth   = (float)Az;
            Obj->TryGetStringField(TEXT("dir"), W.DirName);
            Obj->TryGetBoolField(TEXT("rgb_only"), W.bRgbOnly);
            if (!W.DirName.IsEmpty()) Checkpoint.Windows.Add(W);
        }
    }

    return Checkpoint;
}

bool FCaptureSessionCheckpoint::Save() const
{
    TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetNumberField(TEXT("version"), 1);
    Root->SetStringField(TEXT("batch_timestamp"), BatchTimestamp);
    Root->SetStringField(TEXT("pose_key"), PoseKey);
    Root->SetStringField(TEXT("scene_key"), SceneKey);
    Root->SetBoolField(TEXT("output_mesh"), bOutputMesh);
    Root->SetNumberField(TEXT("gt_texture_resolution"), GTTextureResolution);
    Root->SetBoolField(TEXT("use_capture_reuse"), bUseCaptureReuse);
    Root->SetStringField(TEXT("scene_name"), SceneName);

    TArray<TSharedPtr<FJsonValue>> LabelArray;
    for (const FString& L : TargetLabels)
    {
        LabelArray.Add(MakeShared<FJsonValueString>(L));
    }
    Root->SetArrayField(TEXT("target_labels"), LabelArray);

    // Path: flat number array, 6 values per pose (x,y,z,pitch,yaw,roll), so a resume can restore the
    // FlashPawn path even if the level was never saved before a crash.
    TArray<TSharedPtr<FJsonValue>> PathArray;
    PathArray.Reserve(PathPositions.Num() * 6);
    const int32 N = FMath::Min(PathPositions.Num(), PathRotations.Num());
    for (int32 i = 0; i < N; ++i)
    {
        const FVector& P = PathPositions[i];
        const FRotator& R = PathRotations[i];
        PathArray.Add(MakeShared<FJsonValueNumber>(P.X));
        PathArray.Add(MakeShared<FJsonValueNumber>(P.Y));
        PathArray.Add(MakeShared<FJsonValueNumber>(P.Z));
        PathArray.Add(MakeShared<FJsonValueNumber>(R.Pitch));
        PathArray.Add(MakeShared<FJsonValueNumber>(R.Yaw));
        PathArray.Add(MakeShared<FJsonValueNumber>(R.Roll));
    }
    Root->SetArrayField(TEXT("path"), PathArray);

    TArray<TSharedPtr<FJsonValue>> WindowArray;
    for (const FCaptureWindow& W : Windows)
    {
        TSharedRef<FJsonObject> Obj = MakeShared<FJsonObject>();
        Obj->SetNumberField(TEXT("slot"), W.Slot);
        Obj->SetNumberField(TEXT("elevation"), W.Elevation);
        Obj->SetNumberField(TEXT("azimuth"), W.Azimuth);
        Obj->SetStringField(TEXT("dir"), W.DirName);
        Obj->SetBoolField(TEXT("rgb_only"), W.bRgbOnly);
        WindowArray.Add(MakeShared<FJsonValueObject>(Obj));
    }
    Root->SetArrayField(TEXT("windows"), WindowArray);

    FString JsonStr;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&JsonStr);
    if (!FJsonSerializer::Serialize(Root, Writer))
    {
        return false;
    }

    const FString FinalPath = CapturesRoot / CheckpointFileName;
    const FString TempPath = FinalPath + TEXT(".tmp");
    IFileManager& FM = IFileManager::Get();
    FM.MakeDirectory(*CapturesRoot, true);
    if (!FFileHelper::SaveStringToFile(JsonStr, *TempPath))
    {
        return false;
    }
    FM.Delete(*FinalPath, false, true, true);
    return FM.Move(*FinalPath, *TempPath, true);
}

bool FCaptureSessionCheckpoint::Exists(const FString& CapturesRoot)
{
    return IFileManager::Get().FileExists(*(CapturesRoot / CheckpointFileName));
}

void FCaptureSessionCheckpoint::Clear(const FString& CapturesRoot)
{
    IFileManager::Get().Delete(*(CapturesRoot / CheckpointFileName), false, true, true);
}

void FCaptureSessionCheckpoint::SetWindowRgbOnly(const FString& DirName, bool bInRgbOnly)
{
    for (FCaptureWindow& W : Windows)
    {
        if (W.DirName == DirName)
        {
            W.bRgbOnly = bInRgbOnly;
            return;
        }
    }
}
