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

#include "Utils/CaptureReuseManifest.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/FileManager.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonWriter.h"

DEFINE_LOG_CATEGORY_STATIC(LogCaptureReuseManifest, Log, All);

namespace
{
    const TCHAR* ManifestFileName = TEXT("reuse.json");

    FString ReadOwner(const TSharedPtr<FJsonObject>& Group)
    {
        FString Owner;
        if (Group.IsValid() && Group->TryGetStringField(TEXT("owner"), Owner))
        {
            return Owner;   // JSON null / missing leaves Owner empty (== owns)
        }
        return FString();
    }
}

FCaptureReuseManifest FCaptureReuseManifest::Load(const FString& InCapturesRoot)
{
    FCaptureReuseManifest Manifest;
    Manifest.CapturesRoot = InCapturesRoot;

    FString JsonStr;
    if (!FFileHelper::LoadFileToString(JsonStr, *(InCapturesRoot / ManifestFileName)))
    {
        return Manifest;   // missing → empty manifest
    }

    TSharedPtr<FJsonObject> Root;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonStr);
    if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
    {
        UE_LOG(LogCaptureReuseManifest, Warning, TEXT("reuse.json is unparseable; treating as empty"));
        return Manifest;
    }

    const TSharedPtr<FJsonObject>* Captures = nullptr;
    if (!Root->TryGetObjectField(TEXT("captures"), Captures) || !Captures->IsValid())
    {
        return Manifest;
    }

    for (const auto& Pair : (*Captures)->Values)
    {
        const TSharedPtr<FJsonObject> Obj = Pair.Value.IsValid() ? Pair.Value->AsObject() : nullptr;
        if (!Obj.IsValid()) continue;

        FCaptureReuseEntry Entry;
        Obj->TryGetStringField(TEXT("pose_key"), Entry.PoseKey);
        Obj->TryGetStringField(TEXT("scene_key"), Entry.SceneKey);
        Obj->TryGetStringField(TEXT("gt_materials_key"), Entry.GtMaterialsKey);

        const TSharedPtr<FJsonObject>* ViewGt = nullptr;
        if (Obj->TryGetObjectField(TEXT("view_gt"), ViewGt))
        {
            Entry.ViewGtOwner = ReadOwner(*ViewGt);
        }
        const TSharedPtr<FJsonObject>* GtMat = nullptr;
        if (Obj->TryGetObjectField(TEXT("gt_materials"), GtMat))
        {
            Entry.GtMaterialsOwner = ReadOwner(*GtMat);
        }

        Manifest.Entries.Add(Pair.Key, Entry);
    }

    return Manifest;
}

bool FCaptureReuseManifest::Save() const
{
    TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetNumberField(TEXT("version"), 1);

    TSharedRef<FJsonObject> Captures = MakeShared<FJsonObject>();
    for (const auto& Pair : Entries)
    {
        const FCaptureReuseEntry& E = Pair.Value;
        TSharedRef<FJsonObject> Obj = MakeShared<FJsonObject>();
        Obj->SetStringField(TEXT("pose_key"), E.PoseKey);
        Obj->SetStringField(TEXT("scene_key"), E.SceneKey);
        Obj->SetStringField(TEXT("gt_materials_key"), E.GtMaterialsKey);

        TSharedRef<FJsonObject> ViewGt = MakeShared<FJsonObject>();
        if (E.ViewGtOwner.IsEmpty()) ViewGt->SetField(TEXT("owner"), MakeShared<FJsonValueNull>());
        else                         ViewGt->SetStringField(TEXT("owner"), E.ViewGtOwner);
        Obj->SetObjectField(TEXT("view_gt"), ViewGt);

        TSharedRef<FJsonObject> GtMat = MakeShared<FJsonObject>();
        if (E.GtMaterialsOwner.IsEmpty()) GtMat->SetField(TEXT("owner"), MakeShared<FJsonValueNull>());
        else                              GtMat->SetStringField(TEXT("owner"), E.GtMaterialsOwner);
        Obj->SetObjectField(TEXT("gt_materials"), GtMat);

        Captures->SetObjectField(Pair.Key, Obj);
    }
    Root->SetObjectField(TEXT("captures"), Captures);

    FString JsonStr;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&JsonStr);
    if (!FJsonSerializer::Serialize(Root, Writer))
    {
        return false;
    }

    const FString FinalPath = CapturesRoot / ManifestFileName;
    const FString TempPath = FinalPath + TEXT(".tmp");
    if (!FFileHelper::SaveStringToFile(JsonStr, *TempPath))
    {
        return false;
    }
    IFileManager& FM = IFileManager::Get();
    FM.Delete(*FinalPath, false, true, true);
    return FM.Move(*FinalPath, *TempPath, true);
}

FString FCaptureReuseManifest::FindViewGtOwner(const FString& PoseKey, const FString& SceneKey) const
{
    if (PoseKey.IsEmpty() || SceneKey.IsEmpty())
    {
        return FString();
    }
    for (const auto& Pair : Entries)
    {
        const FCaptureReuseEntry& E = Pair.Value;
        if (E.ViewGtOwner.IsEmpty()                                  // it owns the files
            && E.PoseKey == PoseKey && E.SceneKey == SceneKey
            && IFileManager::Get().DirectoryExists(*(CapturesRoot / Pair.Key)))
        {
            return Pair.Key;
        }
    }
    return FString();
}

FString FCaptureReuseManifest::FindGtMaterialsOwner(const FString& GtMaterialsKey) const
{
    if (GtMaterialsKey.IsEmpty())
    {
        return FString();
    }
    for (const auto& Pair : Entries)
    {
        const FCaptureReuseEntry& E = Pair.Value;
        if (E.GtMaterialsOwner.IsEmpty()
            && E.GtMaterialsKey == GtMaterialsKey
            && IFileManager::Get().DirectoryExists(*(CapturesRoot / Pair.Key / TEXT("gt_materials"))))
        {
            return Pair.Key;
        }
    }
    return FString();
}

void FCaptureReuseManifest::AddOrUpdate(const FString& CaptureName, const FCaptureReuseEntry& Entry)
{
    Entries.Add(CaptureName, Entry);
}

void FCaptureReuseManifest::Remove(const FString& CaptureName)
{
    Entries.Remove(CaptureName);
}
