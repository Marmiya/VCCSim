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

#include "DataStruct_IO/CameraData.h"
#include "Misc/FileHelper.h"
#include "HAL/PlatformFilemanager.h"
#include "Math/UnrealMathUtility.h"

namespace FCameraInfoUtils
{
    FCameraInfo CreateCameraInfo(
        int32 UID,
        const FVector& Position,
        const FQuat& Rotation,
        const FString& ImagePath,
        const FCameraIntrinsics& Intrinsics)
    {
        FCameraInfo CameraInfo;
        
        CameraInfo.UID = UID;
        CameraInfo.Position = Position;
        CameraInfo.Rotation = Rotation;
        CameraInfo.ImagePath = ImagePath;
        CameraInfo.ImageName = FPaths::GetCleanFilename(ImagePath);
        
        // Set intrinsics
        CameraInfo.Width = Intrinsics.Width;
        CameraInfo.Height = Intrinsics.Height;
        CameraInfo.FocalX = Intrinsics.FocalX;
        CameraInfo.FocalY = Intrinsics.FocalY;
        CameraInfo.CenterX = Intrinsics.CenterX;
        CameraInfo.CenterY = Intrinsics.CenterY;
        
        // Calculate FOV from focal length (use horizontal FOV)
        if (Intrinsics.FocalX > 0 && Intrinsics.Width > 0)
        {
            CameraInfo.FOVDegrees = 2.0f * FMath::Atan(Intrinsics.Width * 0.5f / Intrinsics.FocalX) * 180.0f / PI;
        }
        else
        {
            CameraInfo.FOVDegrees = 90.0f; // Default FOV
        }
        
        return CameraInfo;
    }

    FMatrix GetRotationMatrix(const FCameraInfo& CameraInfo)
    {
        return CameraInfo.Rotation.ToMatrix();
    }

    void SetRotationFromMatrix(FCameraInfo& CameraInfo, const FMatrix& RotationMatrix)
    {
        CameraInfo.Rotation = RotationMatrix.ToQuat();
        CameraInfo.Rotation.Normalize();
    }

    bool SaveCameraInfoToFile(
        const TArray<FCameraInfo>& CameraInfos,
        const FString& FilePath,
        const FString& CoordinateSystemDescription)
    {
        FString OutputText;
        OutputText += TEXT("# VCCSim CameraInfo Data Export\n");
        OutputText += TEXT("# Format: ImageName X Y Z QW QX QY QZ\n");
        OutputText += FString::Printf(TEXT("# Coordinate System: %s\n"), *CoordinateSystemDescription);
        OutputText += FString::Printf(TEXT("# Total Cameras: %d\n\n"), CameraInfos.Num());
        
        for (const FCameraInfo& Camera : CameraInfos)
        {
            // Extract image name from full path
            FString ImageName = FPaths::GetCleanFilename(Camera.ImagePath);
            if (ImageName.IsEmpty())
            {
                ImageName = Camera.ImageName;
            }
            
            // Use position and rotation quaternion
            const FVector& Position = Camera.Position;
            const FQuat& Rotation = Camera.Rotation;
            
            OutputText += FString::Printf(TEXT("%s %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n"),
                *ImageName,
                Position.X, Position.Y, Position.Z,
                Rotation.W, Rotation.X, Rotation.Y, Rotation.Z);
        }
        
        // Write to file
        if (!FFileHelper::SaveStringToFile(OutputText, *FilePath))
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to save CameraInfo data to file: %s"), *FilePath);
            return false;
        }
        
        UE_LOG(LogTemp, Log, TEXT("Successfully saved CameraInfo data to: %s"), *FilePath);
        return true;
    }

    bool LoadCameraInfoFromFile(TArray<FCameraInfo>& OutCameraInfos, const FString& FilePath)
    {
        FString FileContent;
        if (!FFileHelper::LoadFileToString(FileContent, *FilePath))
        {
            UE_LOG(LogTemp, Error, TEXT("Failed to load CameraInfo file: %s"), *FilePath);
            return false;
        }
        
        OutCameraInfos.Reset();
        
        TArray<FString> Lines;
        FileContent.ParseIntoArrayLines(Lines);
        
        for (const FString& Line : Lines)
        {
            FString TrimmedLine = Line.TrimStartAndEnd();
            if (TrimmedLine.IsEmpty() || TrimmedLine.StartsWith(TEXT("#")))
            {
                continue;
            }
            
            TArray<FString> Parts;
            TrimmedLine.ParseIntoArray(Parts, TEXT(" "), true);
            
            if (Parts.Num() < 8)
            {
                UE_LOG(LogTemp, Warning, TEXT("Invalid line in CameraInfo file: %s"), *TrimmedLine);
                continue;
            }
            
            FCameraInfo CameraInfo;
            
            try
            {
                CameraInfo.ImageName = Parts[0];
                CameraInfo.ImagePath = Parts[0]; // Set same as name for now
                
                CameraInfo.Position.X = FCString::Atof(*Parts[1]);
                CameraInfo.Position.Y = FCString::Atof(*Parts[2]);
                CameraInfo.Position.Z = FCString::Atof(*Parts[3]);
                
                CameraInfo.Rotation.W = FCString::Atof(*Parts[4]);
                CameraInfo.Rotation.X = FCString::Atof(*Parts[5]);
                CameraInfo.Rotation.Y = FCString::Atof(*Parts[6]);
                CameraInfo.Rotation.Z = FCString::Atof(*Parts[7]);
                
                // Normalize quaternion
                CameraInfo.Rotation.Normalize();
                
                OutCameraInfos.Add(CameraInfo);
            }
            catch (...)
            {
                UE_LOG(LogTemp, Warning, TEXT("Failed to parse CameraInfo line: %s"), *TrimmedLine);
                continue;
            }
        }
        
        UE_LOG(LogTemp, Log, TEXT("Successfully loaded %d cameras from: %s"), OutCameraInfos.Num(), *FilePath);
        return OutCameraInfos.Num() > 0;
    }

    float CalculateAngularDistance(const FCameraInfo& Camera1, const FCameraInfo& Camera2)
    {
        FQuat Q1 = Camera1.Rotation;
        FQuat Q2 = Camera2.Rotation;
        
        // Normalize quaternions
        Q1.Normalize();
        Q2.Normalize();
        
        // Handle quaternion double cover (q and -q represent same rotation)
        float Dot = FMath::Abs(Q1 | Q2);
        Dot = FMath::Clamp(Dot, 0.0f, 1.0f);
        
        // Convert to angle in degrees
        float AngleRad = 2.0f * FMath::Acos(Dot);
        float AngleDeg = FMath::RadiansToDegrees(AngleRad);
        
        return AngleDeg;
    }

    float CalculatePositionDistance(const FCameraInfo& Camera1, const FCameraInfo& Camera2)
    {
        return FVector::Dist(Camera1.Position, Camera2.Position);
    }
}