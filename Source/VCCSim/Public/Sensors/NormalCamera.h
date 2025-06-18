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

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "SensorBase.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Engine/SceneCapture2D.h"
#include "Materials/MaterialInterface.h"
#include "RHIResources.h"
#include "NormalCamera.generated.h"

class ARecorder;

class FNormalCameraConfig: public FSensorConfig
{
public:
    float FOV = 90.0f;
    int32 Width = 1920;
    int32 Height = 1080;
};

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class VCCSIM_API UNormalCameraComponent : public UPrimitiveComponent
{
    GENERATED_BODY()

public:
    UNormalCameraComponent();
    void RConfigure(const FNormalCameraConfig& Config, ARecorder* Recorder);
    bool IsConfigured() const { return bBPConfigured; }
    
    UFUNCTION()
    void SetRecordState(bool RState){ RecordState = RState; }
    
    int32 GetCameraIndex() const { return CameraIndex; }

    void InitializeRenderTargets();
    void SetCaptureComponent() const;

    UFUNCTION(BlueprintCallable, Category = "NormalCamera")
    void CaptureScene();

    // High precision normal data access
    void AsyncGetNormalImageData(TFunction<void(const TArray<FLinearColor>&)> Callback);
    
    std::pair<int32, int32> GetImageSize() const { return {Width, Height}; }
    
protected:
    virtual void BeginPlay() override;
    virtual void OnComponentCreated() override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType,
        FActorComponentTickFunction* ThisTickFunction) override;
    
    void ProcessNormalTexture(TFunction<void(const TArray<FLinearColor>&)> OnComplete);
    TArray<FLinearColor> GetNormalImage();
    
public:
    // Configuration Properties
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NormalCamera|Config")
    float FOV;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NormalCamera|Config")
    int32 Width;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NormalCamera|Config")
    int32 Height;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NormalCamera|Config")
    bool bBPConfigured = false;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "NormalCamera|Config")
    int32 CameraIndex = 0;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "NormalCamera|Debug")
    bool bRecorded = false;

    UPROPERTY()
    UTextureRenderTarget2D* NormalRenderTarget = nullptr;
    
private:
    bool CheckComponentAndRenderTarget() const;
    
    UPROPERTY()
    USceneCaptureComponent2D* CaptureComponent = nullptr;
    
    // Store high precision normal data
    TArray<FLinearColor> NormalData;
    
    UPROPERTY()
    AActor* ParentActor = nullptr;
    
    UPROPERTY()
    ARecorder* RecorderPtr = nullptr;
    
    float RecordInterval = -1.f;
    bool RecordState = false;
    float TimeSinceLastCapture;
    bool Dirty = false;
};
