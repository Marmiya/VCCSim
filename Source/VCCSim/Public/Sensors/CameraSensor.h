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
#include "CameraSensor.generated.h"

class ARecorder;
class UInsMeshHolder;

class FRGBCameraConfig : public FSensorConfig
{
public:
    float FOV = 90.0f;
    int32 Width = 512;
    int32 Height = 512;
    bool bOrthographic = false;
    float OrthoWidth = 512.0f;
};

DECLARE_DYNAMIC_DELEGATE_TwoParams(
    FKeyPointCaptured, const FTransform&, Pose, const FString&, Name);

UCLASS(ClassGroup = (VCCSIM), meta = (BlueprintSpawnableComponent))
class VCCSIM_API URGBCameraComponent : public UPrimitiveComponent
{
    GENERATED_BODY()

public:
    URGBCameraComponent();
    void RConfigure(const FRGBCameraConfig& Config, ARecorder* Recorder);
    bool IsConfigured() const { return bBPConfigured; }
    void SetIgnoreLidar(UInsMeshHolder* MeshHolder);
    UFUNCTION()
    void SetRecordState(bool RState) { RecordState = RState; }
    
    int32 GetCameraIndex() const { return CameraIndex; }
    FString CameraName;
    
    void SetCaptureComponent() const;
    void InitializeRenderTargets();
    
    void ProcessRGBTextureAsyncRaw(TFunction<void()> OnComplete);
    void ProcessRGBTextureAsync(TFunction<void(const TArray<FColor>&)> OnComplete);

    UFUNCTION(BlueprintCallable, Category = "RGBCamera")
    void CaptureRGBScene();

    void ComputeIntrinsics();
    FMatrix44f GetCameraIntrinsics() const { return CameraIntrinsics; }
    // For GRPC call
    void AsyncGetRGBImageData(TFunction<void(const TArray<FColor>&)> Callback);
    std::pair<int32, int32> GetImageSize() const { return {Width, Height}; }
        
protected:
    virtual void BeginPlay() override;
    virtual void OnComponentCreated() override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType,
        FActorComponentTickFunction* ThisTickFunction) override;

public:
    // Configuration Properties
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RGBCamera|Config")
    float FOV;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RGBCamera|Config")
    int32 Width;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RGBCamera|Config")
    int32 Height;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RGBCamera|Config")
    bool bOrthographic;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RGBCamera|Config", 
        meta = (EditCondition = "bOrthographic"))
    float OrthoWidth;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RGBCamera|Config")
    bool bBPConfigured = false;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RGBCamera|Config")
    int32 CameraIndex = 0;

    FKeyPointCaptured
    OnKeyPointCaptured;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "RGBCamera|Debug")
    bool bRecorded = false;
    
    UPROPERTY()
    UTextureRenderTarget2D* RGBRenderTarget = nullptr;
    
private:
    bool CheckComponentAndRenderTarget() const;
    void ExecuteCaptureOnGameThread();

    struct FReadSurfaceContext
    {
        TArray<FColor>* OutData;
        FTextureRenderTargetResource* RenderTarget;
        FIntRect Rect;
        FReadSurfaceDataFlags Flags;
    };

    FMatrix44f CameraIntrinsics;
    
    UPROPERTY()
    USceneCaptureComponent2D* CaptureComponent = nullptr;
    TArray<FColor> RGBData;

    UPROPERTY()
    AActor* ParentActor = nullptr;
    UPROPERTY()
    ARecorder* RecorderPtr = nullptr;
    bool RecordState = false;
    float RecordInterval = -1.f;
    float TimeSinceLastCapture;

    bool Dirty = false;
};