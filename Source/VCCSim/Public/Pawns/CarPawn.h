﻿#pragma once
#include "CoreMinimal.h"
#include "CarPawn.generated.h"

class AVCCSimPath;

UCLASS(HideCategories=("Animation"))
class VCCSIM_API ACarPawn : public APawn
{
	GENERATED_BODY()
public:

	ACarPawn();
	virtual void OnConstruction(const FTransform& Transform) override;
	virtual void Tick(float DeltaSeconds) override;
	
	void SetTraceIgnores();
	void CarMeshSetup();
	void CalculateDistance();
	void FollowThePathAndSteer();
	void ActorGroundTrace();
	void InitialState();

	void WheelsGroundTrace();
	void ChassisRollandPitch();
	void ChassisRelocation();
	
	void WheelsDrive();
	void Vibrations();
	void CalculateSpeed();

	UFUNCTION(BlueprintCallable, Category="Default")
	void AutoMove(double DeltaSec);

	double PerlinNoise(double TimeOffset, double Speed,
		double Amount, double Multiplier);
	double CalculateWheelRotation(USceneComponent* WheelMesh, double Offset);
	void SingleWheelVibration(USceneComponent* WheelMesh, double Offset);
	double WheelVibration(double TimeOffset);
	
public:
	UPROPERTY(BlueprintReadWrite, EditInstanceOnly, Category="Hidden")
	TArray<AActor*> ActorsToIgnore;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="Default")
	TObjectPtr<UChildActorComponent> Chassis;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="Default")
	TObjectPtr<UChildActorComponent> BodyAndAccessories;
	
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="Default")
	TObjectPtr<UStaticMeshComponent> Body;

	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="Default")
	TObjectPtr<UChildActorComponent> Wheels;
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="Default")
	TObjectPtr<UStaticMeshComponent> WheelFL;
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="Default")
	TObjectPtr<UStaticMeshComponent> WheelFR;
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="Default")
	TObjectPtr<UStaticMeshComponent> WheelRL;
	UPROPERTY(BlueprintReadOnly, VisibleAnywhere, Category="Default")
	TObjectPtr<UStaticMeshComponent> WheelRR;	

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly, Category="Car Setup")
	TObjectPtr<UStaticMesh> BodyMesh;
    UPROPERTY(BlueprintReadWrite, EditDefaultsOnly, Category="Car Setup")
	TObjectPtr<UStaticMesh> FrontWheelsMesh;
	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly, Category="Car Setup")
	TObjectPtr<UStaticMesh> RearWheelsMesh;

	UPROPERTY(BlueprintReadWrite, EditInstanceOnly, Category="Driving", Interp)
	TObjectPtr<AVCCSimPath> Path;

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Driving", Interp, meta=(UIMin="0", ClampMin="0"))
	float DistanceTraveled;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Driving")
	bool IfAutoMove;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Car Setup")
	bool TraceGround;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Car Setup")
	bool TrackPath = true;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Car Setup", meta=(UIMin="0", ClampMin="0"))
	float TickFPS = 60;
	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly, Category="Car Setup", meta=(UIMin="0", UIMax="400", ClampMax="400", ClampMin="0"))
	double FrontAxleOffset = 0;
	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly, Category="Car Setup", meta=(UIMin="0", UIMax="400", ClampMax="400", ClampMin="0"))
	double RearAxleOffset = 0;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Driving", Interp, meta=(UIMin="-45", UIMax="45", ClampMax="45", ClampMin="-45"))
	double Steering = 0.;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Suspension", Interp, meta=(UIMin="0", UIMax="10", ClampMax="10", ClampMin="0"))
	double SuspensionStiffness = 4.;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Driving", Interp, meta=(UIMax="40", UIMin="-40", ClampMin="-40", ClampMax="40"))
	double Drift;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Vibrations", Interp, meta=(ClampMin="0", UIMin="0"))
	double OverallVibrationMultiplier;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Car Setup", meta=(UIMin="0", UIMax="400", ClampMax="400", ClampMin="0"))
	double FrontWheelsSideOffset;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Car Setup", meta=(UIMin="0", UIMax="400", ClampMax="400", ClampMin="0"))
	double RearWheelsSideOffset;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Suspension", Interp, meta=(UIMin="0", UIMax="10", ClampMax="10", ClampMin="0"))
	double FrontCamber;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Driving", Interp)
	bool RearBreaksLock;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Driving", Interp)
	bool FrontBreaksLock;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Vibrations", Interp, meta=(UIMin="0", ClampMin="0"))
	double WheelsVibrationAmount;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Vibrations", Interp, meta=(ClampMin="0", UIMin="0"))
	double WheelsVibrationSpeed;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Car Setup", Interp, meta=(UIMin="-100", ClampMin="-100", ClampMax="100", UIMax="100"))
	double BodyHeight;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Suspension", Interp, meta=(UIMin="0", UIMax="10", ClampMax="10", ClampMin="0"))
	double RearCamber;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Read Only")
	float bSpeed; // 1/FPS m/s
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Suspension", Interp, meta=(UIMin="-10", UIMax="10", ClampMax="10", ClampMin="-10"))
	float BodyPitch;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Suspension", Interp, meta=(UIMin="-10", UIMax="10", ClampMax="10", ClampMin="-10"))
	float BodyRoll;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Vibrations", Interp, meta=(UIMin="0", ClampMin="0"))
	double BodyVibrationAmount;
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category="Vibrations", Interp, meta=(ClampMin="0", UIMin="0"))
	double BodyVibrationSpeed;
	
private:
	float CourseDistance;
	float LastCourseDistance;
	float Laps;
	double ChassisRoll;
	double ChassisPitch;
};
