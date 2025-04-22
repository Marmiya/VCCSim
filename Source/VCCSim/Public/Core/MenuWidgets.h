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
#include "Blueprint/UserWidget.h"
#include "Components/Button.h"
#include "Core/VCCSimGameInstance.h"
#include "MenuWidgets.generated.h"


UCLASS()
class VCCSIM_API UMenuWidgets : public UUserWidget
{
	GENERATED_BODY()

protected:
	virtual void NativeConstruct() override;

	UFUNCTION()
	void OnMap1Selected();
    
	UFUNCTION()
	void OnMap2Selected();

	UFUNCTION()
	void OnMap3Selected();

	UFUNCTION()
	void OnMap4Selected();

	UFUNCTION()
	void OnMap5Selected();

	UFUNCTION()
	void OnMap1Hovered();
    
	UFUNCTION()
	void OnMap2Hovered();

	UFUNCTION()
	void OnMap3Hovered();

	UFUNCTION()
	void OnMap4Hovered();

	UFUNCTION()
	void OnMap5Hovered();

	UFUNCTION()
	void OnMap1Unhovered();
    
	UFUNCTION()
	void OnMap2Unhovered();

	UFUNCTION()
	void OnMap3Unhovered();

	UFUNCTION()
	void OnMap4Unhovered();

	UFUNCTION()
	void OnMap5Unhovered();

	UPROPERTY()
	FLinearColor Map1OriginalColor;
    
	UPROPERTY()
	FLinearColor Map2OriginalColor;

	UPROPERTY()
	FLinearColor Map3OriginalColor;

	UPROPERTY()
	FLinearColor Map4OriginalColor;

	UPROPERTY()
	FLinearColor Map5OriginalColor;

	UPROPERTY(EditAnywhere, Category = "Button Colors")
	FLinearColor HoveredColor =
		FLinearColor(1.0f, 0.8f, 0.0f, 1.0f);

	UPROPERTY()
	UVCCSimGameInstance* GameInstance;

	UPROPERTY(meta = (BindWidget))
	class UButton* Map1Button;

	UPROPERTY(meta = (BindWidget))
	class UButton* Map2Button;

	UPROPERTY(meta = (BindWidget))
	class UButton* Map3Button;

	UPROPERTY(meta = (BindWidget))
	class UButton* Map4Button;

	UPROPERTY(meta = (BindWidget))
	class UButton* Map5Button;

	UPROPERTY(meta = (BindWidget))
	class UTextBlock* StatusText;

private:
	FTimerHandle LoadingTimerHandle;
};

UCLASS()
class VCCSIM_API UPauseMenuWidget : public UUserWidget
{
	GENERATED_BODY()

protected:
	virtual void NativeConstruct() override;

	UPROPERTY(meta = (BindWidget))
	class UButton* ResumeButton;

	UPROPERTY(meta = (BindWidget))
	class UButton* MainMenuButton;

	UPROPERTY(meta = (BindWidget))
	class UButton* QuitButton;

	UPROPERTY(meta = (BindWidget))
	class UButton* SemanticButton;

	UFUNCTION()
	void OnResumeClicked();

	UFUNCTION()
	void OnMainMenuClicked();

	UFUNCTION()
	void OnQuitClicked();

	UFUNCTION()
	void OnSemanticClicked();

private:
	UPROPERTY()
	UVCCSimGameInstance* GameInstance;
};