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

#include "Core/MenuWidgets.h"
#include "Kismet/GameplayStatics.h"
#include "Kismet/KismetSystemLibrary.h"
#include "Components/Button.h"
#include "Components/TextBlock.h"
#include "Simulation/SceneAnalysisManager.h"

void UMenuWidgets::NativeConstruct()
{
    Super::NativeConstruct();

    UE_LOG(LogTemp, Warning, TEXT("MenuWidgets: NativeConstruct called"));

    if (!Map1Button || !Map2Button || !Map3Button || !Map4Button || !Map5Button)
    {
        UE_LOG(LogTemp, Error, TEXT("MenuWidgets: One or more buttons are null! "
                                    "Map1: %s, Map2: %s, Map3: %s, Map4: %s, Map5: %s"),
            Map1Button ? TEXT("Valid") : TEXT("Null"),
            Map2Button ? TEXT("Valid") : TEXT("Null"),
            Map3Button ? TEXT("Valid") : TEXT("Null"),
            Map4Button ? TEXT("Valid") : TEXT("Null"),
            Map5Button ? TEXT("Valid") : TEXT("Null"));
        return;
    }

    Map1Button->SetIsEnabled(true);
    Map2Button->SetIsEnabled(true);
    Map3Button->SetIsEnabled(true);
    Map4Button->SetIsEnabled(true);
    Map5Button->SetIsEnabled(true);

    Map1Button->OnClicked.AddDynamic(this, &UMenuWidgets::OnMap1Selected);
    Map2Button->OnClicked.AddDynamic(this, &UMenuWidgets::OnMap2Selected);
    Map3Button->OnClicked.AddDynamic(this, &UMenuWidgets::OnMap3Selected);
    Map4Button->OnClicked.AddDynamic(this, &UMenuWidgets::OnMap4Selected);
    Map5Button->OnClicked.AddDynamic(this, &UMenuWidgets::OnMap4Selected);
    
    Map1OriginalColor = Map1Button->GetColorAndOpacity();
    Map2OriginalColor = Map2Button->GetColorAndOpacity();
    Map3OriginalColor = Map3Button->GetColorAndOpacity();
    Map4OriginalColor = Map4Button->GetColorAndOpacity();
    Map5OriginalColor = Map5Button->GetColorAndOpacity();

    Map1Button->OnHovered.AddDynamic(this, &UMenuWidgets::OnMap1Hovered);
    Map1Button->OnUnhovered.AddDynamic(this, &UMenuWidgets::OnMap1Unhovered);
    Map2Button->OnHovered.AddDynamic(this, &UMenuWidgets::OnMap2Hovered);
    Map2Button->OnUnhovered.AddDynamic(this, &UMenuWidgets::OnMap2Unhovered);
    Map3Button->OnHovered.AddDynamic(this, &UMenuWidgets::OnMap3Hovered);
    Map3Button->OnUnhovered.AddDynamic(this, &UMenuWidgets::OnMap3Unhovered);
    Map4Button->OnHovered.AddDynamic(this, &UMenuWidgets::OnMap4Hovered);
    Map4Button->OnUnhovered.AddDynamic(this, &UMenuWidgets::OnMap4Unhovered);
    Map5Button->OnHovered.AddDynamic(this, &UMenuWidgets::OnMap4Hovered);
    Map5Button->OnUnhovered.AddDynamic(this, &UMenuWidgets::OnMap4Unhovered);

    // Initialize game instance
    GameInstance = Cast<UVCCSimGameInstance>(GetGameInstance());
    
    if (!GameInstance)
    {
        UE_LOG(LogTemp, Error, TEXT("MenuWidgets: Failed to get VCCSimGameInstance"));
        if (StatusText)
        {
            StatusText->SetText(FText::FromString(TEXT("Error: Game instance not found")));
        }
        return;
    }

    // Initialize available maps
    if (GameInstance->AvailableMaps.IsEmpty())
    {
        GameInstance->AvailableMaps.Add(TEXT("Bunker"));
        GameInstance->AvailableMaps.Add(TEXT("Shipping_Port"));
    }

    if (StatusText)
    {
        StatusText->SetText(FText::FromString(TEXT("Ready - Click a map to begin")));
    }
    
    UE_LOG(LogTemp, Warning, TEXT("MenuWidgets: Initialization complete"));
}

void UMenuWidgets::OnMap1Hovered()
{
    if (Map1Button)
    {
        Map1Button->SetColorAndOpacity(HoveredColor);
    }
}

void UMenuWidgets::OnMap2Hovered()
{
    if (Map2Button)
    {
        Map2Button->SetColorAndOpacity(HoveredColor);
    }
}

void UMenuWidgets::OnMap3Hovered()
{
    if (Map3Button)
    {
        Map3Button->SetColorAndOpacity(HoveredColor);
    }
}

void UMenuWidgets::OnMap4Hovered()
{
    if (Map4Button)
    {
        Map4Button->SetColorAndOpacity(HoveredColor);
    }
}

void UMenuWidgets::OnMap5Hovered()
{
    if (Map5Button)
    {
        Map5Button->SetColorAndOpacity(HoveredColor);
    }
}

void UMenuWidgets::OnMap1Unhovered()
{
    if (Map1Button)
    {
        Map1Button->SetColorAndOpacity(Map1OriginalColor);
    }
}

void UMenuWidgets::OnMap2Unhovered()
{
    if (Map2Button)
    {
        Map2Button->SetColorAndOpacity(Map2OriginalColor);
    }
}

void UMenuWidgets::OnMap3Unhovered()
{
    if (Map3Button)
    {
        Map3Button->SetColorAndOpacity(Map3OriginalColor);
    }
}

void UMenuWidgets::OnMap4Unhovered()
{
    if (Map4Button)
    {
        Map4Button->SetColorAndOpacity(Map4OriginalColor);
    }
}

void UMenuWidgets::OnMap5Unhovered()
{
    if (Map5Button)
    {
        Map5Button->SetColorAndOpacity(Map5OriginalColor);
    }
}

void UMenuWidgets::OnMap1Selected()
{
    UE_LOG(LogTemp, Warning, TEXT("MenuWidgets: Map1 Button Clicked"));
    
    if (!GameInstance)
    {
        UE_LOG(LogTemp, Error, TEXT("MenuWidgets: "
                                    "GameInstance is null in OnMap1Selected"));
        return;
    }

    if (StatusText)
    {
        StatusText->SetText(FText::FromString(TEXT("Loading Bunker Map...")));
    }

    // Add a small delay to ensure the loading text is visible
    GetWorld()->GetTimerManager().SetTimer(
        LoadingTimerHandle,
        [this]()
        {
            if (GameInstance)
            {
                GameInstance->LoadMap(GameInstance->AvailableMaps[0]);
            }
        },
        0.1f,
        false
    );
}

void UMenuWidgets::OnMap2Selected()
{
    UE_LOG(LogTemp, Warning, TEXT("MenuWidgets: Map2 Button Clicked"));
    
    if (!GameInstance)
    {
        UE_LOG(LogTemp, Error, TEXT("MenuWidgets: "
                                    "GameInstance is null in OnMap2Selected"));
        return;
    }

    if (StatusText)
    {
        StatusText->SetText(FText::FromString(TEXT("Loading Shipping Port Map...")));
    }

    // Add a small delay to ensure the loading text is visible
    GetWorld()->GetTimerManager().SetTimer(
        LoadingTimerHandle,
        [this]()
        {
            if (GameInstance)
            {
                GameInstance->LoadMap(GameInstance->AvailableMaps[1]);
            }
        },
        0.1f,
        false
    );
}

void UMenuWidgets::OnMap3Selected()
{
    UE_LOG(LogTemp, Warning, TEXT("MenuWidgets: Map3 Button Clicked"));
    
    if (!GameInstance)
    {
        UE_LOG(LogTemp, Error, TEXT("MenuWidgets: "
                                    "GameInstance is null in OnMap3Selected"));
        return;
    }

    if (StatusText)
    {
        StatusText->SetText(FText::FromString(TEXT("Loading Map3...")));
    }

    // Add a small delay to ensure the loading text is visible
    GetWorld()->GetTimerManager().SetTimer(
        LoadingTimerHandle,
        [this]()
        {
            if (GameInstance)
            {
                GameInstance->LoadMap(GameInstance->AvailableMaps[2]);
            }
        },
        0.1f,
        false
    );
}

void UMenuWidgets::OnMap4Selected()
{
    UE_LOG(LogTemp, Warning, TEXT("MenuWidgets: MapTest Button Clicked"));
    
    if (!GameInstance)
    {
        UE_LOG(LogTemp, Error, TEXT("MenuWidgets: "
                                    "GameInstance is null in OnMapTestSelected"));
        return;
    }

    if (StatusText)
    {
        StatusText->SetText(FText::FromString(TEXT("Loading Test Map...")));
    }

    // Add a small delay to ensure the loading text is visible
    GetWorld()->GetTimerManager().SetTimer(
        LoadingTimerHandle,
        [this]()
        {
            if (GameInstance)
            {
                GameInstance->LoadMap(GameInstance->AvailableMaps[3]);
            }
        },
        0.1f,
        false
    );
}

void UMenuWidgets::OnMap5Selected()
{
    UE_LOG(LogTemp, Warning, TEXT("MenuWidgets: Map5 Button Clicked"));
    
    if (!GameInstance)
    {
        UE_LOG(LogTemp, Error, TEXT("MenuWidgets: "
                                    "GameInstance is null in OnMap5Selected"));
        return;
    }

    if (StatusText)
    {
        StatusText->SetText(FText::FromString(TEXT("Loading Map5...")));
    }

    // Add a small delay to ensure the loading text is visible
    GetWorld()->GetTimerManager().SetTimer(
        LoadingTimerHandle,
        [this]()
        {
            if (GameInstance)
            {
                GameInstance->LoadMap(GameInstance->AvailableMaps[4]);
            }
        },
        0.1f,
        false
    );
}

/* --------------------------------Pause Menu---------------------------------*/ 

void UPauseMenuWidget::NativeConstruct()
{
    Super::NativeConstruct();

    // Get GameInstance reference
    GameInstance = Cast<UVCCSimGameInstance>(GetGameInstance());

    if (ResumeButton)
    {
        ResumeButton->OnClicked.AddDynamic(this, &UPauseMenuWidget::OnResumeClicked);
    }
    if (MainMenuButton)
    {
        MainMenuButton->OnClicked.AddDynamic(this, &UPauseMenuWidget::OnMainMenuClicked);
    }
    if (QuitButton)
    {
        QuitButton->OnClicked.AddDynamic(this, &UPauseMenuWidget::OnQuitClicked);
    }
    if (SemanticButton)
    {
        SemanticButton->OnClicked.AddDynamic(this, &UPauseMenuWidget::OnSemanticClicked);
    }
}

void UPauseMenuWidget::OnResumeClicked()
{
    if (APlayerController*PC = GetWorld()->GetFirstPlayerController())
    {
        PC->SetInputMode(FInputModeGameOnly());
        PC->SetShowMouseCursor(false);
    }

    UGameplayStatics::SetGamePaused(GetWorld(), false);
    RemoveFromParent();
}

void UPauseMenuWidget::OnMainMenuClicked()
{
    if (GameInstance)
    {
        // First unpause the game
        UGameplayStatics::SetGamePaused(GetWorld(), false);
        
        // Then return to main menu
        GameInstance->ReturnToMainMenu();
    }
}

void UPauseMenuWidget::OnQuitClicked()
{
    // Save game state before quitting
    if (GameInstance)
    {
        GameInstance->SaveGameState();
    }
    
    UKismetSystemLibrary::QuitGame(GetWorld(), GetWorld()->GetFirstPlayerController(), 
        EQuitPreference::Quit, false);
}

void UPauseMenuWidget::OnSemanticClicked()
{
    if (ASceneAnalysisManager* SceneAnalysisManager = 
        Cast<ASceneAnalysisManager>(UGameplayStatics::GetActorOfClass(GetWorld(), 
        ASceneAnalysisManager::StaticClass())))
    {
        const auto ans = SceneAnalysisManager->InterfaceVisualizeSemanticAnalysis();

        if (SemanticButton && SemanticButton->GetChildAt(0))
        {
            UTextBlock* ButtonText = Cast<UTextBlock>(SemanticButton->GetChildAt(0));
            if (ButtonText)
            {
                FString NewText = ans ? TEXT("Hide Semantic Info") : TEXT("Show Semantic Info");
                ButtonText->SetText(FText::FromString(NewText));
            }
        }
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Scene Analysis Manager not found!"));
    }
}
