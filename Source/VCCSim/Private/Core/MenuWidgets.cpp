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
#include "Engine/World.h"

void UMenuWidgets::NativeConstruct()
{
    Super::NativeConstruct();

    UE_LOG(LogTemp, Warning, TEXT("MenuWidgets: NativeConstruct called"));

    // Initialize game instance first
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
        // Add more maps as needed
        // GameInstance->AvailableMaps.Add(TEXT("Map3"));
        // GameInstance->AvailableMaps.Add(TEXT("Map4"));
        // GameInstance->AvailableMaps.Add(TEXT("Map5"));
    }

    // Set up each button individually
    SetupMapButton(Map1Button, 0, Map1OriginalColor, TEXT("Bunker Map"));
    SetupMapButton(Map2Button, 1, Map2OriginalColor, TEXT("Shipping Port Map"));
    SetupMapButton(Map3Button, 2, Map3OriginalColor, TEXT("Map3"));
    SetupMapButton(Map4Button, 3, Map4OriginalColor, TEXT("Test Map"));
    SetupMapButton(Map5Button, 4, Map5OriginalColor, TEXT("Map5"));

    // Bind click events only for valid buttons
    if (Map1Button && 0 < GameInstance->AvailableMaps.Num())
        Map1Button->OnClicked.AddDynamic(this, &UMenuWidgets::OnMap1Selected);
    
    if (Map2Button && 1 < GameInstance->AvailableMaps.Num())
        Map2Button->OnClicked.AddDynamic(this, &UMenuWidgets::OnMap2Selected);
    
    if (Map3Button && 2 < GameInstance->AvailableMaps.Num())
        Map3Button->OnClicked.AddDynamic(this, &UMenuWidgets::OnMap3Selected);
    
    if (Map4Button && 3 < GameInstance->AvailableMaps.Num())
        Map4Button->OnClicked.AddDynamic(this, &UMenuWidgets::OnMap4Selected);
    
    if (Map5Button && 4 < GameInstance->AvailableMaps.Num())
        Map5Button->OnClicked.AddDynamic(this, &UMenuWidgets::OnMap5Selected);

    // Bind hover events only for valid buttons
    if (Map1Button && 0 < GameInstance->AvailableMaps.Num())
    {
        Map1Button->OnHovered.AddDynamic(this, &UMenuWidgets::OnMap1Hovered);
        Map1Button->OnUnhovered.AddDynamic(this, &UMenuWidgets::OnMap1Unhovered);
    }
    
    if (Map2Button && 1 < GameInstance->AvailableMaps.Num())
    {
        Map2Button->OnHovered.AddDynamic(this, &UMenuWidgets::OnMap2Hovered);
        Map2Button->OnUnhovered.AddDynamic(this, &UMenuWidgets::OnMap2Unhovered);
    }
    
    if (Map3Button && 2 < GameInstance->AvailableMaps.Num())
    {
        Map3Button->OnHovered.AddDynamic(this, &UMenuWidgets::OnMap3Hovered);
        Map3Button->OnUnhovered.AddDynamic(this, &UMenuWidgets::OnMap3Unhovered);
    }
    
    if (Map4Button && 3 < GameInstance->AvailableMaps.Num())
    {
        Map4Button->OnHovered.AddDynamic(this, &UMenuWidgets::OnMap4Hovered);
        Map4Button->OnUnhovered.AddDynamic(this, &UMenuWidgets::OnMap4Unhovered);
    }
    
    if (Map5Button && 4 < GameInstance->AvailableMaps.Num())
    {
        Map5Button->OnHovered.AddDynamic(this, &UMenuWidgets::OnMap5Hovered);
        Map5Button->OnUnhovered.AddDynamic(this, &UMenuWidgets::OnMap5Unhovered);
    }

    if (StatusText)
    {
        StatusText->SetText(FText::FromString(TEXT("Ready - Click a map to begin")));
    }
    
    UE_LOG(LogTemp, Warning, TEXT("MenuWidgets: Initialization complete"));
}

void UMenuWidgets::SetupMapButton(UButton* Button, int32 MapIndex, FLinearColor& OriginalColor, const FString& MapName)
{
    if (!Button)
    {
        UE_LOG(LogTemp, Warning, TEXT("MenuWidgets: Button for map index %d (%s) is null, skipping"), MapIndex, *MapName);
        return;
    }

    // Check if we have a map for this button
    if (MapIndex >= GameInstance->AvailableMaps.Num())
    {
        // No map available for this button, hide it
        Button->SetVisibility(ESlateVisibility::Collapsed);
        Button->SetIsEnabled(false);
        UE_LOG(LogTemp, Warning, TEXT("MenuWidgets: No map available for button index %d (%s), hiding button"), MapIndex, *MapName);
        return;
    }

    // Button exists and we have a map for it, set it up
    Button->SetIsEnabled(true);
    Button->SetVisibility(ESlateVisibility::Visible);
    
    // Store original color
    OriginalColor = Button->GetColorAndOpacity();
    
    UE_LOG(LogTemp, Log, TEXT("MenuWidgets: Successfully set up button for map index %d (%s)"), MapIndex, *MapName);
}

void UMenuWidgets::LoadMapAtIndex(int32 MapIndex, const FString& LoadingText)
{
    if (!GameInstance)
    {
        UE_LOG(LogTemp, Error, TEXT("MenuWidgets: GameInstance is null"));
        return;
    }

    if (MapIndex < 0 || MapIndex >= GameInstance->AvailableMaps.Num())
    {
        UE_LOG(LogTemp, Error, TEXT("MenuWidgets: Invalid map index %d"), MapIndex);
        if (StatusText)
        {
            StatusText->SetText(FText::FromString(TEXT("Error: Map not available")));
        }
        return;
    }

    if (StatusText)
    {
        StatusText->SetText(FText::FromString(LoadingText));
    }

    // Add a small delay to ensure the loading text is visible
    GetWorld()->GetTimerManager().SetTimer(
        LoadingTimerHandle,
        [this, MapIndex]()
        {
            if (GameInstance && MapIndex < GameInstance->AvailableMaps.Num())
            {
                GameInstance->LoadMap(GameInstance->AvailableMaps[MapIndex]);
            }
        },
        0.1f,
        false
    );
}

// Map selection functions - now using the helper function
void UMenuWidgets::OnMap1Selected()
{
    UE_LOG(LogTemp, Warning, TEXT("MenuWidgets: Map1 Button Clicked"));
    LoadMapAtIndex(0, TEXT("Loading Bunker Map..."));
}

void UMenuWidgets::OnMap2Selected()
{
    UE_LOG(LogTemp, Warning, TEXT("MenuWidgets: Map2 Button Clicked"));
    LoadMapAtIndex(1, TEXT("Loading Shipping Port Map..."));
}

void UMenuWidgets::OnMap3Selected()
{
    UE_LOG(LogTemp, Warning, TEXT("MenuWidgets: Map3 Button Clicked"));
    LoadMapAtIndex(2, TEXT("Loading Map3..."));
}

void UMenuWidgets::OnMap4Selected()
{
    UE_LOG(LogTemp, Warning, TEXT("MenuWidgets: Map4 Button Clicked"));
    LoadMapAtIndex(3, TEXT("Loading Test Map..."));
}

void UMenuWidgets::OnMap5Selected()
{
    UE_LOG(LogTemp, Warning, TEXT("MenuWidgets: Map5 Button Clicked"));
    LoadMapAtIndex(4, TEXT("Loading Map5..."));
}

// Hover functions - with null checks
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
    
    // Bind the new reload map button
    if (ReloadMapButton)
    {
        ReloadMapButton->OnClicked.AddDynamic(this, &UPauseMenuWidget::OnReloadMapClicked);
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

void UPauseMenuWidget::OnReloadMapClicked()
{
    UE_LOG(LogTemp, Warning, TEXT("PauseMenuWidget: Reload Map Button Clicked"));
    
    if (!GameInstance)
    {
        UE_LOG(LogTemp, Error, TEXT("PauseMenuWidget: GameInstance is null"));
        return;
    }

    // First unpause the game
    UGameplayStatics::SetGamePaused(GetWorld(), false);
    
    // Use the GameInstance's reload method
    GameInstance->ReloadCurrentMap();
    
    // Remove the pause menu from view
    RemoveFromParent();
}