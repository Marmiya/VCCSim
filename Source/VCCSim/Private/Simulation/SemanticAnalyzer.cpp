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

#include "Simulation/SemanticAnalyzer.h"
#include "Components/ActorComponent.h"
#include "Kismet/GameplayStatics.h"
#include "Components/TextRenderComponent.h"
#include "ProceduralMeshComponent.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Engine/StaticMeshActor.h"
#include "Components/MeshComponent.h"

USemanticAnalyzer::USemanticAnalyzer()
{
	PrimaryComponentTick.bCanEverTick = true;
}

void USemanticAnalyzer::TickComponent(float DeltaTime, enum ELevelTick TickType,
                                      FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	TimeSinceLastUpdate += DeltaTime;
	if (TimeSinceLastUpdate < TickInterval)
	{
		return;
	}
	TimeSinceLastUpdate = 0.f;
	
	if (CenterCharacter)
	{
		if (bShowSemanticAnalysis)
		{
			ShowSemanticAnalysis();
		}
	}
}

void USemanticAnalyzer::ShowSemanticAnalysis()
{
    if (!CenterCharacter || !GetWorld())
    {
        UE_LOG(LogTemp, Warning, TEXT("USemanticAnalyzer: CenterCharacter or World is null!"));
        return;
    }

    FVector CurrentLocation = CenterCharacter->GetActorLocation();
    if (FVector::Dist(CurrentLocation, LastPosition) < 0.4f * DisplayDistance)
    {
        return;    
    }
    LastPosition = CurrentLocation;
    
    // Clear any existing visualization components
    ClearVisualization();
    
    // Find all static mesh actors and actors with mesh components within DisplayDistance
    TArray<AActor*> NearbyActors;
    
    // Get static mesh actors
    TArray<AActor*> StaticMeshActors;
    UGameplayStatics::GetAllActorsOfClass(GetWorld(), AStaticMeshActor::StaticClass(), StaticMeshActors);
    
    for (AActor* Actor : StaticMeshActors)
    {
        // Skip if it's the CenterCharacter or this component's owner
        if (Actor == CenterCharacter || Actor == GetOwner())
        {
            continue;
        }
        
        float Distance = FVector::Dist(CurrentLocation, Actor->GetActorLocation());
        if (Distance <= DisplayDistance)
        {
            NearbyActors.Add(Actor);
            CreateVisualizationForActor(Actor);
        }
    }
    
    // Get actors with mesh components
    TArray<AActor*> AllActors;
    UGameplayStatics::GetAllActorsOfClass(GetWorld(), AActor::StaticClass(), AllActors);
    
    for (AActor* Actor : AllActors)
    {
        // Skip if it's already processed, the CenterCharacter, or this component's owner
        if (NearbyActors.Contains(Actor) || Actor == CenterCharacter || Actor == GetOwner())
        {
            continue;
        }
        
        // Check if the actor has any mesh component
        TArray<UMeshComponent*> MeshComponents;
        Actor->GetComponents<UMeshComponent>(MeshComponents);
        
        if (MeshComponents.Num() > 0)
        {
            float Distance = FVector::Dist(CurrentLocation, Actor->GetActorLocation());
            if (Distance <= DisplayDistance)
            {
                NearbyActors.Add(Actor);
                CreateVisualizationForActor(Actor);
            }
        }
    }
}

void USemanticAnalyzer::ClearVisualization()
{
    for (USceneComponent* Component : VisualizationComponents)
    {
        if (Component)
        {
            Component->DestroyComponent();
        }
    }
    VisualizationComponents.Empty();
}

void USemanticAnalyzer::CreateVisualizationForActor(AActor* Actor)
{
    if (!Actor)
    {
        return;
    }
    
    // Get actor's bounds
    FVector Origin;
    FVector Extent;
    Actor->GetActorBounds(false, Origin, Extent);
    
    // Create a text render component for the actor label
    UTextRenderComponent* NameComponent = NewObject<UTextRenderComponent>(this);
    NameComponent->SetupAttachment(GetOwner()->GetRootComponent());
    NameComponent->RegisterComponent();
    
    // Position the text above the actor
    NameComponent->SetWorldLocation(Origin + FVector(0, 0, Extent.Z + 20.0f));
    
    #if WITH_EDITOR
        FString ActorLabel = Actor->GetActorLabel();
    #else
        FString ActorLabel = Actor->GetName(); // Runtime alternative
    #endif

    if (ActorLabel.StartsWith(TEXT("SM"), ESearchCase::CaseSensitive))
    {
        ActorLabel.RightChopInline(2); // Remove the first 2 characters ("SM")
    }
    
    // Remove underscores and replace with spaces
    ActorLabel.ReplaceInline(TEXT("_"), TEXT(" "));
    
    // Remove numeric indices at the end (like "2" and "3" in "bar_stool2")
    int32 LabelLength = ActorLabel.Len();
    int32 EndIndex = LabelLength;
    
    // Find where the numeric part starts
    for (int32 i = LabelLength - 1; i >= 0; --i)
    {
        TCHAR Char = ActorLabel[i];
        if (!FChar::IsDigit(Char))
        {
            EndIndex = i + 1;
            break;
        }
    }
    
    // Remove the numeric part if found
    if (EndIndex < LabelLength)
    {
        ActorLabel = ActorLabel.Left(EndIndex);
    }
    
    // Trim any trailing spaces
    ActorLabel.TrimEndInline();
    
    NameComponent->SetText(FText::FromString(ActorLabel));
    NameComponent->SetTextRenderColor(FColor::Yellow);
    NameComponent->SetHorizontalAlignment(EHTA_Center);
    NameComponent->SetWorldSize(20.0f);  // Adjust size as needed
    
    // Make the text face the center character
    FVector DirectionToCharacter = (CenterCharacter->GetActorLocation() -
        NameComponent->GetComponentLocation()).GetSafeNormal();
    FRotator LookAtRotation = DirectionToCharacter.Rotation();
    
    // Adjust the rotation to keep text readable (we want to rotate only on the Z axis)
    // This keeps the text upright while facing the character horizontally
    FRotator TextRotation = FRotator(0.0f, LookAtRotation.Yaw, 0.0f);
    NameComponent->SetWorldRotation(TextRotation);
    
    VisualizationComponents.Add(NameComponent);
    
    // Create a procedural mesh for the bounding box
    UProceduralMeshComponent* BoxMesh = NewObject<UProceduralMeshComponent>(this);
    BoxMesh->SetupAttachment(GetOwner()->GetRootComponent());
    BoxMesh->RegisterComponent();
    BoxMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    CreateBoundingBoxMesh(BoxMesh, Origin, Extent);
    VisualizationComponents.Add(BoxMesh);
}

void USemanticAnalyzer::CreateBoundingBoxMesh(
    UProceduralMeshComponent* MeshComponent, const FVector& Origin, const FVector& Extent)
{
    // Define the 8 corners of the box
    TArray<FVector> Vertices;
    Vertices.Add(Origin + FVector(-Extent.X, -Extent.Y, -Extent.Z));
    Vertices.Add(Origin + FVector(-Extent.X, -Extent.Y, Extent.Z));
    Vertices.Add(Origin + FVector(-Extent.X, Extent.Y, -Extent.Z));
    Vertices.Add(Origin + FVector(-Extent.X, Extent.Y, Extent.Z));
    Vertices.Add(Origin + FVector(Extent.X, -Extent.Y, -Extent.Z));
    Vertices.Add(Origin + FVector(Extent.X, -Extent.Y, Extent.Z));
    Vertices.Add(Origin + FVector(Extent.X, Extent.Y, -Extent.Z));
    Vertices.Add(Origin + FVector(Extent.X, Extent.Y, Extent.Z));
    
    // Define the 12 lines (edges) of the box as triangles for the mesh
    TArray<int32> Triangles;
    TArray<FVector> Normals;
    TArray<FVector2D> UV0;
    TArray<FColor> VertexColors;
    TArray<FProcMeshTangent> Tangents;
    
    // Create line segments for each edge of the box
    CreateBoxEdges(Vertices, Triangles, Normals, UV0, VertexColors, Tangents);
    
    // Create the mesh section
    MeshComponent->CreateMeshSection(0, Vertices, Triangles, Normals, UV0, VertexColors, Tangents, false);
    
    // Create a material for the lines
    UMaterial* LineMaterial = LoadObject<UMaterial>(nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial"));
    if (LineMaterial)
    {
        UMaterialInstanceDynamic* DynMaterial = UMaterialInstanceDynamic::Create(LineMaterial, this);
        DynMaterial->SetVectorParameterValue("Color", FLinearColor::Green);
        MeshComponent->SetMaterial(0, DynMaterial);
    }
}

void USemanticAnalyzer::CreateBoxEdges(TArray<FVector>& Vertices, TArray<int32>& Triangles, 
                                      TArray<FVector>& Normals, TArray<FVector2D>& UV0, 
                                      TArray<FColor>& VertexColors, TArray<FProcMeshTangent>& Tangents)
{
    // Define the 12 edges of a cube
    static const int32 EdgeIndices[12][2] = {
        {0, 1}, {0, 2}, {0, 4}, // From bottom left back
        {1, 3}, {1, 5}, // From bottom left front
        {2, 3}, {2, 6}, // From top left back
        {3, 7}, // From top left front
        {4, 5}, {4, 6}, // From bottom right back
        {5, 7}, // From bottom right front
        {6, 7}  // From top right back
    };
    
    // Line thickness
    const float Thickness = 1.0f;
    
    // For each edge, create a thin box represented as triangles
    const int32 BaseVertexCount = Vertices.Num();
    
    for (int32 EdgeIdx = 0; EdgeIdx < 12; EdgeIdx++)
    {
        FVector Start = Vertices[EdgeIndices[EdgeIdx][0]];
        FVector End = Vertices[EdgeIndices[EdgeIdx][1]];
        
        // Create line logic here with vertices and triangles
        // (This is simplified - in a real implementation, you'd create
        // a proper thin 3D line using triangles)
        
        FVector Direction = (End - Start).GetSafeNormal();
        FVector Up, Right;
        Direction.FindBestAxisVectors(Up, Right);
        
        // Scale to desired thickness
        Up *= Thickness;
        Right *= Thickness;
        
        // Add vertices for this edge's box
        int32 VertIdx = Vertices.Num();
        
        // Create 8 vertices for the box representing this edge
        Vertices.Add(Start - Up - Right);
        Vertices.Add(Start - Up + Right);
        Vertices.Add(Start + Up - Right);
        Vertices.Add(Start + Up + Right);
        Vertices.Add(End - Up - Right);
        Vertices.Add(End - Up + Right);
        Vertices.Add(End + Up - Right);
        Vertices.Add(End + Up + Right);
        
        // Add triangles for the box faces
        // Front face
        Triangles.Add(VertIdx + 0); Triangles.Add(VertIdx + 1); Triangles.Add(VertIdx + 2);
        Triangles.Add(VertIdx + 2); Triangles.Add(VertIdx + 1); Triangles.Add(VertIdx + 3);
        // Back face
        Triangles.Add(VertIdx + 4); Triangles.Add(VertIdx + 6); Triangles.Add(VertIdx + 5);
        Triangles.Add(VertIdx + 5); Triangles.Add(VertIdx + 6); Triangles.Add(VertIdx + 7);
        // Left face
        Triangles.Add(VertIdx + 0); Triangles.Add(VertIdx + 2); Triangles.Add(VertIdx + 4);
        Triangles.Add(VertIdx + 4); Triangles.Add(VertIdx + 2); Triangles.Add(VertIdx + 6);
        // Right face
        Triangles.Add(VertIdx + 1); Triangles.Add(VertIdx + 5); Triangles.Add(VertIdx + 3);
        Triangles.Add(VertIdx + 3); Triangles.Add(VertIdx + 5); Triangles.Add(VertIdx + 7);
        // Top face
        Triangles.Add(VertIdx + 2); Triangles.Add(VertIdx + 3); Triangles.Add(VertIdx + 6);
        Triangles.Add(VertIdx + 6); Triangles.Add(VertIdx + 3); Triangles.Add(VertIdx + 7);
        // Bottom face
        Triangles.Add(VertIdx + 0); Triangles.Add(VertIdx + 4); Triangles.Add(VertIdx + 1);
        Triangles.Add(VertIdx + 1); Triangles.Add(VertIdx + 4); Triangles.Add(VertIdx + 5);
        
        // Add simple normals, UVs, and colors
        for (int i = 0; i < 8; ++i)
        {
            Normals.Add(FVector(0, 0, 1));
            UV0.Add(FVector2D(0, 0));
            VertexColors.Add(FColor::Green);
            Tangents.Add(FProcMeshTangent(1, 0, 0));
        }
    }
}