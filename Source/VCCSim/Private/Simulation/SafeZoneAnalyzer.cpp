#include "Simulation/SafeZoneAnalyzer.h"
#include "ProceduralMeshComponent.h"

USafeZoneAnalyzer::USafeZoneAnalyzer()
{
	SafeZoneMaterial = nullptr;
	SafeZoneVisualizationMesh = nullptr;
}

void USafeZoneAnalyzer::InitializeSafeZoneVisualization()
{    
    // Initialize the procedural mesh component if it doesn't exist
    if (!SafeZoneVisualizationMesh)
    {
        SafeZoneVisualizationMesh = NewObject<UProceduralMeshComponent>(this);
        SafeZoneVisualizationMesh->RegisterComponent();
        SafeZoneVisualizationMesh->SetMobility(EComponentMobility::Movable);
        SafeZoneVisualizationMesh->AttachToComponent(
            this, FAttachmentTransformRules::KeepWorldTransform);
        SafeZoneVisualizationMesh->SetCollisionEnabled(ECollisionEnabled::Type::NoCollision);
    }
    
    // Load or create the safe zone material if not already set
    if (!SafeZoneMaterial)
    {
        // Try to load the safe zone material
        SafeZoneMaterial = LoadObject<UMaterialInterface>(nullptr, 
            TEXT("/Script/Engine.Material'/VCCSim/Materials/M_SafeZone.M_SafeZone'"));
        
        if (!SafeZoneMaterial)
        {
            UE_LOG(LogTemp, Error, TEXT("InitializeSafeZoneVisualization: "
                                        "Failed to load safe zone material."));
        }
    }
}

void USafeZoneAnalyzer::VisualizeSafeZone(bool Vis)
{    
    // If not showing, clear visualization and return
    if (!Vis)
    {
        if (SafeZoneVisualizationMesh)
        {
            SafeZoneVisualizationMesh->SetVisibility(false);
        }
        return;
    }
    
    // Check if we have safe zones
    if (MeshSafeZones.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("VisualizeSafeZone: No safe zones "
                                      "to visualize. Generate safe zones first."));
        return;
    }
    
    // Initialize visualization components if needed
    InitializeSafeZoneVisualization();
    
    // If mesh component failed to initialize, return
    if (!SafeZoneVisualizationMesh)
    {
        UE_LOG(LogTemp, Error, TEXT("VisualizeSafeZone: Safe zone mesh "
                                    "component not initialized"));
        return;
    }
    
    // Create the safe zone mesh
    CreateSafeZoneMesh();
    
    // Set visibility
    SafeZoneVisualizationMesh->SetVisibility(true);
}

void USafeZoneAnalyzer::ClearSafeZoneVisualization()
{
    // Clear mesh section
    if (SafeZoneVisualizationMesh)
    {
        SafeZoneVisualizationMesh->ClearAllMeshSections();
        SafeZoneVisualizationMesh->SetVisibility(false);
    }
    
    // Clear safe zones
    MeshSafeZones.Empty();
}

void USafeZoneAnalyzer::GenerateSafeZone(const float& SafeDistance)
{
    if (MeshInfos->Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("GenerateSafeZone: "
                                      "No valid World or no meshes in scene"));
        return;
    }
    
    // Clear previous safe zones
    MeshSafeZones.Empty();
    MeshSafeZones.Reserve(MeshInfos->Num());
    
    // Generate individual safe zone for each mesh
    for (const FMeshInfo& MeshInfo : *MeshInfos)
    {
        // Get the original mesh bounds
        FBox MeshBounds = MeshInfo.Bounds.GetBox();
        
        // Expand by safe distance
        FBox ExpandedMeshBounds = MeshBounds.ExpandBy(SafeDistance);
        
        // Add to our collection
        MeshSafeZones.Add(ExpandedMeshBounds);
    }
    
    UE_LOG(LogTemp, Display, TEXT("Generated %d individual mesh safe "
                                  "zones with expansion distance %.2f"), 
           MeshSafeZones.Num(), SafeDistance);
}

void USafeZoneAnalyzer::CreateSafeZoneMesh()
{
    // Clear existing mesh sections
    SafeZoneVisualizationMesh->ClearAllMeshSections();
    
    // Create arrays for procedural mesh
    TArray<FVector> Vertices;
    TArray<int32> Triangles;
    TArray<FVector> Normals;
    TArray<FVector2D> UV0;
    TArray<FColor> VertexColors;
    TArray<FProcMeshTangent> Tangents;
    
    // Create a semi-transparent color
    FColor BoxColor = SafeZoneColor.ToFColor(false);
    BoxColor.A = 128; // Semi-transparent
    
    // Process each mesh safe zone
    for (int32 BoxIndex = 0; BoxIndex < MeshSafeZones.Num(); BoxIndex++)
    {
        const FBox& SafeZoneBox = MeshSafeZones[BoxIndex];
        
        // Get the box dimensions
        FVector BoxMin = SafeZoneBox.Min;
        FVector BoxMax = SafeZoneBox.Max;
        
        // Base vertex index for this box
        int32 BaseVertexIndex = Vertices.Num();
        
        // Define the 8 corners of the box
        Vertices.Add(FVector(BoxMin.X, BoxMin.Y, BoxMin.Z)); // 0: bottom left back
        Vertices.Add(FVector(BoxMax.X, BoxMin.Y, BoxMin.Z)); // 1: bottom right back
        Vertices.Add(FVector(BoxMax.X, BoxMax.Y, BoxMin.Z)); // 2: bottom right front
        Vertices.Add(FVector(BoxMin.X, BoxMax.Y, BoxMin.Z)); // 3: bottom left front
        Vertices.Add(FVector(BoxMin.X, BoxMin.Y, BoxMax.Z)); // 4: top left back
        Vertices.Add(FVector(BoxMax.X, BoxMin.Y, BoxMax.Z)); // 5: top right back
        Vertices.Add(FVector(BoxMax.X, BoxMax.Y, BoxMax.Z)); // 6: top right front
        Vertices.Add(FVector(BoxMin.X, BoxMax.Y, BoxMax.Z)); // 7: top left front
        
        // Add colors for all 8 vertices
        for (int32 i = 0; i < 8; ++i)
        {
            VertexColors.Add(BoxColor);
        }
        
        // Add texture coordinates
        UV0.Add(FVector2D(0, 0)); // 0
        UV0.Add(FVector2D(1, 0)); // 1
        UV0.Add(FVector2D(1, 1)); // 2
        UV0.Add(FVector2D(0, 1)); // 3
        UV0.Add(FVector2D(0, 0)); // 4
        UV0.Add(FVector2D(1, 0)); // 5
        UV0.Add(FVector2D(1, 1)); // 6
        UV0.Add(FVector2D(0, 1)); // 7
        
        // Add normals
        Normals.Add(FVector(0, 0, -1)); // Bottom face
        Normals.Add(FVector(0, 0, -1));
        Normals.Add(FVector(0, 0, -1));
        Normals.Add(FVector(0, 0, -1));
        Normals.Add(FVector(0, 0, 1));  // Top face
        Normals.Add(FVector(0, 0, 1));
        Normals.Add(FVector(0, 0, 1));
        Normals.Add(FVector(0, 0, 1));
        
        // Add tangents
        for (int32 i = 0; i < 8; ++i)
        {
            Tangents.Add(FProcMeshTangent(1, 0, 0));
        }
        
        // Add triangles for each face (12 triangles total)
        // Bottom face (0,1,2,3)
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 1);
        Triangles.Add(BaseVertexIndex + 2);
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 2);
        Triangles.Add(BaseVertexIndex + 3);
        
        // Top face (4,5,6,7)
        Triangles.Add(BaseVertexIndex + 4);
        Triangles.Add(BaseVertexIndex + 6);
        Triangles.Add(BaseVertexIndex + 5);
        Triangles.Add(BaseVertexIndex + 4);
        Triangles.Add(BaseVertexIndex + 7);
        Triangles.Add(BaseVertexIndex + 6);
        
        // Front face (3,2,6,7)
        Triangles.Add(BaseVertexIndex + 3);
        Triangles.Add(BaseVertexIndex + 2);
        Triangles.Add(BaseVertexIndex + 6);
        Triangles.Add(BaseVertexIndex + 3);
        Triangles.Add(BaseVertexIndex + 6);
        Triangles.Add(BaseVertexIndex + 7);
        
        // Back face (0,1,5,4)
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 5);
        Triangles.Add(BaseVertexIndex + 1);
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 4);
        Triangles.Add(BaseVertexIndex + 5);
        
        // Left face (0,3,7,4)
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 3);
        Triangles.Add(BaseVertexIndex + 7);
        Triangles.Add(BaseVertexIndex + 0);
        Triangles.Add(BaseVertexIndex + 7);
        Triangles.Add(BaseVertexIndex + 4);
        
        // Right face (1,2,6,5)
        Triangles.Add(BaseVertexIndex + 1);
        Triangles.Add(BaseVertexIndex + 6);
        Triangles.Add(BaseVertexIndex + 2);
        Triangles.Add(BaseVertexIndex + 1);
        Triangles.Add(BaseVertexIndex + 5);
        Triangles.Add(BaseVertexIndex + 6);
    }
    
    // Create the mesh section
    if (Vertices.Num() > 0 && Triangles.Num() > 0)
    {
        SafeZoneVisualizationMesh->CreateMeshSection(0, Vertices, Triangles,
            Normals, UV0, VertexColors, Tangents, false);
        
        // Apply material
        if (SafeZoneMaterial)
        {
            SafeZoneVisualizationMesh->SetMaterial(0, SafeZoneMaterial);
        }
        
        UE_LOG(LogTemp, Display, TEXT("Created safe zone visualization "
                                      "with %d individual boxes, %d vertices, %d triangles"),
               MeshSafeZones.Num(), Vertices.Num(), Triangles.Num() / 3);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("CreateSafeZoneMesh: "
                                      "No valid geometry to create mesh"));
    }
}