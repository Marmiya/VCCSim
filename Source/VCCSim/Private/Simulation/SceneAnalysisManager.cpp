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

#include "Simulation/SceneAnalysisManager.h"
#include "Components/StaticMeshComponent.h"
#include "TimerManager.h"
#include "Sensors/CameraSensor.h"
#include "EngineUtils.h"
#include "Engine/StaticMesh.h"
#include "DrawDebugHelpers.h"
#include "Kismet/GameplayStatics.h"
#include "Simulation/SemanticAnalyzer.h"
#include "Simulation/SafeZoneAnalyzer.h"
#include "Simulation/ComplexityAnalyzer.h"
#include "Simulation/CoverageAnalyzer.h"

ASceneAnalysisManager::ASceneAnalysisManager()
{
    World = nullptr;
    TotalPointsInScene = 0;
    TotalTrianglesInScene = 0;
    LogPath = FPaths::ProjectLogDir();

    SetActorEnableCollision(false);

    SemanticAnalyzer = CreateDefaultSubobject<USemanticAnalyzer>(TEXT("SemanticAnalyzer"));
    if (SemanticAnalyzer)
    {
        SemanticAnalyzer->SetupAttachment(RootComponent);
        SemanticAnalyzer->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        SemanticAnalyzer->SetVisibility(false);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("ASceneAnalysisManager::ASceneAnalysisManager:"
                                    " Failed to create SemanticAnalyzer component!"));
    }

    SafeZoneAnalyzer = CreateDefaultSubobject<USafeZoneAnalyzer>(TEXT("SafeZoneAnalyzer"));
    if (SafeZoneAnalyzer)
    {
        SafeZoneAnalyzer->SetupAttachment(RootComponent);
        SafeZoneAnalyzer->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        SafeZoneAnalyzer->SetVisibility(false);
        SafeZoneAnalyzer->MeshInfos = &SceneMeshes;
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("ASceneAnalysisManager::ASceneAnalysisManager:"
                                    " Failed to create SafeZoneAnalyzer component!"));
    }

    ComplexityAnalyzer = CreateDefaultSubobject<UComplexityAnalyzer>(TEXT("ComplexityAnalyzer"));
    if (ComplexityAnalyzer)
    {
        ComplexityAnalyzer->SetupAttachment(RootComponent);
        ComplexityAnalyzer->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        ComplexityAnalyzer->SetVisibility(false);
        ComplexityAnalyzer->GridResolutionPtr = &GridResolution;
        ComplexityAnalyzer->MeshInfos = &SceneMeshes;
        ComplexityAnalyzer->TotalPointsInScenePtr = &TotalPointsInScene;
        ComplexityAnalyzer->TotalTrianglesInScenePtr = &TotalTrianglesInScene;
        ComplexityAnalyzer->UnifiedGridPtr = &UnifiedGrid;
        ComplexityAnalyzer->GridOriginPtr = &GridOrigin;
        ComplexityAnalyzer->GridSizePtr = &GridSize;
        ComplexityAnalyzer->GridInitializedPtr = &bGridInitialized;
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("ASceneAnalysisManager::ASceneAnalysisManager:"
                                    " Failed to create ComplexityAnalyzer component!"));
    }

    CoverageAnalyzer = CreateDefaultSubobject<UCoverageAnalyzer>(TEXT("CoverageAnalyzer"));
    if (CoverageAnalyzer)
    {
        CoverageAnalyzer->SetupAttachment(RootComponent);
        CoverageAnalyzer->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        CoverageAnalyzer->SetVisibility(false);
        CoverageAnalyzer->GridResolutionPtr = &GridResolution;
        CoverageAnalyzer->CameraIntrinsicsPtr = &CameraIntrinsics;
        CoverageAnalyzer->MeshInfos = &SceneMeshes;
        CoverageAnalyzer->UnifiedGridPtr = &UnifiedGrid;
        CoverageAnalyzer->GridOriginPtr = &GridOrigin;
        CoverageAnalyzer->GridSizePtr = &GridSize;
        CoverageAnalyzer->GridInitializedPtr = &bGridInitialized;
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("ASceneAnalysisManager::ASceneAnalysisManager:"
                                    " Failed to create CoverageAnalyzer component!"));
    }
}

bool ASceneAnalysisManager::Initialize(UWorld* InWorld, FString&& Path)
{
    if (!InWorld)
        return false;
    
    World = InWorld;
    CoverageAnalyzer->World = World;
    LogPath = std::move(Path) + "/SceneAnalysisLog";
    return true;
}

void ASceneAnalysisManager::ScanScene()
{
    ScanSceneImpl(TOptional<FBox>());
}

void ASceneAnalysisManager::ScanSceneRegion3D(
    float MinX, float MaxX, float MinY, float MaxY, float MinZ, float MaxZ)
{
    // Create a 3D region bounds
    FBox RegionBounds(
        FVector(MinX, MinY, MinZ),
        FVector(MaxX, MaxY, MaxZ)
    );
    
    ScanSceneImpl(RegionBounds);
}

void ASceneAnalysisManager::ScanSceneImpl(const TOptional<FBox>& RegionBounds)
{
    if (!World)
        return;
    
    // Log region info if bounds are specified
    if (RegionBounds.IsSet())
    {
        const FBox& Bounds = RegionBounds.GetValue();
        UE_LOG(LogTemp, Display, TEXT("Scanning scene within region bounds: "
                                      "X(%.2f to %.2f), Y(%.2f to %.2f), Z(%.2f to %.2f)"),
               Bounds.Min.X, Bounds.Max.X, Bounds.Min.Y, Bounds.Max.Y, Bounds.Min.Z, Bounds.Max.Z);
    }
    else
    {
        UE_LOG(LogTemp, Display, TEXT("Scanning entire scene (no bounds)"));
    }
    
    // Clear previous data
    SceneMeshes.Empty();
    TotalPointsInScene = 0;
    TotalTrianglesInScene = 0;
    
    int32 TotalActorsFound = 0;
    int32 TotalMeshComponentsFound = 0;
    int32 TotalMeshesInRegion = 0;
    
    // Iterate through ALL actors in the world instead of just StaticMeshActors
    for (TActorIterator<AActor> ActorItr(World); ActorItr; ++ActorItr)
    {
        AActor* Actor = *ActorItr;
        TotalActorsFound++;
        
        // Skip actors with the "NotSMActor" tag
        if (Actor->ActorHasTag(FName("NotSMActor")))
        {
            continue;
        }
        
        // Find all static mesh components on this actor
        TArray<UStaticMeshComponent*> MeshComponents;
        Actor->GetComponents<UStaticMeshComponent>(MeshComponents);
        
        for (UStaticMeshComponent* MeshComp : MeshComponents)
        {
            if (MeshComp && MeshComp->GetStaticMesh())
            {
                TotalMeshComponentsFound++;
                
                // If region bounds are specified, check if mesh intersects with region
                bool bShouldIncludeMesh = true;
                
                if (RegionBounds.IsSet())
                {
                    const FBox& RegionBox = RegionBounds.GetValue();
                    FBoxSphereBounds MeshBoundsS = MeshComp->Bounds;
                    FBox MeshBox = MeshBoundsS.GetBox();
                    
                    // Explicit overlap test instead of using Intersect
                    bool bOverlapsX = (MeshBox.Max.X >= RegionBox.Min.X) &&
                        (MeshBox.Min.X <= RegionBox.Max.X);
                    bool bOverlapsY = (MeshBox.Max.Y >= RegionBox.Min.Y) &&
                        (MeshBox.Min.Y <= RegionBox.Max.Y);
                    bool bOverlapsZ = (MeshBox.Max.Z >= RegionBox.Min.Z) &&
                        (MeshBox.Min.Z <= RegionBox.Max.Z);
                    
                    bShouldIncludeMesh = bOverlapsX && bOverlapsY && bOverlapsZ;
                    
                    if (bShouldIncludeMesh)
                    {
                        TotalMeshesInRegion++;
                        UE_LOG(LogTemp, Verbose,
                            TEXT("Mesh from %s is within region - Bounds: "
                                 "X(%.2f to %.2f), Y(%.2f to %.2f), Z(%.2f to %.2f)"),
                              *Actor->GetName(),
                              MeshBox.Min.X, MeshBox.Max.X,
                              MeshBox.Min.Y, MeshBox.Max.Y,
                              MeshBox.Min.Z, MeshBox.Max.Z);
                    }
                    else
                    {
                        UE_LOG(LogTemp, Verbose,
                            TEXT("Mesh from %s is outside region - Bounds:"
                                 " X(%.2f to %.2f), Y(%.2f to %.2f), Z(%.2f to %.2f)"),
                              *Actor->GetName(),
                              MeshBox.Min.X, MeshBox.Max.X,
                              MeshBox.Min.Y, MeshBox.Max.Y,
                              MeshBox.Min.Z, MeshBox.Max.Z);
                    }
                }
                
                if (bShouldIncludeMesh)
                {
                    FMeshInfo MeshInfo;
                    ExtractMeshData(MeshComp, MeshInfo);
                    SceneMeshes.Add(MeshInfo);
                    
                    TotalTrianglesInScene += MeshInfo.NumTriangles;
                    TotalPointsInScene += MeshInfo.NumVertices;
                }
            }
        }
    }
    
    UE_LOG(LogTemp, Display, TEXT("Scan complete: Found %d actors, "
                                  "%d with mesh components, %d within region bounds. "
                                  "Added %d meshes with %d triangles and %d vertices."),
          TotalActorsFound, TotalMeshComponentsFound, TotalMeshesInRegion,
          SceneMeshes.Num(), TotalTrianglesInScene, TotalPointsInScene);
    
    if (SceneMeshes.Num() == 0)
    {
        if (TotalActorsFound == 0)
        {
            UE_LOG(LogTemp, Warning, TEXT("No actors found in the world!"));
        }
        else if (TotalMeshComponentsFound == 0)
        {
            UE_LOG(LogTemp, Warning,
                TEXT("Found %d actors, but none have valid static mesh components!"), 
                TotalActorsFound);
        }
        else if (RegionBounds.IsSet())
        {
            UE_LOG(LogTemp, Warning,
                TEXT("Found %d actors with valid static mesh components, "
                     "but none intersect with the specified region!"), 
                  TotalMeshComponentsFound);
        }
    }
    
    CoverageAnalyzer->ResetCoverage();
    InitializeUnifiedGrid();
}

void ASceneAnalysisManager::RegisterCamera(URGBCameraComponent* CameraComponent)
{
    CameraIntrinsics.Add(CameraComponent->CameraName,
    CameraComponent->GetCameraIntrinsics());
}

FMeshInfo ASceneAnalysisManager::GetMeshInfo(int32 MeshID) const
{
    for (const FMeshInfo& MeshInfo : SceneMeshes)
    {
        if (MeshInfo.MeshID == MeshID)
            return MeshInfo;
    }
    
    return FMeshInfo();
}

TArray<FMeshInfo> ASceneAnalysisManager::GetAllMeshInfo() const
{
    return SceneMeshes;
}

void ASceneAnalysisManager::ExtractMeshData(
    UStaticMeshComponent* MeshComponent, FMeshInfo& OutMeshInfo)
{
    if (!MeshComponent || !MeshComponent->GetStaticMesh())
        return;
    
    UStaticMesh* StaticMesh = MeshComponent->GetStaticMesh();
    
    // Basic mesh info
    OutMeshInfo.MeshID = MeshComponent->GetUniqueID();
    OutMeshInfo.MeshName = MeshComponent->GetName();
    OutMeshInfo.Mesh = StaticMesh;
    OutMeshInfo.Transform = MeshComponent->GetComponentTransform();
    OutMeshInfo.Bounds = MeshComponent->Bounds;
    OutMeshInfo.bIsVisible = MeshComponent->IsVisible();
    
    // Get mesh data
    if (StaticMesh->GetRenderData() && StaticMesh->GetRenderData()->LODResources.Num() > 0)
    {
        const FStaticMeshLODResources& LODModel = StaticMesh->GetRenderData()->LODResources[0];
        
        // Get vertices and indices
        OutMeshInfo.NumVertices = LODModel.VertexBuffers.PositionVertexBuffer.GetNumVertices();
        OutMeshInfo.NumTriangles = LODModel.IndexBuffer.GetNumIndices() / 3;
        
        // Extract vertex positions
        OutMeshInfo.VertexPositions.Reserve(OutMeshInfo.NumVertices);
        for (int32 VertIdx = 0; VertIdx < OutMeshInfo.NumVertices; ++VertIdx)
        {
            // Get the FVector3f from the vertex buffer
            FVector3f VertexPos3f = LODModel.VertexBuffers.
            PositionVertexBuffer.VertexPosition(VertIdx);
            
            // Convert to FVector (explicit conversion)
            FVector VertexPos(VertexPos3f.X, VertexPos3f.Y, VertexPos3f.Z);
            
            // Transform to world space
            VertexPos = OutMeshInfo.Transform.TransformPosition(VertexPos);
            OutMeshInfo.VertexPositions.Add(VertexPos);
        }
        
        // Extract indices
        OutMeshInfo.Indices.Reserve(LODModel.IndexBuffer.GetNumIndices());
        for (int32 IndexIdx = 0; IndexIdx < LODModel.IndexBuffer.GetNumIndices(); ++IndexIdx)
        {
            OutMeshInfo.Indices.Add(LODModel.IndexBuffer.GetIndex(IndexIdx));
        }
    }
}

FIntVector ASceneAnalysisManager::WorldToGridCoordinates(const FVector& WorldPos) const
{
    return FIntVector(
        FMath::FloorToInt((WorldPos.X - GridOrigin.X) / GridResolution),
        FMath::FloorToInt((WorldPos.Y - GridOrigin.Y) / GridResolution),
        FMath::FloorToInt((WorldPos.Z - GridOrigin.Z) / GridResolution)
    );
}

void ASceneAnalysisManager::InitializeUnifiedGrid()
{
    if (!World || SceneMeshes.Num() == 0)
        return;
    
    // Calculate bounds of the scene
    FBox SceneBounds(EForceInit::ForceInit);
    for (const FMeshInfo& MeshInfo : SceneMeshes)
    {
        SceneBounds += MeshInfo.Bounds.GetBox();
    }
    
    // Expand bounds slightly
    SceneBounds = SceneBounds.ExpandBy(GridResolution * 2.0f);
    
    GridOrigin = SceneBounds.Min;
    
    FVector BoundsSize = SceneBounds.GetSize();
    int32 GridSizeX = FMath::Max(1, FMath::CeilToInt(BoundsSize.X / GridResolution));
    int32 GridSizeY = FMath::Max(1, FMath::CeilToInt(BoundsSize.Y / GridResolution));
    int32 GridSizeZ = FMath::Max(1, FMath::CeilToInt(BoundsSize.Z / GridResolution));
    
    GridSize = FVector(GridSizeX, GridSizeY, GridSizeZ);
    
    CoverageAnalyzer->PrepareCoverage();
    
    bGridInitialized = true;
    
    UE_LOG(LogTemp, Display, TEXT("Unified grid initialized: "
                                  "theoretical grid %dx%dx%d, actual populated cells: %d"), 
           GridSizeX, GridSizeY, GridSizeZ, UnifiedGrid.Num());
}

/* -------------------------------- SubModules ------------------------------ */

bool ASceneAnalysisManager::InterfaceVisualizeSemanticAnalysis()
{
    SemanticAnalyzer->bShowSemanticAnalysis = !SemanticAnalyzer->bShowSemanticAnalysis;
    if (UGameplayStatics::IsGamePaused(GetWorld()))
    {
        SemanticAnalyzer->bShowSemanticAnalysis = SemanticAnalyzer->bShowSemanticAnalysis;
        SemanticAnalyzer->ShowSemanticAnalysis();
    }
    return SemanticAnalyzer->bShowSemanticAnalysis;
}

void ASceneAnalysisManager::InterfaceVisualizeSafeZone(bool bShow)
{
    SafeZoneAnalyzer->VisualizeSafeZone(bShow);
}

void ASceneAnalysisManager::InterfaceClearSafeZoneVisualization()
{
    SafeZoneAnalyzer->ClearSafeZoneVisualization();
}

void ASceneAnalysisManager::InterfaceInitializeSafeZoneVisualization()
{
    SafeZoneAnalyzer->InitializeSafeZoneVisualization();
}

void ASceneAnalysisManager::InterfaceGenerateSafeZone(const float& SafeDistance)
{
    SafeZoneAnalyzer->GenerateSafeZone(SafeDistance);
}

void ASceneAnalysisManager::InterfaceVisualizeComplexity(bool bShow)
{
    ComplexityAnalyzer->VisualizeComplexity(bShow);
}

void ASceneAnalysisManager::InterfaceClearComplexityVisualization()
{
    ComplexityAnalyzer->ClearComplexityVisualization();
}

void ASceneAnalysisManager::InterfaceInitializeComplexityVisualization()
{
    ComplexityAnalyzer->InitializeComplexityVisualization();
}

void ASceneAnalysisManager::InterfaceAnalyzeGeometricComplexity()
{
    ComplexityAnalyzer->AnalyzeGeometricComplexity();
}

void ASceneAnalysisManager::InterfaceVisualizeCoverage(bool bShow)
{
    CoverageAnalyzer->VisualizeCoverage(bShow);
}

void ASceneAnalysisManager::InterfaceClearCoverageVisualization()
{
    CoverageAnalyzer->ClearCoverageVisualization();
}

void ASceneAnalysisManager::InterfaceInitializeCoverageVisualization()
{
    CoverageAnalyzer->InitializeCoverageVisualization();
}

void ASceneAnalysisManager::InterfaceUpdateCoverageGrid()
{
    CoverageAnalyzer->UpdateCoverageGrid();
}

void ASceneAnalysisManager::InterfaceComputeCoverage(
    const TArray<FTransform>& CameraTransforms, const FString& CameraName)
{
    CoverageAnalyzer->ComputeCoverage(CameraTransforms, CameraName);
}

/* ----------------------------- Test ----------------------------- */

void ASceneAnalysisManager::ExportMeshesToPly()
{
    if (!World)
    {
        UE_LOG(LogTemp, Warning, TEXT("USceneAnalysisManager::ExportMeshesToPly:"
            " World not set!"));
        return;
    }

    if (SceneMeshes.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("USceneAnalysisManager::ExportMeshesToPly:"
            " No meshes found in the scene!"));
        return;
    }
    
    // Create export directory if it doesn't exist
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    if (!PlatformFile.DirectoryExists(*LogPath))
    {
        PlatformFile.CreateDirectory(*LogPath);
    }

    
    // Export each mesh to a separate PLY file
    for (const FMeshInfo& MeshInfo : SceneMeshes)
    {
        FString FilePath = FPaths::Combine(LogPath, FString::Printf(TEXT("%s_%d.ply"),
            *MeshInfo.MeshName, MeshInfo.MeshID));
        FFileHelper::SaveStringToFile(GeneratePlyContent(MeshInfo), *FilePath);
    }
}

FString ASceneAnalysisManager::GeneratePlyContent(const FMeshInfo& MeshInfo)
{
    FString PLYContent;
    
    // PLY Header
    PLYContent += TEXT("ply\n");
    PLYContent += TEXT("format ascii 1.0\n");
    PLYContent += FString::Printf(TEXT("element vertex %d\n"), MeshInfo.VertexPositions.Num());
    PLYContent += TEXT("property float x\n");
    PLYContent += TEXT("property float y\n");
    PLYContent += TEXT("property float z\n");
    PLYContent += FString::Printf(TEXT("element face %d\n"), MeshInfo.NumTriangles);
    PLYContent += TEXT("property list uchar int vertex_indices\n");
    PLYContent += TEXT("end_header\n");
    
    // Vertex data
    for (const FVector& Vertex : MeshInfo.VertexPositions)
    {
        PLYContent += FString::Printf(TEXT("%f %f %f\n"), Vertex.X, Vertex.Y, Vertex.Z);
    }
    
    // Face data
    for (int32 i = 0; i < MeshInfo.Indices.Num(); i += 3)
    {
        if (i + 2 < MeshInfo.Indices.Num())
        {
            PLYContent += FString::Printf(TEXT("3 %d %d %d\n"), 
                MeshInfo.Indices[i], 
                MeshInfo.Indices[i + 1], 
                MeshInfo.Indices[i + 2]);
        }
    }
    
    return PLYContent;
}

void ASceneAnalysisManager::VisualizeSceneMeshes(
    float Duration, bool bShowWireframe, bool bShowVertices, float VertexSize)
{
    if (!World || SceneMeshes.Num() == 0)
        return;
    
    // Generate a unique color for each mesh for easier distinction
    TArray<FColor> MeshColors;
    for (int32 i = 0; i < SceneMeshes.Num(); ++i)
    {
        // Create visually distinct colors using golden ratio
        const float Hue = fmodf(i * 0.618033988749895f, 1.0f);
        FLinearColor LinearColor = FLinearColor::MakeFromHSV8(Hue * 255.0f, 200, 200);
        MeshColors.Add(LinearColor.ToFColor(false));
    }
    
    // Visualize each mesh
    for (int32 MeshIdx = 0; MeshIdx < SceneMeshes.Num(); ++MeshIdx)
    {
        const FMeshInfo& MeshInfo = SceneMeshes[MeshIdx];
        const FColor& Color = MeshColors[MeshIdx];
        
        // Draw mesh bounds
        DrawDebugBox(World, MeshInfo.Bounds.Origin, MeshInfo.Bounds.BoxExtent,
            Color, false, Duration, 0, 2.0f);
        
        // Draw mesh ID text
        FString MeshText = FString::Printf(TEXT("Mesh ID: %d\nName: %s\nTriangles: %d"), 
            MeshInfo.MeshID, *MeshInfo.MeshName, MeshInfo.NumTriangles);
        DrawDebugString(World, MeshInfo.Bounds.Origin, MeshText, nullptr, Color, Duration);
        
        // Draw wireframe if requested
        if (bShowWireframe)
        {
            for (int32 i = 0; i < MeshInfo.Indices.Num(); i += 3)
            {
                if (i + 2 < MeshInfo.Indices.Num())
                {
                    const FVector& V0 = MeshInfo.VertexPositions[MeshInfo.Indices[i]];
                    const FVector& V1 = MeshInfo.VertexPositions[MeshInfo.Indices[i + 1]];
                    const FVector& V2 = MeshInfo.VertexPositions[MeshInfo.Indices[i + 2]];
                    
                    DrawDebugLine(World, V0, V1, Color, false,
                        Duration, 0, 1.0f);
                    DrawDebugLine(World, V1, V2, Color, false,
                        Duration, 0, 1.0f);
                    DrawDebugLine(World, V2, V0, Color, false,
                        Duration, 0, 1.0f);
                }
            }
        }
        
        // Draw vertices if requested
        if (bShowVertices)
        {
            for (const FVector& Vertex : MeshInfo.VertexPositions)
            {
                DrawDebugPoint(World, Vertex, VertexSize, Color,
                    false, Duration);
            }
        }
    }
    
    UE_LOG(LogTemp, Display, TEXT("Visualized %d meshes with %d total "
                                  "triangles and %d total vertices"), 
        SceneMeshes.Num(), TotalTrianglesInScene, TotalPointsInScene);
}