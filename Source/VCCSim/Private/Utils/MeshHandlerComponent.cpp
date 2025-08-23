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

#include "Utils/MeshHandlerComponent.h"
#include "UObject/ConstructorHelpers.h"
#include "DataStruct_IO/Mesh.h"

UMeshHandlerComponent::UMeshHandlerComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
    CurrentBufferIndex = 0;
    bIsFirstMesh = true;

    bWantsInitializeComponent = true;
    bAutoActivate = true;

    // Make sure component can be moved
    Mobility = EComponentMobility::Movable;

    // Material will be loaded in BeginPlay
    MeshMaterial = nullptr;

    // Initialize mesh buffers
    for (int32 i = 0; i < 2; ++i)
    {
        MeshBuffers[i].bNeedsUpdate = false;
    }
}

void UMeshHandlerComponent::OnRegister()
{
    Super::OnRegister();

    // Create procedural mesh component
    MeshComponent = NewObject<UProceduralMeshComponent>(this);
    if (!MeshComponent)
    {
        UE_LOG(LogTemp, Error,
            TEXT("MeshHandler Failed to create ProceduralMeshComponent"));
        return;
    }
    MeshComponent->SetUsingAbsoluteLocation(true);
    MeshComponent->SetUsingAbsoluteRotation(true);
    MeshComponent->SetUsingAbsoluteScale(true);
    MeshComponent->SetWorldTransform(FTransform::Identity);
    
    MeshComponent->RegisterComponent();
    
    MeshComponent->SetVisibility(true);
    MeshComponent->bVisibleInSceneCaptureOnly = false;
    
    // Optimize for frequent updates
    MeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    MeshComponent->SetTranslucentSortPriority(100);
    MeshComponent->SetCastShadow(false);
    MeshComponent->SetMobility(EComponentMobility::Movable);
    MeshComponent->bUseAsyncCooking = true;
    MeshComponent->bReceivesDecals = false;
    
    // Load and set material
    FString MaterialPath =
        TEXT("/VCCSim/Materials/M_Dynamic_mesh.M_Dynamic_mesh");
    MeshMaterial = Cast<UMaterialInterface>(StaticLoadObject(
        UMaterialInterface::StaticClass(), nullptr, *MaterialPath));
    
    if (MeshMaterial)
    {
        MeshComponent->SetMaterial(0, MeshMaterial);
    }
    else
    {
        UE_LOG(LogTemp, Error,
            TEXT("MeshHandler Failed to load material"));
    }

    // Set up timer for mesh updates
    if (GetWorld())
    {
        GetWorld()->GetTimerManager().SetTimer(
            UpdateTimerHandle,
            this,
            &UMeshHandlerComponent::UpdateMeshInternal,
            0.0333f,
            true
        );
    }
    else
    {
        UE_LOG(LogTemp, Error,
            TEXT("World not available in OnRegister"));
    }
    
}

void UMeshHandlerComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    if (GetWorld())
    {
        GetWorld()->GetTimerManager().ClearTimer(UpdateTimerHandle);
    }
    Super::EndPlay(EndPlayReason);
}

void UMeshHandlerComponent::UpdateMeshFromGRPC(
    const uint8* MeshData, uint32 DataSize, const FTransform& Transform)
{
    // Only lock when accessing the write buffer
    const int32 WriteBufferIndex = 1 - CurrentBufferIndex;
    
    {
        FScopeLock Lock(&MeshUpdateLock);
        if (MeshBuffers[WriteBufferIndex].bNeedsUpdate)
        {
            UE_LOG(LogTemp, Warning,
                TEXT("Mesh buffer is still being processed. "
                     "Skipping update."));
            return;
        }
    }
    
    FMeshData& WriteBuffer = MeshBuffers[WriteBufferIndex];
    
    WriteBuffer.Vertices.Empty();
    WriteBuffer.Normals.Empty();
    WriteBuffer.Triangles.Empty();

    // Check minimum size for header
    if (DataSize < sizeof(FMeshHeader))
    {
        UE_LOG(LogTemp, Error,
            TEXT("Received mesh data is too small for header"));
        return;
    }

    const FMeshHeader* Header = reinterpret_cast<const FMeshHeader*>(MeshData);
    
    if (Header->Magic != 0x48534D55)
    {
        UE_LOG(LogTemp, Error, TEXT("Invalid mesh magic number"));
        return;
    }

    uint32 ExpectedSize = sizeof(FMeshHeader) + 
                         Header->VertexCount * sizeof(FCompactVertex) +
                         Header->IndexCount * sizeof(uint32);
    
    if (DataSize < ExpectedSize)
    {
        UE_LOG(LogTemp, Error, TEXT("Mesh data size mismatch. "
                                    "Expected: %d, Got: %d"), ExpectedSize, DataSize);
        return;
    }

    // Get pointers to vertex and index data
    const FCompactVertex* Vertices =
        reinterpret_cast<const FCompactVertex*>(MeshData + sizeof(FMeshHeader));
    const uint32* Indices =
        reinterpret_cast<const uint32*>(MeshData + sizeof(FMeshHeader) + 
        Header->VertexCount * sizeof(FCompactVertex));

    // Temporary arrays for base mesh
    TArray<FVector> BaseVertices;
    TArray<int32> BaseTriangles;
    BaseVertices.Reserve(Header->VertexCount);
    BaseTriangles.Reserve(Header->IndexCount);

    for (uint32 i = 0; i < Header->VertexCount; ++i)
    {
        BaseVertices.Add(FVector(
            Vertices[i].X * 100.0f,
            -Vertices[i].Y * 100.0f,
            Vertices[i].Z * 100.0f
        ));
    }

    for (uint32 i = 0; i < Header->IndexCount; i += 3)
    {
        BaseTriangles.Add(Indices[i]);
        BaseTriangles.Add(Indices[i + 2]);
        BaseTriangles.Add(Indices[i + 1]);
    }

    TArray<FVector> BaseNormals;
    CalculateVertexNormals(BaseVertices, BaseTriangles, BaseNormals);

    // Now create the extruded mesh
    const float ExtrusionDepth = 5.0f;
    const int32 NumVertices = BaseVertices.Num();

    WriteBuffer.Vertices.Reserve(NumVertices * 2);
    WriteBuffer.Normals.Reserve(NumVertices * 2);
    WriteBuffer.Triangles.Reserve(BaseTriangles.Num() * 2 + NumVertices * 6);

    for (int32 i = 0; i < NumVertices; ++i)
    {
        WriteBuffer.Vertices.Add(BaseVertices[i]);
        WriteBuffer.Normals.Add(BaseNormals[i]);

        WriteBuffer.Vertices.Add(BaseVertices[i] + (BaseNormals[i] * ExtrusionDepth));
        WriteBuffer.Normals.Add(BaseNormals[i]);
    }

    for (int32 i = 0; i < BaseTriangles.Num(); i += 3)
    {
        int32 i0 = BaseTriangles[i] * 2;
        int32 i1 = BaseTriangles[i + 1] * 2;
        int32 i2 = BaseTriangles[i + 2] * 2;

        WriteBuffer.Triangles.Add(i0);
        WriteBuffer.Triangles.Add(i1);
        WriteBuffer.Triangles.Add(i2);

        WriteBuffer.Triangles.Add(i0 + 1);
        WriteBuffer.Triangles.Add(i2 + 1);
        WriteBuffer.Triangles.Add(i1 + 1);
    }

    for (int32 i = 0; i < BaseTriangles.Num(); i += 3)
    {
        int32 i0 = BaseTriangles[i] * 2;
        int32 i1 = BaseTriangles[i + 1] * 2;
        int32 i2 = BaseTriangles[i + 2] * 2;

        WriteBuffer.Triangles.Add(i0);
        WriteBuffer.Triangles.Add(i0 + 1);
        WriteBuffer.Triangles.Add(i1);

        WriteBuffer.Triangles.Add(i1);
        WriteBuffer.Triangles.Add(i0 + 1);
        WriteBuffer.Triangles.Add(i1 + 1);

        WriteBuffer.Triangles.Add(i1);
        WriteBuffer.Triangles.Add(i1 + 1);
        WriteBuffer.Triangles.Add(i2);

        WriteBuffer.Triangles.Add(i2);
        WriteBuffer.Triangles.Add(i1 + 1);
        WriteBuffer.Triangles.Add(i2 + 1);

        WriteBuffer.Triangles.Add(i2);
        WriteBuffer.Triangles.Add(i2 + 1);
        WriteBuffer.Triangles.Add(i0);

        WriteBuffer.Triangles.Add(i0);
        WriteBuffer.Triangles.Add(i2 + 1);
        WriteBuffer.Triangles.Add(i0 + 1);
    }
    
    WriteBuffer.Transform = Transform;
    
    {
        FScopeLock Lock(&MeshUpdateLock);
        WriteBuffer.bNeedsUpdate = true;
    }
}

void UMeshHandlerComponent::UpdateMeshInternal()
{
    bool needsSwap = false;
    {
        FScopeLock Lock(&MeshUpdateLock);
        if (MeshBuffers[1 - CurrentBufferIndex].bNeedsUpdate)
        {
            needsSwap = true;
            // Swap buffers
            CurrentBufferIndex = 1 - CurrentBufferIndex;
        }
    }

    if (!needsSwap)
    {
        return;
    }
    
    FMeshData& ActiveBuffer = MeshBuffers[CurrentBufferIndex];

    // Ensure we have valid mesh component before proceeding
    if (!MeshComponent)
    {
        UE_LOG(LogTemp, Error, TEXT("MeshComponent is null during update!"));
        return;
    }

    if (!bIsFirstMesh)
    {
        MeshComponent->ClearMeshSection(0);
    }
    
    MeshComponent->CreateMeshSection_LinearColor(
        0,
        ActiveBuffer.Vertices,
        ActiveBuffer.Triangles,
        ActiveBuffer.Normals,
        EmptyUVs,
        EmptyColors,
        EmptyTangents,
        false
    );

    bIsFirstMesh = false;

    // Mark buffer as processed
    {
        FScopeLock Lock(&MeshUpdateLock);
        ActiveBuffer.bNeedsUpdate = false;
    }
}

void UMeshHandlerComponent::CalculateVertexNormals(
    const TArray<FVector>& InVertices,
    const TArray<int32>& InTriangles,
    TArray<FVector>& OutNormals)
{
    const int32 NumVertices = InVertices.Num();
    OutNormals.Init(FVector::ZeroVector, NumVertices);
    TArray<int32> NormalCounts;
    NormalCounts.Init(0, NumVertices);

    // Calculate face normals and accumulate them for vertices
    for (int32 i = 0; i < InTriangles.Num(); i += 3)
    {
        const FVector& V0 = InVertices[InTriangles[i]];
        const FVector& V1 = InVertices[InTriangles[i + 1]];
        const FVector& V2 = InVertices[InTriangles[i + 2]];

        // Calculate face normal
        FVector Normal = FVector::CrossProduct(V1 - V0, V2 - V0);
        Normal.Normalize();

        // Accumulate normals for each vertex
        OutNormals[InTriangles[i]] += Normal;
        OutNormals[InTriangles[i + 1]] += Normal;
        OutNormals[InTriangles[i + 2]] += Normal;

        // Count how many faces contribute to each vertex
        NormalCounts[InTriangles[i]]++;
        NormalCounts[InTriangles[i + 1]]++;
        NormalCounts[InTriangles[i + 2]]++;
    }

    // Average the normals
    for (int32 i = 0; i < NumVertices; ++i)
    {
        if (NormalCounts[i] > 0)
        {
            OutNormals[i] /= static_cast<float>(NormalCounts[i]);
            OutNormals[i].Normalize();
        }
        else
        {
            OutNormals[i] = FVector::UpVector;  // Default normal for isolated vertices
        }
    }
}
