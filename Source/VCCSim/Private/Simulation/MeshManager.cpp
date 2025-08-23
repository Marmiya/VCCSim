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

#include "Simulation/MeshManager.h"
#include "Utils/ConfigParser.h"
#include "DataStruct_IO/Mesh.h"
#include "Engine/World.h"
#include "GameFramework/Actor.h"

UFMeshManager::UFMeshManager()
{
    MeshMaterial = nullptr;
}

void UFMeshManager::RConfigure(const FVCCSimConfig& Config)
{
    // Find our owner actor
    OwnerActor = GetTypedOuter<AActor>();
    if (!OwnerActor)
    {
        UE_LOG(LogTemp, Error, TEXT("MeshManager: Failed to find owner actor!"));
        return;
    }

    // Load mesh material from config
    if (Config.VCCSim.MeshMaterial != "")
    {
        MeshMaterial = Cast<UMaterialInterface>(StaticLoadObject(
            UMaterialInterface::StaticClass(), nullptr,
            UTF8_TO_TCHAR(Config.VCCSim.MeshMaterial.c_str())));
    }

    if (MeshMaterial == nullptr)
    {
        // Use UE default material as fallback
        MeshMaterial = Cast<UMaterialInterface>(StaticLoadObject(
            UMaterialInterface::StaticClass(), nullptr,
            TEXT("/Engine/BasicShapes/BasicShapeMaterial")));
            
        if (MeshMaterial == nullptr)
        {
            UE_LOG(LogTemp, Error, TEXT("MeshManager: "
                                        "Failed to load default material!"));
        }
    }

    // Clear any existing meshes (in case RConfigure is called multiple times)
    ClearAllMeshes();
    
    // Set up timer for mesh updates
    if (OwnerActor->GetWorld())
    {
        OwnerActor->GetWorld()->GetTimerManager().SetTimer(
            UpdateTimerHandle,
            this,
            &UFMeshManager::UpdateMeshesInternal,
            MeshUpdateInterval,
            true
        );
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("MeshManager: "
                                    "World not available in RConfigure"));
    }
}

int32 UFMeshManager::AddGlobalMesh()
{
    if (!OwnerActor)
    {
        UE_LOG(LogTemp, Error, TEXT("MeshManager: "
                                    "Cannot add mesh, no owner actor!"));
        return -1;
    }
    
    if (IsInGameThread())
    {
        return AddGlobalMeshInternal();
    }
    
    TPromise<int32> Promise;
    TFuture<int32> Future = Promise.GetFuture();

    AsyncTask(ENamedThreads::GameThread,
        [this, Promise = MoveTemp(Promise)]() mutable
    {
        int32 MeshID = AddGlobalMeshInternal();
        Promise.SetValue(MeshID);
    });

    int32 MeshID = Future.Get();
    return MeshID;
}

bool UFMeshManager::RemoveGlobalMesh(int32 MeshID)
{
    // Check if the mesh ID exists
    {
        FScopeLock Lock(&MeshMapLock);
        if (!MeshComponents.Contains(MeshID) || !MeshInstanceData.Contains(MeshID))
        {
            UE_LOG(LogTemp, Warning, TEXT("MeshManager: Trying to remove "
                                          "non-existent mesh ID %d"), MeshID);
            return false;
        }
    }

    // Check if we're already on the game thread
    if (IsInGameThread())
    {
        return RemoveGlobalMeshInternal(MeshID);
    }
    
    TPromise<bool> Promise;
    TFuture<bool> Future = Promise.GetFuture();

    AsyncTask(ENamedThreads::GameThread,
        [this, MeshID, Promise = MoveTemp(Promise)]() mutable
    {
        bool Success = RemoveGlobalMeshInternal(MeshID);
        Promise.SetValue(Success);
    });

    // Wait for the mesh removal to complete
    bool Success = Future.Get();
    return Success;
}

bool UFMeshManager::UpdateMesh(
    int32 MeshID, const uint8* MeshData, uint32 DataSize, const FTransform& Transform)
{
    if (!MeshData || DataSize < sizeof(FMeshHeader))
    {
        UE_LOG(LogTemp, Error, TEXT("MeshManager: Invalid mesh data "
                                    "provided for ID %d"), MeshID);
        return false;
    }

    // Get the mesh instance data
    TSharedPtr<FMeshInstance> InstanceData = nullptr;
    FCriticalSection* UpdateLock = nullptr;
    
    {
        FScopeLock Lock(&MeshMapLock);
        if (!MeshInstanceData.Contains(MeshID) || !MeshUpdateLocks.Contains(MeshID))
        {
            UE_LOG(LogTemp, Warning, TEXT("MeshManager: Cannot update "
                                          "mesh with ID %d, not found"), MeshID);
            return false;
        }
        
        InstanceData = MeshInstanceData[MeshID];
        UpdateLock = MeshUpdateLocks[MeshID];
    }
    
    if (!InstanceData || !InstanceData->MeshComponent || !UpdateLock)
    {
        UE_LOG(LogTemp, Warning, TEXT("MeshManager: Invalid "
                                      "mesh instance data for ID %d"), MeshID);
        return false;
    }

    // Only lock when accessing the write buffer
    const int32 WriteBufferIndex = 1 - InstanceData->CurrentBufferIndex;
    
    {
        FScopeLock BufferLock(UpdateLock);
        if (InstanceData->MeshBuffers[WriteBufferIndex].bNeedsUpdate)
        {
            // Instead of skipping, we could queue updates if needed
            UE_LOG(LogTemp, Warning, TEXT("Mesh buffer for ID %d is "
                                          "still being processed. Skipping update."), MeshID);
            return false;
        }
        
        // Store the transform immediately - this is lightweight
        InstanceData->MeshBuffers[WriteBufferIndex].Transform = Transform;
        
        // Process mesh data asynchronously using a task graph
        FThreadSafeBool* ProcessingComplete = new FThreadSafeBool(false);
        InstanceData->MeshBuffers[WriteBufferIndex].ProcessingTaskComplete = ProcessingComplete;
        
        // Spawn the task to process the mesh data
        (new FAutoDeleteAsyncTask<FMeshProcessingWorker>(
            MeshData, 
            DataSize, 
            &InstanceData->MeshBuffers[WriteBufferIndex].Vertices,
            &InstanceData->MeshBuffers[WriteBufferIndex].Normals,
            &InstanceData->MeshBuffers[WriteBufferIndex].Triangles,
            this,
            MeshID,
            ProcessingComplete
        ))->StartBackgroundTask();
        
        // Mark for update - the actual mesh will be updated once processing is complete
        InstanceData->MeshBuffers[WriteBufferIndex].bNeedsUpdate = true;
    }
    
    return true;
}

void UFMeshManager::ClearAllMeshes()
{
    FScopeLock Lock(&MeshMapLock);
    
    // Destroy all procedural mesh components
    for (auto& Pair : MeshComponents)
    {
        if (Pair.Value)
        {
            Pair.Value->DestroyComponent();
        }
    }
    
    // Clean up all critical sections
    for (auto& Pair : MeshUpdateLocks)
    {
        if (Pair.Value)
        {
            delete Pair.Value;
        }
    }
    
    // Clear the maps
    MeshComponents.Empty();
    MeshInstanceData.Empty();
    MeshUpdateLocks.Empty();
    
    UE_LOG(LogTemp, Log, TEXT("MeshManager: Cleared all meshes"));
}

UProceduralMeshComponent* UFMeshManager::CreateProceduralMeshComponent()
{
    if (!OwnerActor || !OwnerActor->GetWorld())
    {
        UE_LOG(LogTemp, Error, TEXT("MeshManager: "
                                    "No valid world to create procedural mesh!"));
        return nullptr;
    }

    // Create new procedural mesh component
    UProceduralMeshComponent* NewMeshComponent =
        NewObject<UProceduralMeshComponent>(OwnerActor);
    if (!NewMeshComponent)
    {
        UE_LOG(LogTemp, Error, TEXT("MeshManager: "
                                    "Failed to create ProceduralMeshComponent!"));
        return nullptr;
    }

    // Configure the procedural mesh component
    NewMeshComponent->SetUsingAbsoluteLocation(true);
    NewMeshComponent->SetUsingAbsoluteRotation(true);
    NewMeshComponent->SetUsingAbsoluteScale(true);
    NewMeshComponent->SetWorldTransform(FTransform::Identity);
    
    NewMeshComponent->RegisterComponent();
    
    NewMeshComponent->SetVisibility(true);
    NewMeshComponent->bVisibleInSceneCaptureOnly = false;
    
    // Optimize for frequent updates
    NewMeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    NewMeshComponent->SetTranslucentSortPriority(100);
    NewMeshComponent->SetCastShadow(false);
    NewMeshComponent->SetMobility(EComponentMobility::Movable);
    NewMeshComponent->bUseAsyncCooking = true;
    NewMeshComponent->bReceivesDecals = false;
    
    // Set the material if available
    if (MeshMaterial)
    {
        NewMeshComponent->SetMaterial(0, MeshMaterial);
    }
    
    return NewMeshComponent;
}

FMeshInstance* UFMeshManager::FindMeshInstance(int32 MeshID)
{
    // Note: This function assumes the MeshMapLock is already acquired
    if (MeshInstanceData.Contains(MeshID))
    {
        return MeshInstanceData[MeshID].Get();
    }
    return nullptr;
}

void UFMeshManager::UpdateMeshesInternal()
{
    // OPTIMIZATION: Only check meshes that are flagged for updates
    TArray<int32> MeshesToUpdate;
    
    // Collect mesh IDs that need updating
    {
        FScopeLock Lock(&MeshMapLock);
        for (auto& Pair : MeshInstanceData)
        {
            int32 MeshID = Pair.Key;
            TSharedPtr<FMeshInstance> Instance = Pair.Value;
            
            if (!Instance)
                continue;
                
            // Check if either buffer needs an update
            const int32 WriteBufferIndex = 1 - Instance->CurrentBufferIndex;
            if (Instance->MeshBuffers[WriteBufferIndex].bNeedsUpdate)
            {
                // Check if async processing is complete
                FThreadSafeBool* ProcessingComplete =
                    Instance->MeshBuffers[WriteBufferIndex].ProcessingTaskComplete;
                if (ProcessingComplete && *ProcessingComplete)
                {
                    MeshesToUpdate.Add(MeshID);
                }
            }
        }
    }
    
    // Only process meshes that actually need updates
    for (int32 MeshID : MeshesToUpdate)
    {
        TSharedPtr<FMeshInstance> Instance = nullptr;
        UProceduralMeshComponent* MeshComponent = nullptr;
        FCriticalSection* UpdateLock = nullptr;
        
        {
            FScopeLock Lock(&MeshMapLock);
            if (MeshInstanceData.Contains(MeshID) && MeshComponents.Contains(MeshID)
                && MeshUpdateLocks.Contains(MeshID))
            {
                Instance = MeshInstanceData[MeshID];
                MeshComponent = MeshComponents[MeshID];
                UpdateLock = MeshUpdateLocks[MeshID];
            }
        }
        
        if (!Instance || !MeshComponent || !UpdateLock)
        {
            continue;
        }
        
        // Swap the buffers
        {
            FScopeLock InstanceLock(UpdateLock);
            Instance->CurrentBufferIndex = 1 - Instance->CurrentBufferIndex;
            
            // Clean up the processing complete flag
            delete Instance->MeshBuffers[Instance->CurrentBufferIndex].ProcessingTaskComplete;
            Instance->MeshBuffers[Instance->CurrentBufferIndex].ProcessingTaskComplete = nullptr;
        }
        
        FMeshBufferData& ActiveBuffer = Instance->MeshBuffers[Instance->CurrentBufferIndex];
        
        // Clear previous mesh section if not first mesh
        if (!Instance->bIsFirstMesh)
        {
            MeshComponent->ClearMeshSection(0);
        }
        
        // Empty arrays for unused mesh components
        TArray<FLinearColor> EmptyColors;
        TArray<FProcMeshTangent> EmptyTangents;
        
        // OPTIMIZATION: Use the AsyncTask to update the mesh on the game thread
        // without blocking other operations
        AsyncTask(ENamedThreads::GameThread,
            [MeshComponent, ActiveBuffer, Instance]() mutable
        {
            // Create mesh section
            MeshComponent->CreateMeshSection_LinearColor(
                0,
                ActiveBuffer.Vertices,
                ActiveBuffer.Triangles,
                ActiveBuffer.Normals,
                ActiveBuffer.UVs,
                TArray<FLinearColor>(),
                TArray<FProcMeshTangent>(),
                false
            );
            
            // Set transform
            MeshComponent->SetWorldTransform(ActiveBuffer.Transform);
            
            Instance->bIsFirstMesh = false;
        });
        
        // Mark buffer as processed
        {
            FScopeLock InstanceLock(UpdateLock);
            ActiveBuffer.bNeedsUpdate = false;
        }
    }
}

bool UFMeshManager::ProcessMeshData(const uint8* MeshData, uint32 DataSize, 
                                 TArray<FVector>& OutVertices, TArray<FVector>& OutNormals, 
                                 TArray<int32>& OutTriangles)
{
    // Check minimum size for header
    if (DataSize < sizeof(FMeshHeader))
    {
        UE_LOG(LogTemp, Error, TEXT("Received mesh data is too small for header"));
        return false;
    }

    const FMeshHeader* Header = reinterpret_cast<const FMeshHeader*>(MeshData);
    
    if (Header->Magic != 0x48534D55)
    {
        UE_LOG(LogTemp, Error, TEXT("Invalid mesh magic number"));
        return false;
    }

    uint32 ExpectedSize = sizeof(FMeshHeader) + 
                         Header->VertexCount * sizeof(FCompactVertex) +
                         Header->IndexCount * sizeof(uint32);
    
    if (DataSize < ExpectedSize)
    {
        UE_LOG(LogTemp, Error, TEXT("Mesh data size mismatch. "
                                    "Expected: %d, Got: %d"), ExpectedSize, DataSize);
        return false;
    }

    // Clear output arrays
    OutVertices.Empty();
    OutNormals.Empty();
    OutTriangles.Empty();

    // Get pointers to vertex and index data
    const FCompactVertex* Vertices =
        reinterpret_cast<const FCompactVertex*>(MeshData + sizeof(FMeshHeader));
    const uint32* Indices =
        reinterpret_cast<const uint32*>(MeshData + sizeof(FMeshHeader) + 
        Header->VertexCount * sizeof(FCompactVertex));

    // Reserve space for the mesh data
    OutVertices.Reserve(Header->VertexCount);
    OutTriangles.Reserve(Header->IndexCount);

    // Process vertices - convert to Unreal Engine coordinate system
    for (uint32 i = 0; i < Header->VertexCount; ++i)
    {
        OutVertices.Add(FVector(
            Vertices[i].X * 100.0f,     // Scale and convert to UE coordinates
            -Vertices[i].Y * 100.0f,    // Flip Y axis for UE coordinate system
            Vertices[i].Z * 100.0f
        ));
    }

    // Process indices - swap winding order for UE
    for (uint32 i = 0; i < Header->IndexCount; i += 3)
    {
        OutTriangles.Add(Indices[i]);       // First vertex
        OutTriangles.Add(Indices[i + 2]);   // Third vertex (swapped with second)
        OutTriangles.Add(Indices[i + 1]);   // Second vertex (swapped with third)
    }

    // Calculate vertex normals
    CalculateVertexNormals(OutVertices, OutTriangles, OutNormals);
    
    return true;
}

void UFMeshManager::CalculateVertexNormals(const TArray<FVector>& InVertices, 
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

int32 UFMeshManager::AddGlobalMeshInternal()
{
    check(IsInGameThread());

    // Create new procedural mesh component (now guaranteed to be on game thread)
    UProceduralMeshComponent* NewMeshComponent = CreateProceduralMeshComponent();
    if (!NewMeshComponent)
    {
        UE_LOG(LogTemp, Error, TEXT("MeshManager: "
                                    "Failed to create procedural mesh component!"));
        return -1;
    }

    // Assign a unique ID to the mesh
    int32 MeshID;
    {
        FScopeLock Lock(&MeshMapLock);
        MeshID = NextMeshID++;
        
        // Create new instance data
        TSharedPtr<FMeshInstance> NewInstance = MakeShared<FMeshInstance>();
        NewInstance->MeshComponent = NewMeshComponent;
        NewInstance->bIsActive = true;
        
        // Create critical section for this mesh
        FCriticalSection* NewLock = new FCriticalSection();
        
        // Store everything
        MeshComponents.Add(MeshID, NewMeshComponent);
        MeshInstanceData.Add(MeshID, NewInstance);
        MeshUpdateLocks.Add(MeshID, NewLock);
    }

    UE_LOG(LogTemp, Log, TEXT("MeshManager: "
                              "Added global mesh with ID %d"), MeshID);
    return MeshID;
}

bool UFMeshManager::RemoveGlobalMeshInternal(int32 MeshID)
{
    check(IsInGameThread());
    
    UProceduralMeshComponent* MeshComponent = nullptr;
    FCriticalSection* LockToDelete = nullptr;
    
    // Get the component and lock
    {
        FScopeLock Lock(&MeshMapLock);
        
        if (!MeshComponents.Contains(MeshID) || !MeshInstanceData.Contains(MeshID))
        {
            UE_LOG(LogTemp, Warning, TEXT("MeshManager: Trying to remove "
                                          "non-existent mesh ID %d"), MeshID);
            return false;
        }
        
        MeshComponent = MeshComponents[MeshID];
        LockToDelete = MeshUpdateLocks[MeshID];
        
        // Remove from the maps
        MeshComponents.Remove(MeshID);
        MeshInstanceData.Remove(MeshID);
        MeshUpdateLocks.Remove(MeshID);
    }
    
    // Destroy the procedural mesh component (must be done on game thread)
    if (MeshComponent)
    {
        MeshComponent->DestroyComponent();
    }
    
    // Clean up the critical section
    if (LockToDelete)
    {
        delete LockToDelete;
    }
    
    UE_LOG(LogTemp, Log, TEXT("MeshManager: "
                              "Removed global mesh with ID %d"), MeshID);
    return true;
}