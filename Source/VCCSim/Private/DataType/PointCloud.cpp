#include "DataType/PointCloud.h"
#include "Engine/Engine.h"
#include "Misc/DateTime.h"

// FPLYProperty Constructor
FPLYLoader::FPLYProperty::FPLYProperty(const FString& InType, const FString& InName)
    : Type(InType), Name(InName)
{
    // Determine size based on type
    if (Type == TEXT("float")) 
        Size = 4;
    else if (Type == TEXT("double")) 
        Size = 8;
    else if (Type == TEXT("uchar") || Type == TEXT("char")) 
        Size = 1;
    else if (Type == TEXT("ushort") || Type == TEXT("short")) 
        Size = 2;
    else if (Type == TEXT("uint") || Type == TEXT("int")) 
        Size = 4;
    else 
        Size = 4; // Default
}

// Main PLY loading function - REMOVED CoordinateScale parameter
FPLYLoader::FPLYLoadResult FPLYLoader::LoadPLYFile(const FString& FilePath, 
                                                   const FLinearColor& DefaultColor)
{
    FPLYLoadResult Result;
    
    // Validate file path
    if (!FPaths::FileExists(FilePath))
    {
        Result.ErrorMessage = FString::Printf(TEXT("PLY file not found: %s"), *FilePath);
        UE_LOG(LogTemp, Error, TEXT("%s"), *Result.ErrorMessage);
        return Result;
    }

    // Read file as text to parse header
    TArray<FString> FileLines;
    if (!FFileHelper::LoadFileToStringArray(FileLines, *FilePath))
    {
        Result.ErrorMessage = FString::Printf(TEXT("Failed to read PLY file: %s"), *FilePath);
        UE_LOG(LogTemp, Error, TEXT("%s"), *Result.ErrorMessage);
        return Result;
    }

    // Validate PLY format
    if (FileLines.Num() == 0 || !FileLines[0].StartsWith(TEXT("ply")))
    {
        Result.ErrorMessage = TEXT("Invalid PLY file - missing 'ply' header");
        UE_LOG(LogTemp, Error, TEXT("%s"), *Result.ErrorMessage);
        return Result;
    }

    // Find header end
    int32 HeaderEndIndex = FindHeaderEnd(FileLines);
    if (HeaderEndIndex == INDEX_NONE)
    {
        Result.ErrorMessage = TEXT("Invalid PLY file - no 'end_header' found");
        UE_LOG(LogTemp, Error, TEXT("%s"), *Result.ErrorMessage);
        return Result;
    }

    // Detect format
    EPLYFormat Format = DetectPLYFormat(FileLines);
    if (Format == EPLYFormat::Unknown)
    {
        Result.ErrorMessage = TEXT("Unsupported PLY format");
        UE_LOG(LogTemp, Error, TEXT("%s"), *Result.ErrorMessage);
        return Result;
    }

    // Parse properties and vertex count
    TArray<FPLYProperty> Properties = ParsePLYProperties(FileLines);
    int32 VertexCount = GetVertexCount(FileLines);
    
    if (VertexCount <= 0)
    {
        Result.ErrorMessage = TEXT("Invalid or missing vertex count in PLY file");
        UE_LOG(LogTemp, Error, TEXT("%s"), *Result.ErrorMessage);
        return Result;
    }

    // Check for color properties
    Result.bHasColors = HasColorProperties(Properties);
    
    // Reserve space for points
    Result.Points.Reserve(VertexCount);
    
    UE_LOG(LogTemp, Warning, TEXT("Loading PLY: Format=%s, Vertices=%d, HasColors=%s"), 
        Format == EPLYFormat::ASCII ? TEXT("ASCII") : TEXT("Binary"), 
        VertexCount, 
        Result.bHasColors ? TEXT("Yes") : TEXT("No"));

    // Load data based on format - NO coordinate scaling
    bool bLoadSuccess = false;
    if (Format == EPLYFormat::ASCII)
    {
        bLoadSuccess = LoadASCIIPLY(FileLines, HeaderEndIndex, VertexCount, Properties,
                                   DefaultColor, Result.Points, Result.bHasColors);
    }
    else
    {
        // Calculate header size in bytes for binary files
        FString HeaderText;
        for (int32 i = 0; i <= HeaderEndIndex; ++i)
        {
            HeaderText += FileLines[i] + TEXT("\n");
        }
        int32 HeaderSize = FTCHARToUTF8(HeaderText.GetCharArray().GetData()).Length();
        
        bLoadSuccess = LoadBinaryPLY(FilePath, HeaderSize, VertexCount, Properties, Format,
                                    DefaultColor, Result.Points, Result.bHasColors);
    }

    Result.bSuccess = bLoadSuccess;
    Result.PointCount = Result.Points.Num();
    
    if (Result.bSuccess)
    {
        UE_LOG(LogTemp, Warning, TEXT("Successfully loaded %d points from PLY file (no coordinate transform)"), Result.PointCount);
    }
    else
    {
        Result.ErrorMessage = TEXT("Failed to parse PLY data");
        UE_LOG(LogTemp, Error, TEXT("%s"), *Result.ErrorMessage);
    }

    return Result;
}

bool FPLYLoader::LoadASCIIPLY(const TArray<FString>& Lines, 
                             int32 HeaderEndIndex, 
                             int32 VertexCount, 
                             const TArray<FPLYProperty>& Properties,
                             const FLinearColor& DefaultColor,
                             TArray<FRatPoint>& OutPoints,
                             bool bHasColors)
{
    int32 CurrentVertexIndex = 0;
    
    for (int32 LineIndex = HeaderEndIndex + 1; LineIndex < Lines.Num() && CurrentVertexIndex < VertexCount; ++LineIndex)
    {
        TArray<FString> Components;
        Lines[LineIndex].ParseIntoArray(Components, TEXT(" "), true);
        
        if (Components.Num() >= Properties.Num())
        {
            FVector Position = FVector::ZeroVector;
            FLinearColor Color = DefaultColor;
            
            // Parse each property
            for (int32 PropIndex = 0; PropIndex < Properties.Num() && PropIndex < Components.Num(); ++PropIndex)
            {
                const FPLYProperty& Prop = Properties[PropIndex];
                const FString& Value = Components[PropIndex];
                
                if (Prop.Name == TEXT("x"))
                {
                    Position.X = FCString::Atod(*Value); // NO scaling
                }
                else if (Prop.Name == TEXT("y"))
                {
                    Position.Y = FCString::Atod(*Value); // NO scaling
                }
                else if (Prop.Name == TEXT("z"))
                {
                    Position.Z = FCString::Atod(*Value); // NO scaling
                }
                else if (Prop.Name == TEXT("red") && bHasColors)
                {
                    Color.R = FMath::Clamp(FCString::Atoi(*Value) / 255.0f, 0.0f, 1.0f);
                }
                else if (Prop.Name == TEXT("green") && bHasColors)
                {
                    Color.G = FMath::Clamp(FCString::Atoi(*Value) / 255.0f, 0.0f, 1.0f);
                }
                else if (Prop.Name == TEXT("blue") && bHasColors)
                {
                    Color.B = FMath::Clamp(FCString::Atoi(*Value) / 255.0f, 0.0f, 1.0f);
                }
            }
            
            OutPoints.Add(FRatPoint(Position, Color));
            CurrentVertexIndex++;
        }
    }
    
    return CurrentVertexIndex == VertexCount;
}

bool FPLYLoader::LoadBinaryPLY(const FString& FilePath, 
                              int32 HeaderSize, 
                              int32 VertexCount, 
                              const TArray<FPLYProperty>& Properties, 
                              EPLYFormat Format,
                              const FLinearColor& DefaultColor,
                              TArray<FRatPoint>& OutPoints,
                              bool bHasColors)
{
    // Calculate record size
    int32 RecordSize = 0;
    for (const FPLYProperty& Prop : Properties)
    {
        RecordSize += Prop.Size;
    }
    
    // Read binary data
    TArray<uint8> FileData;
    if (!FFileHelper::LoadFileToArray(FileData, *FilePath))
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to read binary PLY file"));
        return false;
    }
    
    int32 ExpectedDataSize = HeaderSize + (RecordSize * VertexCount);
    if (FileData.Num() < ExpectedDataSize)
    {
        UE_LOG(LogTemp, Error, TEXT("PLY file too small: expected %d bytes, got %d bytes"), 
            ExpectedDataSize, FileData.Num());
        return false;
    }
    
    bool bLittleEndian = (Format == EPLYFormat::BinaryLittleEndian);
    const uint8* DataStart = FileData.GetData() + HeaderSize;
    
    // Parse binary data
    for (int32 VertexIndex = 0; VertexIndex < VertexCount; ++VertexIndex)
    {
        const uint8* RecordStart = DataStart + (VertexIndex * RecordSize);
        int32 ByteOffset = 0;
        
        FVector Position = FVector::ZeroVector;
        FLinearColor Color = DefaultColor;
        
        // Parse each property in the record
        for (const FPLYProperty& Prop : Properties)
        {
            const uint8* PropertyData = RecordStart + ByteOffset;
            
            if (Prop.Name == TEXT("x"))
            {
                if (Prop.Type == TEXT("double"))
                {
                    Position.X = ReadDouble(PropertyData, bLittleEndian); // NO scaling
                }
                else if (Prop.Type == TEXT("float"))
                {
                    Position.X = ReadFloat(PropertyData, bLittleEndian); // NO scaling
                }
            }
            else if (Prop.Name == TEXT("y"))
            {
                if (Prop.Type == TEXT("double"))
                {
                    Position.Y = ReadDouble(PropertyData, bLittleEndian); // NO scaling
                }
                else if (Prop.Type == TEXT("float"))
                {
                    Position.Y = ReadFloat(PropertyData, bLittleEndian); // NO scaling
                }
            }
            else if (Prop.Name == TEXT("z"))
            {
                if (Prop.Type == TEXT("double"))
                {
                    Position.Z = ReadDouble(PropertyData, bLittleEndian); // NO scaling
                }
                else if (Prop.Type == TEXT("float"))
                {
                    Position.Z = ReadFloat(PropertyData, bLittleEndian); // NO scaling
                }
            }
            else if (Prop.Name == TEXT("red") && bHasColors)
            {
                Color.R = FMath::Clamp(ReadUChar(PropertyData) / 255.0f, 0.0f, 1.0f);
            }
            else if (Prop.Name == TEXT("green") && bHasColors)
            {
                Color.G = FMath::Clamp(ReadUChar(PropertyData) / 255.0f, 0.0f, 1.0f);
            }
            else if (Prop.Name == TEXT("blue") && bHasColors)
            {
                Color.B = FMath::Clamp(ReadUChar(PropertyData) / 255.0f, 0.0f, 1.0f);
            }
            
            ByteOffset += Prop.Size;
        }
        
        OutPoints.Add(FRatPoint(Position, Color));
    }
    
    return true;
}

FPLYLoader::EPLYFormat FPLYLoader::DetectPLYFormat(const TArray<FString>& HeaderLines)
{
    for (const FString& Line : HeaderLines)
    {
        if (Line.StartsWith(TEXT("format ascii")))
        {
            return EPLYFormat::ASCII;
        }
        else if (Line.StartsWith(TEXT("format binary_little_endian")))
        {
            return EPLYFormat::BinaryLittleEndian;
        }
        else if (Line.StartsWith(TEXT("format binary_big_endian")))
        {
            return EPLYFormat::BinaryBigEndian;
        }
    }
    return EPLYFormat::Unknown;
}

TArray<FPLYLoader::FPLYProperty> FPLYLoader::ParsePLYProperties(const TArray<FString>& HeaderLines)
{
    TArray<FPLYProperty> Properties;
    bool bInVertexElement = false;
    
    for (const FString& Line : HeaderLines)
    {
        if (Line.StartsWith(TEXT("element vertex")))
        {
            bInVertexElement = true;
        }
        else if (Line.StartsWith(TEXT("element")) && !Line.StartsWith(TEXT("element vertex")))
        {
            bInVertexElement = false;
        }
        else if (bInVertexElement && Line.StartsWith(TEXT("property")))
        {
            TArray<FString> Parts;
            Line.ParseIntoArray(Parts, TEXT(" "), true);
            
            if (Parts.Num() >= 3)
            {
                FString Type = Parts[1];
                FString Name = Parts[2];
                Properties.Add(FPLYProperty(Type, Name));
            }
        }
        else if (Line == TEXT("end_header"))
        {
            break;
        }
    }
    
    return Properties;
}

int32 FPLYLoader::FindHeaderEnd(const TArray<FString>& Lines)
{
    for (int32 i = 0; i < Lines.Num(); ++i)
    {
        if (Lines[i] == TEXT("end_header"))
        {
            return i;
        }
    }
    return INDEX_NONE;
}

int32 FPLYLoader::GetVertexCount(const TArray<FString>& HeaderLines)
{
    for (const FString& Line : HeaderLines)
    {
        if (Line.StartsWith(TEXT("element vertex")))
        {
            FString VertexCountStr = Line.RightChop(14).TrimStartAndEnd();
            return FCString::Atoi(*VertexCountStr);
        }
    }
    return 0;
}

bool FPLYLoader::HasColorProperties(const TArray<FPLYProperty>& Properties)
{
    bool bHasRed = Properties.ContainsByPredicate([](const FPLYProperty& Prop) { return Prop.Name == TEXT("red"); });
    bool bHasGreen = Properties.ContainsByPredicate([](const FPLYProperty& Prop) { return Prop.Name == TEXT("green"); });
    bool bHasBlue = Properties.ContainsByPredicate([](const FPLYProperty& Prop) { return Prop.Name == TEXT("blue"); });
    return bHasRed && bHasGreen && bHasBlue;
}

// Binary reading helper functions (unchanged)
float FPLYLoader::ReadFloat(const uint8* Data, bool bLittleEndian)
{
    uint32 IntValue;
    if (bLittleEndian)
    {
        IntValue = Data[0] | (Data[1] << 8) | (Data[2] << 16) | (Data[3] << 24);
    }
    else
    {
        IntValue = (Data[0] << 24) | (Data[1] << 16) | (Data[2] << 8) | Data[3];
    }
    return *reinterpret_cast<const float*>(&IntValue);
}

double FPLYLoader::ReadDouble(const uint8* Data, bool bLittleEndian)
{
    uint64 LongValue;
    if (bLittleEndian)
    {
        LongValue = static_cast<uint64>(Data[0]) |
                   (static_cast<uint64>(Data[1]) << 8) |
                   (static_cast<uint64>(Data[2]) << 16) |
                   (static_cast<uint64>(Data[3]) << 24) |
                   (static_cast<uint64>(Data[4]) << 32) |
                   (static_cast<uint64>(Data[5]) << 40) |
                   (static_cast<uint64>(Data[6]) << 48) |
                   (static_cast<uint64>(Data[7]) << 56);
    }
    else
    {
        LongValue = (static_cast<uint64>(Data[0]) << 56) |
                   (static_cast<uint64>(Data[1]) << 48) |
                   (static_cast<uint64>(Data[2]) << 40) |
                   (static_cast<uint64>(Data[3]) << 32) |
                   (static_cast<uint64>(Data[4]) << 24) |
                   (static_cast<uint64>(Data[5]) << 16) |
                   (static_cast<uint64>(Data[6]) << 8) |
                   static_cast<uint64>(Data[7]);
    }
    return *reinterpret_cast<const double*>(&LongValue);
}

uint8 FPLYLoader::ReadUChar(const uint8* Data)
{
    return *Data;
}

uint16 FPLYLoader::ReadUShort(const uint8* Data, bool bLittleEndian)
{
    if (bLittleEndian)
    {
        return Data[0] | (Data[1] << 8);
    }
    else
    {
        return (Data[0] << 8) | Data[1];
    }
}

uint32 FPLYLoader::ReadUInt(const uint8* Data, bool bLittleEndian)
{
    if (bLittleEndian)
    {
        return Data[0] | (Data[1] << 8) | (Data[2] << 16) | (Data[3] << 24);
    }
    else
    {
        return (Data[0] << 24) | (Data[1] << 16) | (Data[2] << 8) | Data[3];
    }
}

int32 FPLYLoader::ReadInt(const uint8* Data, bool bLittleEndian)
{
    uint32 UIntValue = ReadUInt(Data, bLittleEndian);
    return *reinterpret_cast<const int32*>(&UIntValue);
}