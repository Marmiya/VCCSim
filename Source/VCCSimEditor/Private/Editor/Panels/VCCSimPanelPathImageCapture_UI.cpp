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

#include "Editor/Panels/VCCSimPanelPathImageCapture.h"
#include "Editor/Panels/VCCSimPanelSelection.h"
#include "Utils/VCCSimUIHelpers.h"
#include "Pawns/FlashPawn.h"
#include "Styling/AppStyle.h"
#include "Styling/CoreStyle.h"
#include "Widgets/Layout/SBorder.h"

// ============================================================================
// UI CONSTRUCTION
// ============================================================================

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreatePathImageCapturePanel()
{
    return FVCCSimUIHelpers::CreateCollapsibleSection(
        "Path Configuration & Image Capture",
        SNew(SVerticalBox)

        // Path Configuration Section
        +SVerticalBox::Slot()
        .AutoHeight()
        [
            CreatePathConfigSection()
        ]

        +SVerticalBox::Slot()
        .MaxHeight(1)
        .Padding(FMargin(0, 8, 0, 8))
        [
           FVCCSimUIHelpers::CreateSeparator()
        ]

        // Image Capture Section
        +SVerticalBox::Slot()
        .AutoHeight()
        [
            CreateImageCaptureSection()
        ]

        +SVerticalBox::Slot()
        .MaxHeight(1)
        .Padding(FMargin(0, 8, 0, 8))
        [
           FVCCSimUIHelpers::CreateSeparator()
        ]

        // Dataset Configuration Section
        +SVerticalBox::Slot()
        .AutoHeight()
        [
            CreateDatasetConfigSection()
        ]

        +SVerticalBox::Slot()
        .MaxHeight(1)
        .Padding(FMargin(0, 4, 0, 4))
        [
            FVCCSimUIHelpers::CreateSeparator()
        ]

        // Lighting Schedule Section
        +SVerticalBox::Slot()
        .AutoHeight()
        [
            CreateLightingScheduleSection()
        ]

        +SVerticalBox::Slot()
        .MaxHeight(1)
        .Padding(FMargin(0, 4, 0, 4))
        [
            FVCCSimUIHelpers::CreateSeparator()
        ]

        // Dataset Capture Section
        +SVerticalBox::Slot()
        .AutoHeight()
        [
            CreateDatasetCaptureSection()
        ],

        bPathImageCaptureSectionExpanded
    );
}

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreatePathConfigSection()
{
    return SNew(SVerticalBox)

    // Orbit parameters (targets come from the shared list in the Object Selection panel)
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 4, 0, 4))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 4, 0))
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
                TEXT("Margin (cm)"), OrbitMarginSpinBox, OrbitMarginValue, OrbitMargin, 50.f, 50.f)
        ]
        +SHorizontalBox::Slot().FillWidth(1.f)
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
                TEXT("Survey Overlap"), OrbitSurveyOverlapSpinBox, OrbitSurveyOverlapValue, OrbitSurveyOverlap, 0.f, 0.05f)
        ]
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 0, 0, 4))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 4, 0))
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
                TEXT("H-FOV (deg)"), OrbitCameraHFOVSpinBox, OrbitCameraHFOVValue, OrbitCameraHFOV, 5.f, 5.f)
        ]
        +SHorizontalBox::Slot().FillWidth(1.f)
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
                TEXT("Start H (cm)"), OrbitStartHeightSpinBox, OrbitStartHeightValue, OrbitStartHeight, 0.f, 50.f)
        ]
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 0, 0, 4))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 4, 0))
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
                TEXT("H-Overlap"), OrbitHOverlapSpinBox, OrbitHOverlapValue, OrbitHOverlap, 0.f, 0.05f)
        ]
        +SHorizontalBox::Slot().FillWidth(1.f)
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
                TEXT("V-Overlap"), OrbitVOverlapSpinBox, OrbitVOverlapValue, OrbitVOverlap, 0.f, 0.05f)
        ]
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 0, 0, 4))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 4, 0))
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
                TEXT("Nadir Alt (cm)"), OrbitNadirAltSpinBox, OrbitNadirAltValue, OrbitNadirAlt, 100.f, 100.f)
        ]
        +SHorizontalBox::Slot().FillWidth(1.f)
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowFloat(
                TEXT("Tilt (deg)"), OrbitNadirTiltSpinBox, OrbitNadirTiltValue, OrbitNadirTiltAngle, 0.f, 1.f)
        ]
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 0, 0, 4))
    [
        SNew(SHorizontalBox)
        +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 8, 0))
        [
            FVCCSimUIHelpers::CreateNumericPropertyRowInt32(
                TEXT("Oblique Rings"), OrbitObliqueRingsSpinBox, OrbitObliqueRingsValue, OrbitObliqueRings, 0, 1)
        ]
        +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(0, 0, 4, 0))
        [
            SNew(SCheckBox)
            .ToolTipText(FText::FromString(TEXT(
                "5-lens oblique survey: a straight-down nadir pass PLUS four passes tilted by 'Tilt (deg)' "
                "toward N/S/E/W (~5x the survey poses). Off = a single lens pitched by 'Tilt (deg)' (set "
                "Tilt to 0 for straight down). Strip spacing always uses the tilted footprint to hold overlap.")))
            .IsChecked_Lambda([this]() { return bOrbitIncludeOblique ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
            .OnCheckStateChanged_Lambda([this](ECheckBoxState S) { bOrbitIncludeOblique = (S == ECheckBoxState::Checked); })
        ]
        +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(0, 0, 12, 0))
        [
            SNew(STextBlock).Text(FText::FromString(TEXT("Oblique")))
        ]
        +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(0, 0, 4, 0))
        [
            SNew(SCheckBox)
            .ToolTipText(FText::FromString(TEXT(
                "Also generate the per-building facade orbit rings. Off = region oblique/nadir survey only.")))
            .IsChecked_Lambda([this]() { return bOrbitSideOrbit ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
            .OnCheckStateChanged_Lambda([this](ECheckBoxState S) { bOrbitSideOrbit = (S == ECheckBoxState::Checked); })
        ]
        +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center)
        [
            SNew(STextBlock).Text(FText::FromString(TEXT("Side Orbit")))
        ]
    ]

    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        FVCCSimUIHelpers::CreateSeparator()
    ]

    // Load/Save Pose buttons
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 4, 0, 4))
    [
        CreatePoseFileButtons()
    ]

    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        FVCCSimUIHelpers::CreateSeparator()
    ]

    // Action buttons
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(FMargin(0, 4, 0, 2))
    [
        CreatePoseActionButtons()
    ];
}

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreateImageCaptureSection()
{
    return SNew(SVerticalBox)

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(0, 0, 0, 4)
    [
        FVCCSimUIHelpers::CreatePropertyRow(TEXT("Tick (s)"),
            SNew(SHorizontalBox)
            + SHorizontalBox::Slot()
            .FillWidth(1.f)
            .Padding(0, 0, 6, 0)
            [
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5, 5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(CaptureTickIntervalSpinBox, SNumericEntryBox<float>)
                    .Value_Lambda([this]() { return CaptureTickIntervalValue; })
                    .MinValue(0.05f).MaxValue(5.f).Delta(0.05f).AllowSpin(true)
                    .ToolTipText(FText::FromString(TEXT(
                        "Interval between capture ticks during auto/dataset capture. "
                        "Takes effect immediately, even while a capture is running.")))
                    .OnValueChanged_Lambda([this](float Val)
                    {
                        CaptureTickInterval = FMath::Clamp(Val, 0.05f, 5.f);
                        CaptureTickIntervalValue = CaptureTickInterval;
                        if (bAutoCaptureInProgress && GEditor)
                        {
                            GEditor->GetTimerManager()->SetTimer(
                                AutoCaptureTimerHandle,
                                FTimerDelegate::CreateLambda([this]() { TickCaptureSession(); }),
                                CaptureTickInterval,
                                true);
                        }
                    })
                ]
            ]
            + SHorizontalBox::Slot()
            .AutoWidth()
            .VAlign(VAlign_Center)
            .Padding(0, 0, 6, 0)
            [
                SNew(STextBlock)
                .Text(FText::FromString(TEXT("Warmup")))
            ]
            + SHorizontalBox::Slot()
            .FillWidth(1.f)
            [
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5, 5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(PoseWarmupFramesSpinBox, SNumericEntryBox<int32>)
                    .Value_Lambda([this]() { return PoseWarmupFramesValue; })
                    .MinValue(1).MaxValue(30).Delta(1).AllowSpin(true)
                    .ToolTipText(FText::FromString(TEXT(
                        "Throwaway frames rendered at each pose before capture, so occlusion culling and "
                        "temporal state converge. Higher = fewer half-loaded buildings but slower capture. "
                        "Takes effect from the next pose.")))
                    .OnValueChanged_Lambda([this](int32 Val)
                    {
                        PoseWarmupFrames = FMath::Clamp(Val, 1, 30);
                        PoseWarmupFramesValue = PoseWarmupFrames;
                    })
                ]
            ]
        )
    ]

    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(0, 0, 0, 4)
    [
        CreateMovementButtons()
    ]
    
    +SVerticalBox::Slot()
    .MaxHeight(1)
    [
        FVCCSimUIHelpers::CreateSeparator()
    ]
    
    +SVerticalBox::Slot()
    .AutoHeight()
    .Padding(0, 4, 0, 0)
    [
        CreateCaptureButtons()
    ];
}

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreatePoseFileButtons()
{
    return SNew(SHorizontalBox)
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(4, 2))
        .Text(FText::FromString("Load Predefined Pose"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() { return OnLoadPoseClicked(); })
        .IsEnabled_Lambda([this]() {
            return SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid();
        })
    ]
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(4, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(4, 2))
        .HAlign(HAlign_Center)
        .Text(FText::FromString("Save Generated Pose"))
        .OnClicked_Lambda([this]() { return OnSavePoseClicked(); })
        .IsEnabled_Lambda([this]() {
            return SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid() && 
                   SelectionManager.Pin()->GetSelectedFlashPawn()->GetPoseCount() > 0;
        })
    ];
}
TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreatePoseActionButtons()
{
    return SNew(SHorizontalBox)
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Success")
        .ContentPadding(FMargin(5, 2))
        .Text(FText::FromString("Generate Poses"))
        .HAlign(HAlign_Center)
        .ToolTipText(FText::FromString(TEXT("Orbit targets come from the Target Actors list in the Object Selection panel")))
        .OnClicked_Lambda([this]() { return OnGeneratePosesClicked(); })
        .IsEnabled_Lambda([this]() {
            return SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid() &&
                SelectionManager.Pin()->HasEnabledTargetActors();
        })
    ]
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(4, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SAssignNew(VisualizePathButton, SButton)
        .ButtonStyle(bPathVisualized ? 
           &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger") : 
           &FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"))
        .ContentPadding(FMargin(5, 2))
        .HAlign(HAlign_Center)
        .Text_Lambda([this]() {
            return FText::FromString(bPathVisualized ? "Hide Path" : "Show Path");
        })
        .OnClicked_Lambda([this]() { return OnTogglePathVisualizationClicked(); })
        .IsEnabled_Lambda([this]() {
            return SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid() && !bPathNeedsUpdate;
        })
    ];
}

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreateMovementButtons()
{
    return SNew(SHorizontalBox)
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(5, 2))
        .Text(FText::FromString("Move Back"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() {
            if (SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid())
            {
                AFlashPawn* Pawn = SelectionManager.Pin()->GetSelectedFlashPawn().Get();
                Pawn->MoveBackward();
                Pawn->PostEditMove(true);
            }
            return FReply::Handled();
        })
        .IsEnabled_Lambda([this]() {
            return SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid();
        })
    ]
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(4, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Default")
        .ContentPadding(FMargin(5, 2))
        .Text(FText::FromString("Move Next"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() {
            if (SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid())
            {
                AFlashPawn* Pawn = SelectionManager.Pin()->GetSelectedFlashPawn().Get();
                Pawn->MoveForward();
                Pawn->PostEditMove(true);
            }
            return FReply::Handled();
        })
        .IsEnabled_Lambda([this]() {
            return SelectionManager.IsValid() &&
                SelectionManager.Pin()->GetSelectedFlashPawn().IsValid();
        })
    ];
}

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreateCaptureButtons()
{
    return SNew(SHorizontalBox)
    
    // Single Capture button
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(0, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Success")
        .ContentPadding(FMargin(5, 2))
        .Text(FText::FromString("Capture This Pose"))
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() { return OnCaptureImagesClicked(); })
        .IsEnabled_Lambda([this]() {
            if (!SelectionManager.IsValid()) return false;
            auto SelectionManagerPin = SelectionManager.Pin();
            if (!SelectionManagerPin.IsValid() || !SelectionManagerPin->GetSelectedFlashPawn().IsValid()) return false;
            
            return (SelectionManagerPin->IsUsingRGBCamera() && SelectionManagerPin->HasRGBCamera()) ||
                   (SelectionManagerPin->IsUsingDepthCamera() && SelectionManagerPin->HasDepthCamera()) ||
                   (SelectionManagerPin->IsUsingNormalCamera() && SelectionManagerPin->HasNormalCamera()) ||
                   (SelectionManagerPin->IsUsingSegmentationCamera() && SelectionManagerPin->HasSegmentationCamera()) ||
                   (SelectionManagerPin->IsUsingBaseColorCamera() && SelectionManagerPin->HasBaseColorCamera()) ||
                   (SelectionManagerPin->IsUsingMaterialPropertiesCamera() && SelectionManagerPin->HasMaterialPropertiesCamera());
        })
    ]
    
    // Auto Capture button
    +SHorizontalBox::Slot()
    .MaxWidth(180)
    .Padding(FMargin(4, 0, 4, 0))
    .HAlign(HAlign_Fill)
    [
        SAssignNew(AutoCaptureButton, SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Primary")
        .ContentPadding(FMargin(5, 2))
        .Text_Lambda([this]() {
            return bAutoCaptureInProgress ? 
                FText::FromString("Stop Capture") :
                FText::FromString("Capture All Poses");
        })
        .HAlign(HAlign_Center)
        .OnClicked_Lambda([this]() {
            if (bAutoCaptureInProgress)
            {
                StopAutoCapture();
                AutoCaptureButton->SetButtonStyle(&FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Primary"));
            }
            else
            {
                StartAutoCapture();
                AutoCaptureButton->SetButtonStyle(&FAppStyle::Get().GetWidgetStyle<FButtonStyle>("FlatButton.Danger"));
            }
            return FReply::Handled();
        })
        .IsEnabled_Lambda([this]() {
            if (!SelectionManager.IsValid()) return false;
            auto SelectionManagerPin = SelectionManager.Pin();
            if (!SelectionManagerPin.IsValid() || !SelectionManagerPin->GetSelectedFlashPawn().IsValid()) return false;

            return SelectionManagerPin->HasAnyActiveCamera();
        })
    ];
}

// ============================================================================
// DATASET CONFIGURATION
// ============================================================================

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreateDatasetConfigSection()
{
    return SNew(SVerticalBox)

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            FVCCSimUIHelpers::CreateSectionHeader(TEXT("Dataset Configuration"))
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            FVCCSimUIHelpers::CreatePropertyRow(TEXT("Output Dir"),
                SNew(SHorizontalBox)
                +SHorizontalBox::Slot().FillWidth(1.f)
                [
                    SAssignNew(OutputDirTextBox, SEditableTextBox)
                    .Text(FText::FromString(OutputDirectory))
                    .OnTextCommitted_Lambda([this](const FText& Text, ETextCommit::Type)
                    {
                        OutputDirectory = Text.ToString();
                        SaveToConfigManager();
                    })
                ]
                +SHorizontalBox::Slot().AutoWidth().Padding(FMargin(4, 0, 0, 0))
                [
                    SNew(SButton)
                    .ButtonStyle(FAppStyle::Get(), "FlatButton")
                    .ContentPadding(FMargin(5, 2))
                    .Text(FText::FromString(TEXT("...")))
                    .OnClicked_Lambda([this]() { return OnBrowseOutputDirClicked(); })
                ]
            )
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            FVCCSimUIHelpers::CreatePropertyRow(TEXT("Scene Name"),
                SAssignNew(SceneNameTextBox, SEditableTextBox)
                .Text(FText::FromString(SceneName))
                .OnTextCommitted_Lambda([this](const FText& Text, ETextCommit::Type)
                {
                    SceneName = Text.ToString();
                    SaveToConfigManager();
                })
            )
        ];
}

// ============================================================================
// LIGHTING SCHEDULE
// ============================================================================

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreateLightingScheduleSection()
{
    TSharedRef<SVerticalBox> Entries = SNew(SVerticalBox);
    for (int32 i = 0; i < NumLightingConditions; ++i)
    {
        Entries->AddSlot().AutoHeight().Padding(FMargin(0, 1))
        [
            CreateLightingEntry(i)
        ];
    }

    return FVCCSimUIHelpers::CreateCollapsibleSection(TEXT("Lighting Schedule"),
        SNew(SVerticalBox)

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            SNew(SBorder)
            .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
            .BorderBackgroundColor(FColor(20, 40, 80, 255))
            .Padding(FMargin(6, 3))
            [
                Entries
            ]
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [ CreateSunPositionCalculatorWidget() ],
        bLightingScheduleExpanded);
}

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreateSunPositionCalculatorWidget()
{
    auto MakeInlineSpinBoxFloat = [this](
        const FString& LabelText,
        TSharedPtr<SNumericEntryBox<float>>& SpinBoxPtr,
        TOptional<float>& ValueOpt,
        float& Var,
        float MinVal, float MaxVal, float Delta) -> TSharedRef<SWidget>
    {
        return SNew(SHorizontalBox)
            +SHorizontalBox::Slot().MaxWidth(76).VAlign(VAlign_Center).Padding(FMargin(0, 0, 4, 0))
            [
                SNew(STextBlock)
                .Text(FText::FromString(LabelText))
                .ColorAndOpacity(FLinearColor(0.85f, 0.85f, 0.85f))
                .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
            ]
            +SHorizontalBox::Slot().FillWidth(1.f)
            [
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5, 5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(SpinBoxPtr, SNumericEntryBox<float>)
                    .Value_Lambda([&ValueOpt]() { return ValueOpt; })
                    .MinValue(MinVal).MaxValue(MaxVal).Delta(Delta).AllowSpin(false)
                    .OnValueChanged_Lambda([&ValueOpt, &Var, MinVal, MaxVal](float Val)
                    {
                        Val = FMath::Clamp(Val, MinVal, MaxVal);
                        Var = Val;
                        ValueOpt = Val;
                    })
                ]
            ];
    };

    auto MakeInlineSpinBoxInt = [this](
        const FString& LabelText,
        TSharedPtr<SNumericEntryBox<int32>>& SpinBoxPtr,
        TOptional<int32>& ValueOpt,
        int32& Var,
        int32 MinVal, int32 MaxVal, int32 Delta) -> TSharedRef<SWidget>
    {
        return SNew(SHorizontalBox)
            +SHorizontalBox::Slot().MaxWidth(40).VAlign(VAlign_Center).Padding(FMargin(0, 0, 2, 0))
            [
                SNew(STextBlock)
                .Text(FText::FromString(LabelText))
                .ColorAndOpacity(FLinearColor(0.85f, 0.85f, 0.85f))
                .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
            ]
            +SHorizontalBox::Slot().FillWidth(1.f)
            [
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5, 5, 5, 255))
                .Padding(4, 0)
                [
                    SAssignNew(SpinBoxPtr, SNumericEntryBox<int32>)
                    .Value_Lambda([&ValueOpt]() { return ValueOpt; })
                    .MinValue(MinVal).MaxValue(MaxVal).Delta(Delta).AllowSpin(false)
                    .OnValueChanged_Lambda([&ValueOpt, &Var, MinVal, MaxVal](int32 Val)
                    {
                        Val = FMath::Clamp(Val, MinVal, MaxVal);
                        Var = Val;
                        ValueOpt = Val;
                    })
                ]
            ];
    };

    return SNew(SBorder)
    .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
    .BorderBackgroundColor(FColor(35, 20, 55, 255))
    .Padding(FMargin(6, 4))
    [
        SNew(SVerticalBox)

        +SVerticalBox::Slot().AutoHeight()
        [
            SNew(STextBlock)
            .Text(FText::FromString(TEXT("Sun Position Calculator")))
            .ColorAndOpacity(FLinearColor(0.75f, 0.5f, 1.f))
            .Font(FCoreStyle::GetDefaultFontStyle("Bold", 9))
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            MakeInlineSpinBoxFloat(TEXT("Latitude"),
                SunCalcLatSpinBox, SunCalcLatValue, SunCalcLatitude, -90.f, 90.f, 0.1f)
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            MakeInlineSpinBoxFloat(TEXT("Longitude"),
                SunCalcLonSpinBox, SunCalcLonValue, SunCalcLongitude, -180.f, 180.f, 0.1f)
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            MakeInlineSpinBoxFloat(TEXT("TZ (UTC±)"),
                SunCalcTZSpinBox, SunCalcTZValue, SunCalcTimeZone, -12.f, 14.f, 0.5f)
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            SNew(SHorizontalBox)

            +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 4, 0))
            [
                MakeInlineSpinBoxInt(TEXT("Year"),
                    SunCalcYearSpinBox, SunCalcYearValue, SunCalcYear, 1900, 2100, 1)
            ]

            +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 4, 0))
            [
                MakeInlineSpinBoxInt(TEXT("Month"),
                    SunCalcMonthSpinBox, SunCalcMonthValue, SunCalcMonth, 1, 12, 1)
            ]

            +SHorizontalBox::Slot().FillWidth(1.f)
            [
                MakeInlineSpinBoxInt(TEXT("Day"),
                    SunCalcDaySpinBox, SunCalcDayValue, SunCalcDay, 1, 31, 1)
            ]
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            SNew(SHorizontalBox)

            +SHorizontalBox::Slot().FillWidth(1.f).Padding(FMargin(0, 0, 4, 0))
            [
                MakeInlineSpinBoxInt(TEXT("Hour"),
                    SunCalcHourSpinBox, SunCalcHourValue, SunCalcHour, 0, 23, 1)
            ]

            +SHorizontalBox::Slot().FillWidth(1.f)
            [
                MakeInlineSpinBoxInt(TEXT("Minute"),
                    SunCalcMinuteSpinBox, SunCalcMinuteValue, SunCalcMinute, 0, 59, 5)
            ]
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            SNew(SHorizontalBox)

            +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(0, 0, 4, 0))
            [
                SNew(STextBlock)
                .Text(FText::FromString(TEXT("Fill slot")))
                .ColorAndOpacity(FLinearColor(0.6f, 0.6f, 0.6f))
                .Font(FCoreStyle::GetDefaultFontStyle("Regular", 8))
            ]

            +SHorizontalBox::Slot().MaxWidth(48).Padding(FMargin(0, 0, 6, 0))
            [
                SNew(SBorder)
                .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
                .BorderBackgroundColor(FColor(5, 5, 5, 255))
                .Padding(2, 0)
                [
                    SAssignNew(SunCalcFillSlotSpinBox, SNumericEntryBox<int32>)
                    .Value_Lambda([this]() { return SunCalcFillSlotValue; })
                    .MinValue(1).MaxValue(NumLightingConditions).Delta(1).AllowSpin(false)
                    .ToolTipText(FText::FromString(TEXT("Target slot index (1-5)")))
                    .OnValueChanged_Lambda([this](int32 Val)
                    {
                        SunCalcFillSlot      = Val;
                        SunCalcFillSlotValue = Val;
                    })
                ]
            ]

            +SHorizontalBox::Slot().AutoWidth().Padding(FMargin(0, 0, 8, 0))
            [
                SNew(SButton)
                .ButtonStyle(FAppStyle::Get(), "FlatButton.Primary")
                .ContentPadding(FMargin(6, 2))
                .Text(FText::FromString(TEXT("Slot")))
                .ToolTipText(FText::FromString(TEXT("Write computed Elevation/Azimuth into the chosen slot")))
                .OnClicked_Lambda([this]() { return OnFillFromSunPositionClicked(); })
            ]

            +SHorizontalBox::Slot().AutoWidth()
            [
                SNew(SButton)
                .ButtonStyle(FAppStyle::Get(), "FlatButton.Primary")
                .ContentPadding(FMargin(8, 2))
                .Text(FText::FromString(TEXT("Calculate")))
                .ToolTipText(FText::FromString(TEXT("Compute sun position and apply to Directional Light")))
                .OnClicked_Lambda([this]() { return OnCalculateSunPositionClicked(); })
            ]
        ]
    ];
}

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreateLightingEntry(int32 Index)
{
    return SNew(SHorizontalBox)

    +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(0, 0, 4, 0))
    [
        SNew(STextBlock)
        .Text(FText::FromString(FString::Printf(TEXT("%d"), Index + 1)))
        .ColorAndOpacity(FLinearColor(0.6f, 0.6f, 0.6f))
        .MinDesiredWidth(18.f)
    ]

    +SHorizontalBox::Slot().MaxWidth(80).Padding(FMargin(0, 0, 2, 0))
    [
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(FColor(5, 5, 5, 255))
        .Padding(2, 0)
        [
            SAssignNew(LightingElevationSpinBox[Index], SNumericEntryBox<float>)
            .Value_Lambda([this, Index]() { return LightingElevationValue[Index]; })
            .MinValue(0.f).MaxValue(90.f).Delta(1.f).AllowSpin(false)
            .ToolTipText(FText::FromString(TEXT("Sun elevation (°)")))
            .OnValueChanged_Lambda([this, Index](float Val)
            {
                LightingElevation[Index] = Val;
                LightingElevationValue[Index] = Val;
                SaveToConfigManager();
            })
        ]
    ]

    +SHorizontalBox::Slot().MaxWidth(80).Padding(FMargin(0, 0, 4, 0))
    [
        SNew(SBorder)
        .BorderImage(FAppStyle::GetBrush("DetailsView.CategoryMiddle"))
        .BorderBackgroundColor(FColor(5, 5, 5, 255))
        .Padding(2, 0)
        [
            SAssignNew(LightingAzimuthSpinBox[Index], SNumericEntryBox<float>)
            .Value_Lambda([this, Index]() { return LightingAzimuthValue[Index]; })
            .MinValue(0.f).MaxValue(360.f).Delta(5.f).AllowSpin(false)
            .ToolTipText(FText::FromString(TEXT("Sun azimuth (°)")))
            .OnValueChanged_Lambda([this, Index](float Val)
            {
                LightingAzimuth[Index] = Val;
                LightingAzimuthValue[Index] = Val;
                SaveToConfigManager();
            })
        ]
    ]

    +SHorizontalBox::Slot().MaxWidth(80).HAlign(HAlign_Fill)
    [
        SNew(SButton)
        .ButtonStyle(FAppStyle::Get(), "FlatButton.Primary")
        .ContentPadding(FMargin(4, 2))
        .Text(FText::FromString(TEXT("Apply")))
        .ToolTipText(FText::FromString(TEXT("Apply this lighting condition to the scene")))
        .OnClicked_Lambda([this, Index]() { return OnApplyLightingClicked(Index); })
    ]

    +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(6, 0, 0, 0))
    [
        SNew(SCheckBox)
        .IsChecked_Lambda([this, Index]()
        {
            return bLightingSelected[Index] ? ECheckBoxState::Checked : ECheckBoxState::Unchecked;
        })
        .OnCheckStateChanged_Lambda([this, Index](ECheckBoxState State)
        {
            bLightingSelected[Index] = (State == ECheckBoxState::Checked);
            SaveToConfigManager();
        })
        .ToolTipText(FText::FromString(TEXT("Include this lighting condition in batch Capture Dataset")))
    ];
}

// ============================================================================
// DATASET CAPTURE
// ============================================================================

TSharedRef<SWidget> FVCCSimPanelPathImageCapture::CreateDatasetCaptureSection()
{
    return SNew(SVerticalBox)

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            FVCCSimUIHelpers::CreateSectionHeader(TEXT("Dataset Capture"))
        ]

        +SVerticalBox::Slot().AutoHeight().Padding(FMargin(0, 2))
        [
            SNew(SHorizontalBox)
            +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(0, 0, 8, 0))
            [
                SNew(SCheckBox)
                .ToolTipText(FText::FromString(TEXT(
                    "Capture RGB / BaseColor / MatProps / Normal images for every FlashPawn pose. "
                    "Off = skip image capture and only export the GT mesh.")))
                .IsEnabled_Lambda([this]() { return !bDatasetCaptureInProgress; })
                .IsChecked_Lambda([this]() { return bOutputImages ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
                .OnCheckStateChanged_Lambda([this](ECheckBoxState S) { bOutputImages = (S == ECheckBoxState::Checked); SaveToConfigManager(); })
                [
                    SNew(STextBlock).Text(FText::FromString(TEXT("Photos")))
                ]
            ]
            +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(0, 0, 8, 0))
            [
                SNew(SCheckBox)
                .ToolTipText(FText::FromString(TEXT(
                    "Export the GT mesh (gt_materials) for the enabled target actors after capturing. "
                    "Reused from a previous capture when unchanged. Off = skip mesh export.")))
                .IsEnabled_Lambda([this]() { return !bDatasetCaptureInProgress; })
                .IsChecked_Lambda([this]() { return bOutputMesh ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
                .OnCheckStateChanged_Lambda([this](ECheckBoxState S) { bOutputMesh = (S == ECheckBoxState::Checked); SaveToConfigManager(); })
                [
                    SNew(STextBlock).Text(FText::FromString(TEXT("Mesh")))
                ]
            ]
            +SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(FMargin(0, 0, 8, 0))
            [
                SNew(SCheckBox)
                .ToolTipText(FText::FromString(TEXT(
                    "Reuse lighting-independent GT channels (BaseColor / MatProps / Normal / gt_materials) "
                    "from a matching earlier capture, so later lighting windows only re-shoot RGB. "
                    "Off = every lighting window captures the full channel set (complete, self-contained dataset).")))
                .IsEnabled_Lambda([this]() { return !bDatasetCaptureInProgress; })
                .IsChecked_Lambda([this]() { return bUseCaptureReuse ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
                .OnCheckStateChanged_Lambda([this](ECheckBoxState S) { bUseCaptureReuse = (S == ECheckBoxState::Checked); SaveToConfigManager(); })
                [
                    SNew(STextBlock).Text(FText::FromString(TEXT("Reuse")))
                ]
            ]
            +SHorizontalBox::Slot().FillWidth(1.f)
            [
                SNew(SButton)
                .ButtonStyle(FAppStyle::Get(), "FlatButton.Success")
                .ContentPadding(FMargin(5, 2))
                .Text_Lambda([this]()
                {
                    return bDatasetCaptureInProgress
                        ? FText::FromString(TEXT("Stop Capture"))
                        : FText::FromString(TEXT("Capture"));
                })
                .ToolTipText(FText::FromString(
                    TEXT("Writes poses.txt + intrinsics.json + lighting.json and captures "
                         "whatever camera channels are checked in Object Selection for every pose on "
                         "the FlashPawn path into a new <Output>/<Scene>/captures/capture_<timestamp> "
                         "directory. Apply the desired lighting first; click again after changing "
                         "lighting to add another capture window. Photos / Mesh checkboxes pick "
                         "which outputs are written. Click again while running to stop "
                         "(the partial output is kept — use Resume to continue it).")))
                .IsEnabled_Lambda([this]()
                {
                    return bDatasetCaptureInProgress
                        || (!OutputDirectory.IsEmpty() && (bOutputImages || bOutputMesh));
                })
                .OnClicked_Lambda([this]() { return OnCaptureDatasetClicked(); })
            ]
            +SHorizontalBox::Slot().AutoWidth().Padding(FMargin(4, 0, 0, 0))
            [
                SNew(SButton)
                .ButtonStyle(FAppStyle::Get(), "FlatButton.Primary")
                .ContentPadding(FMargin(5, 2))
                .Text(FText::FromString(TEXT("Resume")))
                .ToolTipText(FText::FromString(
                    TEXT("Continue the last dataset capture that was stopped or interrupted by a crash "
                         "(reads <Scene>/captures/capture_session.json). Skips poses and lighting "
                         "windows already on disk, re-shoots the last pose as a safety measure, and "
                         "finishes the rest. Requires the same FlashPawn path and scene as the "
                         "interrupted run. Enabled only when a resumable capture exists.")))
                .IsEnabled_Lambda([this]() { return HasResumableCapture(); })
                .OnClicked_Lambda([this]() { return OnResumeCaptureClicked(); })
            ]
        ];
}
