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

#include "Editor/IVCCSimModule.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/Layout/SSeparator.h"
#include "Widgets/Text/STextBlock.h"
#include "Styling/AppStyle.h"

TSharedRef<SWidget> SVCCSimModuleWidget::CreatePropertyRow(const FString& Label, TSharedRef<SWidget> Content)
{
    return SNew(SHorizontalBox)
        + SHorizontalBox::Slot()
        .AutoWidth()
        .VAlign(VAlign_Center)
        .Padding(0, 2, 8, 2)
        [
            SNew(STextBlock)
            .Text(FText::FromString(Label))
            .Font(FAppStyle::GetFontStyle("PropertyWindow.NormalFont"))
        ]
        + SHorizontalBox::Slot()
        .FillWidth(1.0f)
        .VAlign(VAlign_Center)
        .Padding(0, 2)
        [
            Content
        ];
}

TSharedRef<SWidget> SVCCSimModuleWidget::CreateSectionHeader(const FString& Title)
{
    return SNew(STextBlock)
        .Text(FText::FromString(Title))
        .Font(FAppStyle::GetFontStyle("DetailsView.CategoryFontStyle"))
        .Margin(FMargin(0, 4, 0, 2));
}

TSharedRef<SWidget> SVCCSimModuleWidget::CreateSeparator()
{
    return SNew(SSeparator)
        .Orientation(Orient_Horizontal)
        .Thickness(1.0f);
}