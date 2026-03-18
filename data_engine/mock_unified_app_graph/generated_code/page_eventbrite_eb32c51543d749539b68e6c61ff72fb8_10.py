# page_id: page_eventbrite_eb32c51543d749539b68e6c61ff72fb8_10
# screenshot: 2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12.png
# step_index: 10/19
# task: Open Eventbrite. Set the city to San Francisco. Filter for events occurring between May 1st and May 15th under the category 'Music'. Select the first event and check the pricing options available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and UI structure for the mobile date-picker screen.
# Available variables:
# - canvas: PIL Image (1440x2960 RGB)
# - draw: PIL ImageDraw object
# - font_sm, font_md, font_lg, font_xl

# Colors
bg = (255, 255, 255)               # page background (white)
status_bar = (190, 190, 190)       # top status bar (light gray)
header_div = (235, 233, 240)       # subtle divider under header
card_fill = (250, 250, 253)        # very subtle off-white card background
card_border = (222, 218, 230)      # card border / outline
muted_separator = (240, 239, 243)  # faint separators

W, H = canvas.size

# Fill canvas (ensure any prior content is cleared)
draw.rectangle((0, 0, W, H), fill=bg)

# Status bar area (top system bar)
status_h = 72  # approximate height for status area where system icons are
draw.rectangle((0, 0, W, status_h), fill=status_bar)

# Header / toolbar area (below status bar). Keep background white to match design,
# but add a bottom divider line to separate from content.
header_top = status_h
header_h = 120
draw.rectangle((0, header_top, W, header_top + header_h), fill=bg)
# divider line across entire width
draw.line((32, header_top + header_h - 2, W - 32, header_top + header_h - 2), fill=header_div, width=2)

# Main calendar card / section background
# This is the rounded card behind the month title and calendar grid.
card_x0 = 48
card_x1 = W - 48
card_y0 = header_top + header_h + 20   # leave spacing under header
card_y1 = 1360                          # extend down past calendar grid
card_radius = 28

# subtle shadow block behind card (very faint)
shadow_offset = 6
shadow_box = (card_x0 + shadow_offset, card_y0 + shadow_offset, card_x1 + shadow_offset, card_y1 + shadow_offset)
try:
    draw.rounded_rectangle(shadow_box, radius=card_radius, fill=(240, 240, 245))
except Exception:
    draw.rectangle(shadow_box, fill=(240, 240, 245))

# card fill and border
try:
    draw.rounded_rectangle((card_x0, card_y0, card_x1, card_y1), radius=card_radius, fill=card_fill, outline=card_border, width=2)
except Exception:
    draw.rectangle((card_x0, card_y0, card_x1, card_y1), fill=card_fill, outline=card_border)

# Sub-header area inside the card for month navigation (centered visually)
month_bar_h = 96
month_bar_y0 = card_y0 + 28
month_bar_y1 = month_bar_y0 + month_bar_h
month_bar_x0 = card_x0 + 40
month_bar_x1 = card_x1 - 40
# Slightly different tint to separate month label area from calendar grid
try:
    draw.rounded_rectangle((month_bar_x0, month_bar_y0, month_bar_x1, month_bar_y1), radius=20, fill=card_fill, outline=None)
except Exception:
    draw.rectangle((month_bar_x0, month_bar_y0, month_bar_x1, month_bar_y1), fill=card_fill)

# Divider under the month bar to visually separate the header of the card from the grid.
divider_y = month_bar_y1 + 18
draw.line((card_x0 + 20, divider_y, card_x1 - 20, divider_y), fill=muted_separator, width=1)

# Light grid background band where the weekdays row sits (subtle)
week_row_y0 = divider_y + 18
week_row_y1 = week_row_y0 + 54
draw.rectangle((card_x0 + 20, week_row_y0, card_x1 - 20, week_row_y1), fill=(253, 253, 254))

# Faint separators between the calendar card and the end-date section below it.
end_section_y0 = card_y1 + 36
draw.line((card_x0 + 12, end_section_y0, card_x1 - 12, end_section_y0), fill=muted_separator, width=1)

# "End Date" section card/background (keeps it visually grouped but do not draw any text)
end_card_x0 = card_x0
end_card_x1 = card_x1
end_card_y0 = end_section_y0 + 28
end_card_y1 = end_card_y0 + 220
try:
    draw.rounded_rectangle((end_card_x0, end_card_y0, end_card_x1, end_card_y1), radius=20, fill=bg, outline=None)
except Exception:
    draw.rectangle((end_card_x0, end_card_y0, end_card_x1, end_card_y1), fill=bg)

# Add a subtle left padding guide bar inside the end-date area to imply structure (non-intrusive)
guide_x = end_card_x0 + 24
draw.line((guide_x, end_card_y0 + 18, guide_x, end_card_y1 - 18), fill=muted_separator, width=1)

# Bottom safe area: leave the button area clear (do not draw the actual button),
# but add a faint top border above the button area to separate content from the control.
button_area_top = 2768
draw.line((32, button_area_top - 12, W - 32, button_area_top - 12), fill=card_border, width=2)

# Small left & right page margins as subtle vertical guides (non intrusive)
margin_x = 32
draw.line((margin_x, header_top + 8, margin_x, H - 220), fill=(250,250,252), width=1)
draw.line((W - margin_x, header_top + 8, W - margin_x, H - 220), fill=(250,250,252), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/00_icon_23.png
try:
    _c0 = get_crop(0, 132, 120)
    canvas.paste(_c0, (324, 1081), _c0)
except Exception:
    pass
layout["23"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/01_icon_Apply_date_range.png
try:
    _c1 = get_crop(1, 1344, 144)
    canvas.paste(_c1, (48, 2768), _c1)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/02_icon_28.png
try:
    _c2 = get_crop(2, 132, 120)
    canvas.paste(_c2, (60, 1201), _c2)
except Exception:
    pass
layout["28"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/03_icon_24.png
try:
    _c3 = get_crop(3, 132, 120)
    canvas.paste(_c3, (456, 1081), _c3)
except Exception:
    pass
layout["24"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/04_icon_29.png
try:
    _c4 = get_crop(4, 132, 120)
    canvas.paste(_c4, (192, 1201), _c4)
except Exception:
    pass
layout["29"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 52, 71)
    canvas.paste(_c5, (1153, 0), _c5)
except Exception:
    pass
layout["icon_5"] = [1153, 0, 1205, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/06_icon_22.png
try:
    _c6 = get_crop(6, 132, 120)
    canvas.paste(_c6, (192, 1081), _c6)
except Exception:
    pass
layout["22"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/07_icon_30.png
try:
    _c7 = get_crop(7, 132, 120)
    canvas.paste(_c7, (324, 1201), _c7)
except Exception:
    pass
layout["30"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/08_icon_7.47.png
try:
    _c8 = get_crop(8, 61, 65)
    canvas.paste(_c8, (180, 0), _c8)
except Exception:
    pass
layout["7.47"] = [180, 0, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/09_icon_25.png
try:
    _c9 = get_crop(9, 132, 120)
    canvas.paste(_c9, (588, 1081), _c9)
except Exception:
    pass
layout["25"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/10_icon_7.47.png
try:
    _c10 = get_crop(10, 63, 67)
    canvas.paste(_c10, (112, 1), _c10)
except Exception:
    pass
layout["7.47"] = [112, 1, 175, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/11_icon_26.png
try:
    _c11 = get_crop(11, 132, 120)
    canvas.paste(_c11, (720, 1081), _c11)
except Exception:
    pass
layout["26"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 100, 70)
    canvas.paste(_c12, (1210, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1210, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 64, 63)
    canvas.paste(_c13, (309, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [309, 3, 373, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 53, 64)
    canvas.paste(_c14, (247, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [247, 2, 300, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 54, 70)
    canvas.paste(_c15, (1318, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1318, 0, 1372, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/16_icon_27.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (852, 1081), _c16)
except Exception:
    pass
layout["27"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/17_icon_21.png
try:
    _c17 = get_crop(17, 132, 120)
    canvas.paste(_c17, (60, 1081), _c17)
except Exception:
    pass
layout["21"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/18_icon_7.47.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["7.47"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/19_icon_11.png
try:
    _c19 = get_crop(19, 132, 120)
    canvas.paste(_c19, (588, 721), _c19)
except Exception:
    pass
layout["11"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 49, 67)
    canvas.paste(_c20, (382, 1), _c20)
except Exception:
    pass
layout["icon_20"] = [382, 1, 431, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/21_icon_April_2024.png
try:
    _c21 = get_crop(21, 126, 110)
    canvas.paste(_c21, (593, 611), _c21)
except Exception:
    pass
layout["April_2024"] = [593, 611, 719, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/22_icon_7.47.png
try:
    _c22 = get_crop(22, 93, 63)
    canvas.paste(_c22, (16, 2), _c22)
except Exception:
    pass
layout["7.47"] = [16, 2, 109, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/23_icon_Next_month.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (846, 457), _c23)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/24_icon_18.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (588, 961), _c24)
except Exception:
    pass
layout["18"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/25_icon_Choose_a_date.png
try:
    _c25 = get_crop(25, 638, 144)
    canvas.paste(_c25, (48, 1490), _c25)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/26_icon_12.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (720, 721), _c26)
except Exception:
    pass
layout["12"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/27_icon_12.png
try:
    _c27 = get_crop(27, 103, 107)
    canvas.paste(_c27, (734, 614), _c27)
except Exception:
    pass
layout["12"] = [734, 614, 837, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/28_icon_19.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (720, 961), _c28)
except Exception:
    pass
layout["19"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/29_text_What_date.png
try:
    _c29 = get_crop(29, 318, 63)
    canvas.paste(_c29, (563, 117), _c29)
except Exception:
    pass
layout["What_date?"] = [563, 117, 881, 180]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/30_text_Start_Date.png
try:
    _c30 = get_crop(30, 580, 114)
    canvas.paste(_c30, (48, 313), _c30)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 628, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/31_text_April_2024.png
try:
    _c31 = get_crop(31, 203, 54)
    canvas.paste(_c31, (420, 504), _c31)
except Exception:
    pass
layout["April_2024"] = [420, 504, 623, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/32_text_10.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (456, 841), _c32)
except Exception:
    pass
layout["10"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/33_text_11.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (588, 841), _c33)
except Exception:
    pass
layout["11"] = [588, 841, 720, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/34_text_12.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (720, 841), _c34)
except Exception:
    pass
layout["12"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/35_text_13.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (852, 841), _c35)
except Exception:
    pass
layout["13"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/36_text_14.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (60, 961), _c36)
except Exception:
    pass
layout["14"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/37_text_15.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (192, 961), _c37)
except Exception:
    pass
layout["15"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/38_text_16.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (324, 961), _c38)
except Exception:
    pass
layout["16"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/39_text_17.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (456, 961), _c39)
except Exception:
    pass
layout["17"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/40_text_20.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (852, 961), _c40)
except Exception:
    pass
layout["20"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/41_clickable_1.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (192, 721), _c41)
except Exception:
    pass
layout["1"] = [192, 721, 324, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/42_clickable_2.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (324, 721), _c42)
except Exception:
    pass
layout["2"] = [324, 721, 456, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/43_clickable_3.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (456, 721), _c43)
except Exception:
    pass
layout["3"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/44_clickable_6.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (852, 721), _c44)
except Exception:
    pass
layout["6"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/45_clickable_7.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (60, 841), _c45)
except Exception:
    pass
layout["7"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/46_clickable_8.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (192, 841), _c46)
except Exception:
    pass
layout["8"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_10_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-12/47_clickable_9.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (324, 841), _c47)
except Exception:
    pass
layout["9"] = [324, 841, 456, 961]
