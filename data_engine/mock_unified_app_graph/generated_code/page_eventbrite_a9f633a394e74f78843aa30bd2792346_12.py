# page_id: page_eventbrite_a9f633a394e74f78843aa30bd2792346_12
# screenshot: 2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14.png
# step_index: 12/18
# task: Open Eventbrite. Set the city to "Los Angeles". Look for Photography workshops happening next week. What is the price of the tickets for first non-promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural layout for the date-picker UI
# Uses provided variables: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

# ensure full white background (matches screenshot dominant color)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Colors
status_bar_color = (200, 200, 200)        # light grey status bar
header_div_color = (236, 234, 242)        # very light divider
calendar_card_bg = (250, 250, 254)        # subtle off-white / very light bluish
calendar_card_outline = (240, 238, 245)   # faint outline
end_section_bg = (255, 255, 255)          # keep end section white but slightly separated
shadow_strip = (245, 244, 247)            # subtle shadow above bottom button
separator_color = (220, 218, 227)         # another subtle separator

# 1) Status bar area at top (~72px height)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=status_bar_color)

# 2) Header bar area (below status bar) with subtle divider
header_top = status_h
header_bottom = 160
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))
draw.line([(48, header_bottom), (1392, header_bottom)], fill=header_div_color, width=1)

# 3) Main calendar card background (rounded rectangle)
cal_left = 48
cal_top = 200
cal_right = 1392
cal_bottom = 1220
cal_radius = 28
# outer faint outline (slightly larger to simulate subtle border)
draw.rounded_rectangle([(cal_left - 2, cal_top - 2), (cal_right + 2, cal_bottom + 2)],
                       radius=cal_radius + 2, fill=calendar_card_outline)
# main card
draw.rounded_rectangle([(cal_left, cal_top), (cal_right, cal_bottom)],
                       radius=cal_radius, fill=calendar_card_bg)

# subtle horizontal guide line under the month row (non-intrusive)
draw.line([(cal_left + 24, cal_top + 110), (cal_right - 24, cal_top + 110)],
          fill=header_div_color, width=1)

# faint vertical separators for calendar columns (light, wide spacing)
col_width = (cal_right - cal_left - 48) / 7.0
for i in range(1, 7):
    x = cal_left + 24 + int(i * col_width)
    draw.line([(x, cal_top + 140), (x, cal_bottom - 80)], fill=(245, 244, 247), width=1)

# 4) End Date section card (below calendar) as a soft panel/background
end_left = 48
end_top = 1440
end_right = 1392
end_bottom = 1720
end_radius = 20
draw.rounded_rectangle([(end_left, end_top), (end_right, end_bottom)],
                       radius=end_radius, fill=end_section_bg, outline=calendar_card_outline, width=1)

# subtle divider separating the two sections (calendar and end section)
draw.line([(end_left + 8, end_top), (end_right - 8, end_top)], fill=separator_color, width=1)

# 5) Large empty content area is left white (no draws) to avoid duplicating text/icons

# 6) Thin shadow strip above bottom action area (to visually separate)
shadow_top = 2600
shadow_bottom = 2768  # keep above the detected apply-button region (detected button begins at y=2768)
draw.rectangle([(0, shadow_top), (1440, shadow_bottom)], fill=shadow_strip)

# subtle horizontal separator line just above the button area
sep_y = 2720
draw.line([(48, sep_y), (1392, sep_y)], fill=(225, 223, 233), width=1)

# 7) Top subtle border of entire content to frame the layout
draw.line([(0, header_bottom), (1440, header_bottom)], fill=(245, 244, 247), width=1)

# Note: All interactive elements (icons, text, buttons) will be pasted on top separately.
# This code only provides background fills, cards, separators and structural shapes.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/00_icon_24.png
try:
    _c0 = get_crop(0, 132, 120)
    canvas.paste(_c0, (456, 1081), _c0)
except Exception:
    pass
layout["24"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/01_icon_Apply_date_range.png
try:
    _c1 = get_crop(1, 1344, 144)
    canvas.paste(_c1, (48, 2768), _c1)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/02_icon_28.png
try:
    _c2 = get_crop(2, 132, 120)
    canvas.paste(_c2, (60, 1201), _c2)
except Exception:
    pass
layout["28"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/03_icon_29.png
try:
    _c3 = get_crop(3, 132, 120)
    canvas.paste(_c3, (192, 1201), _c3)
except Exception:
    pass
layout["29"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/04_icon_23.png
try:
    _c4 = get_crop(4, 132, 120)
    canvas.paste(_c4, (324, 1081), _c4)
except Exception:
    pass
layout["23"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/05_icon_30.png
try:
    _c5 = get_crop(5, 132, 120)
    canvas.paste(_c5, (324, 1201), _c5)
except Exception:
    pass
layout["30"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/06_icon_25.png
try:
    _c6 = get_crop(6, 132, 120)
    canvas.paste(_c6, (588, 1081), _c6)
except Exception:
    pass
layout["25"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/07_icon_27.png
try:
    _c7 = get_crop(7, 132, 120)
    canvas.paste(_c7, (852, 1081), _c7)
except Exception:
    pass
layout["27"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 52, 71)
    canvas.paste(_c8, (1153, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1153, 0, 1205, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/09_icon_26.png
try:
    _c9 = get_crop(9, 132, 120)
    canvas.paste(_c9, (720, 1081), _c9)
except Exception:
    pass
layout["26"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/10_icon_4.51.png
try:
    _c10 = get_crop(10, 61, 65)
    canvas.paste(_c10, (180, 0), _c10)
except Exception:
    pass
layout["4.51"] = [180, 0, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/11_icon_4.51.png
try:
    _c11 = get_crop(11, 64, 67)
    canvas.paste(_c11, (111, 1), _c11)
except Exception:
    pass
layout["4.51"] = [111, 1, 175, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 64, 63)
    canvas.paste(_c12, (309, 3), _c12)
except Exception:
    pass
layout["icon_12"] = [309, 3, 373, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 100, 70)
    canvas.paste(_c13, (1210, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1210, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/14_icon_21.png
try:
    _c14 = get_crop(14, 132, 120)
    canvas.paste(_c14, (60, 1081), _c14)
except Exception:
    pass
layout["21"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/15_icon_22.png
try:
    _c15 = get_crop(15, 132, 120)
    canvas.paste(_c15, (192, 1081), _c15)
except Exception:
    pass
layout["22"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 53, 64)
    canvas.paste(_c16, (247, 2), _c16)
except Exception:
    pass
layout["icon_16"] = [247, 2, 300, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 54, 70)
    canvas.paste(_c17, (1318, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1318, 0, 1372, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/18_icon_4.51.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (12, 72), _c18)
except Exception:
    pass
layout["4.51"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/19_icon_18.png
try:
    _c19 = get_crop(19, 132, 120)
    canvas.paste(_c19, (588, 961), _c19)
except Exception:
    pass
layout["18"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/20_icon_11.png
try:
    _c20 = get_crop(20, 132, 120)
    canvas.paste(_c20, (588, 721), _c20)
except Exception:
    pass
layout["11"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/21_icon_19.png
try:
    _c21 = get_crop(21, 132, 120)
    canvas.paste(_c21, (720, 961), _c21)
except Exception:
    pass
layout["19"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 49, 67)
    canvas.paste(_c22, (382, 1), _c22)
except Exception:
    pass
layout["icon_22"] = [382, 1, 431, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/23_icon_4.51.png
try:
    _c23 = get_crop(23, 94, 65)
    canvas.paste(_c23, (14, 1), _c23)
except Exception:
    pass
layout["4.51"] = [14, 1, 108, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/24_icon_April_2024.png
try:
    _c24 = get_crop(24, 126, 110)
    canvas.paste(_c24, (593, 611), _c24)
except Exception:
    pass
layout["April_2024"] = [593, 611, 719, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/25_icon_Next_month.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (846, 457), _c25)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/26_icon_12.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (720, 721), _c26)
except Exception:
    pass
layout["12"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/27_icon_12.png
try:
    _c27 = get_crop(27, 104, 106)
    canvas.paste(_c27, (733, 615), _c27)
except Exception:
    pass
layout["12"] = [733, 615, 837, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/28_icon_Choose_a_date.png
try:
    _c28 = get_crop(28, 638, 144)
    canvas.paste(_c28, (48, 1490), _c28)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 103, 99)
    canvas.paste(_c29, (72, 618), _c29)
except Exception:
    pass
layout["icon_29"] = [72, 618, 175, 717]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/30_icon_What_date.png
try:
    _c30 = get_crop(30, 322, 71)
    canvas.paste(_c30, (558, 113), _c30)
except Exception:
    pass
layout["What_date?"] = [558, 113, 880, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/31_icon_16.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (324, 961), _c31)
except Exception:
    pass
layout["16"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/32_icon_10.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (324, 721), _c32)
except Exception:
    pass
layout["10"] = [324, 721, 456, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/33_text_Start_Date.png
try:
    _c33 = get_crop(33, 589, 114)
    canvas.paste(_c33, (48, 313), _c33)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 637, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/34_text_April_2024.png
try:
    _c34 = get_crop(34, 203, 54)
    canvas.paste(_c34, (420, 504), _c34)
except Exception:
    pass
layout["April_2024"] = [420, 504, 623, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/35_text_10.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (456, 841), _c35)
except Exception:
    pass
layout["10"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/36_text_11.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (588, 841), _c36)
except Exception:
    pass
layout["11"] = [588, 841, 720, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/37_text_12.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (720, 841), _c37)
except Exception:
    pass
layout["12"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/38_text_13.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (852, 841), _c38)
except Exception:
    pass
layout["13"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/39_text_14.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (60, 961), _c39)
except Exception:
    pass
layout["14"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/40_text_15.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (192, 961), _c40)
except Exception:
    pass
layout["15"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/41_text_17.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (456, 961), _c41)
except Exception:
    pass
layout["17"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/42_text_20.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (852, 961), _c42)
except Exception:
    pass
layout["20"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/43_clickable_1.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (192, 721), _c43)
except Exception:
    pass
layout["1"] = [192, 721, 324, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/44_clickable_3.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (456, 721), _c44)
except Exception:
    pass
layout["3"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/45_clickable_6.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (852, 721), _c45)
except Exception:
    pass
layout["6"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/46_clickable_7.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (60, 841), _c46)
except Exception:
    pass
layout["7"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/47_clickable_8.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (192, 841), _c47)
except Exception:
    pass
layout["8"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_12_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-14/48_clickable_9.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (324, 841), _c48)
except Exception:
    pass
layout["9"] = [324, 841, 456, 961]
