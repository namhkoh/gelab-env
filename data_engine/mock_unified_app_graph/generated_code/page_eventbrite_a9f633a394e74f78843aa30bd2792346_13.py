# page_id: page_eventbrite_a9f633a394e74f78843aa30bd2792346_13
# screenshot: 2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15.png
# step_index: 13/18
# task: Open Eventbrite. Set the city to "Los Angeles". Look for Photography workshops happening next week. What is the price of the tickets for first non-promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and UI structural elements for the calendar screen
# Available variables: canvas (PIL Image 1440x2960 RGB), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
white = (255, 255, 255)
status_gray = (230, 230, 230)        # status bar background
header_div = (230, 225, 240)         # subtle purple divider
card_bg = (250, 249, 255)            # very light lavender card background
card_outline = (220, 216, 230)       # card outline
section_div = (235, 232, 240)        # section separators
muted_line = (235, 235, 235)         # faint lines
accent_shadow = (240, 238, 245)      # light shadow tone

# Clear background (canvas is already white, but explicitly fill)
draw.rectangle([0, 0, W, H], fill=white)

# Status bar (top area)
status_h = 72
draw.rectangle([0, 0, W, status_h], fill=status_gray)

# Header / toolbar area (below status bar)
header_top = status_h
header_h = 88
draw.rectangle([0, header_top, W, header_top + header_h], fill=white)
# subtle bottom divider under header
draw.line([(32, header_top + header_h), (W - 32, header_top + header_h)], fill=header_div, width=2)

# Big calendar card background (rounded rect)
card_left = 48
card_right = W - 48
card_top = header_top + header_h + 24   # ~184
card_bottom = 1360
card_radius = 28
# gentle shadow behind card (very subtle)
shadow_offset = 8
draw.rounded_rectangle(
    [card_left + shadow_offset, card_top + shadow_offset, card_right + shadow_offset, card_bottom + shadow_offset],
    radius=card_radius, fill=accent_shadow
)
# actual card
draw.rounded_rectangle(
    [card_left, card_top, card_right, card_bottom],
    radius=card_radius, fill=card_bg, outline=card_outline, width=2
)

# Month header strip (visual grouping inside card) - leave text area blank
month_strip_h = 96
ms_top = card_top + 160
ms_left = card_left + 80
ms_right = card_right - 80
draw.rectangle([ms_left, ms_top, ms_right, ms_top + month_strip_h], fill=card_bg)
# small divider under month header
draw.line([(ms_left, ms_top + month_strip_h + 10), (ms_right, ms_top + month_strip_h + 10)], fill=muted_line, width=1)

# Subtle horizontal separators to indicate sections (between calendar and end-date area)
sep_y = card_bottom + 40
draw.line([(48, sep_y), (W - 48, sep_y)], fill=section_div, width=2)

# End Date card header background (visual grouping)
end_card_top = sep_y + 28
end_card_left = 48
end_card_right = W - 48
end_card_bottom = end_card_top + 220
draw.rounded_rectangle([end_card_left, end_card_top, end_card_right, end_card_bottom],
                       radius=20, fill=white, outline=card_outline, width=1)

# Footer separation above the bottom action area (where the button will be placed)
footer_sep_y = 2700
draw.line([(32, footer_sep_y), (W - 32, footer_sep_y)], fill=muted_line, width=2)
# add faint shadow band above the action area
draw.rectangle([32, footer_sep_y - 8, W - 32, footer_sep_y], fill=(248, 247, 250))

# Left and right page margins guides (visual only, very faint)
draw.line([(48, header_top + 8), (48, H - 200)], fill=(250, 250, 250), width=1)
draw.line([(W - 48, header_top + 8), (W - 48, H - 200)], fill=(250, 250, 250), width=1)

# Decorative faint week-day header markers (no text)
week_top = ms_top + month_strip_h + 36
week_height = 40
num_cols = 7
col_w = (ms_right - ms_left) / num_cols
for i in range(num_cols + 1):
    x = ms_left + i * col_w
    # very faint vertical markers (help structure calendar without drawing day numbers)
    draw.line([(x, week_top), (x, week_top + 420)], fill=(248, 247, 250), width=1)

# Add subtle rounded mask for large empty area to give sense of structure (no content)
large_area_top = end_card_bottom + 24
draw.rounded_rectangle([60, large_area_top, W - 60, footer_sep_y - 36],
                       radius=14, fill=white, outline=(245,244,247), width=1)

# Done structural drawing. UI elements (icons, text, buttons) will be pasted on top externally.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/00_icon_28.png
try:
    _c0 = get_crop(0, 132, 120)
    canvas.paste(_c0, (60, 1201), _c0)
except Exception:
    pass
layout["28"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/01_icon_Apply_date_range.png
try:
    _c1 = get_crop(1, 1344, 144)
    canvas.paste(_c1, (48, 2768), _c1)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/02_icon_29.png
try:
    _c2 = get_crop(2, 132, 120)
    canvas.paste(_c2, (192, 1201), _c2)
except Exception:
    pass
layout["29"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/03_icon_24.png
try:
    _c3 = get_crop(3, 132, 120)
    canvas.paste(_c3, (456, 1081), _c3)
except Exception:
    pass
layout["24"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 52, 71)
    canvas.paste(_c4, (1153, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1153, 0, 1205, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/05_icon_30.png
try:
    _c5 = get_crop(5, 132, 120)
    canvas.paste(_c5, (324, 1201), _c5)
except Exception:
    pass
layout["30"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/06_icon_23.png
try:
    _c6 = get_crop(6, 132, 120)
    canvas.paste(_c6, (324, 1081), _c6)
except Exception:
    pass
layout["23"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/07_icon_4.51.png
try:
    _c7 = get_crop(7, 61, 65)
    canvas.paste(_c7, (180, 0), _c7)
except Exception:
    pass
layout["4.51"] = [180, 0, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/08_icon_4.51.png
try:
    _c8 = get_crop(8, 64, 68)
    canvas.paste(_c8, (111, 0), _c8)
except Exception:
    pass
layout["4.51"] = [111, 0, 175, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/09_icon_25.png
try:
    _c9 = get_crop(9, 132, 120)
    canvas.paste(_c9, (588, 1081), _c9)
except Exception:
    pass
layout["25"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 65, 63)
    canvas.paste(_c10, (308, 3), _c10)
except Exception:
    pass
layout["icon_10"] = [308, 3, 373, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 100, 70)
    canvas.paste(_c11, (1210, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1210, 0, 1310, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/12_icon_22.png
try:
    _c12 = get_crop(12, 132, 120)
    canvas.paste(_c12, (192, 1081), _c12)
except Exception:
    pass
layout["22"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 53, 64)
    canvas.paste(_c13, (247, 2), _c13)
except Exception:
    pass
layout["icon_13"] = [247, 2, 300, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 54, 69)
    canvas.paste(_c14, (1318, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1318, 0, 1372, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/15_icon_4.51.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (12, 72), _c15)
except Exception:
    pass
layout["4.51"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/16_icon_26.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (720, 1081), _c16)
except Exception:
    pass
layout["26"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/17_icon_27.png
try:
    _c17 = get_crop(17, 132, 120)
    canvas.paste(_c17, (852, 1081), _c17)
except Exception:
    pass
layout["27"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 49, 67)
    canvas.paste(_c18, (382, 1), _c18)
except Exception:
    pass
layout["icon_18"] = [382, 1, 431, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/19_icon_11.png
try:
    _c19 = get_crop(19, 132, 120)
    canvas.paste(_c19, (588, 721), _c19)
except Exception:
    pass
layout["11"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/20_icon_4.51.png
try:
    _c20 = get_crop(20, 94, 65)
    canvas.paste(_c20, (14, 1), _c20)
except Exception:
    pass
layout["4.51"] = [14, 1, 108, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/21_icon_Choose_a_date.png
try:
    _c21 = get_crop(21, 638, 144)
    canvas.paste(_c21, (48, 1490), _c21)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/22_icon_Next_month.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (846, 457), _c22)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/23_icon_What_date.png
try:
    _c23 = get_crop(23, 322, 71)
    canvas.paste(_c23, (558, 113), _c23)
except Exception:
    pass
layout["What_date?"] = [558, 113, 880, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/24_icon_April_2024.png
try:
    _c24 = get_crop(24, 121, 109)
    canvas.paste(_c24, (596, 611), _c24)
except Exception:
    pass
layout["April_2024"] = [596, 611, 717, 720]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/25_text_Start_Date.png
try:
    _c25 = get_crop(25, 583, 114)
    canvas.paste(_c25, (48, 313), _c25)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 631, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/26_text_April_2024.png
try:
    _c26 = get_crop(26, 203, 54)
    canvas.paste(_c26, (420, 504), _c26)
except Exception:
    pass
layout["April_2024"] = [420, 504, 623, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/27_text_10.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (456, 841), _c27)
except Exception:
    pass
layout["10"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/28_text_11.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (588, 841), _c28)
except Exception:
    pass
layout["11"] = [588, 841, 720, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/29_text_12.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (720, 841), _c29)
except Exception:
    pass
layout["12"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/30_text_13.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (852, 841), _c30)
except Exception:
    pass
layout["13"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/31_text_14.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (60, 961), _c31)
except Exception:
    pass
layout["14"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/32_text_15.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (192, 961), _c32)
except Exception:
    pass
layout["15"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/33_text_16.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (324, 961), _c33)
except Exception:
    pass
layout["16"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/34_text_17.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (456, 961), _c34)
except Exception:
    pass
layout["17"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/35_text_18.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (588, 961), _c35)
except Exception:
    pass
layout["18"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/36_text_19.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (720, 961), _c36)
except Exception:
    pass
layout["19"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/37_text_20.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (852, 961), _c37)
except Exception:
    pass
layout["20"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/38_text_21.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (60, 1081), _c38)
except Exception:
    pass
layout["21"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/39_clickable_1.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (192, 721), _c39)
except Exception:
    pass
layout["1"] = [192, 721, 324, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/40_clickable_2.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (324, 721), _c40)
except Exception:
    pass
layout["2"] = [324, 721, 456, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/41_clickable_3.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (456, 721), _c41)
except Exception:
    pass
layout["3"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/42_clickable_5.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (720, 721), _c42)
except Exception:
    pass
layout["5"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/43_clickable_6.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (852, 721), _c43)
except Exception:
    pass
layout["6"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/44_clickable_7.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (60, 841), _c44)
except Exception:
    pass
layout["7"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/45_clickable_8.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (192, 841), _c45)
except Exception:
    pass
layout["8"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_13_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-15/46_clickable_9.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (324, 841), _c46)
except Exception:
    pass
layout["9"] = [324, 841, 456, 961]
