# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_16
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-18.png
# step_index: 16/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the canvas (1440x2960)
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm/font_md/font_lg/font_xl

# Fill overall background (dominant color: white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar area at the very top (~72px) - light gray to match screenshot status bar
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#CFCFCF")

# Subtle divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#D8D8D8", width=1)

# Header/toolbar area (below status bar). Keep it white but add a faint bottom divider
header_top = status_h
header_bottom = 220
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
draw.line([(48, header_bottom), (1392, header_bottom)], fill="#EDE8F0", width=1)

# Main calendar card/background (rounded rectangle)
cal_x0, cal_y0 = 48, 220
cal_x1, cal_y1 = 1392, 1420  # covers calendar grid area
draw.rounded_rectangle(
    [(cal_x0, cal_y0), (cal_x1, cal_y1)],
    radius=24,
    fill="#FBF8FF",         # very light purple/ivory tint behind calendar
    outline="#F0E8F5",
    width=1
)

# Add a faint inner highlight band near top of the calendar card (navigation row area)
nav_band_h = 60
draw.rectangle(
    [(cal_x0 + 12, cal_y0 + 12), (cal_x1 - 12, cal_y0 + 12 + nav_band_h)],
    fill="#FFFFFF",
    outline=None
)

# End Date section card (rounded) below the calendar
end_x0, end_y0 = 48, 1420
end_x1, end_y1 = 1392, 1860
draw.rounded_rectangle(
    [(end_x0, end_y0), (end_x1, end_y1)],
    radius=20,
    fill="#FFFFFF",
    outline="#F1EAF0",
    width=1
)

# Subtle separator lines dividing major sections
# Under calendar card
sep_y = cal_y1 + 24
draw.line([(cal_x0 + 6, sep_y), (cal_x1 - 6, sep_y)], fill="#F0E8EF", width=1)

# Above bottom action area (leave the actual button area blank for pasted element)
# Draw a thin separator where the sticky action area begins (a bit above the actual button so we don't duplicate)
apply_sep_y = 2720
draw.line([(48, apply_sep_y), (1392, apply_sep_y)], fill="#E6DEE6", width=2)

# Add a faint rounded background behind large empty content area (to give structure)
content_x0, content_y0 = 48, 1860
content_x1, content_y1 = 1392, apply_sep_y - 24
draw.rounded_rectangle(
    [(content_x0, content_y0), (content_x1, content_y1)],
    radius=16,
    fill="#FFFFFF",
    outline="#F7F5F8",
    width=1
)

# Add subtle left and right margins shadow lines for visual structure
draw.line([(48, 0), (48, 2960)], fill="#F3EFF3", width=1)
draw.line([(1392, 0), (1392, 2960)], fill="#F3EFF3", width=1)

# Small accent bar under header (centered, decorative)
accent_x0, accent_x1 = 420, 1020
accent_y = header_bottom - 6
draw.line([(accent_x0, accent_y), (accent_x1, accent_y)], fill="#EDE0F3", width=6)

# (End) Structural drawing complete. Detected icons/text/buttons will be pasted on top of these areas.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 52, 71)
    canvas.paste(_c1, (1153, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1153, 0, 1205, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/02_icon_May.png
try:
    _c2 = get_crop(2, 128, 114)
    canvas.paste(_c2, (195, 610), _c2)
except Exception:
    pass
layout["May"] = [195, 610, 323, 724]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/03_icon_28.png
try:
    _c3 = get_crop(3, 132, 120)
    canvas.paste(_c3, (324, 1201), _c3)
except Exception:
    pass
layout["28"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/04_icon_7.29.png
try:
    _c4 = get_crop(4, 62, 66)
    canvas.paste(_c4, (179, 1), _c4)
except Exception:
    pass
layout["7.29"] = [179, 1, 241, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/05_icon_26.png
try:
    _c5 = get_crop(5, 132, 120)
    canvas.paste(_c5, (60, 1201), _c5)
except Exception:
    pass
layout["26"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/06_icon_May.png
try:
    _c6 = get_crop(6, 139, 115)
    canvas.paste(_c6, (321, 608), _c6)
except Exception:
    pass
layout["May"] = [321, 608, 460, 723]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/07_icon_7.29.png
try:
    _c7 = get_crop(7, 62, 66)
    canvas.paste(_c7, (113, 1), _c7)
except Exception:
    pass
layout["7.29"] = [113, 1, 175, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 65, 65)
    canvas.paste(_c8, (308, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [308, 2, 373, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 101, 70)
    canvas.paste(_c9, (1210, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1210, 0, 1311, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/10_icon_27.png
try:
    _c10 = get_crop(10, 132, 120)
    canvas.paste(_c10, (192, 1201), _c10)
except Exception:
    pass
layout["27"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/11_icon_May.png
try:
    _c11 = get_crop(11, 130, 115)
    canvas.paste(_c11, (455, 609), _c11)
except Exception:
    pass
layout["May"] = [455, 609, 585, 724]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 53, 64)
    canvas.paste(_c12, (247, 2), _c12)
except Exception:
    pass
layout["icon_12"] = [247, 2, 300, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 53, 71)
    canvas.paste(_c13, (1318, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1318, 0, 1371, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/14_icon_29.png
try:
    _c14 = get_crop(14, 132, 120)
    canvas.paste(_c14, (456, 1201), _c14)
except Exception:
    pass
layout["29"] = [456, 1201, 588, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 104, 117)
    canvas.paste(_c15, (71, 611), _c15)
except Exception:
    pass
layout["icon_15"] = [71, 611, 175, 728]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/16_icon_May.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (54, 457), _c16)
except Exception:
    pass
layout["May"] = [54, 457, 198, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/17_icon_7.29.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (12, 72), _c17)
except Exception:
    pass
layout["7.29"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/18_icon_2024.png
try:
    _c18 = get_crop(18, 129, 113)
    canvas.paste(_c18, (591, 609), _c18)
except Exception:
    pass
layout["2024"] = [591, 609, 720, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 49, 68)
    canvas.paste(_c19, (382, 1), _c19)
except Exception:
    pass
layout["icon_19"] = [382, 1, 431, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/20_icon_Next_month.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (846, 457), _c20)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/21_icon_7.29.png
try:
    _c21 = get_crop(21, 93, 64)
    canvas.paste(_c21, (15, 1), _c21)
except Exception:
    pass
layout["7.29"] = [15, 1, 108, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/22_icon_22.png
try:
    _c22 = get_crop(22, 132, 120)
    canvas.paste(_c22, (456, 1081), _c22)
except Exception:
    pass
layout["22"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/23_text_What_date.png
try:
    _c23 = get_crop(23, 318, 63)
    canvas.paste(_c23, (563, 117), _c23)
except Exception:
    pass
layout["What_date?"] = [563, 117, 881, 180]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/24_text_Start_Date.png
try:
    _c24 = get_crop(24, 580, 114)
    canvas.paste(_c24, (48, 313), _c24)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 628, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/25_text_10.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (720, 841), _c25)
except Exception:
    pass
layout["10"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/26_text_11.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (852, 841), _c26)
except Exception:
    pass
layout["11"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/27_text_12.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (60, 961), _c27)
except Exception:
    pass
layout["12"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/28_text_13.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (192, 961), _c28)
except Exception:
    pass
layout["13"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/29_text_14.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (324, 961), _c29)
except Exception:
    pass
layout["14"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/30_text_15.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (456, 961), _c30)
except Exception:
    pass
layout["15"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/31_text_16.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (588, 961), _c31)
except Exception:
    pass
layout["16"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/32_text_17.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (720, 961), _c32)
except Exception:
    pass
layout["17"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/33_text_18.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (852, 961), _c33)
except Exception:
    pass
layout["18"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/34_text_19.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (60, 1081), _c34)
except Exception:
    pass
layout["19"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/35_text_20.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (192, 1081), _c35)
except Exception:
    pass
layout["20"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/36_text_21.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (324, 1081), _c36)
except Exception:
    pass
layout["21"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/37_text_23.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (588, 1081), _c37)
except Exception:
    pass
layout["23"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/38_text_24.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (720, 1081), _c38)
except Exception:
    pass
layout["24"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/39_text_25.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (852, 1081), _c39)
except Exception:
    pass
layout["25"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/40_text_30.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (588, 1201), _c40)
except Exception:
    pass
layout["30"] = [588, 1201, 720, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/41_text_31.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (720, 1201), _c41)
except Exception:
    pass
layout["31"] = [720, 1201, 852, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/42_text_End_Date.png
try:
    _c42 = get_crop(42, 252, 63)
    canvas.paste(_c42, (45, 1453), _c42)
except Exception:
    pass
layout["End_Date"] = [45, 1453, 297, 1516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/43_clickable_1.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (456, 721), _c43)
except Exception:
    pass
layout["1"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/44_clickable_2.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (588, 721), _c44)
except Exception:
    pass
layout["2"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/45_clickable_3.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (720, 721), _c45)
except Exception:
    pass
layout["3"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/46_clickable_4.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (852, 721), _c46)
except Exception:
    pass
layout["4"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/47_clickable_5.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (60, 841), _c47)
except Exception:
    pass
layout["5"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/48_clickable_6.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (192, 841), _c48)
except Exception:
    pass
layout["6"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/49_clickable_7.png
try:
    _c49 = get_crop(49, 132, 120)
    canvas.paste(_c49, (324, 841), _c49)
except Exception:
    pass
layout["7"] = [324, 841, 456, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/50_clickable_8.png
try:
    _c50 = get_crop(50, 132, 120)
    canvas.paste(_c50, (456, 841), _c50)
except Exception:
    pass
layout["8"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/51_clickable_9.png
try:
    _c51 = get_crop(51, 132, 120)
    canvas.paste(_c51, (588, 841), _c51)
except Exception:
    pass
layout["9"] = [588, 841, 720, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_16_2024_4_23_19_27_45f56b06f31541079045047b6d542613-18/52_clickable_Choose_a_date.png
try:
    _c52 = get_crop(52, 638, 144)
    canvas.paste(_c52, (48, 1490), _c52)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]
