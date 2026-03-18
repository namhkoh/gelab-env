# page_id: page_eventbrite_b2798d8b10cc4118ab8cf6648f8a4077_09
# screenshot: 2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11.png
# step_index: 9/12
# task: Open Eventbrite. Search Music event in New York. Select the first one. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for the provided canvas and draw objects.
# Variables available: canvas (PIL.Image 1440x2960), draw (PIL.ImageDraw),
# and fonts: font_sm, font_md, font_lg, font_xl

# Overall background (subtle off-white)
draw.rectangle((0, 0, 1440, 2960), fill="#ffffff")

# Status bar area (top) - subtle muted grey to match screenshot top bar
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill="#bfc5c9")

# Top toolbar area (app header zone) - keep white but add subtle bottom divider/shadow
toolbar_top = status_h
toolbar_bottom = 420
draw.rectangle((0, toolbar_top, 1440, toolbar_bottom), fill="#ffffff")

# Subtle shadow / divider under toolbar (very faint)
draw.line((0, toolbar_bottom, 1440, toolbar_bottom), fill="#ececf0", width=2)

# Prominent blue underline below the page title (spans most of the content width)
# Use safe margins so pasted title/icon crops remain on top
underline_y = 324
underline_left = 48
underline_right = 1392
underline_thickness = 4
draw.rectangle((underline_left, underline_y, underline_right, underline_y + underline_thickness), fill="#2F58FF")

# Thin lighter divider slightly above underline to give a subtle double-line feel (very faint)
draw.line((underline_left, underline_y - 10, underline_right, underline_y - 10), fill="#f3f4fb", width=1)

# Section separators for the list of locations.
# Use the detected clickable row top positions to place separators between rows.
row_tops = [840, 1020, 1200, 1380, 1560, 1740, 2100, 2280, 2460]
sep_left = 40
sep_right = 1400
for y in row_tops:
    # Draw a very light hairline separator
    draw.line((sep_left, y, sep_right, y), fill="#efeff2", width=1)

# Larger section divider where "Found locations" heading sits (to visually separate header area)
found_divider_y = 740
draw.line((sep_left, found_divider_y, sep_right, found_divider_y), fill="#ececf0", width=1)

# Optional subtle background card for the "Nearby / Current location" area (keeps it distinct from list)
# Place it below the title but above the found locations heading; keep it very light and wide but avoid drawing over icon area centers.
nearby_card_top = 420
nearby_card_bottom = 720
card_left = 24
card_right = 1416
draw.rounded_rectangle((card_left, nearby_card_top, card_right, nearby_card_bottom), radius=8, fill="#ffffff", outline=None)

# A faint inner divider inside that card to suggest grouping (very subtle)
draw.line((card_left + 24, nearby_card_bottom - 1, card_right - 24, nearby_card_bottom - 1), fill="#f2f2f4", width=1)

# Footer/edge safe area (very subtle)
footer_top = 2880
draw.rectangle((0, footer_top, 1440, 2960), fill="#ffffff")

# End of structural drawing. (Icons, text and interactive elements will be pasted on top externally.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 46, 69)
    canvas.paste(_c0, (1155, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1155, 0, 1201, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 97, 65)
    canvas.paste(_c1, (1215, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1215, 0, 1312, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/02_icon_9.20.png
try:
    _c2 = get_crop(2, 59, 63)
    canvas.paste(_c2, (178, 1), _c2)
except Exception:
    pass
layout["9.20"] = [178, 1, 237, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/03_icon_9.20.png
try:
    _c3 = get_crop(3, 168, 168)
    canvas.paste(_c3, (0, 72), _c3)
except Exception:
    pass
layout["9.20"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 83, 89)
    canvas.paste(_c4, (1311, 289), _c4)
except Exception:
    pass
layout["icon_4"] = [1311, 289, 1394, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 47, 58)
    canvas.paste(_c5, (1322, 3), _c5)
except Exception:
    pass
layout["icon_5"] = [1322, 3, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/06_icon_9.20.png
try:
    _c6 = get_crop(6, 52, 64)
    canvas.paste(_c6, (116, 1), _c6)
except Exception:
    pass
layout["9.20"] = [116, 1, 168, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 53, 63)
    canvas.paste(_c7, (315, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [315, 1, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 55, 62)
    canvas.paste(_c8, (246, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [246, 1, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/09_icon_Chicago.png
try:
    _c9 = get_crop(9, 1440, 132)
    canvas.paste(_c9, (0, 1380), _c9)
except Exception:
    pass
layout["Chicago"] = [0, 1380, 1440, 1512]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/10_icon_San_Francisco.png
try:
    _c10 = get_crop(10, 1440, 132)
    canvas.paste(_c10, (0, 840), _c10)
except Exception:
    pass
layout["San_Francisco"] = [0, 840, 1440, 972]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/11_icon_Los_Angeles.png
try:
    _c11 = get_crop(11, 1440, 132)
    canvas.paste(_c11, (0, 1020), _c11)
except Exception:
    pass
layout["Los_Angeles"] = [0, 1020, 1440, 1152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/12_icon_Miami.png
try:
    _c12 = get_crop(12, 1440, 132)
    canvas.paste(_c12, (0, 1200), _c12)
except Exception:
    pass
layout["Miami"] = [0, 1200, 1440, 1332]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/13_text_9.20.png
try:
    _c13 = get_crop(13, 91, 43)
    canvas.paste(_c13, (20, 17), _c13)
except Exception:
    pass
layout["9.20"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/14_text_New_York.png
try:
    _c14 = get_crop(14, 1344, 129)
    canvas.paste(_c14, (48, 264), _c14)
except Exception:
    pass
layout["New_York"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/15_text_Nearby.png
try:
    _c15 = get_crop(15, 415, 114)
    canvas.paste(_c15, (48, 465), _c15)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/16_text_Current_location.png
try:
    _c16 = get_crop(16, 415, 114)
    canvas.paste(_c16, (48, 465), _c16)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/17_text_Found_locations.png
try:
    _c17 = get_crop(17, 311, 50)
    canvas.paste(_c17, (44, 740), _c17)
except Exception:
    pass
layout["Found_locations"] = [44, 740, 355, 790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/18_text_Washington.png
try:
    _c18 = get_crop(18, 1440, 132)
    canvas.paste(_c18, (0, 1560), _c18)
except Exception:
    pass
layout["Washington"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/19_text_Boston.png
try:
    _c19 = get_crop(19, 163, 61)
    canvas.paste(_c19, (42, 1746), _c19)
except Exception:
    pass
layout["Boston"] = [42, 1746, 205, 1807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/20_text_Massachusetts.png
try:
    _c20 = get_crop(20, 249, 39)
    canvas.paste(_c20, (47, 1814), _c20)
except Exception:
    pass
layout["Massachusetts"] = [47, 1814, 296, 1853]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/21_text_Philadelphia.png
try:
    _c21 = get_crop(21, 1440, 132)
    canvas.paste(_c21, (0, 1920), _c21)
except Exception:
    pass
layout["Philadelphia"] = [0, 1920, 1440, 2052]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/22_text_Pennsylvania.png
try:
    _c22 = get_crop(22, 214, 43)
    canvas.paste(_c22, (45, 1995), _c22)
except Exception:
    pass
layout["Pennsylvania"] = [45, 1995, 259, 2038]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/23_text_London.png
try:
    _c23 = get_crop(23, 168, 52)
    canvas.paste(_c23, (44, 2109), _c23)
except Exception:
    pass
layout["London"] = [44, 2109, 212, 2161]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/24_text_United_Kingdom.png
try:
    _c24 = get_crop(24, 263, 45)
    canvas.paste(_c24, (45, 2173), _c24)
except Exception:
    pass
layout["United_Kingdom"] = [45, 2173, 308, 2218]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/25_text_New_York.png
try:
    _c25 = get_crop(25, 212, 55)
    canvas.paste(_c25, (44, 2288), _c25)
except Exception:
    pass
layout["New_York"] = [44, 2288, 256, 2343]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/26_text_New_York.png
try:
    _c26 = get_crop(26, 154, 38)
    canvas.paste(_c26, (47, 2353), _c26)
except Exception:
    pass
layout["New_York"] = [47, 2353, 201, 2391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/27_text_Atlanta.png
try:
    _c27 = get_crop(27, 163, 52)
    canvas.paste(_c27, (44, 2468), _c27)
except Exception:
    pass
layout["Atlanta"] = [44, 2468, 207, 2520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/28_text_Georgia.png
try:
    _c28 = get_crop(28, 133, 43)
    canvas.paste(_c28, (45, 2533), _c28)
except Exception:
    pass
layout["Georgia"] = [45, 2533, 178, 2576]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/29_clickable_Boston.png
try:
    _c29 = get_crop(29, 1440, 132)
    canvas.paste(_c29, (0, 1740), _c29)
except Exception:
    pass
layout["Boston"] = [0, 1740, 1440, 1872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/30_clickable_London.png
try:
    _c30 = get_crop(30, 1440, 132)
    canvas.paste(_c30, (0, 2100), _c30)
except Exception:
    pass
layout["London"] = [0, 2100, 1440, 2232]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/31_clickable_New_York.png
try:
    _c31 = get_crop(31, 1440, 132)
    canvas.paste(_c31, (0, 2280), _c31)
except Exception:
    pass
layout["New_York"] = [0, 2280, 1440, 2412]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_09_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-11/32_clickable_Atlanta.png
try:
    _c32 = get_crop(32, 1440, 132)
    canvas.paste(_c32, (0, 2460), _c32)
except Exception:
    pass
layout["Atlanta"] = [0, 2460, 1440, 2592]
