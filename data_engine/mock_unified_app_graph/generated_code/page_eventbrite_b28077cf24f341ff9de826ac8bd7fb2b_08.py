# page_id: page_eventbrite_b28077cf24f341ff9de826ac8bd7fb2b_08
# screenshot: 2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10.png
# step_index: 8/16
# task: Open Eventbrite. Explore 'Wellness' events in Washington. Filter to only show free events. Add the first non-promoted event to favorite and follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (mostly white to match the app)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Top status bar area (background only; icons are pasted later)
status_h = 56
draw.rectangle([(0, 0), (1440, status_h)], fill=(200, 200, 200))
# subtle divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill=(220, 220, 220), width=1)

# Header/search underline (blue accent line across with left/right margins)
underline_left = 48
underline_right = 1440 - 48
underline_y = 420
draw.rectangle([(underline_left, underline_y), (underline_right, underline_y + 4)], fill=(43, 82, 255))

# Light subtle shadow under the header underline to give depth
draw.line([(underline_left, underline_y + 5), (underline_right, underline_y + 5)], fill=(245, 245, 247), width=1)

# Nearby row background (rounded card style behind the row group)
nearby_top = 440
nearby_bottom = 560
card_margin = 36
draw.rounded_rectangle(
    [(card_margin, nearby_top), (1440 - card_margin, nearby_bottom)],
    radius=18,
    fill=(248, 250, 255),
    outline=None
)

# Divider line above the "Found locations" section
found_div_y = 720
draw.line([(48, found_div_y), (1440 - 48, found_div_y)], fill=(235, 236, 241), width=1)

# Draw subtle separators for each found-location row.
# Rows detected begin at y positions: 840, 1020, 1200, 1380, 1560, 1740, 1920, 2100, 2280, 2460
row_starts = [840, 1020, 1200, 1380, 1560, 1740, 1920, 2100, 2280, 2460]
for y in row_starts:
    # a faint full-width separator with left/right padding
    draw.line([(48, y), (1440 - 48, y)], fill=(245, 246, 250), width=1)

# Subtle grouping background for the main list area to separate from header
list_bg_top = found_div_y + 8
list_bg_bottom = 2600
draw.rectangle([(20, list_bg_top), (1440 - 20, list_bg_bottom)], fill=(255, 255, 255))

# Very faint left gutter guide to echo the UI layout (non-intrusive)
draw.line([(48, underline_y - 160), (48, list_bg_bottom)], fill=(250, 250, 251), width=1)

# Bottom safe-area subtle divider
draw.line([(0, 2950), (1440, 2950)], fill=(240, 241, 245), width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 47, 69)
    canvas.paste(_c0, (1155, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1155, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 98, 65)
    canvas.paste(_c1, (1214, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1214, 0, 1312, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/02_icon_4.44.png
try:
    _c2 = get_crop(2, 62, 63)
    canvas.paste(_c2, (179, 1), _c2)
except Exception:
    pass
layout["4.44"] = [179, 1, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/03_icon_4.44.png
try:
    _c3 = get_crop(3, 62, 67)
    canvas.paste(_c3, (112, 0), _c3)
except Exception:
    pass
layout["4.44"] = [112, 0, 174, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/04_icon_4.44.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["4.44"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 63, 61)
    canvas.paste(_c5, (308, 2), _c5)
except Exception:
    pass
layout["icon_5"] = [308, 2, 371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 85, 91)
    canvas.paste(_c6, (1310, 288), _c6)
except Exception:
    pass
layout["icon_6"] = [1310, 288, 1395, 379]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 54, 62)
    canvas.paste(_c7, (246, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [246, 1, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 47, 59)
    canvas.paste(_c8, (1322, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [1322, 2, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/09_icon_Los_Angeles.png
try:
    _c9 = get_crop(9, 1440, 132)
    canvas.paste(_c9, (0, 1020), _c9)
except Exception:
    pass
layout["Los_Angeles"] = [0, 1020, 1440, 1152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/10_icon_San_Francisco.png
try:
    _c10 = get_crop(10, 1440, 132)
    canvas.paste(_c10, (0, 840), _c10)
except Exception:
    pass
layout["San_Francisco"] = [0, 840, 1440, 972]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/11_icon_Chicago.png
try:
    _c11 = get_crop(11, 1440, 132)
    canvas.paste(_c11, (0, 1380), _c11)
except Exception:
    pass
layout["Chicago"] = [0, 1380, 1440, 1512]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/12_icon_4.44.png
try:
    _c12 = get_crop(12, 93, 64)
    canvas.paste(_c12, (15, 1), _c12)
except Exception:
    pass
layout["4.44"] = [15, 1, 108, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/13_icon_Miami.png
try:
    _c13 = get_crop(13, 1440, 132)
    canvas.paste(_c13, (0, 1200), _c13)
except Exception:
    pass
layout["Miami"] = [0, 1200, 1440, 1332]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/14_icon_District_of_Columbia.png
try:
    _c14 = get_crop(14, 1440, 132)
    canvas.paste(_c14, (0, 1560), _c14)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/15_text_Washington.png
try:
    _c15 = get_crop(15, 1344, 129)
    canvas.paste(_c15, (48, 264), _c15)
except Exception:
    pass
layout["Washington"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/16_text_Nearby.png
try:
    _c16 = get_crop(16, 415, 114)
    canvas.paste(_c16, (48, 465), _c16)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/17_text_Current_location.png
try:
    _c17 = get_crop(17, 415, 114)
    canvas.paste(_c17, (48, 465), _c17)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/18_text_Found_locations.png
try:
    _c18 = get_crop(18, 311, 50)
    canvas.paste(_c18, (44, 740), _c18)
except Exception:
    pass
layout["Found_locations"] = [44, 740, 355, 790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/19_text_Boston.png
try:
    _c19 = get_crop(19, 163, 61)
    canvas.paste(_c19, (42, 1746), _c19)
except Exception:
    pass
layout["Boston"] = [42, 1746, 205, 1807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/20_text_Massachusetts.png
try:
    _c20 = get_crop(20, 249, 39)
    canvas.paste(_c20, (47, 1814), _c20)
except Exception:
    pass
layout["Massachusetts"] = [47, 1814, 296, 1853]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/21_text_Philadelphia.png
try:
    _c21 = get_crop(21, 1440, 132)
    canvas.paste(_c21, (0, 1920), _c21)
except Exception:
    pass
layout["Philadelphia"] = [0, 1920, 1440, 2052]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/22_text_Pennsylvania.png
try:
    _c22 = get_crop(22, 214, 43)
    canvas.paste(_c22, (45, 1995), _c22)
except Exception:
    pass
layout["Pennsylvania"] = [45, 1995, 259, 2038]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/23_text_London.png
try:
    _c23 = get_crop(23, 168, 52)
    canvas.paste(_c23, (44, 2109), _c23)
except Exception:
    pass
layout["London"] = [44, 2109, 212, 2161]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/24_text_United_Kingdom.png
try:
    _c24 = get_crop(24, 263, 45)
    canvas.paste(_c24, (45, 2173), _c24)
except Exception:
    pass
layout["United_Kingdom"] = [45, 2173, 308, 2218]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/25_text_New_York.png
try:
    _c25 = get_crop(25, 212, 55)
    canvas.paste(_c25, (44, 2288), _c25)
except Exception:
    pass
layout["New_York"] = [44, 2288, 256, 2343]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/26_text_New_York.png
try:
    _c26 = get_crop(26, 154, 38)
    canvas.paste(_c26, (47, 2353), _c26)
except Exception:
    pass
layout["New_York"] = [47, 2353, 201, 2391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/27_text_Atlanta.png
try:
    _c27 = get_crop(27, 163, 52)
    canvas.paste(_c27, (44, 2468), _c27)
except Exception:
    pass
layout["Atlanta"] = [44, 2468, 207, 2520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/28_text_Georgia.png
try:
    _c28 = get_crop(28, 133, 43)
    canvas.paste(_c28, (45, 2533), _c28)
except Exception:
    pass
layout["Georgia"] = [45, 2533, 178, 2576]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/29_clickable_Boston.png
try:
    _c29 = get_crop(29, 1440, 132)
    canvas.paste(_c29, (0, 1740), _c29)
except Exception:
    pass
layout["Boston"] = [0, 1740, 1440, 1872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/30_clickable_London.png
try:
    _c30 = get_crop(30, 1440, 132)
    canvas.paste(_c30, (0, 2100), _c30)
except Exception:
    pass
layout["London"] = [0, 2100, 1440, 2232]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/31_clickable_New_York.png
try:
    _c31 = get_crop(31, 1440, 132)
    canvas.paste(_c31, (0, 2280), _c31)
except Exception:
    pass
layout["New_York"] = [0, 2280, 1440, 2412]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_08_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-10/32_clickable_Atlanta.png
try:
    _c32 = get_crop(32, 1440, 132)
    canvas.paste(_c32, (0, 2460), _c32)
except Exception:
    pass
layout["Atlanta"] = [0, 2460, 1440, 2592]
