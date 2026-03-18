# page_id: page_seatgeek_3297e3a54de8487d8c2b67798919ed2f_04
# screenshot: 2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-7.png
# step_index: 4/11
# task: Open SeatGeek. Search "Comedy Show in Los Angeles". Find the top recommendation. When is the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar background
status_h = 96
draw.rectangle([(0, 0), (1440, status_h)], fill="#F2F2F2")
draw.line([(0, status_h), (1440, status_h)], fill="#E0E0E0", width=1)

# Search bar background (rounded) with subtle shadow
search_top = 108
search_bottom = 220
search_left = 80
search_right = 1360
# shadow (slightly darker strip behind rounded rect)
shadow_offset = 6
draw.rounded_rectangle(
    [(search_left, search_top + shadow_offset), (search_right, search_bottom + shadow_offset)],
    radius=36, fill="#F0F0F0"
)
# actual search field
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=36, fill="#FAFAFA", outline="#E6E6E6", width=1
)

# Divider under search area
divider_y = search_bottom + 10
draw.line([(48, divider_y), (1392, divider_y)], fill="#EDEDED", width=1)

# "Use my current location" row separators (don't draw text/icons)
use_row_top = 316
use_row_bottom = use_row_top + 194
draw.line([(32, use_row_top), (1408, use_row_top)], fill="#EDEDED", width=1)
draw.line([(32, use_row_bottom), (1408, use_row_bottom)], fill="#EDEDED", width=1)

# Section header separator above "Recent locations"
recent_hdr_y = 562
draw.line([(32, recent_hdr_y - 18), (1408, recent_hdr_y - 18)], fill="#F0F0F0", width=1)

# List item separators for recent locations (two items)
item1_top = 657
item1_bottom = item1_top + 168
item2_top = 825
item2_bottom = item2_top + 168

draw.line([(32, item1_top), (1408, item1_top)], fill="#F2F2F2", width=1)
draw.line([(32, item2_top), (1408, item2_top)], fill="#F2F2F2", width=1)
draw.line([(32, item2_bottom), (1408, item2_bottom)], fill="#F2F2F2", width=1)

# Add a subtle vertical guide on left for visual alignment (very light)
draw.line([(40, status_h + 12), (40, 2800)], fill="#FBFBFB", width=1)

# Bottom area keep white (no additional elements) - ensure full canvas fill stays white
draw.rectangle([(0, 2800), (1440, 2960)], fill="#FFFFFF")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_04_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-7/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 50, 70)
    canvas.paste(_c0, (1152, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1152, 0, 1202, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_04_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-7/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 65, 65)
    canvas.paste(_c1, (241, 2), _c1)
except Exception:
    pass
layout["icon_1"] = [241, 2, 306, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_04_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-7/02_icon_7_10_my.png
try:
    _c2 = get_crop(2, 168, 144)
    canvas.paste(_c2, (0, 122), _c2)
except Exception:
    pass
layout["7:10_my"] = [0, 122, 168, 266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_04_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-7/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 100, 69)
    canvas.paste(_c3, (1213, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1213, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_04_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-7/04_icon_Los_Angeles_CA.png
try:
    _c4 = get_crop(4, 124, 137)
    canvas.paste(_c4, (43, 848), _c4)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [43, 848, 167, 985]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_04_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 63, 65)
    canvas.paste(_c5, (311, 2), _c5)
except Exception:
    pass
layout["icon_5"] = [311, 2, 374, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_04_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 52, 67)
    canvas.paste(_c6, (1319, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1319, 1, 1371, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_04_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-7/07_icon_7_10_my.png
try:
    _c7 = get_crop(7, 49, 65)
    canvas.paste(_c7, (184, 1), _c7)
except Exception:
    pass
layout["7:10_my"] = [184, 1, 233, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_04_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-7/08_icon_Recent_locations.png
try:
    _c8 = get_crop(8, 117, 118)
    canvas.paste(_c8, (45, 357), _c8)
except Exception:
    pass
layout["Recent_locations"] = [45, 357, 162, 475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_04_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-7/09_icon_Clear.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 122), _c9)
except Exception:
    pass
layout["Clear"] = [1248, 122, 1392, 266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_04_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-7/10_icon_pearch_by_city.png
try:
    _c10 = get_crop(10, 100, 122)
    canvas.paste(_c10, (222, 132), _c10)
except Exception:
    pass
layout["pearch_by_city"] = [222, 132, 322, 254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_04_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-7/11_text_7_10_my.png
try:
    _c11 = get_crop(11, 153, 52)
    canvas.paste(_c11, (19, 9), _c11)
except Exception:
    pass
layout["7:10_my"] = [19, 9, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_04_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-7/12_text_pearch_by_city.png
try:
    _c12 = get_crop(12, 936, 144)
    canvas.paste(_c12, (312, 122), _c12)
except Exception:
    pass
layout["pearch_by_city"] = [312, 122, 1248, 266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_04_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-7/13_text_Use_my_current_location.png
try:
    _c13 = get_crop(13, 1440, 194)
    canvas.paste(_c13, (0, 316), _c13)
except Exception:
    pass
layout["Use_my_current_location"] = [0, 316, 1440, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_04_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-7/14_text_Recent_locations.png
try:
    _c14 = get_crop(14, 441, 54)
    canvas.paste(_c14, (44, 562), _c14)
except Exception:
    pass
layout["Recent_locations"] = [44, 562, 485, 616]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_04_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-7/15_text_New_York_NY.png
try:
    _c15 = get_crop(15, 299, 55)
    canvas.paste(_c15, (213, 714), _c15)
except Exception:
    pass
layout["New_York,_NY"] = [213, 714, 512, 769]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_04_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-7/16_text_Los_Angeles_CA.png
try:
    _c16 = get_crop(16, 1440, 168)
    canvas.paste(_c16, (0, 825), _c16)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [0, 825, 1440, 993]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_04_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-7/17_clickable_New_York_NY.png
try:
    _c17 = get_crop(17, 1440, 168)
    canvas.paste(_c17, (0, 657), _c17)
except Exception:
    pass
layout["New_York,_NY"] = [0, 657, 1440, 825]
