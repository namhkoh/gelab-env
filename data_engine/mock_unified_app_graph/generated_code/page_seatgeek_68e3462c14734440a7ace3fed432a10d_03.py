# page_id: page_seatgeek_68e3462c14734440a7ace3fed432a10d_03
# screenshot: 2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-6.png
# step_index: 3/13
# task: Open SeatGeek and change the current location to Los Angeles. Then find the first concert show and track its performer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
draw.rectangle([(0, 0), (1440, 2960)], fill=(250, 250, 250))

# Status bar (top area)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(236, 236, 236))
# subtle bottom hairline under status bar
draw.line([(0, status_h), (1440, status_h)], fill=(220, 220, 220), width=1)

# Search box background (rounded rectangle)
search_left = 70
search_top = 110
search_right = 1370
search_bottom = 266
search_radius = 36
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=search_radius,
    fill=(245, 245, 245),
    outline=None
)
# subtle inner divider under search box
search_div_y = search_bottom + 12
draw.line([(40, search_div_y), (1400, search_div_y)], fill=(230, 230, 230), width=1)

# "Use my current location" row background area (card-like spacing)
use_row_top = 316
use_row_bottom = use_row_top + 194
# keep row white but draw separators above and below
draw.rectangle([(0, use_row_top), (1440, use_row_bottom)], fill=(255, 255, 255))
draw.line([(24, use_row_top), (1416, use_row_top)], fill=(230, 230, 230), width=1)
draw.line([(24, use_row_bottom), (1416, use_row_bottom)], fill=(230, 230, 230), width=1)

# Recent locations header area (no text drawn per instructions)
recent_header_top = use_row_bottom + 36
recent_header_bottom = recent_header_top + 54
# a light left padding band to visually separate section (very subtle)
draw.rectangle([(0, recent_header_top), (1440, recent_header_top+1)], fill=(245, 245, 245))
# vertical spacing preserved; draw a faint guideline under header
draw.line([(24, recent_header_bottom + 24), (1416, recent_header_bottom + 24)], fill=(245, 245, 245), width=1)

# Recent item clickable row background (subtle card / hit area)
recent_row_top = 657
recent_row_bottom = recent_row_top + 168
# keep overall white but add a very faint rounded rect to indicate tappable area
draw.rounded_rectangle(
    [(16, recent_row_top + 6), (1424, recent_row_bottom - 6)],
    radius=12,
    fill=(255, 255, 255),
    outline=(245, 245, 245)
)
# separator line below the recent item
draw.line([(24, recent_row_bottom), (1416, recent_row_bottom)], fill=(240, 240, 240), width=1)

# Large empty content area remains white (no extra drawing) to avoid overlapping pasted content.

# Optional subtle left edge margin guide and an overall light vignette at bottom (very faint)
draw.rectangle([(0, 2800), (1440, 2960)], fill=(250, 250, 250))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_03_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-6/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 49, 69)
    canvas.paste(_c0, (1153, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1153, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_03_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-6/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 100, 69)
    canvas.paste(_c1, (1214, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1214, 0, 1314, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_03_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-6/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 55, 59)
    canvas.paste(_c2, (313, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [313, 3, 368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_03_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-6/03_icon_Recent_locations.png
try:
    _c3 = get_crop(3, 117, 140)
    canvas.paste(_c3, (46, 677), _c3)
except Exception:
    pass
layout["Recent_locations"] = [46, 677, 163, 817]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_03_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-6/04_icon_8.30_my.png
try:
    _c4 = get_crop(4, 168, 144)
    canvas.paste(_c4, (0, 122), _c4)
except Exception:
    pass
layout["8.30_my"] = [0, 122, 168, 266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_03_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-6/05_icon_8.30_my.png
try:
    _c5 = get_crop(5, 56, 61)
    canvas.paste(_c5, (182, 2), _c5)
except Exception:
    pass
layout["8.30_my"] = [182, 2, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_03_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-6/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 54, 60)
    canvas.paste(_c6, (247, 2), _c6)
except Exception:
    pass
layout["icon_6"] = [247, 2, 301, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_03_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-6/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 50, 66)
    canvas.paste(_c7, (1320, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1320, 1, 1370, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_03_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-6/08_icon_Recent_locations.png
try:
    _c8 = get_crop(8, 117, 122)
    canvas.paste(_c8, (44, 354), _c8)
except Exception:
    pass
layout["Recent_locations"] = [44, 354, 161, 476]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_03_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-6/09_icon_Search_by_city.png
try:
    _c9 = get_crop(9, 88, 118)
    canvas.paste(_c9, (232, 140), _c9)
except Exception:
    pass
layout["Search_by_city"] = [232, 140, 320, 258]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_03_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-6/10_icon_8.30_my.png
try:
    _c10 = get_crop(10, 52, 64)
    canvas.paste(_c10, (116, 1), _c10)
except Exception:
    pass
layout["8.30_my"] = [116, 1, 168, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_03_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-6/11_icon_Clear.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 122), _c11)
except Exception:
    pass
layout["Clear"] = [1248, 122, 1392, 266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_03_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-6/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 50, 60)
    canvas.paste(_c12, (382, 2), _c12)
except Exception:
    pass
layout["icon_12"] = [382, 2, 432, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_03_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-6/13_icon_Use_my_current_location.png
try:
    _c13 = get_crop(13, 1440, 194)
    canvas.paste(_c13, (0, 316), _c13)
except Exception:
    pass
layout["Use_my_current_location"] = [0, 316, 1440, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_03_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-6/14_text_Search_by_city.png
try:
    _c14 = get_crop(14, 936, 144)
    canvas.paste(_c14, (312, 122), _c14)
except Exception:
    pass
layout["Search_by_city"] = [312, 122, 1248, 266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_03_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-6/15_text_Recent_locations.png
try:
    _c15 = get_crop(15, 441, 54)
    canvas.paste(_c15, (44, 562), _c15)
except Exception:
    pass
layout["Recent_locations"] = [44, 562, 485, 616]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_03_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-6/16_text_New_York_NY.png
try:
    _c16 = get_crop(16, 299, 55)
    canvas.paste(_c16, (213, 714), _c16)
except Exception:
    pass
layout["New_York,_NY"] = [213, 714, 512, 769]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_03_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-6/17_clickable_New_York_NY.png
try:
    _c17 = get_crop(17, 1440, 168)
    canvas.paste(_c17, (0, 657), _c17)
except Exception:
    pass
layout["New_York,_NY"] = [0, 657, 1440, 825]
