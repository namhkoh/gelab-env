# page_id: page_seatgeek_68e3462c14734440a7ace3fed432a10d_04
# screenshot: 2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-7.png
# step_index: 4/13
# task: Open SeatGeek and change the current location to Los Angeles. Then find the first concert show and track its performer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the provided canvas (1440x2960).
# Available variables: canvas (PIL Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

# --- Colors ---
bg_color = "#FFFFFF"         # main background (dominant)
status_bg = "#F4F4F4"       # status bar background
search_bg = "#F7F7F7"       # search bar background
outline = "#E6E6E6"         # subtle outlines / strokes
divider = "#EDEDED"         # thin separators
muted_divider = "#F2F2F2"   # very light divider

w, h = canvas.size

# Fill overall background (canvas starts white but ensure consistency)
draw.rectangle([0, 0, w, h], fill=bg_color)

# --- Status bar area (top) ---
status_h = 88
draw.rectangle([0, 0, w, status_h], fill=status_bg)
# status bar bottom hairline
draw.line([(0, status_h), (w, status_h)], fill=outline, width=1)

# --- Search bar (rounded) ---
search_left = 48
search_right = w - 48
search_top = 112
search_bottom = search_top + 144   # height ~144 to match screenshot spacing
search_radius = 22

# subtle shadow under the search bar (soft line to imply elevation)
shadow_y = search_bottom + 6
draw.line([(search_left+2, shadow_y), (search_right-2, shadow_y)], fill="#F0F0F0", width=2)

# search box background and border (no icons/text)
try:
    # Pillow's rounded_rectangle if available
    draw.rounded_rectangle([search_left, search_top, search_right, search_bottom],
                           radius=search_radius, fill=search_bg, outline=outline, width=1)
except Exception:
    # fallback: plain rectangle if rounded not available
    draw.rectangle([search_left, search_top, search_right, search_bottom], fill=search_bg, outline=outline)

# --- Divider under search area ---
divider_y = search_bottom + 20
draw.line([(24, divider_y), (w-24, divider_y)], fill=divider, width=1)

# --- "Use my current location" row area (background separators only) ---
use_row_top = 316
use_row_height = 194
use_row_bottom = use_row_top + use_row_height

# Draw thin top and bottom separators for the row
draw.line([(0, use_row_top), (w, use_row_top)], fill=muted_divider, width=1)
draw.line([(0, use_row_bottom), (w, use_row_bottom)], fill=muted_divider, width=1)

# Slight highlight behind the row (very subtle)
highlight_left = 24
highlight_right = w - 24
highlight_top = use_row_top + 8
highlight_bottom = use_row_bottom - 8
draw.rectangle([highlight_left, highlight_top, highlight_right, highlight_bottom], fill=bg_color)

# --- "Recent locations" header separator (under header) ---
# The header text itself will be pasted later; draw a faint rule below the header area to structure page.
recent_header_top = 562
recent_header_bottom = recent_header_top + 54
header_divider_y = recent_header_bottom + 8
draw.line([(24, header_divider_y), (w-24, header_divider_y)], fill=muted_divider, width=1)

# --- Page bottom / large content area background hint ---
# Provide a very subtle left gutter guide and bottom whitespace area to mimic the app's spacious layout.
gutter_x = 44
draw.line([(gutter_x, header_divider_y + 24), (gutter_x, h - 40)], fill=muted_divider, width=1)

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_04_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-7/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 50, 69)
    canvas.paste(_c0, (1152, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1152, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_04_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-7/01_icon_Recent_locations.png
try:
    _c1 = get_crop(1, 117, 137)
    canvas.paste(_c1, (46, 678), _c1)
except Exception:
    pass
layout["Recent_locations"] = [46, 678, 163, 815]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_04_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-7/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 101, 68)
    canvas.paste(_c2, (1213, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1213, 0, 1314, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_04_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-7/03_icon_8.30_my.png
try:
    _c3 = get_crop(3, 168, 144)
    canvas.paste(_c3, (0, 122), _c3)
except Exception:
    pass
layout["8.30_my"] = [0, 122, 168, 266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_04_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-7/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 51, 65)
    canvas.paste(_c4, (1320, 1), _c4)
except Exception:
    pass
layout["icon_4"] = [1320, 1, 1371, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_04_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-7/05_icon_8.30_my.png
try:
    _c5 = get_crop(5, 58, 63)
    canvas.paste(_c5, (181, 2), _c5)
except Exception:
    pass
layout["8.30_my"] = [181, 2, 239, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_04_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 59, 60)
    canvas.paste(_c6, (310, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [310, 3, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_04_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-7/07_icon_Bearch_by_city.png
try:
    _c7 = get_crop(7, 98, 127)
    canvas.paste(_c7, (222, 132), _c7)
except Exception:
    pass
layout["Bearch_by_city"] = [222, 132, 320, 259]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_04_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 59, 62)
    canvas.paste(_c8, (245, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [245, 2, 304, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_04_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-7/09_icon_Recent_locations.png
try:
    _c9 = get_crop(9, 119, 120)
    canvas.paste(_c9, (44, 355), _c9)
except Exception:
    pass
layout["Recent_locations"] = [44, 355, 163, 475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_04_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-7/10_icon_Clear.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 122), _c10)
except Exception:
    pass
layout["Clear"] = [1248, 122, 1392, 266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_04_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-7/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 49, 58)
    canvas.paste(_c11, (383, 5), _c11)
except Exception:
    pass
layout["icon_11"] = [383, 5, 432, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_04_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-7/12_text_8.30_my.png
try:
    _c12 = get_crop(12, 156, 52)
    canvas.paste(_c12, (16, 9), _c12)
except Exception:
    pass
layout["8.30_my"] = [16, 9, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_04_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-7/13_text_Bearch_by_city.png
try:
    _c13 = get_crop(13, 936, 144)
    canvas.paste(_c13, (312, 122), _c13)
except Exception:
    pass
layout["Bearch_by_city"] = [312, 122, 1248, 266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_04_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-7/14_text_Use_my_current_location.png
try:
    _c14 = get_crop(14, 1440, 194)
    canvas.paste(_c14, (0, 316), _c14)
except Exception:
    pass
layout["Use_my_current_location"] = [0, 316, 1440, 510]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_04_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-7/15_text_Recent_locations.png
try:
    _c15 = get_crop(15, 441, 54)
    canvas.paste(_c15, (44, 562), _c15)
except Exception:
    pass
layout["Recent_locations"] = [44, 562, 485, 616]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_04_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-7/16_text_New_York_NY.png
try:
    _c16 = get_crop(16, 299, 55)
    canvas.paste(_c16, (213, 714), _c16)
except Exception:
    pass
layout["New_York,_NY"] = [213, 714, 512, 769]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_04_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-7/17_clickable_New_York_NY.png
try:
    _c17 = get_crop(17, 1440, 168)
    canvas.paste(_c17, (0, 657), _c17)
except Exception:
    pass
layout["New_York,_NY"] = [0, 657, 1440, 825]
