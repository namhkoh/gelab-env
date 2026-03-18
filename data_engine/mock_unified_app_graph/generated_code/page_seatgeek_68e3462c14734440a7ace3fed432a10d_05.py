# page_id: page_seatgeek_68e3462c14734440a7ace3fed432a10d_05
# screenshot: 2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8.png
# step_index: 5/13
# task: Open SeatGeek and change the current location to Los Angeles. Then find the first concert show and track its performer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_bar_height = 72
draw.rectangle([(0, 0), (1440, status_bar_height)], fill="#efefef")

# subtle bottom border under status bar
draw.line([(0, status_bar_height), (1440, status_bar_height)], fill="#dddddd", width=1)

# Search input background (rounded)
search_left = 80
search_top = 84
search_right = 1360
search_bottom = 168
search_radius = 36
# subtle shadow (very light, offset)
draw.rounded_rectangle(
    [(search_left, search_top + 4), (search_right, search_bottom + 6)],
    radius=search_radius,
    fill="#f3f3f3"
)
# actual search box
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=search_radius,
    fill="#ffffff",
    outline="#e6e6e6",
    width=1
)

# Divider line under search area
divider_y = search_bottom + 12
draw.line([(40, divider_y), (1400, divider_y)], fill="#e9e9e9", width=1)

# Group container for first two location results (rounded card-like area)
group1_top = divider_y + 12
group1_bottom = 520
group_left = 24
group_right = 1416
group_radius = 8
# faint background to separate from page (very subtle)
draw.rounded_rectangle(
    [(group_left, group1_top), (group_right, group1_bottom)],
    radius=group_radius,
    fill="#ffffff",
    outline="#f6f6f6",
    width=1
)

# Thin separators between list items inside the group
# First item separator (below "Los Angeles, CA" item)
sep1_y = 316
draw.line([(40, sep1_y), (1400, sep1_y)], fill="#ececec", width=1)

# Second item separator (below "West Los Angeles, CA" item)
sep2_y = 484
draw.line([(40, sep2_y), (1400, sep2_y)], fill="#ececec", width=1)

# Section divider before "Recent locations" heading
recent_heading_divider_y = 700
draw.line([(24, recent_heading_divider_y), (1416, recent_heading_divider_y)], fill="#f0f0f0", width=1)

# Small card background for recent-locations area header area (subtle)
recent_card_top = recent_heading_divider_y + 12
recent_card_bottom = recent_card_top + 140
draw.rectangle([(24, recent_card_top), (1416, recent_card_bottom)], fill="#ffffff", outline="#fafafa")

# additional faint vertical padding separators to keep layout visually consistent
draw.line([(24, group1_top), (24, recent_card_bottom)], fill="#ffffff", width=1)
draw.line([(1416, group1_top), (1416, recent_card_bottom)], fill="#ffffff", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/00_icon_Recent_locations.png
try:
    _c0 = get_crop(0, 121, 142)
    canvas.paste(_c0, (45, 856), _c0)
except Exception:
    pass
layout["Recent_locations"] = [45, 856, 166, 998]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 51, 69)
    canvas.paste(_c1, (1152, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1152, 0, 1203, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 101, 69)
    canvas.paste(_c2, (1212, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1212, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/03_icon_8.30_my.png
try:
    _c3 = get_crop(3, 168, 144)
    canvas.paste(_c3, (0, 122), _c3)
except Exception:
    pass
layout["8.30_my"] = [0, 122, 168, 266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/04_icon_Los_Angeles.png
try:
    _c4 = get_crop(4, 95, 116)
    canvas.paste(_c4, (224, 138), _c4)
except Exception:
    pass
layout["Los_Angeles"] = [224, 138, 319, 254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 52, 66)
    canvas.paste(_c5, (1319, 1), _c5)
except Exception:
    pass
layout["icon_5"] = [1319, 1, 1371, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/06_icon_8.30_my.png
try:
    _c6 = get_crop(6, 59, 63)
    canvas.paste(_c6, (180, 2), _c6)
except Exception:
    pass
layout["8.30_my"] = [180, 2, 239, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 60, 62)
    canvas.paste(_c7, (309, 2), _c7)
except Exception:
    pass
layout["icon_7"] = [309, 2, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/08_icon_Clear.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1248, 122), _c8)
except Exception:
    pass
layout["Clear"] = [1248, 122, 1392, 266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/09_icon_Los_Angeles_CA.png
try:
    _c9 = get_crop(9, 114, 124)
    canvas.paste(_c9, (46, 345), _c9)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [46, 345, 160, 469]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 58, 62)
    canvas.paste(_c10, (246, 2), _c10)
except Exception:
    pass
layout["icon_10"] = [246, 2, 304, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 48, 58)
    canvas.paste(_c11, (383, 5), _c11)
except Exception:
    pass
layout["icon_11"] = [383, 5, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/12_icon_Recent_locations.png
try:
    _c12 = get_crop(12, 121, 133)
    canvas.paste(_c12, (43, 507), _c12)
except Exception:
    pass
layout["Recent_locations"] = [43, 507, 164, 640]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/13_icon_8.30_my.png
try:
    _c13 = get_crop(13, 53, 65)
    canvas.paste(_c13, (115, 1), _c13)
except Exception:
    pass
layout["8.30_my"] = [115, 1, 168, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/14_icon_Los_Angeles.png
try:
    _c14 = get_crop(14, 936, 144)
    canvas.paste(_c14, (312, 122), _c14)
except Exception:
    pass
layout["Los_Angeles"] = [312, 122, 1248, 266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/15_text_Los_Angeles_CA.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 316), _c15)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [0, 316, 1440, 484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/16_text_West_Los_Angeles_CA.png
try:
    _c16 = get_crop(16, 1440, 168)
    canvas.paste(_c16, (0, 484), _c16)
except Exception:
    pass
layout["West_Los_Angeles,_CA"] = [0, 484, 1440, 652]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/17_text_Recent_locations.png
try:
    _c17 = get_crop(17, 441, 55)
    canvas.paste(_c17, (44, 742), _c17)
except Exception:
    pass
layout["Recent_locations"] = [44, 742, 485, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/18_text_New_York_NY.png
try:
    _c18 = get_crop(18, 299, 56)
    canvas.paste(_c18, (213, 894), _c18)
except Exception:
    pass
layout["New_York,_NY"] = [213, 894, 512, 950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_05_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-8/19_clickable_New_York_NY.png
try:
    _c19 = get_crop(19, 1440, 168)
    canvas.paste(_c19, (0, 837), _c19)
except Exception:
    pass
layout["New_York,_NY"] = [0, 837, 1440, 1005]
