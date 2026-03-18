# page_id: page_seatgeek_1e6c9e893d9e4bc99959744188677162_03
# screenshot: 2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6.png
# step_index: 3/8
# task: Open SeatGeek. Search "Radio City Music Hall" and then add the venue to favorite. Who are the performers of the top recommended event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_h = 80
draw.rectangle((0, 0, 1440, status_h), fill="#FBFBFB")
draw.line((0, status_h, 1440, status_h), fill="#E6E6E6", width=1)

# Search bar background (rounded)
search_left = 40
search_top = 110
search_right = 1400
search_bottom = search_top + 144
draw.rounded_rectangle(
    (search_left, search_top, search_right, search_bottom),
    radius=28,
    fill="#F5F5F6",
    outline="#E8E8E8",
    width=1
)

# Subtle divider under search area
divider_y = search_bottom + 6
draw.line((32, divider_y, 1408, divider_y), fill="#E6E6E6", width=1)

# Light horizontal rule separating main lists (between Recent searches and Suggestions)
# Using the known layout of list rows (~1143 top for last recent item + its height 168)
large_divider_y = 1143 + 168 + 5  # ~1316
draw.line((24, large_divider_y, 1416, large_divider_y), fill="#E6E6E6", width=1)

# Bottom navigation bar background and top border/shadow
nav_top = 2792
draw.rectangle((0, nav_top, 1440, 2960), fill="#FFFFFF")
# subtle top border/shadow
draw.line((0, nav_top, 1440, nav_top), fill="#EDEDED", width=1)
draw.rectangle((0, nav_top - 6, 1440, nav_top), fill="#FAFAFA")

# Optional subtle background band behind content area (very light)
# This provides the soft off-white tone visible in the UI without drawing any icons/text.
draw.rectangle((0, divider_y + 16, 1440, nav_top - 16), fill="#FFFFFF")

# Top search area inner divider (thin)
draw.line((search_left + 8, search_top + 144 + 2, search_right - 8, search_top + 144 + 2), fill="#F0F0F0", width=1)

# Small rounded card behind "Suggestions" region header area (very subtle)
suggestion_card_top = large_divider_y + 40
suggestion_card_bottom = suggestion_card_top + 340
draw.rounded_rectangle(
    (32, suggestion_card_top, 1408, suggestion_card_bottom),
    radius=12,
    fill="#FFFFFF",
    outline=None
)

# A faint left margin guideline (non-intrusive) to visually align list content areas
draw.line((32, status_h + 100, 32, nav_top - 20), fill="#FFFFFF", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 47, 70)
    canvas.paste(_c0, (1153, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1153, 0, 1200, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/01_icon_8.32_Wy.png
try:
    _c1 = get_crop(1, 168, 144)
    canvas.paste(_c1, (48, 120), _c1)
except Exception:
    pass
layout["8.32_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/02_icon_Tracking.png
try:
    _c2 = get_crop(2, 288, 168)
    canvas.paste(_c2, (864, 2792), _c2)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/03_icon_Shin_Lim.png
try:
    _c3 = get_crop(3, 1440, 168)
    canvas.paste(_c3, (0, 639), _c3)
except Exception:
    pass
layout["Shin_Lim"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/04_icon_Browse.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (0, 2792), _c4)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/05_icon_Just_Announced_by_My_Performers.png
try:
    _c5 = get_crop(5, 1440, 168)
    canvas.paste(_c5, (0, 1688), _c5)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 63, 64)
    canvas.paste(_c6, (242, 2), _c6)
except Exception:
    pass
layout["icon_6"] = [242, 2, 305, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/07_icon_Tickets.png
try:
    _c7 = get_crop(7, 288, 168)
    canvas.paste(_c7, (576, 2792), _c7)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/08_icon_The_Fonda_Theatre.png
try:
    _c8 = get_crop(8, 1440, 168)
    canvas.paste(_c8, (0, 975), _c8)
except Exception:
    pass
layout["The_Fonda_Theatre"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/09_icon_Dallas_Mavericks.png
try:
    _c9 = get_crop(9, 1440, 168)
    canvas.paste(_c9, (0, 807), _c9)
except Exception:
    pass
layout["Dallas_Mavericks"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 96, 69)
    canvas.paste(_c10, (1216, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1216, 0, 1312, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/11_icon_WWE.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 807), _c11)
except Exception:
    pass
layout["WWE"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/12_icon_Oracle_Arena.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 471), _c12)
except Exception:
    pass
layout["Oracle_Arena"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/13_icon_Clear.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 120), _c13)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/14_icon_Events_by_My_Performers.png
try:
    _c14 = get_crop(14, 1440, 168)
    canvas.paste(_c14, (0, 1520), _c14)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 52, 69)
    canvas.paste(_c15, (1319, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1319, 0, 1371, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/16_icon_Dallas_Mavericks.png
try:
    _c16 = get_crop(16, 1440, 168)
    canvas.paste(_c16, (0, 639), _c16)
except Exception:
    pass
layout["Dallas_Mavericks"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/17_icon_8.32_Wy.png
try:
    _c17 = get_crop(17, 47, 64)
    canvas.paste(_c17, (186, 1), _c17)
except Exception:
    pass
layout["8.32_Wy"] = [186, 1, 233, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/18_icon_Account.png
try:
    _c18 = get_crop(18, 288, 168)
    canvas.paste(_c18, (1152, 2792), _c18)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 59, 64)
    canvas.paste(_c19, (313, 2), _c19)
except Exception:
    pass
layout["icon_19"] = [313, 2, 372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/20_icon_WWE.png
try:
    _c20 = get_crop(20, 1440, 168)
    canvas.paste(_c20, (0, 975), _c20)
except Exception:
    pass
layout["WWE"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/21_icon_Search.png
try:
    _c21 = get_crop(21, 288, 162)
    canvas.paste(_c21, (288, 2792), _c21)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/22_icon_The_Fonda_Theatre.png
try:
    _c22 = get_crop(22, 1440, 168)
    canvas.paste(_c22, (0, 1143), _c22)
except Exception:
    pass
layout["The_Fonda_Theatre"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/23_icon_Performer_event_or_venue.png
try:
    _c23 = get_crop(23, 1032, 144)
    canvas.paste(_c23, (216, 120), _c23)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/24_icon_Search.png
try:
    _c24 = get_crop(24, 288, 162)
    canvas.paste(_c24, (288, 2792), _c24)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/25_icon_Just_Announced_by_My_Performers.png
try:
    _c25 = get_crop(25, 1440, 168)
    canvas.paste(_c25, (0, 1856), _c25)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/26_text_8.32_Wy.png
try:
    _c26 = get_crop(26, 156, 49)
    canvas.paste(_c26, (16, 12), _c26)
except Exception:
    pass
layout["8.32_Wy"] = [16, 12, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/27_text_Recent_searches.png
try:
    _c27 = get_crop(27, 168, 144)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_03_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-6/28_text_Suggestions.png
try:
    _c28 = get_crop(28, 331, 74)
    canvas.paste(_c28, (40, 1423), _c28)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
