# page_id: page_seatgeek_2c6b8c5734894f77ba798a927b118406_02
# screenshot: 2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5.png
# step_index: 2/5
# task: Open SeatGeek. Search "Wembley Stadium". Show the next five football matches. Add to watch list.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
draw.rectangle([(0, 0), (1440, 68)], fill="#EFEFEF")

# Main background (dominant color - white)
draw.rectangle([(0, 68), (1440, 2960)], fill="#FFFFFF")

# Search box (rounded) - background and subtle border/shadow
search_left, search_top = 48, 48
search_right, search_bottom = 1392, 192
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=28,
    fill="#F6F6F6",
    outline="#E6E6E6",
    width=2
)
# subtle inner shadow line under search box
draw.line([(search_left + 8, search_bottom + 8), (search_right - 8, search_bottom + 8)], fill="#F0F0F0", width=1)

# Divider under search area (full-width with side padding)
divider_y = search_bottom + 28
draw.line([(40, divider_y), (1400, divider_y)], fill="#E9E9E9", width=1)

# Section divider between Recent Searches and Suggestions
# Approximate positions inferred from UI; keep as subtle separators
draw.line([(40, 1520), (1400, 1520)], fill="#EAEAEA", width=1)

# Additional subtle separator above the list of recent searches
draw.line([(40, 360), (1400, 360)], fill="#F3F3F3", width=1)

# Light grouping card behind the "Recent searches" area (very subtle, white-on-white elevation)
recent_card_top = 220
recent_card_bottom = 1520
draw.rounded_rectangle(
    [(28, recent_card_top), (1412, recent_card_bottom)],
    radius=12,
    fill="#FFFFFF",
    outline="#F2F2F2",
    width=1
)

# Light grouping card behind the "Suggestions" area
suggest_card_top = 1528
suggest_card_bottom = 2600
draw.rounded_rectangle(
    [(28, suggest_card_top), (1412, suggest_card_bottom)],
    radius=12,
    fill="#FFFFFF",
    outline="#F7F7F7",
    width=1
)

# Bottom navigation bar background and top border/shadow
nav_top = 2792
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#FFFFFF")
# subtle top divider / shadow
draw.line([(0, nav_top), (1440, nav_top)], fill="#E8E8E8", width=2)
# faint shadow gradient band (two lines to imply elevation)
draw.line([(0, nav_top + 2), (1440, nav_top + 2)], fill="#F5F5F5", width=1)

# Small subtle vertical padding guides (very light - non-intrusive)
for y in (120, 280, 520, 760, 1000, 1240):
    draw.line([(40, y), (60, y)], fill="#F6F6F6", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/00_icon_New_York_Knicks.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 639), _c0)
except Exception:
    pass
layout["New_York_Knicks"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/01_icon_Morm.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 807), _c1)
except Exception:
    pass
layout["Morm"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 49, 69)
    canvas.paste(_c2, (1152, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1152, 0, 1201, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 64, 64)
    canvas.paste(_c3, (242, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [242, 2, 306, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/04_icon_Tracking.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (864, 2792), _c4)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/05_icon_Morm.png
try:
    _c5 = get_crop(5, 1440, 168)
    canvas.paste(_c5, (0, 975), _c5)
except Exception:
    pass
layout["Morm"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/06_icon_Just_Announced_by_My_Performers.png
try:
    _c6 = get_crop(6, 1440, 168)
    canvas.paste(_c6, (0, 1688), _c6)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 98, 69)
    canvas.paste(_c7, (1215, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1215, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/08_icon_Browse.png
try:
    _c8 = get_crop(8, 288, 168)
    canvas.paste(_c8, (0, 2792), _c8)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/09_icon_Cryptocom_Arena.png
try:
    _c9 = get_crop(9, 1440, 168)
    canvas.paste(_c9, (0, 471), _c9)
except Exception:
    pass
layout["Cryptocom_Arena"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/10_icon_7.05_my.png
try:
    _c10 = get_crop(10, 168, 144)
    canvas.paste(_c10, (48, 120), _c10)
except Exception:
    pass
layout["7.05_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/11_icon_Tickets.png
try:
    _c11 = get_crop(11, 288, 168)
    canvas.paste(_c11, (576, 2792), _c11)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/12_icon_Clear.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 120), _c12)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/13_icon_Los_Angeles_Clippers.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 975), _c13)
except Exception:
    pass
layout["Los_Angeles_Clippers"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/14_icon_Golden_State_Warriors.png
try:
    _c14 = get_crop(14, 1440, 168)
    canvas.paste(_c14, (0, 807), _c14)
except Exception:
    pass
layout["Golden_State_Warriors"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/15_icon_Golden_State_Warriors.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 639), _c15)
except Exception:
    pass
layout["Golden_State_Warriors"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 52, 68)
    canvas.paste(_c16, (1319, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1319, 0, 1371, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/17_icon_7.05_my.png
try:
    _c17 = get_crop(17, 47, 64)
    canvas.paste(_c17, (186, 1), _c17)
except Exception:
    pass
layout["7.05_my"] = [186, 1, 233, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/18_icon_Events_by_My_Performers.png
try:
    _c18 = get_crop(18, 1440, 168)
    canvas.paste(_c18, (0, 1520), _c18)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 62, 64)
    canvas.paste(_c19, (313, 2), _c19)
except Exception:
    pass
layout["icon_19"] = [313, 2, 375, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/20_icon_Account.png
try:
    _c20 = get_crop(20, 288, 168)
    canvas.paste(_c20, (1152, 2792), _c20)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/21_icon_The_Book_f_Mormon.png
try:
    _c21 = get_crop(21, 1440, 168)
    canvas.paste(_c21, (0, 1143), _c21)
except Exception:
    pass
layout["The_Book_f_Mormon"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/22_icon_7.05_my.png
try:
    _c22 = get_crop(22, 58, 65)
    canvas.paste(_c22, (113, 0), _c22)
except Exception:
    pass
layout["7.05_my"] = [113, 0, 171, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/23_icon_Search.png
try:
    _c23 = get_crop(23, 288, 162)
    canvas.paste(_c23, (288, 2792), _c23)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/24_icon_Just_Announced_by_My_Performers.png
try:
    _c24 = get_crop(24, 1440, 168)
    canvas.paste(_c24, (0, 1856), _c24)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/25_icon_Morm.png
try:
    _c25 = get_crop(25, 1440, 168)
    canvas.paste(_c25, (0, 1143), _c25)
except Exception:
    pass
layout["Morm"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/26_icon_Performer_event_or_venue.png
try:
    _c26 = get_crop(26, 1032, 144)
    canvas.paste(_c26, (216, 120), _c26)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/27_text_Recent_searches.png
try:
    _c27 = get_crop(27, 168, 144)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_02_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-5/28_text_Suggestions.png
try:
    _c28 = get_crop(28, 331, 74)
    canvas.paste(_c28, (40, 1423), _c28)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
