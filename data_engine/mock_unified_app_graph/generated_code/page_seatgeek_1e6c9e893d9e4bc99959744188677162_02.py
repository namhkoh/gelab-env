# page_id: page_seatgeek_1e6c9e893d9e4bc99959744188677162_02
# screenshot: 2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5.png
# step_index: 2/8
# task: Open SeatGeek. Search "Radio City Music Hall" and then add the venue to favorite. Who are the performers of the top recommended event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background with a very light off-white
draw.rectangle([(0, 0), canvas.size], fill="#FAFAFB")

# Status bar area (top)
status_bar_h = 88
draw.rectangle([(0, 0), (canvas.size[0], status_bar_h)], fill="#F2F3F4")

# Subtle bottom edge for status bar to separate from header/search area
draw.line([(0, status_bar_h), (canvas.size[0], status_bar_h)], fill="#E6E7E8", width=1)

# Search bar (rounded) - do not draw any icons or text inside it
search_left = 40
search_top = 60
search_right = canvas.size[0] - 40
search_bottom = 220
search_radius = 28
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=search_radius,
    fill="#F6F6F6",
    outline="#E7E7E7",
    width=1
)

# A subtle shadow / separator under the search bar
draw.line([(search_left + 8, search_bottom + 8), (search_right - 8, search_bottom + 8)], fill="#EFEFEF", width=1)

# Card background panel for the "Recent searches" list (elevated white card)
card_left = 20
card_top = search_bottom + 40
card_right = canvas.size[0] - 20
card_bottom = 1180
card_radius = 12

# Light shadow for the card (a soft rectangle offset behind)
shadow_offset = 6
draw.rounded_rectangle(
    [(card_left + shadow_offset, card_top + shadow_offset), (card_right + shadow_offset, card_bottom + shadow_offset)],
    radius=card_radius + 2,
    fill="#F3F3F3"
)

# Card itself
draw.rounded_rectangle(
    [(card_left, card_top), (card_right, card_bottom)],
    radius=card_radius,
    fill="#FFFFFF",
    outline=None
)

# Thin full-width divider lines to separate sections
# Divider under the initial area (just below the top items region)
draw.line([(20, card_top + 420), (canvas.size[0] - 20, card_top + 420)], fill="#E9E9E9", width=1)

# Divider between recent searches block and suggestions area
divider_y = card_bottom + 40
draw.line([(20, divider_y), (canvas.size[0] - 20, divider_y)], fill="#E9E9E9", width=1)

# Suggestions area background (keeps page visually ordered) - subtle white band
suggestions_top = divider_y + 24
suggestions_bottom = suggestions_top + 520
draw.rectangle([(0, suggestions_top), (canvas.size[0], suggestions_bottom)], fill="#FFFFFF")

# Subtle separators for suggestion rows (do not draw icons or text)
# Based on expected suggestion item vertical positions; these are just faint lines
suggest_item_x0 = 40
suggest_item_x1 = canvas.size[0] - 40
suggest_row_ys = [suggestions_top + 110, suggestions_top + 230, suggestions_top + 350]
for y in suggest_row_ys:
    draw.line([(suggest_item_x0, y), (suggest_item_x1, y)], fill="#F0F0F0", width=1)

# Bottom navigation bar background and top divider/shadow
nav_top = 2792
nav_bottom = canvas.size[1]
draw.rectangle([(0, nav_top), (canvas.size[0], nav_bottom)], fill="#FFFFFF")
draw.line([(0, nav_top), (canvas.size[0], nav_top)], fill="#EDEDED", width=1)

# Very subtle outer edges to frame the layout (optional, thin)
draw.line([(0, 0), (canvas.size[0], 0)], fill="#EDEDED", width=1)
draw.line([(0, canvas.size[1]-1), (canvas.size[0], canvas.size[1]-1)], fill="#EDEDED", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/00_icon_Recent_searches.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 471), _c0)
except Exception:
    pass
layout["Recent_searches"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/01_icon_Shin_Lim.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 639), _c1)
except Exception:
    pass
layout["Shin_Lim"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 49, 70)
    canvas.paste(_c2, (1152, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1152, 0, 1201, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 64, 65)
    canvas.paste(_c3, (242, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [242, 2, 306, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/04_icon_Tracking.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (864, 2792), _c4)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 98, 69)
    canvas.paste(_c5, (1215, 0), _c5)
except Exception:
    pass
layout["icon_5"] = [1215, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/06_icon_The_Fonda_Theatre.png
try:
    _c6 = get_crop(6, 1440, 168)
    canvas.paste(_c6, (0, 975), _c6)
except Exception:
    pass
layout["The_Fonda_Theatre"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/07_icon_Browse.png
try:
    _c7 = get_crop(7, 288, 168)
    canvas.paste(_c7, (0, 2792), _c7)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/08_icon_8.32_my.png
try:
    _c8 = get_crop(8, 168, 144)
    canvas.paste(_c8, (48, 120), _c8)
except Exception:
    pass
layout["8.32_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/09_icon_Dallas_Mavericks.png
try:
    _c9 = get_crop(9, 1440, 168)
    canvas.paste(_c9, (0, 807), _c9)
except Exception:
    pass
layout["Dallas_Mavericks"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/10_icon_Just_Announced_by_My_Performers.png
try:
    _c10 = get_crop(10, 1440, 168)
    canvas.paste(_c10, (0, 1688), _c10)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/11_icon_Tickets.png
try:
    _c11 = get_crop(11, 288, 168)
    canvas.paste(_c11, (576, 2792), _c11)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/12_icon_Clear.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 120), _c12)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/13_icon_WWE.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 807), _c13)
except Exception:
    pass
layout["WWE"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/14_icon_8.32_my.png
try:
    _c14 = get_crop(14, 48, 64)
    canvas.paste(_c14, (185, 1), _c14)
except Exception:
    pass
layout["8.32_my"] = [185, 1, 233, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 52, 69)
    canvas.paste(_c15, (1319, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1319, 0, 1371, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/16_icon_Account.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (1152, 2792), _c16)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/17_icon_Events_by_My_Performers.png
try:
    _c17 = get_crop(17, 1440, 168)
    canvas.paste(_c17, (0, 1520), _c17)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 62, 64)
    canvas.paste(_c18, (313, 2), _c18)
except Exception:
    pass
layout["icon_18"] = [313, 2, 375, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/19_icon_Dallas_Mavericks.png
try:
    _c19 = get_crop(19, 1440, 168)
    canvas.paste(_c19, (0, 639), _c19)
except Exception:
    pass
layout["Dallas_Mavericks"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/20_icon_Oracle_Arena.png
try:
    _c20 = get_crop(20, 1440, 168)
    canvas.paste(_c20, (0, 471), _c20)
except Exception:
    pass
layout["Oracle_Arena"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/21_icon_8.32_my.png
try:
    _c21 = get_crop(21, 57, 65)
    canvas.paste(_c21, (113, 0), _c21)
except Exception:
    pass
layout["8.32_my"] = [113, 0, 170, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/22_icon_The_Fonda_Theatre.png
try:
    _c22 = get_crop(22, 1440, 168)
    canvas.paste(_c22, (0, 1143), _c22)
except Exception:
    pass
layout["The_Fonda_Theatre"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/23_icon_WWE.png
try:
    _c23 = get_crop(23, 1440, 168)
    canvas.paste(_c23, (0, 975), _c23)
except Exception:
    pass
layout["WWE"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/24_icon_Search.png
try:
    _c24 = get_crop(24, 288, 162)
    canvas.paste(_c24, (288, 2792), _c24)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/25_icon_Just_Announced_by_My_Performers.png
try:
    _c25 = get_crop(25, 1440, 168)
    canvas.paste(_c25, (0, 1856), _c25)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/26_icon_Performer_event_or_venue.png
try:
    _c26 = get_crop(26, 1032, 144)
    canvas.paste(_c26, (216, 120), _c26)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/27_text_Recent_searches.png
try:
    _c27 = get_crop(27, 168, 144)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_02_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-5/28_text_Suggestions.png
try:
    _c28 = get_crop(28, 331, 74)
    canvas.paste(_c28, (40, 1423), _c28)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
