# page_id: page_seatgeek_2c6b8c5734894f77ba798a927b118406_03
# screenshot: 2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6.png
# step_index: 3/5
# task: Open SeatGeek. Search "Wembley Stadium". Show the next five football matches. Add to watch list.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# background fill (slightly warm white like the app)
draw.rectangle([0, 0, 1440, 2960], fill=(250, 250, 250))

# status bar area at top (~70px)
status_h = 80
draw.rectangle([0, 0, 1440, status_h], fill=(242, 242, 242))

# subtle bottom shadow in status area to separate from content
draw.line([(0, status_h), (1440, status_h)], fill=(230, 230, 230), width=1)

# Search bar rounded background (do NOT draw icons/text - only background)
search_x0, search_y0 = 48, 120
search_x1, search_y1 = 1392, 264  # height ~144
search_radius = 28
draw.rounded_rectangle([search_x0, search_y0, search_x1, search_y1],
                       radius=search_radius,
                       fill=(247, 247, 248),
                       outline=(226, 226, 227),
                       width=1)

# a subtle divider under the search area to separate sections
divider_y = search_y1 + 24
draw.line([(32, divider_y), (1408, divider_y)], fill=(230, 230, 230), width=1)

# Separator under "Recent searches" list (approx bottom of the list)
# Based on detected list item positions the bottom of that block is around y=1311
recent_list_bottom = 1311
draw.line([(24, recent_list_bottom), (1416, recent_list_bottom)], fill=(230, 230, 230), width=1)

# Light grouping background for suggestions area (subtle, slightly different tint)
# This is a background behind the suggestion section (rounded to look like a card region)
suggestion_card_x0, suggestion_card_y0 = 24, 1360
suggestion_card_x1, suggestion_card_y1 = 1416, 1980
draw.rounded_rectangle([suggestion_card_x0, suggestion_card_y0, suggestion_card_x1, suggestion_card_y1],
                       radius=16,
                       fill=(250, 250, 250),
                       outline=None)

# subtle horizontal separators between suggestion items (but not icons/text)
# positions derived from typical row spacing; these are faint lines only
suggestion_row1 = 1520 + 168  # between rows in suggestions block
suggestion_row2 = 1688 + 168
# Clamp to card region to avoid drawing over unrelated areas
if suggestion_row1 < suggestion_card_y1:
    draw.line([(40, suggestion_row1), (1400, suggestion_row1)], fill=(240, 240, 240), width=1)
if suggestion_row2 < suggestion_card_y1:
    draw.line([(40, suggestion_row2), (1400, suggestion_row2)], fill=(240, 240, 240), width=1)

# Bottom navigation bar background and top shadow (do not draw icons)
nav_top = 2792
draw.rectangle([0, nav_top, 1440, 2960], fill=(255, 255, 255))
# top shadow line to separate nav bar
draw.rectangle([0, nav_top - 3, 1440, nav_top], fill=(235, 235, 235))

# final subtle vertical edges to match app's soft card look
draw.line([(48, search_y0), (48, suggestion_card_y1)], fill=(250, 250, 250), width=1)
draw.line([(1392, search_y0), (1392, suggestion_card_y1)], fill=(250, 250, 250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/00_icon_Morm.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 807), _c0)
except Exception:
    pass
layout["Morm"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/01_icon_New_York_Knicks.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 639), _c1)
except Exception:
    pass
layout["New_York_Knicks"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/02_icon_Cryptocom_Arena.png
try:
    _c2 = get_crop(2, 1440, 168)
    canvas.paste(_c2, (0, 471), _c2)
except Exception:
    pass
layout["Cryptocom_Arena"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 47, 70)
    canvas.paste(_c3, (1153, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1153, 0, 1200, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/04_icon_7.05_my.png
try:
    _c4 = get_crop(4, 168, 144)
    canvas.paste(_c4, (48, 120), _c4)
except Exception:
    pass
layout["7.05_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/05_icon_Tracking.png
try:
    _c5 = get_crop(5, 288, 168)
    canvas.paste(_c5, (864, 2792), _c5)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/06_icon_Morm.png
try:
    _c6 = get_crop(6, 1440, 168)
    canvas.paste(_c6, (0, 975), _c6)
except Exception:
    pass
layout["Morm"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/07_icon_Browse.png
try:
    _c7 = get_crop(7, 288, 168)
    canvas.paste(_c7, (0, 2792), _c7)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/08_icon_Just_Announced_by_My_Performers.png
try:
    _c8 = get_crop(8, 1440, 168)
    canvas.paste(_c8, (0, 1688), _c8)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 61, 64)
    canvas.paste(_c9, (243, 2), _c9)
except Exception:
    pass
layout["icon_9"] = [243, 2, 304, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/10_icon_Tickets.png
try:
    _c10 = get_crop(10, 288, 168)
    canvas.paste(_c10, (576, 2792), _c10)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 96, 69)
    canvas.paste(_c11, (1216, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1216, 0, 1312, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/12_icon_Los_Angeles_Clippers.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 975), _c12)
except Exception:
    pass
layout["Los_Angeles_Clippers"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/13_icon_Golden_State_Warriors.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 807), _c13)
except Exception:
    pass
layout["Golden_State_Warriors"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/14_icon_Clear.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 120), _c14)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/15_icon_Golden_State_Warriors.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 639), _c15)
except Exception:
    pass
layout["Golden_State_Warriors"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/16_icon_Events_by_My_Performers.png
try:
    _c16 = get_crop(16, 1440, 168)
    canvas.paste(_c16, (0, 1520), _c16)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 53, 68)
    canvas.paste(_c17, (1319, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1319, 0, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/18_icon_Account.png
try:
    _c18 = get_crop(18, 288, 168)
    canvas.paste(_c18, (1152, 2792), _c18)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 59, 64)
    canvas.paste(_c19, (313, 2), _c19)
except Exception:
    pass
layout["icon_19"] = [313, 2, 372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/20_icon_7.05_my.png
try:
    _c20 = get_crop(20, 47, 64)
    canvas.paste(_c20, (186, 1), _c20)
except Exception:
    pass
layout["7.05_my"] = [186, 1, 233, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/21_icon_Search.png
try:
    _c21 = get_crop(21, 288, 162)
    canvas.paste(_c21, (288, 2792), _c21)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/22_icon_The_Book_f_Mormon.png
try:
    _c22 = get_crop(22, 1440, 168)
    canvas.paste(_c22, (0, 1143), _c22)
except Exception:
    pass
layout["The_Book_f_Mormon"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/23_icon_Performer_event_or_venue.png
try:
    _c23 = get_crop(23, 1032, 144)
    canvas.paste(_c23, (216, 120), _c23)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/24_icon_Search.png
try:
    _c24 = get_crop(24, 288, 162)
    canvas.paste(_c24, (288, 2792), _c24)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/25_icon_Just_Announced_by_My_Performers.png
try:
    _c25 = get_crop(25, 1440, 168)
    canvas.paste(_c25, (0, 1856), _c25)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/26_text_7.05_my.png
try:
    _c26 = get_crop(26, 151, 52)
    canvas.paste(_c26, (21, 9), _c26)
except Exception:
    pass
layout["7.05_my"] = [21, 9, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/27_text_Recent_searches.png
try:
    _c27 = get_crop(27, 168, 144)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_03_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-6/28_text_Suggestions.png
try:
    _c28 = get_crop(28, 331, 74)
    canvas.paste(_c28, (40, 1423), _c28)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
