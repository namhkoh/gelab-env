# page_id: page_seatgeek_2d936b7d0af74c20ba6cd92218729838_03
# screenshot: 2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6.png
# step_index: 3/12
# task: Open SeatGeek. Track "Los Angeles Clippers" and "Golden State Warriors".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (ensure canvas is filled white)
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar background (top area)
status_bar_h = 120
draw.rectangle((0, 0, 1440, status_bar_h), fill=(242, 242, 242))

# Search bar background (rounded)
search_x0, search_y0 = 40, 120
search_x1, search_y1 = 1400, 264
draw.rounded_rectangle((search_x0, search_y0, search_x1, search_y1), radius=24, fill=(246, 246, 246), outline=None)

# Subtle separator line under the search bar area
sep_color = (227, 227, 227)
draw.line((40, search_y1 + 20, 1400, search_y1 + 20), fill=sep_color, width=2)

# Divider between "Recent searches" block and next section (approx where list ends)
recent_list_bottom = 1143 + 168  # using detected item positions to calculate an approximate bottom
draw.line((40, recent_list_bottom + 8, 1400, recent_list_bottom + 8), fill=sep_color, width=2)

# Another subtle divider further down before bottom nav area
bottom_nav_top = 2792 - 48
draw.line((20, bottom_nav_top, 1420, bottom_nav_top), fill=(230, 230, 230), width=2)

# Light grouping background for the "Suggestions" region to hint a card area (subtle)
suggestions_top = 1420
suggestions_bottom = suggestions_top + 420
draw.rounded_rectangle((28, suggestions_top - 20, 1412, suggestions_bottom), radius=12, fill=(255, 255, 255), outline=(245, 245, 245))

# Very faint shadow/line under the search bar to lift it visually
draw.line((search_x0 + 6, search_y1 + 8, search_x1 - 6, search_y1 + 8), fill=(240, 240, 240), width=1)

# Thin separators between main sections (one near top content area)
draw.line((40, 320, 1400, 320), fill=(238, 238, 238), width=1)
draw.line((40, 720, 1400, 720), fill=(248, 248, 248), width=1)

# Draw a faint rounded container behind the recent searches area to suggest grouping (very subtle)
recent_top = 360
recent_bottom = recent_list_bottom + 4
draw.rounded_rectangle((28, recent_top - 20, 1412, recent_bottom), radius=14, fill=(255, 255, 255), outline=(245, 245, 245))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/00_icon_The_Book_of_Mormon.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 471), _c0)
except Exception:
    pass
layout["The_Book_of_Mormon"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/01_icon_Boston_Celtics.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 975), _c1)
except Exception:
    pass
layout["Boston_Celtics"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 45, 70)
    canvas.paste(_c2, (1154, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1154, 0, 1199, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/03_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c3 = get_crop(3, 1440, 168)
    canvas.paste(_c3, (0, 639), _c3)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/04_icon_The_Lion_King.png
try:
    _c4 = get_crop(4, 1440, 168)
    canvas.paste(_c4, (0, 807), _c4)
except Exception:
    pass
layout["The_Lion_King"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/05_icon_Suggestions.png
try:
    _c5 = get_crop(5, 1440, 168)
    canvas.paste(_c5, (0, 1143), _c5)
except Exception:
    pass
layout["Suggestions"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/06_icon_6.53_Wy.png
try:
    _c6 = get_crop(6, 168, 144)
    canvas.paste(_c6, (48, 120), _c6)
except Exception:
    pass
layout["6.53_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/07_icon_Tracking.png
try:
    _c7 = get_crop(7, 288, 168)
    canvas.paste(_c7, (864, 2792), _c7)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 65, 61)
    canvas.paste(_c8, (243, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [243, 3, 308, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/09_icon_Browse.png
try:
    _c9 = get_crop(9, 288, 168)
    canvas.paste(_c9, (0, 2792), _c9)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/10_icon_Tickets.png
try:
    _c10 = get_crop(10, 288, 168)
    canvas.paste(_c10, (576, 2792), _c10)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/11_icon_The_Phantom_of_the_Opera.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 975), _c11)
except Exception:
    pass
layout["The_Phantom_of_the_Opera"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/12_icon_Just_Announced_by_My_Performers.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 1688), _c12)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/13_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 807), _c13)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 99, 68)
    canvas.paste(_c14, (1216, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1216, 0, 1315, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/15_icon_Clear.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1248, 120), _c15)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/16_icon_Events_by_My_Performers.png
try:
    _c16 = get_crop(16, 1440, 168)
    canvas.paste(_c16, (0, 1520), _c16)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 54, 59)
    canvas.paste(_c17, (315, 5), _c17)
except Exception:
    pass
layout["icon_17"] = [315, 5, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/18_icon_Account.png
try:
    _c18 = get_crop(18, 288, 168)
    canvas.paste(_c18, (1152, 2792), _c18)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/19_icon_The_Phantom_of_the_Opera.png
try:
    _c19 = get_crop(19, 1440, 168)
    canvas.paste(_c19, (0, 1143), _c19)
except Exception:
    pass
layout["The_Phantom_of_the_Opera"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 45, 66)
    canvas.paste(_c20, (1327, 2), _c20)
except Exception:
    pass
layout["icon_20"] = [1327, 2, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/21_icon_6.53_Wy.png
try:
    _c21 = get_crop(21, 46, 62)
    canvas.paste(_c21, (186, 2), _c21)
except Exception:
    pass
layout["6.53_Wy"] = [186, 2, 232, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/22_icon_Search.png
try:
    _c22 = get_crop(22, 288, 162)
    canvas.paste(_c22, (288, 2792), _c22)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/23_icon_Performer_event_or_venue.png
try:
    _c23 = get_crop(23, 1032, 144)
    canvas.paste(_c23, (216, 120), _c23)
except Exception:
    pass
layout["Performer;_event;_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/24_icon_Search.png
try:
    _c24 = get_crop(24, 288, 162)
    canvas.paste(_c24, (288, 2792), _c24)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/25_icon_Morm.png
try:
    _c25 = get_crop(25, 1440, 168)
    canvas.paste(_c25, (0, 639), _c25)
except Exception:
    pass
layout["Morm"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/26_text_6.53_Wy.png
try:
    _c26 = get_crop(26, 153, 49)
    canvas.paste(_c26, (19, 12), _c26)
except Exception:
    pass
layout["6.53_Wy"] = [19, 12, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/27_text_Recent_searches.png
try:
    _c27 = get_crop(27, 168, 144)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/28_text_Suggestions.png
try:
    _c28 = get_crop(28, 331, 74)
    canvas.paste(_c28, (40, 1423), _c28)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_03_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-6/29_text_Just_Announced_by_My_Performers.png
try:
    _c29 = get_crop(29, 1440, 168)
    canvas.paste(_c29, (0, 1856), _c29)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]
