# page_id: page_seatgeek_2d936b7d0af74c20ba6cd92218729838_09
# screenshot: 2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12.png
# step_index: 9/12
# task: Open SeatGeek. Track "Los Angeles Clippers" and "Golden State Warriors".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the UI page (using provided canvas and draw)

# Colors
status_bar_color = (238, 238, 238)    # light grey status bar
search_bg = (247, 247, 247)           # very light grey for search field background
divider_color = (225, 225, 225)       # subtle divider lines
nav_border = (230, 230, 230)          # top border of bottom nav
shadow_color = (245, 245, 245)        # faint shadow areas

# Canvas size (given)
W, H = 1440, 2960

# 1) Status bar area at top (~56px)
draw.rectangle([(0, 0), (W, 56)], fill=status_bar_color)

# 2) Search bar / header background (rounded rect)
# Position chosen to match screenshot spacing: left/right margins ~48, height ~144, y ~96..240
search_left = 48
search_top = 96
search_right = W - 48
search_bottom = search_top + 144
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=18,
    fill=search_bg,
    outline=(235, 235, 235)
)

# 3) Thin divider below the search area
divider_y_1 = search_bottom + 40
draw.line([(40, divider_y_1), (W-40, divider_y_1)], fill=divider_color, width=1)

# 4) Horizontal separator between "Recent searches" block and "Suggestions" block
# Placed to align roughly with where recent items end and suggestions begin in screenshot
divider_y_2 = 1200
draw.line([(24, divider_y_2), (W-24, divider_y_2)], fill=divider_color, width=2)

# 5) Subtle top shadow for the suggestions area (very light band)
draw.rectangle([(0, divider_y_2-8), (W, divider_y_2)], fill=shadow_color)

# 6) Content area background bands (subtle breaks for large white areas)
# A faint band behind the "Recent searches" list to gently separate it from page background
recent_band_top = divider_y_1 + 24
recent_band_bottom = divider_y_2 - 24
draw.rectangle([(0, recent_band_top), (W, recent_band_bottom)], fill=(255,255,255))

# 7) Bottom navigation area background and top border
nav_top = 2792
draw.rectangle([(0, nav_top), (W, H)], fill=(255, 255, 255))
draw.line([(0, nav_top), (W, nav_top)], fill=nav_border, width=2)

# 8) Very faint shadow above bottom nav to lift it off the page
draw.rectangle([(0, nav_top-12), (W, nav_top)], fill=(250,250,250))

# 9) Optional subtle left content margin guideline (visual structure only, very faint)
# This helps separate circles/avatars from text area without drawing icons/text.
# Use a very faint vertical guide line (kept extremely subtle so it reads as structure)
draw.line([(40, search_bottom + 20), (40, nav_top - 20)], fill=(250,250,250), width=1)

# End of structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/00_icon_Suggestions.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 1143), _c0)
except Exception:
    pass
layout["Suggestions"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/01_icon_Morm.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 471), _c1)
except Exception:
    pass
layout["Morm"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/02_icon_The_Lion_King.png
try:
    _c2 = get_crop(2, 1440, 168)
    canvas.paste(_c2, (0, 975), _c2)
except Exception:
    pass
layout["The_Lion_King"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 45, 71)
    canvas.paste(_c3, (1154, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1154, 0, 1199, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/04_icon_Los_Angeles_Clippers.png
try:
    _c4 = get_crop(4, 1440, 168)
    canvas.paste(_c4, (0, 471), _c4)
except Exception:
    pass
layout["Los_Angeles_Clippers"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/05_icon_Tracking.png
try:
    _c5 = get_crop(5, 288, 168)
    canvas.paste(_c5, (864, 2792), _c5)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/06_icon_6.54_Wy.png
try:
    _c6 = get_crop(6, 168, 144)
    canvas.paste(_c6, (48, 120), _c6)
except Exception:
    pass
layout["6.54_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/07_icon_Tickets.png
try:
    _c7 = get_crop(7, 288, 168)
    canvas.paste(_c7, (576, 2792), _c7)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/08_icon_Browse.png
try:
    _c8 = get_crop(8, 288, 168)
    canvas.paste(_c8, (0, 2792), _c8)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 64, 61)
    canvas.paste(_c9, (243, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [243, 3, 307, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/10_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c10 = get_crop(10, 1440, 168)
    canvas.paste(_c10, (0, 975), _c10)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/11_icon_Just_Announced_by_My_Performers.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 1688), _c11)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/12_icon_The_Book_of_Mormon.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 639), _c12)
except Exception:
    pass
layout["The_Book_of_Mormon"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 94, 69)
    canvas.paste(_c13, (1218, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1218, 0, 1312, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/14_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c14 = get_crop(14, 1440, 168)
    canvas.paste(_c14, (0, 807), _c14)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/15_icon_Events_by_My_Performers.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 1520), _c15)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/16_icon_Clear.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 120), _c16)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/17_icon_The_Phantom_of_the_Opera.png
try:
    _c17 = get_crop(17, 1440, 168)
    canvas.paste(_c17, (0, 1143), _c17)
except Exception:
    pass
layout["The_Phantom_of_the_Opera"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/18_icon_Account.png
try:
    _c18 = get_crop(18, 288, 168)
    canvas.paste(_c18, (1152, 2792), _c18)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 45, 66)
    canvas.paste(_c19, (1327, 2), _c19)
except Exception:
    pass
layout["icon_19"] = [1327, 2, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/20_icon_6.54_Wy.png
try:
    _c20 = get_crop(20, 46, 64)
    canvas.paste(_c20, (185, 1), _c20)
except Exception:
    pass
layout["6.54_Wy"] = [185, 1, 231, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/21_icon_Search.png
try:
    _c21 = get_crop(21, 288, 162)
    canvas.paste(_c21, (288, 2792), _c21)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/22_icon_Morm.png
try:
    _c22 = get_crop(22, 1440, 168)
    canvas.paste(_c22, (0, 807), _c22)
except Exception:
    pass
layout["Morm"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 54, 60)
    canvas.paste(_c23, (315, 4), _c23)
except Exception:
    pass
layout["icon_23"] = [315, 4, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/24_icon_Search.png
try:
    _c24 = get_crop(24, 288, 162)
    canvas.paste(_c24, (288, 2792), _c24)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/25_icon_Performer_event_or_venue.png
try:
    _c25 = get_crop(25, 1032, 144)
    canvas.paste(_c25, (216, 120), _c25)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/26_text_6.54_Wy.png
try:
    _c26 = get_crop(26, 153, 49)
    canvas.paste(_c26, (19, 12), _c26)
except Exception:
    pass
layout["6.54_Wy"] = [19, 12, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/27_text_Recent_searches.png
try:
    _c27 = get_crop(27, 168, 144)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/28_text_Suggestions.png
try:
    _c28 = get_crop(28, 331, 74)
    canvas.paste(_c28, (40, 1423), _c28)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_09_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-12/29_text_Just_Announced_by_My_Performers.png
try:
    _c29 = get_crop(29, 1440, 168)
    canvas.paste(_c29, (0, 1856), _c29)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]
