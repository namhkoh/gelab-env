# page_id: page_seatgeek_2d936b7d0af74c20ba6cd92218729838_02
# screenshot: 2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5.png
# step_index: 2/12
# task: Open SeatGeek. Track "Los Angeles Clippers" and "Golden State Warriors".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_h = 72
status_color = (242, 242, 244)  # light grey/status background
draw.rectangle((0, 0, 1440, status_h), fill=status_color)
draw.line((0, status_h-1, 1440, status_h-1), fill=(225, 225, 228), width=1)

# Header background / toolbar area behind the search field (subtle)
header_top = 60
header_bottom = 280
header_bg = (249, 249, 249)
draw.rectangle((0, header_top, 1440, header_bottom), fill=header_bg)

# Light rounded "search bar" background shape (ONLY background, no icons/text)
search_bg_box = (40, 100, 1400, 250)  # larger rounded background for the search area
draw.rounded_rectangle(search_bg_box, radius=20, fill=(255,255,255), outline=(236,236,236), width=1)

# Thin divider line under the header/search area
divider_y = 270
draw.line((40, divider_y, 1400, divider_y), fill=(234,234,234), width=1)

# "Recent searches" card background (subtle card shape behind the list)
recent_card_top = divider_y + 20
recent_card_bottom = 1320
recent_card_box = (30, recent_card_top, 1410, recent_card_bottom)
draw.rounded_rectangle(recent_card_box, radius=12, fill=(255,255,255), outline=(240,240,240), width=1)

# Separator line between recent searches and suggestions
sep_y = 1320
draw.line((30, sep_y, 1410, sep_y), fill=(235,235,235), width=1)

# "Suggestions" card background area
suggestions_top = sep_y + 20
suggestions_bottom = 2240
suggestions_box = (30, suggestions_top, 1410, suggestions_bottom)
draw.rounded_rectangle(suggestions_box, radius=12, fill=(255,255,255), outline=(240,240,240), width=1)

# Subtle section headings background bands (behind the heading areas only)
# Recent searches heading band
draw.rectangle((30, recent_card_top, 1410, recent_card_top+90), fill=(255,255,255))
# Suggestions heading band
draw.rectangle((30, suggestions_top, 1410, suggestions_top+90), fill=(255,255,255))

# Large content area background (keeps page feeling airy; no text/icons drawn)
content_bg_top = suggestions_top + 120
draw.rectangle((0, content_bg_top, 1440, 2790), fill=(255,255,255))

# Bottom navigation bar background and top divider (leave icons area blank)
nav_top = 2792
draw.rectangle((0, nav_top, 1440, 2960), fill=(255,255,255))
draw.line((0, nav_top, 1440, nav_top), fill=(230,230,230), width=1)

# Add a faint drop shadow under header to separate from content
shadow_y_start = divider_y
for i, alpha in enumerate([18,14,10,6], start=0):
    y = shadow_y_start + i
    # semi-manual "shadow" by drawing very light lines
    draw.line((40, y, 1400, y), fill=(0 + alpha, 0 + alpha, 0 + alpha), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/00_icon_Boston_Celtics.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 975), _c0)
except Exception:
    pass
layout["Boston_Celtics"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/01_icon_The_Book_of_Mormon.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 471), _c1)
except Exception:
    pass
layout["The_Book_of_Mormon"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 47, 70)
    canvas.paste(_c2, (1153, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1153, 0, 1200, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/03_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c3 = get_crop(3, 1440, 168)
    canvas.paste(_c3, (0, 639), _c3)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/04_icon_Tracking.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (864, 2792), _c4)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 97, 69)
    canvas.paste(_c5, (1216, 0), _c5)
except Exception:
    pass
layout["icon_5"] = [1216, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 66, 63)
    canvas.paste(_c6, (242, 2), _c6)
except Exception:
    pass
layout["icon_6"] = [242, 2, 308, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/07_icon_Browse.png
try:
    _c7 = get_crop(7, 288, 168)
    canvas.paste(_c7, (0, 2792), _c7)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/08_icon_Tickets.png
try:
    _c8 = get_crop(8, 288, 168)
    canvas.paste(_c8, (576, 2792), _c8)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/09_icon_The_Phantom_of_the_Opera.png
try:
    _c9 = get_crop(9, 1440, 168)
    canvas.paste(_c9, (0, 975), _c9)
except Exception:
    pass
layout["The_Phantom_of_the_Opera"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/10_icon_Suggestions.png
try:
    _c10 = get_crop(10, 1440, 168)
    canvas.paste(_c10, (0, 1143), _c10)
except Exception:
    pass
layout["Suggestions"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/11_icon_The_Lion_King.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 807), _c11)
except Exception:
    pass
layout["The_Lion_King"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/12_icon_Just_Announced_by_My_Performers.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 1688), _c12)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/13_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 807), _c13)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/14_icon_Clear.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 120), _c14)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/15_icon_6.53_Wy.png
try:
    _c15 = get_crop(15, 47, 63)
    canvas.paste(_c15, (186, 1), _c15)
except Exception:
    pass
layout["6.53_Wy"] = [186, 1, 233, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/16_icon_Events_by_My_Performers.png
try:
    _c16 = get_crop(16, 1440, 168)
    canvas.paste(_c16, (0, 1520), _c16)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/17_icon_Account.png
try:
    _c17 = get_crop(17, 288, 168)
    canvas.paste(_c17, (1152, 2792), _c17)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/18_icon_6.53_Wy.png
try:
    _c18 = get_crop(18, 168, 144)
    canvas.paste(_c18, (48, 120), _c18)
except Exception:
    pass
layout["6.53_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 52, 69)
    canvas.paste(_c19, (1319, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1319, 0, 1371, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/20_icon_6.53_Wy.png
try:
    _c20 = get_crop(20, 57, 65)
    canvas.paste(_c20, (113, 0), _c20)
except Exception:
    pass
layout["6.53_Wy"] = [113, 0, 170, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/21_icon_The_Phantom_of_the_Opera.png
try:
    _c21 = get_crop(21, 1440, 168)
    canvas.paste(_c21, (0, 1143), _c21)
except Exception:
    pass
layout["The_Phantom_of_the_Opera"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/22_icon_Morm.png
try:
    _c22 = get_crop(22, 1440, 168)
    canvas.paste(_c22, (0, 639), _c22)
except Exception:
    pass
layout["Morm"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/23_icon_Search.png
try:
    _c23 = get_crop(23, 288, 162)
    canvas.paste(_c23, (288, 2792), _c23)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/24_icon_Performer_event_or_venue.png
try:
    _c24 = get_crop(24, 1032, 144)
    canvas.paste(_c24, (216, 120), _c24)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/25_icon_Just_Announced_by_My_Performers.png
try:
    _c25 = get_crop(25, 1440, 168)
    canvas.paste(_c25, (0, 1856), _c25)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 45, 52)
    canvas.paste(_c26, (385, 9), _c26)
except Exception:
    pass
layout["icon_26"] = [385, 9, 430, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/27_icon_Morm.png
try:
    _c27 = get_crop(27, 1440, 168)
    canvas.paste(_c27, (0, 471), _c27)
except Exception:
    pass
layout["Morm"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/28_text_Recent_searches.png
try:
    _c28 = get_crop(28, 168, 144)
    canvas.paste(_c28, (48, 120), _c28)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_02_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-5/29_text_Suggestions.png
try:
    _c29 = get_crop(29, 331, 74)
    canvas.paste(_c29, (40, 1423), _c29)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
