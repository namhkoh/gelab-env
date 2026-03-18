# page_id: page_seatgeek_6623dffea11a48f2955bafde23a3f1c7_03
# screenshot: 2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6.png
# step_index: 3/9
# task: Open SeatGeek. Search "New York Knicks" and select the second upcoming event, show the location of the event and track the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar background
draw.rectangle([(0, 0), (1440, 120)], fill="#f2f2f2")

# Subtle bottom border for status bar
draw.line([(0, 120), (1440, 120)], fill="#e0e0e0", width=2)

# Search field background (large rounded container behind the input)
search_rect = (48, 72, 1392, 216)
draw.rounded_rectangle(search_rect, radius=28, fill="#fafafa", outline="#f0f0f0", width=1)

# Very faint inner bottom shadow for search container
draw.line([(search_rect[0]+8, search_rect[3]-4), (search_rect[2]-8, search_rect[3]-4)], fill="#f0f0f0", width=1)

# Primary horizontal divider below header / search area
draw.line([(32, 312), (1408, 312)], fill="#e8e8e8", width=2)

# Card background for the "Recent searches" block (a subtle card-shaped area behind rows)
recent_card = (24, 328, 1416, 1168)
draw.rounded_rectangle(recent_card, radius=12, fill="#ffffff", outline="#f5f5f5", width=1)

# Subtle separators to visually group sections (do not draw icons/text)
# Divider between recent searches block and suggestions area
draw.line([(32, 1168), (1408, 1168)], fill="#ececec", width=1)

# Additional divider between suggestion header area and suggestion items
draw.line([(32, 1416), (1408, 1416)], fill="#ececec", width=1)

# Light background card for the Suggestions section to separate from page background
suggestions_card = (24, 1428, 1416, 1848)
draw.rounded_rectangle(suggestions_card, radius=12, fill="#ffffff", outline="#f6f6f6", width=1)

# Subtle horizontal separators inside suggestions area (spaced like list rows; not drawing icons/text)
for y in (1560, 1692, 1824):
    draw.line([(72, y), (1368, y)], fill="#f0f0f0", width=1)

# Large whitespace/content area is intentionally left plain (main background is white)
# But add a very faint vertical gradient band to suggest depth (top-to-bottom subtle overlay)
# (Drawn as semi-opaque rectangles stacked to avoid imports)
for i in range(0, 200, 20):
    alpha_shade = int(4 + i * 0.05)  # tiny incremental shading
    shade_color = (250 - alpha_shade, 250 - alpha_shade, 250 - alpha_shade)
    draw.rectangle([(0, 1848 + i), (1440, 1848 + i + 20)], fill=shade_color)

# Top border for bottom navigation area (thin separator)
nav_top = 2768
draw.line([(0, nav_top), (1440, nav_top)], fill="#e8e8e8", width=2)

# Bottom navigation background area (keeps it visually separate)
draw.rectangle([(0, 2792), (1440, 2960)], fill="#ffffff")

# Slight inner top shadow for nav to give it elevation
draw.line([(0, 2792), (1440, 2792)], fill="#f2f2f2", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/00_icon_Golden_State_Warriors.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 471), _c0)
except Exception:
    pass
layout["Golden_State_Warriors"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/01_icon_Mormi.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 639), _c1)
except Exception:
    pass
layout["Mormi"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/02_icon_Suggestions.png
try:
    _c2 = get_crop(2, 1440, 168)
    canvas.paste(_c2, (0, 1143), _c2)
except Exception:
    pass
layout["Suggestions"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/03_icon_Recent_searches.png
try:
    _c3 = get_crop(3, 1440, 168)
    canvas.paste(_c3, (0, 471), _c3)
except Exception:
    pass
layout["Recent_searches"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 45, 70)
    canvas.paste(_c4, (1154, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1154, 0, 1199, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/05_icon_Just_Announced_by_My_Performers.png
try:
    _c5 = get_crop(5, 1440, 168)
    canvas.paste(_c5, (0, 1688), _c5)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/06_icon_Tracking.png
try:
    _c6 = get_crop(6, 288, 168)
    canvas.paste(_c6, (864, 2792), _c6)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/07_icon_Tickets.png
try:
    _c7 = get_crop(7, 288, 168)
    canvas.paste(_c7, (576, 2792), _c7)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/08_icon_6.57_Wy.png
try:
    _c8 = get_crop(8, 168, 144)
    canvas.paste(_c8, (48, 120), _c8)
except Exception:
    pass
layout["6.57_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 64, 61)
    canvas.paste(_c9, (243, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [243, 3, 307, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/10_icon_Browse.png
try:
    _c10 = get_crop(10, 288, 168)
    canvas.paste(_c10, (0, 2792), _c10)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/11_icon_The_Book_f_Mormon.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 807), _c11)
except Exception:
    pass
layout["The_Book_f_Mormon"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/12_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 975), _c12)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 94, 69)
    canvas.paste(_c13, (1218, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1218, 0, 1312, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/14_icon_Events_by_My_Performers.png
try:
    _c14 = get_crop(14, 1440, 168)
    canvas.paste(_c14, (0, 1520), _c14)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/15_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 1143), _c15)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/16_icon_Los_Angeles_Clippers.png
try:
    _c16 = get_crop(16, 1440, 168)
    canvas.paste(_c16, (0, 639), _c16)
except Exception:
    pass
layout["Los_Angeles_Clippers"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/17_icon_Clear.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 120), _c17)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/18_icon_The_Lion_King.png
try:
    _c18 = get_crop(18, 1440, 168)
    canvas.paste(_c18, (0, 975), _c18)
except Exception:
    pass
layout["The_Lion_King"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 45, 66)
    canvas.paste(_c19, (1327, 2), _c19)
except Exception:
    pass
layout["icon_19"] = [1327, 2, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/20_icon_Account.png
try:
    _c20 = get_crop(20, 288, 168)
    canvas.paste(_c20, (1152, 2792), _c20)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/21_icon_6.57_Wy.png
try:
    _c21 = get_crop(21, 46, 63)
    canvas.paste(_c21, (185, 2), _c21)
except Exception:
    pass
layout["6.57_Wy"] = [185, 2, 231, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/22_icon_Search.png
try:
    _c22 = get_crop(22, 288, 162)
    canvas.paste(_c22, (288, 2792), _c22)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/23_icon_Just_Announced_by_My_Performers.png
try:
    _c23 = get_crop(23, 1440, 168)
    canvas.paste(_c23, (0, 1856), _c23)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/24_icon_Search.png
try:
    _c24 = get_crop(24, 288, 162)
    canvas.paste(_c24, (288, 2792), _c24)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 55, 61)
    canvas.paste(_c25, (315, 4), _c25)
except Exception:
    pass
layout["icon_25"] = [315, 4, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/26_icon_Performer_event_or_venue.png
try:
    _c26 = get_crop(26, 1032, 144)
    canvas.paste(_c26, (216, 120), _c26)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/27_text_6.57_Wy.png
try:
    _c27 = get_crop(27, 153, 49)
    canvas.paste(_c27, (19, 12), _c27)
except Exception:
    pass
layout["6.57_Wy"] = [19, 12, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/28_text_Recent_searches.png
try:
    _c28 = get_crop(28, 168, 144)
    canvas.paste(_c28, (48, 120), _c28)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_03_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-6/29_text_Suggestions.png
try:
    _c29 = get_crop(29, 331, 74)
    canvas.paste(_c29, (40, 1423), _c29)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
