# page_id: page_seatgeek_2d936b7d0af74c20ba6cd92218729838_10
# screenshot: 2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13.png
# step_index: 10/12
# task: Open SeatGeek. Track "Los Angeles Clippers" and "Golden State Warriors".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background & structure for SeatGeek-like mobile UI (canvas and draw are provided)

# Full canvas background (slightly warm off-white)
draw.rectangle((0, 0, 1440, 2960), fill="#fbfbfb")

# Status bar (top ~70px) - subtle light gray
status_h = 70
draw.rectangle((0, 0, 1440, status_h), fill="#efefef")

# Subtle bottom divider for status bar
draw.line((20, status_h-1, 1420, status_h-1), fill="#e3e3e3", width=1)

# Search bar area - rounded white box with subtle border
search_left, search_top, search_right, search_bottom = 40, 80, 1400, 160
draw.rounded_rectangle(
    (search_left, search_top, search_right, search_bottom),
    radius=28,
    fill="#ffffff",
    outline="#e6e6e6",
    width=2
)

# Soft shadow under search bar (thin)
draw.rectangle((search_left, search_bottom, search_right, search_bottom+6), fill="#f3f3f3")

# Main section separators and group card backgrounds
pad = 20
content_left = pad
content_right = 1440 - pad

# Top results card background
top_results_top = 220
top_results_bottom = 560
draw.rounded_rectangle(
    (content_left, top_results_top, content_right, top_results_bottom),
    radius=12,
    fill="#ffffff",
    outline="#efefef",
    width=1
)
# separator line under heading area inside card
draw.line((content_left+24, top_results_top+84, content_right-24, top_results_top+84), fill="#ececec", width=1)

# Performers card background
performers_top = 640
performers_bottom = 1040
draw.rounded_rectangle(
    (content_left, performers_top, content_right, performers_bottom),
    radius=12,
    fill="#ffffff",
    outline="#efefef",
    width=1
)
# subtle separators for performer items
for y in (performers_top+86, performers_top+172):
    draw.line((content_left+24, y, content_right-24, y), fill="#f0f0f0", width=1)

# Recent searches large card background (multiple items)
recent_top = 1160
recent_bottom = 2160
draw.rounded_rectangle(
    (content_left, recent_top, content_right, recent_bottom),
    radius=12,
    fill="#ffffff",
    outline="#efefef",
    width=1
)
# horizontal separators for list items (approx positions)
for i in range(1,6):
    y = recent_top + 120 * i
    if recent_top < y < recent_bottom - 24:
        draw.line((content_left+24, y, content_right-24, y), fill="#f2f2f2", width=1)

# Suggestions card background
suggestions_top = 2280
suggestions_bottom = 2680
draw.rounded_rectangle(
    (content_left, suggestions_top, content_right, suggestions_bottom),
    radius=12,
    fill="#ffffff",
    outline="#efefef",
    width=1
)
# small separators in suggestions
draw.line((content_left+24, suggestions_top+120, content_right-24, suggestions_top+120), fill="#f2f2f2", width=1)

# Full-width subtle separators between major sections
section_dividers = [
    (search_bottom + 20),          # under search area
    (top_results_bottom + 30),     # after top results
    (performers_bottom + 20),      # after performers
    (recent_bottom + 20),          # after recent searches
    (suggestions_bottom + 20)      # before bottom nav
]
for y in section_dividers:
    draw.line((24, y, 1416, y), fill="#ececec", width=1)

# Bottom navigation bar background (reserved area) and top divider
nav_top = 2792
draw.rectangle((0, nav_top, 1440, 2960), fill="#ffffff")
draw.line((0, nav_top, 1440, nav_top), fill="#e6e6e6", width=1)

# Light card shadows (subtle) under major group cards to give depth
# (drawn as very light strokes)
for rect in [
    (content_left, top_results_top, content_right, top_results_bottom),
    (content_left, performers_top, content_right, performers_bottom),
    (content_left, recent_top, content_right, recent_bottom),
    (content_left, suggestions_top, content_right, suggestions_bottom),
]:
    x0, y0, x1, y1 = rect
    # bottom shadow line
    draw.line((x0+6, y1, x1-6, y1), fill="#f5f5f5", width=2)
    # tiny highlight on top
    draw.line((x0+6, y0, x1-6, y0), fill="#ffffff", width=1)

# Done - structural elements drawn. Icons/text will be layered on top separately.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/00_icon_Performers.png
try:
    _c0 = get_crop(0, 1440, 179)
    canvas.paste(_c0, (0, 859), _c0)
except Exception:
    pass
layout["Performers"] = [0, 859, 1440, 1038]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/01_icon_Suggestions.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 2098), _c1)
except Exception:
    pass
layout["Suggestions"] = [0, 2098, 1440, 2266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/02_icon_The_Lion_King.png
try:
    _c2 = get_crop(2, 1440, 168)
    canvas.paste(_c2, (0, 1930), _c2)
except Exception:
    pass
layout["The_Lion_King"] = [0, 1930, 1440, 2098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/03_icon_Morm.png
try:
    _c3 = get_crop(3, 1440, 168)
    canvas.paste(_c3, (0, 1426), _c3)
except Exception:
    pass
layout["Morm"] = [0, 1426, 1440, 1594]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/04_icon_Top_results.png
try:
    _c4 = get_crop(4, 1440, 179)
    canvas.paste(_c4, (0, 471), _c4)
except Exception:
    pass
layout["Top_results"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/05_icon_No_events.png
try:
    _c5 = get_crop(5, 1440, 179)
    canvas.paste(_c5, (0, 1038), _c5)
except Exception:
    pass
layout["No_events"] = [0, 1038, 1440, 1217]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/06_icon_Morm.png
try:
    _c6 = get_crop(6, 1440, 168)
    canvas.paste(_c6, (0, 1762), _c6)
except Exception:
    pass
layout["Morm"] = [0, 1762, 1440, 1930]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 61, 59)
    canvas.paste(_c7, (244, 4), _c7)
except Exception:
    pass
layout["icon_7"] = [244, 4, 305, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/08_icon_Browse.png
try:
    _c8 = get_crop(8, 288, 168)
    canvas.paste(_c8, (0, 2792), _c8)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 43, 70)
    canvas.paste(_c9, (1155, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1155, 0, 1198, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/10_icon_The_Book_of_Mormon.png
try:
    _c10 = get_crop(10, 1440, 168)
    canvas.paste(_c10, (0, 1594), _c10)
except Exception:
    pass
layout["The_Book_of_Mormon"] = [0, 1594, 1440, 1762]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 96, 67)
    canvas.paste(_c11, (1218, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1218, 0, 1314, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/12_icon_Los_Angeles_Clippers.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 1426), _c12)
except Exception:
    pass
layout["Los_Angeles_Clippers"] = [0, 1426, 1440, 1594]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/13_icon_Events_by_My_Performers.png
try:
    _c13 = get_crop(13, 288, 162)
    canvas.paste(_c13, (288, 2792), _c13)
except Exception:
    pass
layout["Events_by_My_Performers"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/14_icon_6.54_Wy.png
try:
    _c14 = get_crop(14, 46, 60)
    canvas.paste(_c14, (185, 3), _c14)
except Exception:
    pass
layout["6.54_Wy"] = [185, 3, 231, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 46, 57)
    canvas.paste(_c15, (318, 6), _c15)
except Exception:
    pass
layout["icon_15"] = [318, 6, 364, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/16_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c16 = get_crop(16, 1440, 168)
    canvas.paste(_c16, (0, 1930), _c16)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 1930, 1440, 2098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/17_icon_FREE_Guest_List_for_Official_Golden_Stat.png
try:
    _c17 = get_crop(17, 1440, 179)
    canvas.paste(_c17, (0, 1038), _c17)
except Exception:
    pass
layout["FREE_Guest_List_for_Offic"] = [0, 1038, 1440, 1217]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 45, 64)
    canvas.paste(_c18, (1326, 3), _c18)
except Exception:
    pass
layout["icon_18"] = [1326, 3, 1371, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/19_icon_Golden_State_Warriors.png
try:
    _c19 = get_crop(19, 1032, 144)
    canvas.paste(_c19, (216, 120), _c19)
except Exception:
    pass
layout["Golden_State_Warriors]"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/20_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c20 = get_crop(20, 1440, 168)
    canvas.paste(_c20, (0, 1762), _c20)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 1762, 1440, 1930]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/21_icon_Golden_State_Warriors.png
try:
    _c21 = get_crop(21, 1440, 179)
    canvas.paste(_c21, (0, 471), _c21)
except Exception:
    pass
layout["Golden_State_Warriors"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/22_icon_Clear.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1248, 120), _c22)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/23_icon_FREE_Guest_List_for_Official_Golden_Stat.png
try:
    _c23 = get_crop(23, 1440, 179)
    canvas.paste(_c23, (0, 859), _c23)
except Exception:
    pass
layout["FREE_Guest_List_for_Offic"] = [0, 859, 1440, 1038]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/24_icon_6.54_Wy.png
try:
    _c24 = get_crop(24, 53, 62)
    canvas.paste(_c24, (116, 1), _c24)
except Exception:
    pass
layout["6.54_Wy"] = [116, 1, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/25_icon_6.54_Wy.png
try:
    _c25 = get_crop(25, 168, 144)
    canvas.paste(_c25, (48, 120), _c25)
except Exception:
    pass
layout["6.54_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/26_icon_The_Phantom_of_the_Opera.png
try:
    _c26 = get_crop(26, 1440, 168)
    canvas.paste(_c26, (0, 2098), _c26)
except Exception:
    pass
layout["The_Phantom_of_the_Opera"] = [0, 2098, 1440, 2266]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/27_icon_Account.png
try:
    _c27 = get_crop(27, 288, 168)
    canvas.paste(_c27, (1152, 2792), _c27)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/28_icon_Morm.png
try:
    _c28 = get_crop(28, 1440, 168)
    canvas.paste(_c28, (0, 1594), _c28)
except Exception:
    pass
layout["Morm"] = [0, 1594, 1440, 1762]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/29_text_Top_results.png
try:
    _c29 = get_crop(29, 295, 72)
    canvas.paste(_c29, (40, 373), _c29)
except Exception:
    pass
layout["Top_results"] = [40, 373, 335, 445]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/30_text_Performers.png
try:
    _c30 = get_crop(30, 296, 64)
    canvas.paste(_c30, (42, 760), _c30)
except Exception:
    pass
layout["Performers"] = [42, 760, 338, 824]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/31_text_Recent_searches.png
try:
    _c31 = get_crop(31, 436, 54)
    canvas.paste(_c31, (44, 1332), _c31)
except Exception:
    pass
layout["Recent_searches"] = [44, 1332, 480, 1386]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/32_text_Suggestions.png
try:
    _c32 = get_crop(32, 331, 72)
    canvas.paste(_c32, (40, 2378), _c32)
except Exception:
    pass
layout["Suggestions"] = [40, 2378, 371, 2450]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/33_text_Popular.png
try:
    _c33 = get_crop(33, 177, 56)
    canvas.paste(_c33, (234, 2535), _c33)
except Exception:
    pass
layout["Popular"] = [234, 2535, 411, 2591]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/34_text_Events_by_My_Performers.png
try:
    _c34 = get_crop(34, 1440, 149)
    canvas.paste(_c34, (0, 2643), _c34)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 2643, 1440, 2792]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/35_clickable_Popular.png
try:
    _c35 = get_crop(35, 1440, 168)
    canvas.paste(_c35, (0, 2475), _c35)
except Exception:
    pass
layout["Popular"] = [0, 2475, 1440, 2643]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/36_clickable_Tickets.png
try:
    _c36 = get_crop(36, 288, 168)
    canvas.paste(_c36, (576, 2792), _c36)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_10_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-13/37_clickable_Tracking.png
try:
    _c37 = get_crop(37, 288, 168)
    canvas.paste(_c37, (864, 2792), _c37)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]
