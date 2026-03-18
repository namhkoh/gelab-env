# page_id: page_seatgeek_6623dffea11a48f2955bafde23a3f1c7_02
# screenshot: 2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5.png
# step_index: 2/9
# task: Open SeatGeek. Search "New York Knicks" and select the second upcoming event, show the location of the event and track the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background with a very light off-white to match the app background
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFB")

# Status bar area (top ~72px) - subtle light gray
draw.rectangle([(0, 0), (1440, 72)], fill="#EDEDED")

# Search bar background (rounded) below the status bar
search_left = 48
search_top = 72
search_right = 1440 - 48
search_bottom = search_top + 144
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=20,
    fill="#F5F5F5",
    outline="#E6E6E6",
    width=1
)

# Thin divider under the search area
divider_y = search_bottom + 24
draw.line([(24, divider_y), (1440 - 24, divider_y)], fill="#E6E6E6", width=1)

# Background "card" for Recent searches (rounded white card)
recent_card_top = divider_y + 40
recent_card_bottom = recent_card_top + 700  # covers the list area
card_margin_x = 24
# subtle base shadow layer
draw.rounded_rectangle(
    [(card_margin_x, recent_card_top + 8), (1440 - card_margin_x, recent_card_bottom + 8)],
    radius=18,
    fill="#F6F6F6"
)
# white card
draw.rounded_rectangle(
    [(card_margin_x, recent_card_top), (1440 - card_margin_x, recent_card_bottom)],
    radius=18,
    fill="#FFFFFF",
    outline=None
)

# Thin separator line across the card (to hint section boundary)
sep_y = recent_card_bottom + 8
draw.line([(card_margin_x + 8, sep_y), (1440 - card_margin_x - 8, sep_y)], fill="#ECECEC", width=1)

# Background "card" for Suggestions (rounded white card)
suggestions_top = sep_y + 40
suggestions_bottom = suggestions_top + 420
# subtle shadow
draw.rounded_rectangle(
    [(card_margin_x, suggestions_top + 8), (1440 - card_margin_x, suggestions_bottom + 8)],
    radius=18,
    fill="#F6F6F6"
)
# white card
draw.rounded_rectangle(
    [(card_margin_x, suggestions_top), (1440 - card_margin_x, suggestions_bottom)],
    radius=18,
    fill="#FFFFFF",
    outline=None
)

# Separator lines inside suggestions card to create row cues (light, subtle)
row_height = 100
for i in range(1, 3):
    y = suggestions_top + i * row_height
    draw.line([(card_margin_x + 24, y), (1440 - card_margin_x - 24, y)], fill="#F1F1F1", width=1)

# Bottom navigation area: white panel with a subtle top border/shadow
nav_top = 2792
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#FFFFFF")
draw.line([(0, nav_top), (1440, nav_top)], fill="#E8E8E8", width=2)

# Subtle left and right page margins shadow (very light) to match app depth
draw.rectangle([(0, 72), (12, 2960 - 168)], fill="#FAFAFA")
draw.rectangle([(1440 - 12, 72), (1440, 2960 - 168)], fill="#FAFAFA")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/00_icon_Recent_searches.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 471), _c0)
except Exception:
    pass
layout["Recent_searches"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/01_icon_Mormi.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 639), _c1)
except Exception:
    pass
layout["Mormi"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/02_icon_Suggestions.png
try:
    _c2 = get_crop(2, 1440, 168)
    canvas.paste(_c2, (0, 1143), _c2)
except Exception:
    pass
layout["Suggestions"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/03_icon_Golden_State_Warriors.png
try:
    _c3 = get_crop(3, 1440, 168)
    canvas.paste(_c3, (0, 471), _c3)
except Exception:
    pass
layout["Golden_State_Warriors"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 47, 70)
    canvas.paste(_c4, (1153, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1153, 0, 1200, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/05_icon_Just_Announced_by_My_Performers.png
try:
    _c5 = get_crop(5, 1440, 168)
    canvas.paste(_c5, (0, 1688), _c5)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/06_icon_Tracking.png
try:
    _c6 = get_crop(6, 288, 168)
    canvas.paste(_c6, (864, 2792), _c6)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 66, 63)
    canvas.paste(_c7, (242, 2), _c7)
except Exception:
    pass
layout["icon_7"] = [242, 2, 308, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/08_icon_Tickets.png
try:
    _c8 = get_crop(8, 288, 168)
    canvas.paste(_c8, (576, 2792), _c8)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 96, 69)
    canvas.paste(_c9, (1217, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1217, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/10_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c10 = get_crop(10, 1440, 168)
    canvas.paste(_c10, (0, 975), _c10)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/11_icon_Browse.png
try:
    _c11 = get_crop(11, 288, 168)
    canvas.paste(_c11, (0, 2792), _c11)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/12_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 1143), _c12)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/13_icon_The_Book_f_Mormon.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 807), _c13)
except Exception:
    pass
layout["The_Book_f_Mormon"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/14_icon_The_Lion_King.png
try:
    _c14 = get_crop(14, 1440, 168)
    canvas.paste(_c14, (0, 975), _c14)
except Exception:
    pass
layout["The_Lion_King"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/15_icon_Los_Angeles_Clippers.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 639), _c15)
except Exception:
    pass
layout["Los_Angeles_Clippers"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/16_icon_Events_by_My_Performers.png
try:
    _c16 = get_crop(16, 1440, 168)
    canvas.paste(_c16, (0, 1520), _c16)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/17_icon_Clear.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 120), _c17)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/18_icon_6.57_Wy.png
try:
    _c18 = get_crop(18, 47, 63)
    canvas.paste(_c18, (186, 1), _c18)
except Exception:
    pass
layout["6.57_Wy"] = [186, 1, 233, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/19_icon_6.57_Wy.png
try:
    _c19 = get_crop(19, 57, 65)
    canvas.paste(_c19, (113, 0), _c19)
except Exception:
    pass
layout["6.57_Wy"] = [113, 0, 170, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/20_icon_Account.png
try:
    _c20 = get_crop(20, 288, 168)
    canvas.paste(_c20, (1152, 2792), _c20)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 52, 69)
    canvas.paste(_c21, (1319, 0), _c21)
except Exception:
    pass
layout["icon_21"] = [1319, 0, 1371, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/22_icon_6.57_Wy.png
try:
    _c22 = get_crop(22, 168, 144)
    canvas.paste(_c22, (48, 120), _c22)
except Exception:
    pass
layout["6.57_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/23_icon_Just_Announced_by_My_Performers.png
try:
    _c23 = get_crop(23, 1440, 168)
    canvas.paste(_c23, (0, 1856), _c23)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/24_icon_Search.png
try:
    _c24 = get_crop(24, 288, 162)
    canvas.paste(_c24, (288, 2792), _c24)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/25_text_Performer_event_or_venue.png
try:
    _c25 = get_crop(25, 1032, 144)
    canvas.paste(_c25, (216, 120), _c25)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/26_text_Recent_searches.png
try:
    _c26 = get_crop(26, 168, 144)
    canvas.paste(_c26, (48, 120), _c26)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/6623dffea11a48f2955bafde23a3f1c7/step_02_2024_4_22_18_56_6623dffea11a48f2955bafde23a3f1c7-5/27_text_Suggestions.png
try:
    _c27 = get_crop(27, 331, 74)
    canvas.paste(_c27, (40, 1423), _c27)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
