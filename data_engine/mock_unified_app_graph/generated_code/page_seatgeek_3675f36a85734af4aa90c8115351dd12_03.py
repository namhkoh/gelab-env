# page_id: page_seatgeek_3675f36a85734af4aa90c8115351dd12_03
# screenshot: 2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6.png
# step_index: 3/9
# task: Open SeatGeek. Search "The Fonda Theatre". Select the top popular event and track it. What is the lowest price?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural elements for the UI mockup
# Uses provided 'canvas' (PIL.Image) and 'draw' (PIL.ImageDraw)

# Colors
BG = (255, 255, 255)
STATUS_BG = (244, 244, 244)
SEARCH_FILL = (250, 250, 250)
STROKE = (230, 230, 230)
CARD_FILL = (252, 252, 252)
DIVIDER = (235, 235, 235)
BOTTOM_BG = (255, 255, 255)
SHADOW = (240, 240, 240)

W, H = canvas.size

# Clear canvas to background
draw.rectangle([(0, 0), (W, H)], fill=BG)

# Status bar (top)
status_h = 80
draw.rectangle([(0, 0), (W, status_h)], fill=STATUS_BG)

# Small subtle bottom line under status bar
draw.line([(0, status_h - 1), (W, status_h - 1)], fill=STROKE, width=1)

# Search bar (rounded rectangle) -- do not draw text or icons inside
search_x0 = 216
search_y0 = 120
search_w = 1032
search_h = 144
search_x1 = search_x0 + search_w
search_y1 = search_y0 + search_h
draw.rounded_rectangle(
    [(search_x0, search_y0), (search_x1, search_y1)],
    radius=28, fill=SEARCH_FILL, outline=STROKE, width=2
)

# Thin divider below search area
divider_y = search_y1 + 36
draw.line([(48, divider_y), (W - 48, divider_y)], fill=DIVIDER, width=2)

# "Recent searches" card background (rounded rectangle grouping the list)
card_left = 32
card_right = W - 32
card_top = divider_y + 24
# We'll assume 5 items at ~168px height (as in the detected layout)
item_h = 168
num_items = 5
card_bottom = card_top + item_h * num_items
draw.rounded_rectangle(
    [(card_left, card_top), (card_right, card_bottom)],
    radius=20, fill=CARD_FILL, outline=STROKE, width=1
)

# Internal separators for each list item inside the recent searches card
for i in range(1, num_items):
    y = card_top + i * item_h
    draw.line([(card_left + 24, y), (card_right - 24, y)], fill=DIVIDER, width=1)

# Big thin divider between Recent Searches and Suggestions (matches screenshot)
divider2_y = card_bottom + 24
draw.line([(48, divider2_y), (W - 48, divider2_y)], fill=DIVIDER, width=2)

# Suggestions card grouping
suggestion_top = divider2_y + 24
# Make suggestions card extend to include a few suggestion rows (3 rows)
suggestion_rows = 3
suggestion_row_h = 168
suggestion_bottom = suggestion_top + suggestion_rows * suggestion_row_h
draw.rounded_rectangle(
    [(card_left, suggestion_top), (card_right, suggestion_bottom)],
    radius=20, fill=CARD_FILL, outline=STROKE, width=1
)

# Separators inside suggestions
for i in range(1, suggestion_rows):
    y = suggestion_top + i * suggestion_row_h
    draw.line([(card_left + 24, y), (card_right - 24, y)], fill=DIVIDER, width=1)

# Large subtle divider line below Suggestions block
draw.line([(48, suggestion_bottom + 12), (W - 48, suggestion_bottom + 12)], fill=DIVIDER, width=1)

# Bottom navigation bar background and top border
nav_top = 2792
draw.rectangle([(0, nav_top), (W, H)], fill=BOTTOM_BG)
draw.line([(0, nav_top), (W, nav_top)], fill=STROKE, width=1)

# Slight shadow under cards to separate from background
# (soft single-line shadows for both cards)
shadow_offset = 8
draw.line([(card_left + 8, card_bottom + shadow_offset - 2),
           (card_right - 8, card_bottom + shadow_offset - 2)], fill=SHADOW, width=2)
draw.line([(card_left + 8, suggestion_bottom + shadow_offset - 2),
           (card_right - 8, suggestion_bottom + shadow_offset - 2)], fill=SHADOW, width=2)

# End of structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/00_icon_Madison_Square_Garden.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 471), _c0)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/01_icon_8.11_my.png
try:
    _c1 = get_crop(1, 168, 144)
    canvas.paste(_c1, (48, 120), _c1)
except Exception:
    pass
layout["8.11_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/02_icon_Tracking.png
try:
    _c2 = get_crop(2, 288, 168)
    canvas.paste(_c2, (864, 2792), _c2)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 47, 70)
    canvas.paste(_c3, (1153, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1153, 0, 1200, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/04_icon_Browse.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (0, 2792), _c4)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/05_icon_Just_Announced_by_My_Performers.png
try:
    _c5 = get_crop(5, 1440, 168)
    canvas.paste(_c5, (0, 1688), _c5)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/06_icon_Ed_Sheeran.png
try:
    _c6 = get_crop(6, 1440, 168)
    canvas.paste(_c6, (0, 807), _c6)
except Exception:
    pass
layout["Ed_Sheeran"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/07_icon_Tickets.png
try:
    _c7 = get_crop(7, 288, 168)
    canvas.paste(_c7, (576, 2792), _c7)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/08_icon_Ed_Sheeran.png
try:
    _c8 = get_crop(8, 1440, 168)
    canvas.paste(_c8, (0, 639), _c8)
except Exception:
    pass
layout["Ed_Sheeran"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 61, 64)
    canvas.paste(_c9, (243, 2), _c9)
except Exception:
    pass
layout["icon_9"] = [243, 2, 304, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/10_icon_8.11_my.png
try:
    _c10 = get_crop(10, 54, 64)
    canvas.paste(_c10, (115, 1), _c10)
except Exception:
    pass
layout["8.11_my"] = [115, 1, 169, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/11_icon_Metropolitan_Opera.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 807), _c11)
except Exception:
    pass
layout["Metropolitan_Opera"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/12_icon_Metropolitan_Opera.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 975), _c12)
except Exception:
    pass
layout["Metropolitan_Opera"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 102, 68)
    canvas.paste(_c13, (1213, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1213, 0, 1315, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/14_icon_Clear.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 120), _c14)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/15_icon_Events_by_My_Performers.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 1520), _c15)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/16_icon_Madison_Square_Garden.png
try:
    _c16 = get_crop(16, 1440, 168)
    canvas.paste(_c16, (0, 639), _c16)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/17_icon_8.11_my.png
try:
    _c17 = get_crop(17, 45, 63)
    canvas.paste(_c17, (187, 1), _c17)
except Exception:
    pass
layout["8.11_my"] = [187, 1, 232, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/18_icon_Suggestions.png
try:
    _c18 = get_crop(18, 1440, 168)
    canvas.paste(_c18, (0, 1143), _c18)
except Exception:
    pass
layout["Suggestions"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/19_icon_Account.png
try:
    _c19 = get_crop(19, 288, 168)
    canvas.paste(_c19, (1152, 2792), _c19)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 59, 64)
    canvas.paste(_c20, (313, 2), _c20)
except Exception:
    pass
layout["icon_20"] = [313, 2, 372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 53, 68)
    canvas.paste(_c21, (1319, 0), _c21)
except Exception:
    pass
layout["icon_21"] = [1319, 0, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/22_icon_Search.png
try:
    _c22 = get_crop(22, 288, 162)
    canvas.paste(_c22, (288, 2792), _c22)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/23_icon_Performer_event_or_venue.png
try:
    _c23 = get_crop(23, 1032, 144)
    canvas.paste(_c23, (216, 120), _c23)
except Exception:
    pass
layout["Performer;_event;_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/24_icon_Just_Announced_by_My_Performers.png
try:
    _c24 = get_crop(24, 1440, 168)
    canvas.paste(_c24, (0, 1856), _c24)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/25_icon_Search.png
try:
    _c25 = get_crop(25, 288, 162)
    canvas.paste(_c25, (288, 2792), _c25)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/26_icon_Los_Angeles_Lakers.png
try:
    _c26 = get_crop(26, 1440, 168)
    canvas.paste(_c26, (0, 1143), _c26)
except Exception:
    pass
layout["Los_Angeles_Lakers"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/27_text_Recent_searches.png
try:
    _c27 = get_crop(27, 168, 144)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3675f36a85734af4aa90c8115351dd12/step_03_2024_4_22_20_10_3675f36a85734af4aa90c8115351dd12-6/28_text_Suggestions.png
try:
    _c28 = get_crop(28, 331, 74)
    canvas.paste(_c28, (40, 1423), _c28)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
