# page_id: page_seatgeek_1e6c9e893d9e4bc99959744188677162_04
# screenshot: 2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7.png
# step_index: 4/8
# task: Open SeatGeek. Search "Radio City Music Hall" and then add the venue to favorite. Who are the performers of the top recommended event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI elements for SeatGeek-like search page
# Uses provided canvas (PIL Image) and draw (ImageDraw)

# Colors
bg_color = (255, 255, 255)           # page background (dominant white)
muted_bg = (248, 248, 249)           # subtle off-white used for cards/search
line_color = (230, 230, 230)         # separators and borders
shadow_color = (0, 0, 0, 18)         # translucent shadow (if needed)
status_bar_color = (245, 245, 245)   # top status area
search_fill = (250, 250, 250)        # search bar fill
card_outline = (240, 240, 240)

w, h = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar area (top)
status_h = 84
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)
# subtle bottom divider for status bar
draw.line([(0, status_h), (w, status_h)], fill=line_color, width=1)

# Search bar card (rounded)
search_margin_x = 48
search_top = status_h + 12
search_bottom = search_top + 104
search_radius = 36
draw.rounded_rectangle(
    [(search_margin_x, search_top), (w - search_margin_x, search_bottom)],
    radius=search_radius,
    fill=search_fill,
    outline=card_outline,
    width=1
)

# Divider below search area
divider_y = search_bottom + 18
draw.line([(40, divider_y), (w - 40, divider_y)], fill=line_color, width=1)

# Section card backgrounds (rounded rectangles behind groups)
card_margin_x = 24
card_radius = 18

# Top Results card
top_results_top = divider_y + 28
top_results_bottom = top_results_top + 260
draw.rounded_rectangle(
    [(card_margin_x, top_results_top), (w - card_margin_x, top_results_bottom)],
    radius=card_radius,
    fill=bg_color,
    outline=card_outline,
    width=1
)

# Events card
events_top = top_results_bottom + 60
events_bottom = events_top + 420
draw.rounded_rectangle(
    [(card_margin_x, events_top), (w - card_margin_x, events_bottom)],
    radius=card_radius,
    fill=bg_color,
    outline=card_outline,
    width=1
)

# Venues card
venues_top = events_bottom + 80
venues_bottom = venues_top + 260
draw.rounded_rectangle(
    [(card_margin_x, venues_top), (w - card_margin_x, venues_bottom)],
    radius=card_radius,
    fill=bg_color,
    outline=card_outline,
    width=1
)

# Recent searches card
recent_top = venues_bottom + 160
recent_bottom = recent_top + 520
draw.rounded_rectangle(
    [(card_margin_x, recent_top), (w - card_margin_x, recent_bottom)],
    radius=card_radius,
    fill=bg_color,
    outline=card_outline,
    width=1
)

# Section separators (full-width thin lines)
separator_lines = [
    divider_y,                      # under search
    top_results_bottom + 20,        # after top results
    events_bottom + 40,             # after events
    venues_bottom + 40,             # after venues
    recent_bottom - 220             # inside recent area (visual grouping)
]
for y_pos in separator_lines:
    draw.line([(24, int(y_pos)), (w - 24, int(y_pos))], fill=line_color, width=1)

# Subtle shadow/crease above the bottom navigation bar
nav_top = h - 168  # matches detected nav icons area starting y ~2792 for 2960 canvas
# Draw top border line for nav and a faint shadow
draw.line([(0, nav_top), (w, nav_top)], fill=line_color, width=1)
# Faint translucent shadow strokes (simulate elevation)
for i, alpha in enumerate((14, 10, 6)):
    y = nav_top + 1 + i
    draw.line([(0, y), (w, y)], fill=(0, 0, 0, alpha), width=1)

# Bottom navigation background (keeps icons area clear for pasting)
draw.rectangle([(0, nav_top), (w, h)], fill=bg_color)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/00_icon_Recent_searches.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 2530), _c0)
except Exception:
    pass
layout["Recent_searches"] = [0, 2530, 1440, 2698]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/01_icon_New_York_NY.png
try:
    _c1 = get_crop(1, 1440, 179)
    canvas.paste(_c1, (0, 1217), _c1)
except Exception:
    pass
layout["New_York,_NY"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/02_icon_New_York_NY.png
try:
    _c2 = get_crop(2, 1440, 179)
    canvas.paste(_c2, (0, 1575), _c2)
except Exception:
    pass
layout["New_York,_NY"] = [0, 1575, 1440, 1754]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 59, 60)
    canvas.paste(_c3, (244, 4), _c3)
except Exception:
    pass
layout["icon_3"] = [244, 4, 303, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/04_icon_Shin.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (0, 2792), _c4)
except Exception:
    pass
layout["Shin"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/05_icon_Radio_City_Music_Hall_Stage_Door_Tour.png
try:
    _c5 = get_crop(5, 1440, 179)
    canvas.paste(_c5, (0, 471), _c5)
except Exception:
    pass
layout["Radio_City_Music_Hall_Sta"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/06_icon_Fri.png
try:
    _c6 = get_crop(6, 1440, 179)
    canvas.paste(_c6, (0, 829), _c6)
except Exception:
    pass
layout["Fri,"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 42, 70)
    canvas.paste(_c7, (1156, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1156, 0, 1198, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/08_icon_im.png
try:
    _c8 = get_crop(8, 288, 162)
    canvas.paste(_c8, (288, 2792), _c8)
except Exception:
    pass
layout["im"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 53, 61)
    canvas.paste(_c9, (315, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [315, 3, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/10_icon_New_York_NY.png
try:
    _c10 = get_crop(10, 1440, 179)
    canvas.paste(_c10, (0, 829), _c10)
except Exception:
    pass
layout["New_York,_NY"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/11_icon_8.32_Wy.png
try:
    _c11 = get_crop(11, 168, 144)
    canvas.paste(_c11, (48, 120), _c11)
except Exception:
    pass
layout["8.32_Wy"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/12_icon_8.32_Wy.png
try:
    _c12 = get_crop(12, 44, 61)
    canvas.paste(_c12, (187, 2), _c12)
except Exception:
    pass
layout["8.32_Wy"] = [187, 2, 231, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 96, 68)
    canvas.paste(_c13, (1218, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1218, 0, 1314, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/14_icon_Music_Hall_Stage_Door_Tour.png
try:
    _c14 = get_crop(14, 1440, 179)
    canvas.paste(_c14, (0, 1963), _c14)
except Exception:
    pass
layout["Music_Hall_Stage_Door_Tou"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/15_icon_Radio_City_Music_Hall.png
try:
    _c15 = get_crop(15, 1032, 144)
    canvas.paste(_c15, (216, 120), _c15)
except Exception:
    pass
layout["Radio_City_Music_Hall"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/16_icon_Tickets.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (576, 2792), _c16)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/17_icon_8.32_Wy.png
try:
    _c17 = get_crop(17, 53, 62)
    canvas.paste(_c17, (116, 1), _c17)
except Exception:
    pass
layout["8.32_Wy"] = [116, 1, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 44, 65)
    canvas.paste(_c18, (1326, 3), _c18)
except Exception:
    pass
layout["icon_18"] = [1326, 3, 1370, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/19_icon_Tracking.png
try:
    _c19 = get_crop(19, 288, 168)
    canvas.paste(_c19, (864, 2792), _c19)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/20_icon_Clear.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 120), _c20)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/21_icon_Radio_City_Music_Hall_Stage_Door_Tour.png
try:
    _c21 = get_crop(21, 1440, 179)
    canvas.paste(_c21, (0, 650), _c21)
except Exception:
    pass
layout["Radio_City_Music_Hall_Sta"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/22_icon_Account.png
try:
    _c22 = get_crop(22, 288, 168)
    canvas.paste(_c22, (1152, 2792), _c22)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/23_icon_Music_Hall_Stage_Door_Tour.png
try:
    _c23 = get_crop(23, 1440, 179)
    canvas.paste(_c23, (0, 2142), _c23)
except Exception:
    pass
layout["Music_Hall_Stage_Door_Tou"] = [0, 2142, 1440, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/24_icon_New_York_NY.png
try:
    _c24 = get_crop(24, 1440, 179)
    canvas.paste(_c24, (0, 1396), _c24)
except Exception:
    pass
layout["New_York,_NY"] = [0, 1396, 1440, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/25_icon_girl_in_red.png
try:
    _c25 = get_crop(25, 205, 59)
    canvas.paste(_c25, (234, 1426), _c25)
except Exception:
    pass
layout["girl_in_red"] = [234, 1426, 439, 1485]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/26_text_Top_results.png
try:
    _c26 = get_crop(26, 292, 68)
    canvas.paste(_c26, (41, 376), _c26)
except Exception:
    pass
layout["Top_results"] = [41, 376, 333, 444]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/27_text_Events.png
try:
    _c27 = get_crop(27, 177, 54)
    canvas.paste(_c27, (46, 1122), _c27)
except Exception:
    pass
layout["Events"] = [46, 1122, 223, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/28_text_Venues.png
try:
    _c28 = get_crop(28, 195, 56)
    canvas.paste(_c28, (43, 1868), _c28)
except Exception:
    pass
layout["Venues"] = [43, 1868, 238, 1924]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/29_text_Recent_searches.png
try:
    _c29 = get_crop(29, 436, 54)
    canvas.paste(_c29, (44, 2435), _c29)
except Exception:
    pass
layout["Recent_searches"] = [44, 2435, 480, 2489]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/30_text_Oracle_Arena.png
try:
    _c30 = get_crop(30, 294, 53)
    canvas.paste(_c30, (234, 2590), _c30)
except Exception:
    pass
layout["Oracle_Arena"] = [234, 2590, 528, 2643]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/31_text_Shin.png
try:
    _c31 = get_crop(31, 99, 41)
    canvas.paste(_c31, (237, 2760), _c31)
except Exception:
    pass
layout["Shin"] = [237, 2760, 336, 2801]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/1e6c9e893d9e4bc99959744188677162/step_04_2024_4_22_20_31_1e6c9e893d9e4bc99959744188677162-7/32_text_im.png
try:
    _c32 = get_crop(32, 57, 41)
    canvas.paste(_c32, (362, 2760), _c32)
except Exception:
    pass
layout["im"] = [362, 2760, 419, 2801]
