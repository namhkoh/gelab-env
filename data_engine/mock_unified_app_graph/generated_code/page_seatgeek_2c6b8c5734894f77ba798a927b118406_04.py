# page_id: page_seatgeek_2c6b8c5734894f77ba798a927b118406_04
# screenshot: 2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7.png
# step_index: 4/5
# task: Open SeatGeek. Search "Wembley Stadium". Show the next five football matches. Add to watch list.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
bg_color = (250, 250, 250)  # very light off-white
draw.rectangle([(0, 0), canvas.size], fill=bg_color)

# Status bar area (approx 0-50px)
status_h = 50
status_color = (236, 236, 236)  # subtle light gray
draw.rectangle([(0, 0), (canvas.width, status_h)], fill=status_color)

# Thin subtle bottom hairline under status bar
hairline_color = (220, 220, 220)
draw.line([(0, status_h), (canvas.width, status_h)], fill=hairline_color, width=1)

# Search bar background (rounded rect)
search_left = 40
search_top = 60
search_right = canvas.width - 40
search_bottom = 204
search_radius = 22
search_fill = (245, 245, 246)
search_outline = (230, 230, 230)
draw.rounded_rectangle([(search_left, search_top), (search_right, search_bottom)],
                       radius=search_radius, fill=search_fill, outline=search_outline, width=1)

# Divider under search area
divider_y = search_bottom + 12
draw.line([(40, divider_y), (canvas.width - 40, divider_y)], fill=hairline_color, width=1)

# Section card backgrounds (subtle white cards with rounded corners)
card_radius = 16
card_outline = (235, 235, 235)
card_fill = (255, 255, 255)

# Top Results card
top_results_top = divider_y + 30
top_results_bottom = top_results_top + 220
draw.rounded_rectangle([(24, top_results_top), (canvas.width - 24, top_results_bottom)],
                       radius=card_radius, fill=card_fill, outline=card_outline, width=1)

# Divider after Top Results
y = top_results_bottom + 18
draw.line([(24, y), (canvas.width - 24, y)], fill=hairline_color, width=1)

# Performers card
performers_top = y + 20
performers_bottom = performers_top + 120
draw.rounded_rectangle([(24, performers_top), (canvas.width - 24, performers_bottom)],
                       radius=card_radius, fill=card_fill, outline=card_outline, width=1)

# Divider after Performers
y2 = performers_bottom + 18
draw.line([(24, y2), (canvas.width - 24, y2)], fill=hairline_color, width=1)

# Events section card (larger area for event rows)
events_top = y2 + 40
events_bottom = events_top + 520
draw.rounded_rectangle([(16, events_top), (canvas.width - 16, events_bottom)],
                       radius=18, fill=card_fill, outline=card_outline, width=1)

# Inside Events: subtle separators for three event rows (approximate)
row_x1 = 24
row_x2 = canvas.width - 24
row_height = 120
for i in range(1, 4):
    ry = events_top + i * row_height + 8
    draw.line([(row_x1, ry), (row_x2, ry)], fill=hairline_color, width=1)

# Divider after Events
events_div_y = events_bottom + 18
draw.line([(24, events_div_y), (canvas.width - 24, events_div_y)], fill=hairline_color, width=1)

# Venues card (list of venue rows)
venues_top = events_div_y + 24
venues_bottom = venues_top + 660
draw.rounded_rectangle([(24, venues_top), (canvas.width - 24, venues_bottom)],
                       radius=card_radius, fill=card_fill, outline=card_outline, width=1)

# Subtle separators between venue rows (approx every 140px)
venue_row_h = 140
for i in range(1, 4):
    vy = venues_top + i * venue_row_h
    if vy < venues_bottom - 8:
        draw.line([(40, vy), (canvas.width - 40, vy)], fill=hairline_color, width=1)

# Floating subtle section separators across full width (to mirror subtle app dividers)
divider_positions = [search_bottom + 12, top_results_bottom + 18, performers_bottom + 18,
                     events_bottom + 18, venues_top + 0]
for dy in divider_positions:
    draw.line([(0, dy), (canvas.width, dy)], fill=(245,245,245), width=1)

# Bottom navigation bar background and top divider
nav_top = 2792
nav_color = (255, 255, 255)
draw.rectangle([(0, nav_top), (canvas.width, canvas.height)], fill=nav_color)
draw.line([(24, nav_top), (canvas.width - 24, nav_top)], fill=hairline_color, width=1)

# Subtle shadow above the bottom nav to lift it visually
shadow_color = (240, 240, 240)
draw.line([(0, nav_top + 1), (canvas.width, nav_top + 1)], fill=shadow_color, width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/00_icon_No_events.png
try:
    _c0 = get_crop(0, 1440, 179)
    canvas.paste(_c0, (0, 829), _c0)
except Exception:
    pass
layout["No_events"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/01_icon_Performers.png
try:
    _c1 = get_crop(1, 1440, 179)
    canvas.paste(_c1, (0, 1217), _c1)
except Exception:
    pass
layout["Performers"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 59, 61)
    canvas.paste(_c2, (244, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [244, 3, 303, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/03_icon_Gog.png
try:
    _c3 = get_crop(3, 1440, 179)
    canvas.paste(_c3, (0, 1784), _c3)
except Exception:
    pass
layout["Gog"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 43, 70)
    canvas.paste(_c4, (1155, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1155, 0, 1198, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/05_icon_7.05_my.png
try:
    _c5 = get_crop(5, 168, 144)
    canvas.paste(_c5, (48, 120), _c5)
except Exception:
    pass
layout["7.05_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/06_icon_Fri.png
try:
    _c6 = get_crop(6, 1440, 179)
    canvas.paste(_c6, (0, 1963), _c6)
except Exception:
    pass
layout["Fri,"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 54, 61)
    canvas.paste(_c7, (315, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [315, 3, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 92, 69)
    canvas.paste(_c8, (1219, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1219, 0, 1311, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/09_icon_Santa_Clara_CA.png
try:
    _c9 = get_crop(9, 1440, 179)
    canvas.paste(_c9, (0, 1963), _c9)
except Exception:
    pass
layout["Santa_Clara,_CA"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/10_icon_Account.png
try:
    _c10 = get_crop(10, 288, 168)
    canvas.paste(_c10, (1152, 2792), _c10)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/11_icon_Tickets.png
try:
    _c11 = get_crop(11, 288, 168)
    canvas.paste(_c11, (576, 2792), _c11)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/12_icon_Columbus_OH.png
try:
    _c12 = get_crop(12, 1440, 179)
    canvas.paste(_c12, (0, 1784), _c12)
except Exception:
    pass
layout["Columbus,_OH"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/13_icon_Wembley_Stadium.png
try:
    _c13 = get_crop(13, 1440, 179)
    canvas.paste(_c13, (0, 471), _c13)
except Exception:
    pass
layout["Wembley_Stadium"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/14_icon_Wembley_Stadium_Tour.png
try:
    _c14 = get_crop(14, 1440, 179)
    canvas.paste(_c14, (0, 1217), _c14)
except Exception:
    pass
layout["Wembley_Stadium_Tour"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/15_icon_7.05_my.png
try:
    _c15 = get_crop(15, 45, 61)
    canvas.paste(_c15, (187, 2), _c15)
except Exception:
    pass
layout["7.05_my"] = [187, 2, 232, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/16_icon_Tracking.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (864, 2792), _c16)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 45, 66)
    canvas.paste(_c17, (1326, 2), _c17)
except Exception:
    pass
layout["icon_17"] = [1326, 2, 1371, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/18_icon_Wembley_Stadium.png
try:
    _c18 = get_crop(18, 288, 162)
    canvas.paste(_c18, (288, 2792), _c18)
except Exception:
    pass
layout["Wembley_Stadium"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/19_icon_Clear.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 120), _c19)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/20_icon_Glendale_AZ.png
try:
    _c20 = get_crop(20, 1440, 179)
    canvas.paste(_c20, (0, 1605), _c20)
except Exception:
    pass
layout["Glendale,_AZ"] = [0, 1605, 1440, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/21_icon_Wembley_Stadium_Tour.png
try:
    _c21 = get_crop(21, 1440, 179)
    canvas.paste(_c21, (0, 650), _c21)
except Exception:
    pass
layout["Wembley_Stadium_Tour"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/22_icon_Wembley_Stadium.png
try:
    _c22 = get_crop(22, 1440, 179)
    canvas.paste(_c22, (0, 2351), _c22)
except Exception:
    pass
layout["Wembley_Stadium"] = [0, 2351, 1440, 2530]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/23_icon_Wembley_Stadium_Tour.png
try:
    _c23 = get_crop(23, 1440, 179)
    canvas.paste(_c23, (0, 829), _c23)
except Exception:
    pass
layout["Wembley_Stadium_Tour"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/24_icon_7.05_my.png
try:
    _c24 = get_crop(24, 52, 62)
    canvas.paste(_c24, (117, 1), _c24)
except Exception:
    pass
layout["7.05_my"] = [117, 1, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/25_icon_Gog.png
try:
    _c25 = get_crop(25, 1440, 179)
    canvas.paste(_c25, (0, 1605), _c25)
except Exception:
    pass
layout["Gog"] = [0, 1605, 1440, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/26_text_Wembley_Stadium.png
try:
    _c26 = get_crop(26, 1032, 144)
    canvas.paste(_c26, (216, 120), _c26)
except Exception:
    pass
layout["Wembley_Stadium"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/27_text_Top_results.png
try:
    _c27 = get_crop(27, 295, 72)
    canvas.paste(_c27, (40, 373), _c27)
except Exception:
    pass
layout["Top_results"] = [40, 373, 335, 445]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/28_text_No_events.png
try:
    _c28 = get_crop(28, 201, 43)
    canvas.paste(_c28, (239, 931), _c28)
except Exception:
    pass
layout["No_events"] = [239, 931, 440, 974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/29_text_Performers.png
try:
    _c29 = get_crop(29, 293, 54)
    canvas.paste(_c29, (44, 1122), _c29)
except Exception:
    pass
layout["Performers"] = [44, 1122, 337, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/30_text_Events.png
try:
    _c30 = get_crop(30, 177, 54)
    canvas.paste(_c30, (46, 1510), _c30)
except Exception:
    pass
layout["Events"] = [46, 1510, 223, 1564]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/31_text_Venues.png
try:
    _c31 = get_crop(31, 197, 60)
    canvas.paste(_c31, (42, 2253), _c31)
except Exception:
    pass
layout["Venues"] = [42, 2253, 239, 2313]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/32_text_Wembley_Stadium.png
try:
    _c32 = get_crop(32, 1440, 179)
    canvas.paste(_c32, (0, 2530), _c32)
except Exception:
    pass
layout["Wembley_Stadium"] = [0, 2530, 1440, 2709]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/33_text_No_events.png
try:
    _c33 = get_crop(33, 201, 40)
    canvas.paste(_c33, (239, 2633), _c33)
except Exception:
    pass
layout["No_events"] = [239, 2633, 440, 2673]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/34_text_Wembley_Stadium.png
try:
    _c34 = get_crop(34, 288, 162)
    canvas.paste(_c34, (288, 2792), _c34)
except Exception:
    pass
layout["Wembley_Stadium"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2c6b8c5734894f77ba798a927b118406/step_04_2024_4_22_19_4_2c6b8c5734894f77ba798a927b118406-7/35_clickable_Browse.png
try:
    _c35 = get_crop(35, 288, 168)
    canvas.paste(_c35, (0, 2792), _c35)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]
