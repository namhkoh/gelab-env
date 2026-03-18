# page_id: page_seatgeek_03681efe5b614d869915a8728b755f3d_04
# screenshot: 2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7.png
# step_index: 4/10
# task: Open SeatGeek. Search "Metropolitan Opera". Find the next available show. Filter by "best seats". What section are they in for the lowest price tickets?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (canvas provided, default white)
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar area (top)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(245, 245, 245))  # subtle light gray status background
# subtle bottom divider under status bar
draw.line((0, status_h, 1440, status_h), fill=(230, 230, 230), width=1)

# Search bar / header area (rounded input background)
search_margin_x = 40
search_top = 76
search_bottom = 220
search_rect = (search_margin_x, search_top, 1440 - search_margin_x, search_bottom)
try:
    draw.rounded_rectangle(search_rect, radius=28, fill=(255, 255, 255), outline=(235, 235, 235), width=2)
except Exception:
    draw.rectangle(search_rect, fill=(255, 255, 255), outline=(235, 235, 235))

# Thin divider under the search area
divider_y = search_bottom + 20
draw.line((24, divider_y, 1440 - 24, divider_y), fill=(235, 235, 235), width=2)

# Section card/background containers (soft off-white cards to group list areas)
card_margin_x = 28
# Top Results card (contains "Top results" list)
top_results_top = divider_y + 24
top_results_bottom = 920
try:
    draw.rounded_rectangle((card_margin_x, top_results_top, 1440 - card_margin_x, top_results_bottom),
                           radius=12, fill=(250, 250, 250), outline=(240, 240, 240), width=1)
except Exception:
    draw.rectangle((card_margin_x, top_results_top, 1440 - card_margin_x, top_results_bottom),
                   fill=(250, 250, 250), outline=(240, 240, 240))

# Performers card
performers_top = top_results_bottom + 24
performers_bottom = 1400
try:
    draw.rounded_rectangle((card_margin_x, performers_top, 1440 - card_margin_x, performers_bottom),
                           radius=12, fill=(250, 250, 250), outline=(240, 240, 240), width=1)
except Exception:
    draw.rectangle((card_margin_x, performers_top, 1440 - card_margin_x, performers_bottom),
                   fill=(250, 250, 250), outline=(240, 240, 240))

# Events card
events_top = performers_bottom + 28
events_bottom = 2000
try:
    draw.rounded_rectangle((card_margin_x, events_top, 1440 - card_margin_x, events_bottom),
                           radius=12, fill=(250, 250, 250), outline=(240, 240, 240), width=1)
except Exception:
    draw.rectangle((card_margin_x, events_top, 1440 - card_margin_x, events_bottom),
                   fill=(250, 250, 250), outline=(240, 240, 240))

# Venues card
venues_top = events_bottom + 36
venues_bottom = 2600
try:
    draw.rounded_rectangle((card_margin_x, venues_top, 1440 - card_margin_x, venues_bottom),
                           radius=12, fill=(250, 250, 250), outline=(240, 240, 240), width=1)
except Exception:
    draw.rectangle((card_margin_x, venues_top, 1440 - card_margin_x, venues_bottom),
                   fill=(250, 250, 250), outline=(240, 240, 240))

# Subtle separators between list items / sections (full width thin lines)
sep_color = (235, 235, 235)
separator_positions = [
    divider_y + 120,   # within top results
    divider_y + 300,
    divider_y + 480,
    performers_top + 120,
    performers_top + 300,
    events_top + 120,
    events_top + 300,
    venues_top + 120,
]
for y in separator_positions:
    if 0 < y < 2792:  # avoid drawing over bottom nav area
        draw.line((24, y, 1440 - 24, y), fill=sep_color, width=1)

# Light section heading rule (spaced separators to delimit groups)
draw.line((24, top_results_bottom + 12, 1440 - 24, top_results_bottom + 12), fill=(230, 230, 230), width=1)
draw.line((24, performers_bottom + 12, 1440 - 24, performers_bottom + 12), fill=(230, 230, 230), width=1)
draw.line((24, events_bottom + 12, 1440 - 24, events_bottom + 12), fill=(230, 230, 230), width=1)

# Bottom navigation bar area (reserve space, draw subtle top border and light background)
bottom_nav_top = 2792
draw.rectangle((0, bottom_nav_top, 1440, 2960), fill=(255, 255, 255))
draw.line((0, bottom_nav_top, 1440, bottom_nav_top), fill=(230, 230, 230), width=2)

# Subtle drop shadows under the large cards (very light)
shadow_color = (245, 245, 245)
# shadow under top_results
draw.rectangle((card_margin_x, top_results_bottom + 1, 1440 - card_margin_x, top_results_bottom + 6), fill=shadow_color)
# shadow under performers
draw.rectangle((card_margin_x, performers_bottom + 1, 1440 - card_margin_x, performers_bottom + 6), fill=shadow_color)
# shadow under events
draw.rectangle((card_margin_x, events_bottom + 1, 1440 - card_margin_x, events_bottom + 6), fill=shadow_color)

# Decorative left gutter line (very subtle) to guide eye down the page
gutter_x = 40
draw.line((gutter_x, divider_y + 8, gutter_x, venues_bottom - 8), fill=(250, 250, 250), width=8)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/00_icon_Performers.png
try:
    _c0 = get_crop(0, 1440, 179)
    canvas.paste(_c0, (0, 1217), _c0)
except Exception:
    pass
layout["Performers"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/01_icon_2_events.png
try:
    _c1 = get_crop(1, 1440, 179)
    canvas.paste(_c1, (0, 650), _c1)
except Exception:
    pass
layout["2_events"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/02_icon_No_events.png
try:
    _c2 = get_crop(2, 1440, 179)
    canvas.paste(_c2, (0, 829), _c2)
except Exception:
    pass
layout["No_events"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/03_icon_No_events.png
try:
    _c3 = get_crop(3, 1440, 179)
    canvas.paste(_c3, (0, 1396), _c3)
except Exception:
    pass
layout["No_events"] = [0, 1396, 1440, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/04_icon_Metropolitan_Operal.png
try:
    _c4 = get_crop(4, 1032, 144)
    canvas.paste(_c4, (216, 120), _c4)
except Exception:
    pass
layout["Metropolitan_Operal"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/05_icon_Events.png
try:
    _c5 = get_crop(5, 1440, 179)
    canvas.paste(_c5, (0, 1784), _c5)
except Exception:
    pass
layout["Events"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/06_icon_Sat.png
try:
    _c6 = get_crop(6, 1440, 179)
    canvas.paste(_c6, (0, 1963), _c6)
except Exception:
    pass
layout["Sat,"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 59, 61)
    canvas.paste(_c7, (244, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [244, 3, 303, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/08_icon_New_York_NY.png
try:
    _c8 = get_crop(8, 1440, 179)
    canvas.paste(_c8, (0, 1784), _c8)
except Exception:
    pass
layout["New_York,_NY"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/09_icon_New_York.png
try:
    _c9 = get_crop(9, 1440, 179)
    canvas.paste(_c9, (0, 2142), _c9)
except Exception:
    pass
layout["New_York"] = [0, 2142, 1440, 2321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 43, 69)
    canvas.paste(_c10, (1155, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1155, 0, 1198, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 93, 69)
    canvas.paste(_c11, (1219, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1219, 0, 1312, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/12_icon_7.56_my.png
try:
    _c12 = get_crop(12, 168, 144)
    canvas.paste(_c12, (48, 120), _c12)
except Exception:
    pass
layout["7.56_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 54, 61)
    canvas.paste(_c13, (315, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [315, 3, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/14_icon_57_events.png
try:
    _c14 = get_crop(14, 288, 168)
    canvas.paste(_c14, (0, 2792), _c14)
except Exception:
    pass
layout["57_events"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/15_icon_Metropolitan_Opera.png
try:
    _c15 = get_crop(15, 1440, 179)
    canvas.paste(_c15, (0, 471), _c15)
except Exception:
    pass
layout["Metropolitan_Opera"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/16_icon_Tracking.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (864, 2792), _c16)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/17_icon_7.56_my.png
try:
    _c17 = get_crop(17, 44, 61)
    canvas.paste(_c17, (187, 2), _c17)
except Exception:
    pass
layout["7.56_my"] = [187, 2, 231, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 45, 64)
    canvas.paste(_c18, (1326, 3), _c18)
except Exception:
    pass
layout["icon_18"] = [1326, 3, 1371, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/19_icon_LA_TRAVIATA-Opera_Metropolitana_de_Barce.png
try:
    _c19 = get_crop(19, 1440, 179)
    canvas.paste(_c19, (0, 1217), _c19)
except Exception:
    pass
layout["LA_TRAVIATA-Opera_Metropo"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/20_icon_Account.png
try:
    _c20 = get_crop(20, 288, 168)
    canvas.paste(_c20, (1152, 2792), _c20)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/21_icon_57_events.png
try:
    _c21 = get_crop(21, 288, 162)
    canvas.paste(_c21, (288, 2792), _c21)
except Exception:
    pass
layout["57_events"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/22_icon_LA_TRAVIATA-Opera_Metropolitana_de_Barce.png
try:
    _c22 = get_crop(22, 1440, 179)
    canvas.paste(_c22, (0, 1396), _c22)
except Exception:
    pass
layout["LA_TRAVIATA-Opera_Metropo"] = [0, 1396, 1440, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/23_icon_Clear.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1248, 120), _c23)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/24_icon_LA_TRAVIATA-Opera_Metropolitana_de_Barce.png
try:
    _c24 = get_crop(24, 1440, 179)
    canvas.paste(_c24, (0, 650), _c24)
except Exception:
    pass
layout["LA_TRAVIATA-Opera_Metropo"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/25_icon_Metropolitan_Opera.png
try:
    _c25 = get_crop(25, 1440, 179)
    canvas.paste(_c25, (0, 650), _c25)
except Exception:
    pass
layout["Metropolitan_Opera"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/26_icon_7.56_my.png
try:
    _c26 = get_crop(26, 52, 62)
    canvas.paste(_c26, (116, 1), _c26)
except Exception:
    pass
layout["7.56_my"] = [116, 1, 168, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/27_icon_Tickets.png
try:
    _c27 = get_crop(27, 288, 168)
    canvas.paste(_c27, (576, 2792), _c27)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/28_text_Top_results.png
try:
    _c28 = get_crop(28, 295, 72)
    canvas.paste(_c28, (40, 373), _c28)
except Exception:
    pass
layout["Top_results"] = [40, 373, 335, 445]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/29_text_No_events.png
try:
    _c29 = get_crop(29, 201, 43)
    canvas.paste(_c29, (239, 931), _c29)
except Exception:
    pass
layout["No_events"] = [239, 931, 440, 974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/30_text_Performers.png
try:
    _c30 = get_crop(30, 293, 54)
    canvas.paste(_c30, (44, 1122), _c30)
except Exception:
    pass
layout["Performers"] = [44, 1122, 337, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/31_text_Events.png
try:
    _c31 = get_crop(31, 179, 52)
    canvas.paste(_c31, (44, 1691), _c31)
except Exception:
    pass
layout["Events"] = [44, 1691, 223, 1743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/32_text_Venues.png
try:
    _c32 = get_crop(32, 197, 62)
    canvas.paste(_c32, (42, 2433), _c32)
except Exception:
    pass
layout["Venues"] = [42, 2433, 239, 2495]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/33_text_Metropolitan_Opera.png
try:
    _c33 = get_crop(33, 1440, 179)
    canvas.paste(_c33, (0, 2530), _c33)
except Exception:
    pass
layout["Metropolitan_Opera"] = [0, 2530, 1440, 2709]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/03681efe5b614d869915a8728b755f3d/step_04_2024_4_22_19_56_03681efe5b614d869915a8728b755f3d-7/34_text_57_events.png
try:
    _c34 = get_crop(34, 196, 43)
    canvas.paste(_c34, (237, 2630), _c34)
except Exception:
    pass
layout["57_events"] = [237, 2630, 433, 2673]
