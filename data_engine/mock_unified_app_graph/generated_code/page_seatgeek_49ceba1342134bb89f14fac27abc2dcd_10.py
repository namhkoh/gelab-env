# page_id: page_seatgeek_49ceba1342134bb89f14fac27abc2dcd_10
# screenshot: 2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13.png
# step_index: 10/12
# task: Open SeatGeek. Track "New York Yankees", "Boston Red Sox".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background (dominant light/off-white from screenshot)
bg_color = (250, 250, 250)  # very light grey / off-white
draw.rectangle([0, 0, canvas.width, canvas.height], fill=bg_color)

# Status bar area at top (~70px tall)
status_bar_h = 72
status_color = (236, 236, 236)  # subtle light grey
draw.rectangle([0, 0, canvas.width, status_bar_h], fill=status_color)

# Thin divider under status bar
draw.line([(0, status_bar_h), (canvas.width, status_bar_h)], fill=(220,220,220), width=1)

# Search box background (rounded) below status bar
search_left = 40
search_top = status_bar_h + 16
search_right = canvas.width - 40
search_bottom = search_top + 112
search_fill = (245, 245, 245)  # very light grey for input background
search_border = (226, 226, 226)
radius = 28
draw.rounded_rectangle([search_left, search_top, search_right, search_bottom],
                       radius=radius, fill=search_fill, outline=search_border, width=1)

# Subtle divider line under the search area
divider_y = search_bottom + 20
draw.line([(36, divider_y), (canvas.width-36, divider_y)], fill=(235,235,235), width=1)

# Section card backgrounds (rounded rectangles behind groups)
card_radius = 12
card_fill = (255, 255, 255)  # white cards slightly contrasted against BG
card_border = (240, 240, 240)

# Top results card
top_results_top = divider_y + 18
top_results_bottom = top_results_top + 420
draw.rounded_rectangle([28, top_results_top, canvas.width-28, top_results_bottom],
                       radius=card_radius, fill=card_fill, outline=card_border, width=1)

# Performers small card (separator-style single item row area)
performers_top = top_results_bottom + 40
performers_bottom = performers_top + 140
draw.rounded_rectangle([28, performers_top, canvas.width-28, performers_bottom],
                       radius=card_radius, fill=card_fill, outline=card_border, width=1)

# Events card (list)
events_top = performers_bottom + 40
events_bottom = events_top + 440
draw.rounded_rectangle([28, events_top, canvas.width-28, events_bottom],
                       radius=card_radius, fill=card_fill, outline=card_border, width=1)

# Recent searches card
recent_top = events_bottom + 40
recent_bottom = recent_top + 540
draw.rounded_rectangle([28, recent_top, canvas.width-28, recent_bottom],
                       radius=card_radius, fill=card_fill, outline=card_border, width=1)

# Separator lines between list rows inside cards (subtle)
sep_color = (243, 243, 243)
# Inside Top results card: draw separators for item rows (approx positions)
for y in (top_results_top + 110, top_results_top + 220, top_results_top + 330):
    draw.line([(40, y), (canvas.width-40, y)], fill=sep_color, width=1)

# Inside Events card: separators for three event rows
for y in (events_top + 120, events_top + 240, events_top + 360):
    draw.line([(40, y), (canvas.width-40, y)], fill=sep_color, width=1)

# Inside Recent searches card: faint separators for rows (vertical spacing)
for i in range(1,6):
    y = recent_top + i*80
    draw.line([(40, y), (canvas.width-40, y)], fill=sep_color, width=1)

# Thin section headings separators (to emphasize headings area)
heading_sep_color = (230, 230, 230)
draw.line([(28, top_results_top - 8), (canvas.width-28, top_results_top - 8)], fill=heading_sep_color, width=1)
draw.line([(28, performers_top - 8), (canvas.width-28, performers_top - 8)], fill=heading_sep_color, width=1)
draw.line([(28, events_top - 8), (canvas.width-28, events_top - 8)], fill=heading_sep_color, width=1)
draw.line([(28, recent_top - 8), (canvas.width-28, recent_top - 8)], fill=heading_sep_color, width=1)

# Bottom navigation bar background and top divider
nav_top = 2792
draw.rectangle([0, nav_top, canvas.width, canvas.height], fill=(255,255,255))
draw.line([(0, nav_top), (canvas.width, nav_top)], fill=(230,230,230), width=1)

# Slight shadow under some cards to suggest elevation (soft, single line)
shadow_color = (245,245,245)
draw.line([(28, top_results_bottom+1), (canvas.width-28, top_results_bottom+1)], fill=shadow_color, width=2)
draw.line([(28, events_bottom+1), (canvas.width-28, events_bottom+1)], fill=shadow_color, width=2)
draw.line([(28, recent_bottom+1), (canvas.width-28, recent_bottom+1)], fill=shadow_color, width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/00_icon_Boston_MA.png
try:
    _c0 = get_crop(0, 1440, 179)
    canvas.paste(_c0, (0, 1605), _c0)
except Exception:
    pass
layout["Boston,_MA"] = [0, 1605, 1440, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 59, 62)
    canvas.paste(_c1, (244, 3), _c1)
except Exception:
    pass
layout["icon_1"] = [244, 3, 303, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 43, 70)
    canvas.paste(_c2, (1155, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1155, 0, 1198, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/03_icon_Boston_Red_Sox.png
try:
    _c3 = get_crop(3, 1440, 179)
    canvas.paste(_c3, (0, 1217), _c3)
except Exception:
    pass
layout["Boston_Red_Sox"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/04_icon_Boston_Red_Sox_at_Cleveland_Guardians.png
try:
    _c4 = get_crop(4, 1440, 179)
    canvas.paste(_c4, (0, 1784), _c4)
except Exception:
    pass
layout["Boston_Red_Sox_at_Clevela"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/05_icon_8.35_my.png
try:
    _c5 = get_crop(5, 168, 144)
    canvas.paste(_c5, (48, 120), _c5)
except Exception:
    pass
layout["8.35_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/06_icon_Chicago_Cubs_at_Boston_Red_Sox.png
try:
    _c6 = get_crop(6, 1440, 179)
    canvas.paste(_c6, (0, 471), _c6)
except Exception:
    pass
layout["Chicago_Cubs_at_Boston_Re"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 92, 69)
    canvas.paste(_c7, (1219, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1219, 0, 1311, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/08_icon_Boston_Red_Sox_at_Cleveland_Guardians.png
try:
    _c8 = get_crop(8, 1440, 179)
    canvas.paste(_c8, (0, 829), _c8)
except Exception:
    pass
layout["Boston_Red_Sox_at_Clevela"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/09_icon_Boston_MA.png
try:
    _c9 = get_crop(9, 1440, 179)
    canvas.paste(_c9, (0, 1963), _c9)
except Exception:
    pass
layout["Boston,_MA"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 54, 61)
    canvas.paste(_c10, (315, 3), _c10)
except Exception:
    pass
layout["icon_10"] = [315, 3, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/11_icon_Account.png
try:
    _c11 = get_crop(11, 288, 168)
    canvas.paste(_c11, (1152, 2792), _c11)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/12_icon_Oracle_Arena.png
try:
    _c12 = get_crop(12, 288, 162)
    canvas.paste(_c12, (288, 2792), _c12)
except Exception:
    pass
layout["Oracle_Arena"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/13_icon_Boston_MA.png
try:
    _c13 = get_crop(13, 1440, 179)
    canvas.paste(_c13, (0, 650), _c13)
except Exception:
    pass
layout["Boston,_MA"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/14_icon_8.35_my.png
try:
    _c14 = get_crop(14, 44, 61)
    canvas.paste(_c14, (187, 2), _c14)
except Exception:
    pass
layout["8.35_my"] = [187, 2, 231, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/15_icon_Tracking.png
try:
    _c15 = get_crop(15, 288, 168)
    canvas.paste(_c15, (864, 2792), _c15)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/16_icon_Clear.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 120), _c16)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 45, 66)
    canvas.paste(_c17, (1326, 2), _c17)
except Exception:
    pass
layout["icon_17"] = [1326, 2, 1371, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/18_icon_Oracle_Arena.png
try:
    _c18 = get_crop(18, 288, 168)
    canvas.paste(_c18, (0, 2792), _c18)
except Exception:
    pass
layout["Oracle_Arena"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/19_icon_Tickets.png
try:
    _c19 = get_crop(19, 288, 168)
    canvas.paste(_c19, (576, 2792), _c19)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/20_icon_139_events.png
try:
    _c20 = get_crop(20, 1440, 179)
    canvas.paste(_c20, (0, 650), _c20)
except Exception:
    pass
layout["139_events"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/21_icon_Music_Hall.png
try:
    _c21 = get_crop(21, 1440, 168)
    canvas.paste(_c21, (0, 2351), _c21)
except Exception:
    pass
layout["Music_Hall"] = [0, 2351, 1440, 2519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/22_icon_8.35_my.png
try:
    _c22 = get_crop(22, 53, 62)
    canvas.paste(_c22, (116, 1), _c22)
except Exception:
    pass
layout["8.35_my"] = [116, 1, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/23_icon_Tomorrow.png
try:
    _c23 = get_crop(23, 1440, 179)
    canvas.paste(_c23, (0, 829), _c23)
except Exception:
    pass
layout["Tomorrow"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/24_icon_Boston_Red_Sox.png
try:
    _c24 = get_crop(24, 1032, 144)
    canvas.paste(_c24, (216, 120), _c24)
except Exception:
    pass
layout["Boston_Red_Sox"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/25_text_Top_results.png
try:
    _c25 = get_crop(25, 295, 72)
    canvas.paste(_c25, (40, 373), _c25)
except Exception:
    pass
layout["Top_results"] = [40, 373, 335, 445]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/26_text_Performers.png
try:
    _c26 = get_crop(26, 293, 54)
    canvas.paste(_c26, (44, 1122), _c26)
except Exception:
    pass
layout["Performers"] = [44, 1122, 337, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/27_text_Events.png
try:
    _c27 = get_crop(27, 177, 54)
    canvas.paste(_c27, (46, 1510), _c27)
except Exception:
    pass
layout["Events"] = [46, 1510, 223, 1564]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/28_text_Recent_searches.png
try:
    _c28 = get_crop(28, 436, 57)
    canvas.paste(_c28, (44, 2257), _c28)
except Exception:
    pass
layout["Recent_searches"] = [44, 2257, 480, 2314]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/29_text_Radio.png
try:
    _c29 = get_crop(29, 131, 49)
    canvas.paste(_c29, (236, 2579), _c29)
except Exception:
    pass
layout["Radio"] = [236, 2579, 367, 2628]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/30_text_Music_Hall.png
try:
    _c30 = get_crop(30, 1440, 168)
    canvas.paste(_c30, (0, 2519), _c30)
except Exception:
    pass
layout["Music_Hall"] = [0, 2519, 1440, 2687]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_10_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-13/31_text_Oracle_Arena.png
try:
    _c31 = get_crop(31, 1440, 105)
    canvas.paste(_c31, (0, 2687), _c31)
except Exception:
    pass
layout["Oracle_Arena"] = [0, 2687, 1440, 2792]
