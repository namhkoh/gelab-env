# page_id: page_seatgeek_49ceba1342134bb89f14fac27abc2dcd_04
# screenshot: 2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7.png
# step_index: 4/12
# task: Open SeatGeek. Track "New York Yankees", "Boston Red Sox".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill (dominant light color)
draw.rectangle((0, 0, 1440, 2960), fill=(250, 250, 250))

# Status bar area at top (~56-72px)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(242, 242, 242))

# Thin divider under status bar
draw.line((0, status_h - 1, 1440, status_h - 1), fill=(230, 230, 230), width=1)

# Header / search bar background (rounded, leave space for icons/text to be pasted)
search_margin = 48
search_y1 = 84
search_y2 = 176
draw.rounded_rectangle(
    (search_margin, search_y1, 1440 - search_margin, search_y2),
    radius=28,
    fill=(255, 255, 255),
    outline=(230, 230, 230),
    width=1
)

# Subtle shadow / separator under the header area
draw.line((search_margin, search_y2 + 18, 1440 - search_margin, search_y2 + 18), fill=(240, 240, 240), width=1)

# Main content panel backgrounds (soft white cards to group sections)
# Panels are intentionally large and neutral so pasted content (icons/text) sits on top
panel_padding_left = 40
panel_padding_right = 40
panels = [
    # Top results block
    (panel_padding_left, 320, 1440 - panel_padding_right, 670),
    # Performers block
    (panel_padding_left, 700, 1440 - panel_padding_right, 1100),
    # Events block (taller)
    (panel_padding_left, 1120, 1440 - panel_padding_right, 1740),
    # Venues / Recent searches block
    (panel_padding_left, 1760, 1440 - panel_padding_right, 2360),
]
for (x1, y1, x2, y2) in panels:
    draw.rounded_rectangle((x1, y1, x2, y2), radius=12, fill=(255, 255, 255))

# Horizontal separators between logical sections (subtle, inset from sides)
separator_x1 = 48
separator_x2 = 1440 - 48
separators = [200, 460, 640, 820, 1000, 1200, 1380, 1605, 1784, 1963, 2140, 2351, 2580, 2792]
for y in separators:
    draw.line((separator_x1, y, separator_x2, y), fill=(235, 235, 235), width=1)

# Internal faint row separators inside panels (very light)
for y in range(420, 1700, 180):
    draw.line((80, y, 1360, y), fill=(245, 245, 245), width=1)

# Bottom navigation bar background and top divider
nav_top = 2792
draw.line((0, nav_top, 1440, nav_top), fill=(230, 230, 230), width=2)
draw.rectangle((0, nav_top, 1440, 2960), fill=(255, 255, 255))

# Slight vertical guideline (subtle) to imply content padding (non-intrusive)
draw.line((48, 90, 48, 2700), fill=(250, 250, 250), width=1)

# Final subtle highlight under status/header for depth
draw.rectangle((0, status_h, 1440, status_h + 4), fill=(220, 220, 220))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/00_icon_Oakland_Athletics_at_New_York_Yankees.png
try:
    _c0 = get_crop(0, 1440, 179)
    canvas.paste(_c0, (0, 650), _c0)
except Exception:
    pass
layout["Oakland_Athletics_at_New_"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/01_icon_Today.png
try:
    _c1 = get_crop(1, 1440, 179)
    canvas.paste(_c1, (0, 829), _c1)
except Exception:
    pass
layout["Today"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 59, 61)
    canvas.paste(_c2, (244, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [244, 3, 303, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/03_icon_New_York_Yankeest.png
try:
    _c3 = get_crop(3, 1032, 144)
    canvas.paste(_c3, (216, 120), _c3)
except Exception:
    pass
layout["New_York_Yankeest"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/04_icon_Bronx_NY.png
try:
    _c4 = get_crop(4, 1440, 179)
    canvas.paste(_c4, (0, 1605), _c4)
except Exception:
    pass
layout["Bronx,_NY"] = [0, 1605, 1440, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 42, 70)
    canvas.paste(_c5, (1156, 0), _c5)
except Exception:
    pass
layout["icon_5"] = [1156, 0, 1198, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/06_icon_8.35_my.png
try:
    _c6 = get_crop(6, 168, 144)
    canvas.paste(_c6, (48, 120), _c6)
except Exception:
    pass
layout["8.35_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/07_icon_New_York_Yankees.png
try:
    _c7 = get_crop(7, 1440, 179)
    canvas.paste(_c7, (0, 1217), _c7)
except Exception:
    pass
layout["New_York_Yankees"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 53, 61)
    canvas.paste(_c8, (315, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [315, 3, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/09_icon_Recent_searches.png
try:
    _c9 = get_crop(9, 288, 162)
    canvas.paste(_c9, (288, 2792), _c9)
except Exception:
    pass
layout["Recent_searches"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/10_icon_Today.png
try:
    _c10 = get_crop(10, 1440, 179)
    canvas.paste(_c10, (0, 1963), _c10)
except Exception:
    pass
layout["Today"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/11_icon_Events.png
try:
    _c11 = get_crop(11, 1440, 179)
    canvas.paste(_c11, (0, 1605), _c11)
except Exception:
    pass
layout["Events"] = [0, 1605, 1440, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/12_icon_New_York_Yankees.png
try:
    _c12 = get_crop(12, 1440, 179)
    canvas.paste(_c12, (0, 471), _c12)
except Exception:
    pass
layout["New_York_Yankees"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/13_icon_Tomorrow.png
try:
    _c13 = get_crop(13, 1440, 179)
    canvas.paste(_c13, (0, 1784), _c13)
except Exception:
    pass
layout["Tomorrow"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 93, 69)
    canvas.paste(_c14, (1219, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1219, 0, 1312, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/15_icon_Bronx_NY.png
try:
    _c15 = get_crop(15, 1440, 179)
    canvas.paste(_c15, (0, 1784), _c15)
except Exception:
    pass
layout["Bronx,_NY"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/16_icon_8.35_my.png
try:
    _c16 = get_crop(16, 44, 61)
    canvas.paste(_c16, (187, 2), _c16)
except Exception:
    pass
layout["8.35_my"] = [187, 2, 231, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 45, 66)
    canvas.paste(_c17, (1326, 2), _c17)
except Exception:
    pass
layout["icon_17"] = [1326, 2, 1371, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/18_icon_Clear.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 120), _c18)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/19_icon_Bronx_NY.png
try:
    _c19 = get_crop(19, 1440, 179)
    canvas.paste(_c19, (0, 1963), _c19)
except Exception:
    pass
layout["Bronx,_NY"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/20_icon_Yankee_Stadium.png
try:
    _c20 = get_crop(20, 1440, 179)
    canvas.paste(_c20, (0, 2351), _c20)
except Exception:
    pass
layout["Yankee_Stadium"] = [0, 2351, 1440, 2530]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/21_icon_Account.png
try:
    _c21 = get_crop(21, 288, 168)
    canvas.paste(_c21, (1152, 2792), _c21)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/22_icon_Recent_searches.png
try:
    _c22 = get_crop(22, 288, 168)
    canvas.paste(_c22, (0, 2792), _c22)
except Exception:
    pass
layout["Recent_searches"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/23_icon_8.35_my.png
try:
    _c23 = get_crop(23, 53, 62)
    canvas.paste(_c23, (116, 1), _c23)
except Exception:
    pass
layout["8.35_my"] = [116, 1, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/24_icon_Tickets.png
try:
    _c24 = get_crop(24, 288, 168)
    canvas.paste(_c24, (576, 2792), _c24)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/25_icon_Tracking.png
try:
    _c25 = get_crop(25, 288, 168)
    canvas.paste(_c25, (864, 2792), _c25)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/26_icon_Performers.png
try:
    _c26 = get_crop(26, 1440, 179)
    canvas.paste(_c26, (0, 1217), _c26)
except Exception:
    pass
layout["Performers"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/27_icon_Top.png
try:
    _c27 = get_crop(27, 1440, 179)
    canvas.paste(_c27, (0, 471), _c27)
except Exception:
    pass
layout["Top"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/28_text_results.png
try:
    _c28 = get_crop(28, 186, 54)
    canvas.paste(_c28, (146, 377), _c28)
except Exception:
    pass
layout["results"] = [146, 377, 332, 431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/29_text_Performers.png
try:
    _c29 = get_crop(29, 293, 54)
    canvas.paste(_c29, (44, 1122), _c29)
except Exception:
    pass
layout["Performers"] = [44, 1122, 337, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/30_text_Events.png
try:
    _c30 = get_crop(30, 177, 54)
    canvas.paste(_c30, (46, 1510), _c30)
except Exception:
    pass
layout["Events"] = [46, 1510, 223, 1564]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/31_text_Venues.png
try:
    _c31 = get_crop(31, 197, 60)
    canvas.paste(_c31, (42, 2253), _c31)
except Exception:
    pass
layout["Venues"] = [42, 2253, 239, 2313]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_04_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-7/32_text_Recent_searches.png
try:
    _c32 = get_crop(32, 288, 168)
    canvas.paste(_c32, (0, 2792), _c32)
except Exception:
    pass
layout["Recent_searches"] = [0, 2792, 288, 2960]
