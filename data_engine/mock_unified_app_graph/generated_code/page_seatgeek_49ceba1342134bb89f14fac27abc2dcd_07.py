# page_id: page_seatgeek_49ceba1342134bb89f14fac27abc2dcd_07
# screenshot: 2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10.png
# step_index: 7/12
# task: Open SeatGeek. Track "New York Yankees", "Boston Red Sox".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the provided canvas
# Available: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
bg_color = (250, 250, 250)        # overall app background (very light gray)
status_bar_color = (235, 235, 235)  # top status bar
content_bg = (255, 255, 255)      # main content background (white)
search_bg = (242, 242, 242)       # search field background
divider = (230, 230, 230)         # subtle dividers
muted_shadow = (245, 245, 245)    # slight shadow area
nav_bg = (255, 255, 255)          # bottom nav background

W, H = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (W, H)], fill=bg_color)

# Status bar area (top)
status_h = 68
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# Main content panel (below status bar, ends above bottom nav)
bottom_nav_top = 2792
draw.rectangle([(0, status_h), (W, bottom_nav_top)], fill=content_bg)

# Search bar background (rounded)
search_left = 48
search_top = 120
search_right = W - 48
search_height = 144
search_box = (search_left, search_top, search_right, search_top + search_height)
draw.rounded_rectangle(search_box, radius=20, fill=search_bg, outline=(220, 220, 220), width=1)

# Divider immediately under the search bar
draw.line([(search_left, search_top + search_height), (search_right, search_top + search_height)], fill=divider, width=1)

# Major horizontal separators (approximate positions based on detected elements)
separator_x1 = 48
separator_x2 = W - 48
separator_positions = [
    650,   # after first block of results
    1008,  # after top results / before performers
    1370,  # between performer and events area
    1784,  # mid-list separator
    1963,  # another section divider
    2351,  # venues section divider
]

for y in separator_positions:
    draw.line([(separator_x1, y), (separator_x2, y)], fill=divider, width=1)

# Add faint wide dividers for grouping further up (e.g., under headings)
draw.line([(32, 264), (W - 32, 264)], fill=divider, width=1)    # below search bar area / thin rule

# Slight band/shadow above the bottom nav to separate content from nav
draw.rectangle([(0, bottom_nav_top - 6), (W, bottom_nav_top)], fill=muted_shadow)

# Bottom navigation bar background and top divider
draw.rectangle([(0, bottom_nav_top), (W, H)], fill=nav_bg)
draw.line([(0, bottom_nav_top), (W, bottom_nav_top)], fill=divider, width=1)

# Subtle rounded "section cards" behind groups to visually separate blocks (very light)
# Top results card
card_margin = 28
draw.rounded_rectangle(
    [(card_margin, 324), (W - card_margin, 840)],
    radius=12,
    fill=(255, 255, 255),
    outline=None
)

# Performers small card
draw.rounded_rectangle(
    [(card_margin, 1080), (W - card_margin, 1224)],
    radius=10,
    fill=(255, 255, 255),
    outline=None
)

# Events grouping card (three items area)
draw.rounded_rectangle(
    [(card_margin, 1440), (W - card_margin, 2016)],
    radius=10,
    fill=(255, 255, 255),
    outline=None
)

# Venues card
draw.rounded_rectangle(
    [(card_margin, 2208), (W - card_margin, 2388)],
    radius=10,
    fill=(255, 255, 255),
    outline=None
)

# Recent searches background area (slightly different tint)
recent_top = 2688
draw.rectangle([(0, recent_top), (W, bottom_nav_top)], fill=(250, 250, 250))

# final subtle top shadow on content (under status bar) to create separation
draw.line([(0, status_h), (W, status_h)], fill=(225, 225, 225), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/00_icon_Today.png
try:
    _c0 = get_crop(0, 1440, 179)
    canvas.paste(_c0, (0, 829), _c0)
except Exception:
    pass
layout["Today"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/01_icon_Bronx_NY.png
try:
    _c1 = get_crop(1, 1440, 179)
    canvas.paste(_c1, (0, 1605), _c1)
except Exception:
    pass
layout["Bronx,_NY"] = [0, 1605, 1440, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/02_icon_Oakland_Athletics_at_New_York_Yankees.png
try:
    _c2 = get_crop(2, 1440, 179)
    canvas.paste(_c2, (0, 650), _c2)
except Exception:
    pass
layout["Oakland_Athletics_at_New_"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 62, 63)
    canvas.paste(_c3, (243, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [243, 2, 305, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/04_icon_8.35_my.png
try:
    _c4 = get_crop(4, 53, 62)
    canvas.paste(_c4, (116, 1), _c4)
except Exception:
    pass
layout["8.35_my"] = [116, 1, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 43, 70)
    canvas.paste(_c5, (1155, 0), _c5)
except Exception:
    pass
layout["icon_5"] = [1155, 0, 1198, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/06_icon_Bronx_NY.png
try:
    _c6 = get_crop(6, 1440, 179)
    canvas.paste(_c6, (0, 1784), _c6)
except Exception:
    pass
layout["Bronx,_NY"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/07_icon_New_York_Yankees.png
try:
    _c7 = get_crop(7, 1440, 179)
    canvas.paste(_c7, (0, 1217), _c7)
except Exception:
    pass
layout["New_York_Yankees"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 54, 60)
    canvas.paste(_c8, (315, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [315, 3, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/09_icon_Tomorrow.png
try:
    _c9 = get_crop(9, 1440, 179)
    canvas.paste(_c9, (0, 1784), _c9)
except Exception:
    pass
layout["Tomorrow"] = [0, 1784, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 93, 69)
    canvas.paste(_c10, (1219, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1219, 0, 1312, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/11_icon_8.35_my.png
try:
    _c11 = get_crop(11, 46, 63)
    canvas.paste(_c11, (186, 1), _c11)
except Exception:
    pass
layout["8.35_my"] = [186, 1, 232, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/12_icon_Recent_searches.png
try:
    _c12 = get_crop(12, 288, 162)
    canvas.paste(_c12, (288, 2792), _c12)
except Exception:
    pass
layout["Recent_searches"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/13_icon_Today.png
try:
    _c13 = get_crop(13, 1440, 179)
    canvas.paste(_c13, (0, 1963), _c13)
except Exception:
    pass
layout["Today"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/14_icon_Events.png
try:
    _c14 = get_crop(14, 1440, 179)
    canvas.paste(_c14, (0, 1605), _c14)
except Exception:
    pass
layout["Events"] = [0, 1605, 1440, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/15_icon_New_York_Yankees.png
try:
    _c15 = get_crop(15, 1440, 179)
    canvas.paste(_c15, (0, 471), _c15)
except Exception:
    pass
layout["New_York_Yankees"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/16_icon_Clear.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 120), _c16)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/17_icon_Bronx_NY.png
try:
    _c17 = get_crop(17, 1440, 179)
    canvas.paste(_c17, (0, 1963), _c17)
except Exception:
    pass
layout["Bronx,_NY"] = [0, 1963, 1440, 2142]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 44, 67)
    canvas.paste(_c18, (1326, 2), _c18)
except Exception:
    pass
layout["icon_18"] = [1326, 2, 1370, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/19_icon_Recent_searches.png
try:
    _c19 = get_crop(19, 288, 168)
    canvas.paste(_c19, (0, 2792), _c19)
except Exception:
    pass
layout["Recent_searches"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/20_icon_New_York_Yankees.png
try:
    _c20 = get_crop(20, 1032, 144)
    canvas.paste(_c20, (216, 120), _c20)
except Exception:
    pass
layout["New_York_Yankees"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/21_icon_Top.png
try:
    _c21 = get_crop(21, 1440, 179)
    canvas.paste(_c21, (0, 471), _c21)
except Exception:
    pass
layout["Top"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/22_icon_Yankee_Stadium.png
try:
    _c22 = get_crop(22, 1440, 179)
    canvas.paste(_c22, (0, 2351), _c22)
except Exception:
    pass
layout["Yankee_Stadium"] = [0, 2351, 1440, 2530]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/23_icon_Account.png
try:
    _c23 = get_crop(23, 288, 168)
    canvas.paste(_c23, (1152, 2792), _c23)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/24_icon_8.35_my.png
try:
    _c24 = get_crop(24, 168, 144)
    canvas.paste(_c24, (48, 120), _c24)
except Exception:
    pass
layout["8.35_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/25_icon_Performers.png
try:
    _c25 = get_crop(25, 1440, 179)
    canvas.paste(_c25, (0, 1217), _c25)
except Exception:
    pass
layout["Performers"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/26_icon_Tickets.png
try:
    _c26 = get_crop(26, 288, 168)
    canvas.paste(_c26, (576, 2792), _c26)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/27_icon_Tracking.png
try:
    _c27 = get_crop(27, 288, 168)
    canvas.paste(_c27, (864, 2792), _c27)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/28_text_results.png
try:
    _c28 = get_crop(28, 186, 54)
    canvas.paste(_c28, (146, 377), _c28)
except Exception:
    pass
layout["results"] = [146, 377, 332, 431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/29_text_Performers.png
try:
    _c29 = get_crop(29, 293, 54)
    canvas.paste(_c29, (44, 1122), _c29)
except Exception:
    pass
layout["Performers"] = [44, 1122, 337, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/30_text_Events.png
try:
    _c30 = get_crop(30, 177, 54)
    canvas.paste(_c30, (46, 1510), _c30)
except Exception:
    pass
layout["Events"] = [46, 1510, 223, 1564]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/31_text_Venues.png
try:
    _c31 = get_crop(31, 197, 60)
    canvas.paste(_c31, (42, 2253), _c31)
except Exception:
    pass
layout["Venues"] = [42, 2253, 239, 2313]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/49ceba1342134bb89f14fac27abc2dcd/step_07_2024_4_22_20_34_49ceba1342134bb89f14fac27abc2dcd-10/32_text_Recent_searches.png
try:
    _c32 = get_crop(32, 288, 168)
    canvas.paste(_c32, (0, 2792), _c32)
except Exception:
    pass
layout["Recent_searches"] = [0, 2792, 288, 2960]
