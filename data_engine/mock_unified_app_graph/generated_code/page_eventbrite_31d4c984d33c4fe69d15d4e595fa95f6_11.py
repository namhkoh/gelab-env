# page_id: page_eventbrite_31d4c984d33c4fe69d15d4e595fa95f6_11
# screenshot: 2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13.png
# step_index: 11/14
# task: Open Eventbrite. Look for 'community events' in 'Chicago'. Select the first event happening tomorrow that is not promoted. Check if they have an option for 'refund policy'.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for Eventbrite-like page
# Available variables: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
bg_color = (255, 255, 255)            # overall white background
status_bar_color = (194, 194, 194)    # status bar grey
header_bg = (255, 255, 255)           # header area white
accent_blue = (30, 103, 255)          # blue accent for underline / dividers
muted_divider = (230, 230, 230)       # light grey dividers
card_bg = (250, 250, 251)             # very light card background
bottom_nav_bg = (255, 255, 255)       # bottom nav white
soft_shadow = (240, 240, 240)

# Clear canvas to background color
draw.rectangle([(0,0),(w,h)], fill=bg_color)

# Status bar (top ~60px)
status_h = 60
draw.rectangle([(0,0),(w,status_h)], fill=status_bar_color)

# Header / search area (below status bar)
header_top = status_h
header_h = 120
header_bottom = header_top + header_h
# white header background (keeps icons/text pasted on top)
draw.rectangle([(0, header_top), (w, header_bottom)], fill=header_bg)

# Thin blue accent underline below the header (represents active search underline)
underline_y = header_bottom - 10
draw.rectangle([(48, underline_y), (w-48, underline_y+6)], fill=accent_blue)

# subtle shadow line under header
draw.rectangle([(0, header_bottom), (w, header_bottom+1)], fill=muted_divider)

# Event list area: draw subtle rounded card backgrounds for each event grouping row
# Rows as detected (y positions from detected elements); each has height ~396, x inset 48
row_x0 = 48
row_w = 1344
row_h = 396
row_radius = 6

row_ys = [390, 786, 1182, 1578, 1974, 2370]
for y in row_ys:
    x0 = row_x0
    y0 = y
    x1 = x0 + row_w
    y1 = y0 + row_h
    # Draw a very subtle card background with rounded corners
    draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=row_radius, fill=card_bg, outline=None)
    # inner top separator shadow for depth
    draw.line([(x0+8, y0+1), (x1-8, y0+1)], fill=soft_shadow, width=1)
    # inner bottom separator subtle
    draw.line([(x0+8, y1-1), (x1-8, y1-1)], fill=muted_divider, width=1)

# Draw separators between rows across full content width (light grey)
for i in range(len(row_ys)-1):
    sep_y = row_ys[i] + row_h + 18  # small gap between cards
    if sep_y < h - 200:
        draw.line([(48, sep_y), (w-48, sep_y)], fill=muted_divider, width=1)

# Large content area background (main content column already white; add subtle center column guide)
center_x0 = 48
center_x1 = w - 48
draw.rectangle([(center_x0, header_bottom+8), (center_x1, h - 200)], outline=None, fill=None)

# Bottom navigation bar background and top divider
bottom_nav_h = 156
bottom_nav_top = h - bottom_nav_h
draw.rectangle([(0, bottom_nav_top), (w, h)], fill=bottom_nav_bg)
# Top divider of bottom nav
draw.rectangle([(0, bottom_nav_top), (w, bottom_nav_top+1)], fill=muted_divider)

# Subtle rounded top corners for bottom nav to separate it visually
draw.line([(24, bottom_nav_top), (w-24, bottom_nav_top)], fill=muted_divider, width=1)

# Final subtle outer frame / safe-area guide (very faint)
draw.rectangle([(8, status_h+8), (w-8, h-bottom_nav_h-8)], outline=(245,245,245))

# Note: No icons, text, thumbnails, or buttons are drawn — only backgrounds, bars, cards, and separators.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/00_icon_Ise_of_an_Automate.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 390), _c0)
except Exception:
    pass
layout["Ise_of_an_Automate"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/01_icon_2911_E_Z9th_St.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 2370), _c1)
except Exception:
    pass
layout["2911_E_Z9th_St"] = [48, 2370, 1392, 2766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/02_icon_Sat_Apr_27.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1974), _c2)
except Exception:
    pass
layout["Sat,_Apr_27"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/03_icon_community_eventsl.png
try:
    _c3 = get_crop(3, 1344, 191)
    canvas.paste(_c3, (48, 72), _c3)
except Exception:
    pass
layout["community_eventsl"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/04_icon_Gathering_of_Vibrations_A.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 1974), _c4)
except Exception:
    pass
layout["Gathering_of_Vibrations:_"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 56)
    canvas.paste(_c5, (316, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [316, 6, 366, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 41, 54)
    canvas.paste(_c6, (254, 7), _c6)
except Exception:
    pass
layout["icon_6"] = [254, 7, 295, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/07_icon_8.07.png
try:
    _c7 = get_crop(7, 51, 60)
    canvas.paste(_c7, (185, 3), _c7)
except Exception:
    pass
layout["8.07"] = [185, 3, 236, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/08_icon_8.07.png
try:
    _c8 = get_crop(8, 58, 61)
    canvas.paste(_c8, (113, 3), _c8)
except Exception:
    pass
layout["8.07"] = [113, 3, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/09_icon_Sat_Apr_27.png
try:
    _c9 = get_crop(9, 1344, 396)
    canvas.paste(_c9, (48, 1578), _c9)
except Exception:
    pass
layout["Sat,_Apr_27"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 42, 66)
    canvas.paste(_c10, (1158, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1158, 1, 1200, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/11_icon_Cancel.png
try:
    _c11 = get_crop(11, 90, 64)
    canvas.paste(_c11, (1216, 1), _c11)
except Exception:
    pass
layout["Cancel"] = [1216, 1, 1306, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/12_icon_Sun.png
try:
    _c12 = get_crop(12, 1344, 396)
    canvas.paste(_c12, (48, 1182), _c12)
except Exception:
    pass
layout["Sun,"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/13_icon_Cancel.png
try:
    _c13 = get_crop(13, 50, 63)
    canvas.paste(_c13, (1320, 1), _c13)
except Exception:
    pass
layout["Cancel"] = [1320, 1, 1370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/14_icon_8.07.png
try:
    _c14 = get_crop(14, 124, 105)
    canvas.paste(_c14, (51, 119), _c14)
except Exception:
    pass
layout["8.07"] = [51, 119, 175, 224]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/15_icon_ATIH_Community_Event_Chicago.png
try:
    _c15 = get_crop(15, 1344, 396)
    canvas.paste(_c15, (48, 786), _c15)
except Exception:
    pass
layout["ATIH_Community_Event:_Chi"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 46, 54)
    canvas.paste(_c16, (384, 7), _c16)
except Exception:
    pass
layout["icon_16"] = [384, 7, 430, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/17_icon_Community_Farm_Kick_Off_Event.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 1578), _c17)
except Exception:
    pass
layout["Community_Farm_Kick_Off_E"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/18_icon_Cancel.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1099, 96), _c18)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/19_icon_6_00_PM_CDT.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 786), _c19)
except Exception:
    pass
layout["6:00_PM_CDT"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/20_icon_Choir_and_Solo_Artists_for_Events.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 1182), _c20)
except Exception:
    pass
layout["Choir_and_Solo_Artists_fo"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/21_icon_Gathering_of_Vibrations_A.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 2370), _c21)
except Exception:
    pass
layout["Gathering_of_Vibrations:_"] = [48, 2370, 1392, 2766]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/22_icon_First_Aid_and_Manual_Handling_for.png
try:
    _c22 = get_crop(22, 1344, 396)
    canvas.paste(_c22, (48, 390), _c22)
except Exception:
    pass
layout["First_Aid_and_Manual_Hand"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/23_icon_requlated.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 786), _c23)
except Exception:
    pass
layout["requlated"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/24_icon_The_Fromus_Centre.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 390), _c24)
except Exception:
    pass
layout["The_Fromus_Centre"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/25_icon_Cancel.png
try:
    _c25 = get_crop(25, 149, 144)
    canvas.paste(_c25, (1243, 97), _c25)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/26_icon_Just_Roots_Presents_Saint_James.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 1578), _c26)
except Exception:
    pass
layout["Just_Roots_Presents:_Sain"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/27_icon_Young_Adult_Singers_Wanted_Community.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 1182), _c27)
except Exception:
    pass
layout["Young_Adult_Singers_Wante"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/28_icon_ATIH_Community_Event_Chicago.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 786), _c28)
except Exception:
    pass
layout["ATIH_Community_Event:_Chi"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/29_text_8.07.png
try:
    _c29 = get_crop(29, 89, 43)
    canvas.paste(_c29, (20, 17), _c29)
except Exception:
    pass
layout["8.07"] = [20, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/30_text_Events.png
try:
    _c30 = get_crop(30, 186, 56)
    canvas.paste(_c30, (46, 301), _c30)
except Exception:
    pass
layout["Events"] = [46, 301, 232, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/31_text_Tue_Apr_23.png
try:
    _c31 = get_crop(31, 200, 43)
    canvas.paste(_c31, (390, 448), _c31)
except Exception:
    pass
layout["Tue,_Apr_23"] = [390, 448, 590, 491]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/32_text_9_00_AM_GMT_01_00.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 390), _c32)
except Exception:
    pass
layout["9:00_AM_GMT+01:00"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/33_text_22_creator_followers.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 390), _c33)
except Exception:
    pass
layout["22_creator_followers"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/34_clickable_Home.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (0, 2804), _c34)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/35_clickable_Search_events.png
try:
    _c35 = get_crop(35, 288, 156)
    canvas.paste(_c35, (288, 2804), _c35)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/36_clickable_Favorites.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (576, 2804), _c36)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/37_clickable_Tickets.png
try:
    _c37 = get_crop(37, 288, 156)
    canvas.paste(_c37, (864, 2804), _c37)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_11_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-13/38_clickable_More.png
try:
    _c38 = get_crop(38, 288, 156)
    canvas.paste(_c38, (1152, 2804), _c38)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
