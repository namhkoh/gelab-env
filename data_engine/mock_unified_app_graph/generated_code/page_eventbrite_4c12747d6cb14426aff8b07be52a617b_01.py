# page_id: page_eventbrite_4c12747d6cb14426aff8b07be52a617b_01
# screenshot: 2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3.png
# step_index: 1/11
# task: Open Eventbrite. Search 'Art'. Filter event type "Performance". Select the first event. Follow the organizer and save the event to favorite. What is the price of the ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (canvas is already white, but ensure dominant subtle tint)
bg_color = (255, 255, 255)
draw.rectangle([(0, 0), canvas.size], fill=bg_color)

# Status bar (top)
status_h = 76
status_color = (199, 199, 199)  # light gray status bar
draw.rectangle([(0, 0), (canvas.width, status_h)], fill=status_color)

# Subtle divider below status bar
draw.line([(0, status_h), (canvas.width, status_h)], fill=(210,210,210), width=1)

# Header area (space that contains the search bar) - keep background clean and add subtle shadow line
header_top = status_h
header_bottom = 220
header_bg = (255, 255, 255)
draw.rectangle([(0, header_top), (canvas.width, header_bottom)], fill=header_bg)
draw.line([(0, header_bottom), (canvas.width, header_bottom)], fill=(235,235,238), width=1)

# Large subtle rounded background behind the search area (but do not draw the actual search content)
search_bg_box = (172, 88, 1268, 180)  # behind the search bar area
draw.rounded_rectangle(search_bg_box, radius=32, fill=(250,250,251), outline=(240,240,243), width=1)

# Main content area: draw card backgrounds for event list entries.
# Use detected card positions/sizes from the UI crop list (x=48, widths=1344)
cards = [
    (48, 490, 48+1344, 490+396),
    (48, 886, 48+1344, 886+396),
    (48, 1282, 48+1344, 1282+396),
    (48, 1678, 48+1344, 1678+396),
    (48, 2074, 48+1344, 2074+396),
    (48, 2470, 48+1344, 2470+346),
    (0, 2804, 288, 2804+156),   # leftmost bottom clickable area - keep nav items free but draw underlying bar separately
    (288, 2804, 576, 2804+156),
    (576, 2804, 864, 2804+156),
    (864, 2804, 1152, 2804+156),
    (1152, 2804, 1440, 2804+156)
]

card_fill = (255, 255, 255)
card_edge = (243, 243, 246)
shadow_color = (245, 246, 248)

# Draw subtle drop shadows and rounded card backgrounds for the main list (except explicit bottom nav pieces)
for box in cards[:6]:
    x0, y0, x1, y1 = box
    # shadow
    shadow_offset = 6
    draw.rounded_rectangle([x0+shadow_offset, y0+shadow_offset, x1+shadow_offset, y1+shadow_offset],
                           radius=14, fill=shadow_color, outline=None)
    # card
    draw.rounded_rectangle([x0, y0, x1, y1], radius=12, fill=card_fill, outline=card_edge, width=1)

# Separator thin lines between cards (subtle)
for box in cards[:6]:
    x0, y0, x1, y1 = box
    # draw a faint horizontal separator near the bottom edge to visually separate items
    sep_y = y1 + 8
    draw.line([(x0+8, sep_y), (x1-8, sep_y)], fill=(240,240,243), width=1)

# Floating content area hint: a light colored banner behind where some image posts appear (do not draw images/text)
# Place one banner roughly behind the "More events you'll love" heading area
banner_box = (48, 340, 1392, 420)
draw.rectangle(banner_box, fill=(255,255,255))  # keep it white but ensure crisp separation
draw.line([(48, 420), (1392, 420)], fill=(240,240,243), width=1)

# Bottom navigation bar background (full-width)
nav_top = 2804
nav_bottom = canvas.height
nav_bg = (255, 255, 255)
draw.rectangle([(0, nav_top), (canvas.width, nav_bottom)], fill=nav_bg)

# Top divider of nav bar
draw.line([(0, nav_top), (canvas.width, nav_top)], fill=(230,230,234), width=1)

# Highlight the selected nav area (leftmost) with subtle tint (do not draw icons)
selected_area = (0, nav_top, 360, nav_bottom)  # slightly wider than one icon column to be a background only
draw.rectangle(selected_area, fill=(255, 247, 240))

# Subtle rounded top for the nav bar (soften corners)
draw.polygon([(0, nav_top), (20, nav_top), (20, nav_top+6), (0, nav_top+6)], fill=nav_bg)
draw.polygon([(canvas.width, nav_top), (canvas.width-20, nav_top), (canvas.width-20, nav_top+6), (canvas.width, nav_top+6)], fill=nav_bg)

# Additional subtle separators for visual grouping near the middle (between header and list)
draw.line([(48, 320), (1392, 320)], fill=(245,245,247), width=1)

# Done drawing background & structure. The detected UI elements will be pasted on top.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/00_icon_ering_to_soothe_the_brokel.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1678), _c0)
except Exception:
    pass
layout["ering_to_soothe_the_broke"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/01_icon_Spring-Zing_Happy_Hour.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 2074), _c1)
except Exception:
    pass
layout["Spring-Zing_Happy_Hour"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/02_icon_NDIE.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 490), _c2)
except Exception:
    pass
layout["NDIE"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/03_icon_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/04_icon_Sat.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 886), _c4)
except Exception:
    pass
layout["Sat,"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/05_icon_San_Francisco.png
try:
    _c5 = get_crop(5, 495, 117)
    canvas.paste(_c5, (473, 2651), _c5)
except Exception:
    pass
layout["San_Francisco"] = [473, 2651, 968, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/06_icon_Spring-Zing_Happy.png
try:
    _c6 = get_crop(6, 144, 123)
    canvas.paste(_c6, (1140, 1951), _c6)
except Exception:
    pass
layout["Spring-Zing_Happy"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 139)
    canvas.paste(_c7, (1140, 747), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/08_icon_City.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 1539), _c8)
except Exception:
    pass
layout["City"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/09_icon_Bissa.png
try:
    _c9 = get_crop(9, 288, 156)
    canvas.paste(_c9, (288, 2804), _c9)
except Exception:
    pass
layout["Bissa}"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/10_icon_Reggaeton.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 2347), _c10)
except Exception:
    pass
layout["Reggaeton__"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/11_icon_RIEF_MEDICIN.png
try:
    _c11 = get_crop(11, 1344, 396)
    canvas.paste(_c11, (48, 1282), _c11)
except Exception:
    pass
layout["RIEF_MEDICIN"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/12_icon_Spring-Zing_Happy.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1284, 1951), _c12)
except Exception:
    pass
layout["Spring-Zing_Happy"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 139)
    canvas.paste(_c13, (1284, 747), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/14_icon_7.51.png
try:
    _c14 = get_crop(14, 108, 101)
    canvas.paste(_c14, (38, 121), _c14)
except Exception:
    pass
layout["7.51"] = [38, 121, 146, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/15_icon_City.png
try:
    _c15 = get_crop(15, 144, 139)
    canvas.paste(_c15, (1284, 1539), _c15)
except Exception:
    pass
layout["City"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/16_icon_Reggaeton.png
try:
    _c16 = get_crop(16, 144, 123)
    canvas.paste(_c16, (1284, 2347), _c16)
except Exception:
    pass
layout["Reggaeton__"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/17_icon_7.51.png
try:
    _c17 = get_crop(17, 54, 60)
    canvas.paste(_c17, (184, 2), _c17)
except Exception:
    pass
layout["7.51"] = [184, 2, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/18_icon_City.png
try:
    _c18 = get_crop(18, 144, 139)
    canvas.paste(_c18, (1140, 1143), _c18)
except Exception:
    pass
layout["City"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/19_icon_SatvaonG.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (0, 2804), _c19)
except Exception:
    pass
layout["SatvaonG"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/20_icon_PDO_Thread_Training.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 1282), _c20)
except Exception:
    pass
layout["PDO_Thread_Training_|"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/21_icon_Overflow_menu_button.png
try:
    _c21 = get_crop(21, 144, 139)
    canvas.paste(_c21, (1284, 1143), _c21)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 58, 57)
    canvas.paste(_c22, (313, 4), _c22)
except Exception:
    pass
layout["icon_22"] = [313, 4, 371, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 47, 58)
    canvas.paste(_c23, (250, 3), _c23)
except Exception:
    pass
layout["icon_23"] = [250, 3, 297, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/24_icon_7.51.png
try:
    _c24 = get_crop(24, 59, 59)
    canvas.paste(_c24, (113, 3), _c24)
except Exception:
    pass
layout["7.51"] = [113, 3, 172, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 47, 53)
    canvas.paste(_c25, (1321, 7), _c25)
except Exception:
    pass
layout["icon_25"] = [1321, 7, 1368, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/26_icon_8_29_creator_followers.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 886), _c26)
except Exception:
    pass
layout["8_29_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/27_icon_59_creator_followers.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 490), _c27)
except Exception:
    pass
layout["59_creator_followers"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 56, 57)
    canvas.paste(_c28, (1213, 5), _c28)
except Exception:
    pass
layout["icon_28"] = [1213, 5, 1269, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/29_icon_8_100_creator_followers.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 1678), _c29)
except Exception:
    pass
layout["8_100_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/30_icon_Free.png
try:
    _c30 = get_crop(30, 125, 73)
    canvas.paste(_c30, (248, 561), _c30)
except Exception:
    pass
layout["Free"] = [248, 561, 373, 634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/31_icon_Area.png
try:
    _c31 = get_crop(31, 1344, 396)
    canvas.paste(_c31, (48, 2074), _c31)
except Exception:
    pass
layout["Area"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/32_icon_icon_32.png
try:
    _c32 = get_crop(32, 41, 55)
    canvas.paste(_c32, (1272, 6), _c32)
except Exception:
    pass
layout["icon_32"] = [1272, 6, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/33_icon_Salsa.png
try:
    _c33 = get_crop(33, 1344, 346)
    canvas.paste(_c33, (48, 2470), _c33)
except Exception:
    pass
layout["Salsa"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/34_icon_Grief_Medicine_A_Gathering_to_Soothe_the.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 1678), _c34)
except Exception:
    pass
layout["Grief_Medicine:_A_Gatheri"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/35_icon_icon_35.png
try:
    _c35 = get_crop(35, 43, 55)
    canvas.paste(_c35, (385, 7), _c35)
except Exception:
    pass
layout["icon_35"] = [385, 7, 428, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/36_icon_Processing_Grief_Self-Care_for_Loss.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 886), _c36)
except Exception:
    pass
layout["Processing_Grief:_Self-Ca"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/37_icon_Yggae.png
try:
    _c37 = get_crop(37, 150, 68)
    canvas.paste(_c37, (933, 2643), _c37)
except Exception:
    pass
layout["Yggae"] = [933, 2643, 1083, 2711]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/38_icon_8_100_creator_followers.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 1678), _c38)
except Exception:
    pass
layout["8_100_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/39_text_7.51.png
try:
    _c39 = get_crop(39, 87, 43)
    canvas.paste(_c39, (22, 17), _c39)
except Exception:
    pass
layout["7.51"] = [22, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/40_text_More_events_you_II_love.png
try:
    _c40 = get_crop(40, 1344, 396)
    canvas.paste(_c40, (48, 490), _c40)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/41_text_Mon.png
try:
    _c41 = get_crop(41, 92, 43)
    canvas.paste(_c41, (393, 2129), _c41)
except Exception:
    pass
layout["Mon,"] = [393, 2129, 485, 2172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/42_text_13.png
try:
    _c42 = get_crop(42, 54, 38)
    canvas.paste(_c42, (561, 2129), _c42)
except Exception:
    pass
layout["13"] = [561, 2129, 615, 2167]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/43_text_5_00_PM_PDT.png
try:
    _c43 = get_crop(43, 1344, 396)
    canvas.paste(_c43, (48, 2074), _c43)
except Exception:
    pass
layout["5:00_PM_PDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/44_text_Hour_The_Lookout.png
try:
    _c44 = get_crop(44, 1344, 396)
    canvas.paste(_c44, (48, 2074), _c44)
except Exception:
    pass
layout["Hour_@_The_Lookout"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/45_text_3600_16th_St.png
try:
    _c45 = get_crop(45, 223, 38)
    canvas.paste(_c45, (392, 2328), _c45)
except Exception:
    pass
layout["3600_16th_St"] = [392, 2328, 615, 2366]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/46_text_8_9266_creator_followers.png
try:
    _c46 = get_crop(46, 1344, 396)
    canvas.paste(_c46, (48, 2074), _c46)
except Exception:
    pass
layout["8_9266_creator_followers"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/47_text_Aanonananal.png
try:
    _c47 = get_crop(47, 194, 14)
    canvas.paste(_c47, (98, 2542), _c47)
except Exception:
    pass
layout["Aanonananal"] = [98, 2542, 292, 2556]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/48_text_Sat_May_4.png
try:
    _c48 = get_crop(48, 186, 43)
    canvas.paste(_c48, (392, 2525), _c48)
except Exception:
    pass
layout["Sat,_May_4"] = [392, 2525, 578, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/49_text_IO_00_AM_PDT.png
try:
    _c49 = get_crop(49, 1344, 346)
    canvas.paste(_c49, (48, 2470), _c49)
except Exception:
    pass
layout["IO:00_AM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/50_text_hellaGood.png
try:
    _c50 = get_crop(50, 186, 41)
    canvas.paste(_c50, (101, 2556), _c50)
except Exception:
    pass
layout["hellaGood"] = [101, 2556, 287, 2597]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/51_text_ssan.png
try:
    _c51 = get_crop(51, 25, 9)
    canvas.paste(_c51, (252, 2636), _c51)
except Exception:
    pass
layout["ssan"] = [252, 2636, 277, 2645]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/52_text_featuring-.png
try:
    _c52 = get_crop(52, 43, 15)
    canvas.paste(_c52, (215, 2650), _c52)
except Exception:
    pass
layout["'featuring-"] = [215, 2650, 258, 2665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/53_text_Jah_Wafeidk_SHELTER.png
try:
    _c53 = get_crop(53, 129, 13)
    canvas.paste(_c53, (142, 2702), _c53)
except Exception:
    pass
layout["Jah_Wafeidk_SHELTER"] = [142, 2702, 271, 2715]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/54_text_DJGREENB_DJAGANA_DJMALIIGZ.png
try:
    _c54 = get_crop(54, 215, 18)
    canvas.paste(_c54, (91, 2718), _c54)
except Exception:
    pass
layout["DJGREENB_DJAGANA_DJMALIIG"] = [91, 2718, 306, 2736]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/55_text_Log_AETa.png
try:
    _c55 = get_crop(55, 41, 6)
    canvas.paste(_c55, (111, 2738), _c55)
except Exception:
    pass
layout["Log__AETa"] = [111, 2738, 152, 2744]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/56_text_atrobcats.png
try:
    _c56 = get_crop(56, 43, 13)
    canvas.paste(_c56, (156, 2746), _c56)
except Exception:
    pass
layout["atrobcats"] = [156, 2746, 199, 2759]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/57_text_nalceuani.png
try:
    _c57 = get_crop(57, 37, 7)
    canvas.paste(_c57, (212, 2742), _c57)
except Exception:
    pass
layout["nalceuani"] = [212, 2742, 249, 2749]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/58_text_Lrocaa_Rnrae.png
try:
    _c58 = get_crop(58, 53, 9)
    canvas.paste(_c58, (240, 2763), _c58)
except Exception:
    pass
layout["Lrocaa_Rnrae"] = [240, 2763, 293, 2772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/59_text_SatvaonG.png
try:
    _c59 = get_crop(59, 60, 29)
    canvas.paste(_c59, (92, 2761), _c59)
except Exception:
    pass
layout["SatvaonG"] = [92, 2761, 152, 2790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/60_text_t0ph.png
try:
    _c60 = get_crop(60, 32, 15)
    canvas.paste(_c60, (158, 2767), _c60)
except Exception:
    pass
layout["t0ph"] = [158, 2767, 190, 2782]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/61_text_Z44.png
try:
    _c61 = get_crop(61, 23, 15)
    canvas.paste(_c61, (197, 2767), _c61)
except Exception:
    pass
layout["Z44"] = [197, 2767, 220, 2782]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/62_text_71J_Nissiom_St.st.png
try:
    _c62 = get_crop(62, 74, 13)
    canvas.paste(_c62, (232, 2774), _c62)
except Exception:
    pass
layout["{71J_Nissiom_St.st"] = [232, 2774, 306, 2787]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/63_clickable_Favorites.png
try:
    _c63 = get_crop(63, 288, 156)
    canvas.paste(_c63, (576, 2804), _c63)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/64_clickable_Tickets.png
try:
    _c64 = get_crop(64, 288, 156)
    canvas.paste(_c64, (864, 2804), _c64)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_01_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-3/65_clickable_More.png
try:
    _c65 = get_crop(65, 288, 156)
    canvas.paste(_c65, (1152, 2804), _c65)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
