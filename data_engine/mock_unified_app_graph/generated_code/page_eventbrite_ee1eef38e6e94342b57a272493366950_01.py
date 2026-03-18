# page_id: page_eventbrite_ee1eef38e6e94342b57a272493366950_01
# screenshot: 2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3.png
# step_index: 1/10
# task: Open Eventbrite. Open "Fashion" category. Apply filter for free events. From the list, select the first non-promoted event and add it to your favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the mobile page.
# Uses provided variables: canvas (PIL Image), draw (ImageDraw)

bg_color = (250, 250, 250)      # subtle off-white page background
status_bar_color = (189, 189, 189)  # grey status bar
header_search_bg = (244, 245, 248)  # search pill background
divider_color = (230, 230, 235)   # light divider lines
nav_bar_color = (245, 245, 247)   # bottom nav background
card_shadow = (242, 242, 245)     # very subtle card background

# Fill full background
draw.rectangle([(0, 0), canvas.size], fill=bg_color)

width, height = canvas.size

# Status bar at top (~56px)
status_h = 56
draw.rectangle([(0, 0), (width, status_h)], fill=status_bar_color)

# Header area under status bar
header_top = status_h
header_bottom = 160
draw.rectangle([(0, header_top), (width, header_bottom)], fill=bg_color)

# Rounded search pill (background only, icons/text will be pasted later)
search_left = 160
search_right = width - 120
search_top = header_top + 16
search_bottom = search_top + 64
radius = 36
try:
    draw.rounded_rectangle([(search_left, search_top), (search_right, search_bottom)],
                           radius=radius, fill=header_search_bg, outline=(222,222,227))
except Exception:
    # fallback if rounded_rectangle missing: draw rectangle
    draw.rectangle([(search_left, search_top), (search_right, search_bottom)],
                   fill=header_search_bg, outline=(222,222,227))

# Thin divider under header
draw.line([(40, header_bottom - 2), (width - 40, header_bottom - 2)], fill=divider_color, width=1)

# Subtle card-like banding for row groups (behind content)
band_margin_x = 32
band_width = width - band_margin_x * 2
row_bands = [
    (420, 580),
    (1210, 1370),
    (1606, 1766),
    (2002, 2162),
    (2400, 2560)
]
for top, bottom in row_bands:
    try:
        draw.rounded_rectangle([(band_margin_x, top), (band_margin_x + band_width, bottom)],
                               radius=8, fill=card_shadow)
    except Exception:
        draw.rectangle([(band_margin_x, top), (band_margin_x + band_width, bottom)], fill=card_shadow)

# Horizontal separators between rows (thin lines)
separator_ys = [440, 760, 1100, 1520, 1900, 2280, 2660]
for y in separator_ys:
    draw.line([(48, y), (width - 48, y)], fill=divider_color, width=1)

# Floating search/sort bar area that sits above content (visual background only)
float_left = 200
float_right = width - 200
float_top = 2680
float_bottom = float_top + 96
try:
    draw.rounded_rectangle([(float_left, float_top), (float_right, float_bottom)],
                           radius=48, fill=(255,255,255), outline=(230,230,235))
except Exception:
    draw.rectangle([(float_left, float_top), (float_right, float_bottom)],
                   fill=(255,255,255), outline=(230,230,235))

# Bottom navigation bar background and top divider
nav_top = height - 120
draw.rectangle([(0, nav_top), (width, height)], fill=nav_bar_color)
draw.line([(0, nav_top), (width, nav_top)], fill=divider_color, width=1)

# Ensure we do not draw over any detected elements by clearing those rectangles back to bg_color.
detected_boxes = [
    (48,490,1344,396),
    (195,93,1179,144),
    (48,2074,1344,396),
    (288,2804,288,156),
    (1140,2345,144,125),
    (48,1282,1344,396),
    (1284,2345,144,125),
    (37,120,109,102),
    (1284,1539,144,139),
    (1140,1935,144,139),
    (1284,1935,144,139),
    (0,2804,288,156),
    (1284,747,144,139),
    (1284,1143,144,139),
    (184,2,54,61),
    (312,3,60,59),
    (248,3,51,59),
    (48,2470,1344,346),
    (1321,6,48,55),
    (115,3,57,60),
    (48,886,1344,396),
    (576,2804,288,156),
    (1140,747,144,139),
    (1140,1143,144,139),
    (1321,3,94,60),
    (1140,1539,144,139),
    (48,1678,1344,396),
    (48,1678,1344,396),
    (385,6,44,57),
    (864,2804,288,156),
    (427,2651,586,117),
    (390,703,112,54),
    (48,2074,1344,396),
    (1272,7,41,54),
    (22,17,87,43),
    (48,490,1344,396),
    (390,2583,165,45),
    (48,2470,1344,346),
    (1031,2646,110,57),
    (1152,2804,288,156)
]

# Draw white (bg_color) rectangles over detected areas to avoid duplicating icons/text.
for x, y, w, h in detected_boxes:
    # Defensive: ensure boxes are within canvas bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)
    if x2 > x1 and y2 > y1:
        draw.rectangle([(x1, y1), (x2, y2)], fill=bg_color)

# Small final top shadow under status bar (subtle)
draw.line([(0, status_h), (width, status_h)], fill=(210,210,210), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/00_icon_Online.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 490), _c0)
except Exception:
    pass
layout["Online"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/01_icon_Search_events.png
try:
    _c1 = get_crop(1, 1179, 144)
    canvas.paste(_c1, (195, 93), _c1)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/02_icon_Online.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 2074), _c2)
except Exception:
    pass
layout["Online"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/03_icon_Or.png
try:
    _c3 = get_crop(3, 288, 156)
    canvas.paste(_c3, (288, 2804), _c3)
except Exception:
    pass
layout["Or"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/04_icon_Loss.png
try:
    _c4 = get_crop(4, 144, 125)
    canvas.paste(_c4, (1140, 2345), _c4)
except Exception:
    pass
layout["Loss"] = [1140, 2345, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/05_icon_Understanding_Grief_and_Loss.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1282), _c5)
except Exception:
    pass
layout["Understanding_Grief_and_L"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 125)
    canvas.paste(_c6, (1284, 2345), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2345, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/07_icon_5.27.png
try:
    _c7 = get_crop(7, 109, 102)
    canvas.paste(_c7, (37, 120), _c7)
except Exception:
    pass
layout["5.27"] = [37, 120, 146, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1284, 1539), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/09_icon_Favorite_button.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1140, 1935), _c9)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 139)
    canvas.paste(_c10, (1284, 1935), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/11_icon_Fing_You_t0_understand_yo.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (0, 2804), _c11)
except Exception:
    pass
layout["Fing_You_t0_understand_yo"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 139)
    canvas.paste(_c12, (1284, 747), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 139)
    canvas.paste(_c13, (1284, 1143), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/14_icon_5.27.png
try:
    _c14 = get_crop(14, 54, 61)
    canvas.paste(_c14, (184, 2), _c14)
except Exception:
    pass
layout["5.27"] = [184, 2, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 60, 59)
    canvas.paste(_c15, (312, 3), _c15)
except Exception:
    pass
layout["icon_15"] = [312, 3, 372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/16_icon_YG.png
try:
    _c16 = get_crop(16, 1344, 346)
    canvas.paste(_c16, (48, 2470), _c16)
except Exception:
    pass
layout["YG"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 51, 59)
    canvas.paste(_c17, (248, 3), _c17)
except Exception:
    pass
layout["icon_17"] = [248, 3, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/18_icon_Working_with_Grief_and_Loss.png
try:
    _c18 = get_crop(18, 1344, 396)
    canvas.paste(_c18, (48, 490), _c18)
except Exception:
    pass
layout["Working_with_Grief_and_Lo"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/19_icon_Favorite_button.png
try:
    _c19 = get_crop(19, 144, 139)
    canvas.paste(_c19, (1140, 747), _c19)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 48, 55)
    canvas.paste(_c20, (1321, 6), _c20)
except Exception:
    pass
layout["icon_20"] = [1321, 6, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/21_icon_5.27.png
try:
    _c21 = get_crop(21, 57, 60)
    canvas.paste(_c21, (115, 3), _c21)
except Exception:
    pass
layout["5.27"] = [115, 3, 172, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/22_icon_1253_creator_followers.png
try:
    _c22 = get_crop(22, 1344, 396)
    canvas.paste(_c22, (48, 886), _c22)
except Exception:
    pass
layout["1253_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/23_icon_Online_events.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (576, 2804), _c23)
except Exception:
    pass
layout["Online_events"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/24_icon_Favorite_button.png
try:
    _c24 = get_crop(24, 144, 139)
    canvas.paste(_c24, (1140, 1143), _c24)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 94, 60)
    canvas.paste(_c25, (1211, 3), _c25)
except Exception:
    pass
layout["icon_25"] = [1211, 3, 1305, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/26_icon_Favorite_button.png
try:
    _c26 = get_crop(26, 144, 139)
    canvas.paste(_c26, (1140, 1539), _c26)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/27_icon_Art_for_Grief_and_Loss.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 1678), _c27)
except Exception:
    pass
layout["Art_for_Grief_and_Loss"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/28_icon_Weeruy_se55.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 1678), _c28)
except Exception:
    pass
layout["Weeruy_se55"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 44, 57)
    canvas.paste(_c29, (385, 6), _c29)
except Exception:
    pass
layout["icon_29"] = [385, 6, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/30_icon_Loss.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (864, 2804), _c30)
except Exception:
    pass
layout["Loss"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/31_icon_Online_events.png
try:
    _c31 = get_crop(31, 586, 117)
    canvas.paste(_c31, (427, 2651), _c31)
except Exception:
    pass
layout["Online_events"] = [427, 2651, 1013, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/32_icon_Online.png
try:
    _c32 = get_crop(32, 112, 54)
    canvas.paste(_c32, (390, 703), _c32)
except Exception:
    pass
layout["Online"] = [390, 703, 502, 757]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/33_icon_5_00_AM_EST.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 2074), _c33)
except Exception:
    pass
layout["5:00_AM_EST"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/34_icon_icon_34.png
try:
    _c34 = get_crop(34, 41, 54)
    canvas.paste(_c34, (1272, 7), _c34)
except Exception:
    pass
layout["icon_34"] = [1272, 7, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/35_text_5.27.png
try:
    _c35 = get_crop(35, 87, 43)
    canvas.paste(_c35, (22, 17), _c35)
except Exception:
    pass
layout["5.27"] = [22, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/36_text_More_events_you_II_love.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 490), _c36)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/37_text_Sat_Jun_1.png
try:
    _c37 = get_crop(37, 165, 45)
    canvas.paste(_c37, (390, 2583), _c37)
except Exception:
    pass
layout["Sat,_Jun_1"] = [390, 2583, 555, 2628]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/38_text_5_00_AM_EDT.png
try:
    _c38 = get_crop(38, 1344, 346)
    canvas.paste(_c38, (48, 2470), _c38)
except Exception:
    pass
layout["5:00_AM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/39_text_Loss.png
try:
    _c39 = get_crop(39, 110, 57)
    canvas.paste(_c39, (1031, 2646), _c39)
except Exception:
    pass
layout["Loss"] = [1031, 2646, 1141, 2703]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_01_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-3/40_clickable_More.png
try:
    _c40 = get_crop(40, 288, 156)
    canvas.paste(_c40, (1152, 2804), _c40)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
