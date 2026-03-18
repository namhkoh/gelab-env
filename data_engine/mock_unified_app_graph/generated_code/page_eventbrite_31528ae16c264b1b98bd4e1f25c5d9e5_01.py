# page_id: page_eventbrite_31528ae16c264b1b98bd4e1f25c5d9e5_01
# screenshot: 2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3.png
# step_index: 1/11
# task: Open Eventbrite. Search 'Fitness'. Filter for free events. Browse and select any 'Yoga' event. Note the location.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (canvas: PIL.Image, draw: PIL.ImageDraw)
w, h = canvas.size

# Colors
bg_color = (250,250,252)        # very light off-white background
status_bar_color = (200,200,200) # light grey for status bar
divider_color = (238,236,242)   # subtle divider / separator (very light purple-gray)
card_shadow = (243,243,245)     # faint shadow for cards
card_bg = (255,255,255)         # card white
nav_bg = (255,255,255)          # bottom nav background (white)
top_divider = (220,220,224)     # stronger divider under status bar

# Fill overall background
draw.rectangle([(0,0),(w,h)], fill=bg_color)

# Status bar area (top)
status_h = 80
draw.rectangle([(0,0),(w,status_h)], fill=status_bar_color)
# fine divider under status bar
draw.line([(0,status_h),(w,status_h)], fill=top_divider, width=1)

# Content area margins and card geometry (use detected block Y positions)
card_left = 48
card_right = w - 48  # 1392 for 1440 canvas
card_height = 396
card_tops = [490, 886, 1282, 1678, 2074, 2470]  # from detections: top positions of list items

# Draw rounded card backgrounds with subtle shadow for each listed section
radius = 18
for y_top in card_tops:
    y_bottom = y_top + card_height
    # shadow (slightly offset down)
    shadow_box = [card_left, y_top+6, card_right, y_bottom+6]
    draw.rounded_rectangle(shadow_box, radius=radius, fill=card_shadow)
    # white card on top
    card_box = [card_left, y_top, card_right, y_bottom]
    draw.rounded_rectangle(card_box, radius=radius, fill=card_bg)

# Draw horizontal separators between cards (matches bottoms of each card)
separator_ys = [top + card_height for top in card_tops]  # [886,1282,...,2804]
for y in separator_ys:
    # draw a faint line across the content width
    draw.line([(card_left+8, y), (card_right-8, y)], fill=divider_color, width=1)

# Draw a subtle leading divider under the page heading area (around first card top)
# This gives a visual separation between header and list
heading_div_y = card_tops[0] - 36
draw.line([(card_left, heading_div_y), (card_right, heading_div_y)], fill=divider_color, width=1)

# Bottom navigation background area
nav_top = 2804
draw.rectangle([(0, nav_top), (w, h)], fill=nav_bg)
# top divider of nav
draw.line([(0, nav_top), (w, nav_top)], fill=top_divider, width=1)

# Small subtle top/bottom shadows to anchor the nav and header visually
# shadow under last card (above nav)
draw.rectangle([(card_left, separator_ys[-1]-1), (card_right, separator_ys[-1]+3)], fill=(245,245,247))
# slight highlight at top content area (below status bar)
draw.rectangle([(card_left, status_h+1), (card_right, status_h+5)], fill=(255,255,255))

# Note: All icons/text/content will be pasted on top of these drawn backgrounds.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/00_icon_ering_to_soothe_the_brokel.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1678), _c0)
except Exception:
    pass
layout["ering_to_soothe_the_broke"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/01_icon_Spring-Zing_Happy_Hour.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 2074), _c1)
except Exception:
    pass
layout["Spring-Zing_Happy_Hour"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/02_icon_NDIE.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 490), _c2)
except Exception:
    pass
layout["NDIE"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/03_icon_San_Francisco.png
try:
    _c3 = get_crop(3, 495, 117)
    canvas.paste(_c3, (473, 2651), _c3)
except Exception:
    pass
layout["San_Francisco"] = [473, 2651, 968, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/04_icon_Q_Search_events.png
try:
    _c4 = get_crop(4, 1179, 144)
    canvas.paste(_c4, (195, 93), _c4)
except Exception:
    pass
layout["Q_Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/05_icon_Sat.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 886), _c5)
except Exception:
    pass
layout["Sat,"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/06_icon_Spring-Zing_Happy.png
try:
    _c6 = get_crop(6, 144, 123)
    canvas.paste(_c6, (1140, 1951), _c6)
except Exception:
    pass
layout["Spring-Zing_Happy"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 139)
    canvas.paste(_c7, (1140, 747), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/08_icon_Spring-Zing_Happy.png
try:
    _c8 = get_crop(8, 144, 123)
    canvas.paste(_c8, (1140, 2347), _c8)
except Exception:
    pass
layout["Spring-Zing_Happy"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/09_icon_City.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1140, 1539), _c9)
except Exception:
    pass
layout["City"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/10_icon_320.png
try:
    _c10 = get_crop(10, 288, 156)
    canvas.paste(_c10, (288, 2804), _c10)
except Exception:
    pass
layout["320"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1284, 747), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/12_icon_7.54.png
try:
    _c12 = get_crop(12, 107, 100)
    canvas.paste(_c12, (39, 122), _c12)
except Exception:
    pass
layout["7.54"] = [39, 122, 146, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/13_icon_Spring-Zing_Happy.png
try:
    _c13 = get_crop(13, 144, 123)
    canvas.paste(_c13, (1284, 1951), _c13)
except Exception:
    pass
layout["Spring-Zing_Happy"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/14_icon_RIEF_MEDICIN.png
try:
    _c14 = get_crop(14, 1344, 396)
    canvas.paste(_c14, (48, 1282), _c14)
except Exception:
    pass
layout["RIEF_MEDICIN"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 123)
    canvas.paste(_c15, (1284, 2347), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/16_icon_City.png
try:
    _c16 = get_crop(16, 144, 139)
    canvas.paste(_c16, (1284, 1539), _c16)
except Exception:
    pass
layout["City"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/17_icon_7.54.png
try:
    _c17 = get_crop(17, 55, 60)
    canvas.paste(_c17, (183, 2), _c17)
except Exception:
    pass
layout["7.54"] = [183, 2, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/18_icon_City.png
try:
    _c18 = get_crop(18, 144, 139)
    canvas.paste(_c18, (1140, 1143), _c18)
except Exception:
    pass
layout["City"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/19_icon_PDO_Thread_Training.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 1282), _c19)
except Exception:
    pass
layout["PDO_Thread_Training_|"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 58, 57)
    canvas.paste(_c20, (313, 4), _c20)
except Exception:
    pass
layout["icon_20"] = [313, 4, 371, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/21_icon_Overflow_menu_button.png
try:
    _c21 = get_crop(21, 144, 139)
    canvas.paste(_c21, (1284, 1143), _c21)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 47, 58)
    canvas.paste(_c22, (250, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [250, 3, 297, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 47, 53)
    canvas.paste(_c23, (1321, 7), _c23)
except Exception:
    pass
layout["icon_23"] = [1321, 7, 1368, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/24_icon_7.54.png
try:
    _c24 = get_crop(24, 58, 59)
    canvas.paste(_c24, (115, 3), _c24)
except Exception:
    pass
layout["7.54"] = [115, 3, 173, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/25_icon_8_29_creator_followers.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 886), _c25)
except Exception:
    pass
layout["8_29_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/26_icon_Fommunity_Center.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Fommunity_Center"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/27_icon_59_creator_followers.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 490), _c27)
except Exception:
    pass
layout["59_creator_followers"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 55, 57)
    canvas.paste(_c28, (1213, 5), _c28)
except Exception:
    pass
layout["icon_28"] = [1213, 5, 1268, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/29_icon_Area.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 2074), _c29)
except Exception:
    pass
layout["Area"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/30_icon_Free.png
try:
    _c30 = get_crop(30, 122, 74)
    canvas.paste(_c30, (250, 560), _c30)
except Exception:
    pass
layout["Free"] = [250, 560, 372, 634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 41, 53)
    canvas.paste(_c31, (1272, 7), _c31)
except Exception:
    pass
layout["icon_31"] = [1272, 7, 1313, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/32_icon_8_100_creator_followers.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 1678), _c32)
except Exception:
    pass
layout["8_100_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/33_icon_Q_Search_events.png
try:
    _c33 = get_crop(33, 43, 55)
    canvas.paste(_c33, (385, 7), _c33)
except Exception:
    pass
layout["Q_Search_events"] = [385, 7, 428, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/34_icon_Grief_Medicine_A_Gathering_to_Soothe_the.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 1678), _c34)
except Exception:
    pass
layout["Grief_Medicine:_A_Gatheri"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/35_icon_Tickets.png
try:
    _c35 = get_crop(35, 288, 156)
    canvas.paste(_c35, (864, 2804), _c35)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/36_icon_Processing_Grief_Self-Care_for_Loss.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 886), _c36)
except Exception:
    pass
layout["Processing_Grief:_Self-Ca"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/37_icon_8_100_creator_followers.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 1678), _c37)
except Exception:
    pass
layout["8_100_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/38_icon_g_NAMI_CC_to_Preventively_Su.png
try:
    _c38 = get_crop(38, 1344, 346)
    canvas.paste(_c38, (48, 2470), _c38)
except Exception:
    pass
layout["g_NAMI_CC_to_Preventively"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/39_icon_8_29_creator_followers.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 886), _c39)
except Exception:
    pass
layout["8_29_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/40_text_7.54.png
try:
    _c40 = get_crop(40, 92, 43)
    canvas.paste(_c40, (22, 17), _c40)
except Exception:
    pass
layout["7.54"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/41_text_More_events_you_II_love.png
try:
    _c41 = get_crop(41, 1344, 396)
    canvas.paste(_c41, (48, 490), _c41)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/42_text_Mon.png
try:
    _c42 = get_crop(42, 92, 43)
    canvas.paste(_c42, (393, 2129), _c42)
except Exception:
    pass
layout["Mon,"] = [393, 2129, 485, 2172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/43_text_13.png
try:
    _c43 = get_crop(43, 54, 38)
    canvas.paste(_c43, (561, 2129), _c43)
except Exception:
    pass
layout["13"] = [561, 2129, 615, 2167]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/44_text_5_00_PM_PDT.png
try:
    _c44 = get_crop(44, 1344, 396)
    canvas.paste(_c44, (48, 2074), _c44)
except Exception:
    pass
layout["5:00_PM_PDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/45_text_Hour_The_Lookout.png
try:
    _c45 = get_crop(45, 1344, 396)
    canvas.paste(_c45, (48, 2074), _c45)
except Exception:
    pass
layout["Hour_@_The_Lookout"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/46_text_3600_16th_St.png
try:
    _c46 = get_crop(46, 223, 38)
    canvas.paste(_c46, (392, 2328), _c46)
except Exception:
    pass
layout["3600_16th_St"] = [392, 2328, 615, 2366]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/47_text_8_9266_creator_followers.png
try:
    _c47 = get_crop(47, 1344, 396)
    canvas.paste(_c47, (48, 2074), _c47)
except Exception:
    pass
layout["8_9266_creator_followers"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/48_text_Wed_May_22.png
try:
    _c48 = get_crop(48, 221, 43)
    canvas.paste(_c48, (394, 2525), _c48)
except Exception:
    pass
layout["Wed,_May_22"] = [394, 2525, 615, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/49_text_5.30_PM_PDT.png
try:
    _c49 = get_crop(49, 1344, 346)
    canvas.paste(_c49, (48, 2470), _c49)
except Exception:
    pass
layout["5.30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/50_text_320.png
try:
    _c50 = get_crop(50, 75, 38)
    canvas.paste(_c50, (392, 2723), _c50)
except Exception:
    pass
layout["320"] = [392, 2723, 467, 2761]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/51_clickable_Favorites.png
try:
    _c51 = get_crop(51, 288, 156)
    canvas.paste(_c51, (576, 2804), _c51)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_01_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-3/52_clickable_More.png
try:
    _c52 = get_crop(52, 288, 156)
    canvas.paste(_c52, (1152, 2804), _c52)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
