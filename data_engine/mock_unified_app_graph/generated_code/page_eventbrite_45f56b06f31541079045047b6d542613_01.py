# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_01
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-3.png
# step_index: 1/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint overall background
draw.rectangle((0, 0, 1440, 2960), fill=(250, 246, 255))  # soft off-white/pale lavender background

# Status bar (top) - light gray strip
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(200, 200, 200))

# Header area under status bar (toolbar background)
header_top = status_h
header_bottom = 176
draw.rectangle((0, header_top, 1440, header_bottom), fill=(250, 246, 255))  # keep subtle same BG
# thin divider under header
draw.line((48, header_bottom, 1392, header_bottom), fill=(235, 231, 241), width=2)

# List card areas (rounded white cards with subtle border and separators)
card_x1, card_x2 = 48, 1392
card_height = 396
card_radius = 16
card_fill = (255, 255, 255)
card_outline = (230, 228, 236)

card_tops = [490, 886, 1282, 1678, 2074, 2470, 2804]
for y in card_tops:
    # Draw rounded card background
    draw.rounded_rectangle((card_x1, y, card_x2, y + card_height), radius=card_radius, fill=card_fill, outline=card_outline, width=1)
    # subtle bottom divider/shadow line
    draw.line((card_x1 + 8, y + card_height, card_x2 - 8, y + card_height), fill=(245, 243, 247), width=2)

# Horizontal separators between list items (extra subtle)
for y in [490 + card_height, 886 + card_height, 1282 + card_height, 1678 + card_height, 2074 + card_height, 2470 + card_height]:
    draw.line((48, y + 8, 1392, y + 8), fill=(245, 243, 247), width=1)

# Accent left edge guide (very subtle vertical line to separate thumbnail area from text)
thumb_sep_x = 48 + 160  # approximate thumbnail column width
draw.line((thumb_sep_x, 450, thumb_sep_x, 2760), fill=(245, 243, 247), width=1)

# Floating location pill shadow area (background shadow under the floating city pill)
# approximate center of floating pill from screenshot; draw faint rounded rectangle/ellipse as shadow
pill_shadow_box = (360, 2610, 1080, 2700)
draw.rounded_rectangle(pill_shadow_box, radius=36, fill=(246, 246, 247))

# Bottom navigation bar background and top divider
nav_top = 2804
draw.rectangle((0, nav_top, 1440, 2960), fill=(255, 255, 255))
draw.line((0, nav_top, 1440, nav_top), fill=(230, 228, 236), width=2)

# Subtle elevation shadow above bottom nav (thin gradient-esque strokes)
for i, alpha_offset in enumerate(range(1, 6)):
    y = nav_top - (6 - i)
    shade = 240 + i  # slightly changing shade
    draw.line((0, y, 1440, y), fill=(shade, shade, shade), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/00_icon_ering_to_soothe_the_brokel.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1678), _c0)
except Exception:
    pass
layout["ering_to_soothe_the_broke"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/01_icon_Spring-Zing_Happy_Hour.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 2074), _c1)
except Exception:
    pass
layout["Spring-Zing_Happy_Hour"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/02_icon_NDIE.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 490), _c2)
except Exception:
    pass
layout["NDIE"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/03_icon_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/04_icon_Sat.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 886), _c4)
except Exception:
    pass
layout["Sat,"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/05_icon_San_Francisco.png
try:
    _c5 = get_crop(5, 495, 117)
    canvas.paste(_c5, (473, 2651), _c5)
except Exception:
    pass
layout["San_Francisco"] = [473, 2651, 968, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/06_icon_Spring-Zing_Happy.png
try:
    _c6 = get_crop(6, 144, 123)
    canvas.paste(_c6, (1140, 1951), _c6)
except Exception:
    pass
layout["Spring-Zing_Happy"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/07_icon_RIEF_MEDICIN.png
try:
    _c7 = get_crop(7, 1344, 396)
    canvas.paste(_c7, (48, 1282), _c7)
except Exception:
    pass
layout["RIEF_MEDICIN"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/08_icon_8_22L1SOLU1.png
try:
    _c8 = get_crop(8, 288, 156)
    canvas.paste(_c8, (288, 2804), _c8)
except Exception:
    pass
layout["8_22L1SOLU1"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/09_icon_Favorite_button.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1140, 747), _c9)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/10_icon_City.png
try:
    _c10 = get_crop(10, 144, 139)
    canvas.paste(_c10, (1140, 1539), _c10)
except Exception:
    pass
layout["City"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/11_icon_Spring-Zing_Happy.png
try:
    _c11 = get_crop(11, 144, 123)
    canvas.paste(_c11, (1140, 2347), _c11)
except Exception:
    pass
layout["Spring-Zing_Happy"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/12_icon_Spring-Zing_Happy.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1284, 1951), _c12)
except Exception:
    pass
layout["Spring-Zing_Happy"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 139)
    canvas.paste(_c13, (1284, 747), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/14_icon_7.28.png
try:
    _c14 = get_crop(14, 111, 103)
    canvas.paste(_c14, (37, 120), _c14)
except Exception:
    pass
layout["7.28"] = [37, 120, 148, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 123)
    canvas.paste(_c15, (1284, 2347), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/16_icon_City.png
try:
    _c16 = get_crop(16, 144, 139)
    canvas.paste(_c16, (1284, 1539), _c16)
except Exception:
    pass
layout["City"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/17_icon_7.28.png
try:
    _c17 = get_crop(17, 54, 60)
    canvas.paste(_c17, (184, 2), _c17)
except Exception:
    pass
layout["7.28"] = [184, 2, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/18_icon_PDO_Thread_Training.png
try:
    _c18 = get_crop(18, 1344, 396)
    canvas.paste(_c18, (48, 1282), _c18)
except Exception:
    pass
layout["PDO_Thread_Training_|"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/19_icon_City.png
try:
    _c19 = get_crop(19, 144, 139)
    canvas.paste(_c19, (1140, 1143), _c19)
except Exception:
    pass
layout["City"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/20_icon_Overflow_menu_button.png
try:
    _c20 = get_crop(20, 144, 139)
    canvas.paste(_c20, (1284, 1143), _c20)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 58, 57)
    canvas.paste(_c21, (313, 4), _c21)
except Exception:
    pass
layout["icon_21"] = [313, 4, 371, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/22_icon_ce_for_our_commun.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (0, 2804), _c22)
except Exception:
    pass
layout["ce_for_our_commun"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 47, 58)
    canvas.paste(_c23, (250, 3), _c23)
except Exception:
    pass
layout["icon_23"] = [250, 3, 297, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/24_icon_8_29_creator_followers.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 886), _c24)
except Exception:
    pass
layout["8_29_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 47, 53)
    canvas.paste(_c25, (1321, 7), _c25)
except Exception:
    pass
layout["icon_25"] = [1321, 7, 1368, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/26_icon_7.28.png
try:
    _c26 = get_crop(26, 58, 59)
    canvas.paste(_c26, (115, 3), _c26)
except Exception:
    pass
layout["7.28"] = [115, 3, 173, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/27_icon_59_creator_followers.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 490), _c27)
except Exception:
    pass
layout["59_creator_followers"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 56, 58)
    canvas.paste(_c28, (1213, 4), _c28)
except Exception:
    pass
layout["icon_28"] = [1213, 4, 1269, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/29_icon_8_100_creator_followers.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 1678), _c29)
except Exception:
    pass
layout["8_100_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/30_icon_Free.png
try:
    _c30 = get_crop(30, 125, 73)
    canvas.paste(_c30, (248, 561), _c30)
except Exception:
    pass
layout["Free"] = [248, 561, 373, 634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/31_icon_Area.png
try:
    _c31 = get_crop(31, 1344, 396)
    canvas.paste(_c31, (48, 2074), _c31)
except Exception:
    pass
layout["Area"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/32_icon_icon_32.png
try:
    _c32 = get_crop(32, 41, 55)
    canvas.paste(_c32, (1272, 6), _c32)
except Exception:
    pass
layout["icon_32"] = [1272, 6, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/33_icon_Free.png
try:
    _c33 = get_crop(33, 53, 52)
    canvas.paste(_c33, (290, 2618), _c33)
except Exception:
    pass
layout["Free"] = [290, 2618, 343, 2670]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/34_icon_Grief_Medicine_A_Gathering_to_Soothe_the.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 1678), _c34)
except Exception:
    pass
layout["Grief_Medicine:_A_Gatheri"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/35_icon_icon_35.png
try:
    _c35 = get_crop(35, 43, 55)
    canvas.paste(_c35, (385, 7), _c35)
except Exception:
    pass
layout["icon_35"] = [385, 7, 428, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/36_icon_Processing_Grief_Self-Care_for_Loss.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 886), _c36)
except Exception:
    pass
layout["Processing_Grief:_Self-Ca"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/37_icon_Free.png
try:
    _c37 = get_crop(37, 119, 73)
    canvas.paste(_c37, (248, 2541), _c37)
except Exception:
    pass
layout["Free"] = [248, 2541, 367, 2614]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/38_icon_San_Francisco.png
try:
    _c38 = get_crop(38, 1344, 346)
    canvas.paste(_c38, (48, 2470), _c38)
except Exception:
    pass
layout["San_Francisco"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/39_icon_8_100_creator_followers.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 1678), _c39)
except Exception:
    pass
layout["8_100_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/40_text_7.28.png
try:
    _c40 = get_crop(40, 91, 45)
    canvas.paste(_c40, (20, 15), _c40)
except Exception:
    pass
layout["7.28"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/41_text_More_events_you_II_love.png
try:
    _c41 = get_crop(41, 1344, 396)
    canvas.paste(_c41, (48, 490), _c41)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/42_text_Mon.png
try:
    _c42 = get_crop(42, 92, 43)
    canvas.paste(_c42, (393, 2129), _c42)
except Exception:
    pass
layout["Mon,"] = [393, 2129, 485, 2172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/43_text_13.png
try:
    _c43 = get_crop(43, 54, 38)
    canvas.paste(_c43, (561, 2129), _c43)
except Exception:
    pass
layout["13"] = [561, 2129, 615, 2167]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/44_text_5_00_PM_PDT.png
try:
    _c44 = get_crop(44, 1344, 396)
    canvas.paste(_c44, (48, 2074), _c44)
except Exception:
    pass
layout["5:00_PM_PDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/45_text_Hour_The_Lookout.png
try:
    _c45 = get_crop(45, 1344, 396)
    canvas.paste(_c45, (48, 2074), _c45)
except Exception:
    pass
layout["Hour_@_The_Lookout"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/46_text_3600_16th_St.png
try:
    _c46 = get_crop(46, 223, 38)
    canvas.paste(_c46, (392, 2328), _c46)
except Exception:
    pass
layout["3600_16th_St"] = [392, 2328, 615, 2366]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/47_text_8_9265_creator_followers.png
try:
    _c47 = get_crop(47, 1344, 396)
    canvas.paste(_c47, (48, 2074), _c47)
except Exception:
    pass
layout["8_9265_creator_followers"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/48_text_THE.png
try:
    _c48 = get_crop(48, 59, 37)
    canvas.paste(_c48, (170, 2519), _c48)
except Exception:
    pass
layout["THE"] = [170, 2519, 229, 2556]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/49_text_Tue_Apr_23.png
try:
    _c49 = get_crop(49, 200, 43)
    canvas.paste(_c49, (390, 2557), _c49)
except Exception:
    pass
layout["Tue,_Apr_23"] = [390, 2557, 590, 2600]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/50_text_8_00_AM_PDT.png
try:
    _c50 = get_crop(50, 1344, 346)
    canvas.paste(_c50, (48, 2470), _c50)
except Exception:
    pass
layout["8:00_AM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/51_text_Cafe_at.png
try:
    _c51 = get_crop(51, 157, 52)
    canvas.paste(_c51, (112, 2631), _c51)
except Exception:
    pass
layout["Cafe_at"] = [112, 2631, 269, 2683]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/52_text_The.png
try:
    _c52 = get_crop(52, 66, 38)
    canvas.paste(_c52, (394, 2693), _c52)
except Exception:
    pass
layout["The"] = [394, 2693, 460, 2731]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/53_text_pw_be.png
try:
    _c53 = get_crop(53, 92, 31)
    canvas.paste(_c53, (44, 2736), _c53)
except Exception:
    pass
layout["pw_be"] = [44, 2736, 136, 2767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/54_text_ce_for_our_commun.png
try:
    _c54 = get_crop(54, 288, 156)
    canvas.paste(_c54, (0, 2804), _c54)
except Exception:
    pass
layout["ce_for_our_commun"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/55_clickable_Favorites.png
try:
    _c55 = get_crop(55, 288, 156)
    canvas.paste(_c55, (576, 2804), _c55)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/56_clickable_Tickets.png
try:
    _c56 = get_crop(56, 288, 156)
    canvas.paste(_c56, (864, 2804), _c56)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_01_2024_4_23_19_27_45f56b06f31541079045047b6d542613-3/57_clickable_More.png
try:
    _c57 = get_crop(57, 288, 156)
    canvas.paste(_c57, (1152, 2804), _c57)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
