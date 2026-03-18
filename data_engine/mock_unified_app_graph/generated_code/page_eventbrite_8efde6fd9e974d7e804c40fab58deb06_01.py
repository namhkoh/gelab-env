# page_id: page_eventbrite_8efde6fd9e974d7e804c40fab58deb06_01
# screenshot: 2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3.png
# step_index: 1/8
# task: Open Eventbrite. Search for "Education". Filter only online events. Note how many events are available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_height = 56
draw.rectangle([(0, 0), (1440, status_height)], fill=(230, 230, 230))  # light gray status background
draw.line([(0, status_height), (1440, status_height)], fill=(200, 200, 200), width=1)  # divider under status bar

# Header area (behind search)
header_top = status_height
header_bottom = 200
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))  # keep header white

# Search bar background (rounded) - behind detected search element
search_x0, search_y0 = 195, 93
search_w, search_h = 1179, 144
search_box = [search_x0, search_y0, search_x0 + search_w, search_y0 + search_h]
draw.rounded_rectangle(search_box, radius=72, fill=(250, 250, 252), outline=(220, 220, 225), width=2)

# Main content subtle background (very light warm white)
draw.rectangle([(0, header_bottom), (1440, 2960 - 160)], fill=(255, 255, 255))

# Event row card backgrounds (rounded rects) - positioned to sit behind detected content blocks
card_x = 48
card_w = 1344
card_positions = [
    (48, 490, 48 + card_w, 490 + 396),   # first block
    (48, 886, 48 + card_w, 886 + 396),
    (48, 1282, 48 + card_w, 1282 + 396),
    (48, 1678, 48 + card_w, 1678 + 396),
    (48, 2074, 48 + card_w, 2074 + 396),
    (48, 2470, 48 + card_w, 2470 + 346)  # last block slightly shorter
]
for rect in card_positions:
    # subtle card shadow (top-left lighter & bottom-right slight)
    x0, y0, x1, y1 = rect
    # shadow layer
    shadow_rect = (x0 + 4, y0 + 6, x1 + 4, y1 + 6)
    draw.rounded_rectangle(shadow_rect, radius=16, fill=(245, 245, 247))
    # main card
    draw.rounded_rectangle(rect, radius=16, fill=(255, 255, 255), outline=(236, 236, 239), width=1)

    # left thumbnail placeholder background (do not draw any image content; just a neutral area)
    thumb_x0 = x0 + 8
    thumb_y0 = y0 + 16
    thumb_x1 = thumb_x0 + 160
    thumb_y1 = thumb_y0 + 160
    draw.rectangle([(thumb_x0, thumb_y0), (thumb_x1, thumb_y1)], fill=(240, 240, 242), outline=(230, 230, 234))

    # subtle divider line at bottom of card
    draw.line([(x0 + 8, y1 - 8), (x1 - 8, y1 - 8)], fill=(240, 240, 243), width=1)

# Additional thin separators between cards (full-bleed subtle rules)
separator_ys = [ (490 + 396), (886 + 396), (1282 + 396), (1678 + 396), (2074 + 396), (2470 + 346) ]
for y in separator_ys:
    draw.line([(48, y + 6), (1392, y + 6)], fill=(245, 245, 247), width=1)

# Floating small location pill background (do not draw its text/icon — just the pill behind it)
# The pill is visible around y ~ 2630 area in screenshot; draw faint rounded pill background
pill_center_x = 720
pill_center_y = 2651
pill_w = 420
pill_h = 96
pill_box = [pill_center_x - pill_w // 2, pill_center_y - pill_h // 2, pill_center_x + pill_w // 2, pill_center_y + pill_h // 2]
draw.rounded_rectangle(pill_box, radius=48, fill=(255, 255, 255), outline=(230, 230, 235), width=1)
# slight shadow under pill
draw.rounded_rectangle([pill_box[0], pill_box[1]+6, pill_box[2], pill_box[3]+6], radius=48, fill=(250,250,250))

# Bottom navigation bar background and top divider
nav_y = 2804
draw.rectangle([(0, nav_y), (1440, 2960)], fill=(255, 255, 255))
draw.line([(0, nav_y), (1440, nav_y)], fill=(230, 230, 235), width=1)

# Active home indicator (a subtle orange dot under left nav area to indicate active; avoid drawing icon shapes)
home_indicator_center = (144, nav_y + 18)
draw.ellipse([(home_indicator_center[0]-6, home_indicator_center[1]-6), (home_indicator_center[0]+6, home_indicator_center[1]+6)], fill=(238, 84, 30))

# End: subtle overall vignette to match screenshot's soft feel (very light)
overlay_color = (255, 255, 255, 0)
# No blending imports allowed; keep to simple shapes already drawn.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/00_icon_City.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 886), _c0)
except Exception:
    pass
layout["City,"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/01_icon_Search_events.png
try:
    _c1 = get_crop(1, 1179, 144)
    canvas.paste(_c1, (195, 93), _c1)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/02_icon_New_York.png
try:
    _c2 = get_crop(2, 144, 139)
    canvas.paste(_c2, (1140, 747), _c2)
except Exception:
    pass
layout["New_York"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/03_icon_Conference_Connections.png
try:
    _c3 = get_crop(3, 144, 139)
    canvas.paste(_c3, (1140, 1935), _c3)
except Exception:
    pass
layout["Conference_Connections"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/04_icon_Free.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 1282), _c4)
except Exception:
    pass
layout["Free"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/05_icon_Conference_Connections.png
try:
    _c5 = get_crop(5, 144, 123)
    canvas.paste(_c5, (1140, 2347), _c5)
except Exception:
    pass
layout["Conference_Connections"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/06_icon_VOSCHINO.png
try:
    _c6 = get_crop(6, 1344, 396)
    canvas.paste(_c6, (48, 1678), _c6)
except Exception:
    pass
layout["VOSCHINO"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/07_icon_New_York.png
try:
    _c7 = get_crop(7, 144, 123)
    canvas.paste(_c7, (1140, 1159), _c7)
except Exception:
    pass
layout["New_York"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/08_icon_New_York.png
try:
    _c8 = get_crop(8, 144, 123)
    canvas.paste(_c8, (1284, 1159), _c8)
except Exception:
    pass
layout["New_York"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/09_icon_New_York.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1284, 747), _c9)
except Exception:
    pass
layout["New_York"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1284, 2347), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/11_icon_Union_H.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (288, 2804), _c11)
except Exception:
    pass
layout["Union_H"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/12_icon_Good_Afternoon_New_York.png
try:
    _c12 = get_crop(12, 1344, 396)
    canvas.paste(_c12, (48, 1678), _c12)
except Exception:
    pass
layout["Good_Afternoon_New_York"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/13_icon_New_York.png
try:
    _c13 = get_crop(13, 405, 117)
    canvas.paste(_c13, (518, 2651), _c13)
except Exception:
    pass
layout["New_York"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1284, 1935), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 123)
    canvas.paste(_c15, (1284, 1555), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1555, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/16_icon_Medical_Hair_Loss_Therapy_Training.png
try:
    _c16 = get_crop(16, 1344, 396)
    canvas.paste(_c16, (48, 886), _c16)
except Exception:
    pass
layout["Medical_Hair_Loss_Therapy"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/17_icon_free_Grief_and_Loss_Healing_Circle.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 490), _c17)
except Exception:
    pass
layout["(free)_Grief_and_Loss_Hea"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/18_icon_139_creator_followers.png
try:
    _c18 = get_crop(18, 1344, 396)
    canvas.paste(_c18, (48, 1282), _c18)
except Exception:
    pass
layout["139_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/19_icon_Favorite_button.png
try:
    _c19 = get_crop(19, 144, 123)
    canvas.paste(_c19, (1140, 1555), _c19)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1555, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/20_icon_Home.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (0, 2804), _c20)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/21_icon_6.58.png
try:
    _c21 = get_crop(21, 102, 96)
    canvas.paste(_c21, (42, 123), _c21)
except Exception:
    pass
layout["6.58"] = [42, 123, 144, 219]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 59, 58)
    canvas.paste(_c22, (312, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [312, 3, 371, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/23_icon_6.58.png
try:
    _c23 = get_crop(23, 55, 60)
    canvas.paste(_c23, (183, 2), _c23)
except Exception:
    pass
layout["6.58"] = [183, 2, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/24_icon_Free.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 490), _c24)
except Exception:
    pass
layout["Free"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 50, 59)
    canvas.paste(_c25, (248, 3), _c25)
except Exception:
    pass
layout["icon_25"] = [248, 3, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 47, 52)
    canvas.paste(_c26, (1321, 8), _c26)
except Exception:
    pass
layout["icon_26"] = [1321, 8, 1368, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/27_icon_6.58.png
try:
    _c27 = get_crop(27, 58, 61)
    canvas.paste(_c27, (115, 2), _c27)
except Exception:
    pass
layout["6.58"] = [115, 2, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 58, 57)
    canvas.paste(_c28, (1212, 5), _c28)
except Exception:
    pass
layout["icon_28"] = [1212, 5, 1270, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/29_icon_Free.png
try:
    _c29 = get_crop(29, 130, 74)
    canvas.paste(_c29, (244, 560), _c29)
except Exception:
    pass
layout["Free"] = [244, 560, 374, 634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 41, 54)
    canvas.paste(_c30, (1272, 7), _c30)
except Exception:
    pass
layout["icon_30"] = [1272, 7, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 44, 56)
    canvas.paste(_c31, (385, 6), _c31)
except Exception:
    pass
layout["icon_31"] = [385, 6, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/32_icon_Free.png
try:
    _c32 = get_crop(32, 127, 75)
    canvas.paste(_c32, (245, 2540), _c32)
except Exception:
    pass
layout["Free"] = [245, 2540, 372, 2615]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/33_icon_free_Grief_and_Loss_Healing_Circle.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 490), _c33)
except Exception:
    pass
layout["(free)_Grief_and_Loss_Hea"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/34_icon_8_7107_creator_followers.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 2074), _c34)
except Exception:
    pass
layout["8_7107_creator_followers"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/35_icon_8_1646_creator_followers.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 886), _c35)
except Exception:
    pass
layout["8_1646_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/36_icon_City.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 2074), _c36)
except Exception:
    pass
layout["City"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/37_icon_Sat_May_4_._11_59_PM_EDT.png
try:
    _c37 = get_crop(37, 1344, 346)
    canvas.paste(_c37, (48, 2470), _c37)
except Exception:
    pass
layout["Sat;_May_4_._11:59_PM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/38_text_6.58.png
try:
    _c38 = get_crop(38, 89, 43)
    canvas.paste(_c38, (22, 17), _c38)
except Exception:
    pass
layout["6.58"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/39_text_More_events_you_II_love.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 490), _c39)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/40_text_vLt.png
try:
    _c40 = get_crop(40, 59, 49)
    canvas.paste(_c40, (42, 2535), _c40)
except Exception:
    pass
layout["vLt"] = [42, 2535, 101, 2584]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/41_clickable_Favorites.png
try:
    _c41 = get_crop(41, 288, 156)
    canvas.paste(_c41, (576, 2804), _c41)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/42_clickable_Tickets.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (864, 2804), _c42)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_01_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-3/43_clickable_More.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (1152, 2804), _c43)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
