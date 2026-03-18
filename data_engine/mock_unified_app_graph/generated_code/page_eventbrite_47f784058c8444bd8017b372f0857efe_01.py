# page_id: page_eventbrite_47f784058c8444bd8017b372f0857efe_01
# screenshot: 2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3.png
# step_index: 1/11
# task: Open Eventbrite. Explore local events scheduled for this weekend. Select the first event from the 'Science' category. Read details of the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint overall background
bg_color = "#FBFBFD"        # very light off-white background
status_color = "#BFBDBD"    # light grey status bar
header_bg = "#FFFFFF"       # header white
card_bg = "#FFFFFF"         # card white (slightly stands out against page background)
divider = "#E9E9EE"         # subtle divider color
shadow1 = "#E6E2E8"
shadow2 = "#EFEAF0"
nav_bg = "#FFFFFF"

w, h = canvas.size

# Fill canvas background
draw.rectangle((0, 0, w, h), fill=bg_color)

# Status bar area (~56px tall)
status_h = 56
draw.rectangle((0, 0, w, status_h), fill=status_color)

# Header / toolbar area (below status bar, contains search field - do not redraw the field itself)
header_top = status_h
header_bottom = 200
draw.rectangle((0, header_top, w, header_bottom), fill=header_bg)

# Soft shadow / divider under header
draw.line((48, header_bottom, w-48, header_bottom), fill=shadow1, width=1)
draw.line((48, header_bottom+1, w-48, header_bottom+1), fill=shadow2, width=1)
draw.line((48, header_bottom+2, w-48, header_bottom+2), fill=divider, width=1)

# Section cards / rows backgrounds
# Use the detected vertical grouping positions as guides (top y positions approximated)
card_tops = [470, 866, 1262, 1658, 2054, 2450]  # approximated tops for each list item block
card_height = 180
card_left = 48
card_right = w - 48
card_radius = 14

for top in card_tops:
    # Slightly lift cards off the page with a very faint shadow line above
    draw.line((card_left+2, top-2, card_right-2, top-2), fill="#F3F1F4", width=1)
    # Card background
    draw.rounded_rectangle((card_left, top, card_right, top + card_height),
                           radius=card_radius, fill=card_bg, outline=None)

    # subtle separator inside the card near the bottom to give visual separation between content areas
    sep_y = top + card_height - 1
    draw.line((card_left+12, sep_y, card_right-12, sep_y), fill=divider, width=1)

# Horizontal separators between major sections (full-bleed but inset to respect margins)
section_dividers = [480, 876, 1272, 1668, 2064, 2460]
for y in section_dividers:
    draw.line((card_left, y, card_right, y), fill=divider, width=1)

# Floating location/search pill shadow area (do not draw the pill itself; just provide a soft highlight behind it)
# The detected pill is around y ~2525 with width ~495; provide a subtle backdrop shadow only
pill_center_x = 473 + 495/2  # using detected pos (473,2651) size=(495x117) from available data
pill_center_y = 2651
pill_w = 520
pill_h = 130
pill_bbox = (pill_center_x - pill_w/2, pill_center_y - pill_h/2,
             pill_center_x + pill_w/2, pill_center_y + pill_h/2)
# Soft shadow (do not draw the pill itself)
draw.rectangle((pill_bbox[0]-6, pill_bbox[1]-6, pill_bbox[2]+6, pill_bbox[3]+6), fill="#FBFBFD")

# Bottom navigation bar background area
nav_top = 2804
draw.rectangle((0, nav_top, w, h), fill=nav_bg)
# subtle top divider for nav bar
draw.line((0, nav_top, w, nav_top), fill=shadow1, width=1)

# final subtle edge accents on left/right to match screenshot margins
edge_fill = "#F8F7FA"
draw.rectangle((0, 0, 48, h), fill=edge_fill)
draw.rectangle((w-48, 0, w, h), fill=edge_fill)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/00_icon_ering_to_soothe_the_brokel.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1678), _c0)
except Exception:
    pass
layout["ering_to_soothe_the_broke"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/01_icon_NDIE.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 490), _c1)
except Exception:
    pass
layout["NDIE"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/02_icon_San_Francisco.png
try:
    _c2 = get_crop(2, 495, 117)
    canvas.paste(_c2, (473, 2651), _c2)
except Exception:
    pass
layout["San_Francisco"] = [473, 2651, 968, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/03_icon_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/04_icon_QUEEN.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 2074), _c4)
except Exception:
    pass
layout["QUEEN"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/05_icon_Sat.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 886), _c5)
except Exception:
    pass
layout["Sat,"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 139)
    canvas.paste(_c6, (1140, 747), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/07_icon_3600.png
try:
    _c7 = get_crop(7, 288, 156)
    canvas.paste(_c7, (288, 2804), _c7)
except Exception:
    pass
layout["3600"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/08_icon_City.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 1539), _c8)
except Exception:
    pass
layout["City"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1284, 747), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/10_icon_City.png
try:
    _c10 = get_crop(10, 144, 139)
    canvas.paste(_c10, (1284, 1539), _c10)
except Exception:
    pass
layout["City"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/11_icon_7.57.png
try:
    _c11 = get_crop(11, 108, 102)
    canvas.paste(_c11, (38, 121), _c11)
except Exception:
    pass
layout["7.57"] = [38, 121, 146, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/12_icon_Favorite_button.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1140, 1951), _c12)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 123)
    canvas.paste(_c13, (1284, 1951), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/14_icon_Spring-Zing_Happy.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1140, 2331), _c14)
except Exception:
    pass
layout["Spring-Zing_Happy"] = [1140, 2331, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 139)
    canvas.paste(_c15, (1284, 1143), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/16_icon_Overflow_menu_button.png
try:
    _c16 = get_crop(16, 144, 139)
    canvas.paste(_c16, (1284, 2331), _c16)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2331, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/17_icon_City.png
try:
    _c17 = get_crop(17, 144, 139)
    canvas.paste(_c17, (1140, 1143), _c17)
except Exception:
    pass
layout["City"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/18_icon_7.57.png
try:
    _c18 = get_crop(18, 55, 61)
    canvas.paste(_c18, (183, 2), _c18)
except Exception:
    pass
layout["7.57"] = [183, 2, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/19_icon_RIEF_MEDICIN.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 1282), _c19)
except Exception:
    pass
layout["RIEF_MEDICIN"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/20_icon_PDO_Thread_Training.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 1282), _c20)
except Exception:
    pass
layout["PDO_Thread_Training_|"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 60, 58)
    canvas.paste(_c21, (312, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [312, 3, 372, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 50, 59)
    canvas.paste(_c22, (248, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [248, 3, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 47, 53)
    canvas.paste(_c23, (1321, 7), _c23)
except Exception:
    pass
layout["icon_23"] = [1321, 7, 1368, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/24_icon_7.57.png
try:
    _c24 = get_crop(24, 59, 60)
    canvas.paste(_c24, (114, 3), _c24)
except Exception:
    pass
layout["7.57"] = [114, 3, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/25_icon_Register_Nowl.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["Register_Nowl"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/26_icon_8_29_creator_followers.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 886), _c26)
except Exception:
    pass
layout["8_29_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/27_icon_59_creator_followers.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 490), _c27)
except Exception:
    pass
layout["59_creator_followers"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 57, 58)
    canvas.paste(_c28, (1213, 4), _c28)
except Exception:
    pass
layout["icon_28"] = [1213, 4, 1270, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/29_icon_Spring-Zing_Happy_Hour.png
try:
    _c29 = get_crop(29, 1344, 346)
    canvas.paste(_c29, (48, 2470), _c29)
except Exception:
    pass
layout["Spring-Zing_Happy_Hour"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/30_icon_Free.png
try:
    _c30 = get_crop(30, 125, 73)
    canvas.paste(_c30, (248, 561), _c30)
except Exception:
    pass
layout["Free"] = [248, 561, 373, 634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 41, 55)
    canvas.paste(_c31, (1272, 6), _c31)
except Exception:
    pass
layout["icon_31"] = [1272, 6, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/32_icon_Grief_Medicine_A_Gathering_to_Soothe_the.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 1678), _c32)
except Exception:
    pass
layout["Grief_Medicine:_A_Gatheri"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/33_icon_icon_33.png
try:
    _c33 = get_crop(33, 44, 56)
    canvas.paste(_c33, (385, 6), _c33)
except Exception:
    pass
layout["icon_33"] = [385, 6, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/34_icon_Queen_of_Indies_2024.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 2074), _c34)
except Exception:
    pass
layout["Queen_of_Indies_2024"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/35_icon_Processing_Grief_Self-Care_for_Loss.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 886), _c35)
except Exception:
    pass
layout["Processing_Grief:_Self-Ca"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/36_icon_8_100_creator_followers.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 1678), _c36)
except Exception:
    pass
layout["8_100_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/37_icon_Tickets.png
try:
    _c37 = get_crop(37, 288, 156)
    canvas.paste(_c37, (864, 2804), _c37)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/38_icon_7_00_PM_PDT.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 2074), _c38)
except Exception:
    pass
layout["7:00_PM_PDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/39_icon_7.57.png
try:
    _c39 = get_crop(39, 91, 59)
    canvas.paste(_c39, (16, 4), _c39)
except Exception:
    pass
layout["7.57"] = [16, 4, 107, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/40_text_More_events_you_II_love.png
try:
    _c40 = get_crop(40, 1344, 396)
    canvas.paste(_c40, (48, 490), _c40)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/41_text_Mon.png
try:
    _c41 = get_crop(41, 92, 43)
    canvas.paste(_c41, (393, 2525), _c41)
except Exception:
    pass
layout["Mon,"] = [393, 2525, 485, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/42_text_13_S_00_PM_PDT.png
try:
    _c42 = get_crop(42, 1344, 346)
    canvas.paste(_c42, (48, 2470), _c42)
except Exception:
    pass
layout["13_+_S:00_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/43_text_Out_in_Tech_SF_Bay_Area.png
try:
    _c43 = get_crop(43, 1344, 346)
    canvas.paste(_c43, (48, 2470), _c43)
except Exception:
    pass
layout["Out_in_Tech_SF_Bay_Area"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/44_clickable_Favorites.png
try:
    _c44 = get_crop(44, 288, 156)
    canvas.paste(_c44, (576, 2804), _c44)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_01_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-3/45_clickable_More.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (1152, 2804), _c45)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
