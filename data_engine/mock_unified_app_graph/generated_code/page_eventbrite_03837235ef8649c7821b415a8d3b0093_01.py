# page_id: page_eventbrite_03837235ef8649c7821b415a8d3b0093_01
# screenshot: 2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3.png
# step_index: 1/8
# task: Open Eventbrite. Locate the 'Conference' category. Filter the results to only show virtual events. Choose the first event from the results. What is the duration of this event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall page chrome / backgrounds for Eventbrite-like mobile UI
# Assumes variables provided: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Page background (subtle warm white)
draw.rectangle((0, 0, W, H), fill="#FFFFFF")

# Status bar (top area ~56px) - light gray to match screenshot status area
status_h = 56
draw.rectangle((0, 0, W, status_h), fill="#CFCFCF")

# Thin divider below status bar to separate it from header
draw.line((0, status_h, W, status_h), fill="#BEBEBE", width=1)

# Header area / toolbar background (below status bar)
header_top = status_h
header_bottom = 220  # includes search area region (search widget will be pasted later)
draw.rectangle((0, header_top, W, header_bottom), fill="#FFFFFF")

# subtle shadow line under header
draw.line((24, header_bottom, W-24, header_bottom), fill="#E8E8E8", width=1)

# Main content container background (a large rounded card area behind the list)
list_left = 32
list_right = W - 32
list_top = 360
list_bottom = H - 220
shadow_offset = 8

# Draw a subtle shadow for the container
draw.rounded_rectangle(
    (list_left + shadow_offset, list_top + shadow_offset, list_right + shadow_offset, list_bottom + shadow_offset),
    radius=28, fill="#F5F5F6"
)

# Draw the main white container on top
draw.rounded_rectangle(
    (list_left, list_top, list_right, list_bottom),
    radius=28, fill="#FFFFFF", outline=None
)

# Section separators: draw thin separators where list items are separated
# Use detected vertical positions to place separators between items
separator_x1 = 48
separator_x2 = W - 48
# Positions derived from detected item top Ys and heights in the image:
separators_y = [886, 1282, 1678, 2074, 2470]  # horizontal lines between rows
for y in separators_y:
    draw.line((separator_x1, y, separator_x2, y), fill="#F0F0F2", width=1)

# Slight inner left margin vertical guide (visual alignment aid background, very subtle)
guide_x = 48
draw.line((guide_x, list_top + 12, guide_x, list_bottom - 12), fill="#FFFFFF", width=1)

# Small section header background area (behind the "More events you'll love" heading)
# Place a pale purple wash behind the heading location (but do NOT draw any text)
heading_bg_left = list_left + 8
heading_bg_right = list_right - 8
heading_bg_top = 400
heading_bg_bottom = 460
draw.rectangle((heading_bg_left, heading_bg_top, heading_bg_right, heading_bg_bottom), fill="#FFFFFF")

# Draw a faint divider under the heading area
draw.line((heading_bg_left, heading_bg_bottom, heading_bg_right, heading_bg_bottom), fill="#F0EDF6", width=1)

# Floating location/search pill background area placeholder shadow region (do NOT draw the pill itself)
# We only add a faint shadow to indicate space reserved for the overlayed pill (keeps from duplicating icons)
pill_shadow_bbox = (440, 2620, 980, 2688)
draw.rectangle((pill_shadow_bbox[0]+6, pill_shadow_bbox[1]+6, pill_shadow_bbox[2]+6, pill_shadow_bbox[3]+6), fill="#EFEFF1")
# Do not draw the pill itself (it will be pasted later)

# Bottom navigation bar background and top divider
nav_h = 116
nav_top = H - nav_h
draw.rectangle((0, nav_top, W, H), fill="#FFFFFF")
# Divider line above nav
draw.line((0, nav_top, W, nav_top), fill="#E6E6E8", width=1)

# Optional subtle top and bottom shadows for visual separation (very light)
draw.line((0, nav_top+2, W, nav_top+2), fill="#F7F7F8", width=1)
draw.line((0, 2, W, 2), fill="#FFFFFF", width=1)

# Right-side persistent vertical safe margin (do not draw over overflow/menu icon areas)
# This is only a subtle visual margin stroke
draw.line((W-48, list_top + 12, W-48, list_bottom - 12), fill="#FFFFFF", width=1)

# Done drawing structural UI backgrounds (no text/icons drawn)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/00_icon_ering_to_soothe_the_brokel.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1678), _c0)
except Exception:
    pass
layout["ering_to_soothe_the_broke"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/01_icon_NDIE.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 490), _c1)
except Exception:
    pass
layout["NDIE"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/02_icon_Search_events.png
try:
    _c2 = get_crop(2, 1179, 144)
    canvas.paste(_c2, (195, 93), _c2)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/03_icon_Sat.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 886), _c3)
except Exception:
    pass
layout["Sat,"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/04_icon_QUEEN.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 2074), _c4)
except Exception:
    pass
layout["QUEEN"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 139)
    canvas.paste(_c5, (1140, 747), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/06_icon_City.png
try:
    _c6 = get_crop(6, 144, 139)
    canvas.paste(_c6, (1140, 1539), _c6)
except Exception:
    pass
layout["City"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 139)
    canvas.paste(_c7, (1284, 747), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/08_icon_San_Francisco.png
try:
    _c8 = get_crop(8, 495, 117)
    canvas.paste(_c8, (473, 2651), _c8)
except Exception:
    pass
layout["San_Francisco"] = [473, 2651, 968, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/09_icon_Bissa.png
try:
    _c9 = get_crop(9, 288, 156)
    canvas.paste(_c9, (288, 2804), _c9)
except Exception:
    pass
layout["Bissa}"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/10_icon_Reggaeton.png
try:
    _c10 = get_crop(10, 144, 139)
    canvas.paste(_c10, (1140, 2331), _c10)
except Exception:
    pass
layout["Reggaeton__"] = [1140, 2331, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/11_icon_City.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1284, 1539), _c11)
except Exception:
    pass
layout["City"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/12_icon_4.40.png
try:
    _c12 = get_crop(12, 110, 103)
    canvas.paste(_c12, (37, 120), _c12)
except Exception:
    pass
layout["4.40"] = [37, 120, 147, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/13_icon_Favorite_button.png
try:
    _c13 = get_crop(13, 144, 123)
    canvas.paste(_c13, (1140, 1951), _c13)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 123)
    canvas.paste(_c14, (1284, 1951), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/15_icon_Reggaeton.png
try:
    _c15 = get_crop(15, 144, 139)
    canvas.paste(_c15, (1284, 2331), _c15)
except Exception:
    pass
layout["Reggaeton__"] = [1284, 2331, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/16_icon_Overflow_menu_button.png
try:
    _c16 = get_crop(16, 144, 139)
    canvas.paste(_c16, (1284, 1143), _c16)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/17_icon_City.png
try:
    _c17 = get_crop(17, 144, 139)
    canvas.paste(_c17, (1140, 1143), _c17)
except Exception:
    pass
layout["City"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/18_icon_SatvaonG.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (0, 2804), _c18)
except Exception:
    pass
layout["SatvaonG"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/19_icon_PDO_Thread_Training.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 1282), _c19)
except Exception:
    pass
layout["PDO_Thread_Training_|"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 59, 58)
    canvas.paste(_c20, (313, 3), _c20)
except Exception:
    pass
layout["icon_20"] = [313, 3, 372, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/21_icon_4.40.png
try:
    _c21 = get_crop(21, 55, 59)
    canvas.paste(_c21, (183, 3), _c21)
except Exception:
    pass
layout["4.40"] = [183, 3, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 50, 59)
    canvas.paste(_c22, (248, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [248, 3, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 47, 53)
    canvas.paste(_c23, (1321, 7), _c23)
except Exception:
    pass
layout["icon_23"] = [1321, 7, 1368, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/24_icon_9_00_PM_PDT.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 490), _c24)
except Exception:
    pass
layout["9:00_PM_PDT"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 63, 58)
    canvas.paste(_c25, (1213, 4), _c25)
except Exception:
    pass
layout["icon_25"] = [1213, 4, 1276, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/26_icon_8_30_creator_followers.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 886), _c26)
except Exception:
    pass
layout["8_30_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/27_icon_Queen_of_Indies_2024.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 2074), _c27)
except Exception:
    pass
layout["Queen_of_Indies_2024"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/28_icon_Free.png
try:
    _c28 = get_crop(28, 125, 73)
    canvas.paste(_c28, (248, 561), _c28)
except Exception:
    pass
layout["Free"] = [248, 561, 373, 634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/29_icon_4.40.png
try:
    _c29 = get_crop(29, 57, 61)
    canvas.paste(_c29, (116, 2), _c29)
except Exception:
    pass
layout["4.40"] = [116, 2, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 41, 55)
    canvas.paste(_c30, (1272, 6), _c30)
except Exception:
    pass
layout["icon_30"] = [1272, 6, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/31_icon_Grief_Medicine_A_Gathering_to_Soothe_the.png
try:
    _c31 = get_crop(31, 1344, 396)
    canvas.paste(_c31, (48, 1678), _c31)
except Exception:
    pass
layout["Grief_Medicine:_A_Gatheri"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/32_icon_8_100_creator_followers.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 1678), _c32)
except Exception:
    pass
layout["8_100_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/33_icon_Salsa.png
try:
    _c33 = get_crop(33, 1344, 346)
    canvas.paste(_c33, (48, 2470), _c33)
except Exception:
    pass
layout["Salsa"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/34_icon_Sales_ended.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 1282), _c34)
except Exception:
    pass
layout["Sales_ended"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/35_icon_icon_35.png
try:
    _c35 = get_crop(35, 43, 57)
    canvas.paste(_c35, (385, 6), _c35)
except Exception:
    pass
layout["icon_35"] = [385, 6, 428, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/36_icon_Processing_Grief_Self-Care_for_Loss.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 886), _c36)
except Exception:
    pass
layout["Processing_Grief:_Self-Ca"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/37_icon_7_00_PM_PDT.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 2074), _c37)
except Exception:
    pass
layout["7:00_PM_PDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/38_text_4.40.png
try:
    _c38 = get_crop(38, 89, 43)
    canvas.paste(_c38, (22, 15), _c38)
except Exception:
    pass
layout["4.40"] = [22, 15, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/39_text_More_events_you_II_love.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 490), _c39)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/40_text_Aanonalone.png
try:
    _c40 = get_crop(40, 192, 14)
    canvas.paste(_c40, (98, 2542), _c40)
except Exception:
    pass
layout["Aanonalone"] = [98, 2542, 290, 2556]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/41_text_Sat_May_4_._10_00_AM_PDT.png
try:
    _c41 = get_crop(41, 1344, 346)
    canvas.paste(_c41, (48, 2470), _c41)
except Exception:
    pass
layout["Sat,_May_4_._10:00_AM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/42_text_hellaGood.png
try:
    _c42 = get_crop(42, 186, 41)
    canvas.paste(_c42, (101, 2556), _c42)
except Exception:
    pass
layout["hellaGood"] = [101, 2556, 287, 2597]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/43_text_ssan.png
try:
    _c43 = get_crop(43, 25, 9)
    canvas.paste(_c43, (252, 2636), _c43)
except Exception:
    pass
layout["ssan"] = [252, 2636, 277, 2645]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/44_text_featuring-.png
try:
    _c44 = get_crop(44, 43, 15)
    canvas.paste(_c44, (215, 2650), _c44)
except Exception:
    pass
layout["'featuring-"] = [215, 2650, 258, 2665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/45_text_Jah_Wafeidk_SHELTER.png
try:
    _c45 = get_crop(45, 129, 13)
    canvas.paste(_c45, (142, 2702), _c45)
except Exception:
    pass
layout["Jah_Wafeidk_SHELTER"] = [142, 2702, 271, 2715]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/46_text_DJGREENB_DJAGANA_DJMALIIGZ.png
try:
    _c46 = get_crop(46, 215, 18)
    canvas.paste(_c46, (91, 2718), _c46)
except Exception:
    pass
layout["DJGREENB_DJAGANA_DJMALIIG"] = [91, 2718, 306, 2736]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/47_text_Log_AETa.png
try:
    _c47 = get_crop(47, 41, 6)
    canvas.paste(_c47, (111, 2738), _c47)
except Exception:
    pass
layout["Log__AETa"] = [111, 2738, 152, 2744]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/48_text_atrobcats.png
try:
    _c48 = get_crop(48, 43, 13)
    canvas.paste(_c48, (156, 2746), _c48)
except Exception:
    pass
layout["atrobcats"] = [156, 2746, 199, 2759]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/49_text_nalceuani.png
try:
    _c49 = get_crop(49, 37, 7)
    canvas.paste(_c49, (212, 2742), _c49)
except Exception:
    pass
layout["nalceuani"] = [212, 2742, 249, 2749]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/50_text_Lrocaa_Rnrae.png
try:
    _c50 = get_crop(50, 53, 9)
    canvas.paste(_c50, (240, 2763), _c50)
except Exception:
    pass
layout["Lrocaa_Rnrae"] = [240, 2763, 293, 2772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/51_text_SatvaonG.png
try:
    _c51 = get_crop(51, 60, 29)
    canvas.paste(_c51, (92, 2761), _c51)
except Exception:
    pass
layout["SatvaonG"] = [92, 2761, 152, 2790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/52_text_t0ph.png
try:
    _c52 = get_crop(52, 32, 15)
    canvas.paste(_c52, (158, 2767), _c52)
except Exception:
    pass
layout["t0ph"] = [158, 2767, 190, 2782]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/53_text_Z44.png
try:
    _c53 = get_crop(53, 23, 15)
    canvas.paste(_c53, (197, 2767), _c53)
except Exception:
    pass
layout["Z44"] = [197, 2767, 220, 2782]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/54_text_71J_Nissiom_St.st.png
try:
    _c54 = get_crop(54, 74, 13)
    canvas.paste(_c54, (232, 2774), _c54)
except Exception:
    pass
layout["{71J_Nissiom_St.st"] = [232, 2774, 306, 2787]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/55_text_9_232_creator_followers.png
try:
    _c55 = get_crop(55, 1344, 346)
    canvas.paste(_c55, (48, 2470), _c55)
except Exception:
    pass
layout["9_232_creator_followers"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/56_clickable_Favorites.png
try:
    _c56 = get_crop(56, 288, 156)
    canvas.paste(_c56, (576, 2804), _c56)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/57_clickable_Tickets.png
try:
    _c57 = get_crop(57, 288, 156)
    canvas.paste(_c57, (864, 2804), _c57)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_01_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-3/58_clickable_More.png
try:
    _c58 = get_crop(58, 288, 156)
    canvas.paste(_c58, (1152, 2804), _c58)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
