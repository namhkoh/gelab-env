# page_id: page_eventbrite_03837235ef8649c7821b415a8d3b0093_07
# screenshot: 2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9.png
# step_index: 7/8
# task: Open Eventbrite. Locate the 'Conference' category. Filter the results to only show virtual events. Choose the first event from the results. What is the duration of this event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for 1440x2960 canvas
# Available: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Base background (very light warm white)
draw.rectangle((0, 0, 1440, 2960), fill=(250, 250, 252))

# Status bar (top dark area)
status_h = 64
draw.rectangle((0, 0, 1440, status_h), fill=(45, 48, 54))

# Subtle divider under status bar
draw.line((0, status_h, 1440, status_h), fill=(230, 230, 235), width=1)

# Search/header area background (keeps white but add faint bottom divider)
search_top = status_h
search_h = 140
draw.rectangle((0, search_top, 1440, search_top + search_h), fill=(250, 250, 252))
draw.line((48, search_top + search_h - 2, 1440 - 48, search_top + search_h - 2), fill=(225, 225, 230), width=2)

# Thin subtle section divider under filter area
filters_bottom = 360
draw.line((36, filters_bottom, 1440 - 36, filters_bottom), fill=(236, 236, 240), width=1)

# Shadow helper: draw a soft rectangle shadow by layering semi-opaque strokes
def _shadow(outline_box, shadow_color, expand=8):
    x0, y0, x1, y1 = outline_box
    # layered darker strokes to simulate soft shadow
    for i, alpha in enumerate((40, 30, 20, 12), start=1):
        inset = expand - i*2
        if inset < 0: inset = 0
        draw.rounded_rectangle(
            (x0 - inset, y0 - inset + 6 + i, x1 + inset, y1 + inset + 6 + i),
            radius=28 + i,
            outline=(shadow_color[0], shadow_color[1], shadow_color[2]),
            width=1
        )

# First event card background (rounded card behind the image & title)
card1_box = (36, 636, 1404, 1720)
_shadow(card1_box, (210, 210, 215), expand=10)
draw.rounded_rectangle(card1_box, radius=24, fill=(255, 255, 255), outline=(235, 235, 240), width=1)

# Separator (small gap) between image area and metadata region inside card (visual only)
sep_y = 636 + 420
draw.line((60, sep_y, 1440 - 60, sep_y), fill=(245, 245, 247), width=1)

# Second event card background
card2_box = (36, 1700, 1404, 2796)
_shadow(card2_box, (210, 210, 215), expand=10)
draw.rounded_rectangle(card2_box, radius=24, fill=(255, 255, 255), outline=(235, 235, 240), width=1)

# Subtle inner divider on second card (under image area)
sep2_y = 1700 + 420
draw.line((60, sep2_y, 1440 - 60, sep2_y), fill=(245, 245, 247), width=1)

# Light page-wide divider above the bottom navigation
nav_top = 2804
draw.line((0, nav_top, 1440, nav_top), fill=(230, 230, 235), width=2)

# Bottom navigation bar background
draw.rectangle((0, nav_top, 1440, 2960), fill=(255, 255, 255))
draw.line((0, nav_top + 2, 1440, nav_top + 2), fill=(240, 240, 244), width=1)

# Highlight for active nav item (second slot) - background accent circle (kept subtle)
nav_item_w = 288
active_center_x = nav_item_w * 1 + nav_item_w // 2  # second item
active_center_y = nav_top + 78
accent_radius = 44
draw.ellipse(
    (active_center_x - accent_radius, active_center_y - accent_radius,
     active_center_x + accent_radius, active_center_y + accent_radius),
    fill=(250, 240, 232)
)

# Gentle rounded corners for the whole content area left/right margins visually
# (thin decorative lines to match material card spacing)
draw.line((36, filters_bottom + 24, 36, nav_top - 24), fill=(248, 248, 250), width=1)
draw.line((1404, filters_bottom + 24, 1404, nav_top - 24), fill=(248, 248, 250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (850, 410), _c0)
except Exception:
    pass
layout["Music"] = [850, 410, 1037, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1049, 410), _c1)
except Exception:
    pass
layout["Business"] = [1049, 410, 1290, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (438, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [438, 410, 838, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/05_icon_Fo.png
try:
    _c5 = get_crop(5, 131, 111)
    canvas.paste(_c5, (1296, 406), _c5)
except Exception:
    pass
layout["Fo("] = [1296, 406, 1427, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2252), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2252, 1236, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 2252), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2252, 1380, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/09_icon_12pm_MDt.png
try:
    _c9 = get_crop(9, 1344, 1012)
    canvas.paste(_c9, (48, 676), _c9)
except Exception:
    pass
layout["12pm_MDt'"] = [48, 676, 1392, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/10_icon_Advanced_Clinical_Supervisor_Next_Level.png
try:
    _c10 = get_crop(10, 1344, 1080)
    canvas.paste(_c10, (48, 1736), _c10)
except Exception:
    pass
layout["Advanced_Clinical_Supervi"] = [48, 1736, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/11_icon_4.41.png
try:
    _c11 = get_crop(11, 125, 116)
    canvas.paste(_c11, (55, 113), _c11)
except Exception:
    pass
layout["4.41"] = [55, 113, 180, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/12_icon_4.41.png
try:
    _c12 = get_crop(12, 61, 64)
    canvas.paste(_c12, (180, 0), _c12)
except Exception:
    pass
layout["4.41"] = [180, 0, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/13_icon_Search_forae.png
try:
    _c13 = get_crop(13, 68, 63)
    canvas.paste(_c13, (307, 0), _c13)
except Exception:
    pass
layout["Search_forae"] = [307, 0, 375, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/14_icon_4.41.png
try:
    _c14 = get_crop(14, 62, 65)
    canvas.paste(_c14, (113, 0), _c14)
except Exception:
    pass
layout["4.41"] = [113, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 54, 64)
    canvas.paste(_c15, (246, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [246, 0, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 62, 60)
    canvas.paste(_c16, (1316, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1316, 0, 1378, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 66, 61)
    canvas.paste(_c17, (1208, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1208, 0, 1274, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/18_icon_8.30AM_EDT.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (576, 2804), _c18)
except Exception:
    pass
layout["8.30AM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/19_icon_Search_forae.png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/20_icon_Online.png
try:
    _c20 = get_crop(20, 377, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 50, 60)
    canvas.paste(_c21, (1263, 0), _c21)
except Exception:
    pass
layout["icon_21"] = [1263, 0, 1313, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/22_icon_Promoted.png
try:
    _c22 = get_crop(22, 42, 55)
    canvas.paste(_c22, (283, 2726), _c22)
except Exception:
    pass
layout["Promoted"] = [283, 2726, 325, 2781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/23_icon_Advanced_Clinical_Supervisor_Next_Level.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["Advanced_Clinical_Supervi"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/24_icon_More.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/25_icon_8.30AM_EDT.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (288, 2804), _c25)
except Exception:
    pass
layout["8.30AM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/26_icon_Search_forae.png
try:
    _c26 = get_crop(26, 50, 62)
    canvas.paste(_c26, (384, 2), _c26)
except Exception:
    pass
layout["Search_forae"] = [384, 2, 434, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/27_icon_4.41.png
try:
    _c27 = get_crop(27, 94, 64)
    canvas.paste(_c27, (10, 0), _c27)
except Exception:
    pass
layout["4.41"] = [10, 0, 104, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/28_icon_Advanced_Clinical_Supervisor_Next_Level.png
try:
    _c28 = get_crop(28, 1344, 1080)
    canvas.paste(_c28, (48, 1736), _c28)
except Exception:
    pass
layout["Advanced_Clinical_Supervi"] = [48, 1736, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/29_text_3_222_events.png
try:
    _c29 = get_crop(29, 372, 103)
    canvas.paste(_c29, (54, 410), _c29)
except Exception:
    pass
layout["3,222_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/30_text_The_Faces_of_Feminine.png
try:
    _c30 = get_crop(30, 1344, 1012)
    canvas.paste(_c30, (48, 676), _c30)
except Exception:
    pass
layout["The_Faces_of_Feminine"] = [48, 676, 1392, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/31_text_Sat_May_11.png
try:
    _c31 = get_crop(31, 232, 55)
    canvas.paste(_c31, (91, 1456), _c31)
except Exception:
    pass
layout["Sat,_May_11"] = [91, 1456, 323, 1511]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/32_text_2_00_PM_EDT.png
try:
    _c32 = get_crop(32, 251, 49)
    canvas.paste(_c32, (347, 1455), _c32)
except Exception:
    pass
layout["2:00_PM_EDT"] = [347, 1455, 598, 1504]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/33_text_Online.png
try:
    _c33 = get_crop(33, 126, 43)
    canvas.paste(_c33, (94, 1527), _c33)
except Exception:
    pass
layout["Online"] = [94, 1527, 220, 1570]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/34_text_Promoted.png
try:
    _c34 = get_crop(34, 195, 43)
    canvas.paste(_c34, (94, 1594), _c34)
except Exception:
    pass
layout["Promoted"] = [94, 1594, 289, 1637]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/35_text_Online.png
try:
    _c35 = get_crop(35, 126, 43)
    canvas.paste(_c35, (94, 2665), _c35)
except Exception:
    pass
layout["Online"] = [94, 2665, 220, 2708]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_07_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-9/36_clickable_Home.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (0, 2804), _c36)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
