# page_id: page_eventbrite_1a166da440f24e2e9152f2c0e40eb7aa_06
# screenshot: 2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8.png
# step_index: 6/16
# task: Open Eventbrite. Check "Sports" category. Filter events happening next month. Add the first event to your wishlist.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural elements for the mobile UI page
# Uses provided canvas (1440x2960) and draw (ImageDraw) objects.

# Colors
bg_color = "#FBFCFD"           # very light page background
status_bar_color = "#CFCFCF"   # gray status bar
header_bg = "#FFFFFF"          # white header/search background
divider_color = "#E6E7EA"      # subtle divider lines
card_bg = "#FFFFFF"            # card background (white)
card_shadow = "#EEF1F4"        # faint shadow for cards
bottom_nav_bg = "#FFFFFF"      # bottom nav background
bottom_nav_shadow = "#E9EBEE"  # subtle top divider/shadow for bottom nav

# Clear canvas / fill background
draw.rectangle((0, 0, 1440, 2960), fill=bg_color)

# Status bar area (top)
status_h = 96
draw.rectangle((0, 0, 1440, status_h), fill=status_bar_color)

# Header / Search area
header_top = status_h
header_bottom = 320
draw.rectangle((0, header_top, 1440, header_bottom), fill=header_bg)
# subtle bottom divider under header
draw.line((48, header_bottom, 1392, header_bottom), fill=divider_color, width=2)

# Thin separator under filter/chips area
filters_div_y = 420
draw.line((48, filters_div_y, 1392, filters_div_y), fill=divider_color, width=1)

# First event card (rounded rectangle with shadow)
card1_x1, card1_y1 = 48, 440
card1_x2, card1_y2 = 1392, 1120
radius = 28
# shadow (simple offset rectangle for shadow effect)
draw.rounded_rectangle((card1_x1+6, card1_y1+8, card1_x2+6, card1_y2+8), radius=radius, fill=card_shadow)
# card background
draw.rounded_rectangle((card1_x1, card1_y1, card1_x2, card1_y2), radius=radius, fill=card_bg, outline=divider_color, width=1)

# Separator area between image and text area within the card (subtle)
inner_sep_y = card1_y1 + (card1_y2 - card1_y1) * 0.62
draw.line((card1_x1+24, inner_sep_y, card1_x2-24, inner_sep_y), fill=bg_color, width=1)

# Second event card (rounded rectangle with shadow)
card2_x1, card2_y1 = 48, 1480
card2_x2, card2_y2 = 1392, 2160
# shadow
draw.rounded_rectangle((card2_x1+6, card2_y1+8, card2_x2+6, card2_y2+8), radius=radius, fill=card_shadow)
# card background
draw.rounded_rectangle((card2_x1, card2_y1, card2_x2, card2_y2), radius=radius, fill=card_bg, outline=divider_color, width=1)

# Small horizontal separator between results list sections
draw.line((48, 1320, 1392, 1320), fill=divider_color, width=1)
draw.line((48, 2320, 1392, 2320), fill=divider_color, width=1)

# Bottom navigation area
nav_top = 2820
draw.line((0, nav_top, 1440, nav_top), fill=bottom_nav_shadow, width=2)
draw.rectangle((0, nav_top, 1440, 2960), fill=bottom_nav_bg)

# Light edge/shadow above first card group to give depth
draw.rectangle((24, card1_y1-12, 1416, card1_y1-10), fill=card_shadow)

# Decorative subtle left/right page gutters (very subtle)
draw.rectangle((0, 0, 24, 2960), fill=bg_color)
draw.rectangle((1416, 0, 1440, 2960), fill=bg_color)

# Final top toolbar divider (very thin) to separate status bar from content
draw.line((0, status_h, 1440, status_h), fill=divider_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/00_icon_Anytime.png
try:
    _c0 = get_crop(0, 400, 135)
    canvas.paste(_c0, (438, 390), _c0)
except Exception:
    pass
layout["Anytime"] = [438, 390, 838, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/01_icon_1_Filter.png
try:
    _c1 = get_crop(1, 372, 135)
    canvas.paste(_c1, (54, 390), _c1)
except Exception:
    pass
layout["1_Filter"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/02_icon_Sports_Fitness.png
try:
    _c2 = get_crop(2, 378, 135)
    canvas.paste(_c2, (850, 390), _c2)
except Exception:
    pass
layout["Sports_&_Fitness"] = [850, 390, 1228, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/03_icon_Favorite_button.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 1192), _c3)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/04_icon_Overflow_menu_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1236, 1192), _c4)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2252), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2252, 1236, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2252), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2252, 1380, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/07_icon_5.31.png
try:
    _c7 = get_crop(7, 124, 112)
    canvas.paste(_c7, (55, 115), _c7)
except Exception:
    pass
layout["5.31"] = [55, 115, 179, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/08_icon_5.31.png
try:
    _c8 = get_crop(8, 61, 65)
    canvas.paste(_c8, (180, 0), _c8)
except Exception:
    pass
layout["5.31"] = [180, 0, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/09_icon_Search_forae.png
try:
    _c9 = get_crop(9, 69, 64)
    canvas.paste(_c9, (307, 0), _c9)
except Exception:
    pass
layout["Search_forae"] = [307, 0, 376, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/10_icon_5.31.png
try:
    _c10 = get_crop(10, 62, 66)
    canvas.paste(_c10, (113, 0), _c10)
except Exception:
    pass
layout["5.31"] = [113, 0, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 54, 64)
    canvas.paste(_c11, (246, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [246, 0, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 68, 61)
    canvas.paste(_c12, (1209, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1209, 0, 1277, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 61, 60)
    canvas.paste(_c13, (1316, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1316, 0, 1377, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/14_icon_I_00_PM_EDT.png
try:
    _c14 = get_crop(14, 288, 156)
    canvas.paste(_c14, (288, 2804), _c14)
except Exception:
    pass
layout["I:00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/15_icon_Search_forae.png
try:
    _c15 = get_crop(15, 1344, 191)
    canvas.paste(_c15, (48, 72), _c15)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/16_icon_Wwwactivesouthdu.png
try:
    _c16 = get_crop(16, 1344, 1080)
    canvas.paste(_c16, (48, 1736), _c16)
except Exception:
    pass
layout["Wwwactivesouthdu"] = [48, 1736, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/17_icon_Online.png
try:
    _c17 = get_crop(17, 377, 144)
    canvas.paste(_c17, (0, 259), _c17)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/18_icon_Basics_of_Roller_Skating_balance_power.png
try:
    _c18 = get_crop(18, 1344, 1012)
    canvas.paste(_c18, (48, 676), _c18)
except Exception:
    pass
layout["Basics_of_Roller_Skating_"] = [48, 676, 1392, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/19_icon_I_00_PM_EDT.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (576, 2804), _c19)
except Exception:
    pass
layout["I:00_PM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/20_icon_Search_forae.png
try:
    _c20 = get_crop(20, 50, 62)
    canvas.paste(_c20, (384, 2), _c20)
except Exception:
    pass
layout["Search_forae"] = [384, 2, 434, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/21_icon_5.31.png
try:
    _c21 = get_crop(21, 95, 64)
    canvas.paste(_c21, (11, 0), _c21)
except Exception:
    pass
layout["5.31"] = [11, 0, 106, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/22_icon_Basic_Awareness_Course.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (864, 2804), _c22)
except Exception:
    pass
layout["Basic_Awareness_Course"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 40, 61)
    canvas.paste(_c23, (1274, 0), _c23)
except Exception:
    pass
layout["icon_23"] = [1274, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/24_icon_Online.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (0, 2804), _c24)
except Exception:
    pass
layout["Online"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/25_icon_Promoted.png
try:
    _c25 = get_crop(25, 251, 62)
    canvas.paste(_c25, (81, 1582), _c25)
except Exception:
    pass
layout["Promoted"] = [81, 1582, 332, 1644]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/26_icon_Promoted.png
try:
    _c26 = get_crop(26, 41, 55)
    canvas.paste(_c26, (286, 1586), _c26)
except Exception:
    pass
layout["Promoted"] = [286, 1586, 327, 1641]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/27_text_690_events.png
try:
    _c27 = get_crop(27, 372, 135)
    canvas.paste(_c27, (54, 390), _c27)
except Exception:
    pass
layout["690_events"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/28_text_SAFEGUARDING.png
try:
    _c28 = get_crop(28, 1344, 1080)
    canvas.paste(_c28, (48, 1736), _c28)
except Exception:
    pass
layout["SAFEGUARDING"] = [48, 1736, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/29_text_Online.png
try:
    _c29 = get_crop(29, 186, 63)
    canvas.paste(_c29, (94, 2615), _c29)
except Exception:
    pass
layout["Online"] = [94, 2615, 280, 2678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/30_text_25th_April.png
try:
    _c30 = get_crop(30, 278, 72)
    canvas.paste(_c30, (313, 2614), _c30)
except Exception:
    pass
layout["25th_April"] = [313, 2614, 591, 2686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/31_text_Thu_Apr_25.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (0, 2804), _c31)
except Exception:
    pass
layout["Thu,_Apr_25"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/32_text_I_00_PM_EDT.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (288, 2804), _c32)
except Exception:
    pass
layout["I:00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/33_text_Online.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (0, 2804), _c33)
except Exception:
    pass
layout["Online"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_06_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-8/34_clickable_More.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (1152, 2804), _c34)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
