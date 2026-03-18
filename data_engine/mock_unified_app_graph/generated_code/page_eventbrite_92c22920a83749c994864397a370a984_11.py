# page_id: page_eventbrite_92c22920a83749c994864397a370a984_11
# screenshot: 2024_4_24_16_59_92c22920a83749c994864397a370a984-13.png
# step_index: 11/13
# task: Open Eventbrite. Set the city to "Chicago". Select the "Sports" category and view the recommended events. See the date of the first non-promoted event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas (provided): 1440x2960 RGB, variable name `canvas`
# Draw object (provided): `draw`
# Fonts available: font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
bg_color = (247, 248, 250)       # overall very light background
status_color = (189, 189, 189)   # status bar grey
search_bg = (255, 255, 255)      # search bar white
divider_color = (220, 223, 227)  # subtle divider
card_bg = (255, 255, 255)        # card background
card_shadow = (230, 234, 238)    # card shadow
image_overlay1 = (208, 238, 249) # bluish overlay for first card image area
image_overlay2 = (255, 244, 224) # warm overlay for second card image area
nav_bg = (255, 255, 255)         # bottom nav background
nav_top_border = (224, 227, 231)

# 1) Full background
draw.rectangle([0, 0, w, h], fill=bg_color)

# 2) Status bar area at top (~56px)
status_h = 56
draw.rectangle([0, 0, w, status_h], fill=status_color)
# thin bottom divider under status bar
draw.line([(0, status_h), (w, status_h)], fill=divider_color, width=1)

# 3) Search/header area background (rounded white search container)
search_x = 48
search_y = 72
search_w = 1344
search_h = 140  # approximate header/search container height
search_radius = 12
# subtle shadow behind search bar (drawn as a thin rounded rect beneath)
shadow_offset = 6
draw.rounded_rectangle(
    [search_x, search_y + shadow_offset, search_x + search_w, search_y + search_h + shadow_offset],
    radius=search_radius, fill=card_shadow
)
draw.rounded_rectangle(
    [search_x, search_y, search_x + search_w, search_y + search_h],
    radius=search_radius, fill=search_bg
)
# divider under header/search area
divider_y = search_y + search_h + 20
draw.line([(48, divider_y), (w - 48, divider_y)], fill=divider_color, width=1)

# 4) Top filters area separator (subtle space and line)
filters_sep_y = divider_y + 80
draw.line([(48, filters_sep_y), (w - 48, filters_sep_y)], fill=divider_color, width=1)

# 5) Event cards (rounded white cards with slight shadow)
card_radius = 26
card_shadow_offset = 8

# Card 1 (first event)
card1_x, card1_y = 48, 676
card1_w, card1_h = 1344, 1012
# shadow
draw.rounded_rectangle(
    [card1_x, card1_y + card_shadow_offset, card1_x + card1_w, card1_y + card1_h + card_shadow_offset],
    radius=card_radius, fill=card_shadow
)
# card background
draw.rounded_rectangle(
    [card1_x, card1_y, card1_x + card1_w, card1_y + card1_h],
    radius=card_radius, fill=card_bg
)
# image area background inside card1 (top portion)
card1_image_h = 420
img1_x0, img1_y0 = card1_x, card1_y
img1_x1, img1_y1 = card1_x + card1_w, card1_y + card1_image_h
# slightly rounded top corners for the image area (match card)
draw.rounded_rectangle([img1_x0, img1_y0, img1_x1, img1_y1], radius=20, fill=image_overlay1)
# subtle divider between image and text section
draw.line([(card1_x + 28, card1_y + card1_image_h + 18), (card1_x + card1_w - 28, card1_y + card1_image_h + 18)],
          fill=divider_color, width=1)

# Card 2 (second event)
card2_x, card2_y = 48, 1736
card2_w, card2_h = 1344, 1012
# shadow
draw.rounded_rectangle(
    [card2_x, card2_y + card_shadow_offset, card2_x + card2_w, card2_y + card2_h + card_shadow_offset],
    radius=card_radius, fill=card_shadow
)
# card background
draw.rounded_rectangle(
    [card2_x, card2_y, card2_x + card2_w, card2_y + card2_h],
    radius=card_radius, fill=card_bg
)
# image area background inside card2 (top portion)
card2_image_h = 420
img2_x0, img2_y0 = card2_x, card2_y
img2_x1, img2_y1 = card2_x + card2_w, card2_y + card2_image_h
draw.rounded_rectangle([img2_x0, img2_y0, img2_x1, img2_y1], radius=20, fill=image_overlay2)
# subtle divider between image and text section
draw.line([(card2_x + 28, card2_y + card2_image_h + 18), (card2_x + card2_w - 28, card2_y + card2_image_h + 18)],
          fill=divider_color, width=1)

# 6) Light separators between major sections
draw.line([(48, card1_y - 20), (w - 48, card1_y - 20)], fill=divider_color, width=1)
draw.line([(48, card2_y - 20), (w - 48, card2_y - 20)], fill=divider_color, width=1)

# 7) Bottom navigation bar background
nav_y = 2804
nav_h = 156
draw.rectangle([0, nav_y, w, nav_y + nav_h], fill=nav_bg)
# top border for nav
draw.line([(0, nav_y), (w, nav_y)], fill=nav_top_border, width=2)

# 8) Small rounded highlight behind floating action areas (do not draw icons)
# These are subtle circular backgrounds where favorite/share icons will be pasted;
# draw soft white circles to act as card-level decoration (allows icons to remain visible).
# Circle 1 near first card image (position chosen not to duplicate icon shapes exactly)
circle_radius = 44
c1_cx, c1_cy = card1_x + card1_w - 132, card1_y + card1_image_h - 36
draw.ellipse([c1_cx - circle_radius, c1_cy - circle_radius, c1_cx + circle_radius, c1_cy + circle_radius],
             fill=(255, 255, 255))
# Circle 2 near second card image
c2_cx, c2_cy = card2_x + card2_w - 132, card2_y + card2_image_h - 36
draw.ellipse([c2_cx - circle_radius, c2_cy - circle_radius, c2_cx + circle_radius, c2_cy + circle_radius],
             fill=(255, 255, 255))

# 9) Final subtle vignette/shadow at very bottom edge to ground the nav bar
draw.rectangle([0, nav_y + nav_h - 6, w, nav_y + nav_h], fill=(235, 238, 241))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/00_icon_Anytime.png
try:
    _c0 = get_crop(0, 400, 135)
    canvas.paste(_c0, (438, 390), _c0)
except Exception:
    pass
layout["Anytime"] = [438, 390, 838, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/01_icon_1_Filter.png
try:
    _c1 = get_crop(1, 372, 135)
    canvas.paste(_c1, (54, 390), _c1)
except Exception:
    pass
layout["1_Filter"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/02_icon_Sports_Fitness.png
try:
    _c2 = get_crop(2, 378, 135)
    canvas.paste(_c2, (850, 390), _c2)
except Exception:
    pass
layout["Sports_&_Fitness"] = [850, 390, 1228, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/03_icon_Favorite_button.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 1192), _c3)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2252), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2252, 1236, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/05_icon_Overflow_menu_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 1192), _c5)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2252), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2252, 1380, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 52, 65)
    canvas.paste(_c7, (1152, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1152, 0, 1204, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/08_icon_5.00.png
try:
    _c8 = get_crop(8, 117, 110)
    canvas.paste(_c8, (59, 116), _c8)
except Exception:
    pass
layout["5.00"] = [59, 116, 176, 226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/09_icon_5.00.png
try:
    _c9 = get_crop(9, 60, 65)
    canvas.paste(_c9, (180, 0), _c9)
except Exception:
    pass
layout["5.00"] = [180, 0, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/10_icon_Search_forae.png
try:
    _c10 = get_crop(10, 68, 63)
    canvas.paste(_c10, (307, 0), _c10)
except Exception:
    pass
layout["Search_forae"] = [307, 0, 375, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 66, 62)
    canvas.paste(_c11, (1212, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1212, 0, 1278, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/12_icon_5.00.png
try:
    _c12 = get_crop(12, 59, 66)
    canvas.paste(_c12, (115, 0), _c12)
except Exception:
    pass
layout["5.00"] = [115, 0, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 54, 63)
    canvas.paste(_c13, (246, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [246, 1, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 57, 59)
    canvas.paste(_c14, (1318, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1318, 0, 1375, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/15_icon_Search_forae.png
try:
    _c15 = get_crop(15, 1344, 191)
    canvas.paste(_c15, (48, 72), _c15)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/16_icon_Basics_of_Roller_Skating_balance_power.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (576, 2804), _c16)
except Exception:
    pass
layout["Basics_of_Roller_Skating_"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/17_icon_Chicago.png
try:
    _c17 = get_crop(17, 417, 144)
    canvas.paste(_c17, (0, 259), _c17)
except Exception:
    pass
layout["Chicago"] = [0, 259, 417, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/18_icon_BiG_RQVE.png
try:
    _c18 = get_crop(18, 1344, 1012)
    canvas.paste(_c18, (48, 676), _c18)
except Exception:
    pass
layout["BiG_RQVE"] = [48, 676, 1392, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/19_icon_Search_forae.png
try:
    _c19 = get_crop(19, 49, 62)
    canvas.paste(_c19, (384, 2), _c19)
except Exception:
    pass
layout["Search_forae"] = [384, 2, 433, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/20_icon_Basics_of_Roller_Skating_balance_power.png
try:
    _c20 = get_crop(20, 1344, 1012)
    canvas.paste(_c20, (48, 1736), _c20)
except Exception:
    pass
layout["Basics_of_Roller_Skating_"] = [48, 1736, 1392, 2748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/21_icon_Basics_of_Roller_Skating_balance_power.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (864, 2804), _c21)
except Exception:
    pass
layout["Basics_of_Roller_Skating_"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/22_icon_More.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (1152, 2804), _c22)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/23_icon_Promoted.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (288, 2804), _c23)
except Exception:
    pass
layout["Promoted"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 41, 61)
    canvas.paste(_c24, (1273, 0), _c24)
except Exception:
    pass
layout["icon_24"] = [1273, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/25_icon_Promoted.png
try:
    _c25 = get_crop(25, 45, 53)
    canvas.paste(_c25, (285, 2647), _c25)
except Exception:
    pass
layout["Promoted"] = [285, 2647, 330, 2700]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/26_icon_Big_Grove_Brewery_Taproom.png
try:
    _c26 = get_crop(26, 44, 57)
    canvas.paste(_c26, (284, 1585), _c26)
except Exception:
    pass
layout["Big_Grove_Brewery_&_Tapro"] = [284, 1585, 328, 1642]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/27_icon_Promoted.png
try:
    _c27 = get_crop(27, 244, 60)
    canvas.paste(_c27, (85, 2644), _c27)
except Exception:
    pass
layout["Promoted"] = [85, 2644, 329, 2704]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/28_text_5.00.png
try:
    _c28 = get_crop(28, 91, 45)
    canvas.paste(_c28, (20, 15), _c28)
except Exception:
    pass
layout["5.00"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/29_text_426_events.png
try:
    _c29 = get_crop(29, 372, 135)
    canvas.paste(_c29, (54, 390), _c29)
except Exception:
    pass
layout["426_events"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/92c22920a83749c994864397a370a984/step_11_2024_4_24_16_59_92c22920a83749c994864397a370a984-13/30_clickable_Home.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (0, 2804), _c30)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
