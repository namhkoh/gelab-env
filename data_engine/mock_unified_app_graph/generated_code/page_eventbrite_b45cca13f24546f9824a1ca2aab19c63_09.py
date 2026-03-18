# page_id: page_eventbrite_b45cca13f24546f9824a1ca2aab19c63_09
# screenshot: 2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11.png
# step_index: 9/11
# task: Open Eventbrite. Search for "Art". Filter for events in New York. Select first recommended event. Save it to wishlist. What is the duration of the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (1440x2960 RGB). font_sm, font_md, font_lg, font_xl available.
w, h = canvas.size

# Colors (matching the screenshot's neutral / light UI)
bg_color = (255, 255, 255)           # main background (white)
status_bar_color = (156, 162, 166)   # muted grey for status bar
header_divider = (226, 229, 233)     # subtle divider lines
card_shadow = (235, 239, 243)        # shadow under cards
card_bg = (255, 255, 255)            # card background (white)
image_dark = (28, 28, 28)            # dark image placeholder
image_deep_green = (20, 54, 46)      # second image deep green tone
nav_bar_bg = (250, 250, 250)         # bottom nav background
muted_sep = (240, 243, 246)

# Fill entire background (canvas is already white, but ensure dominant color)
draw.rectangle([0, 0, w, h], fill=bg_color)

# Status bar area (top) ~ 72px high
status_h = 72
draw.rectangle([0, 0, w, status_h], fill=status_bar_color)

# Header / search area background (below status bar).
# Keep it visually distinct but mostly white; draw a subtle top/bottom divider.
header_top = status_h
header_bottom = 264  # approximate bottom of search/header area (based on screenshot)
draw.rectangle([0, header_top, w, header_bottom], fill=bg_color)
draw.line([24, header_bottom, w-24, header_bottom], fill=header_divider, width=1)

# Thin divider under the row of filters (approx y around 360)
filters_div_y = 360
draw.line([24, filters_div_y, w-24, filters_div_y], fill=header_divider, width=1)

# Large horizontal spacer area for the "8,638 events" label area (do not draw text)
events_area_top = filters_div_y + 20
events_area_bottom = events_area_top + 40
# subtle background is same as canvas; add a faint separator below the event count region
draw.line([24, events_area_bottom+12, w-24, events_area_bottom+12], fill=muted_sep, width=1)

# Event card 1: background rounded rectangle + dark image area
card_x = 48
card_w = w - (card_x * 2)  # 1344 as seen in detections
card1_y = 480
card1_h = 420
card1_box = [card_x, card1_y, card_x + card_w, card1_y + card1_h]

# subtle shadow
shadow_offset = 8
draw.rounded_rectangle(
    [card1_box[0], card1_box[1] + shadow_offset, card1_box[2], card1_box[3] + shadow_offset],
    radius=26, fill=card_shadow
)
# card background
draw.rounded_rectangle(card1_box, radius=26, fill=card_bg)

# image placeholder inside card (full-width, rounded corners)
img_margin = 12
img_box = [
    card1_box[0] + img_margin,
    card1_box[1] + img_margin,
    card1_box[2] - img_margin,
    card1_box[1] + 220  # image height portion
]
draw.rounded_rectangle(img_box, radius=16, fill=image_dark)

# subtle divider between image and meta area
sep_y = img_box[3] + 14
draw.line([card1_box[0] + 8, sep_y, card1_box[2] - 8, sep_y], fill=header_divider, width=1)

# Event card 2: spaced below card1
card2_y = card1_box[3] + 48
card2_h = 420
card2_box = [card_x, card2_y, card_x + card_w, card2_y + card2_h]

# shadow and card background
draw.rounded_rectangle(
    [card2_box[0], card2_box[1] + shadow_offset, card2_box[2], card2_box[3] + shadow_offset],
    radius=26, fill=card_shadow
)
draw.rounded_rectangle(card2_box, radius=26, fill=card_bg)

# second card image placeholder (colored background to hint at image)
img2_box = [
    card2_box[0] + img_margin,
    card2_box[1] + img_margin,
    card2_box[2] - img_margin,
    card2_box[1] + 220
]
# draw a two-tone band to suggest a colorful event image without drawing content
draw.rounded_rectangle(img2_box, radius=16, fill=image_deep_green)
# subtle lighter top band
draw.rectangle([img2_box[0], img2_box[1], img2_box[2], img2_box[1]+56], fill=(48, 90, 78))

# small separator below second image
sep2_y = img2_box[3] + 14
draw.line([card2_box[0] + 8, sep2_y, card2_box[2] - 8, sep2_y], fill=header_divider, width=1)

# Global thin separators between major sections
draw.line([24, card1_box[3] + 12, w-24, card1_box[3] + 12], fill=muted_sep, width=1)
draw.line([24, card2_box[3] + 12, w-24, card2_box[3] + 12], fill=muted_sep, width=1)

# Bottom navigation bar area (~120px high)
nav_h = 120
nav_top = h - nav_h
draw.rectangle([0, nav_top, w, h], fill=nav_bar_bg)
# top divider for nav bar
draw.line([24, nav_top, w-24, nav_top], fill=header_divider, width=1)

# Left and right safe margins (subtle vertical guides) - very faint
draw.line([24, header_top, 24, h - nav_h], fill=(250, 250, 250), width=1)
draw.line([w-24, header_top, w-24, h - nav_h], fill=(250, 250, 250), width=1)

# Done - structure, backgrounds and separators have been drawn.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/04_icon_Foo.png
try:
    _c4 = get_crop(4, 147, 110)
    canvas.paste(_c4, (1283, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1283, 406, 1430, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/05_icon_Juia.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2252), _c5)
except Exception:
    pass
layout["Juia"] = [1092, 2252, 1236, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/06_icon_Juia.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2252), _c6)
except Exception:
    pass
layout["Juia"] = [1236, 2252, 1380, 2396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1092, 1192), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/09_icon_Foo.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/10_icon_7.06.png
try:
    _c10 = get_crop(10, 124, 115)
    canvas.paste(_c10, (54, 113), _c10)
except Exception:
    pass
layout["7.06"] = [54, 113, 178, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 53, 66)
    canvas.paste(_c11, (1151, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1151, 0, 1204, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/12_icon_7.06.png
try:
    _c12 = get_crop(12, 62, 65)
    canvas.paste(_c12, (179, 0), _c12)
except Exception:
    pass
layout["7.06"] = [179, 0, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/13_icon_Art.png
try:
    _c13 = get_crop(13, 68, 63)
    canvas.paste(_c13, (308, 0), _c13)
except Exception:
    pass
layout["Art"] = [308, 0, 376, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 98, 63)
    canvas.paste(_c14, (1212, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1212, 0, 1310, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/15_icon_7.06.png
try:
    _c15 = get_crop(15, 62, 66)
    canvas.paste(_c15, (114, 0), _c15)
except Exception:
    pass
layout["7.06"] = [114, 0, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/16_icon_Art.png
try:
    _c16 = get_crop(16, 55, 65)
    canvas.paste(_c16, (246, 0), _c16)
except Exception:
    pass
layout["Art"] = [246, 0, 301, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 56, 61)
    canvas.paste(_c17, (1319, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1319, 0, 1375, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/18_icon_New_York.png
try:
    _c18 = get_crop(18, 434, 144)
    canvas.paste(_c18, (0, 259), _c18)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/19_icon_I_00_PM_EDT.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["I:00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/20_icon_aclh.png
try:
    _c20 = get_crop(20, 1344, 1080)
    canvas.paste(_c20, (48, 1736), _c20)
except Exception:
    pass
layout["[aclh"] = [48, 1736, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/21_icon_THE_LIFE_DEATH_OF_ART.png
try:
    _c21 = get_crop(21, 1344, 1012)
    canvas.paste(_c21, (48, 676), _c21)
except Exception:
    pass
layout["THE_LIFE_&_DEATH_OF_ART"] = [48, 676, 1392, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 53, 61)
    canvas.paste(_c22, (383, 2), _c22)
except Exception:
    pass
layout["icon_22"] = [383, 2, 436, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/23_icon_Elegance_Spirits_A_Charcuterie_and_Cockt.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["Elegance_&_Spirits:_A_Cha"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/24_icon_I_00_PM_EDT.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["I:00_PM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/25_icon_Promoted.png
try:
    _c25 = get_crop(25, 280, 67)
    canvas.paste(_c25, (54, 1580), _c25)
except Exception:
    pass
layout["Promoted"] = [54, 1580, 334, 1647]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/26_icon_Elegance_Spirits_A_Charcuterie_and_Cockt.png
try:
    _c26 = get_crop(26, 1344, 1080)
    canvas.paste(_c26, (48, 1736), _c26)
except Exception:
    pass
layout["Elegance_&_Spirits:_A_Cha"] = [48, 1736, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/27_icon_Fae_WLCI.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (1152, 2804), _c27)
except Exception:
    pass
layout["Fae_WLCI"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/28_text_7.06.png
try:
    _c28 = get_crop(28, 89, 41)
    canvas.paste(_c28, (22, 17), _c28)
except Exception:
    pass
layout["7.06"] = [22, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/29_text_Art.png
try:
    _c29 = get_crop(29, 119, 65)
    canvas.paste(_c29, (205, 138), _c29)
except Exception:
    pass
layout["Art"] = [205, 138, 324, 203]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/30_text_8_638_events.png
try:
    _c30 = get_crop(30, 359, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["8,638_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/31_text_THE_LIFE_DEATH_OF_ART.png
try:
    _c31 = get_crop(31, 1344, 1012)
    canvas.paste(_c31, (48, 676), _c31)
except Exception:
    pass
layout["THE_LIFE_&_DEATH_OF_ART"] = [48, 676, 1392, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/32_text_Sun_Apr_28_._2_00_PM_EDT.png
try:
    _c32 = get_crop(32, 507, 55)
    canvas.paste(_c32, (91, 1454), _c32)
except Exception:
    pass
layout["Sun,_Apr_28_._2:00_PM_EDT"] = [91, 1454, 598, 1509]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/33_text_JACK.png
try:
    _c33 = get_crop(33, 106, 43)
    canvas.paste(_c33, (91, 1525), _c33)
except Exception:
    pass
layout["JACK"] = [91, 1525, 197, 1568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/34_text_Sat_Apr_27.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (0, 2804), _c34)
except Exception:
    pass
layout["Sat,_Apr_27_="] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/35_text_I_00_PM_EDT.png
try:
    _c35 = get_crop(35, 288, 156)
    canvas.paste(_c35, (288, 2804), _c35)
except Exception:
    pass
layout["I:00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/36_text_367E_Oth_St.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (0, 2804), _c36)
except Exception:
    pass
layout["367E_]Oth_St"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_09_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-11/37_clickable_Art.png
try:
    _c37 = get_crop(37, 1344, 191)
    canvas.paste(_c37, (48, 72), _c37)
except Exception:
    pass
layout["Art"] = [48, 72, 1392, 263]
