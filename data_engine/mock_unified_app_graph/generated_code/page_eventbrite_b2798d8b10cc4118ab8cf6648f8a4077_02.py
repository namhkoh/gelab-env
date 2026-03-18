# page_id: page_eventbrite_b2798d8b10cc4118ab8cf6648f8a4077_02
# screenshot: 2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4.png
# step_index: 2/12
# task: Open Eventbrite. Search Music event in New York. Select the first one. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
bg_color = (247, 248, 250)
draw.rectangle([(0, 0), (1440, 2960)], fill=bg_color)

# Status bar (top)
status_h = 56
status_color = (158, 158, 158)
draw.rectangle([(0, 0), (1440, status_h)], fill=status_color)

# Header / search area background
header_top = status_h
header_bottom = 320
header_color = (255, 255, 255)
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=header_color)

# Thin divider under the search area (matches screenshot thin rule)
divider_color = (216, 216, 222)
divider_y = 263  # just below the detected search crop
draw.line([(48, divider_y), (1392, divider_y)], fill=divider_color, width=2)

# Subtle separator above filters row
sep_y = 360
draw.line([(32, sep_y), (1408, sep_y)], fill=(230, 230, 235), width=1)

# First event card shadow and background (big rounded card starting at detected area)
card1_x0, card1_y0 = 48, 676
card1_w, card1_h = 1344, 1175
card1_x1, card1_y1 = card1_x0 + card1_w, card1_y0 + card1_h

# Shadow
shadow_offset = 8
shadow_color = (230, 230, 235)
draw.rounded_rectangle(
    [(card1_x0 + shadow_offset, card1_y0 + shadow_offset), (card1_x1 + shadow_offset, card1_y1 + shadow_offset)],
    radius=28, fill=shadow_color
)

# Card background
card_bg = (255, 255, 255)
draw.rounded_rectangle([(card1_x0, card1_y0), (card1_x1, card1_y1)], radius=28, fill=card_bg)

# Inner image/content area for first card (dark banner area behind actual image)
img1_top = card1_y0 + 0
img1_bottom = card1_y0 + 408
image_dark = (22, 36, 80)
draw.rounded_rectangle([(card1_x0 + 16, img1_top + 16), (card1_x1 - 16, img1_bottom - 8)], radius=12, fill=image_dark)

# Subtle separator line inside card (below the hero image)
draw.line([(card1_x0 + 24, img1_bottom + 12), (card1_x1 - 24, img1_bottom + 12)], fill=(240, 240, 244), width=1)

# Second event card shadow and background (detected area)
card2_x0, card2_y0 = 48, 1899
card2_w, card2_h = 1344, 917
card2_x1, card2_y1 = card2_x0 + card2_w, card2_y0 + card2_h

# Shadow for second card
draw.rounded_rectangle(
    [(card2_x0 + shadow_offset, card2_y0 + shadow_offset), (card2_x1 + shadow_offset, card2_y1 + shadow_offset)],
    radius=22, fill=shadow_color
)

# Card background for second card
draw.rounded_rectangle([(card2_x0, card2_y0), (card2_x1, card2_y1)], radius=22, fill=card_bg)

# Inner image/content area for second card (light banner area behind actual image)
img2_top = card2_y0 + 12
img2_bottom = card2_y0 + 420
banner_light = (245, 247, 250)
draw.rounded_rectangle([(card2_x0 + 16, img2_top + 12), (card2_x1 - 16, img2_bottom - 12)], radius=10, fill=banner_light)

# Separator between the two major sections
draw.line([(24, card1_y1 + 18), (1416, card1_y1 + 18)], fill=(233, 233, 236), width=1)

# Additional subtle content-area background band (to hint grouping under filters)
band_top = 520
band_bottom = 590
band_color = (250, 251, 253)
draw.rectangle([(0, band_top), (1440, band_bottom)], fill=band_color)

# Bottom navigation bar background and top border
nav_top = 2804
nav_bottom = 2960
nav_bg = (255, 255, 255)
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill=nav_bg)
draw.line([(0, nav_top), (1440, nav_top)], fill=(225, 225, 230), width=2)

# Small indicator for selected nav item (subtle underline area, not an icon)
indicator_w = 72
indicator_h = 4
indicator_x = 360  # arbitrary position; icons will be pasted later
indicator_y = nav_top + 10
draw.rounded_rectangle(
    [(indicator_x, indicator_y), (indicator_x + indicator_w, indicator_y + indicator_h)],
    radius=2, fill=(230, 90, 30)
)

# Top app-level thin shadow under status bar for depth
draw.line([(0, status_h), (1440, status_h)], fill=(200, 200, 205), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 103)
    canvas.paste(_c1, (425, 410), _c1)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1036, 410), _c2)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/04_icon_Foo.png
try:
    _c4 = get_crop(4, 121, 109)
    canvas.paste(_c4, (1284, 407), _c4)
except Exception:
    pass
layout["Foo"] = [1284, 407, 1405, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/05_icon_Jpcio.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Jpcio"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/06_icon_Jpcio.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["Jpcio"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/07_icon_9.18.png
try:
    _c7 = get_crop(7, 128, 120)
    canvas.paste(_c7, (53, 111), _c7)
except Exception:
    pass
layout["9.18"] = [53, 111, 181, 231]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 55, 61)
    canvas.paste(_c8, (247, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [247, 1, 302, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/09_icon_WEDNESDAY_6_PM.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1092, 2415), _c9)
except Exception:
    pass
layout["WEDNESDAY_6_PM"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/10_icon_9.18.png
try:
    _c10 = get_crop(10, 55, 62)
    canvas.paste(_c10, (182, 0), _c10)
except Exception:
    pass
layout["9.18"] = [182, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/11_icon_Online.png
try:
    _c11 = get_crop(11, 377, 144)
    canvas.paste(_c11, (0, 259), _c11)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/12_icon_WEDNESDAY_6_PM.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1236, 2415), _c12)
except Exception:
    pass
layout["WEDNESDAY_6_PM"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 60, 60)
    canvas.paste(_c13, (1317, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1317, 0, 1377, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 73, 60)
    canvas.paste(_c14, (1209, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1209, 0, 1282, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 58, 61)
    canvas.paste(_c15, (312, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [312, 1, 370, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/16_icon_9.18.png
try:
    _c16 = get_crop(16, 54, 64)
    canvas.paste(_c16, (116, 0), _c16)
except Exception:
    pass
layout["9.18"] = [116, 0, 170, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/17_icon_Webinar.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (288, 2804), _c17)
except Exception:
    pass
layout["Webinar"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/18_icon_Search_forae.png
try:
    _c18 = get_crop(18, 50, 60)
    canvas.paste(_c18, (383, 2), _c18)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 433, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/19_icon_Promoted.png
try:
    _c19 = get_crop(19, 252, 66)
    canvas.paste(_c19, (76, 1742), _c19)
except Exception:
    pass
layout["Promoted"] = [76, 1742, 328, 1808]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/20_icon_Search_forae.png
try:
    _c20 = get_crop(20, 1344, 191)
    canvas.paste(_c20, (48, 72), _c20)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/21_icon_deepcowl.png
try:
    _c21 = get_crop(21, 1344, 1175)
    canvas.paste(_c21, (48, 676), _c21)
except Exception:
    pass
layout["deepcowl"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/22_icon_a_home_with_ZERO_down.png
try:
    _c22 = get_crop(22, 1344, 917)
    canvas.paste(_c22, (48, 1899), _c22)
except Exception:
    pass
layout["a_home_with_ZERO_down"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 41, 61)
    canvas.paste(_c23, (1273, 0), _c23)
except Exception:
    pass
layout["icon_23"] = [1273, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/24_icon_Webinar.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (0, 2804), _c24)
except Exception:
    pass
layout["Webinar"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/25_icon_Free.png
try:
    _c25 = get_crop(25, 125, 76)
    canvas.paste(_c25, (91, 2592), _c25)
except Exception:
    pass
layout["Free"] = [91, 2592, 216, 2668]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/26_icon_Active_Military_Veterans_VA_Homebuyer.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (864, 2804), _c26)
except Exception:
    pass
layout["Active_Military_&_Veteran"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/27_icon_Active_Military_Veterans_VA_Homebuyer.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (576, 2804), _c27)
except Exception:
    pass
layout["Active_Military_&_Veteran"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/28_icon_Promoted.png
try:
    _c28 = get_crop(28, 41, 59)
    canvas.paste(_c28, (285, 1748), _c28)
except Exception:
    pass
layout["Promoted"] = [285, 1748, 326, 1807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/29_icon_More.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (1152, 2804), _c29)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/30_icon_Free.png
try:
    _c30 = get_crop(30, 125, 77)
    canvas.paste(_c30, (90, 1369), _c30)
except Exception:
    pass
layout["Free"] = [90, 1369, 215, 1446]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/31_text_9.18.png
try:
    _c31 = get_crop(31, 91, 43)
    canvas.paste(_c31, (20, 17), _c31)
except Exception:
    pass
layout["9.18"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/32_text_10_000_events.png
try:
    _c32 = get_crop(32, 359, 103)
    canvas.paste(_c32, (54, 410), _c32)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/33_text_Tow.png
try:
    _c33 = get_crop(33, 69, 37)
    canvas.paste(_c33, (106, 698), _c33)
except Exception:
    pass
layout["Tow"] = [106, 698, 175, 735]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/34_text_SEo_TRACK.png
try:
    _c34 = get_crop(34, 152, 36)
    canvas.paste(_c34, (186, 700), _c34)
except Exception:
    pass
layout["SEo_TRACK"] = [186, 700, 338, 736]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/35_text_deepaow.png
try:
    _c35 = get_crop(35, 149, 38)
    canvas.paste(_c35, (406, 700), _c35)
except Exception:
    pass
layout["deepaow"] = [406, 700, 555, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/36_text_Slo_TRaCK.png
try:
    _c36 = get_crop(36, 149, 36)
    canvas.paste(_c36, (584, 700), _c36)
except Exception:
    pass
layout["Slo_TRaCK"] = [584, 700, 733, 736]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/37_text_deepciam.png
try:
    _c37 = get_crop(37, 149, 38)
    canvas.paste(_c37, (811, 700), _c37)
except Exception:
    pass
layout["deepciam"] = [811, 700, 960, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/38_text_seo_TracK.png
try:
    _c38 = get_crop(38, 150, 32)
    canvas.paste(_c38, (992, 703), _c38)
except Exception:
    pass
layout["seo_TracK"] = [992, 703, 1142, 735]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/39_text_deepao.png
try:
    _c39 = get_crop(39, 121, 41)
    canvas.paste(_c39, (1212, 699), _c39)
except Exception:
    pass
layout["deepao"] = [1212, 699, 1333, 740]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/40_text_SEO_MASTERCLASS_HOW_TO_RANKANY.png
try:
    _c40 = get_crop(40, 1344, 1175)
    canvas.paste(_c40, (48, 676), _c40)
except Exception:
    pass
layout["SEO_MASTERCLASS:_HOW_TO_R"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/41_text_WEBSITE.png
try:
    _c41 = get_crop(41, 255, 60)
    canvas.paste(_c41, (94, 1535), _c41)
except Exception:
    pass
layout["WEBSITE"] = [94, 1535, 349, 1595]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/42_text_Wed_Mar_20_._5_00_PM_GMT.png
try:
    _c42 = get_crop(42, 537, 50)
    canvas.paste(_c42, (93, 1619), _c42)
except Exception:
    pass
layout["Wed,_Mar_20_._5:00_PM_GMT"] = [93, 1619, 630, 1669]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/43_text_Online.png
try:
    _c43 = get_crop(43, 129, 45)
    canvas.paste(_c43, (91, 1687), _c43)
except Exception:
    pass
layout["Online"] = [91, 1687, 220, 1732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/44_text_Edz.png
try:
    _c44 = get_crop(44, 166, 65)
    canvas.paste(_c44, (132, 2502), _c44)
except Exception:
    pass
layout["Edz"] = [132, 2502, 298, 2567]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/45_text_Cf.png
try:
    _c45 = get_crop(45, 151, 52)
    canvas.paste(_c45, (432, 2508), _c45)
except Exception:
    pass
layout["Cf"] = [432, 2508, 583, 2560]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_02_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-4/46_text_Active_Military_Veterans_VA_Homebuyer.png
try:
    _c46 = get_crop(46, 1344, 917)
    canvas.paste(_c46, (48, 1899), _c46)
except Exception:
    pass
layout["Active_Military_&_Veteran"] = [48, 1899, 1392, 2816]
