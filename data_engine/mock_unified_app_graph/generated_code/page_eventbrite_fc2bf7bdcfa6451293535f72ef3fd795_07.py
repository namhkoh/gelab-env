# page_id: page_eventbrite_fc2bf7bdcfa6451293535f72ef3fd795_07
# screenshot: 2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9.png
# step_index: 7/8
# task: Open Eventbrite. Search for events by 'Music' under online events. Choose the second event in the list. Get the event's duration information.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
draw.rectangle((0, 0, 1440, 2960), fill=(247, 247, 250))  # soft off-white background

# Status bar (top ~60px)
status_h = 60
draw.rectangle((0, 0, 1440, status_h), fill=(160, 160, 160))  # muted grey status bar

# Header / toolbar area below status bar
toolbar_top = status_h
toolbar_bottom = 220
draw.rectangle((0, toolbar_top, 1440, toolbar_bottom), fill=(255, 255, 255))  # white toolbar

# Thin divider under the toolbar / search area
divider_y = 263  # matches visual divider under search area
draw.line((48, divider_y, 1392, divider_y), fill=(220, 220, 223), width=2)

# Subtle horizontal rule separating filter area and content
draw.line((36, 420, 1404, 420), fill=(235, 235, 238), width=1)

# First event card background (shadow + rounded white card)
card1_left = 36
card1_top = 640
card1_right = 1404
card1_bottom = 1810
shadow_offset = 8

# shadow (soft flat shadow look)
draw.rounded_rectangle(
    (card1_left + shadow_offset, card1_top + shadow_offset, card1_right + shadow_offset, card1_bottom + shadow_offset),
    radius=28,
    fill=(235, 235, 238)
)

# main card
draw.rounded_rectangle((card1_left, card1_top, card1_right, card1_bottom), radius=28, fill=(255, 255, 255))

# subtle inner separator on card (where image ends and text section begins)
# The event image for this card is pasted at y=676 with height 1091 -> image bottom = 676+1091 = 1767
image1_bottom = 676 + 1091
sep_y = image1_bottom + 12
draw.line((card1_left + 24, sep_y, card1_right - 24, sep_y), fill=(245, 245, 247), width=1)

# Second event card background (shadow + rounded white card)
card2_left = 36
card2_top = 1796
card2_right = 1404
card2_bottom = 2780
draw.rounded_rectangle(
    (card2_left + shadow_offset, card2_top + shadow_offset, card2_right + shadow_offset, card2_bottom + shadow_offset),
    radius=24,
    fill=(235, 235, 238)
)
draw.rounded_rectangle((card2_left, card2_top, card2_right, card2_bottom), radius=24, fill=(255, 255, 255))

# inner separator for second card (where its image ends)
image2_bottom = 1815 + 1001  # y of image top + its height
sep2_y = image2_bottom + 10
draw.line((card2_left + 20, sep2_y, card2_right - 20, sep2_y), fill=(245, 245, 247), width=1)

# Light section divider lines between list items (subtle)
draw.line((48, 1840, 1392, 1840), fill=(240, 240, 243), width=1)
draw.line((48, 2760, 1392, 2760), fill=(240, 240, 243), width=1)

# Bottom navigation bar background
nav_top = 2804
nav_bottom = 2960
draw.rectangle((0, nav_top, 1440, nav_bottom), fill=(255, 255, 255))
# top border for nav
draw.line((0, nav_top, 1440, nav_top), fill=(225, 225, 228), width=2)

# Very subtle left/right page padding shadow to frame the content area
draw.line((24, toolbar_bottom + 8, 24, nav_top - 8), fill=(250, 250, 251), width=1)
draw.line((1416, toolbar_bottom + 8, 1416, nav_top - 8), fill=(250, 250, 251), width=1)

# Decorative subtle accent band behind the top of the first card (doesn't duplicate any icon/text)
accent_band_top = card1_top + 6
accent_band_bottom = card1_top + 36
draw.rectangle((card1_left + 8, accent_band_top, card1_right - 8, accent_band_bottom), fill=(250, 246, 250))

# Small rounded chip background placeholders (no text/icons drawn) to suggest filter area blocks
# These are intentionally abstract and do not match any detected element shapes exactly.
chip_y = 344
chip_w = 220
for i, x in enumerate((72, 340, 820)):
    left = x
    top = chip_y
    right = left + chip_w
    bottom = top + 64
    draw.rounded_rectangle((left, top, right, bottom), radius=36, fill=(244, 248, 255) if i != 1 else (233, 243, 255))

# End of layout drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 135)
    canvas.paste(_c0, (850, 390), _c0)
except Exception:
    pass
layout["Music"] = [850, 390, 1037, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 135)
    canvas.paste(_c1, (438, 390), _c1)
except Exception:
    pass
layout["Anytime"] = [438, 390, 838, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/02_icon_1_Filter.png
try:
    _c2 = get_crop(2, 372, 135)
    canvas.paste(_c2, (54, 390), _c2)
except Exception:
    pass
layout["1_Filter"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/03_icon_Favorite_button.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 2331), _c3)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2331, 1236, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/04_icon_Overflow_menu_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1236, 2331), _c4)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2331, 1380, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/05_icon_PAsS.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["PAsS"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/06_icon_THE_BALLrooM.png
try:
    _c6 = get_crop(6, 1344, 1091)
    canvas.paste(_c6, (48, 676), _c6)
except Exception:
    pass
layout["THE_BALLrooM"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/08_icon_8.04.png
try:
    _c8 = get_crop(8, 124, 113)
    canvas.paste(_c8, (55, 114), _c8)
except Exception:
    pass
layout["8.04"] = [55, 114, 179, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/09_icon_May.png
try:
    _c9 = get_crop(9, 1344, 1091)
    canvas.paste(_c9, (48, 676), _c9)
except Exception:
    pass
layout["May"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/10_icon_Search_forae.png
try:
    _c10 = get_crop(10, 68, 63)
    canvas.paste(_c10, (307, 0), _c10)
except Exception:
    pass
layout["Search_forae"] = [307, 0, 375, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/11_icon_8.04.png
try:
    _c11 = get_crop(11, 61, 65)
    canvas.paste(_c11, (180, 0), _c11)
except Exception:
    pass
layout["8.04"] = [180, 0, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/12_icon_8.04.png
try:
    _c12 = get_crop(12, 60, 66)
    canvas.paste(_c12, (115, 0), _c12)
except Exception:
    pass
layout["8.04"] = [115, 0, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 55, 64)
    canvas.paste(_c13, (246, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [246, 0, 301, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 63, 59)
    canvas.paste(_c14, (1318, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1318, 0, 1381, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 67, 61)
    canvas.paste(_c15, (1208, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1208, 0, 1275, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/16_icon_Search_forae.png
try:
    _c16 = get_crop(16, 1344, 191)
    canvas.paste(_c16, (48, 72), _c16)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/17_icon_Online.png
try:
    _c17 = get_crop(17, 377, 144)
    canvas.paste(_c17, (0, 259), _c17)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/18_icon_Q.Q0_DiA_CDT.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (288, 2804), _c18)
except Exception:
    pass
layout["Q.Q0_DiA_CDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 41, 61)
    canvas.paste(_c19, (1273, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1273, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/20_icon_RNB_VIBES_reloaded_OASIS_FIRST_R_B.png
try:
    _c20 = get_crop(20, 1344, 1001)
    canvas.paste(_c20, (48, 1815), _c20)
except Exception:
    pass
layout["RNB_VIBES_reloaded@OASIS_"] = [48, 1815, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/21_icon_Search_forae.png
try:
    _c21 = get_crop(21, 50, 63)
    canvas.paste(_c21, (383, 1), _c21)
except Exception:
    pass
layout["Search_forae"] = [383, 1, 433, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/22_icon_RNB_VIBES_reloaded_OASIS_FIRST_R_B.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (864, 2804), _c22)
except Exception:
    pass
layout["RNB_VIBES_reloaded@OASIS_"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/23_icon_RNB_VIBES_reloaded_OASIS_FIRST_R_B.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (576, 2804), _c23)
except Exception:
    pass
layout["RNB_VIBES_reloaded@OASIS_"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/24_icon_Promoted.png
try:
    _c24 = get_crop(24, 43, 60)
    canvas.paste(_c24, (286, 1661), _c24)
except Exception:
    pass
layout["Promoted"] = [286, 1661, 329, 1721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/25_icon_Cot.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["Cot"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/26_icon_More.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (1152, 2804), _c26)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/27_text_8.04.png
try:
    _c27 = get_crop(27, 94, 45)
    canvas.paste(_c27, (20, 15), _c27)
except Exception:
    pass
layout["8.04"] = [20, 15, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/28_text_357_events.png
try:
    _c28 = get_crop(28, 372, 135)
    canvas.paste(_c28, (54, 390), _c28)
except Exception:
    pass
layout["357_events"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/29_text_Wed_May_15.png
try:
    _c29 = get_crop(29, 257, 57)
    canvas.paste(_c29, (93, 1533), _c29)
except Exception:
    pass
layout["Wed,_May_15"] = [93, 1533, 350, 1590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/30_text_3.00_PM_EDT.png
try:
    _c30 = get_crop(30, 253, 50)
    canvas.paste(_c30, (368, 1533), _c30)
except Exception:
    pass
layout["3.00_PM_EDT"] = [368, 1533, 621, 1583]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/31_text_Online.png
try:
    _c31 = get_crop(31, 129, 45)
    canvas.paste(_c31, (91, 1604), _c31)
except Exception:
    pass
layout["Online"] = [91, 1604, 220, 1649]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/32_text_Cot.png
try:
    _c32 = get_crop(32, 72, 36)
    canvas.paste(_c32, (92, 2777), _c32)
except Exception:
    pass
layout["Cot"] = [92, 2777, 164, 2813]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/33_text_AAzu_11.png
try:
    _c33 = get_crop(33, 145, 27)
    canvas.paste(_c33, (178, 2782), _c33)
except Exception:
    pass
layout["AAzu_11"] = [178, 2782, 323, 2809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_07_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-9/34_text_Q.Q0_DiA_CDT.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (288, 2804), _c34)
except Exception:
    pass
layout["Q.Q0_DiA_CDT"] = [288, 2804, 576, 2960]
