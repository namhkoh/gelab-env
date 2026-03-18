# page_id: page_eventbrite_892383ad791c46b6aa65c632ee882bf0_06
# screenshot: 2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-8.png
# step_index: 6/12
# task: Open Eventbrite. Search for online "Music" events happening next weekend.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# Status bar (top strip behind system icons)
status_h = 96
draw.rectangle((0, 0, 1440, status_h), fill="#D0D0D0")

# Header area (toolbar) below status bar with a subtle divider
header_top = status_h
header_bottom = 200
draw.rectangle((0, header_top, 1440, header_bottom), fill="#FFFFFF")
draw.line((24, header_bottom, 1440-24, header_bottom), fill="#E6E1EA", width=2)

# Light background card behind the date selection area (rounded, subtle purple-tinted)
card_left, card_top, card_right, card_bottom = 30, 240, 1410, 720
draw.rounded_rectangle((card_left, card_top, card_right, card_bottom),
                       radius=28, fill="#FBF8FD", outline="#EDE9F3", width=2)

# Soft shadow/edge beneath the card (subtle horizontal band)
draw.rectangle((card_left+8, card_bottom+2, card_right-8, card_bottom+8), fill="#F0E6F5")

# Separator line further down the page to hint sectioning (keeps clear of detected text areas)
sep_y = 820
draw.line((40, sep_y, 1440-40, sep_y), fill="#F1EEF4", width=1)

# Top of the bottom interactive area (thin divider above the bottom controls)
bottom_div_y = 2720
draw.line((48, bottom_div_y, 1440-48, bottom_div_y), fill="#E6E1EA", width=1)

# Very subtle footer band below everything (do not overlap detected button area)
draw.rectangle((0, 2916, 1440, 2960), fill="#F6F6F8")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_06_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-8/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_06_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-8/01_icon_5.23.png
try:
    _c1 = get_crop(1, 57, 63)
    canvas.paste(_c1, (182, 2), _c1)
except Exception:
    pass
layout["5.23"] = [182, 2, 239, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_06_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-8/02_icon_5.23.png
try:
    _c2 = get_crop(2, 57, 63)
    canvas.paste(_c2, (115, 3), _c2)
except Exception:
    pass
layout["5.23"] = [115, 3, 172, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_06_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-8/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 61, 61)
    canvas.paste(_c3, (310, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [310, 3, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_06_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-8/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 50, 59)
    canvas.paste(_c4, (249, 6), _c4)
except Exception:
    pass
layout["icon_4"] = [249, 6, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_06_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-8/05_icon_5.23.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (12, 72), _c5)
except Exception:
    pass
layout["5.23"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_06_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-8/06_icon_5.23.png
try:
    _c6 = get_crop(6, 90, 62)
    canvas.paste(_c6, (17, 3), _c6)
except Exception:
    pass
layout["5.23"] = [17, 3, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_06_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-8/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 45, 66)
    canvas.paste(_c7, (1325, 2), _c7)
except Exception:
    pass
layout["icon_7"] = [1325, 2, 1370, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_06_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 68, 67)
    canvas.paste(_c8, (1214, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1214, 0, 1282, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_06_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-8/09_icon_What_date.png
try:
    _c9 = get_crop(9, 318, 73)
    canvas.paste(_c9, (558, 111), _c9)
except Exception:
    pass
layout["What_date?"] = [558, 111, 876, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_06_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-8/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 57, 68)
    canvas.paste(_c10, (1257, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1257, 0, 1314, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_06_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-8/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 47, 64)
    canvas.paste(_c11, (384, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [384, 2, 431, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_06_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-8/12_text_Start_Date.png
try:
    _c12 = get_crop(12, 589, 144)
    canvas.paste(_c12, (48, 313), _c12)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 637, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_06_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-8/13_text_End_Date.png
try:
    _c13 = get_crop(13, 253, 67)
    canvas.paste(_c13, (45, 437), _c13)
except Exception:
    pass
layout["End_Date"] = [45, 437, 298, 504]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_06_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-8/14_clickable_Choose_a_date.png
try:
    _c14 = get_crop(14, 638, 144)
    canvas.paste(_c14, (48, 476), _c14)
except Exception:
    pass
layout["Choose_a_date"] = [48, 476, 686, 620]
