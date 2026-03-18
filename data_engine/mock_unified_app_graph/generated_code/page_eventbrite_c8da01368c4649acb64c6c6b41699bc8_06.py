# page_id: page_eventbrite_c8da01368c4649acb64c6c6b41699bc8_06
# screenshot: 2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-8.png
# step_index: 6/13
# task: Open Eventbrite. Look up "Animal" events. Filter by events happening next week. Select the first event - who is the organizer?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (canvas already provided as white, but ensure full fill)
draw.rectangle((0, 0, 1440, 2960), fill="#ffffff")

# Status bar (top area)
status_bar_height = 88
draw.rectangle((0, 0, 1440, status_bar_height), fill="#cfcfcf")

# Header strip under status bar (keeps header area visually distinct)
header_top = status_bar_height
header_bottom = 176
draw.rectangle((0, header_top, 1440, header_bottom), fill="#ffffff")

# Subtle divider under header
draw.line((40, header_bottom, 1400, header_bottom), fill="#e9e6ec", width=2)

# Light rounded content card behind the date fields (acts as grouping background)
card_left = 36
card_top = 232
card_right = 1404
card_bottom = 600
draw.rounded_rectangle(
    (card_left, card_top, card_right, card_bottom),
    radius=22,
    fill="#fbfbfd",
    outline="#efe9f5",
    width=2
)

# Soft drop shadow below the card for depth
shadow_top = card_bottom
shadow_bottom = card_bottom + 12
draw.rectangle((card_left+6, shadow_top, card_right-6, shadow_bottom), fill="#f1eef4")

# Subtle thin separator slightly below card to separate content area from remainder
sep_y = card_bottom + 48
draw.line((40, sep_y, 1400, sep_y), fill="#f0eef2", width=1)

# Large empty content area remains white (no text/icons drawn)

# Top-of-button separator above the bottom action area (keeps button area visually separated)
button_sep_y = 2720
draw.line((48, button_sep_y, 1392, button_sep_y), fill="#dcd7df", width=2)

# Outer subtle rounded frame for the screen (very light) to mimic device inset
draw.rounded_rectangle((12, 12, 1428, 2948), radius=8, outline="#f0eef2", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_06_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-8/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_06_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-8/01_icon_5.15.png
try:
    _c1 = get_crop(1, 57, 64)
    canvas.paste(_c1, (182, 1), _c1)
except Exception:
    pass
layout["5.15"] = [182, 1, 239, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_06_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-8/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 61, 61)
    canvas.paste(_c2, (310, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [310, 3, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_06_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-8/03_icon_5.15.png
try:
    _c3 = get_crop(3, 56, 65)
    canvas.paste(_c3, (117, 2), _c3)
except Exception:
    pass
layout["5.15"] = [117, 2, 173, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_06_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-8/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 50, 59)
    canvas.paste(_c4, (249, 5), _c4)
except Exception:
    pass
layout["icon_4"] = [249, 5, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_06_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-8/05_icon_5.15.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (12, 72), _c5)
except Exception:
    pass
layout["5.15"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_06_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-8/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 45, 66)
    canvas.paste(_c6, (1325, 2), _c6)
except Exception:
    pass
layout["icon_6"] = [1325, 2, 1370, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_06_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-8/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 68, 68)
    canvas.paste(_c7, (1214, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1214, 0, 1282, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_06_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-8/08_icon_What_date.png
try:
    _c8 = get_crop(8, 318, 72)
    canvas.paste(_c8, (558, 112), _c8)
except Exception:
    pass
layout["What_date?"] = [558, 112, 876, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_06_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-8/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 58, 68)
    canvas.paste(_c9, (1256, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1256, 0, 1314, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_06_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-8/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 47, 63)
    canvas.paste(_c10, (384, 3), _c10)
except Exception:
    pass
layout["icon_10"] = [384, 3, 431, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_06_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-8/11_text_5.15.png
try:
    _c11 = get_crop(11, 92, 43)
    canvas.paste(_c11, (22, 17), _c11)
except Exception:
    pass
layout["5.15"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_06_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-8/12_text_Start_Date.png
try:
    _c12 = get_crop(12, 589, 144)
    canvas.paste(_c12, (48, 313), _c12)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 637, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_06_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-8/13_text_End_Date.png
try:
    _c13 = get_crop(13, 253, 67)
    canvas.paste(_c13, (45, 437), _c13)
except Exception:
    pass
layout["End_Date"] = [45, 437, 298, 504]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_06_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-8/14_clickable_Choose_a_date.png
try:
    _c14 = get_crop(14, 638, 144)
    canvas.paste(_c14, (48, 476), _c14)
except Exception:
    pass
layout["Choose_a_date"] = [48, 476, 686, 620]
