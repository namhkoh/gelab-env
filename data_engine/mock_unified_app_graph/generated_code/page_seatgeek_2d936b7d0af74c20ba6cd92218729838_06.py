# page_id: page_seatgeek_2d936b7d0af74c20ba6cd92218729838_06
# screenshot: 2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9.png
# step_index: 6/12
# task: Open SeatGeek. Track "Los Angeles Clippers" and "Golden State Warriors".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background canvas is provided as `canvas` (1440x2960 RGB) and `draw` (ImageDraw).
# Fonts available: font_sm, font_md, font_lg, font_xl
# This script paints the page background, status bar, header cards, section cards, and separators.

W, H = canvas.size

# Base background (slightly warm white to match screenshot)
draw.rectangle([(0, 0), (W, H)], fill="#fbfbfb")

# --- Top hero area (image background) ---
hero_height = 520
hero_color = "#2b2b2b"  # dark muted to match arena photo background
draw.rectangle([(0, 0), (W, hero_height)], fill=hero_color)

# Status bar overlay at very top (darker band behind status icons)
status_h = 110
draw.rectangle([(0, 0), (W, status_h)], fill="#0f0f0f")

# Subtle vignette/top gradient feel: a semi-opaque darker bar below status for visual depth
# (simulate by a darker thin band)
draw.rectangle([(0, status_h - 6), (W, status_h + 6)], fill="#111111")

# --- Title / header card below hero image ---
card_pad_h = 40
title_card_top = hero_height - 0  # start just below hero
title_card_left = 36
title_card_right = W - 36
title_card_height = 220
title_card_bottom = title_card_top + title_card_height

# Drop shadow for title card (soft grey rectangle offset)
shadow_offset = 6
shadow_box = [title_card_left, title_card_top + shadow_offset,
              title_card_right, title_card_bottom + shadow_offset]
draw.rounded_rectangle(shadow_box, radius=16, fill="#e9e9e9")

# Main title card (white)
title_card_box = [title_card_left, title_card_top, title_card_right, title_card_bottom]
draw.rounded_rectangle(title_card_box, radius=16, fill="#ffffff")

# Thin divider at bottom of title card
draw.line([(title_card_left + 12, title_card_bottom), (title_card_right - 12, title_card_bottom)],
          fill="#e6e6e6", width=1)

# --- Small info row / notice section (No Games / buyer guarantee area) ---
# This is a full-width flat section under the title card
notice_top = title_card_bottom + 34
notice_left = 0
notice_right = W
notice_height = 126
notice_bottom = notice_top + notice_height

# Use plain white (matches list background) and a faint border on top
draw.rectangle([(notice_left, notice_top), (notice_right, notice_bottom)], fill="#ffffff")
draw.line([(16, notice_top), (W - 16, notice_top)], fill="#f0f0f0", width=1)
draw.line([(16, notice_bottom), (W - 16, notice_bottom)], fill="#f0f0f0", width=1)

# --- Separator before list ---
sep_y = notice_bottom + 28
draw.line([(16, sep_y), (W - 16, sep_y)], fill="#e9e9e9", width=1)

# --- "All Games" list container ---
list_top = sep_y + 28
list_left = 16
list_right = W - 16
list_bottom = H - 8

# Drop shadow for list container
list_shadow = [list_left, list_top + 6, list_right, list_bottom + 6]
draw.rounded_rectangle(list_shadow, radius=20, fill="#eef0f1")

# Main list background (rounded)
draw.rounded_rectangle([list_left, list_top, list_right, list_bottom], radius=20, fill="#ffffff")

# Inner padding and subtle separators for individual rows.
# Based on detected row positions in this UI, draw separators across the list.
# Known row top positions (approx): 1596, 1963, 2330, 2697
# These are absolute coordinates from the original screenshot; translate them if needed.
separators = [1596, 1963, 2330, 2697]
for y in separators:
    if list_top + 8 < y < list_bottom - 8:
        draw.line([(list_left + 24, y), (list_right - 24, y)], fill="#f0f0f0", width=1)

# Also draw a faint left inset divider to visually separate date-pill area (no pills drawn)
# Draw a shadowed vertical guideline area on the left where date-pill lives (subtle only)
pill_area_x = list_left + 160
draw.rectangle([(list_left + 24, list_top + 16), (pill_area_x, list_bottom - 24)], fill=None, outline="#fafafa")
# Add a very subtle drop for the pill column (light gradient simulated by small rect)
draw.rectangle([(list_left + 24, list_top + 16), (pill_area_x, list_top + 40)], fill="#fbfbfb")

# Top and bottom edges for the list to separate from surrounding white
draw.line([(list_left + 12, list_top), (list_right - 12, list_top)], fill="#f5f5f5", width=1)
draw.line([(list_left + 12, list_bottom), (list_right - 12, list_bottom)], fill="#f5f5f5", width=1)

# --- Final subtle overlays / polish ---
# Slight horizontal band under hero image to transition into content area
transition_y = hero_height - 12
draw.rectangle([(0, transition_y), (W, transition_y + 24)], fill="#1f1f1f")

# Light right-side safe margin shadow to anchor content
draw.rectangle([(W - 12, 0), (W, H)], fill="#ffffff")

# No text or icons are drawn here — those are supplied later and will be pasted on top.
# This completes the background, structural cards, and separators.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_06_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9/00_icon_23.png
try:
    _c0 = get_crop(0, 1440, 367)
    canvas.paste(_c0, (0, 1963), _c0)
except Exception:
    pass
layout["23"] = [0, 1963, 1440, 2330]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_06_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9/01_icon_28.png
try:
    _c1 = get_crop(1, 1440, 263)
    canvas.paste(_c1, (0, 2697), _c1)
except Exception:
    pass
layout["28"] = [0, 2697, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_06_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9/02_icon_Track_this_performer.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1104, 84), _c2)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_06_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9/03_icon_Los_Angeles_CA.png
try:
    _c3 = get_crop(3, 1440, 367)
    canvas.paste(_c3, (0, 1596), _c3)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [0, 1596, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_06_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9/04_icon_26.png
try:
    _c4 = get_crop(4, 1440, 367)
    canvas.paste(_c4, (0, 2330), _c4)
except Exception:
    pass
layout["26"] = [0, 2330, 1440, 2697]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_06_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9/05_icon_Mavericks_at_LA_Clippers_Game_2.png
try:
    _c5 = get_crop(5, 1440, 367)
    canvas.paste(_c5, (0, 1963), _c5)
except Exception:
    pass
layout["Mavericks_at_LA_Clippers_"] = [0, 1963, 1440, 2330]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_06_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9/06_icon_21.png
try:
    _c6 = get_crop(6, 1440, 367)
    canvas.paste(_c6, (0, 1596), _c6)
except Exception:
    pass
layout["21"] = [0, 1596, 1440, 1963]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_06_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9/07_icon_Share_this_performer.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1260, 84), _c7)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_06_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9/08_icon_Western_Conference_First_Round_LA_Clippe.png
try:
    _c8 = get_crop(8, 1440, 367)
    canvas.paste(_c8, (0, 2330), _c8)
except Exception:
    pass
layout["Western_Conference_First_"] = [0, 2330, 1440, 2697]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_06_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9/09_icon_6.53_W.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (36, 84), _c9)
except Exception:
    pass
layout["6.53_W"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_06_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9/10_icon_Western_Conference_First_Round_LA_Clippe.png
try:
    _c10 = get_crop(10, 1440, 263)
    canvas.paste(_c10, (0, 2697), _c10)
except Exception:
    pass
layout["Western_Conference_First_"] = [0, 2697, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_06_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9/11_icon_No_upcoming_events_for_Los_Angeles_Clipp.png
try:
    _c11 = get_crop(11, 1440, 126)
    canvas.paste(_c11, (0, 933), _c11)
except Exception:
    pass
layout["No_upcoming_events_for_Lo"] = [0, 933, 1440, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_06_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 88, 103)
    canvas.paste(_c12, (1304, 952), _c12)
except Exception:
    pass
layout["icon_12"] = [1304, 952, 1392, 1055]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_06_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 54, 63)
    canvas.paste(_c13, (1150, 3), _c13)
except Exception:
    pass
layout["icon_13"] = [1150, 3, 1204, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_06_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 110, 66)
    canvas.paste(_c14, (1214, 3), _c14)
except Exception:
    pass
layout["icon_14"] = [1214, 3, 1324, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_06_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9/15_icon_6.53_W.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (36, 84), _c15)
except Exception:
    pass
layout["6.53_W"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_06_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9/16_icon_Protected_by_our_Buyer_Guarantee.png
try:
    _c16 = get_crop(16, 1440, 126)
    canvas.paste(_c16, (0, 933), _c16)
except Exception:
    pass
layout["Protected_by_our_Buyer_Gu"] = [0, 933, 1440, 1059]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_06_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 49, 55)
    canvas.paste(_c17, (1323, 8), _c17)
except Exception:
    pass
layout["icon_17"] = [1323, 8, 1372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_06_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-9/18_text_AIl_Games.png
try:
    _c18 = get_crop(18, 266, 55)
    canvas.paste(_c18, (60, 1496), _c18)
except Exception:
    pass
layout["AIl_Games"] = [60, 1496, 326, 1551]
