# page_id: page_seatgeek_2d936b7d0af74c20ba6cd92218729838_11
# screenshot: 2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-14.png
# step_index: 11/12
# task: Open SeatGeek. Track "Los Angeles Clippers" and "Golden State Warriors".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the page on the provided canvas and draw objects.
# Available variables: canvas (PIL Image 1440x2960 RGB), draw (ImageDraw), fonts: font_sm, font_md, font_lg, font_xl

# Colors
navy_dark = (6, 34, 52)       # deep status/navy
navy_mid  = (12, 68, 96)      # main header navy
navy_soft = (10, 56, 80)      # decorative navy shade
white     = (255, 255, 255)
light_grey = (246, 247, 248)  # subtle content card background
divider_grey = (225, 228, 231)

w, h = canvas.size

# Ensure base is white
draw.rectangle([(0, 0), (w, h)], fill=white)

# Status bar (top ~80px)
status_bar_h = 88
draw.rectangle([(0, 0), (w, status_bar_h)], fill=navy_dark)

# Main header / hero background (below status bar)
header_top = status_bar_h
header_bottom = 820  # tall hero area like screenshot
draw.rectangle([(0, header_top), (w, header_bottom)], fill=navy_mid)

# Decorative header shapes (abstract ticket/plus shapes)
# Right-side rounded block
draw.rounded_rectangle([(920, -160), (1700, 220)], radius=140, fill=navy_soft)
# Left-side rounded block
draw.rounded_rectangle([(-240, 360), (520, 980)], radius=180, fill=navy_soft)
# Small plus-like subtle block (center-left)
draw.rounded_rectangle([(540, 260), (700, 380)], radius=36, fill=navy_soft)

# Create a white rounded overlay at the bottom of the header to form the curved transition to content
overlay_top = header_bottom - 80
overlay_bottom = header_bottom + 80
draw.rounded_rectangle([(0, overlay_top), (w, overlay_bottom)], radius=44, fill=white)

# Thin separator line along the header/content boundary
draw.line([(32, overlay_bottom), (w - 32, overlay_bottom)], fill=divider_grey, width=1)

# Large subtle content card / background panel (keeps plenty of whitespace for pasted elements)
card_left = 72
card_right = w - 72
card_top = overlay_bottom + 80
card_bottom = card_top + 920
draw.rounded_rectangle([(card_left, card_top), (card_right, card_bottom)], radius=28, fill=light_grey)

# Subtle inner separators to denote sections within the content area
sep_y1 = card_top + 220
sep_y2 = card_top + 520
draw.line([(card_left + 24, sep_y1), (card_right - 24, sep_y1)], fill=divider_grey, width=1)
draw.line([(card_left + 24, sep_y2), (card_right - 24, sep_y2)], fill=divider_grey, width=1)

# Additional faint divider near center of the page
center_div_y = int(h * 0.55)
draw.line([(48, center_div_y), (w - 48, center_div_y)], fill=(240, 241, 242), width=1)

# Footer/content grounding bar (very light)
footer_top = h - 180
draw.rectangle([(0, footer_top), (w, h)], fill=white)
draw.line([(0, footer_top), (w, footer_top)], fill=divider_grey, width=1)

# Small left and right padding background strips to frame content columns
draw.rectangle([(0, overlay_bottom), (48, footer_top)], fill=white)
draw.rectangle([(w - 48, overlay_bottom), (w, footer_top)], fill=white)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_11_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-14/00_icon_Track_Now.png
try:
    _c0 = get_crop(0, 337, 153)
    canvas.paste(_c0, (551, 1638), _c0)
except Exception:
    pass
layout["Track_Now"] = [551, 1638, 888, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_11_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-14/01_icon_Track_this_performer.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1104, 84), _c1)
except Exception:
    pass
layout["Track_this_performer"] = [1104, 84, 1248, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_11_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-14/02_icon_Share_this_performer.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1260, 84), _c2)
except Exception:
    pass
layout["Share_this_performer"] = [1260, 84, 1404, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_11_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-14/03_icon_6.54_W.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (36, 84), _c3)
except Exception:
    pass
layout["6.54_W"] = [36, 84, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_11_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-14/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 57, 59)
    canvas.paste(_c4, (245, 5), _c4)
except Exception:
    pass
layout["icon_4"] = [245, 5, 302, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_11_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-14/05_icon_6.54_W.png
try:
    _c5 = get_crop(5, 54, 59)
    canvas.paste(_c5, (182, 5), _c5)
except Exception:
    pass
layout["6.54_W"] = [182, 5, 236, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_11_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-14/06_icon_6.54_W.png
try:
    _c6 = get_crop(6, 59, 62)
    canvas.paste(_c6, (114, 3), _c6)
except Exception:
    pass
layout["6.54_W"] = [114, 3, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_11_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-14/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 57, 69)
    canvas.paste(_c7, (380, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [380, 1, 437, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_11_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-14/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 54, 64)
    canvas.paste(_c8, (1151, 5), _c8)
except Exception:
    pass
layout["icon_8"] = [1151, 5, 1205, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_11_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-14/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 48, 49)
    canvas.paste(_c9, (317, 12), _c9)
except Exception:
    pass
layout["icon_9"] = [317, 12, 365, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_11_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-14/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 57, 63)
    canvas.paste(_c10, (1216, 5), _c10)
except Exception:
    pass
layout["icon_10"] = [1216, 5, 1273, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_11_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-14/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 43, 61)
    canvas.paste(_c11, (1325, 6), _c11)
except Exception:
    pass
layout["icon_11"] = [1325, 6, 1368, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_11_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-14/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 46, 64)
    canvas.paste(_c12, (1270, 5), _c12)
except Exception:
    pass
layout["icon_12"] = [1270, 5, 1316, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_11_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-14/13_text_Golden_State_Warriors.png
try:
    _c13 = get_crop(13, 653, 66)
    canvas.paste(_c13, (55, 857), _c13)
except Exception:
    pass
layout["Golden_State_Warriors"] = [55, 857, 708, 923]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_11_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-14/14_text_No_upcoming_Games.png
try:
    _c14 = get_crop(14, 337, 153)
    canvas.paste(_c14, (551, 1638), _c14)
except Exception:
    pass
layout["No_upcoming_Games"] = [551, 1638, 888, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_11_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-14/15_text_Track_Golden_State_Warriors_for_event_up.png
try:
    _c15 = get_crop(15, 337, 153)
    canvas.paste(_c15, (551, 1638), _c15)
except Exception:
    pass
layout["Track_Golden_State_Warrio"] = [551, 1638, 888, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2d936b7d0af74c20ba6cd92218729838/step_11_2024_4_22_18_52_2d936b7d0af74c20ba6cd92218729838-14/16_text_1Zz.png
try:
    _c16 = get_crop(16, 156, 142)
    canvas.paste(_c16, (645, 1176), _c16)
except Exception:
    pass
layout["1Zz"] = [645, 1176, 801, 1318]
