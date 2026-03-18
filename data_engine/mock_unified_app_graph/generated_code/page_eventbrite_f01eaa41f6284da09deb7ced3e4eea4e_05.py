# page_id: page_eventbrite_f01eaa41f6284da09deb7ced3e4eea4e_05
# screenshot: 2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-7.png
# step_index: 5/11
# task: Open Eventbrite. Check out 'Sports' events. Apply filters for events happening this week. Select the first event. Check similar events and add the first similar event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (1440x2960). This script paints the background and structural UI chrome.
# It intentionally avoids drawing any detected text/icons (they will be pasted on top).

# Colors
bg_white = (255, 255, 255)
status_gray = (197, 197, 197)        # status bar background
header_shadow = (230, 230, 230)      # subtle divider/shadow
card_bg = (250, 250, 251)            # very light card background
card_border = (240, 240, 240)        # card border / separators
accent_purple = (60, 40, 75)         # subtle header accent (used only for small divider)
separator = (242, 242, 243)          # separators between list items

W, H = canvas.size

# Fill overall background (dominant color from screenshot)
draw.rectangle([0, 0, W, H], fill=bg_white)

# Status bar area (top ~72px)
status_h = 72
draw.rectangle([0, 0, W, status_h], fill=status_gray)

# Subtle bottom line/shadow under the status bar
draw.line([(0, status_h - 1), (W, status_h - 1)], fill=header_shadow, width=1)

# Header / toolbar area (below status bar)
header_top = status_h
header_bottom = 200
# Use white so text pasted later remains visible; add a faint accent divider near bottom
draw.rectangle([0, header_top, W, header_bottom], fill=bg_white)
draw.line([(24, header_bottom - 1), (W - 24, header_bottom - 1)], fill=header_shadow, width=1)
# tiny accent center line to hint the header area (very subtle)
draw.line([(48, header_bottom - 2), (W - 48, header_bottom - 2)], fill=accent_purple, width=1)

# Content area card (rounded rectangle) behind the list of options
card_left = 24
card_right = W - 24
card_top = header_bottom + 8
card_bottom = 1400
radius = 22
# rounded_rectangle is available in modern PIL; draw a light card to separate content area subtly
try:
    draw.rounded_rectangle([card_left, card_top, card_right, card_bottom],
                           radius=radius, fill=card_bg, outline=card_border, width=1)
except Exception:
    # fallback: draw simple rectangle if rounded_rectangle unavailable
    draw.rectangle([card_left, card_top, card_right, card_bottom], fill=card_bg, outline=card_border)

# Decorative left and right padding vertical guides (very subtle)
guide_x1 = 48
guide_x2 = W - 48
draw.line([(guide_x1, card_top + 8), (guide_x1, card_bottom - 8)], fill=(248, 248, 249), width=1)
draw.line([(guide_x2, card_top + 8), (guide_x2, card_bottom - 8)], fill=(248, 248, 249), width=1)

# Separators between the detected text sections.
# Detected text boxes (y, height=144): 234, 414, 594, 774, 954, 1134
# Compute the gaps between boxes and draw separators centered in those gaps.
boxes = [234, 414, 594, 774, 954, 1134]
box_h = 144
separators_y = []
for i in range(len(boxes) - 1):
    bottom_i = boxes[i] + box_h
    top_next = boxes[i + 1]
    # only draw separator if there is a visible gap
    if top_next > bottom_i:
        sep_y = (bottom_i + top_next) // 2
        separators_y.append(sep_y)

# Draw the separators across the inner content width (aligned with detected text width)
sep_left = 48
sep_right = W - 48
for y in separators_y:
    draw.line([(sep_left, y), (sep_right, y)], fill=separator, width=1)

# Add a faint large-area shadow toward the top of the list to visually anchor the header
shadow_top = card_top
shadow_bottom = card_top + 12
for i in range(6):
    alpha = 12 - i*2
    if alpha <= 0:
        continue
    color = (240 + i, 240 + i, 241 + i)
    draw.line([(card_left + 2, shadow_top + i), (card_right - 2, shadow_top + i)], fill=color, width=1)

# Footer area subtle divider to end content card if the viewport continued (keeps structure consistent)
footer_div_y = card_bottom + 8
draw.line([(card_left + 4, footer_div_y), (card_right - 4, footer_div_y)], fill=header_shadow, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_05_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-7/00_icon_4.36.png
try:
    _c0 = get_crop(0, 60, 62)
    canvas.paste(_c0, (180, 2), _c0)
except Exception:
    pass
layout["4.36"] = [180, 2, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_05_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-7/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 64, 61)
    canvas.paste(_c1, (308, 3), _c1)
except Exception:
    pass
layout["icon_1"] = [308, 3, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_05_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-7/02_icon_4.36.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (12, 72), _c2)
except Exception:
    pass
layout["4.36"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_05_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-7/03_icon_Anytime.png
try:
    _c3 = get_crop(3, 1344, 144)
    canvas.paste(_c3, (48, 234), _c3)
except Exception:
    pass
layout["Anytime"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_05_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-7/04_icon_4.36.png
try:
    _c4 = get_crop(4, 57, 64)
    canvas.paste(_c4, (116, 2), _c4)
except Exception:
    pass
layout["4.36"] = [116, 2, 173, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_05_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 51, 58)
    canvas.paste(_c5, (248, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 5, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_05_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 100, 60)
    canvas.paste(_c6, (1212, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 1, 1312, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_05_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 44, 60)
    canvas.paste(_c7, (1326, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1326, 3, 1370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_05_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 123, 129)
    canvas.paste(_c8, (1291, 246), _c8)
except Exception:
    pass
layout["icon_8"] = [1291, 246, 1414, 375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_05_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-7/09_icon_4.36.png
try:
    _c9 = get_crop(9, 91, 61)
    canvas.paste(_c9, (16, 4), _c9)
except Exception:
    pass
layout["4.36"] = [16, 4, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_05_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-7/10_icon_Tomorrow.png
try:
    _c10 = get_crop(10, 1344, 144)
    canvas.paste(_c10, (48, 594), _c10)
except Exception:
    pass
layout["Tomorrow"] = [48, 594, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_05_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-7/11_text_When_do_you_want_to_go_out.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 234), _c11)
except Exception:
    pass
layout["When_do_you_want_to_go_ou"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_05_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-7/12_text_Today.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 414), _c12)
except Exception:
    pass
layout["Today"] = [48, 414, 1392, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_05_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-7/13_text_This_Week.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 774), _c13)
except Exception:
    pass
layout["This_Week"] = [48, 774, 1392, 918]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_05_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-7/14_text_This_Weekend.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 954), _c14)
except Exception:
    pass
layout["This_Weekend"] = [48, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_05_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-7/15_text_Choose_a_date-.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 1134), _c15)
except Exception:
    pass
layout["Choose_a_date-"] = [48, 1134, 1392, 1278]
