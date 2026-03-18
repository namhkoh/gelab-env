# page_id: page_eventbrite_f1e087441f9e44d997c2a58b9c8b0258_06
# screenshot: 2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-8.png
# step_index: 6/10
# task: Open Eventbrite. Find the 'Arts' category. Select events that are available for this weekend. From the results, open the first item and add it to favorite. Follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural layout drawing for the mobile UI page
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Fill overall background (dominant white)
draw.rectangle([0, 0, 1440, 2960], fill=(255, 255, 255))

# Status bar area at the very top (~50-80px)
STATUS_H = 84
draw.rectangle([0, 0, 1440, STATUS_H], fill=(189, 189, 189))
# subtle bottom border for status bar
draw.line([(0, STATUS_H - 1), (1440, STATUS_H - 1)], fill=(160, 160, 160), width=1)

# Header / toolbar area beneath status bar (keeps white but with subtle divider)
HEADER_H = 108
header_top = STATUS_H
header_bottom = STATUS_H + HEADER_H
draw.rectangle([0, header_top, 1440, header_bottom], fill=(255, 255, 255))
# divider under header
draw.line([(24, header_bottom), (1440 - 24, header_bottom)], fill=(230, 230, 230), width=2)

# Group container (rounded card) that holds the list options
# It's deliberately a white card with a faint outline/shadow to structure the page
card_x0, card_x1 = 32, 1440 - 32
card_y0, card_y1 = 200, 1280
card_radius = 20
# faint shadow by drawing a subtle darker band under the card
for i, shade in enumerate([245, 244, 243, 242]):
    draw.rounded_rectangle(
        [card_x0, card_y1 + 2 + i, card_x1, card_y1 + 6 + i],
        radius=card_radius,
        fill=(shade, shade, shade),
        outline=None
    )
# card fill (matches page white) and a very subtle outline to indicate boundary
draw.rounded_rectangle([card_x0, card_y0, card_x1, card_y1],
                       radius=card_radius,
                       fill=(255, 255, 255),
                       outline=(235, 235, 235),
                       width=1)

# Separator lines between the list items (using detected item positions)
# Detected item Y positions and heights (from metadata)
items = [
    (48, 234, 1344, 144),   # Anytime (top)
    (48, 414, 1344, 144),   # Today
    (48, 594, 1344, 144),   # Tomorrow
    (48, 774, 1344, 144),   # This Week
    (48, 954, 1344, 144),   # This Weekend
    (48, 1134, 1344, 144),  # Choose a date...
]
# draw subtle separators at the bottom edge of each item area
for (_, y, _, h) in items:
    bottom_y = y + h
    # Only draw separators that fall inside the card area to avoid drawing across status/header
    if card_y0 <= bottom_y <= card_y1 + 8:
        draw.line([(card_x0 + 24, bottom_y), (card_x1 - 24, bottom_y)], fill=(245, 245, 245), width=1)

# Accent vertical spacing guides (very faint) to visually separate left content area from right icons
# These are structural only and very subtle so they don't conflict with pasted icons/text
guide_x = 48
draw.line([(guide_x - 4, card_y0 + 8), (guide_x - 4, card_y1 - 8)], fill=(250, 250, 250), width=1)
guide_right_x = 1440 - 48
draw.line([(guide_right_x + 4, card_y0 + 8), (guide_right_x + 4, card_y1 - 8)], fill=(250, 250, 250), width=1)

# Bottom area remains empty (white) for further content; add one final faint baseline near top of content region
draw.line([(24, card_y1 + 20), (1440 - 24, card_y1 + 20)], fill=(250, 250, 250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_06_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-8/00_icon_4.32.png
try:
    _c0 = get_crop(0, 60, 62)
    canvas.paste(_c0, (180, 2), _c0)
except Exception:
    pass
layout["4.32"] = [180, 2, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_06_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-8/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 63, 61)
    canvas.paste(_c1, (309, 3), _c1)
except Exception:
    pass
layout["icon_1"] = [309, 3, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_06_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-8/02_icon_4.32.png
try:
    _c2 = get_crop(2, 56, 64)
    canvas.paste(_c2, (116, 2), _c2)
except Exception:
    pass
layout["4.32"] = [116, 2, 172, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_06_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-8/03_icon_4.32.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (12, 72), _c3)
except Exception:
    pass
layout["4.32"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_06_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-8/04_icon_Anytime.png
try:
    _c4 = get_crop(4, 1344, 144)
    canvas.paste(_c4, (48, 234), _c4)
except Exception:
    pass
layout["Anytime"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_06_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-8/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 51, 58)
    canvas.paste(_c5, (248, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 5, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_06_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-8/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 100, 60)
    canvas.paste(_c6, (1212, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 1, 1312, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_06_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-8/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 44, 60)
    canvas.paste(_c7, (1326, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1326, 3, 1370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_06_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-8/08_icon_4.32.png
try:
    _c8 = get_crop(8, 93, 61)
    canvas.paste(_c8, (16, 3), _c8)
except Exception:
    pass
layout["4.32"] = [16, 3, 109, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_06_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-8/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 123, 129)
    canvas.paste(_c9, (1291, 246), _c9)
except Exception:
    pass
layout["icon_9"] = [1291, 246, 1414, 375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_06_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-8/10_icon_Tomorrow.png
try:
    _c10 = get_crop(10, 1344, 144)
    canvas.paste(_c10, (48, 594), _c10)
except Exception:
    pass
layout["Tomorrow"] = [48, 594, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_06_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-8/11_text_When_do_you_want_to_go_out.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 234), _c11)
except Exception:
    pass
layout["When_do_you_want_to_go_ou"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_06_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-8/12_text_Today.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 414), _c12)
except Exception:
    pass
layout["Today"] = [48, 414, 1392, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_06_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-8/13_text_This_Week.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 774), _c13)
except Exception:
    pass
layout["This_Week"] = [48, 774, 1392, 918]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_06_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-8/14_text_This_Weekend.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 954), _c14)
except Exception:
    pass
layout["This_Weekend"] = [48, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_06_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-8/15_text_Choose_a_date-.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 1134), _c15)
except Exception:
    pass
layout["Choose_a_date-"] = [48, 1134, 1392, 1278]
