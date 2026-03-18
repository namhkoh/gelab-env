# page_id: page_eventbrite_9fdb2ee43d5a49adac5304bdd5dacfc2_05
# screenshot: 2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-7.png
# step_index: 5/8
# task: Open Eventbrite. Look up 'Pet' events. Filter by events happening this weekend. Select the third non-promoted event from the results - how much are the tickets for the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (match screenshot dominant white)
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar area at top (~50px) - a neutral light gray bar
status_h = 84
draw.rectangle((0, 0, 1440, status_h), fill=(200, 200, 200))

# Header/toolbar area below status bar (keeps white but add subtle bottom divider)
header_top = status_h
header_bottom = 160
draw.rectangle((0, header_top, 1440, header_bottom), fill=(255, 255, 255))
# subtle divider line under the header
draw.line((24, header_bottom, 1440 - 24, header_bottom), fill=(230, 230, 235), width=2)

# Slight shadow / accent just below header to separate from content
draw.line((24, header_bottom + 2, 1440 - 24, header_bottom + 2), fill=(240, 237, 245), width=1)

# Content area card (a very subtle off-white rounded rectangle behind the list)
card_x0 = 36
card_x1 = 1440 - 36
card_y0 = 200
card_y1 = 1320
# Off-white fill and very light border to create a gentle card effect
try:
    draw.rounded_rectangle((card_x0, card_y0, card_x1, card_y1),
                           radius=18,
                           fill=(250, 250, 251),
                           outline=(240, 239, 243),
                           width=1)
except Exception:
    # Fallback if rounded_rectangle is not available: draw normal rect + small corner circles
    draw.rectangle((card_x0, card_y0, card_x1, card_y1), fill=(250, 250, 251), outline=(240, 239, 243))

# Draw subtle separators between the option rows (do not draw text or icons)
# Positions correspond to the detected text blocks' bottoms
text_blocks = [234, 414, 594, 774, 954, 1134]  # y positions from detections
sep_color = (244, 244, 247)
sep_x0 = 48
sep_x1 = 1440 - 48
for y in text_blocks:
    sep_y = y + 144  # draw separator just below each detected text block
    # Only draw separators inside the card area to avoid duplicating header/status regions
    if card_y0 + 4 <= sep_y <= card_y1 - 4:
        draw.line((sep_x0, sep_y, sep_x1, sep_y), fill=sep_color, width=2)

# Add a faint left edge guide line inside the card to suggest item alignment (very subtle)
draw.line((card_x0 + 12, card_y0 + 12, card_x0 + 12, card_y1 - 12), fill=(247, 247, 249), width=1)

# Draw a faint right edge guideline inside card (helps visual balance, non-intrusive)
draw.line((card_x1 - 12, card_y0 + 12, card_x1 - 12, card_y1 - 12), fill=(247, 247, 249), width=1)

# Footer subtle divider near the end of the card area
draw.line((card_x0 + 8, card_y1 - 8, card_x1 - 8, card_y1 - 8), fill=(240, 240, 242), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_05_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-7/00_icon_4.47.png
try:
    _c0 = get_crop(0, 60, 63)
    canvas.paste(_c0, (180, 2), _c0)
except Exception:
    pass
layout["4.47"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_05_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-7/01_icon_4.47.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (12, 72), _c1)
except Exception:
    pass
layout["4.47"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_05_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-7/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 64, 61)
    canvas.paste(_c2, (308, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [308, 3, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_05_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-7/03_icon_Anytime.png
try:
    _c3 = get_crop(3, 1344, 144)
    canvas.paste(_c3, (48, 234), _c3)
except Exception:
    pass
layout["Anytime"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_05_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-7/04_icon_4.47.png
try:
    _c4 = get_crop(4, 59, 65)
    canvas.paste(_c4, (114, 2), _c4)
except Exception:
    pass
layout["4.47"] = [114, 2, 173, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_05_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 51, 59)
    canvas.paste(_c5, (248, 4), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 4, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_05_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 44, 61)
    canvas.paste(_c6, (1326, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [1326, 3, 1370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_05_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 100, 61)
    canvas.paste(_c7, (1212, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1212, 0, 1312, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_05_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-7/08_icon_4.47.png
try:
    _c8 = get_crop(8, 89, 61)
    canvas.paste(_c8, (17, 3), _c8)
except Exception:
    pass
layout["4.47"] = [17, 3, 106, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_05_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-7/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 123, 129)
    canvas.paste(_c9, (1291, 246), _c9)
except Exception:
    pass
layout["icon_9"] = [1291, 246, 1414, 375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_05_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-7/10_icon_Tomorrow.png
try:
    _c10 = get_crop(10, 1344, 144)
    canvas.paste(_c10, (48, 594), _c10)
except Exception:
    pass
layout["Tomorrow"] = [48, 594, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_05_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-7/11_text_When_do_you_want_to_go_out.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 234), _c11)
except Exception:
    pass
layout["When_do_you_want_to_go_ou"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_05_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-7/12_text_Today.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 414), _c12)
except Exception:
    pass
layout["Today"] = [48, 414, 1392, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_05_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-7/13_text_This_Week.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 774), _c13)
except Exception:
    pass
layout["This_Week"] = [48, 774, 1392, 918]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_05_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-7/14_text_This_Weekend.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 954), _c14)
except Exception:
    pass
layout["This_Weekend"] = [48, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_05_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-7/15_text_Choose_a_date-.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 1134), _c15)
except Exception:
    pass
layout["Choose_a_date-"] = [48, 1134, 1392, 1278]
