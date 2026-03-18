# page_id: page_eventbrite_a9f633a394e74f78843aa30bd2792346_10
# screenshot: 2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-12.png
# step_index: 10/18
# task: Open Eventbrite. Set the city to "Los Angeles". Look for Photography workshops happening next week. What is the price of the tickets for first non-promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for a 1440x2960 canvas.
# Assumes available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Full background (match dominant white)
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# Status bar area (top) - light gray
status_h = 96
draw.rectangle((0, 0, 1440, status_h), fill="#CFCFCF")

# Header / toolbar area below status bar
header_top = status_h
header_bottom = status_h + 88
draw.rectangle((0, header_top, 1440, header_bottom), fill="#FFFFFF")
# subtle divider under header
draw.line((32, header_bottom, 1408, header_bottom), fill="#E8E8E8", width=2)

# Large content card background (rounded) behind the list items
card_left, card_top = 32, 200
card_right, card_bottom = 1408, 1260
shadow_offset = 8

# simple shadow
draw.rounded_rectangle(
    (card_left + shadow_offset, card_top + shadow_offset,
     card_right + shadow_offset, card_bottom + shadow_offset),
    radius=24, fill="#F3F3F3"
)

# card surface
draw.rounded_rectangle(
    (card_left, card_top, card_right, card_bottom),
    radius=24, fill="#FFFFFF", outline="#EFEFEF", width=1
)

# Horizontal separators between the option rows (subtle)
# Detected rows centers (approx): 234, 414, 594, 774, 954, 1134
row_centers = [234, 414, 594, 774, 954, 1134]
# draw a faint separator beneath each row (except last) to structure the card
for y in row_centers:
    sep_y = y + 110  # place separator between items (visually between large text rows)
    if sep_y < card_bottom - 24:
        draw.line((card_left + 16, sep_y, card_right - 16, sep_y), fill="#FAFAFA", width=2)

# Accent background for the first (selected) row area — subtle warm band (behind text)
first_row_top = 200
first_row_bottom = 200 + 156
accent_left = card_left + 8
accent_right = card_right - 8
draw.rectangle((accent_left, first_row_top, accent_right, first_row_bottom), fill="#FFF8F3")

# Thin accent mark on the left for the selected row
accent_mark_w = 10
draw.rectangle((card_left + 8, first_row_top + 12, card_left + 8 + accent_mark_w, first_row_bottom - 12), fill="#D95319")

# A faint vertical divider to separate header area from content (subtle)
divider_x = 32
draw.line((divider_x, header_bottom + 8, divider_x, card_bottom - 8), fill="#FFFFFF", width=1)

# Bottom safe-area footer subtle divider
footer_top = 2840
draw.line((32, footer_top, 1408, footer_top), fill="#F0F0F0", width=2)

# End of structural drawing. The detected icons/text will be pasted on top of these elements.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_10_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-12/00_icon_4.51.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (12, 72), _c0)
except Exception:
    pass
layout["4.51"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_10_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-12/01_icon_4.51.png
try:
    _c1 = get_crop(1, 60, 62)
    canvas.paste(_c1, (180, 2), _c1)
except Exception:
    pass
layout["4.51"] = [180, 2, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_10_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-12/02_icon_4.51.png
try:
    _c2 = get_crop(2, 59, 64)
    canvas.paste(_c2, (114, 2), _c2)
except Exception:
    pass
layout["4.51"] = [114, 2, 173, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_10_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-12/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 65, 61)
    canvas.paste(_c3, (308, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [308, 3, 373, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_10_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-12/04_icon_Anytime.png
try:
    _c4 = get_crop(4, 1344, 144)
    canvas.paste(_c4, (48, 234), _c4)
except Exception:
    pass
layout["Anytime"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_10_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-12/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 48, 63)
    canvas.paste(_c5, (1154, 4), _c5)
except Exception:
    pass
layout["icon_5"] = [1154, 4, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_10_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-12/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 98, 61)
    canvas.paste(_c6, (1216, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [1216, 3, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_10_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-12/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 51, 59)
    canvas.paste(_c7, (248, 4), _c7)
except Exception:
    pass
layout["icon_7"] = [248, 4, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_10_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-12/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 45, 60)
    canvas.paste(_c8, (1326, 4), _c8)
except Exception:
    pass
layout["icon_8"] = [1326, 4, 1371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_10_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-12/09_icon_4.51.png
try:
    _c9 = get_crop(9, 89, 61)
    canvas.paste(_c9, (16, 4), _c9)
except Exception:
    pass
layout["4.51"] = [16, 4, 105, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_10_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-12/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 123, 128)
    canvas.paste(_c10, (1291, 247), _c10)
except Exception:
    pass
layout["icon_10"] = [1291, 247, 1414, 375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_10_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-12/11_icon_Tomorrow.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 594), _c11)
except Exception:
    pass
layout["Tomorrow"] = [48, 594, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_10_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-12/12_text_When_do_you_want_to_go_out.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 234), _c12)
except Exception:
    pass
layout["When_do_you_want_to_go_ou"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_10_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-12/13_text_Today.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 414), _c13)
except Exception:
    pass
layout["Today"] = [48, 414, 1392, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_10_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-12/14_text_This_Week.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 774), _c14)
except Exception:
    pass
layout["This_Week"] = [48, 774, 1392, 918]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_10_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-12/15_text_This_Weekend.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 954), _c15)
except Exception:
    pass
layout["This_Weekend"] = [48, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_10_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-12/16_text_Choose_a_date-.png
try:
    _c16 = get_crop(16, 1344, 144)
    canvas.paste(_c16, (48, 1134), _c16)
except Exception:
    pass
layout["Choose_a_date-"] = [48, 1134, 1392, 1278]
