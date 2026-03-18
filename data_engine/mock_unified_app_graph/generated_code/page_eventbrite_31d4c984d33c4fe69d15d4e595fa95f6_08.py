# page_id: page_eventbrite_31d4c984d33c4fe69d15d4e595fa95f6_08
# screenshot: 2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-10.png
# step_index: 8/14
# task: Open Eventbrite. Look for 'community events' in 'Chicago'. Select the first event happening tomorrow that is not promoted. Check if they have an option for 'refund policy'.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and UI structure for the mobile page (uses existing `canvas`, `draw`, and font variables)

w, h = canvas.size

# Background fill (dominant color: white)
draw.rectangle((0, 0, w, h), fill=(255, 255, 255))

# Status bar at the very top (approx 0..88 px)
status_bar_h = 88
draw.rectangle((0, 0, w, status_bar_h), fill=(221, 221, 221))  # light grey status bar
# subtle inner top highlight and bottom shadow to give depth
draw.line((0, status_bar_h-1, w, status_bar_h-1), fill=(200, 200, 200), width=1)
draw.line((0, 1, w, 1), fill=(235, 235, 235), width=1)

# Header / toolbar area below the status bar (keeps background white but add subtle divider)
header_top = status_bar_h
header_h = 140
draw.rectangle((0, header_top, w, header_top + header_h), fill=(255, 255, 255))
# faint bottom border under header
draw.line((24, header_top + header_h - 1, w - 24, header_top + header_h - 1), fill=(243, 242, 247), width=2)

# Large subtle rounded card behind the list of choices (group container)
list_card_top = 200
list_card_bottom = 1300
card_radius = 20
draw.rounded_rectangle((24, list_card_top, w - 24, list_card_bottom), radius=card_radius, fill=(255, 255, 255), outline=(245,245,248), width=1)

# Add a very light vertical left margin guideline (visual structure only)
left_margin_x = 48
draw.line((left_margin_x, list_card_top + 8, left_margin_x, list_card_bottom - 8), fill=(250,250,251), width=1)

# Separator lines between the rows / sections (positions derived from expected item y positions)
rows = [
    {"top": 234, "height":144},  # title row
    {"top": 414, "height":144},
    {"top": 594, "height":144},
    {"top": 774, "height":144},
    {"top": 954, "height":144},
    {"top": 1134, "height":144},
]
sep_color = (245, 245, 247)
for r in rows:
    # draw a faint separator under each row (except maybe the last)
    y_sep = r["top"] + r["height"] - 10
    # Clip separators to inside card container
    if y_sep > list_card_top and y_sep < list_card_bottom:
        draw.line((left_margin_x, y_sep, w - 48, y_sep), fill=sep_color, width=2)
        # add a slightly darker 1px line immediately below to enhance separation
        draw.line((left_margin_x, y_sep + 2, w - 48, y_sep + 2), fill=(250,250,251), width=1)

# Add faint section dividers for visual grouping (longer subtle bars to the left)
for idx, r in enumerate(rows):
    y_center = r["top"] + 32
    if y_center > list_card_top and y_center < list_card_bottom:
        # a small left-side accent bar (very faint)
        bar_x1 = left_margin_x - 10
        bar_x2 = left_margin_x - 6
        draw.rectangle((bar_x1, y_center - 18, bar_x2, y_center + 18), fill=(250,250,251))

# Bottom shadow under the card for subtle elevation
shadow_top = list_card_bottom
for i in range(6):
    alpha = int(12 - i*2)
    y = shadow_top + i
    # make progressively lighter lines
    shade = 240 + i
    draw.line((24, y, w - 24, y), fill=(shade, shade, shade), width=1)

# Small decorative top-left back area background (no icon/text drawn)
back_area_rect = (24, header_top + 18, 140, header_top + header_h - 18)
draw.rectangle(back_area_rect, fill=(255,255,255))

# Final very subtle overall vignette edges to match screenshot feel (extremely light)
edge_shade = 4
for i in range(edge_shade):
    alpha = int(6 - i)
    inset = 6 + i
    draw.rectangle((inset, inset, w - inset, h - inset), outline=(255 - i, 255 - i, 255 - i))

# End of structural/background drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_08_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-10/00_icon_8.07.png
try:
    _c0 = get_crop(0, 57, 63)
    canvas.paste(_c0, (115, 3), _c0)
except Exception:
    pass
layout["8.07"] = [115, 3, 172, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_08_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-10/01_icon_8.07.png
try:
    _c1 = get_crop(1, 58, 61)
    canvas.paste(_c1, (181, 3), _c1)
except Exception:
    pass
layout["8.07"] = [181, 3, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_08_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-10/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 64, 61)
    canvas.paste(_c2, (308, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [308, 3, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_08_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-10/03_icon_8.07.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (12, 72), _c3)
except Exception:
    pass
layout["8.07"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_08_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-10/04_icon_Anytime.png
try:
    _c4 = get_crop(4, 1344, 144)
    canvas.paste(_c4, (48, 234), _c4)
except Exception:
    pass
layout["Anytime"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_08_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-10/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 48, 63)
    canvas.paste(_c5, (1154, 4), _c5)
except Exception:
    pass
layout["icon_5"] = [1154, 4, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_08_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 98, 61)
    canvas.paste(_c6, (1216, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [1216, 3, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_08_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-10/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 51, 58)
    canvas.paste(_c7, (248, 5), _c7)
except Exception:
    pass
layout["icon_7"] = [248, 5, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_08_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 46, 60)
    canvas.paste(_c8, (1325, 4), _c8)
except Exception:
    pass
layout["icon_8"] = [1325, 4, 1371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_08_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-10/09_icon_8.07.png
try:
    _c9 = get_crop(9, 92, 60)
    canvas.paste(_c9, (15, 5), _c9)
except Exception:
    pass
layout["8.07"] = [15, 5, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_08_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-10/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 123, 128)
    canvas.paste(_c10, (1291, 247), _c10)
except Exception:
    pass
layout["icon_10"] = [1291, 247, 1414, 375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_08_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-10/11_icon_Tomorrow.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 594), _c11)
except Exception:
    pass
layout["Tomorrow"] = [48, 594, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_08_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-10/12_text_When_do_you_want_to_go_out.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 234), _c12)
except Exception:
    pass
layout["When_do_you_want_to_go_ou"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_08_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-10/13_text_Today.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 414), _c13)
except Exception:
    pass
layout["Today"] = [48, 414, 1392, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_08_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-10/14_text_This_Week.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 774), _c14)
except Exception:
    pass
layout["This_Week"] = [48, 774, 1392, 918]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_08_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-10/15_text_This_Weekend.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 954), _c15)
except Exception:
    pass
layout["This_Weekend"] = [48, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_08_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-10/16_text_Choose_a_date-.png
try:
    _c16 = get_crop(16, 1344, 144)
    canvas.paste(_c16, (48, 1134), _c16)
except Exception:
    pass
layout["Choose_a_date-"] = [48, 1134, 1392, 1278]
