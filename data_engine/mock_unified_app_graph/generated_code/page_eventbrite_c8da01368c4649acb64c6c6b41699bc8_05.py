# page_id: page_eventbrite_c8da01368c4649acb64c6c6b41699bc8_05
# screenshot: 2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-7.png
# step_index: 5/13
# task: Open Eventbrite. Look up "Animal" events. Filter by events happening next week. Select the first event - who is the organizer?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# background and structural elements for the mobile UI (canvas and draw provided)

# Fill full background (dominant color is white)
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar area (top ~72px) - light gray
status_bar_h = 72
draw.rectangle((0, 0, 1440, status_bar_h), fill=(189, 189, 189))

# Thin subtle divider under status bar
draw.line([(0, status_bar_h), (1440, status_bar_h)], fill=(210, 210, 210), width=1)

# Toolbar / header area (beneath status bar) - keep white but add a bottom divider
header_top = status_bar_h
header_bottom = 140
draw.rectangle((0, header_top, 1440, header_bottom), fill=(255, 255, 255))
draw.line([(24, header_bottom), (1416, header_bottom)], fill=(236, 234, 240), width=2)

# List content container background (subtle group area)
list_left = 24
list_right = 1416
list_top = 200
list_bottom = 1320
# Very subtle off-white card so pasted text/icons remain unchanged
draw.rounded_rectangle((list_left, list_top, list_right, list_bottom),
                       radius=12, fill=(255, 255, 255), outline=None)

# Separator lines between list items (use detected item vertical positions)
item_positions = [234, 414, 594, 774, 954, 1134]
item_height = 144
sep_color = (243, 243, 244)  # very light separator
for y in item_positions:
    # draw a thin separator at the bottom edge of each item area
    sep_y = y + item_height
    draw.line([(48, sep_y), (1392, sep_y)], fill=sep_color, width=1)

# Subtle left guide margin (visual structure) - light vertical guideline (non-intrusive)
draw.line([(48, header_top + 8), (48, list_bottom)], fill=(248, 248, 249), width=1)

# Light bottom shadow for the list container to give depth
shadow_top = list_bottom
shadow_bottom = list_bottom + 18
for i in range(8):
    alpha = int(12 * (1 - i / 8))  # decreasing opacity
    y0 = shadow_top + i * 2
    y1 = y0 + 2
    # simulate shadow by drawing slightly darker translucent lines (approx with RGB blends)
    shade = 240 - i * 2
    draw.rectangle((list_left + 6, y0, list_right - 6, y1), fill=(shade, shade, shade))

# final subtle footer divider near top of remaining content area
draw.line([(24, 1280), (1416, 1280)], fill=(245, 245, 246), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_05_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-7/00_icon_5.15.png
try:
    _c0 = get_crop(0, 60, 62)
    canvas.paste(_c0, (180, 2), _c0)
except Exception:
    pass
layout["5.15"] = [180, 2, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_05_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-7/01_icon_5.15.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (12, 72), _c1)
except Exception:
    pass
layout["5.15"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_05_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-7/02_icon_5.15.png
try:
    _c2 = get_crop(2, 57, 64)
    canvas.paste(_c2, (116, 2), _c2)
except Exception:
    pass
layout["5.15"] = [116, 2, 173, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_05_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-7/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 63, 61)
    canvas.paste(_c3, (309, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [309, 3, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_05_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-7/04_icon_Anytime.png
try:
    _c4 = get_crop(4, 1344, 144)
    canvas.paste(_c4, (48, 234), _c4)
except Exception:
    pass
layout["Anytime"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_05_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 53, 59)
    canvas.paste(_c5, (248, 4), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 4, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_05_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 100, 61)
    canvas.paste(_c6, (1212, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 0, 1312, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_05_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 44, 60)
    canvas.paste(_c7, (1326, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1326, 3, 1370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_05_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 123, 129)
    canvas.paste(_c8, (1291, 246), _c8)
except Exception:
    pass
layout["icon_8"] = [1291, 246, 1414, 375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_05_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-7/09_icon_Tomorrow.png
try:
    _c9 = get_crop(9, 1344, 144)
    canvas.paste(_c9, (48, 594), _c9)
except Exception:
    pass
layout["Tomorrow"] = [48, 594, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_05_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-7/10_text_5.15.png
try:
    _c10 = get_crop(10, 92, 43)
    canvas.paste(_c10, (22, 17), _c10)
except Exception:
    pass
layout["5.15"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_05_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-7/11_text_When_do_you_want_to_go_out.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 234), _c11)
except Exception:
    pass
layout["When_do_you_want_to_go_ou"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_05_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-7/12_text_Today.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 414), _c12)
except Exception:
    pass
layout["Today"] = [48, 414, 1392, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_05_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-7/13_text_This_Week.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 774), _c13)
except Exception:
    pass
layout["This_Week"] = [48, 774, 1392, 918]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_05_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-7/14_text_This_Weekend.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 954), _c14)
except Exception:
    pass
layout["This_Weekend"] = [48, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_05_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-7/15_text_Choose_a_date-.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 1134), _c15)
except Exception:
    pass
layout["Choose_a_date-"] = [48, 1134, 1392, 1278]
