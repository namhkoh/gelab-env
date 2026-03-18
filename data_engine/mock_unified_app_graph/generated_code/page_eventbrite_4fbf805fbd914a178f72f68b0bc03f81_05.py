# page_id: page_eventbrite_4fbf805fbd914a178f72f68b0bc03f81_05
# screenshot: 2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-7.png
# step_index: 5/10
# task: Open Eventbrite. Explore "Education" events. Apply filters for events happening tomorrow. From the list, select the third event and check out its description.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background (dominant color: white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar area at the very top (slightly muted grey)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(205, 205, 205))

# Subtle divider under status bar to separate it from header
draw.line([(0, status_h), (1440, status_h)], fill=(190, 190, 190), width=1)

# Header / toolbar area (kept visually distinct but still light)
header_top = status_h
header_bottom = 168
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))

# Thin divider under header
draw.line([(24, header_bottom), (1440 - 24, header_bottom)], fill=(235, 235, 240), width=2)

# Rounded card/background behind the list of date choices
card_left = 32
card_top = 200
card_right = 1440 - 32
card_bottom = 1280
card_radius = 28
# Slightly off-white fill so it reads as a separate surface
draw.rounded_rectangle(
    [(card_left, card_top), (card_right, card_bottom)],
    radius=card_radius,
    fill=(250, 250, 251),
    outline=(235, 235, 240),
    width=1
)

# Subtle separators between list sections (do not draw text/icons)
# Positions are placed to run behind where text/icons will be pasted.
separator_x1 = card_left + 32
separator_x2 = card_right - 32
separator_positions = [
    414 - 20,  # between header/first item area
    594 - 20,
    774 - 20,
    954 - 20,
    1134 - 20
]
for y in separator_positions:
    # Light hairline separator
    draw.line([(separator_x1, y), (separator_x2, y)], fill=(242, 242, 245), width=1)

# A faint left guideline to indicate grouping (very subtle)
draw.line([(card_left + 24, card_top + 12), (card_left + 24, card_bottom - 12)], fill=(248, 248, 249), width=2)

# Bottom area left intentionally blank (large white space for scrolling content)
# End of UI background/structure drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_05_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-7/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 64, 61)
    canvas.paste(_c0, (308, 3), _c0)
except Exception:
    pass
layout["icon_0"] = [308, 3, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_05_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-7/01_icon_4.56.png
try:
    _c1 = get_crop(1, 60, 63)
    canvas.paste(_c1, (180, 2), _c1)
except Exception:
    pass
layout["4.56"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_05_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-7/02_icon_4.56.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (12, 72), _c2)
except Exception:
    pass
layout["4.56"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_05_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-7/03_icon_Anytime.png
try:
    _c3 = get_crop(3, 1344, 144)
    canvas.paste(_c3, (48, 234), _c3)
except Exception:
    pass
layout["Anytime"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_05_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-7/04_icon_4.56.png
try:
    _c4 = get_crop(4, 57, 64)
    canvas.paste(_c4, (116, 2), _c4)
except Exception:
    pass
layout["4.56"] = [116, 2, 173, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_05_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 51, 58)
    canvas.paste(_c5, (248, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 5, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_05_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 100, 60)
    canvas.paste(_c6, (1212, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 1, 1312, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_05_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 44, 60)
    canvas.paste(_c7, (1326, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1326, 3, 1370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_05_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 123, 129)
    canvas.paste(_c8, (1291, 246), _c8)
except Exception:
    pass
layout["icon_8"] = [1291, 246, 1414, 375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_05_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-7/09_icon_Tomorrow.png
try:
    _c9 = get_crop(9, 1344, 144)
    canvas.paste(_c9, (48, 594), _c9)
except Exception:
    pass
layout["Tomorrow"] = [48, 594, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_05_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-7/10_text_4.56.png
try:
    _c10 = get_crop(10, 92, 43)
    canvas.paste(_c10, (22, 17), _c10)
except Exception:
    pass
layout["4.56"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_05_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-7/11_text_When_do_you_want_to_go_out.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 234), _c11)
except Exception:
    pass
layout["When_do_you_want_to_go_ou"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_05_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-7/12_text_Today.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 414), _c12)
except Exception:
    pass
layout["Today"] = [48, 414, 1392, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_05_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-7/13_text_This_Week.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 774), _c13)
except Exception:
    pass
layout["This_Week"] = [48, 774, 1392, 918]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_05_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-7/14_text_This_Weekend.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 954), _c14)
except Exception:
    pass
layout["This_Weekend"] = [48, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_05_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-7/15_text_Choose_a_date-.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 1134), _c15)
except Exception:
    pass
layout["Choose_a_date-"] = [48, 1134, 1392, 1278]
