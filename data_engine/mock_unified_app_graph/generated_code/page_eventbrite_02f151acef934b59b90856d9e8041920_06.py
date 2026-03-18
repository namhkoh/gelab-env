# page_id: page_eventbrite_02f151acef934b59b90856d9e8041920_06
# screenshot: 2024_4_24_17_24_02f151acef934b59b90856d9e8041920-8.png
# step_index: 6/11
# task: Open Eventbrite. Check the "Tech" events happening this month. Open the first event and check its date and time.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (1440x2960 white)
w, h = canvas.size

# Background fill (ensure clean white base)
draw.rectangle((0, 0, w, h), fill=(255, 255, 255))

# Status bar area at top (approx ~72px tall) - neutral grey background
status_h = 72
status_color = (200, 200, 200)  # light neutral grey
draw.rectangle((0, 0, w, status_h), fill=status_color)

# Header/toolbar area (below status bar) - keep white but add subtle divider/shadow
header_top = status_h
header_bottom = 160
draw.rectangle((0, header_top, w, header_bottom), fill=(255, 255, 255))

# Subtle divider line / shadow under header to separate content
divider_color = (236, 230, 243)  # very light lavender-gray
draw.rectangle((0, header_bottom, w, header_bottom + 4), fill=divider_color)

# Additional faint horizontal separator (mid-top area) to echo UI structure
sep_y = 240
draw.line((40, sep_y, w - 40, sep_y), fill=(245, 244, 247), width=1)

# Large content area remains white (no text/icons drawn here)
# Optionally add a very faint full-width guideline near where the date fields sit
guideline_y = 520
draw.line((40, guideline_y, w - 40, guideline_y), fill=(250, 249, 251), width=1)

# Bottom "Apply date range" bar background (rounded container behind the button)
# NOTE: the actual button/content will be pasted on top; this draws the structural background only.
btn_margin_x = 48
btn_margin_bottom = 40
btn_height = 144
btn_left = btn_margin_x
btn_right = w - btn_margin_x
btn_bottom = h - btn_margin_bottom
btn_top = btn_bottom - btn_height

# Draw a very light shadow above the bar
shadow_top = btn_top - 8
draw.rectangle((0, shadow_top, w, btn_top), fill=(248, 247, 249))

# Rounded rectangle container for the button area (subtle border)
outer_radius = 18
container_fill = (255, 255, 255)
container_outline = (142, 132, 148)  # muted purple-gray outline
draw.rounded_rectangle([btn_left, btn_top, btn_right, btn_bottom],
                       radius=outer_radius, fill=container_fill,
                       outline=container_outline, width=5)

# Thin inner separator line above the container to further define the area
draw.line((btn_left + 6, btn_top, btn_right - 6, btn_top), fill=(240, 238, 245), width=1)

# Final subtle side gutters to emphasize safe content area (non-intrusive)
gutter_color = (250, 250, 250)
draw.rectangle((0, header_bottom + 4, 24, h - 220), fill=gutter_color)
draw.rectangle((w - 24, header_bottom + 4, w, h - 220), fill=gutter_color)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_06_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-8/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_06_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-8/01_icon_5.25.png
try:
    _c1 = get_crop(1, 57, 63)
    canvas.paste(_c1, (182, 2), _c1)
except Exception:
    pass
layout["5.25"] = [182, 2, 239, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_06_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-8/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 61, 61)
    canvas.paste(_c2, (310, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [310, 3, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_06_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-8/03_icon_5.25.png
try:
    _c3 = get_crop(3, 57, 65)
    canvas.paste(_c3, (116, 2), _c3)
except Exception:
    pass
layout["5.25"] = [116, 2, 173, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_06_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-8/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 50, 59)
    canvas.paste(_c4, (249, 6), _c4)
except Exception:
    pass
layout["icon_4"] = [249, 6, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_06_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-8/05_icon_5.25.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (12, 72), _c5)
except Exception:
    pass
layout["5.25"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_06_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-8/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 45, 66)
    canvas.paste(_c6, (1325, 2), _c6)
except Exception:
    pass
layout["icon_6"] = [1325, 2, 1370, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_06_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-8/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 67, 68)
    canvas.paste(_c7, (1214, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1214, 0, 1281, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_06_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-8/08_icon_What_date.png
try:
    _c8 = get_crop(8, 318, 73)
    canvas.paste(_c8, (558, 111), _c8)
except Exception:
    pass
layout["What_date?"] = [558, 111, 876, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_06_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-8/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 56, 68)
    canvas.paste(_c9, (1258, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1258, 0, 1314, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_06_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-8/10_icon_5.25.png
try:
    _c10 = get_crop(10, 90, 62)
    canvas.paste(_c10, (17, 3), _c10)
except Exception:
    pass
layout["5.25"] = [17, 3, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_06_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-8/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 47, 63)
    canvas.paste(_c11, (384, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [384, 3, 431, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_06_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-8/12_text_Start_Date.png
try:
    _c12 = get_crop(12, 589, 144)
    canvas.paste(_c12, (48, 313), _c12)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 637, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_06_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-8/13_text_End_Date.png
try:
    _c13 = get_crop(13, 253, 67)
    canvas.paste(_c13, (45, 437), _c13)
except Exception:
    pass
layout["End_Date"] = [45, 437, 298, 504]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/02f151acef934b59b90856d9e8041920/step_06_2024_4_24_17_24_02f151acef934b59b90856d9e8041920-8/14_clickable_Choose_a_date.png
try:
    _c14 = get_crop(14, 638, 144)
    canvas.paste(_c14, (48, 476), _c14)
except Exception:
    pass
layout["Choose_a_date"] = [48, 476, 686, 620]
