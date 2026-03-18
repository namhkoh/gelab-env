# page_id: page_eventbrite_a9f633a394e74f78843aa30bd2792346_04
# screenshot: 2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-6.png
# step_index: 4/18
# task: Open Eventbrite. Set the city to "Los Angeles". Look for Photography workshops happening next week. What is the price of the tickets for first non-promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (canvas: 1440x2960 RGB, draw: ImageDraw)
w, h = canvas.size

# Colors
bg = (250, 250, 252)         # very light off-white background
status_bar = (190, 190, 190) # light grey status bar
status_border = (180, 180, 180)
accent_blue = (43, 90, 255)  # vivid blue used for underline
chip_shadow = (230, 230, 235)
chip_bg = (255, 255, 255)
divider = (235, 235, 240)

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg)

# Status bar (top area)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar)
# subtle bottom border under status
draw.line([(0, status_h - 1), (w, status_h - 1)], fill=status_border, width=1)

# Header area (below status) - keep it visually separated (mostly white)
header_top = status_h
header_bottom = 160
draw.rectangle([(0, header_top), (w, header_bottom)], fill=bg)

# Blue underline below the location input (spans with horizontal margins)
underline_y = header_bottom
left_margin = 48
right_margin = 48
draw.line([(left_margin, underline_y), (w - right_margin, underline_y)], fill=accent_blue, width=4)

# Thin divider below underline for subtle separation
draw.line([(left_margin, underline_y + 6), (w - right_margin, underline_y + 6)], fill=divider, width=1)

# "Chips" / option group backgrounds (rounded rectangles behind Nearby / Online events)
# Left chip (matches approximate detected bounds)
left_chip_x = 48
left_chip_y = 430
left_chip_w = 415
left_chip_h = 120
radius = 20

# Shadow for left chip
draw.rounded_rectangle(
    [(left_chip_x + 0, left_chip_y + 6), (left_chip_x + left_chip_w, left_chip_y + left_chip_h + 6)],
    radius=radius,
    fill=chip_shadow
)
# Chip background
draw.rounded_rectangle(
    [(left_chip_x, left_chip_y), (left_chip_x + left_chip_w, left_chip_y + left_chip_h)],
    radius=radius,
    fill=chip_bg
)
# Right chip (approx bounds)
right_chip_x = 511
right_chip_y = left_chip_y
right_chip_w = 452
right_chip_h = 120

# Shadow for right chip
draw.rounded_rectangle(
    [(right_chip_x + 0, right_chip_y + 6), (right_chip_x + right_chip_w, right_chip_y + right_chip_h + 6)],
    radius=radius,
    fill=chip_shadow
)
# Chip background
draw.rounded_rectangle(
    [(right_chip_x, right_chip_y), (right_chip_x + right_chip_w, right_chip_y + right_chip_h)],
    radius=radius,
    fill=chip_bg
)

# Subtle horizontal separator line beneath the chips area
sep_y = right_chip_y + right_chip_h + 28
draw.line([(32, sep_y), (w - 32, sep_y)], fill=divider, width=1)

# Large content area background (leave most white, but add a faint centered section band to hint content region)
band_top = sep_y + 40
band_bottom = band_top + 420
band_margin = 80
band_color = (248, 249, 251)
draw.rectangle([(band_margin, band_top), (w - band_margin, band_bottom)], fill=band_color, outline=None)

# Subtle inner divider to indicate where a list of events would start
inner_div_y = band_top + 140
draw.line([(band_margin + 12, inner_div_y), (w - band_margin - 12, inner_div_y)], fill=divider, width=1)

# Footer subtle top separator (near bottom area)
draw.line([(32, h - 220), (w - 32, h - 220)], fill=divider, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_04_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-6/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 47, 69)
    canvas.paste(_c0, (1155, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1155, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_04_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-6/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 95, 66)
    canvas.paste(_c1, (1214, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1214, 0, 1309, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_04_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-6/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 60, 62)
    canvas.paste(_c2, (310, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [310, 3, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_04_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-6/03_icon_4.50.png
try:
    _c3 = get_crop(3, 168, 168)
    canvas.paste(_c3, (0, 72), _c3)
except Exception:
    pass
layout["4.50"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_04_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-6/04_icon_4.50.png
try:
    _c4 = get_crop(4, 61, 65)
    canvas.paste(_c4, (179, 1), _c4)
except Exception:
    pass
layout["4.50"] = [179, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_04_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-6/05_icon_4.50.png
try:
    _c5 = get_crop(5, 61, 66)
    canvas.paste(_c5, (114, 1), _c5)
except Exception:
    pass
layout["4.50"] = [114, 1, 175, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_04_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-6/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 47, 57)
    canvas.paste(_c6, (251, 6), _c6)
except Exception:
    pass
layout["icon_6"] = [251, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_04_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-6/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 45, 60)
    canvas.paste(_c7, (1326, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1326, 3, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_04_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 83, 86)
    canvas.paste(_c8, (1312, 290), _c8)
except Exception:
    pass
layout["icon_8"] = [1312, 290, 1395, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_04_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-6/09_icon_Nearby.png
try:
    _c9 = get_crop(9, 415, 114)
    canvas.paste(_c9, (48, 465), _c9)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_04_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-6/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 50, 66)
    canvas.paste(_c10, (383, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [383, 1, 433, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_04_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-6/11_icon_Los_Angeles.png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Los_Angeles"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_04_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-6/12_text_4.50.png
try:
    _c12 = get_crop(12, 89, 43)
    canvas.paste(_c12, (22, 17), _c12)
except Exception:
    pass
layout["4.50"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_04_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-6/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_04_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-6/14_text_Virtual_attendance.png
try:
    _c14 = get_crop(14, 452, 114)
    canvas.paste(_c14, (511, 465), _c14)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_04_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-6/15_text_Loading.png
try:
    _c15 = get_crop(15, 156, 55)
    canvas.paste(_c15, (641, 1970), _c15)
except Exception:
    pass
layout["Loading"] = [641, 1970, 797, 2025]
