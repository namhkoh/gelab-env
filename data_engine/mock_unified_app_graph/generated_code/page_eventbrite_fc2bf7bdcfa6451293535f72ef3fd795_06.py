# page_id: page_eventbrite_fc2bf7bdcfa6451293535f72ef3fd795_06
# screenshot: 2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-8.png
# step_index: 6/8
# task: Open Eventbrite. Search for events by 'Music' under online events. Choose the second event in the list. Get the event's duration information.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile page
# Assumes: canvas (1440x2960 RGB), draw (ImageDraw), font_sm/font_md/font_lg/font_xl available.

w, h = canvas.size

# 1) Status bar area (top ~72px) - light grey bar
status_bar_h = 72
draw.rectangle([(0, 0), (w, status_bar_h)], fill=(190, 190, 190))

# subtle thin bottom edge for status bar
draw.line([(0, status_bar_h-1), (w, status_bar_h-1)], fill=(170,170,170), width=1)

# 2) Header / section divider area
# Heading text "Find events in..." occupies roughly y 264..393 in detection; draw a subtle divider below it.
heading_bottom = 393
divider_y = heading_bottom + 24  # place divider slightly below heading area
draw.line([(48, divider_y), (w-48, divider_y)], fill=(205,200,208), width=2)

# 3) Row background for option icons (Nearby / Online events)
# Create a very subtle off-white band behind the options row so icons pasted on top remain visible
options_top = divider_y + 20
options_bottom = options_top + 160  # cover the area around the detected icons row (~y 465 height 114)
draw.rectangle([(24, options_top), (w-24, options_bottom)], fill=(250, 251, 253))

# subtle separator under the options row
draw.line([(24, options_bottom+12), (w-24, options_bottom+12)], fill=(230,226,232), width=1)

# 4) "Browsing in" / Location selection card background
# Draw a rounded selection card behind the location block (Los Angeles). Keep it subtle and mostly white.
loc_card_top = 700
loc_card_bottom = 980
loc_card_margin = 28
card_radius = 16
draw.rounded_rectangle(
    [(loc_card_margin, loc_card_top), (w - loc_card_margin, loc_card_bottom)],
    radius=card_radius,
    fill=(255,255,255),
    outline=(235,232,238),
    width=1
)

# 5) Divider lines separating main sections further down the page
# Add a light divider above the location card and one below to create structure
draw.line([(24, loc_card_top-12), (w-24, loc_card_top-12)], fill=(235,232,238), width=1)
draw.line([(24, loc_card_bottom+12), (w-24, loc_card_bottom+12)], fill=(245,243,247), width=1)

# 6) Large content background area (main content remains white; draw faint tint at bottom to avoid stark white)
# A very subtle gradient-like fill using horizontal translucent bands (simulated by slightly different rectangles)
band_top = loc_card_bottom + 40
band_height = 1600
band_step = 200
tints = [(255,255,255), (252,252,253), (250,250,251), (249,249,250)]
y = band_top
i = 0
while y < h:
    color = tints[i % len(tints)]
    draw.rectangle([(0, y), (w, min(h, y+band_step))], fill=color)
    y += band_step
    i += 1

# 7) Left side vertical gutter guideline (very subtle) to anchor content visually
gutter_x = 48
draw.line([(gutter_x, status_bar_h+8), (gutter_x, h-32)], fill=(245,243,247), width=1)

# 8) Right side vertical gutter guideline
gutter_x_r = w - 48
draw.line([(gutter_x_r, status_bar_h+8), (gutter_x_r, h-32)], fill=(245,243,247), width=1)

# Note: actual UI icons/text/buttons will be pasted on top of these structural elements at their detected positions.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_06_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-8/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_06_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-8/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 60, 62)
    canvas.paste(_c1, (310, 3), _c1)
except Exception:
    pass
layout["icon_1"] = [310, 3, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_06_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-8/02_icon_8.04.png
try:
    _c2 = get_crop(2, 60, 63)
    canvas.paste(_c2, (180, 2), _c2)
except Exception:
    pass
layout["8.04"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_06_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-8/03_icon_8.04.png
try:
    _c3 = get_crop(3, 59, 65)
    canvas.paste(_c3, (115, 1), _c3)
except Exception:
    pass
layout["8.04"] = [115, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_06_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-8/04_icon_8.04.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["8.04"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_06_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-8/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 57)
    canvas.paste(_c5, (248, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_06_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-8/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 62, 65)
    canvas.paste(_c6, (1212, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 0, 1274, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_06_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-8/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 46, 59)
    canvas.paste(_c7, (1322, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1322, 3, 1368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_06_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 43, 63)
    canvas.paste(_c8, (1270, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1270, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_06_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-8/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 52, 66)
    canvas.paste(_c9, (382, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [382, 1, 434, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_06_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-8/10_icon_Nearby.png
try:
    _c10 = get_crop(10, 415, 114)
    canvas.paste(_c10, (48, 465), _c10)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_06_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-8/11_text_8.04.png
try:
    _c11 = get_crop(11, 97, 50)
    canvas.paste(_c11, (18, 13), _c11)
except Exception:
    pass
layout["8.04"] = [18, 13, 115, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_06_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-8/12_text_Find_events_in..png
try:
    _c12 = get_crop(12, 1344, 129)
    canvas.paste(_c12, (48, 264), _c12)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_06_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-8/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_06_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-8/14_text_Virtual_attendance.png
try:
    _c14 = get_crop(14, 452, 114)
    canvas.paste(_c14, (511, 465), _c14)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_06_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-8/15_text_Browsing_in.png
try:
    _c15 = get_crop(15, 228, 55)
    canvas.paste(_c15, (44, 742), _c15)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/fc2bf7bdcfa6451293535f72ef3fd795/step_06_2024_4_23_20_3_fc2bf7bdcfa6451293535f72ef3fd795-8/16_text_Los_Angeles.png
try:
    _c16 = get_crop(16, 1440, 138)
    canvas.paste(_c16, (0, 816), _c16)
except Exception:
    pass
layout["Los_Angeles"] = [0, 816, 1440, 954]
