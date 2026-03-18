# page_id: page_eventbrite_5362d15941a843c5a00f4a85c7ff0a9a_04
# screenshot: 2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-6.png
# step_index: 4/12
# task: Open Eventbrite. Set the city to 'Los Angeles'. Search 'Business'. Filter 'French' speaking events. Add the first event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structure drawing for mobile UI (canvas and draw are provided)

# Canvas dimensions
W, H = 1440, 2960

# Colors
WHITE = "#FFFFFF"
STATUS_BAR = "#CFCFCF"      # light gray status bar
HEADER_DIVIDER_BLUE = "#2A56D7"  # bright blue underline
SUBTLE_LINE = "#E6E6E9"     # subtle divider lines
CARD_BG = "#F5FAFF"        # very light blue card background
PUNCH_BG = "#FBFBFD"       # very light neutral for large background shapes

# Clear/fill background
draw.rectangle([(0, 0), (W, H)], fill=WHITE)

# Status bar (top)
status_h = 72
draw.rectangle([(0, 0), (W, status_h)], fill=STATUS_BAR)

# Header area (below status bar)
header_top = status_h
header_bottom = 220
draw.rectangle([(0, header_top), (W, header_bottom)], fill=WHITE)

# Blue underline for the header/title (spans most of the width with side margins)
uline_left = 48
uline_right = W - 48
uline_y = header_bottom - 8
uline_height = 4
draw.rectangle([(uline_left, uline_y), (uline_right, uline_y + uline_height)], fill=HEADER_DIVIDER_BLUE)

# Thin subtle divider below header for separation
divider_y = header_bottom + 12
draw.rectangle([(24, divider_y), (W - 24, divider_y + 1)], fill=SUBTLE_LINE)

# Section group backgrounds (rounded cards behind "Nearby" and "Online events")
# Left card (Nearby group)
left_card = (36, 420, 480, 540)
draw.rounded_rectangle(left_card, radius=18, fill=CARD_BG, outline=None)

# Right card (Online events / Virtual attendance)
right_card = (496, 420, 980, 540)
draw.rounded_rectangle(right_card, radius=18, fill=CARD_BG, outline=None)

# Subtle horizontal separator line below the section group
sep_y = 560
draw.rectangle([(24, sep_y), (W - 24, sep_y + 1)], fill=SUBTLE_LINE)

# Large faint circular background around the middle where loading spinner appears
# (keeps it as a soft background element, spinner and "Loading" text will be pasted on top)
circle_center = (W // 2, 1700)
circle_radius = 220
draw.ellipse([
    (circle_center[0] - circle_radius, circle_center[1] - circle_radius),
    (circle_center[0] + circle_radius, circle_center[1] + circle_radius)
], fill=PUNCH_BG)

# Another faint radial accent (subtle arc-like band) to echo the spinner area
arc_outer = (circle_center[0] - 260, circle_center[1] - 260, circle_center[0] + 260, circle_center[1] + 260)
arc_inner = (circle_center[0] - 180, circle_center[1] - 180, circle_center[0] + 180, circle_center[1] + 180)
# draw outer light ring
draw.ellipse(arc_outer, outline="#F1F2F4", width=2)
# draw inner lighter ring
draw.ellipse(arc_inner, outline="#F7F8FA", width=2)

# Final subtle bottom divider to ground the page (near bottom)
bottom_div_y = H - 160
draw.rectangle([(24, bottom_div_y), (W - 24, bottom_div_y + 1)], fill=SUBTLE_LINE)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_04_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-6/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 47, 69)
    canvas.paste(_c0, (1155, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1155, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_04_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-6/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 96, 66)
    canvas.paste(_c1, (1214, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1214, 0, 1310, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_04_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-6/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 61, 62)
    canvas.paste(_c2, (309, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [309, 3, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_04_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-6/03_icon_8.01.png
try:
    _c3 = get_crop(3, 168, 168)
    canvas.paste(_c3, (0, 72), _c3)
except Exception:
    pass
layout["8.01"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_04_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-6/04_icon_8.01.png
try:
    _c4 = get_crop(4, 61, 65)
    canvas.paste(_c4, (179, 1), _c4)
except Exception:
    pass
layout["8.01"] = [179, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_04_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-6/05_icon_8.01.png
try:
    _c5 = get_crop(5, 61, 65)
    canvas.paste(_c5, (113, 1), _c5)
except Exception:
    pass
layout["8.01"] = [113, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_04_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-6/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 48, 57)
    canvas.paste(_c6, (250, 6), _c6)
except Exception:
    pass
layout["icon_6"] = [250, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_04_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-6/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 45, 60)
    canvas.paste(_c7, (1326, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1326, 3, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_04_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 81, 93)
    canvas.paste(_c8, (1313, 287), _c8)
except Exception:
    pass
layout["icon_8"] = [1313, 287, 1394, 380]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_04_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-6/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 50, 66)
    canvas.paste(_c9, (383, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 1, 433, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_04_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-6/10_text_8.01.png
try:
    _c10 = get_crop(10, 89, 43)
    canvas.paste(_c10, (20, 17), _c10)
except Exception:
    pass
layout["8.01"] = [20, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_04_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-6/11_text_Los_Angeles.png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Los_Angeles"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_04_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-6/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_04_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-6/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_04_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-6/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_04_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-6/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_04_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-6/16_text_Loading.png
try:
    _c16 = get_crop(16, 156, 55)
    canvas.paste(_c16, (641, 1970), _c16)
except Exception:
    pass
layout["Loading"] = [641, 1970, 797, 2025]
