# page_id: page_eventbrite_5362d15941a843c5a00f4a85c7ff0a9a_03
# screenshot: 2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5.png
# step_index: 3/12
# task: Open Eventbrite. Set the city to 'Los Angeles'. Search 'Business'. Filter 'French' speaking events. Add the first event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for 1440x2960 canvas
# Available: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Full background (canvas already white, but ensure fill)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar area (top ~72px) - muted gray
STATUS_H = 72
draw.rectangle([(0, 0), (1440, STATUS_H)], fill=(190, 190, 190))

# Toolbar area (below status) - keep light/neutral background
TOOLBAR_H = 72  # height for toolbar region (from STATUS_H to STATUS_H+TOOLBAR_H)
draw.rectangle([(0, STATUS_H), (1440, STATUS_H + TOOLBAR_H)], fill=(255, 255, 255))

# Subtle bottom divider under toolbar
toolbar_bottom_y = STATUS_H + TOOLBAR_H
draw.line([(40, toolbar_bottom_y), (1400, toolbar_bottom_y)], fill=(230, 230, 235), width=2)

# Card/background area for search/options group (subtle tinted panel)
# This sits under the main search prompt and behind the two option groups.
search_card_top = 300
search_card_bottom = 480
draw.rounded_rectangle(
    [(36, search_card_top), (1404, search_card_bottom)],
    radius=14,
    fill=(249, 250, 255),
    outline=None
)

# Prominent accent underline for the search input (thin blue line)
# Positioned to align under the search prompt area (will be covered by pasted text)
underline_y = 392
draw.line([(48, underline_y), (1392, underline_y)], fill=(59, 84, 255), width=6)

# Separator between options area and browsing section (light hairline)
sep_y = 488
draw.line([(40, sep_y), (1400, sep_y)], fill=(235, 235, 238), width=1)

# Browsing-in section background (subtle grouped area)
browsing_top = 720
browsing_bottom = 980
draw.rounded_rectangle(
    [(36, browsing_top), (1404, browsing_bottom)],
    radius=12,
    fill=(255, 255, 255),
    outline=(245, 245, 248),
    width=1
)

# Light vertical guide left margin to define content column (very subtle)
draw.line([(48, 200), (48, 2600)], fill=(250, 250, 250), width=1)

# Right-side subtle divider to visually balance (will not overlap icons)
draw.line([(1400, 200), (1400, 2600)], fill=(250, 250, 250), width=1)

# Decorative faint horizontal grid lines to help structure whitespace (very subtle)
for y in (560, 720, 980, 1300, 1700, 2100):
    draw.line([(48, y), (1392, y)], fill=(250, 250, 251), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_03_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_03_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5/01_icon_8.01.png
try:
    _c1 = get_crop(1, 60, 63)
    canvas.paste(_c1, (180, 2), _c1)
except Exception:
    pass
layout["8.01"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_03_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5/02_icon_8.01.png
try:
    _c2 = get_crop(2, 61, 65)
    canvas.paste(_c2, (113, 1), _c2)
except Exception:
    pass
layout["8.01"] = [113, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_03_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 62, 62)
    canvas.paste(_c3, (309, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [309, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_03_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5/04_icon_8.01.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["8.01"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_03_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 59)
    canvas.paste(_c5, (248, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 5, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_03_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 65, 63)
    canvas.paste(_c6, (1212, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 1, 1277, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_03_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 47, 58)
    canvas.paste(_c7, (1322, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1322, 3, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_03_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 50, 62)
    canvas.paste(_c8, (1263, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1263, 1, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_03_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 52, 65)
    canvas.paste(_c9, (382, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [382, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_03_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5/10_icon_8.01.png
try:
    _c10 = get_crop(10, 93, 63)
    canvas.paste(_c10, (12, 2), _c10)
except Exception:
    pass
layout["8.01"] = [12, 2, 105, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_03_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5/11_text_FFind_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["FFind_events_in."] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_03_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_03_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_03_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_03_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_03_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_03_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5/17_text_San_Francisco.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["San_Francisco"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_03_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-5/18_text_California.png
try:
    _c18 = get_crop(18, 188, 50)
    canvas.paste(_c18, (42, 902), _c18)
except Exception:
    pass
layout["California"] = [42, 902, 230, 952]
