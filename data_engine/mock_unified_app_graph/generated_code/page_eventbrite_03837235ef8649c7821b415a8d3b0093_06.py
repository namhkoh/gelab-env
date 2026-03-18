# page_id: page_eventbrite_03837235ef8649c7821b415a8d3b0093_06
# screenshot: 2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-8.png
# step_index: 6/8
# task: Open Eventbrite. Locate the 'Conference' category. Filter the results to only show virtual events. Choose the first event from the results. What is the duration of this event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint background and structural UI elements for the page
# Assumes: canvas (1440x2960 PIL Image), draw (ImageDraw), fonts available

# Full background (slightly warm white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar area (top system bar)
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill="#F2F3F5")
# Thin divider under status bar
draw.line([(24, status_h), (1416, status_h)], fill="#E6E3E9", width=1)

# App header area (space for back arrow etc)
header_top = status_h
header_bottom = 200
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
# Subtle bottom divider under header/toolbar
draw.line([(24, header_bottom), (1416, header_bottom)], fill="#E9E6EE", width=1)

# Main title divider (the long thin rule under the "Find events in..." area)
title_divider_y = 250
draw.line([(48, title_divider_y), (1392, title_divider_y)], fill="#D9D4DC", width=2)

# Option row background (rounded card area that holds the "Nearby" and "Online events" groups)
options_top = 360
options_bottom = 600
options_padding_lr = 36
draw.rounded_rectangle(
    [(options_padding_lr, options_top), (1440 - options_padding_lr, options_bottom)],
    radius=20,
    fill="#FFFFFF",
    outline=None
)
# Very subtle shadow line under the options card to separate from content below
draw.line([(options_padding_lr + 8, options_bottom + 2), (1440 - options_padding_lr - 8, options_bottom + 2)],
          fill="#F0EDF2", width=2)

# Divider between the two option columns (visual separator aligned roughly between the two groups)
mid_x = 512
sep_y1 = options_top + 24
sep_y2 = options_bottom - 24
draw.line([(mid_x, sep_y1), (mid_x, sep_y2)], fill="#FBFBFD", width=1)

# "Browsing in" section header area separator
browse_top = 700
draw.line([(48, browse_top), (1392, browse_top)], fill="#F0EDF2", width=1)

# Location card background (subtle area behind the location content)
loc_card_top = 760
loc_card_bottom = 980
draw.rectangle([(24, loc_card_top), (1416, loc_card_bottom)], fill="#FFFFFF")
# Bottom separator under location card
draw.line([(24, loc_card_bottom), (1416, loc_card_bottom)], fill="#F0EDF2", width=1)

# Long faint divider to suggest content area separation further down
draw.line([(48, 1120), (1392, 1120)], fill="#F7F6F8", width=1)

# Add a subtle vertical rhythm lines on left to guide layout (very faint)
for y in range(1300, 2800, 300):
    draw.line([(48, y), (1392, y)], fill="#FBFBFD", width=1)

# Final subtle bottom area tint to give depth (very light)
draw.rectangle([(0, 2800), (1440, 2960)], fill="#FFFFFF")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_06_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-8/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_06_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-8/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 60, 61)
    canvas.paste(_c1, (310, 4), _c1)
except Exception:
    pass
layout["icon_1"] = [310, 4, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_06_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-8/02_icon_4.41.png
try:
    _c2 = get_crop(2, 60, 65)
    canvas.paste(_c2, (114, 1), _c2)
except Exception:
    pass
layout["4.41"] = [114, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_06_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-8/03_icon_4.41.png
try:
    _c3 = get_crop(3, 60, 64)
    canvas.paste(_c3, (180, 2), _c3)
except Exception:
    pass
layout["4.41"] = [180, 2, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_06_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-8/04_icon_4.41.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["4.41"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_06_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-8/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 49, 58)
    canvas.paste(_c5, (249, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 6, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_06_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-8/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 45, 59)
    canvas.paste(_c6, (1323, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [1323, 3, 1368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_06_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-8/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 60, 64)
    canvas.paste(_c7, (1213, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1213, 1, 1273, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_06_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 42, 63)
    canvas.paste(_c8, (1271, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1271, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_06_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-8/09_icon_4.41.png
try:
    _c9 = get_crop(9, 92, 64)
    canvas.paste(_c9, (14, 1), _c9)
except Exception:
    pass
layout["4.41"] = [14, 1, 106, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_06_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-8/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 51, 65)
    canvas.paste(_c10, (383, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [383, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_06_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-8/11_icon_Nearby.png
try:
    _c11 = get_crop(11, 415, 114)
    canvas.paste(_c11, (48, 465), _c11)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_06_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-8/12_text_Find_events_in..png
try:
    _c12 = get_crop(12, 1344, 129)
    canvas.paste(_c12, (48, 264), _c12)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_06_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-8/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_06_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-8/14_text_Virtual_attendance.png
try:
    _c14 = get_crop(14, 452, 114)
    canvas.paste(_c14, (511, 465), _c14)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_06_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-8/15_text_Browsing_in.png
try:
    _c15 = get_crop(15, 228, 55)
    canvas.paste(_c15, (44, 742), _c15)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_06_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-8/16_text_San_Francisco.png
try:
    _c16 = get_crop(16, 1440, 138)
    canvas.paste(_c16, (0, 816), _c16)
except Exception:
    pass
layout["San_Francisco"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_06_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-8/17_text_California.png
try:
    _c17 = get_crop(17, 188, 50)
    canvas.paste(_c17, (42, 902), _c17)
except Exception:
    pass
layout["California"] = [42, 902, 230, 952]
