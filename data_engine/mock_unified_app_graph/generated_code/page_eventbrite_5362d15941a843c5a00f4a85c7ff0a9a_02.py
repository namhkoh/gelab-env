# page_id: page_eventbrite_5362d15941a843c5a00f4a85c7ff0a9a_02
# screenshot: 2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4.png
# step_index: 2/12
# task: Open Eventbrite. Set the city to 'Los Angeles'. Search 'Business'. Filter 'French' speaking events. Add the first event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (canvas already white, reinforce)
draw.rectangle([(0, 0), canvas.size], fill=(255, 255, 255))

# Colors
status_bar_color = (189, 189, 189)        # light gray for status bar
divider_color = (210, 205, 218)           # pale purple/gray divider
thin_divider_color = (230, 229, 235)      # very light divider
card_shadow = (236, 234, 240)             # subtle shadow for cards
card_border = (242, 240, 245)             # card border
card_fill = (255, 255, 255)               # card background (white)

w, h = canvas.size

# Status bar area (top)
status_bar_height = 72
draw.rectangle([(0, 0), (w, status_bar_height)], fill=status_bar_color)

# Header/toolbar area under status bar
header_top = status_bar_height
header_bottom = 144
draw.rectangle([(0, header_top), (w, header_bottom)], fill=card_fill)

# Subtle bottom border for header
draw.line([(32, header_bottom), (w-32, header_bottom)], fill=thin_divider_color, width=1)

# Underline for the "Find events in..." title
# Title detected at y ~264 with height ~129 -> underline just below that
title_underline_y = 264 + 129 + 0  # ~393
draw.line([(48, title_underline_y), (w-48, title_underline_y)], fill=divider_color, width=2)

# Separator line between the options area and the browsing section
# Place it somewhat below the chips region (chips text around 465)
separator_y = 560
draw.line([(32, separator_y), (w-32, separator_y)], fill=thin_divider_color, width=1)

# "Browsing in" card background behind the city row
# Detected city text around y ~816; draw a subtle rounded card behind to anchor it
card_x0 = 32
card_x1 = w - 32
card_y0 = 700
card_y1 = 980
card_radius = 20

# Draw shadow (slightly offset)
shadow_offset = 6
draw.rounded_rectangle(
    [(card_x0+shadow_offset, card_y0+shadow_offset), (card_x1+shadow_offset, card_y1+shadow_offset)],
    radius=card_radius, fill=card_shadow
)

# Draw main card
draw.rounded_rectangle(
    [(card_x0, card_y0), (card_x1, card_y1)],
    radius=card_radius, fill=card_fill, outline=card_border, width=1
)

# Small subtle divider above the card to visually separate the area
draw.line([(card_x0+12, card_y0), (card_x1-12, card_y0)], fill=thin_divider_color, width=1)

# Add a very light left accent line for the "Browsing in" section
accent_x = card_x0 + 12
draw.line([(accent_x, card_y0 + 20), (accent_x, card_y1 - 20)], fill=(241, 238, 245), width=4)

# Final subtle bottom global divider near page fold
draw.line([(32, h-220), (w-32, h-220)], fill=thin_divider_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_02_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_02_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 60, 61)
    canvas.paste(_c1, (310, 4), _c1)
except Exception:
    pass
layout["icon_1"] = [310, 4, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_02_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4/02_icon_8.01.png
try:
    _c2 = get_crop(2, 60, 63)
    canvas.paste(_c2, (180, 2), _c2)
except Exception:
    pass
layout["8.01"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_02_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4/03_icon_8.01.png
try:
    _c3 = get_crop(3, 60, 65)
    canvas.paste(_c3, (114, 1), _c3)
except Exception:
    pass
layout["8.01"] = [114, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_02_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4/04_icon_8.01.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["8.01"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_02_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 58)
    canvas.paste(_c5, (248, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 6, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_02_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 61, 64)
    canvas.paste(_c6, (1212, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 1, 1273, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_02_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 46, 59)
    canvas.paste(_c7, (1322, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1322, 3, 1368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_02_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 43, 63)
    canvas.paste(_c8, (1270, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1270, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_02_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 52, 65)
    canvas.paste(_c9, (382, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [382, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_02_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4/10_icon_8.01.png
try:
    _c10 = get_crop(10, 93, 64)
    canvas.paste(_c10, (12, 2), _c10)
except Exception:
    pass
layout["8.01"] = [12, 2, 105, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_02_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_02_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_02_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_02_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_02_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_02_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_02_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4/17_text_San_Francisco.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["San_Francisco"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_02_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-4/18_text_California.png
try:
    _c18 = get_crop(18, 188, 50)
    canvas.paste(_c18, (42, 902), _c18)
except Exception:
    pass
layout["California"] = [42, 902, 230, 952]
