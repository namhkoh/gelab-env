# page_id: page_eventbrite_86c0bd1901f44c94916665f4058f9b6d_02
# screenshot: 2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-4.png
# step_index: 2/11
# task: Open Eventbrite. Set the city to Los Angeles. Select the 'Food & Drink' category. What's the date of the first event that is not promoted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar (top area)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(200, 200, 200))
# subtle bottom border under status bar
draw.line((0, status_h, 1440, status_h), fill=(170, 170, 170), width=1)

# Thin header divider under the main title area
# The "Find events in..." text block ends around y ~ 393, place divider slightly below that
divider_y = 396
draw.line((48, divider_y, 1392, divider_y), fill=(190, 185, 195), width=2)

# Group background for the "Nearby / Online events" option row
opts_x0, opts_x1 = 48, 1392
opts_y0, opts_y1 = 410, 600
draw.rounded_rectangle((opts_x0, opts_y0, opts_x1, opts_y1),
                       radius=24,
                       fill=(250, 251, 253),
                       outline=(235, 235, 238))

# subtle inner top divider for options group
draw.line((opts_x0 + 24, opts_y0 + 72, opts_x1 - 24, opts_y0 + 72), fill=(235, 235, 238), width=1)

# Separator between sections above the browsing block
sep_y = 720
draw.line((48, sep_y, 1392, sep_y), fill=(240, 240, 242), width=1)

# Card background for the "Browsing in" / "Chicago" selection
browse_x0, browse_x1 = 48, 1392
browse_y0, browse_y1 = 800, 980
draw.rounded_rectangle((browse_x0, browse_y0, browse_x1, browse_y1),
                       radius=20,
                       fill=(255, 255, 255),
                       outline=(230, 230, 235))

# Light drop shadow under the browsing card (very subtle)
shadow_y = browse_y1 + 2
draw.line((browse_x0 + 6, shadow_y, browse_x1 - 6, shadow_y), fill=(245, 245, 247), width=2)

# Another faint horizontal separator lower on the page to structure content area
draw.line((48, 1100, 1392, 1100), fill=(245, 245, 247), width=1)

# Bottom area remains plain white for content to be pasted later
# (No text/icons drawn here — these will be overlaid by detected elements)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_02_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-4/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_02_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-4/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 60, 60)
    canvas.paste(_c1, (310, 4), _c1)
except Exception:
    pass
layout["icon_1"] = [310, 4, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_02_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-4/02_icon_7.13.png
try:
    _c2 = get_crop(2, 59, 62)
    canvas.paste(_c2, (180, 2), _c2)
except Exception:
    pass
layout["7.13"] = [180, 2, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_02_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-4/03_icon_7.13.png
try:
    _c3 = get_crop(3, 57, 63)
    canvas.paste(_c3, (116, 2), _c3)
except Exception:
    pass
layout["7.13"] = [116, 2, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_02_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-4/04_icon_7.13.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["7.13"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_02_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-4/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 49, 57)
    canvas.paste(_c5, (249, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_02_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-4/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 46, 59)
    canvas.paste(_c6, (1322, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [1322, 3, 1368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_02_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-4/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 60, 64)
    canvas.paste(_c7, (1213, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1213, 1, 1273, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_02_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 43, 63)
    canvas.paste(_c8, (1270, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1270, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_02_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-4/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 51, 65)
    canvas.paste(_c9, (383, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_02_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-4/10_icon_7.13.png
try:
    _c10 = get_crop(10, 93, 62)
    canvas.paste(_c10, (14, 2), _c10)
except Exception:
    pass
layout["7.13"] = [14, 2, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_02_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-4/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_02_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-4/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_02_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-4/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_02_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-4/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_02_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-4/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_02_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-4/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_02_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-4/17_text_Chicago.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["Chicago"] = [0, 816, 1440, 954]
