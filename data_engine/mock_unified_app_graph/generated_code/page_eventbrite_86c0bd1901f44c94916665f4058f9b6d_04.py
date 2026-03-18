# page_id: page_eventbrite_86c0bd1901f44c94916665f4058f9b6d_04
# screenshot: 2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-6.png
# step_index: 4/11
# task: Open Eventbrite. Set the city to Los Angeles. Select the 'Food & Drink' category. What's the date of the first event that is not promoted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# paint overall background
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# status bar area (top)
status_h = 80
draw.rectangle((0, 0, 1440, status_h), fill="#E6E6E6")
# subtle bottom edge for status bar
draw.line((0, status_h - 1, 1440, status_h - 1), fill="#D0D0D0", width=1)

# header area (behind the "Los Angeles" label and back arrow)
header_top = status_h
header_bottom = 420
draw.rectangle((0, header_top, 1440, header_bottom), fill="#FFFFFF")

# prominent blue underline under the header (matches screenshot accent)
underline_y = 400
underline_x0 = 48
underline_x1 = 1392
draw.line((underline_x0, underline_y, underline_x1, underline_y), fill="#2F56FF", width=4)

# faint divider under the underline to give depth
draw.line((underline_x0, underline_y + 6, underline_x1, underline_y + 6), fill="#F0F2FF", width=1)

# circular pill backgrounds behind the two header options ("Nearby" and "Online events")
# Left option background (center computed relative to detected text box)
left_circle_center = (48 + 60, 465 + 40)
right_circle_center = (511 + 60, 465 + 40)
circle_r = 46

# light blue circular backgrounds
draw.ellipse(
    (
        left_circle_center[0] - circle_r,
        left_circle_center[1] - circle_r,
        left_circle_center[0] + circle_r,
        left_circle_center[1] + circle_r,
    ),
    fill="#EAF3FF",
    outline="#D6E9FF",
)

draw.ellipse(
    (
        right_circle_center[0] - circle_r,
        right_circle_center[1] - circle_r,
        right_circle_center[0] + circle_r,
        right_circle_center[1] + circle_r,
    ),
    fill="#EAF3FF",
    outline="#D6E9FF",
)

# subtle shadow dots beneath the option circles to anchor them
shadow_offset = 6
draw.ellipse(
    (
        left_circle_center[0] - circle_r + 2,
        left_circle_center[1] + circle_r - 6,
        left_circle_center[0] + circle_r - 2,
        left_circle_center[1] + circle_r - 2,
    ),
    fill="#FBFBFC",
    outline="#F0F0F0",
)

draw.ellipse(
    (
        right_circle_center[0] - circle_r + 2,
        right_circle_center[1] + circle_r - 6,
        right_circle_center[0] + circle_r - 2,
        right_circle_center[1] + circle_r - 2,
    ),
    fill="#FBFBFC",
    outline="#F0F0F0",
)

# large content card area placeholder (rounded rect) for event list background
card_x0 = 48
card_x1 = 1392
card_y0 = 520
card_y1 = 920
draw.rounded_rectangle((card_x0, card_y0, card_x1, card_y1), radius=18, fill="#FBFBFC", outline="#EEEEF3", width=1)

# small separator lines to indicate sections lower down
sep_y = card_y1 + 40
draw.line((card_x0, sep_y, card_x1, sep_y), fill="#F2F2F5", width=1)
draw.line((card_x0, sep_y + 40, card_x1, sep_y + 40), fill="#F2F2F5", width=1)

# another faint large card further down (placeholder for additional content)
card2_y0 = sep_y + 80
card2_y1 = card2_y0 + 360
draw.rounded_rectangle((card_x0, card2_y0, card_x1, card2_y1), radius=18, fill="#FFFFFF", outline="#F0F0F3", width=1)

# subtle page bottom edge line
draw.line((0, 2958, 1440, 2958), fill="#EFEFF1", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_04_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-6/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 47, 69)
    canvas.paste(_c0, (1155, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1155, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_04_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-6/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 60, 62)
    canvas.paste(_c1, (310, 3), _c1)
except Exception:
    pass
layout["icon_1"] = [310, 3, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_04_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-6/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 91, 66)
    canvas.paste(_c2, (1215, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1215, 0, 1306, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_04_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-6/03_icon_7.13.png
try:
    _c3 = get_crop(3, 168, 168)
    canvas.paste(_c3, (0, 72), _c3)
except Exception:
    pass
layout["7.13"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_04_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-6/04_icon_7.13.png
try:
    _c4 = get_crop(4, 61, 64)
    canvas.paste(_c4, (179, 1), _c4)
except Exception:
    pass
layout["7.13"] = [179, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_04_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-6/05_icon_7.13.png
try:
    _c5 = get_crop(5, 59, 65)
    canvas.paste(_c5, (115, 1), _c5)
except Exception:
    pass
layout["7.13"] = [115, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_04_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-6/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 49, 56)
    canvas.paste(_c6, (249, 7), _c6)
except Exception:
    pass
layout["icon_6"] = [249, 7, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_04_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-6/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 81, 93)
    canvas.paste(_c7, (1313, 287), _c7)
except Exception:
    pass
layout["icon_7"] = [1313, 287, 1394, 380]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_04_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 45, 60)
    canvas.paste(_c8, (1326, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1326, 3, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_04_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-6/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 50, 66)
    canvas.paste(_c9, (383, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 1, 433, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_04_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-6/10_icon_7.13.png
try:
    _c10 = get_crop(10, 93, 63)
    canvas.paste(_c10, (15, 2), _c10)
except Exception:
    pass
layout["7.13"] = [15, 2, 108, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_04_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-6/11_text_Los_Angeles.png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Los_Angeles"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_04_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-6/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_04_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-6/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_04_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-6/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_04_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-6/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_04_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-6/16_text_Loading.png
try:
    _c16 = get_crop(16, 156, 55)
    canvas.paste(_c16, (641, 1970), _c16)
except Exception:
    pass
layout["Loading"] = [641, 1970, 797, 2025]
