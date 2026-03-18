# page_id: page_eventbrite_cee80d2c889b45cd86e30248baf590b8_02
# screenshot: 2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-4.png
# step_index: 2/13
# task: Open Eventbrite. Search Food & Drink party event in New York. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (canvas already provided)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Top status bar (dark grey strip)
status_h = 84
draw.rectangle([(0, 0), (1440, status_h)], fill="#9AA0A6")

# Subtle darker top edge for depth
draw.rectangle([(0, 0), (1440, 6)], fill="#8A8F93")

# Thin divider below header / search area (light muted purple/grey)
divider_y = 340
draw.line([(48, divider_y), (1392, divider_y)], fill="#C8BFD0", width=3)

# Light subtle horizontal guide under the "Find events" area
draw.line([(48, divider_y + 48), (1392, divider_y + 48)], fill="#EFEAF0", width=1)

# Two option cards (background panels for Nearby / Online events)
left_card = (32, 430, 464, 554)   # corresponds to left group region
right_card = (492, 430, 964, 554) # corresponds to right group region

# Soft rounded backgrounds (very light hues to separate from white)
try:
    draw.rounded_rectangle(left_card, radius=16, fill="#F6F8FF", outline="#E6EAF7", width=1)
    draw.rounded_rectangle(right_card, radius=16, fill="#F6F8FF", outline="#E6EAF7", width=1)
except AttributeError:
    # Fallback if rounded_rectangle is not available
    draw.rectangle(left_card, fill="#F6F8FF", outline="#E6EAF7")
    draw.rectangle(right_card, fill="#F6F8FF", outline="#E6EAF7")

# Subtle separator below the option cards
draw.line([(32, 580), (1408, 580)], fill="#F0EFF2", width=1)

# Large location selection card (background for "Browsing in / Los Angeles")
loc_card = (32, 720, 1408, 920)
try:
    draw.rounded_rectangle(loc_card, radius=10, fill="#FFFFFF", outline="#ECE9EE", width=1)
except AttributeError:
    draw.rectangle(loc_card, fill="#FFFFFF", outline="#ECE9EE")

# Soft shadow effect under the location card (very subtle)
shadow_top = 920
draw.rectangle([(32, shadow_top), (1408, shadow_top + 2)], fill="#F3F1F4")

# Faint circular background for the check mark area on the right (behind the pasted icon)
check_bg_center = (1290 + 103 // 2, 835 + 105 // 2)
check_bg_radius_x = 64
check_bg_radius_y = 64
check_bg_box = [
    (check_bg_center[0] - check_bg_radius_x, check_bg_center[1] - check_bg_radius_y),
    (check_bg_center[0] + check_bg_radius_x, check_bg_center[1] + check_bg_radius_y),
]
draw.ellipse([check_bg_box[0], check_bg_box[1]], fill="#F6F3FA", outline=None)

# Top-left back arrow area hint (a subtle touch area background, not an icon)
nav_hint = (24, 96, 120, 176)
try:
    draw.rounded_rectangle(nav_hint, radius=8, fill="#FFFFFF", outline="#EFEFF1", width=1)
except AttributeError:
    draw.rectangle(nav_hint, fill="#FFFFFF", outline="#EFEFF1")

# Horizontal spacing guide under header (thin line)
draw.line([(48, 220), (1392, 220)], fill="#FFFFFF", width=10)

# Footer separator near bottom of main content area (to indicate end of section)
draw.line([(48, 980), (1392, 980)], fill="#F4F2F6", width=1)

# Additional subtle vertical guide on the left (visual gutter)
draw.line([(32, 120), (32, 1200)], fill="#FFFFFF", width=24)

# Final slight vignette at very top to emulate the app chrome transition
draw.rectangle([(0, 0), (1440, 18)], fill="#8E9397")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_02_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-4/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_02_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-4/01_icon_9.44.png
try:
    _c1 = get_crop(1, 58, 64)
    canvas.paste(_c1, (180, 1), _c1)
except Exception:
    pass
layout["9.44"] = [180, 1, 238, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_02_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-4/02_icon_9.44.png
try:
    _c2 = get_crop(2, 168, 168)
    canvas.paste(_c2, (0, 72), _c2)
except Exception:
    pass
layout["9.44"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_02_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-4/03_icon_9.44.png
try:
    _c3 = get_crop(3, 52, 63)
    canvas.paste(_c3, (116, 2), _c3)
except Exception:
    pass
layout["9.44"] = [116, 2, 168, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_02_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-4/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 53, 62)
    canvas.paste(_c4, (315, 2), _c4)
except Exception:
    pass
layout["icon_4"] = [315, 2, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_02_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-4/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 61, 63)
    canvas.paste(_c5, (1213, 1), _c5)
except Exception:
    pass
layout["icon_5"] = [1213, 1, 1274, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_02_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-4/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 54, 62)
    canvas.paste(_c6, (247, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [247, 1, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_02_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-4/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 46, 58)
    canvas.paste(_c7, (1322, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1322, 3, 1368, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_02_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 50, 62)
    canvas.paste(_c8, (1263, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1263, 1, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_02_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-4/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 48, 65)
    canvas.paste(_c9, (383, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 0, 431, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_02_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-4/10_icon_9.44.png
try:
    _c10 = get_crop(10, 91, 63)
    canvas.paste(_c10, (16, 1), _c10)
except Exception:
    pass
layout["9.44"] = [16, 1, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_02_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-4/11_icon_Nearby.png
try:
    _c11 = get_crop(11, 415, 114)
    canvas.paste(_c11, (48, 465), _c11)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_02_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-4/12_text_Find_events_in..png
try:
    _c12 = get_crop(12, 1344, 129)
    canvas.paste(_c12, (48, 264), _c12)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_02_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-4/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_02_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-4/14_text_Virtual_attendance.png
try:
    _c14 = get_crop(14, 452, 114)
    canvas.paste(_c14, (511, 465), _c14)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_02_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-4/15_text_Browsing_in.png
try:
    _c15 = get_crop(15, 228, 55)
    canvas.paste(_c15, (44, 742), _c15)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_02_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-4/16_text_Los_Angeles.png
try:
    _c16 = get_crop(16, 1440, 138)
    canvas.paste(_c16, (0, 816), _c16)
except Exception:
    pass
layout["Los_Angeles"] = [0, 816, 1440, 954]
