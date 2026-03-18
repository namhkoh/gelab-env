# page_id: page_eventbrite_5059e61823f845cf95cbd1d8d54254c6_02
# screenshot: 2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4.png
# step_index: 2/19
# task: Open Eventbrite. Look for 'Education' in Los Angeles happening on May 4. Filter to show only free events. How many events are posted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background base
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar area at top (dark grey background)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(150, 150, 150))

# Thin subtle divider under status bar / app bar
draw.line((0, status_h, 1440, status_h), fill=(190, 190, 190), width=1)

# Main header area (keeps white but add a subtle bottom divider under the "Find events in..." area)
# The detected "Find events in..." text top is at y=264 with height 129 -> underline roughly below that.
underline_y = 264 + 129 + 10
draw.line((48, underline_y, 1440-48, underline_y), fill=(215, 206, 225), width=2)

# Light separator under the icon options block (approx below icon area)
icons_block_bottom = 465 + 114
sep_y = icons_block_bottom + 20
draw.line((24, sep_y, 1440-24, sep_y), fill=(240, 238, 243), width=1)

# Section card background for "Browsing in / New York" group
card_left = 36
card_right = 1440 - 36
card_top = 700
card_bottom = 1000
card_radius = 22
# subtle shadow (a slightly darker thin band behind the card)
shadow_offset = 6
draw.rounded_rectangle(
    (card_left + shadow_offset, card_top + shadow_offset, card_right + shadow_offset, card_bottom + shadow_offset),
    radius=card_radius, fill=(245, 244, 246)
)
# actual card
draw.rounded_rectangle(
    (card_left, card_top, card_right, card_bottom),
    radius=card_radius, fill=(250, 249, 252), outline=(235, 232, 239), width=1
)

# Subtle left grouping bar to visually anchor the "Browsing in" heading (background stripe)
stripe_x0 = card_left + 18
stripe_x1 = stripe_x0 + 6
draw.rectangle((stripe_x0, card_top + 22, stripe_x1, card_top + 22 + 48), fill=(230, 224, 236))

# A faint large-divider near the middle to separate header area from content (very subtle)
mid_div_y = int((underline_y + card_top) / 2)
draw.line((24, mid_div_y, 1440-24, mid_div_y), fill=(247, 246, 248), width=1)

# Bottom area remains white (no content drawn) to allow pasted elements to appear on top

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_02_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_02_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4/01_icon_7.34.png
try:
    _c1 = get_crop(1, 58, 63)
    canvas.paste(_c1, (115, 2), _c1)
except Exception:
    pass
layout["7.34"] = [115, 2, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_02_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 60, 61)
    canvas.paste(_c2, (310, 4), _c2)
except Exception:
    pass
layout["icon_2"] = [310, 4, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_02_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4/03_icon_7.34.png
try:
    _c3 = get_crop(3, 59, 63)
    canvas.paste(_c3, (180, 2), _c3)
except Exception:
    pass
layout["7.34"] = [180, 2, 239, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_02_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4/04_icon_7.34.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["7.34"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_02_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 49, 57)
    canvas.paste(_c5, (249, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_02_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4/06_icon_7.34.png
try:
    _c6 = get_crop(6, 93, 63)
    canvas.paste(_c6, (15, 1), _c6)
except Exception:
    pass
layout["7.34"] = [15, 1, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_02_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 59, 64)
    canvas.paste(_c7, (1213, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1213, 1, 1272, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_02_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 42, 63)
    canvas.paste(_c8, (1271, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1271, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_02_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 45, 59)
    canvas.paste(_c9, (1323, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [1323, 3, 1368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_02_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 51, 65)
    canvas.paste(_c10, (383, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [383, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_02_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_02_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_02_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_02_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_02_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_02_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_02_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4/17_text_New_York.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["New_York"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_02_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-4/18_text_New_York.png
try:
    _c18 = get_crop(18, 182, 50)
    canvas.paste(_c18, (44, 905), _c18)
except Exception:
    pass
layout["New_York"] = [44, 905, 226, 955]
