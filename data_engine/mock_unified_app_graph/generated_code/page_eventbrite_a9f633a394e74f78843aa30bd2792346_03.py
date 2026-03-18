# page_id: page_eventbrite_a9f633a394e74f78843aa30bd2792346_03
# screenshot: 2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-5.png
# step_index: 3/18
# task: Open Eventbrite. Set the city to "Los Angeles". Look for Photography workshops happening next week. What is the price of the tickets for first non-promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI elements for Eventbrite-like page

# Fill overall background (canvas starts white, but ensure exact tone)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar at top (~50-70px) - light grey
status_h = 70
draw.rectangle([(0, 0), (1440, status_h)], fill="#D0D0D0")

# App header area below status bar (keeps white, add subtle shadow line)
header_top = status_h
header_bottom = 280
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
# subtle bottom divider (very light)
draw.line([(32, header_bottom), (1440 - 32, header_bottom)], fill="#F0F0F2", width=1)

# Prominent blue underline/divider under the "Find events in..." heading
# (matches screenshot's bright blue accent)
blue_y = 380
draw.rectangle([(48, blue_y), (1440 - 48, blue_y + 6)], fill="#3B5BFF")

# Thin lighter divider just below blue line
draw.line([(48, blue_y + 10), (1440 - 48, blue_y + 10)], fill="#E9E9F5", width=1)

# Two circular badge backgrounds for "Nearby" and "Online events" (only backgrounds, no icons/text)
# Left badge (Nearby) - pale blue circle
left_badge_center = (140, 520)
left_badge_r = 64
draw.ellipse([
    (left_badge_center[0] - left_badge_r, left_badge_center[1] - left_badge_r),
    (left_badge_center[0] + left_badge_r, left_badge_center[1] + left_badge_r)
], fill="#EAF4FF")
# subtle inner highlight ring
draw.ellipse([
    (left_badge_center[0] - left_badge_r + 10, left_badge_center[1] - left_badge_r + 10),
    (left_badge_center[0] + left_badge_r - 10, left_badge_center[1] + left_badge_r - 10)
], outline="#D1E8FF", width=2)

# Right badge (Online events) - pale blue circle
right_badge_center = (620, 520)
right_badge_r = 64
draw.ellipse([
    (right_badge_center[0] - right_badge_r, right_badge_center[1] - right_badge_r),
    (right_badge_center[0] + right_badge_r, right_badge_center[1] + right_badge_r)
], fill="#EAF4FF")
draw.ellipse([
    (right_badge_center[0] - right_badge_r + 10, right_badge_center[1] - right_badge_r + 10),
    (right_badge_center[0] + right_badge_r - 10, right_badge_center[1] + right_badge_r - 10)
], outline="#D1E8FF", width=2)

# Group card background behind the two options (subtle grouping)
group_card_top = 460
group_card_bottom = 620
draw.rounded_rectangle([(28, group_card_top), (1440 - 28, group_card_bottom)], radius=12, fill="#FFFFFF", outline="#F2F2F6", width=1)

# "Browsing in" section background (rounded panel) where city selection appears
browse_top = 720
browse_bottom = 980
draw.rounded_rectangle([(24, browse_top), (1440 - 24, browse_bottom)], radius=14, fill="#FFFFFF", outline="#F4F4F7", width=1)

# Subtle circular selection background on the right (where a checkmark appears)
sel_center = (1320, 880)
sel_r = 48
draw.ellipse([
    (sel_center[0] - sel_r, sel_center[1] - sel_r),
    (sel_center[0] + sel_r, sel_center[1] + sel_r)
], fill="#F6F4FB")

# Divider line between header area and content (very faint)
draw.line([(48, header_bottom + 12), (1440 - 48, header_bottom + 12)], fill="#F3F3F7", width=1)

# Large empty content area (keeps white) with a faint top gradient-like divider
# (drawn as a very light rectangle to imply separation)
content_divider_top = browse_bottom + 24
draw.rectangle([(32, content_divider_top), (1440 - 32, content_divider_top + 8)], fill="#FBFBFD")

# Bottom safe area hint (subtle)
bottom_hint_top = 2960 - 80
draw.rectangle([(0, bottom_hint_top), (1440, 2960)], fill="#FFFFFF")
draw.line([(0, bottom_hint_top), (1440, bottom_hint_top)], fill="#F1F1F3", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_03_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-5/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_03_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-5/01_icon_4.50.png
try:
    _c1 = get_crop(1, 61, 65)
    canvas.paste(_c1, (179, 1), _c1)
except Exception:
    pass
layout["4.50"] = [179, 1, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_03_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-5/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 61, 62)
    canvas.paste(_c2, (309, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [309, 3, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_03_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-5/03_icon_4.50.png
try:
    _c3 = get_crop(3, 168, 168)
    canvas.paste(_c3, (0, 72), _c3)
except Exception:
    pass
layout["4.50"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_03_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-5/04_icon_4.50.png
try:
    _c4 = get_crop(4, 59, 65)
    canvas.paste(_c4, (115, 1), _c4)
except Exception:
    pass
layout["4.50"] = [115, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_03_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 58)
    canvas.paste(_c5, (248, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 6, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_03_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 63, 64)
    canvas.paste(_c6, (1212, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 1, 1275, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_03_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 46, 58)
    canvas.paste(_c7, (1322, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1322, 3, 1368, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_03_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-5/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 47, 62)
    canvas.paste(_c8, (1266, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1266, 1, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_03_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-5/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 52, 65)
    canvas.paste(_c9, (382, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [382, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_03_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-5/10_icon_Nearby.png
try:
    _c10 = get_crop(10, 415, 114)
    canvas.paste(_c10, (48, 465), _c10)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_03_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-5/11_text_4.50.png
try:
    _c11 = get_crop(11, 89, 43)
    canvas.paste(_c11, (22, 17), _c11)
except Exception:
    pass
layout["4.50"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_03_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-5/12_text_Find_events_in..png
try:
    _c12 = get_crop(12, 1344, 129)
    canvas.paste(_c12, (48, 264), _c12)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_03_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-5/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_03_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-5/14_text_Virtual_attendance.png
try:
    _c14 = get_crop(14, 452, 114)
    canvas.paste(_c14, (511, 465), _c14)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_03_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-5/15_text_Browsing_in.png
try:
    _c15 = get_crop(15, 228, 55)
    canvas.paste(_c15, (44, 742), _c15)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_03_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-5/16_text_Washington.png
try:
    _c16 = get_crop(16, 1440, 138)
    canvas.paste(_c16, (0, 816), _c16)
except Exception:
    pass
layout["Washington"] = [0, 816, 1440, 954]
