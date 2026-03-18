# page_id: page_eventbrite_3c2c0d71896b45acb211a472de4b4c9e_04
# screenshot: 2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6.png
# step_index: 4/15
# task: Open Eventbrite. Search free Health event in Los Angeles. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (canvas already white, but ensure consistent fill)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Top status bar area (subtle light gray)
status_h = 88
draw.rectangle([(0, 0), (1440, status_h)], fill="#F2F3F5")

# Top toolbar / header area (keeps white but with subtle bottom divider)
toolbar_top = status_h
toolbar_bottom = 176
draw.rectangle([(0, toolbar_top), (1440, toolbar_bottom)], fill="#FFFFFF")
# bottom divider under toolbar
draw.line([(24, toolbar_bottom), (1440-24, toolbar_bottom)], fill="#E6E7EA", width=1)

# Main title underline (accent colored rule under the "Find events in…" area)
# Position tuned to sit just under the detected title bounding box (y ~264..)
title_line_y = 340
draw.line([(48, title_line_y), (1440-48, title_line_y)], fill="#4A57F2", width=4)

# Two rounded "cards" / selection pills for Nearby and Online events (background panels)
left_card = (36, 420, 480, 560)
right_card = (495, 420, 1000, 560)
card_radius = 20
# subtle card fills
draw.rounded_rectangle(left_card, radius=card_radius, fill="#FFFFFF", outline="#ECECF1", width=1)
draw.rounded_rectangle(right_card, radius=card_radius, fill="#FFFFFF", outline="#ECECF1", width=1)
# faint shadow beneath cards (soft oval)
draw.ellipse([(left_card[0]+6, left_card[3]-6), (left_card[2]-6, left_card[3]+8)], fill="#FBFBFC")
draw.ellipse([(right_card[0]+6, right_card[3]-6), (right_card[2]-6, right_card[3]+8)], fill="#FBFBFC")

# Inside each card draw a faint circular backdrop for the icons (so pasted icons sit atop)
# Left circle (behind the icon) - light blue
left_circle_center = (left_card[0] + 64, (left_card[1] + left_card[3]) // 2)
left_circle_r = 44
draw.ellipse([
    (left_circle_center[0] - left_circle_r, left_circle_center[1] - left_circle_r),
    (left_circle_center[0] + left_circle_r, left_circle_center[1] + left_circle_r)],
    fill="#EAF0FF", outline=None)
# Right circle (behind the icon) - light blue
right_circle_center = (right_card[0] + 64, (right_card[1] + right_card[3]) // 2)
right_circle_r = 44
draw.ellipse([
    (right_circle_center[0] - right_circle_r, right_circle_center[1] - right_circle_r),
    (right_circle_center[0] + right_circle_r, right_circle_center[1] + right_circle_r)],
    fill="#EAF0FF", outline=None)

# Separator line before the "Browsing in" section
sep_y = 700
draw.line([(48, sep_y), (1440-48, sep_y)], fill="#F1F2F5", width=1)

# Large area for the "Browsing in" selection - keep background white but add subtle circle on right for check UI
# faint circular selectable background (where the check icon will be pasted on top)
check_center = (1310, 860)
check_r = 54
draw.ellipse([
    (check_center[0] - check_r, check_center[1] - check_r),
    (check_center[0] + check_r, check_center[1] + check_r)],
    fill="#F7F6FB", outline=None)

# Thin divider under the city row area to subtly separate from the rest of the content
city_row_bottom = 970
draw.line([(48, city_row_bottom), (1440-48, city_row_bottom)], fill="#F3F4F6", width=1)

# Add a very faint large background tint to the entire content area to match screenshot's soft tone near top
overlay_top = toolbar_bottom
overlay_bottom = 1100
draw.rectangle([(0, overlay_top), (1440, overlay_bottom)], fill="#FFFFFF")

# End of structural/background drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_04_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_04_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6/01_icon_9.41.png
try:
    _c1 = get_crop(1, 58, 64)
    canvas.paste(_c1, (180, 1), _c1)
except Exception:
    pass
layout["9.41"] = [180, 1, 238, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_04_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6/02_icon_9.41.png
try:
    _c2 = get_crop(2, 53, 63)
    canvas.paste(_c2, (115, 2), _c2)
except Exception:
    pass
layout["9.41"] = [115, 2, 168, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_04_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6/03_icon_9.41.png
try:
    _c3 = get_crop(3, 168, 168)
    canvas.paste(_c3, (0, 72), _c3)
except Exception:
    pass
layout["9.41"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_04_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 53, 62)
    canvas.paste(_c4, (315, 2), _c4)
except Exception:
    pass
layout["icon_4"] = [315, 2, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_04_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 54, 62)
    canvas.paste(_c5, (247, 1), _c5)
except Exception:
    pass
layout["icon_5"] = [247, 1, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_04_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 66, 63)
    canvas.paste(_c6, (1212, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 1, 1278, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_04_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 47, 58)
    canvas.paste(_c7, (1322, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1322, 3, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_04_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 42, 60)
    canvas.paste(_c8, (1272, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1272, 3, 1314, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_04_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 49, 64)
    canvas.paste(_c9, (383, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 0, 432, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_04_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6/10_text_9.41.png
try:
    _c10 = get_crop(10, 93, 50)
    canvas.paste(_c10, (18, 12), _c10)
except Exception:
    pass
layout["9.41"] = [18, 12, 111, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_04_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_04_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_04_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_04_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_04_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_04_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_04_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6/17_text_New_York.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["New_York"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_04_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-6/18_text_New_York.png
try:
    _c18 = get_crop(18, 182, 50)
    canvas.paste(_c18, (44, 905), _c18)
except Exception:
    pass
layout["New_York"] = [44, 905, 226, 955]
