# page_id: page_eventbrite_b45cca13f24546f9824a1ca2aab19c63_06
# screenshot: 2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-8.png
# step_index: 6/11
# task: Open Eventbrite. Search for "Art". Filter for events in New York. Select first recommended event. Save it to wishlist. What is the duration of the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw background and structural UI elements for the page
w, h = canvas.size

# Background fill (dominant white)
draw.rectangle([0, 0, w, h], fill=(255, 255, 255))

# Status bar (top area ~72px) - muted gray
status_h = 72
draw.rectangle([0, 0, w, status_h], fill=(168, 168, 168))

# Header / toolbar area under the status bar (subtle off-white)
header_h = 140
header_top = status_h
header_bottom = status_h + header_h
draw.rectangle([0, header_top, w, header_bottom], fill=(250, 250, 252))

# Subtle divider below header
divider_y = header_bottom
draw.line([48, divider_y, w - 48, divider_y], fill=(220, 220, 230), width=2)

# "Nearby" option card background (rounded rectangle)
card_margin_x = 36
card_y = 320
card_h = 160
card_rect = [card_margin_x, card_y, w - card_margin_x, card_y + card_h]
draw.rounded_rectangle(card_rect, radius=12, fill=(250, 250, 255), outline=(230, 230, 235), width=1)

# Light circular background behind the nearby icon (keeps icon layer separate)
badge_center_x = card_margin_x + 66
badge_center_y = card_y + 54
badge_r = 48
draw.ellipse([badge_center_x - badge_r, badge_center_y - badge_r, badge_center_x + badge_r, badge_center_y + badge_r],
             fill=(227, 242, 255))

# Thin separator below the Nearby card
sep_y = card_y + card_h + 28
draw.line([card_margin_x, sep_y, w - card_margin_x, sep_y], fill=(240, 240, 245), width=1)

# "Browsing in" small label region separator (visual grouping)
group_top = sep_y + 26
draw.rectangle([card_margin_x, group_top, w - card_margin_x, group_top + 6], fill=(245, 245, 248))

# Large section card for "Online events" area (subtle background to group the selection)
online_top = group_top + 24
online_h = 140
online_rect = [0, online_top, w, online_top + online_h]
draw.rectangle(online_rect, fill=(255, 255, 255))

# Right-side pale circle background where the selection/checkmark sits (behind pasted check icon)
right_circle_center = (w - 130, online_top + online_h // 2)
right_circle_r = 56
draw.ellipse([right_circle_center[0] - right_circle_r, right_circle_center[1] - right_circle_r,
              right_circle_center[0] + right_circle_r, right_circle_center[1] + right_circle_r],
             fill=(250, 249, 252))

# Subtle bottom divider under the online section
draw.line([card_margin_x, online_top + online_h + 12, w - card_margin_x, online_top + online_h + 12],
          fill=(235, 235, 240), width=1)

# Small visual accent: a faint vertical guideline at left (to align content, not text)
draw.line([card_margin_x + 12, header_bottom + 6, card_margin_x + 12, h - 120], fill=(245, 245, 248), width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_06_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-8/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 47, 69)
    canvas.paste(_c0, (1155, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1155, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_06_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-8/01_icon_7.05.png
try:
    _c1 = get_crop(1, 168, 168)
    canvas.paste(_c1, (0, 72), _c1)
except Exception:
    pass
layout["7.05"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_06_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-8/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 98, 63)
    canvas.paste(_c2, (1214, 1), _c2)
except Exception:
    pass
layout["icon_2"] = [1214, 1, 1312, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_06_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-8/03_icon_7.05.png
try:
    _c3 = get_crop(3, 62, 64)
    canvas.paste(_c3, (179, 1), _c3)
except Exception:
    pass
layout["7.05"] = [179, 1, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_06_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-8/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 63, 62)
    canvas.paste(_c4, (308, 2), _c4)
except Exception:
    pass
layout["icon_4"] = [308, 2, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_06_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-8/05_icon_7.05.png
try:
    _c5 = get_crop(5, 62, 65)
    canvas.paste(_c5, (113, 1), _c5)
except Exception:
    pass
layout["7.05"] = [113, 1, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_06_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-8/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 52, 58)
    canvas.paste(_c6, (247, 5), _c6)
except Exception:
    pass
layout["icon_6"] = [247, 5, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_06_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-8/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 103, 107)
    canvas.paste(_c7, (1291, 836), _c7)
except Exception:
    pass
layout["icon_7"] = [1291, 836, 1394, 943]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_06_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 47, 57)
    canvas.paste(_c8, (1322, 4), _c8)
except Exception:
    pass
layout["icon_8"] = [1322, 4, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_06_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-8/09_text_FFind_events_in..png
try:
    _c9 = get_crop(9, 1344, 129)
    canvas.paste(_c9, (48, 264), _c9)
except Exception:
    pass
layout["FFind_events_in."] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_06_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-8/10_text_Nearby.png
try:
    _c10 = get_crop(10, 415, 114)
    canvas.paste(_c10, (48, 465), _c10)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_06_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-8/11_text_Current_location.png
try:
    _c11 = get_crop(11, 415, 114)
    canvas.paste(_c11, (48, 465), _c11)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_06_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-8/12_text_Browsing_in.png
try:
    _c12 = get_crop(12, 228, 55)
    canvas.paste(_c12, (44, 742), _c12)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_06_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-8/13_text_Online_events.png
try:
    _c13 = get_crop(13, 1440, 138)
    canvas.paste(_c13, (0, 816), _c13)
except Exception:
    pass
layout["Online_events"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_06_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-8/14_text_Virtual_attendance.png
try:
    _c14 = get_crop(14, 1440, 138)
    canvas.paste(_c14, (0, 816), _c14)
except Exception:
    pass
layout["Virtual_attendance"] = [0, 816, 1440, 954]
