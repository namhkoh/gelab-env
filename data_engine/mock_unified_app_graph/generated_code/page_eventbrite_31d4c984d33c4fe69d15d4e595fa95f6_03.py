# page_id: page_eventbrite_31d4c984d33c4fe69d15d4e595fa95f6_03
# screenshot: 2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-5.png
# step_index: 3/14
# task: Open Eventbrite. Look for 'community events' in 'Chicago'. Select the first event happening tomorrow that is not promoted. Check if they have an option for 'refund policy'.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided. This script paints only backgrounds, bars, dividers and section surfaces.

w, h = 1440, 2960

# Clear canvas to white (canvas already white, but ensure)
draw.rectangle((0, 0, w, h), fill=(255, 255, 255))

# 1) Status bar (top area)
status_h = 72
status_color = (189, 189, 189)  # light muted gray
draw.rectangle((0, 0, w, status_h), fill=status_color)

# subtle inner highlight and shadow for status bar
draw.line((0, status_h - 1, w, status_h - 1), fill=(210, 210, 210))
draw.line((0, status_h, w, status_h), fill=(160, 160, 160))

# 2) App header area (below status bar)
header_top = status_h
header_bottom = 220
# keep header white but give a faint bottom divider to separate from content
draw.rectangle((0, header_top, w, header_bottom), fill=(255, 255, 255))
draw.line((48, header_bottom, w - 48, header_bottom), fill=(240, 240, 240), width=1)

# 3) Title underline (accent thin bar under "Find events in..." area)
# placed relative to detected title area (detected text y ~264 height ~129)
underline_y = 330
accent_left = 48
accent_right = w - 48
accent_color = (86, 101, 255)  # vivid bluish-purple accent
# draw a slightly thick accent line with subtle glow
draw.rectangle((accent_left, underline_y - 3, accent_right, underline_y + 3), fill=accent_color)
# subtle lighter highlight above the accent
draw.line((accent_left, underline_y - 4, accent_right, underline_y - 4), fill=(140, 150, 255))
# subtle shadow below the accent
draw.line((accent_left, underline_y + 4, accent_right, underline_y + 4), fill=(70, 80, 180))

# 4) Section separators
# separator under the "Nearby / Current location" block (keep thin, light)
sep1_y = 440
draw.line((48, sep1_y, w - 48, sep1_y), fill=(245, 245, 245), width=2)

# separator above "Browsing in" area
sep2_y = 720
draw.line((44, sep2_y, w - 44, sep2_y), fill=(248, 248, 248), width=1)

# 5) Subtle grouping card behind "Browsing in" heading (very pale background)
# This is a faint rounded panel to anchor that section, keep it extremely pale so it won't conflict with pasted content.
group_x0 = 36
group_y0 = 700
group_x1 = w - 36
group_y1 = 900
panel_color = (250, 250, 252)  # almost white, slight cool tint
draw.rounded_rectangle((group_x0, group_y0, group_x1, group_y1), radius=12, fill=panel_color, outline=None)

# add faint bottom divider for that card
draw.line((group_x0 + 8, group_y1 - 1, group_x1 - 8, group_y1 - 1), fill=(245, 245, 245))

# 6) Large empty content area background (remain white) - add a very faint overall vignette edge (subtle)
vignette_color = (255, 255, 255)
# Draw faint border to suggest screen edge (extremely subtle)
draw.rectangle((1, 1, w - 2, h - 2), outline=(250, 250, 250))

# Done: only background, bars, dividers and subtle section surfaces drawn.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_03_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-5/00_icon_8.07.png
try:
    _c0 = get_crop(0, 168, 168)
    canvas.paste(_c0, (0, 72), _c0)
except Exception:
    pass
layout["8.07"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_03_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-5/01_icon_8.07.png
try:
    _c1 = get_crop(1, 60, 64)
    canvas.paste(_c1, (180, 1), _c1)
except Exception:
    pass
layout["8.07"] = [180, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_03_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-5/02_icon_8.07.png
try:
    _c2 = get_crop(2, 62, 65)
    canvas.paste(_c2, (112, 1), _c2)
except Exception:
    pass
layout["8.07"] = [112, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_03_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 63, 61)
    canvas.paste(_c3, (308, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [308, 3, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_03_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-5/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 52, 58)
    canvas.paste(_c4, (247, 5), _c4)
except Exception:
    pass
layout["icon_4"] = [247, 5, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_03_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 103, 108)
    canvas.paste(_c5, (1291, 836), _c5)
except Exception:
    pass
layout["icon_5"] = [1291, 836, 1394, 944]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_03_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 48, 57)
    canvas.paste(_c6, (1321, 4), _c6)
except Exception:
    pass
layout["icon_6"] = [1321, 4, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_03_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 91, 61)
    canvas.paste(_c7, (1212, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1212, 1, 1303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_03_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-5/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 42, 59)
    canvas.paste(_c8, (1272, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1272, 3, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_03_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-5/09_icon_8.07.png
try:
    _c9 = get_crop(9, 95, 60)
    canvas.paste(_c9, (12, 4), _c9)
except Exception:
    pass
layout["8.07"] = [12, 4, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_03_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-5/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 53, 65)
    canvas.paste(_c10, (382, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [382, 1, 435, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_03_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-5/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_03_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-5/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_03_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-5/13_text_Current_location.png
try:
    _c13 = get_crop(13, 415, 114)
    canvas.paste(_c13, (48, 465), _c13)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_03_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-5/14_text_Browsing_in.png
try:
    _c14 = get_crop(14, 228, 55)
    canvas.paste(_c14, (44, 742), _c14)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_03_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-5/15_text_Online_events.png
try:
    _c15 = get_crop(15, 1440, 138)
    canvas.paste(_c15, (0, 816), _c15)
except Exception:
    pass
layout["Online_events"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_03_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-5/16_text_Virtual_attendance.png
try:
    _c16 = get_crop(16, 1440, 138)
    canvas.paste(_c16, (0, 816), _c16)
except Exception:
    pass
layout["Virtual_attendance"] = [0, 816, 1440, 954]
