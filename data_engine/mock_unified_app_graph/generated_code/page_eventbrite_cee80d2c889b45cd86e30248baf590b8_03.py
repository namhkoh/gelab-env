# page_id: page_eventbrite_cee80d2c889b45cd86e30248baf590b8_03
# screenshot: 2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-5.png
# step_index: 3/13
# task: Open Eventbrite. Search Food & Drink party event in New York. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for 1440x2960 canvas
# Available: canvas (PIL Image), draw (ImageDraw), font_* variables

# Base background (mostly white/off-white)
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# Status bar area (top strip)
status_bar_h = 72
draw.rectangle((0, 0, 1440, status_bar_h), fill="#9A9A9A")
# subtle bottom divider under status bar
draw.line((0, status_bar_h, 1440, status_bar_h), fill="#7E7E7E", width=2)

# Header area (toolbar) - keep it visually separated but mostly white
header_top = status_bar_h
header_bottom = 220
draw.rectangle((0, header_top, 1440, header_bottom), fill="#FFFFFF")
# light divider to separate header from content
draw.line((48, header_bottom, 1392, header_bottom), fill="#E6E6E9", width=1)

# Accent underline for search area (thin purple/blue)
# Placed a bit below header to represent the accent divider
accent_y = 320
draw.line((48, accent_y, 1392, accent_y), fill="#4B46FF", width=4)

# Two subtle pill/card backgrounds for "Nearby" and "Online events" groups
# Left card (Nearby)
card1_bbox = (36, 420, 480, 560)
draw.rounded_rectangle(card1_bbox, radius=18, fill="#F3F8FF", outline=None)
# little circular background inside left card (behind icon area)
circle1_center = (36 + 72, 420 + 64)
circle1_r = 44
draw.ellipse((circle1_center[0]-circle1_r, circle1_center[1]-circle1_r,
              circle1_center[0]+circle1_r, circle1_center[1]+circle1_r), fill="#DCEEFF")

# Right card (Online events)
card2_bbox = (500, 420, 944, 560)
draw.rounded_rectangle(card2_bbox, radius=18, fill="#F3F8FF", outline=None)
circle2_center = (500 + 72, 420 + 64)
circle2_r = 44
draw.ellipse((circle2_center[0]-circle2_r, circle2_center[1]-circle2_r,
              circle2_center[0]+circle2_r, circle2_center[1]+circle2_r), fill="#DCEEFF")

# Separator line below the card area
sep_y = 620
draw.line((48, sep_y, 1392, sep_y), fill="#ECECF0", width=1)

# Section header background area for "Browsing in" (subtle)
browse_top = 700
browse_bottom = 820
draw.rectangle((0, browse_top, 1440, browse_bottom), fill="#FFFFFF")
# subtle top divider for the browsing section
draw.line((48, browse_top, 1392, browse_top), fill="#F0F0F3", width=1)

# Large selectable row background for the city selection (rounded)
city_row_bbox = (24, 800, 1416, 960)
draw.rounded_rectangle(city_row_bbox, radius=12, fill="#FFFFFF", outline="#F1F1F4")
# faint top border for the city row to separate from content above
draw.line((48, 800, 1392, 800), fill="#F4F4F6", width=1)

# Additional subtle vertical guide on left margin
draw.line((48, header_bottom + 12, 48, 2960 - 48), fill="#FBFBFC", width=2)

# Bottom area remains clean white for content to be pasted later
# Add a faint footer divider near bottom (visual balance)
footer_div_y = 2860
draw.line((48, footer_div_y, 1392, footer_div_y), fill="#F2F2F4", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_03_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-5/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_03_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-5/01_icon_9.44.png
try:
    _c1 = get_crop(1, 58, 64)
    canvas.paste(_c1, (180, 1), _c1)
except Exception:
    pass
layout["9.44"] = [180, 1, 238, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_03_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-5/02_icon_9.44.png
try:
    _c2 = get_crop(2, 52, 63)
    canvas.paste(_c2, (116, 2), _c2)
except Exception:
    pass
layout["9.44"] = [116, 2, 168, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_03_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-5/03_icon_9.44.png
try:
    _c3 = get_crop(3, 168, 168)
    canvas.paste(_c3, (0, 72), _c3)
except Exception:
    pass
layout["9.44"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_03_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-5/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 53, 63)
    canvas.paste(_c4, (315, 1), _c4)
except Exception:
    pass
layout["icon_4"] = [315, 1, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_03_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 54, 62)
    canvas.paste(_c5, (247, 1), _c5)
except Exception:
    pass
layout["icon_5"] = [247, 1, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_03_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 66, 63)
    canvas.paste(_c6, (1212, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 1, 1278, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_03_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 48, 58)
    canvas.paste(_c7, (1321, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1321, 3, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_03_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-5/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 42, 60)
    canvas.paste(_c8, (1272, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1272, 3, 1314, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_03_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-5/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 49, 64)
    canvas.paste(_c9, (383, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 0, 432, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_03_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-5/10_icon_9.44.png
try:
    _c10 = get_crop(10, 91, 63)
    canvas.paste(_c10, (16, 1), _c10)
except Exception:
    pass
layout["9.44"] = [16, 1, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_03_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-5/11_icon_Nearby.png
try:
    _c11 = get_crop(11, 415, 114)
    canvas.paste(_c11, (48, 465), _c11)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_03_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-5/12_text_FFind_events_in..png
try:
    _c12 = get_crop(12, 1344, 129)
    canvas.paste(_c12, (48, 264), _c12)
except Exception:
    pass
layout["FFind_events_in."] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_03_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-5/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_03_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-5/14_text_Virtual_attendance.png
try:
    _c14 = get_crop(14, 452, 114)
    canvas.paste(_c14, (511, 465), _c14)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_03_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-5/15_text_Browsing_in.png
try:
    _c15 = get_crop(15, 228, 55)
    canvas.paste(_c15, (44, 742), _c15)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_03_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-5/16_text_Los_Angeles.png
try:
    _c16 = get_crop(16, 1440, 138)
    canvas.paste(_c16, (0, 816), _c16)
except Exception:
    pass
layout["Los_Angeles"] = [0, 816, 1440, 954]
