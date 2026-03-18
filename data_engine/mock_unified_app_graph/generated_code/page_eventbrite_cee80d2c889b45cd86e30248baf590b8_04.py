# page_id: page_eventbrite_cee80d2c889b45cd86e30248baf590b8_04
# screenshot: 2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-6.png
# step_index: 4/13
# task: Open Eventbrite. Search Food & Drink party event in New York. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided. This script paints the background and structural UI elements.
# It intentionally avoids drawing any icons or text that will be pasted later.

# Overall page background (slightly warm white to match the screenshot tone)
draw.rectangle((0, 0, 1440, 2960), fill=(250, 251, 253))

# Status bar area (top ~72px) - subtle gray to separate from content area
status_bar_h = 72
draw.rectangle((0, 0, 1440, status_bar_h), fill=(243, 244, 246))

# Top navigation area below the status bar (nav bar / back-arrow area)
nav_top = status_bar_h
nav_bottom = 160
draw.rectangle((0, nav_top, 1440, nav_bottom), fill=(255, 255, 255))

# Subtle bottom hairline of nav area
draw.line((32, nav_bottom - 1, 1408, nav_bottom - 1), fill=(230, 232, 235), width=1)

# "New York" section underline (primary accent blue)
# The New York text bounding box starts at y ~264; place a strong accent line slightly below the top of that box.
newy_top = 264
underline_y = newy_top + 85  # line runs under the large header text
accent_blue = (46, 83, 255)
draw.line((48, underline_y, 1392, underline_y), fill=accent_blue, width=6)

# Light divider just below the accent line to add structure
draw.line((48, underline_y + 12, 1392, underline_y + 12), fill=(235, 238, 246), width=1)

# Option group container area (row that will contain "Nearby" and "Online events")
# Draw a very subtle background band to visually group the option icons and labels.
options_top = underline_y + 28
options_bottom = options_top + 140
draw.rectangle((24, options_top, 1416, options_bottom), fill=(255, 255, 255))

# Soft drop shadow under the options band to separate from the rest of the page
shadow_y1 = options_bottom
shadow_y2 = shadow_y1 + 8
for i in range(6):
    alpha = int(10 - i * 2)  # decreasing alpha for gradient
    if alpha <= 0:
        continue
    shade = (220 + i, 223 + i, 227 + i)
    draw.line((24, shadow_y1 + i, 1416, shadow_y1 + i), fill=shade, width=1)

# Two subtle rounded card outlines to visually group each option (no icons/text drawn)
card_radius = 18
left_card = (36, options_top + 12, 480, options_bottom - 12)
right_card = (492, options_top + 12, 936, options_bottom - 12)
# Very light fill to hint grouping, but largely transparent / white so pasted icons remain prominent
draw.rounded_rectangle(left_card, radius=card_radius, fill=(250, 252, 255), outline=(235, 239, 255), width=1)
draw.rounded_rectangle(right_card, radius=card_radius, fill=(250, 252, 255), outline=(235, 239, 255), width=1)

# Bottom separator to indicate end of the options section
sep_y = options_bottom + 30
draw.line((32, sep_y, 1408, sep_y), fill=(241, 243, 245), width=1)

# Large content area remains white; add a very faint center guide area for "Loading" region (no text)
# We'll draw a faint small dot to hint loading area but keep it subtle so pasted "Loading" text sits on top.
center_x = 1440 // 2
center_y = 1970  # near where the "Loading" text will be pasted
draw.ellipse((center_x - 2, center_y - 2, center_x + 2, center_y + 2), fill=(227, 230, 236))

# Final subtle footer line near bottom to frame the page (very faint)
draw.line((32, 2920, 1408, 2920), fill=(245, 246, 248), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_04_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-6/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 45, 69)
    canvas.paste(_c0, (1156, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1156, 0, 1201, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_04_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-6/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 95, 66)
    canvas.paste(_c1, (1215, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1215, 0, 1310, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_04_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-6/02_icon_9.44.png
try:
    _c2 = get_crop(2, 58, 65)
    canvas.paste(_c2, (180, 1), _c2)
except Exception:
    pass
layout["9.44"] = [180, 1, 238, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_04_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-6/03_icon_9.44.png
try:
    _c3 = get_crop(3, 54, 65)
    canvas.paste(_c3, (115, 1), _c3)
except Exception:
    pass
layout["9.44"] = [115, 1, 169, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_04_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-6/04_icon_9.44.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["9.44"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_04_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-6/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 79, 90)
    canvas.paste(_c5, (1314, 289), _c5)
except Exception:
    pass
layout["icon_5"] = [1314, 289, 1393, 379]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_04_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-6/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 53, 64)
    canvas.paste(_c6, (315, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [315, 1, 368, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_04_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-6/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 45, 60)
    canvas.paste(_c7, (1326, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1326, 3, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_04_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 55, 63)
    canvas.paste(_c8, (246, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [246, 1, 301, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_04_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-6/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 46, 65)
    canvas.paste(_c9, (384, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [384, 0, 430, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_04_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-6/10_text_9.44.png
try:
    _c10 = get_crop(10, 94, 43)
    canvas.paste(_c10, (20, 15), _c10)
except Exception:
    pass
layout["9.44"] = [20, 15, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_04_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-6/11_text_New_York.png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["New_York"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_04_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-6/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_04_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-6/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_04_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-6/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_04_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-6/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_04_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-6/16_text_Loading.png
try:
    _c16 = get_crop(16, 156, 55)
    canvas.paste(_c16, (641, 1970), _c16)
except Exception:
    pass
layout["Loading"] = [641, 1970, 797, 2025]
