# page_id: page_eventbrite_b2798d8b10cc4118ab8cf6648f8a4077_06
# screenshot: 2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-8.png
# step_index: 6/12
# task: Open Eventbrite. Search Music event in New York. Select the first one. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background & structure (assumes `canvas` and `draw` are provided)
# Canvas: 1440x2960 RGB

# Full background (slightly off-white like the screenshot)
draw.rectangle((0, 0, 1440, 2960), fill=(250, 250, 252))

# Status bar area (top ~72px) - muted grey
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(158, 160, 162))

# Subtle bottom shadow under status bar
draw.rectangle((0, status_h, 1440, status_h + 4), fill=(200, 200, 202))

# Toolbar area below status bar (keeps light background)
toolbar_top = status_h
toolbar_bottom = 148
draw.rectangle((0, toolbar_top, 1440, toolbar_bottom), fill=(250, 250, 252))

# Main title divider (thin muted purple/grey line under the title)
# Title region is around y ≈ 264..393 (detected). Place divider just below that.
divider_y = 396
draw.line((48, divider_y, 1392, divider_y), fill=(140, 128, 150), width=2)

# "Nearby" section card background (rounded rect)
nearby_card = (24, 420, 1416, 600)
draw.rounded_rectangle(nearby_card, radius=18, fill=(247, 249, 252), outline=None)

# Subtle divider below the Nearby card
draw.line((32, 604, 1408, 604), fill=(235, 235, 238), width=1)

# "Browsing in / Online events" section background (rounded rect)
browsing_card = (24, 720, 1416, 940)
draw.rounded_rectangle(browsing_card, radius=20, fill=(255, 255, 255), outline=None)

# Subtle shadow along the top edge of the browsing card to suggest elevation
draw.rectangle((24, 716, 1416, 720), fill=(245, 245, 247))

# Horizontal separators to structure white space further down the page
draw.line((24, 960, 1416, 960), fill=(245, 245, 247), width=1)
draw.line((24, 1320, 1416, 1320), fill=(245, 245, 247), width=1)

# Bottom area remains clean/white; add a very light footer divider near the bottom area
draw.line((24, 2800, 1416, 2800), fill=(245, 245, 247), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_06_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-8/00_icon_9_19.png
try:
    _c0 = get_crop(0, 56, 63)
    canvas.paste(_c0, (180, 1), _c0)
except Exception:
    pass
layout["9:19"] = [180, 1, 236, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_06_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-8/01_icon_9_19.png
try:
    _c1 = get_crop(1, 168, 168)
    canvas.paste(_c1, (0, 72), _c1)
except Exception:
    pass
layout["9:19"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_06_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-8/02_icon_9_19.png
try:
    _c2 = get_crop(2, 51, 63)
    canvas.paste(_c2, (117, 2), _c2)
except Exception:
    pass
layout["9:19"] = [117, 2, 168, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_06_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-8/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 55, 62)
    canvas.paste(_c3, (246, 1), _c3)
except Exception:
    pass
layout["icon_3"] = [246, 1, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_06_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-8/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 53, 63)
    canvas.paste(_c4, (315, 1), _c4)
except Exception:
    pass
layout["icon_4"] = [315, 1, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_06_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-8/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 48, 56)
    canvas.paste(_c5, (1321, 4), _c5)
except Exception:
    pass
layout["icon_5"] = [1321, 4, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_06_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-8/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 86, 61)
    canvas.paste(_c6, (1213, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1213, 1, 1299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_06_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-8/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 103, 109)
    canvas.paste(_c7, (1291, 837), _c7)
except Exception:
    pass
layout["icon_7"] = [1291, 837, 1394, 946]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_06_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 41, 59)
    canvas.paste(_c8, (1272, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1272, 3, 1313, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_06_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-8/09_text_9_19.png
try:
    _c9 = get_crop(9, 94, 45)
    canvas.paste(_c9, (17, 15), _c9)
except Exception:
    pass
layout["9:19"] = [17, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_06_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-8/10_text_Find_events_in..png
try:
    _c10 = get_crop(10, 1344, 129)
    canvas.paste(_c10, (48, 264), _c10)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_06_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-8/11_text_Nearby.png
try:
    _c11 = get_crop(11, 415, 114)
    canvas.paste(_c11, (48, 465), _c11)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_06_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-8/12_text_Current_location.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_06_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-8/13_text_Browsing_in.png
try:
    _c13 = get_crop(13, 228, 55)
    canvas.paste(_c13, (44, 742), _c13)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_06_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-8/14_text_Online_events.png
try:
    _c14 = get_crop(14, 1440, 138)
    canvas.paste(_c14, (0, 816), _c14)
except Exception:
    pass
layout["Online_events"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_06_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-8/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 1440, 138)
    canvas.paste(_c15, (0, 816), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [0, 816, 1440, 954]
