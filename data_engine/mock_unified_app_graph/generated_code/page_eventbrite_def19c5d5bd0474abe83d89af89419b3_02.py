# page_id: page_eventbrite_def19c5d5bd0474abe83d89af89419b3_02
# screenshot: 2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-4.png
# step_index: 2/8
# task: Open Eventbrite. Set the city to Los Angeles. Select the second recommendation on the home tab. Follow the organizer and look for the time and date of the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural elements for mobile UI (1440x2960)
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Ensure overall background is solid white (dominant color)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar area at top (approx ~80px)
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill=(200, 200, 200))  # neutral light gray status bar

# Toolbar area directly under status bar (white background, with subtle bottom divider)
toolbar_h = 120
toolbar_y0 = status_h
draw.rectangle([(0, toolbar_y0), (1440, toolbar_y0 + toolbar_h)], fill=(255, 255, 255))
# toolbar bottom divider (soft purple/gray)
draw.line([(48, toolbar_y0 + toolbar_h - 1), (1392, toolbar_y0 + toolbar_h - 1)], fill=(225, 215, 235), width=2)

# Thin faint horizontal guide under the "Find events in..." heading area
# Use the detected heading box to place the divider just below it:
heading_box_y = 264
heading_box_h = 129
divider_y = heading_box_y + heading_box_h
draw.line([(48, divider_y), (1392, divider_y)], fill=(210, 200, 220), width=3)

# "Nearby" row background - subtle rounded card behind the row (do NOT draw icons/text)
near_x0, near_y0 = 32, 430
near_x1, near_y1 = 1440 - 32, near_y0 + 120
draw.rounded_rectangle([(near_x0, near_y0), (near_x1, near_y1)],
                       radius=14, fill=(250, 252, 255), outline=(230, 235, 245), width=1)

# Subtle separator beneath the Nearby card
sep_y = near_y1 + 26
draw.line([(48, sep_y), (1392, sep_y)], fill=(245, 243, 247), width=1)

# "Browsing in" / "Online events" section background - a gentle pale purple card to indicate selection area
browsing_y0 = 720
browsing_h = 220
browsing_x0, browsing_x1 = 32, 1440 - 32
draw.rounded_rectangle([(browsing_x0, browsing_y0), (browsing_x1, browsing_y0 + browsing_h)],
                       radius=16, fill=(251, 249, 255), outline=None)

# Subtle bottom divider under browsing section
draw.line([(48, browsing_y0 + browsing_h + 6), (1392, browsing_y0 + browsing_h + 6)], fill=(236, 232, 242), width=1)

# Large content area remains blank/white for pasted elements; add a very faint vertical margin guide on left/right
draw.line([(32, browsing_y0 + browsing_h + 60), (32, 2960)], fill=(248, 247, 250), width=1)
draw.line([(1440 - 32, browsing_y0 + browsing_h + 60), (1440 - 32, 2960)], fill=(248, 247, 250), width=1)

# Subtle overall bottom shadow to anchor the page (very faint)
draw.rectangle([(0, 2920), (1440, 2960)], fill=(255, 255, 255, 0))
for i in range(6):
    alpha = int(6 - i)
    y = 2920 + i
    draw.line([(0, y), (1440, y)], fill=(245 - i, 244 - i, 246 - i), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_02_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-4/00_icon_5.34.png
try:
    _c0 = get_crop(0, 168, 168)
    canvas.paste(_c0, (0, 72), _c0)
except Exception:
    pass
layout["5.34"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_02_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-4/01_icon_5.34.png
try:
    _c1 = get_crop(1, 60, 62)
    canvas.paste(_c1, (180, 2), _c1)
except Exception:
    pass
layout["5.34"] = [180, 2, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_02_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-4/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 61, 61)
    canvas.paste(_c2, (309, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [309, 3, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_02_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-4/03_icon_5.34.png
try:
    _c3 = get_crop(3, 59, 65)
    canvas.paste(_c3, (114, 1), _c3)
except Exception:
    pass
layout["5.34"] = [114, 1, 173, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_02_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-4/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 49, 56)
    canvas.paste(_c4, (249, 6), _c4)
except Exception:
    pass
layout["icon_4"] = [249, 6, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_02_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-4/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 47, 57)
    canvas.paste(_c5, (1322, 4), _c5)
except Exception:
    pass
layout["icon_5"] = [1322, 4, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_02_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-4/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 103, 108)
    canvas.paste(_c6, (1291, 836), _c6)
except Exception:
    pass
layout["icon_6"] = [1291, 836, 1394, 944]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_02_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-4/07_icon_5.34.png
try:
    _c7 = get_crop(7, 94, 62)
    canvas.paste(_c7, (14, 2), _c7)
except Exception:
    pass
layout["5.34"] = [14, 2, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_02_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 71, 61)
    canvas.paste(_c8, (1213, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1213, 1, 1284, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_02_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-4/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 42, 59)
    canvas.paste(_c9, (1272, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [1272, 3, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_02_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-4/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 53, 65)
    canvas.paste(_c10, (382, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [382, 1, 435, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_02_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-4/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_02_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-4/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_02_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-4/13_text_Current_location.png
try:
    _c13 = get_crop(13, 415, 114)
    canvas.paste(_c13, (48, 465), _c13)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_02_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-4/14_text_Browsing_in.png
try:
    _c14 = get_crop(14, 228, 55)
    canvas.paste(_c14, (44, 742), _c14)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_02_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-4/15_text_Online_events.png
try:
    _c15 = get_crop(15, 1440, 138)
    canvas.paste(_c15, (0, 816), _c15)
except Exception:
    pass
layout["Online_events"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_02_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-4/16_text_Virtual_attendance.png
try:
    _c16 = get_crop(16, 1440, 138)
    canvas.paste(_c16, (0, 816), _c16)
except Exception:
    pass
layout["Virtual_attendance"] = [0, 816, 1440, 954]
