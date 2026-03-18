# page_id: page_eventbrite_def19c5d5bd0474abe83d89af89419b3_03
# screenshot: 2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-5.png
# step_index: 3/8
# task: Open Eventbrite. Set the city to Los Angeles. Select the second recommendation on the home tab. Follow the organizer and look for the time and date of the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar area (top)
STATUS_H = 72
draw.rectangle([(0, 0), (1440, STATUS_H)], fill=(189, 189, 189))

# Header / toolbar area (below status bar)
TOOLBAR_H = 160
draw.rectangle([(0, STATUS_H), (1440, TOOLBAR_H)], fill=(255, 255, 255))
# subtle divider under toolbar
draw.line([(24, TOOLBAR_H), (1416, TOOLBAR_H)], fill=(230, 230, 230), width=1)

# Search underline (accent line under the search input area)
# Matches the wide indigo underline across the page
UNDERLINE_X0 = 48
UNDERLINE_X1 = 1392
UNDERLINE_Y = 360
draw.line([(UNDERLINE_X0, UNDERLINE_Y), (UNDERLINE_X1, UNDERLINE_Y)], fill=(63, 81, 181), width=6)

# "Nearby" row background (rounded card behind the row)
NEARBY_X0 = 40
NEARBY_Y0 = 420
NEARBY_X1 = 1400
NEARBY_Y1 = 560
draw.rounded_rectangle([(NEARBY_X0, NEARBY_Y0), (NEARBY_X1, NEARBY_Y1)], radius=18, fill=(242, 246, 255))
# subtle bottom edge shadow for the nearby card
draw.line([(NEARBY_X0 + 6, NEARBY_Y1), (NEARBY_X1 - 6, NEARBY_Y1)], fill=(230, 232, 240), width=1)

# Divider above "Browsing in" section
BROWSE_DIV_Y = 720
draw.line([(48, BROWSE_DIV_Y), (1392, BROWSE_DIV_Y)], fill=(245, 245, 248), width=1)

# Section header background area for the "Online events" block (subtle tinted area)
ONLINE_AREA_X0 = 0
ONLINE_AREA_Y0 = 800
ONLINE_AREA_X1 = 1440
ONLINE_AREA_Y1 = 960
draw.rectangle([(ONLINE_AREA_X0, ONLINE_AREA_Y0), (ONLINE_AREA_X1, ONLINE_AREA_Y1)], fill=(255, 255, 255))
# light left accent to separate content visually
draw.line([(48, ONLINE_AREA_Y0 + 8), (48, ONLINE_AREA_Y1 - 8)], fill=(245, 245, 248), width=8)

# Long faint right-side separator to suggest content column (doesn't overlap detected icons/text)
draw.line([(1392, TOOLBAR_H + 8), (1392, 2960 - 8)], fill=(250, 250, 252), width=8)

# Small subtle horizontal separators to structure the large whitespace
for y in (980, 1260, 1640, 2020):
    draw.line([(48, y), (1392, y)], fill=(250, 250, 252), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_03_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-5/00_icon_5.34.png
try:
    _c0 = get_crop(0, 168, 168)
    canvas.paste(_c0, (0, 72), _c0)
except Exception:
    pass
layout["5.34"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_03_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-5/01_icon_5.34.png
try:
    _c1 = get_crop(1, 60, 64)
    canvas.paste(_c1, (180, 1), _c1)
except Exception:
    pass
layout["5.34"] = [180, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_03_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-5/02_icon_5.34.png
try:
    _c2 = get_crop(2, 60, 65)
    canvas.paste(_c2, (114, 1), _c2)
except Exception:
    pass
layout["5.34"] = [114, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_03_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 62, 61)
    canvas.paste(_c3, (309, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [309, 3, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_03_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-5/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 51, 58)
    canvas.paste(_c4, (247, 5), _c4)
except Exception:
    pass
layout["icon_4"] = [247, 5, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_03_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 103, 108)
    canvas.paste(_c5, (1291, 836), _c5)
except Exception:
    pass
layout["icon_5"] = [1291, 836, 1394, 944]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_03_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 48, 57)
    canvas.paste(_c6, (1321, 4), _c6)
except Exception:
    pass
layout["icon_6"] = [1321, 4, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_03_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 89, 61)
    canvas.paste(_c7, (1212, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1212, 1, 1301, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_03_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-5/08_icon_5.34.png
try:
    _c8 = get_crop(8, 94, 62)
    canvas.paste(_c8, (14, 2), _c8)
except Exception:
    pass
layout["5.34"] = [14, 2, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_03_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-5/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 42, 59)
    canvas.paste(_c9, (1272, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [1272, 3, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_03_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-5/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 53, 65)
    canvas.paste(_c10, (382, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [382, 1, 435, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_03_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-5/11_text_FFind_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["FFind_events_in."] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_03_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-5/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_03_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-5/13_text_Current_location.png
try:
    _c13 = get_crop(13, 415, 114)
    canvas.paste(_c13, (48, 465), _c13)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_03_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-5/14_text_Browsing_in.png
try:
    _c14 = get_crop(14, 228, 55)
    canvas.paste(_c14, (44, 742), _c14)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_03_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-5/15_text_Online_events.png
try:
    _c15 = get_crop(15, 1440, 138)
    canvas.paste(_c15, (0, 816), _c15)
except Exception:
    pass
layout["Online_events"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/def19c5d5bd0474abe83d89af89419b3/step_03_2024_4_24_17_33_def19c5d5bd0474abe83d89af89419b3-5/16_text_Virtual_attendance.png
try:
    _c16 = get_crop(16, 1440, 138)
    canvas.paste(_c16, (0, 816), _c16)
except Exception:
    pass
layout["Virtual_attendance"] = [0, 816, 1440, 954]
