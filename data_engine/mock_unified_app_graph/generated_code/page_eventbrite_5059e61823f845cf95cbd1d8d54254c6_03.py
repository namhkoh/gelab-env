# page_id: page_eventbrite_5059e61823f845cf95cbd1d8d54254c6_03
# screenshot: 2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5.png
# step_index: 3/19
# task: Open Eventbrite. Look for 'Education' in Los Angeles happening on May 4. Filter to show only free events. How many events are posted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the UI page using the provided
# canvas (1440x2960) and draw (ImageDraw) objects.

# Colors
bg_fill = (250, 250, 252)            # very light off-white background
status_bar_color = (120, 120, 120)   # top status bar grey
status_bar_bottom = (105, 105, 105)  # slightly darker bottom border
search_underline_color = (61, 90, 255)  # vivid blue/purple underline
search_outline = (225, 225, 230)     # subtle outline for search area
chip_bg = (235, 244, 255)            # pale blue for chip backgrounds
divider_color = (235, 235, 240)      # light divider lines
card_bg = (255, 255, 255)            # white card background
card_outline = (235, 235, 240)       # card outline

w, h = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg_fill)

# Status bar area (top)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)
# subtle bottom border of status bar
draw.line([(0, status_h), (w, status_h)], fill=status_bar_bottom, width=2)

# Header / search area background (subtle, behind text; do NOT draw any text)
search_x1 = 48
search_x2 = w - 48
search_y1 = 120
search_y2 = 320
draw.rounded_rectangle(
    [(search_x1, search_y1), (search_x2, search_y2)],
    radius=8,
    fill=card_bg,
    outline=search_outline,
    width=1
)

# Prominent underline below the search field
underline_y = 286
draw.line([(search_x1, underline_y), (search_x2, underline_y)], fill=search_underline_color, width=4)

# Chip / option background pills (two group areas beneath search)
chips_top = 420
chips_bottom = 540
# Left chip group background
draw.rounded_rectangle(
    [(48, chips_top), (380, chips_bottom)],
    radius=28,
    fill=chip_bg,
    outline=None
)
# Right chip group background (wider to accommodate "Online events" area)
draw.rounded_rectangle(
    [(416, chips_top), (920, chips_bottom)],
    radius=28,
    fill=chip_bg,
    outline=None
)

# Divider line below chips
divider_y = chips_bottom + 20
draw.line([(48, divider_y), (w - 48, divider_y)], fill=divider_color, width=1)

# "Browsing in" / location card background (rounded card behind the location group)
card_x1 = 36
card_x2 = w - 36
card_y1 = 720
card_y2 = 1020
# subtle card outline and fill
draw.rounded_rectangle(
    [(card_x1, card_y1), (card_x2, card_y2)],
    radius=12,
    fill=card_bg,
    outline=card_outline,
    width=1
)

# Small separator between the browsing title area and the list area (thin line)
sep_y = card_y1 + 110
draw.line([(card_x1 + 20, sep_y), (card_x2 - 20, sep_y)], fill=divider_color, width=1)

# A faint vertical guide line on the right edge of the card area (visual structure, not an icon)
draw.line([(card_x2 - 72, card_y1 + 24), (card_x2 - 72, card_y2 - 24)], fill=divider_color, width=1)

# Large lower content area background (subtle to separate from top)
content_top = card_y2 + 40
draw.rectangle([(0, content_top), (w, h)], fill=bg_fill)

# Final subtle horizontal rule near the very top under the status/search cluster
draw.line([(24, search_y1 - 24), (w - 24, search_y1 - 24)], fill=divider_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_03_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_03_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5/01_icon_7.34.png
try:
    _c1 = get_crop(1, 58, 63)
    canvas.paste(_c1, (114, 2), _c1)
except Exception:
    pass
layout["7.34"] = [114, 2, 172, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_03_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5/02_icon_7.34.png
try:
    _c2 = get_crop(2, 60, 63)
    canvas.paste(_c2, (180, 2), _c2)
except Exception:
    pass
layout["7.34"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_03_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 60, 62)
    canvas.paste(_c3, (310, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [310, 3, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_03_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5/04_icon_7.34.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["7.34"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_03_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 58)
    canvas.paste(_c5, (248, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 6, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_03_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5/06_icon_7.34.png
try:
    _c6 = get_crop(6, 93, 63)
    canvas.paste(_c6, (15, 1), _c6)
except Exception:
    pass
layout["7.34"] = [15, 1, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_03_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 61, 64)
    canvas.paste(_c7, (1213, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1213, 1, 1274, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_03_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 46, 59)
    canvas.paste(_c8, (1322, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1322, 3, 1368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_03_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 45, 63)
    canvas.paste(_c9, (1268, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [1268, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_03_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 52, 65)
    canvas.paste(_c10, (382, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [382, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_03_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5/11_text_FFind_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["FFind_events_in."] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_03_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_03_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_03_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_03_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_03_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_03_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5/17_text_New_York.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["New_York"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_03_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-5/18_text_New_York.png
try:
    _c18 = get_crop(18, 182, 50)
    canvas.paste(_c18, (44, 905), _c18)
except Exception:
    pass
layout["New_York"] = [44, 905, 226, 955]
