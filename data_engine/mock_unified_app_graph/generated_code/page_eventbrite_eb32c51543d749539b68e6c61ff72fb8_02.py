# page_id: page_eventbrite_eb32c51543d749539b68e6c61ff72fb8_02
# screenshot: 2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-4.png
# step_index: 2/19
# task: Open Eventbrite. Set the city to San Francisco. Filter for events occurring between May 1st and May 15th under the category 'Music'. Select the first event and check the pricing options available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background & structural UI elements for a 1440x2960 canvas
w, h = canvas.size

# Colors
status_bar_color = (210, 210, 210)        # light gray for status bar
divider_color = (171, 164, 180)           # soft muted purple/gray divider
muted_bg = (245, 249, 255)                # very light bluish card background
card_border = (230, 230, 235)             # subtle card border
section_bg = (255, 255, 255)              # white (explicit)
page_bg = (255, 255, 255)                 # main page background

# Clear canvas to page background (in case not already)
draw.rectangle([(0, 0), (w, h)], fill=page_bg)

# Status bar (top area)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# Top toolbar area (slightly below status bar) kept white but draw subtle bottom divider
toolbar_top = status_h
toolbar_bottom = status_h + 120
# keep it white (no fill needed, canvas already white), add a faint bottom divider
divider_y = toolbar_bottom
draw.line([(48, divider_y), (w - 48, divider_y)], fill=divider_color, width=2)

# Category / chips background card (behind "Nearby" and "Online events")
chips_card_top = 420
chips_card_bottom = 560
chips_card_left = 32
chips_card_right = w - 32
chips_radius = 20
draw.rounded_rectangle(
    [(chips_card_left, chips_card_top), (chips_card_right, chips_card_bottom)],
    radius=chips_radius,
    fill=muted_bg,
    outline=card_border,
    width=1
)

# Subtle horizontal separator below chips area
sep_y = chips_card_bottom + 40
draw.line([(48, sep_y), (w - 48, sep_y)], fill=(240, 238, 243), width=1)

# Browsing location card area (behind "Browsing in" and "Los Angeles")
loc_card_top = 760
loc_card_bottom = 920
loc_card_left = 24
loc_card_right = w - 24
loc_radius = 18
# Draw a white card with a very light border/shadow effect
# subtle shadow: draw a slightly darker thin rectangle below to imply elevation
shadow_offset = 6
draw.rounded_rectangle(
    [(loc_card_left, loc_card_top), (loc_card_right, loc_card_bottom)],
    radius=loc_radius,
    fill=section_bg,
    outline=card_border,
    width=1
)
# Draw faint shadow line beneath card
draw.rectangle(
    [(loc_card_left + 6, loc_card_bottom + 2),
     (loc_card_right - 6, loc_card_bottom + 2 + shadow_offset)],
    fill=(250, 250, 252)
)

# Large empty content area remains white (events list) - add a very faint vertical guideline on left for layout alignment
draw.line([(48, loc_card_bottom + 120), (48, h - 120)], fill=(248, 248, 250), width=1)

# Final subtle bottom divider near top search area (to match subtle UI divider under "Find events in…")
top_search_divider_y = 300
draw.line([(48, top_search_divider_y), (w - 48, top_search_divider_y)], fill=(236, 233, 240), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_02_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-4/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_02_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-4/01_icon_7.47.png
try:
    _c1 = get_crop(1, 59, 65)
    canvas.paste(_c1, (114, 1), _c1)
except Exception:
    pass
layout["7.47"] = [114, 1, 173, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_02_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-4/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 60, 61)
    canvas.paste(_c2, (310, 4), _c2)
except Exception:
    pass
layout["icon_2"] = [310, 4, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_02_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-4/03_icon_7.47.png
try:
    _c3 = get_crop(3, 60, 63)
    canvas.paste(_c3, (180, 2), _c3)
except Exception:
    pass
layout["7.47"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_02_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-4/04_icon_7.47.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["7.47"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_02_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-4/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 49, 57)
    canvas.paste(_c5, (249, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_02_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-4/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 60, 64)
    canvas.paste(_c6, (1213, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1213, 1, 1273, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_02_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-4/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 46, 59)
    canvas.paste(_c7, (1322, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1322, 3, 1368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_02_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 43, 63)
    canvas.paste(_c8, (1270, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1270, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_02_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-4/09_icon_7.47.png
try:
    _c9 = get_crop(9, 91, 64)
    canvas.paste(_c9, (16, 1), _c9)
except Exception:
    pass
layout["7.47"] = [16, 1, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_02_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-4/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 51, 65)
    canvas.paste(_c10, (383, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [383, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_02_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-4/11_icon_Nearby.png
try:
    _c11 = get_crop(11, 415, 114)
    canvas.paste(_c11, (48, 465), _c11)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_02_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-4/12_text_Find_events_in..png
try:
    _c12 = get_crop(12, 1344, 129)
    canvas.paste(_c12, (48, 264), _c12)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_02_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-4/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_02_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-4/14_text_Virtual_attendance.png
try:
    _c14 = get_crop(14, 452, 114)
    canvas.paste(_c14, (511, 465), _c14)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_02_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-4/15_text_Browsing_in.png
try:
    _c15 = get_crop(15, 228, 55)
    canvas.paste(_c15, (44, 742), _c15)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_02_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-4/16_text_Los_Angeles.png
try:
    _c16 = get_crop(16, 1440, 138)
    canvas.paste(_c16, (0, 816), _c16)
except Exception:
    pass
layout["Los_Angeles"] = [0, 816, 1440, 954]
