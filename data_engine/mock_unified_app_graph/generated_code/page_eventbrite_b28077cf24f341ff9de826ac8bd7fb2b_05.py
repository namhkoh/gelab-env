# page_id: page_eventbrite_b28077cf24f341ff9de826ac8bd7fb2b_05
# screenshot: 2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-7.png
# step_index: 5/16
# task: Open Eventbrite. Explore 'Wellness' events in Washington. Filter to only show free events. Add the first non-promoted event to favorite and follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural elements for the mobile UI page
# Assumes `canvas` (1440x2960) and `draw` are provided.

# Colors
bg_color = "#FFFFFF"            # page background
status_bar_color = "#9a9a9a"    # top status bar (muted gray)
divider_color = "#C8BFD0"       # thin purple-gray divider under header
card_bg_color = "#F6F8FB"       # subtle card background for rows
muted_divider = "#EFEFF1"       # very light divider
online_card_bg = "#FBF7FF"      # pale lavender background for "Online events" area
shadow_color = (0, 0, 0, 25)    # faint shadow (if needed)

w, h = canvas.size

# Fill overall background (canvas may already be white, but fill to ensure consistency)
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar (top area with time/signal icons)
status_bar_height = 88
draw.rectangle([(0, 0), (w, status_bar_height)], fill=status_bar_color)

# A slightly lighter strip below status bar to transition to content area
transition_height = status_bar_height + 8
draw.rectangle([(0, status_bar_height), (w, transition_height)], fill="#EDEBEB")

# Main header area (kept white but provide a very subtle top shadow line)
header_top = transition_height
header_bottom = 200
draw.rectangle([(0, header_top), (w, header_bottom)], fill=bg_color)
# subtle hairline under header
draw.line([(48, header_bottom), (w-48, header_bottom)], fill=divider_color, width=2)

# Thin divider under the "Find events in..." heading (approximate position)
divider_y = 320
draw.line([(48, divider_y), (w-48, divider_y)], fill=divider_color, width=2)

# Nearby row background (rounded card behind the Nearby item)
nearby_top = 360
nearby_bottom = 520
nearby_left = 32
nearby_right = w - 32
draw.rounded_rectangle([(nearby_left, nearby_top), (nearby_right, nearby_bottom)],
                       radius=20, fill=card_bg_color, outline=None)

# Very subtle inner divider within the nearby card (to mimic separation)
draw.line([(nearby_left+24, nearby_top+92), (nearby_right-24, nearby_top+92)],
          fill=muted_divider, width=1)

# "Browsing in" label area (keeps white, but add a faint separator above)
browsing_y = 720
draw.line([(48, browsing_y-24), (w-48, browsing_y-24)], fill=muted_divider, width=1)

# Online events section background (rounded, light lavender block)
online_top = 760
online_bottom = 920
online_left = 28
online_right = w - 28
draw.rounded_rectangle([(online_left, online_top), (online_right, online_bottom)],
                       radius=28, fill=online_card_bg, outline=None)

# Add a faint right-side margin area to hint the selectable check area (no icon drawn)
# This is just a soft circular highlight behind where the check icon will be pasted.
# Keep it very pale so it doesn't duplicate the actual icon content.
check_hint_center = (1291 + 103//2, 836 + 108//2)  # center based on detected check icon area
hint_radius = 56
# Draw a very pale circle background (ensure it's subtle)
draw.ellipse([
    (check_hint_center[0] - hint_radius, check_hint_center[1] - hint_radius),
    (check_hint_center[0] + hint_radius, check_hint_center[1] + hint_radius)],
    fill="#FBF9FB")

# Additional separators to structure the large empty content area
# faint horizontal separators every 420px starting below the online card
sep_y = online_bottom + 60
while sep_y < h - 200:
    draw.line([(48, sep_y), (w-48, sep_y)], fill=muted_divider, width=1)
    sep_y += 420

# Subtle bottom shadow under the header area to add depth
shadow_top = header_bottom - 2
shadow_bottom = header_bottom + 6
draw.rectangle([(0, shadow_top), (w, shadow_bottom)], fill="#F2EFF2")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_05_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-7/00_icon_4.44.png
try:
    _c0 = get_crop(0, 168, 168)
    canvas.paste(_c0, (0, 72), _c0)
except Exception:
    pass
layout["4.44"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_05_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-7/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 61, 61)
    canvas.paste(_c1, (309, 3), _c1)
except Exception:
    pass
layout["icon_1"] = [309, 3, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_05_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-7/02_icon_4.44.png
try:
    _c2 = get_crop(2, 60, 63)
    canvas.paste(_c2, (180, 2), _c2)
except Exception:
    pass
layout["4.44"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_05_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-7/03_icon_4.44.png
try:
    _c3 = get_crop(3, 60, 65)
    canvas.paste(_c3, (113, 1), _c3)
except Exception:
    pass
layout["4.44"] = [113, 1, 173, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_05_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-7/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 49, 56)
    canvas.paste(_c4, (249, 6), _c4)
except Exception:
    pass
layout["icon_4"] = [249, 6, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_05_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-7/05_icon_4.44.png
try:
    _c5 = get_crop(5, 93, 62)
    canvas.paste(_c5, (15, 2), _c5)
except Exception:
    pass
layout["4.44"] = [15, 2, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_05_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 103, 108)
    canvas.paste(_c6, (1291, 836), _c6)
except Exception:
    pass
layout["icon_6"] = [1291, 836, 1394, 944]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_05_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 46, 57)
    canvas.paste(_c7, (1322, 4), _c7)
except Exception:
    pass
layout["icon_7"] = [1322, 4, 1368, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_05_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 69, 61)
    canvas.paste(_c8, (1213, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1213, 1, 1282, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_05_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-7/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 42, 59)
    canvas.paste(_c9, (1272, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [1272, 3, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_05_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-7/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 53, 65)
    canvas.paste(_c10, (382, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [382, 1, 435, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_05_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-7/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_05_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-7/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_05_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-7/13_text_Current_location.png
try:
    _c13 = get_crop(13, 415, 114)
    canvas.paste(_c13, (48, 465), _c13)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_05_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-7/14_text_Browsing_in.png
try:
    _c14 = get_crop(14, 228, 55)
    canvas.paste(_c14, (44, 742), _c14)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_05_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-7/15_text_Online_events.png
try:
    _c15 = get_crop(15, 1440, 138)
    canvas.paste(_c15, (0, 816), _c15)
except Exception:
    pass
layout["Online_events"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_05_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-7/16_text_Virtual_attendance.png
try:
    _c16 = get_crop(16, 1440, 138)
    canvas.paste(_c16, (0, 816), _c16)
except Exception:
    pass
layout["Virtual_attendance"] = [0, 816, 1440, 954]
