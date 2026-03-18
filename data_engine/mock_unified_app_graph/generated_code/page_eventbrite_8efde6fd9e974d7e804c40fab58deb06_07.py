# page_id: page_eventbrite_8efde6fd9e974d7e804c40fab58deb06_07
# screenshot: 2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9.png
# step_index: 7/8
# task: Open Eventbrite. Search for "Education". Filter only online events. Note how many events are available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and UI structure for the mobile page (canvas: 1440x2960)
# Assumes `canvas` (PIL Image) and `draw` (PIL.ImageDraw.Draw) are available.

# Fill overall background (dominant color: white)
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# Status bar area (top ~72px) - light grey
STATUS_H = 72
draw.rectangle((0, 0, 1440, STATUS_H), fill="#D0D0D0")

# Subtle bottom divider of status/header area
draw.line((0, STATUS_H-1, 1440, STATUS_H-1), fill="#C8C4C9", width=1)

# Header region (below status bar) - keep it light (mostly white)
HEADER_TOP = STATUS_H
HEADER_BOTTOM = 240
draw.rectangle((0, HEADER_TOP, 1440, HEADER_BOTTOM), fill="#FFFFFF")

# Soft underline below the main header/title area (thin divider)
# The "Find events in..." text sits roughly in this area; draw the divider below it
divider_y = 360
draw.line((48, divider_y, 1392, divider_y), fill="#E0DDE1", width=2)

# Slight shadow line under the divider to give subtle separation
draw.line((48, divider_y+2, 1392, divider_y+2), fill="#F6F6F7", width=1)

# Light card area behind the filter chips / options (subtle pale surface)
# Positioned above the "Browsing in" section
options_card_top = 300
options_card_bottom = 560
draw.rounded_rectangle((32, options_card_top, 1408, options_card_bottom),
                       radius=14, fill="#FAFBFD", outline="#F0EDF2", width=1)

# Separator between options card and the browsing section
sep_y = options_card_bottom + 24
draw.line((32, sep_y, 1408, sep_y), fill="#F0EAF0", width=1)

# "Browsing in" header area (no text drawn) - leave white but add a faint label area separator
browsing_top = sep_y + 24
draw.rectangle((0, browsing_top - 8, 1440, browsing_top + 8), fill="#FFFFFF")

# Location selection card (rounded) - background for New York entry
loc_card_top = 740
loc_card_bottom = 920
draw.rounded_rectangle((32, loc_card_top, 1408, loc_card_bottom),
                       radius=18, fill="#FBFAFF", outline="#ECE8EE", width=1)

# Small subtle inner separator under the "Browsing in" label
draw.line((48, browsing_top + 56, 1392, browsing_top + 56), fill="#F3EEF4", width=1)

# Add a faint divider further down the page for content separation
draw.line((32, 1000, 1408, 1000), fill="#F4F2F6", width=1)

# Bottom area left intentionally blank (main content region)
# Add a subtle vertical rhythm line on the left to anchor content blocks
for y in range(1100, 2800, 220):
    draw.line((48, y, 1392, y), fill="#FBF9FB", width=1)

# Final subtle edge shading at the bottom of the location card to lift it visually
draw.line((32, loc_card_bottom, 1408, loc_card_bottom), fill="#ECE8EE", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_07_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_07_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 60, 61)
    canvas.paste(_c1, (310, 4), _c1)
except Exception:
    pass
layout["icon_1"] = [310, 4, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_07_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9/02_icon_7.00.png
try:
    _c2 = get_crop(2, 60, 64)
    canvas.paste(_c2, (180, 2), _c2)
except Exception:
    pass
layout["7.00"] = [180, 2, 240, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_07_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9/03_icon_7.00.png
try:
    _c3 = get_crop(3, 58, 65)
    canvas.paste(_c3, (116, 1), _c3)
except Exception:
    pass
layout["7.00"] = [116, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_07_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9/04_icon_7.00.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["7.00"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_07_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 57)
    canvas.paste(_c5, (248, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_07_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 60, 65)
    canvas.paste(_c6, (1213, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1213, 0, 1273, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_07_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 42, 63)
    canvas.paste(_c7, (1271, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1271, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_07_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 46, 59)
    canvas.paste(_c8, (1322, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1322, 3, 1368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_07_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 51, 66)
    canvas.paste(_c9, (383, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 1, 434, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_07_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9/10_text_7.00.png
try:
    _c10 = get_crop(10, 91, 45)
    canvas.paste(_c10, (20, 15), _c10)
except Exception:
    pass
layout["7.00"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_07_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_07_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_07_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_07_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_07_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_07_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_07_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9/17_text_New_York.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["New_York"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_07_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-9/18_text_New_York.png
try:
    _c18 = get_crop(18, 182, 50)
    canvas.paste(_c18, (44, 905), _c18)
except Exception:
    pass
layout["New_York"] = [44, 905, 226, 955]
