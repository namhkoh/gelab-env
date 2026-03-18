# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_06
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8.png
# step_index: 6/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the mobile page
# Available variables:
# - canvas: PIL Image 1440x2960 RGB (white)
# - draw: PIL ImageDraw object
# - font_sm, font_md, font_lg, font_xl

# Fill overall background (dominant color = white)
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# Top status bar area (~72-88px tall)
status_bar_height = 88
draw.rectangle((0, 0, 1440, status_bar_height), fill="#F2F2F2")

# Hairline under status bar
draw.line((0, status_bar_height, 1440, status_bar_height), fill="#E0E0E0", width=1)

# Main header divider under the "Find events in..." heading area
# The detected heading text block ends around y = 264 + 129 = 393, draw a subtle divider there
header_div_y = 393
draw.line((48, header_div_y, 1392, header_div_y), fill="#C8C0C8", width=2)

# A lighter highlight directly above the divider to mimic subtle depth
draw.line((48, header_div_y-2, 1392, header_div_y-2), fill="#EFEFF1", width=1)

# Section card / selection area for "Browsing in" and the selected city
# Position chosen to sit under the header and options, leaving room for pasted icons/text
card_left = 32
card_top = 720
card_right = 1408
card_bottom = 980
card_radius = 20

# Card background (white to blend with page) with a thin subtle border/shadow to separate areas
draw.rounded_rectangle(
    (card_left, card_top, card_right, card_bottom),
    radius=card_radius,
    fill="#FFFFFF",
    outline="#F0EDF2",
    width=1
)

# Subtle inner horizontal separator within the card (to visually separate label and details)
sep_y = card_top + 48
draw.line((card_left + 20, sep_y, card_right - 20, sep_y), fill="#F3F2F5", width=1)

# Additional faint separators for visual grouping below the header options region
# (These are purely structural background lines; actual icons/text will be pasted on top.)
opt_region_top = 360
opt_region_bottom = 560
# light divider above options
draw.line((48, opt_region_top, 1392, opt_region_top), fill="#F4F3F6", width=1)
# light divider below options
draw.line((48, opt_region_bottom, 1392, opt_region_bottom), fill="#F4F3F6", width=1)

# Final subtle footer divider near top of content area to create depth
draw.line((48, opt_region_bottom + 160, 1392, opt_region_bottom + 160), fill="#F6F5F7", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_06_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_06_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8/01_icon_9.12.png
try:
    _c1 = get_crop(1, 58, 64)
    canvas.paste(_c1, (180, 1), _c1)
except Exception:
    pass
layout["9.12"] = [180, 1, 238, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_06_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8/02_icon_9.12.png
try:
    _c2 = get_crop(2, 51, 63)
    canvas.paste(_c2, (117, 2), _c2)
except Exception:
    pass
layout["9.12"] = [117, 2, 168, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_06_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8/03_icon_9.12.png
try:
    _c3 = get_crop(3, 168, 168)
    canvas.paste(_c3, (0, 72), _c3)
except Exception:
    pass
layout["9.12"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_06_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 53, 62)
    canvas.paste(_c4, (315, 2), _c4)
except Exception:
    pass
layout["icon_4"] = [315, 2, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_06_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 54, 62)
    canvas.paste(_c5, (247, 1), _c5)
except Exception:
    pass
layout["icon_5"] = [247, 1, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_06_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 61, 63)
    canvas.paste(_c6, (1213, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1213, 1, 1274, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_06_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 46, 58)
    canvas.paste(_c7, (1322, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1322, 3, 1368, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_06_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 51, 62)
    canvas.paste(_c8, (1262, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1262, 1, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_06_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 48, 64)
    canvas.paste(_c9, (383, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 0, 431, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_06_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8/10_text_9.12.png
try:
    _c10 = get_crop(10, 91, 43)
    canvas.paste(_c10, (20, 17), _c10)
except Exception:
    pass
layout["9.12"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_06_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_06_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_06_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_06_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_06_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_06_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_06_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8/17_text_New_York.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["New_York"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_06_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-8/18_text_New_York.png
try:
    _c18 = get_crop(18, 182, 50)
    canvas.paste(_c18, (44, 905), _c18)
except Exception:
    pass
layout["New_York"] = [44, 905, 226, 955]
