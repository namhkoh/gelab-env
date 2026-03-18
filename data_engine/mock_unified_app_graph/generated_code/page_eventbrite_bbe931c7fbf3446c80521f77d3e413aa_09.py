# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_09
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11.png
# step_index: 9/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background & structure drawing for the UI (uses provided `canvas` and `draw`)

# 1) Base background (white)
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# 2) Status bar area at top (~72px) - muted gray
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(160, 160, 160))

# 3) Header/search area under status bar
header_top = status_h
header_bottom = 220
draw.rectangle((0, header_top, 1440, header_bottom), fill=(255, 255, 255))

# blue underline for the search field (prominent accent)
blue_y = status_h + 128
draw.line((48, blue_y, 1392, blue_y), fill=(36, 82, 255), width=6)

# subtle thin divider under header area
draw.line((24, header_bottom, 1416, header_bottom), fill=(240, 240, 242), width=1)

# 4) Option cards area (Nearby / Online events) - light pill backgrounds
opts_top = 420
opts_bottom = 540
pill_fill = (235, 246, 255)  # very light blue
pill_radius = 24

# left pill background
left_pill = (48, opts_top, 463, opts_bottom)
draw.rounded_rectangle(left_pill, radius=pill_radius, fill=pill_fill)

# right pill background
right_pill = (511, opts_top, 926, opts_bottom)
draw.rounded_rectangle(right_pill, radius=pill_radius, fill=pill_fill)

# subtle inner circular highlights behind icon areas (light circle)
circle_fill = (220, 238, 255)
draw.ellipse((68, opts_top + 8, 68 + 96, opts_top + 8 + 96), fill=circle_fill)
draw.ellipse((531, opts_top + 8, 531 + 96, opts_top + 8 + 96), fill=circle_fill)

# faint separator under the option cards area
draw.line((24, opts_bottom + 20, 1416, opts_bottom + 20), fill=(240, 240, 242), width=1)

# 5) "Found locations" header separator (subtle)
found_header_y = 740
draw.line((24, found_header_y + 56, 1416, found_header_y + 56), fill=(245, 245, 247), width=1)

# 6) List background bands and separators for found locations
# Rows observed at tops: 840, 1020, 1200, 1380, 1560, 1740, 1920, 2100, 2280, 2460
row_tops = list(range(840, 2461, 180))
row_height = 132

band_fill = (250, 250, 252)  # very subtle off-white band for alternating rows
sep_color = (240, 240, 242)

for i, y in enumerate(row_tops):
    # alternate subtle band for even rows
    if i % 2 == 0:
        draw.rectangle((0, y, 1440, y + row_height), fill=band_fill)
    # thin horizontal separator line under each row
    draw.line((48, y + row_height, 1392, y + row_height), fill=sep_color, width=1)

# 7) Large whitespace area is intentionally left white for the rest of the content

# 8) Bottom subtle footer divider (near the end of the list)
draw.line((24, 2800, 1416, 2800), fill=(245, 245, 247), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 46, 68)
    canvas.paste(_c0, (1155, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1155, 0, 1201, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 97, 65)
    canvas.paste(_c1, (1215, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1215, 0, 1312, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/02_icon_9.12.png
try:
    _c2 = get_crop(2, 51, 64)
    canvas.paste(_c2, (117, 1), _c2)
except Exception:
    pass
layout["9.12"] = [117, 1, 168, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/03_icon_9.12.png
try:
    _c3 = get_crop(3, 58, 62)
    canvas.paste(_c3, (179, 1), _c3)
except Exception:
    pass
layout["9.12"] = [179, 1, 237, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/04_icon_9.12.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["9.12"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 62)
    canvas.paste(_c5, (1320, 1), _c5)
except Exception:
    pass
layout["icon_5"] = [1320, 1, 1370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 85, 95)
    canvas.paste(_c6, (1310, 286), _c6)
except Exception:
    pass
layout["icon_6"] = [1310, 286, 1395, 381]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/07_icon_San_Francisco.png
try:
    _c7 = get_crop(7, 1440, 132)
    canvas.paste(_c7, (0, 840), _c7)
except Exception:
    pass
layout["San_Francisco"] = [0, 840, 1440, 972]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/08_icon_District_of_Columbia.png
try:
    _c8 = get_crop(8, 1440, 132)
    canvas.paste(_c8, (0, 1740), _c8)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1740, 1440, 1872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 53, 63)
    canvas.paste(_c9, (315, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [315, 1, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/10_icon_Chicago.png
try:
    _c10 = get_crop(10, 1440, 132)
    canvas.paste(_c10, (0, 1380), _c10)
except Exception:
    pass
layout["Chicago"] = [0, 1380, 1440, 1512]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 55, 61)
    canvas.paste(_c11, (247, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [247, 2, 302, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/12_icon_Los_Angeles.png
try:
    _c12 = get_crop(12, 1440, 132)
    canvas.paste(_c12, (0, 1020), _c12)
except Exception:
    pass
layout["Los_Angeles"] = [0, 1020, 1440, 1152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/13_icon_United_Kingdom.png
try:
    _c13 = get_crop(13, 1440, 132)
    canvas.paste(_c13, (0, 2100), _c13)
except Exception:
    pass
layout["United_Kingdom"] = [0, 2100, 1440, 2232]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/14_icon_Miami.png
try:
    _c14 = get_crop(14, 1440, 132)
    canvas.paste(_c14, (0, 1200), _c14)
except Exception:
    pass
layout["Miami"] = [0, 1200, 1440, 1332]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/15_icon_District_of_Columbia.png
try:
    _c15 = get_crop(15, 1440, 132)
    canvas.paste(_c15, (0, 1560), _c15)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/16_icon_Philadelphia.png
try:
    _c16 = get_crop(16, 1440, 132)
    canvas.paste(_c16, (0, 1920), _c16)
except Exception:
    pass
layout["Philadelphia"] = [0, 1920, 1440, 2052]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 50, 63)
    canvas.paste(_c17, (382, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [382, 0, 432, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/18_icon_Nearby.png
try:
    _c18 = get_crop(18, 415, 114)
    canvas.paste(_c18, (48, 465), _c18)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/19_text_9.12.png
try:
    _c19 = get_crop(19, 91, 43)
    canvas.paste(_c19, (20, 17), _c19)
except Exception:
    pass
layout["9.12"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/20_text_Los_Angeles.png
try:
    _c20 = get_crop(20, 1344, 129)
    canvas.paste(_c20, (48, 264), _c20)
except Exception:
    pass
layout["Los_Angeles"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/21_text_Online_events.png
try:
    _c21 = get_crop(21, 452, 114)
    canvas.paste(_c21, (511, 465), _c21)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/22_text_Virtual_attendance.png
try:
    _c22 = get_crop(22, 452, 114)
    canvas.paste(_c22, (511, 465), _c22)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/23_text_Found_locations.png
try:
    _c23 = get_crop(23, 311, 50)
    canvas.paste(_c23, (44, 740), _c23)
except Exception:
    pass
layout["Found_locations"] = [44, 740, 355, 790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/24_text_New_York.png
try:
    _c24 = get_crop(24, 212, 55)
    canvas.paste(_c24, (44, 2288), _c24)
except Exception:
    pass
layout["New_York"] = [44, 2288, 256, 2343]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/25_text_New_York.png
try:
    _c25 = get_crop(25, 154, 38)
    canvas.paste(_c25, (47, 2353), _c25)
except Exception:
    pass
layout["New_York"] = [47, 2353, 201, 2391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/26_text_Atlanta.png
try:
    _c26 = get_crop(26, 163, 52)
    canvas.paste(_c26, (44, 2468), _c26)
except Exception:
    pass
layout["Atlanta"] = [44, 2468, 207, 2520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/27_text_Georgia.png
try:
    _c27 = get_crop(27, 133, 43)
    canvas.paste(_c27, (45, 2533), _c27)
except Exception:
    pass
layout["Georgia"] = [45, 2533, 178, 2576]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/28_clickable_New_York.png
try:
    _c28 = get_crop(28, 1440, 132)
    canvas.paste(_c28, (0, 2280), _c28)
except Exception:
    pass
layout["New_York"] = [0, 2280, 1440, 2412]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_09_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-11/29_clickable_Atlanta.png
try:
    _c29 = get_crop(29, 1440, 132)
    canvas.paste(_c29, (0, 2460), _c29)
except Exception:
    pass
layout["Atlanta"] = [0, 2460, 1440, 2592]
