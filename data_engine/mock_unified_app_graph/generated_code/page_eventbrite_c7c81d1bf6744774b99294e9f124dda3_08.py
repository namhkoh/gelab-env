# page_id: page_eventbrite_c7c81d1bf6744774b99294e9f124dda3_08
# screenshot: 2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10.png
# step_index: 8/10
# task: Open Eventbrite. Search for "Fitness". Select the events in the location "Chicago". What is the price of the first event in listing?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for the provided canvas (1440x2960)
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Base background (dominant color: white)
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar area (top) - light gray background for status icons area
status_bar_height = 72
draw.rectangle((0, 0, 1440, status_bar_height), fill=(228, 228, 228))

# Top toolbar / header area background (below status bar)
header_top = status_bar_height
header_bottom = 420
draw.rectangle((0, header_top, 1440, header_bottom), fill=(255, 255, 255))

# Thin subtle bottom divider under header
draw.line((24, header_bottom, 1440 - 24, header_bottom), fill=(220, 220, 220), width=2)

# Search underline (accent blue) - place a prominent blue line where search input underline is
underline_y = header_bottom + 0  # just at header bottom
draw.line((48, underline_y, 1440 - 48, underline_y), fill=(20, 90, 240), width=4)

# Two rounded background "chips" (containers) behind the Nearby / Online events area
# Left chip background (behind detected elements around y ~465-579)
left_chip_box = (28, 435, 480, 595)
draw.rounded_rectangle(left_chip_box, radius=28, fill=(235, 246, 255), outline=None)

# Right chip background
right_chip_box = (488, 435, 1004, 595)
draw.rounded_rectangle(right_chip_box, radius=28, fill=(235, 246, 255), outline=None)

# Subtle divider below chips area
chips_bottom = 620
draw.line((24, chips_bottom, 1440 - 24, chips_bottom), fill=(240, 240, 240), width=1)

# "Found locations" section divider (header area) - subtle spacing
found_header_y = 740
draw.line((24, found_header_y - 20, 1440 - 24, found_header_y - 20), fill=(245, 245, 245), width=1)

# List rows backgrounds and separators
# Detected full-width rows (y tops) from detection: 840, 1020, 1200, 1380, 1560, 1740, 1920, 2100, 2280, 2460
row_tops = [840, 1020, 1200, 1380, 1560, 1740, 1920, 2100, 2280, 2460]
row_height = 132  # from detections
separator_color = (240, 240, 240)
stripe_color = (250, 250, 251)

for i, top in enumerate(row_tops):
    bottom = top + row_height
    # Alternate subtle stripe backgrounds for rows (do not draw strong fills to avoid masking pasted text)
    if i % 2 == 0:
        draw.rectangle((0, top, 1440, bottom), fill=stripe_color)
    # Separator line at top of each row
    draw.line((24, top, 1440 - 24, top), fill=separator_color, width=1)

# Final bottom separator after last known row
draw.line((24, row_tops[-1] + row_height, 1440 - 24, row_tops[-1] + row_height), fill=separator_color, width=1)

# Soft left margin guideline (visual structure only)
draw.line((48, underline_y - 120, 48, 2960 - 48), fill=(245, 245, 245), width=1)

# Large whitespace area remains (no text/icons drawn here)
# Done drawing structural backgrounds and separators.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 48, 69)
    canvas.paste(_c0, (1154, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1154, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 98, 65)
    canvas.paste(_c1, (1214, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1214, 0, 1312, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/02_icon_7.10.png
try:
    _c2 = get_crop(2, 62, 63)
    canvas.paste(_c2, (179, 1), _c2)
except Exception:
    pass
layout["7.10"] = [179, 1, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 63, 62)
    canvas.paste(_c3, (308, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [308, 2, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/04_icon_7.10.png
try:
    _c4 = get_crop(4, 59, 64)
    canvas.paste(_c4, (116, 1), _c4)
except Exception:
    pass
layout["7.10"] = [116, 1, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/05_icon_7.10.png
try:
    _c5 = get_crop(5, 168, 168)
    canvas.paste(_c5, (0, 72), _c5)
except Exception:
    pass
layout["7.10"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 52, 64)
    canvas.paste(_c6, (1319, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1319, 0, 1371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 51, 58)
    canvas.paste(_c7, (247, 5), _c7)
except Exception:
    pass
layout["icon_7"] = [247, 5, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 85, 96)
    canvas.paste(_c8, (1310, 286), _c8)
except Exception:
    pass
layout["icon_8"] = [1310, 286, 1395, 382]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/09_icon_District_of_Columbia.png
try:
    _c9 = get_crop(9, 1440, 132)
    canvas.paste(_c9, (0, 1740), _c9)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1740, 1440, 1872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/10_icon_San_Francisco.png
try:
    _c10 = get_crop(10, 1440, 132)
    canvas.paste(_c10, (0, 840), _c10)
except Exception:
    pass
layout["San_Francisco"] = [0, 840, 1440, 972]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/11_icon_Chicago.png
try:
    _c11 = get_crop(11, 1440, 132)
    canvas.paste(_c11, (0, 1380), _c11)
except Exception:
    pass
layout["Chicago"] = [0, 1380, 1440, 1512]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/12_icon_United_Kingdom.png
try:
    _c12 = get_crop(12, 1440, 132)
    canvas.paste(_c12, (0, 2100), _c12)
except Exception:
    pass
layout["United_Kingdom"] = [0, 2100, 1440, 2232]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/13_icon_District_of_Columbia.png
try:
    _c13 = get_crop(13, 1440, 132)
    canvas.paste(_c13, (0, 1560), _c13)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/14_icon_Los_Angeles.png
try:
    _c14 = get_crop(14, 1440, 132)
    canvas.paste(_c14, (0, 1020), _c14)
except Exception:
    pass
layout["Los_Angeles"] = [0, 1020, 1440, 1152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/15_icon_Philadelphia.png
try:
    _c15 = get_crop(15, 1440, 132)
    canvas.paste(_c15, (0, 1920), _c15)
except Exception:
    pass
layout["Philadelphia"] = [0, 1920, 1440, 2052]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/16_icon_Miami.png
try:
    _c16 = get_crop(16, 1440, 132)
    canvas.paste(_c16, (0, 1200), _c16)
except Exception:
    pass
layout["Miami"] = [0, 1200, 1440, 1332]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 53, 65)
    canvas.paste(_c17, (382, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [382, 1, 435, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/18_text_7.10.png
try:
    _c18 = get_crop(18, 89, 41)
    canvas.paste(_c18, (22, 17), _c18)
except Exception:
    pass
layout["7.10"] = [22, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/19_text_Chicago.png
try:
    _c19 = get_crop(19, 1344, 129)
    canvas.paste(_c19, (48, 264), _c19)
except Exception:
    pass
layout["Chicago"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/20_text_Nearby.png
try:
    _c20 = get_crop(20, 415, 114)
    canvas.paste(_c20, (48, 465), _c20)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/21_text_Online_events.png
try:
    _c21 = get_crop(21, 452, 114)
    canvas.paste(_c21, (511, 465), _c21)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/22_text_Current_location.png
try:
    _c22 = get_crop(22, 415, 114)
    canvas.paste(_c22, (48, 465), _c22)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/23_text_Virtual_attendance.png
try:
    _c23 = get_crop(23, 452, 114)
    canvas.paste(_c23, (511, 465), _c23)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/24_text_Found_locations.png
try:
    _c24 = get_crop(24, 311, 50)
    canvas.paste(_c24, (44, 740), _c24)
except Exception:
    pass
layout["Found_locations"] = [44, 740, 355, 790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/25_text_New_York.png
try:
    _c25 = get_crop(25, 212, 55)
    canvas.paste(_c25, (44, 2288), _c25)
except Exception:
    pass
layout["New_York"] = [44, 2288, 256, 2343]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/26_text_New_York.png
try:
    _c26 = get_crop(26, 154, 38)
    canvas.paste(_c26, (47, 2353), _c26)
except Exception:
    pass
layout["New_York"] = [47, 2353, 201, 2391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/27_text_Atlanta.png
try:
    _c27 = get_crop(27, 163, 52)
    canvas.paste(_c27, (44, 2468), _c27)
except Exception:
    pass
layout["Atlanta"] = [44, 2468, 207, 2520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/28_text_Georgia.png
try:
    _c28 = get_crop(28, 133, 43)
    canvas.paste(_c28, (45, 2533), _c28)
except Exception:
    pass
layout["Georgia"] = [45, 2533, 178, 2576]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/29_clickable_New_York.png
try:
    _c29 = get_crop(29, 1440, 132)
    canvas.paste(_c29, (0, 2280), _c29)
except Exception:
    pass
layout["New_York"] = [0, 2280, 1440, 2412]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_08_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-10/30_clickable_Atlanta.png
try:
    _c30 = get_crop(30, 1440, 132)
    canvas.paste(_c30, (0, 2460), _c30)
except Exception:
    pass
layout["Atlanta"] = [0, 2460, 1440, 2592]
