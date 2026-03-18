# page_id: page_eventbrite_eb32c51543d749539b68e6c61ff72fb8_05
# screenshot: 2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7.png
# step_index: 5/19
# task: Open Eventbrite. Set the city to San Francisco. Filter for events occurring between May 1st and May 15th under the category 'Music'. Select the first event and check the pricing options available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
draw.rectangle((0, 0, 1440, 2960), fill="#FCFCFD")

# Status bar area (top)
status_height = 72
draw.rectangle((0, 0, 1440, status_height), fill="#BDBDBD")

# Subtle bottom shadow of status bar
for i, a in enumerate((220, 210, 200, 190, 180)):
    alpha = int(6 - i)  # decreasing intensity for subtle banding
    # simulate subtle darker band using slightly darker grays
    shade = (200 - i * 6)
    draw.line((0, status_height + i, 1440, status_height + i), fill=(shade, shade, shade))

# Header underline (search field divider)
underline_y = 224
draw.line((48, underline_y, 1392, underline_y), fill="#2B6FF6", width=4)

# Rounded card behind the 'Nearby' / 'Online events' group
card_left, card_top, card_right, card_bottom = 36, 300, 1404, 540
draw.rounded_rectangle((card_left, card_top, card_right, card_bottom),
                       radius=20, fill="#FFFFFF", outline="#E6EEF9", width=1)

# Very subtle inner highlight at top of that card
draw.line((card_left + 2, card_top + 2, card_right - 2, card_top + 2), fill="#F5F9FF", width=1)

# Large subtle divider separating header area from list
divider_y = 600
draw.line((36, divider_y, 1404, divider_y), fill="#F1F3F6", width=1)

# Draw separators between location rows (light hairlines)
row_tops = [840, 1020, 1200, 1380, 1560, 1740, 1920, 2100, 2280, 2460]
for y in row_tops:
    # separator above each row
    draw.line((36, y, 1404, y), fill="#F3F4F6", width=1)

# Add faint section header background behind "Found locations" top area
found_section_top = 700
found_section_bottom = 820
draw.rectangle((36, found_section_top, 1404, found_section_bottom), fill="#FFFFFF")

# Subtle left edge guideline/padding accent for list area
draw.rectangle((36, 700, 44, 2960), fill="#FFFFFF")

# A light bottom padding band so the page doesn't feel abrupt at the end
draw.rectangle((0, 2890, 1440, 2960), fill="#FBFBFC")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 48, 69)
    canvas.paste(_c0, (1154, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1154, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 98, 65)
    canvas.paste(_c1, (1214, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1214, 0, 1312, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/02_icon_7.47.png
try:
    _c2 = get_crop(2, 62, 63)
    canvas.paste(_c2, (179, 1), _c2)
except Exception:
    pass
layout["7.47"] = [179, 1, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/03_icon_7.47.png
try:
    _c3 = get_crop(3, 62, 65)
    canvas.paste(_c3, (112, 1), _c3)
except Exception:
    pass
layout["7.47"] = [112, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/04_icon_7.47.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["7.47"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 63, 62)
    canvas.paste(_c5, (308, 2), _c5)
except Exception:
    pass
layout["icon_5"] = [308, 2, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 51, 58)
    canvas.paste(_c6, (247, 5), _c6)
except Exception:
    pass
layout["icon_6"] = [247, 5, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 51, 63)
    canvas.paste(_c7, (1320, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1320, 0, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 85, 97)
    canvas.paste(_c8, (1310, 285), _c8)
except Exception:
    pass
layout["icon_8"] = [1310, 285, 1395, 382]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/09_icon_San_Francisco.png
try:
    _c9 = get_crop(9, 1440, 132)
    canvas.paste(_c9, (0, 840), _c9)
except Exception:
    pass
layout["San_Francisco"] = [0, 840, 1440, 972]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/10_icon_District_of_Columbia.png
try:
    _c10 = get_crop(10, 1440, 132)
    canvas.paste(_c10, (0, 1740), _c10)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1740, 1440, 1872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/11_icon_Chicago.png
try:
    _c11 = get_crop(11, 1440, 132)
    canvas.paste(_c11, (0, 1380), _c11)
except Exception:
    pass
layout["Chicago"] = [0, 1380, 1440, 1512]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/12_icon_Los_Angeles.png
try:
    _c12 = get_crop(12, 1440, 132)
    canvas.paste(_c12, (0, 1020), _c12)
except Exception:
    pass
layout["Los_Angeles"] = [0, 1020, 1440, 1152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/13_icon_Miami.png
try:
    _c13 = get_crop(13, 1440, 132)
    canvas.paste(_c13, (0, 1200), _c13)
except Exception:
    pass
layout["Miami"] = [0, 1200, 1440, 1332]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/14_icon_United_Kingdom.png
try:
    _c14 = get_crop(14, 1440, 132)
    canvas.paste(_c14, (0, 2100), _c14)
except Exception:
    pass
layout["United_Kingdom"] = [0, 2100, 1440, 2232]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/15_icon_7.47.png
try:
    _c15 = get_crop(15, 92, 64)
    canvas.paste(_c15, (15, 1), _c15)
except Exception:
    pass
layout["7.47"] = [15, 1, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/16_icon_District_of_Columbia.png
try:
    _c16 = get_crop(16, 1440, 132)
    canvas.paste(_c16, (0, 1560), _c16)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/17_icon_Philadelphia.png
try:
    _c17 = get_crop(17, 1440, 132)
    canvas.paste(_c17, (0, 1920), _c17)
except Exception:
    pass
layout["Philadelphia"] = [0, 1920, 1440, 2052]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/18_icon_District_of_Columbia.png
try:
    _c18 = get_crop(18, 1440, 132)
    canvas.paste(_c18, (0, 1560), _c18)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 53, 65)
    canvas.paste(_c19, (382, 1), _c19)
except Exception:
    pass
layout["icon_19"] = [382, 1, 435, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/20_icon_Nearby.png
try:
    _c20 = get_crop(20, 415, 114)
    canvas.paste(_c20, (48, 465), _c20)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/21_text_San_Francisco.png
try:
    _c21 = get_crop(21, 1344, 129)
    canvas.paste(_c21, (48, 264), _c21)
except Exception:
    pass
layout["San_Francisco"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/22_text_Online_events.png
try:
    _c22 = get_crop(22, 452, 114)
    canvas.paste(_c22, (511, 465), _c22)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/23_text_Virtual_attendance.png
try:
    _c23 = get_crop(23, 452, 114)
    canvas.paste(_c23, (511, 465), _c23)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/24_text_Found_locations.png
try:
    _c24 = get_crop(24, 311, 50)
    canvas.paste(_c24, (44, 740), _c24)
except Exception:
    pass
layout["Found_locations"] = [44, 740, 355, 790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/25_text_New_York.png
try:
    _c25 = get_crop(25, 212, 55)
    canvas.paste(_c25, (44, 2288), _c25)
except Exception:
    pass
layout["New_York"] = [44, 2288, 256, 2343]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/26_text_New_York.png
try:
    _c26 = get_crop(26, 154, 38)
    canvas.paste(_c26, (47, 2353), _c26)
except Exception:
    pass
layout["New_York"] = [47, 2353, 201, 2391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/27_text_Atlanta.png
try:
    _c27 = get_crop(27, 163, 52)
    canvas.paste(_c27, (44, 2468), _c27)
except Exception:
    pass
layout["Atlanta"] = [44, 2468, 207, 2520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/28_text_Georgia.png
try:
    _c28 = get_crop(28, 133, 43)
    canvas.paste(_c28, (45, 2533), _c28)
except Exception:
    pass
layout["Georgia"] = [45, 2533, 178, 2576]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/29_clickable_New_York.png
try:
    _c29 = get_crop(29, 1440, 132)
    canvas.paste(_c29, (0, 2280), _c29)
except Exception:
    pass
layout["New_York"] = [0, 2280, 1440, 2412]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_05_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-7/30_clickable_Atlanta.png
try:
    _c30 = get_crop(30, 1440, 132)
    canvas.paste(_c30, (0, 2460), _c30)
except Exception:
    pass
layout["Atlanta"] = [0, 2460, 1440, 2592]
