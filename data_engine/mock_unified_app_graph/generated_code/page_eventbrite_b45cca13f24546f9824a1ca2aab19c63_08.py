# page_id: page_eventbrite_b45cca13f24546f9824a1ca2aab19c63_08
# screenshot: 2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10.png
# step_index: 8/11
# task: Open Eventbrite. Search for "Art". Filter for events in New York. Select first recommended event. Save it to wishlist. What is the duration of the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided in the environment.
# Draw the overall background
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar area (top ~72px) - light gray background to match screenshot
STATUS_BAR_H = 72
draw.rectangle([(0, 0), (1440, STATUS_BAR_H)], fill="#E9E9E9")

# Subtle top divider shadow under status bar
draw.line([(0, STATUS_BAR_H), (1440, STATUS_BAR_H)], fill="#D6D6D6", width=1)

# Header / search area background (below status bar)
HEADER_TOP = STATUS_BAR_H
HEADER_BOTTOM = 320
draw.rectangle([(0, HEADER_TOP), (1440, HEADER_BOTTOM)], fill="#FFFFFF")

# Prominent blue underline for the search field (accent color)
underline_y = 240
draw.line([(48, underline_y), (1392, underline_y)], fill="#2746FF", width=6)

# Slight bottom divider under header
draw.line([(0, HEADER_BOTTOM), (1440, HEADER_BOTTOM)], fill="#EFEFF2", width=1)

# Decorative circular background behind where the left back/accessory icon would be
# (background element only — actual icon will be pasted on top)
draw.ellipse([(36, HEADER_TOP + 36), (128, HEADER_TOP + 128)], fill="#F2F6FF")

# "Nearby" row card: rounded rectangle behind the nearby/current location row
nearby_card_top = 420
nearby_card_bottom = 540
draw.rounded_rectangle(
    [(36, nearby_card_top), (1404, nearby_card_bottom)],
    radius=20,
    fill="#F7FBFF",
    outline=None
)

# Subtle divider between the "Nearby" card and the list header area
draw.line([(36, nearby_card_bottom + 12), (1404, nearby_card_bottom + 12)], fill="#F0F2F6", width=1)

# Found locations section background (keeps the list visually grouped)
found_list_top = 720
draw.rectangle([(0, found_list_top), (1440, 2960)], fill="#FFFFFF")

# Thin header label separator (above the list title area)
draw.line([(36, found_list_top - 8), (1404, found_list_top - 8)], fill="#F0F2F6", width=1)

# Draw separators for each detected clickable list row.
# Using the known Y positions for the list items (they will be pasted on top)
row_tops = [840, 1020, 1200, 1380, 1560, 1740, 1920, 2100, 2280, 2460]
row_height = 132
separator_color = "#F3F4F8"
for y in row_tops:
    bottom = y + row_height
    # subtle bottom separator line for each row
    draw.line([(48, bottom), (1392, bottom)], fill=separator_color, width=1)

# Add faint left margin guideline (visual structure, very subtle)
draw.line([(48, HEADER_BOTTOM + 12), (48, 2960)], fill="#F6F6F8", width=1)

# Optional: faint section dividers to group every few rows (subtle)
group_divider_color = "#F2F3F6"
group_dividers = [1560, 1920]  # approximate grouping lines seen in the layout
for gy in group_dividers:
    draw.line([(36, gy - 6), (1404, gy - 6)], fill=group_divider_color, width=1)

# Footer bottom edge subtle shadow to ground the page
draw.rectangle([(0, 2940), (1440, 2960)], fill="#FFFFFF")
draw.line([(0, 2940), (1440, 2940)], fill="#F0F0F0", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 48, 69)
    canvas.paste(_c0, (1154, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1154, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 98, 65)
    canvas.paste(_c1, (1214, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1214, 0, 1312, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/02_icon_7.06.png
try:
    _c2 = get_crop(2, 64, 65)
    canvas.paste(_c2, (178, 0), _c2)
except Exception:
    pass
layout["7.06"] = [178, 0, 242, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/03_icon_7.06.png
try:
    _c3 = get_crop(3, 168, 168)
    canvas.paste(_c3, (0, 72), _c3)
except Exception:
    pass
layout["7.06"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 66, 63)
    canvas.paste(_c4, (307, 1), _c4)
except Exception:
    pass
layout["icon_4"] = [307, 1, 373, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/05_icon_7.06.png
try:
    _c5 = get_crop(5, 62, 65)
    canvas.paste(_c5, (114, 1), _c5)
except Exception:
    pass
layout["7.06"] = [114, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 85, 91)
    canvas.paste(_c6, (1310, 288), _c6)
except Exception:
    pass
layout["icon_6"] = [1310, 288, 1395, 379]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 55, 63)
    canvas.paste(_c7, (245, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [245, 1, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 52, 62)
    canvas.paste(_c8, (1319, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1319, 1, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/09_icon_Los_Angeles.png
try:
    _c9 = get_crop(9, 1440, 132)
    canvas.paste(_c9, (0, 1020), _c9)
except Exception:
    pass
layout["Los_Angeles"] = [0, 1020, 1440, 1152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/10_icon_San_Francisco.png
try:
    _c10 = get_crop(10, 1440, 132)
    canvas.paste(_c10, (0, 840), _c10)
except Exception:
    pass
layout["San_Francisco"] = [0, 840, 1440, 972]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/11_icon_Chicago.png
try:
    _c11 = get_crop(11, 1440, 132)
    canvas.paste(_c11, (0, 1380), _c11)
except Exception:
    pass
layout["Chicago"] = [0, 1380, 1440, 1512]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/12_icon_Miami.png
try:
    _c12 = get_crop(12, 1440, 132)
    canvas.paste(_c12, (0, 1200), _c12)
except Exception:
    pass
layout["Miami"] = [0, 1200, 1440, 1332]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/13_icon_District_of_Columbia.png
try:
    _c13 = get_crop(13, 1440, 132)
    canvas.paste(_c13, (0, 1560), _c13)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/14_text_7.06.png
try:
    _c14 = get_crop(14, 91, 45)
    canvas.paste(_c14, (20, 15), _c14)
except Exception:
    pass
layout["7.06"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/15_text_New_Yorkl.png
try:
    _c15 = get_crop(15, 1344, 129)
    canvas.paste(_c15, (48, 264), _c15)
except Exception:
    pass
layout["New_Yorkl"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/16_text_Nearby.png
try:
    _c16 = get_crop(16, 415, 114)
    canvas.paste(_c16, (48, 465), _c16)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/17_text_Current_location.png
try:
    _c17 = get_crop(17, 415, 114)
    canvas.paste(_c17, (48, 465), _c17)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/18_text_Found_locations.png
try:
    _c18 = get_crop(18, 311, 50)
    canvas.paste(_c18, (44, 740), _c18)
except Exception:
    pass
layout["Found_locations"] = [44, 740, 355, 790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/19_text_Boston.png
try:
    _c19 = get_crop(19, 163, 61)
    canvas.paste(_c19, (42, 1746), _c19)
except Exception:
    pass
layout["Boston"] = [42, 1746, 205, 1807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/20_text_Massachusetts.png
try:
    _c20 = get_crop(20, 249, 39)
    canvas.paste(_c20, (47, 1814), _c20)
except Exception:
    pass
layout["Massachusetts"] = [47, 1814, 296, 1853]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/21_text_Philadelphia.png
try:
    _c21 = get_crop(21, 1440, 132)
    canvas.paste(_c21, (0, 1920), _c21)
except Exception:
    pass
layout["Philadelphia"] = [0, 1920, 1440, 2052]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/22_text_Pennsylvania.png
try:
    _c22 = get_crop(22, 214, 43)
    canvas.paste(_c22, (45, 1995), _c22)
except Exception:
    pass
layout["Pennsylvania"] = [45, 1995, 259, 2038]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/23_text_London.png
try:
    _c23 = get_crop(23, 168, 52)
    canvas.paste(_c23, (44, 2109), _c23)
except Exception:
    pass
layout["London"] = [44, 2109, 212, 2161]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/24_text_United_Kingdom.png
try:
    _c24 = get_crop(24, 263, 45)
    canvas.paste(_c24, (45, 2173), _c24)
except Exception:
    pass
layout["United_Kingdom"] = [45, 2173, 308, 2218]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/25_text_New_York.png
try:
    _c25 = get_crop(25, 212, 55)
    canvas.paste(_c25, (44, 2288), _c25)
except Exception:
    pass
layout["New_York"] = [44, 2288, 256, 2343]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/26_text_New_York.png
try:
    _c26 = get_crop(26, 154, 38)
    canvas.paste(_c26, (47, 2353), _c26)
except Exception:
    pass
layout["New_York"] = [47, 2353, 201, 2391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/27_text_Atlanta.png
try:
    _c27 = get_crop(27, 163, 52)
    canvas.paste(_c27, (44, 2468), _c27)
except Exception:
    pass
layout["Atlanta"] = [44, 2468, 207, 2520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/28_text_Georgia.png
try:
    _c28 = get_crop(28, 133, 43)
    canvas.paste(_c28, (45, 2533), _c28)
except Exception:
    pass
layout["Georgia"] = [45, 2533, 178, 2576]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/29_clickable_Boston.png
try:
    _c29 = get_crop(29, 1440, 132)
    canvas.paste(_c29, (0, 1740), _c29)
except Exception:
    pass
layout["Boston"] = [0, 1740, 1440, 1872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/30_clickable_London.png
try:
    _c30 = get_crop(30, 1440, 132)
    canvas.paste(_c30, (0, 2100), _c30)
except Exception:
    pass
layout["London"] = [0, 2100, 1440, 2232]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/31_clickable_New_York.png
try:
    _c31 = get_crop(31, 1440, 132)
    canvas.paste(_c31, (0, 2280), _c31)
except Exception:
    pass
layout["New_York"] = [0, 2280, 1440, 2412]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_08_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-10/32_clickable_Atlanta.png
try:
    _c32 = get_crop(32, 1440, 132)
    canvas.paste(_c32, (0, 2460), _c32)
except Exception:
    pass
layout["Atlanta"] = [0, 2460, 1440, 2592]
