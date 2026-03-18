# page_id: page_eventbrite_31d4c984d33c4fe69d15d4e595fa95f6_05
# screenshot: 2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7.png
# step_index: 5/14
# task: Open Eventbrite. Look for 'community events' in 'Chicago'. Select the first event happening tomorrow that is not promoted. Check if they have an option for 'refund policy'.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the mobile UI page
# Available variables:
# - canvas: PIL Image (1440x2960 RGB)
# - draw: PIL ImageDraw object
# - font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# 1) Base background (white) - canvas already white, but ensure fill
draw.rectangle([0, 0, w, h], fill="#FFFFFF")

# 2) Status bar area at top (~72px) - light neutral gray
status_h = 72
draw.rectangle([0, 0, w, status_h], fill="#E6E6E6")

# subtle inner divider under status bar to separate it from toolbar
draw.line([(0, status_h), (w, status_h)], fill="#D0D0D0", width=1)

# 3) Toolbar / header area (below status bar)
toolbar_top = status_h
toolbar_bottom = 220  # roomy header to match screenshot spacing
draw.rectangle([0, toolbar_top, w, toolbar_bottom], fill="#FFFFFF")

# prominent blue underline below header (title underline)
underline_y = toolbar_bottom - 4
draw.line([(48, underline_y), (w-48, underline_y)], fill="#2F51E8", width=4)

# thin hairline directly under the blue underline for crisp separation
draw.line([(48, underline_y+6), (w-48, underline_y+6)], fill="#E8EAF6", width=1)

# 4) "Nearby" section card background (rounded rectangle container)
# Position chosen to leave space for header and follow the screenshot layout.
nearby_top = toolbar_bottom + 32
nearby_bottom = nearby_top + 140
nearby_margin_x = 36
card_bbox = [nearby_margin_x, nearby_top, w - nearby_margin_x, nearby_bottom]
draw.rounded_rectangle(card_bbox, radius=16, fill="#FBFCFF", outline="#E9EEF9", width=1)

# subtle drop shadow for the card (very faint)
shadow_bbox = [card_bbox[0], card_bbox[1]+6, card_bbox[2], card_bbox[3]+6]
draw.rounded_rectangle(shadow_bbox, radius=16, fill=None, outline=None)
# manual faint line to imply depth
draw.line([(card_bbox[0], card_bbox[3]+2), (card_bbox[2], card_bbox[3]+2)], fill="#F3F5F9", width=1)

# 5) Separator / section label divider above the "Found locations" list
found_label_y = 740  # anchor near detected "Found locations" area
draw.line([(44, found_label_y), (w-44, found_label_y)], fill="#F1F2F6", width=1)

# 6) List rows background bands (subtle) and separators for the found locations list
# Detected list rows start around y = 840 and repeat every ~180px; draw separators at those positions.
list_row_ys = [840, 1020, 1200, 1380, 1560, 1740, 1920, 2100, 2280, 2460]
for y in list_row_ys:
    # very light divider across the full width with left/right padding
    draw.line([(36, y), (w-36, y)], fill="#F3F4F6", width=1)

# 7) Subtle left column guide (visual alignment aid) - very light, not duplicating any UI element
guide_x = 44
draw.line([(guide_x, toolbar_bottom+12), (guide_x, h-40)], fill="#FCFCFD", width=1)

# 8) Bottom area subtle finish line near end of content
draw.line([(36, h-80), (w-36, h-80)], fill="#FAFBFC", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 48, 69)
    canvas.paste(_c0, (1154, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1154, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 98, 65)
    canvas.paste(_c1, (1214, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1214, 0, 1312, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/02_icon_8.07.png
try:
    _c2 = get_crop(2, 168, 168)
    canvas.paste(_c2, (0, 72), _c2)
except Exception:
    pass
layout["8.07"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/03_icon_8.07.png
try:
    _c3 = get_crop(3, 62, 63)
    canvas.paste(_c3, (179, 1), _c3)
except Exception:
    pass
layout["8.07"] = [179, 1, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/04_icon_8.07.png
try:
    _c4 = get_crop(4, 64, 67)
    canvas.paste(_c4, (111, 0), _c4)
except Exception:
    pass
layout["8.07"] = [111, 0, 175, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 67, 63)
    canvas.paste(_c5, (307, 1), _c5)
except Exception:
    pass
layout["icon_5"] = [307, 1, 374, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 85, 91)
    canvas.paste(_c6, (1310, 288), _c6)
except Exception:
    pass
layout["icon_6"] = [1310, 288, 1395, 379]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 54, 62)
    canvas.paste(_c7, (246, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [246, 1, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 48, 59)
    canvas.paste(_c8, (1322, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [1322, 2, 1370, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/09_icon_San_Francisco.png
try:
    _c9 = get_crop(9, 1440, 132)
    canvas.paste(_c9, (0, 840), _c9)
except Exception:
    pass
layout["San_Francisco"] = [0, 840, 1440, 972]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/10_icon_Los_Angeles.png
try:
    _c10 = get_crop(10, 1440, 132)
    canvas.paste(_c10, (0, 1020), _c10)
except Exception:
    pass
layout["Los_Angeles"] = [0, 1020, 1440, 1152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/11_icon_Chicago.png
try:
    _c11 = get_crop(11, 1440, 132)
    canvas.paste(_c11, (0, 1380), _c11)
except Exception:
    pass
layout["Chicago"] = [0, 1380, 1440, 1512]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/12_icon_Miami.png
try:
    _c12 = get_crop(12, 1440, 132)
    canvas.paste(_c12, (0, 1200), _c12)
except Exception:
    pass
layout["Miami"] = [0, 1200, 1440, 1332]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/13_icon_District_of_Columbia.png
try:
    _c13 = get_crop(13, 1440, 132)
    canvas.paste(_c13, (0, 1560), _c13)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/14_text_8.07.png
try:
    _c14 = get_crop(14, 89, 43)
    canvas.paste(_c14, (20, 17), _c14)
except Exception:
    pass
layout["8.07"] = [20, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/15_text_Chicago.png
try:
    _c15 = get_crop(15, 1344, 129)
    canvas.paste(_c15, (48, 264), _c15)
except Exception:
    pass
layout["Chicago"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/16_text_Nearby.png
try:
    _c16 = get_crop(16, 415, 114)
    canvas.paste(_c16, (48, 465), _c16)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/17_text_Current_location.png
try:
    _c17 = get_crop(17, 415, 114)
    canvas.paste(_c17, (48, 465), _c17)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/18_text_Found_locations.png
try:
    _c18 = get_crop(18, 311, 50)
    canvas.paste(_c18, (44, 740), _c18)
except Exception:
    pass
layout["Found_locations"] = [44, 740, 355, 790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/19_text_Boston.png
try:
    _c19 = get_crop(19, 163, 61)
    canvas.paste(_c19, (42, 1746), _c19)
except Exception:
    pass
layout["Boston"] = [42, 1746, 205, 1807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/20_text_Massachusetts.png
try:
    _c20 = get_crop(20, 249, 39)
    canvas.paste(_c20, (47, 1814), _c20)
except Exception:
    pass
layout["Massachusetts"] = [47, 1814, 296, 1853]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/21_text_Philadelphia.png
try:
    _c21 = get_crop(21, 1440, 132)
    canvas.paste(_c21, (0, 1920), _c21)
except Exception:
    pass
layout["Philadelphia"] = [0, 1920, 1440, 2052]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/22_text_Pennsylvania.png
try:
    _c22 = get_crop(22, 214, 43)
    canvas.paste(_c22, (45, 1995), _c22)
except Exception:
    pass
layout["Pennsylvania"] = [45, 1995, 259, 2038]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/23_text_London.png
try:
    _c23 = get_crop(23, 168, 52)
    canvas.paste(_c23, (44, 2109), _c23)
except Exception:
    pass
layout["London"] = [44, 2109, 212, 2161]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/24_text_United_Kingdom.png
try:
    _c24 = get_crop(24, 263, 45)
    canvas.paste(_c24, (45, 2173), _c24)
except Exception:
    pass
layout["United_Kingdom"] = [45, 2173, 308, 2218]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/25_text_New_York.png
try:
    _c25 = get_crop(25, 212, 55)
    canvas.paste(_c25, (44, 2288), _c25)
except Exception:
    pass
layout["New_York"] = [44, 2288, 256, 2343]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/26_text_New_York.png
try:
    _c26 = get_crop(26, 154, 38)
    canvas.paste(_c26, (47, 2353), _c26)
except Exception:
    pass
layout["New_York"] = [47, 2353, 201, 2391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/27_text_Atlanta.png
try:
    _c27 = get_crop(27, 163, 52)
    canvas.paste(_c27, (44, 2468), _c27)
except Exception:
    pass
layout["Atlanta"] = [44, 2468, 207, 2520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/28_text_Georgia.png
try:
    _c28 = get_crop(28, 133, 43)
    canvas.paste(_c28, (45, 2533), _c28)
except Exception:
    pass
layout["Georgia"] = [45, 2533, 178, 2576]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/29_clickable_Boston.png
try:
    _c29 = get_crop(29, 1440, 132)
    canvas.paste(_c29, (0, 1740), _c29)
except Exception:
    pass
layout["Boston"] = [0, 1740, 1440, 1872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/30_clickable_London.png
try:
    _c30 = get_crop(30, 1440, 132)
    canvas.paste(_c30, (0, 2100), _c30)
except Exception:
    pass
layout["London"] = [0, 2100, 1440, 2232]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/31_clickable_New_York.png
try:
    _c31 = get_crop(31, 1440, 132)
    canvas.paste(_c31, (0, 2280), _c31)
except Exception:
    pass
layout["New_York"] = [0, 2280, 1440, 2412]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31d4c984d33c4fe69d15d4e595fa95f6/step_05_2024_4_23_20_6_31d4c984d33c4fe69d15d4e595fa95f6-7/32_clickable_Atlanta.png
try:
    _c32 = get_crop(32, 1440, 132)
    canvas.paste(_c32, (0, 2460), _c32)
except Exception:
    pass
layout["Atlanta"] = [0, 2460, 1440, 2592]
