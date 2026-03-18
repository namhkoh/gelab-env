# page_id: page_eventbrite_182a9ea236924e9eb43ccbe82fd1506e_12
# screenshot: 2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14.png
# step_index: 12/13
# task: Open Eventbrite. Set time to tomorrow. Clear all search filters. Select the third one in New York. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural elements for the Eventbrite UI mockup
# Uses available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Full-canvas background fill (very light off-white dominant color)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFD")

# Status bar area (top ~84px) - muted gray
draw.rectangle([(0, 0), (1440, 84)], fill="#CFCFCF")

# Thin subtle overlay line at bottom of status bar to separate it from header
draw.line([(0, 84), (1440, 84)], fill="#E6E6EA", width=1)

# Header / toolbar background (white card area under status bar)
header_top = 90
header_bottom = 188
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")

# Header subtle bottom divider
draw.line([(24, header_bottom), (1416, header_bottom)], fill="#EAEAF0", width=2)

# Main content card container (rounded rectangle) under the header to group the top details
card1_bbox = (36, 210, 1404, 820)
draw.rounded_rectangle(card1_bbox, radius=20, fill="#FFFFFF", outline="#F1F2F7", width=2)

# Divider between top details and the "About this event" area
divider_y1 = 620
draw.line([(48, divider_y1), (1392, divider_y1)], fill="#ECEEF3", width=2)

# Light section separator lines for content sections
separators = [920, 1180, 1560, 1960]
for y in separators:
    draw.line([(48, y), (1392, y)], fill="#F0F1F6", width=1)

# Subtle section header background strip for "About this event" area (keeps it visually distinct)
about_strip_bbox = (24, 720, 1416, 780)
draw.rectangle(about_strip_bbox, fill="#FBFBFE")

# Subtle rounded card behind the location block to visually group the location content
loc_card_bbox = (36, 980, 1404, 1320)
draw.rounded_rectangle(loc_card_bbox, radius=16, fill="#FFFFFF", outline="#F5F6FA", width=1)

# Very light dividing stroke inside the location card to echo the UI separators
draw.line([(60, 1120), (1380, 1120)], fill="#F2F3F8", width=1)

# Thin faint bottom shadow under mid content areas to add depth
draw.line([(36, 1326), (1404, 1326)], fill="#F6F7FB", width=6)

# A few subtle vertical guide edges to frame content columns (non-intrusive)
draw.line([(48, 200), (48, 1960)], fill="#FBFBFD", width=24)  # wide almost transparent margin
draw.line([(1392, 200), (1392, 1960)], fill="#FBFBFD", width=24)

# Note: Interactive footer / "Reserve a spot" area is provided externally and must not be redrawn.
# All drawn elements above are background/structure only and avoid rendering any detected icons/text.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/00_icon_More.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1116, 108), _c0)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/01_icon_Share.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 108), _c1)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/02_icon_Job_Seekers.png
try:
    _c2 = get_crop(2, 240, 240)
    canvas.paste(_c2, (600, 1990), _c2)
except Exception:
    pass
layout["Job_Seekers"] = [600, 1990, 840, 2230]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 113, 104)
    canvas.paste(_c3, (987, 2440), _c3)
except Exception:
    pass
layout["icon_3"] = [987, 2440, 1100, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 106, 102)
    canvas.paste(_c4, (1216, 2442), _c4)
except Exception:
    pass
layout["icon_4"] = [1216, 2442, 1322, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/05_icon_Business_Professional.png
try:
    _c5 = get_crop(5, 234, 144)
    canvas.paste(_c5, (48, 1235), _c5)
except Exception:
    pass
layout["Business_&_Professional"] = [48, 1235, 282, 1379]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 93, 103)
    canvas.paste(_c6, (1107, 2441), _c6)
except Exception:
    pass
layout["icon_6"] = [1107, 2441, 1200, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/07_icon_4hrs.png
try:
    _c7 = get_crop(7, 198, 74)
    canvas.paste(_c7, (50, 470), _c7)
except Exception:
    pass
layout["4hrs"] = [50, 470, 248, 544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/08_icon_Reserve_a_spot.png
try:
    _c8 = get_crop(8, 1440, 636)
    canvas.paste(_c8, (0, 2324), _c8)
except Exception:
    pass
layout["Reserve_a_spot"] = [0, 2324, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/09_icon_9.33.png
try:
    _c9 = get_crop(9, 51, 60)
    canvas.paste(_c9, (184, 1), _c9)
except Exception:
    pass
layout["9.33"] = [184, 1, 235, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/10_icon_Free.png
try:
    _c10 = get_crop(10, 99, 106)
    canvas.paste(_c10, (236, 2573), _c10)
except Exception:
    pass
layout["Free"] = [236, 2573, 335, 2679]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 55, 60)
    canvas.paste(_c11, (247, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [247, 1, 302, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/12_icon_New_York.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 325), _c12)
except Exception:
    pass
layout["New_York"] = [48, 325, 1392, 469]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/13_icon_9.33.png
try:
    _c13 = get_crop(13, 52, 61)
    canvas.paste(_c13, (117, 2), _c13)
except Exception:
    pass
layout["9.33"] = [117, 2, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 97, 59)
    canvas.paste(_c14, (1214, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [1214, 1, 1311, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/15_icon_Free.png
try:
    _c15 = get_crop(15, 139, 104)
    canvas.paste(_c15, (96, 2573), _c15)
except Exception:
    pass
layout["Free"] = [96, 2573, 235, 2677]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 51, 56)
    canvas.paste(_c16, (1320, 4), _c16)
except Exception:
    pass
layout["icon_16"] = [1320, 4, 1371, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/17_icon_9.33.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (36, 108), _c17)
except Exception:
    pass
layout["9.33"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/18_icon_The_organizer_will_review_refund_request.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 325), _c18)
except Exception:
    pass
layout["The_organizer_will_review"] = [48, 325, 1392, 469]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/19_icon_Show_map.png
try:
    _c19 = get_crop(19, 226, 144)
    canvas.paste(_c19, (1166, 1453), _c19)
except Exception:
    pass
layout["Show_map"] = [1166, 1453, 1392, 1597]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 54, 62)
    canvas.paste(_c20, (314, 2), _c20)
except Exception:
    pass
layout["icon_20"] = [314, 2, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/21_icon_you_looking_for_ajob_in_New_York_If_you_.png
try:
    _c21 = get_crop(21, 234, 144)
    canvas.paste(_c21, (48, 1235), _c21)
except Exception:
    pass
layout["you_looking_for_ajob_in_N"] = [48, 1235, 282, 1379]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/22_icon_Refund_policy.png
try:
    _c22 = get_crop(22, 368, 73)
    canvas.paste(_c22, (65, 579), _c22)
except Exception:
    pass
layout["Refund_policy"] = [65, 579, 433, 652]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/23_icon_New_York_Virtual_Job_Fair_New_York_NY_10.png
try:
    _c23 = get_crop(23, 226, 144)
    canvas.paste(_c23, (1166, 1453), _c23)
except Exception:
    pass
layout["New_York;_Virtual_Job_Fai"] = [1166, 1453, 1392, 1597]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/24_icon_Read_more.png
try:
    _c24 = get_crop(24, 234, 144)
    canvas.paste(_c24, (48, 1235), _c24)
except Exception:
    pass
layout["Read_more"] = [48, 1235, 282, 1379]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/25_text_9.33.png
try:
    _c25 = get_crop(25, 96, 49)
    canvas.paste(_c25, (16, 12), _c25)
except Exception:
    pass
layout["9.33"] = [16, 12, 112, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/26_text_General_Admission.png
try:
    _c26 = get_crop(26, 415, 55)
    canvas.paste(_c26, (116, 2451), _c26)
except Exception:
    pass
layout["General_Admission"] = [116, 2451, 531, 2506]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_12_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-14/27_text_Job_Seekers.png
try:
    _c27 = get_crop(27, 1440, 636)
    canvas.paste(_c27, (0, 2324), _c27)
except Exception:
    pass
layout["Job_Seekers"] = [0, 2324, 1440, 2960]
