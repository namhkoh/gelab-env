# page_id: page_eventbrite_a9f633a394e74f78843aa30bd2792346_05
# screenshot: 2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7.png
# step_index: 5/18
# task: Open Eventbrite. Set the city to "Los Angeles". Look for Photography workshops happening next week. What is the price of the tickets for first non-promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the UI page.
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
status_bar_color = "#D0D0D0"     # light gray for status bar
header_underline_color = "#2C51F0"  # eventbrite-like blue underline
divider_color = "#E9E9EA"        # subtle divider
muted_bg = "#F7F8FA"             # very light gray background accents
circle_bg = "#EAF3FF"            # pale blue for icon background
circle_border = "#CFE3FF"        # border for icon background
card_border = "#F0F0F2"          # card subtle border

W, H = canvas.size

# 1) Background fill (canvas is already white, but ensure consistent fill)
draw.rectangle([(0, 0), (W, H)], fill="#FFFFFF")

# 2) Status bar area at top (~72px high)
status_h = 72
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)
# subtle bottom divider under status bar
draw.line([(0, status_h-1), (W, status_h-1)], fill=divider_color, width=1)

# 3) Header / toolbar area (below status bar)
header_top = status_h
header_bottom = 180
draw.rectangle([(0, header_top), (W, header_bottom)], fill="#FFFFFF")
# Blue underline for the search/header input (thin, prominent)
underline_y = 158
draw.line([(48, underline_y), (W-48, underline_y)], fill=header_underline_color, width=6)

# subtle divider below header area
draw.line([(0, header_bottom-1), (W, header_bottom-1)], fill=divider_color, width=1)

# 4) Option pills area (Nearby / Online events) - pale circular backgrounds behind icons
# Left option circle
left_circle_center = (130, 520)
circle_radius = 44
lc = left_circle_center
draw.ellipse([(lc[0]-circle_radius, lc[1]-circle_radius), (lc[0]+circle_radius, lc[1]+circle_radius)],
             fill=circle_bg, outline=circle_border, width=2)
# Right option circle
right_circle_center = (600, 520)
rc = right_circle_center
draw.ellipse([(rc[0]-circle_radius, rc[1]-circle_radius), (rc[0]+circle_radius, rc[1]+circle_radius)],
             fill=circle_bg, outline=circle_border, width=2)

# Subtle divider below the options area
options_bottom = 600
draw.line([(0, options_bottom), (W, options_bottom)], fill=divider_color, width=1)

# 5) Found locations section background card (subtle)
found_top = 720
found_bottom = H - 120
card_margin = 24
draw.rounded_rectangle([(card_margin, found_top), (W - card_margin, found_bottom)],
                       radius=8, fill="#FFFFFF", outline=card_border, width=1)

# Soft top divider for the found locations card
draw.line([(card_margin + 8, found_top + 1), (W - card_margin - 8, found_top + 1)], fill="#F4F5F6", width=1)

# 6) Section separators between main logical blocks
# Under status bar (already), under header (already), under options (already),
# additionally add a faint separator above the list area title region
draw.line([(48, found_top - 64), (W - 48, found_top - 64)], fill="#F3F4F6", width=1)

# 7) Light guides for list rows (subtle horizontal markers to structure spacing)
# Use the detected rows as guides (do not draw any text or icons)
row_starts = [840, 1020, 1200, 1380, 1560, 1740, 1920, 2100, 2280, 2460]
row_height = 132
for y in row_starts:
    # very subtle baseline marker near the bottom of each row area
    baseline_y = y + row_height - 8
    if baseline_y < found_bottom - 24:
        draw.line([(48, baseline_y), (W - 48, baseline_y)], fill="#FBFBFC", width=1)

# 8) Right-side floating accent (subtle) to mimic UI balance (no icons drawn)
# small rounded rectangle near the upper-right header area (purely decorative)
draw.rounded_rectangle([(W-200, header_top+18), (W-80, header_top+78)],
                       radius=12, fill=muted_bg, outline="#ECEEF0", width=1)

# End of structural drawing. Actual icons/text will be pasted on top of these elements.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 48, 69)
    canvas.paste(_c0, (1154, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1154, 0, 1202, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 98, 65)
    canvas.paste(_c1, (1214, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1214, 0, 1312, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/02_icon_4.50.png
try:
    _c2 = get_crop(2, 62, 64)
    canvas.paste(_c2, (179, 1), _c2)
except Exception:
    pass
layout["4.50"] = [179, 1, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 63, 62)
    canvas.paste(_c3, (308, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [308, 2, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/04_icon_4.50.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["4.50"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/05_icon_4.50.png
try:
    _c5 = get_crop(5, 61, 65)
    canvas.paste(_c5, (114, 1), _c5)
except Exception:
    pass
layout["4.50"] = [114, 1, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 50, 58)
    canvas.paste(_c6, (248, 5), _c6)
except Exception:
    pass
layout["icon_6"] = [248, 5, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 51, 63)
    canvas.paste(_c7, (1320, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1320, 0, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 86, 97)
    canvas.paste(_c8, (1310, 285), _c8)
except Exception:
    pass
layout["icon_8"] = [1310, 285, 1396, 382]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/09_icon_San_Francisco.png
try:
    _c9 = get_crop(9, 1440, 132)
    canvas.paste(_c9, (0, 840), _c9)
except Exception:
    pass
layout["San_Francisco"] = [0, 840, 1440, 972]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/10_icon_District_of_Columbia.png
try:
    _c10 = get_crop(10, 1440, 132)
    canvas.paste(_c10, (0, 1740), _c10)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1740, 1440, 1872]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/11_icon_Chicago.png
try:
    _c11 = get_crop(11, 1440, 132)
    canvas.paste(_c11, (0, 1380), _c11)
except Exception:
    pass
layout["Chicago"] = [0, 1380, 1440, 1512]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/12_icon_Los_Angeles.png
try:
    _c12 = get_crop(12, 1440, 132)
    canvas.paste(_c12, (0, 1020), _c12)
except Exception:
    pass
layout["Los_Angeles"] = [0, 1020, 1440, 1152]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/13_icon_United_Kingdom.png
try:
    _c13 = get_crop(13, 1440, 132)
    canvas.paste(_c13, (0, 2100), _c13)
except Exception:
    pass
layout["United_Kingdom"] = [0, 2100, 1440, 2232]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/14_icon_District_of_Columbia.png
try:
    _c14 = get_crop(14, 1440, 132)
    canvas.paste(_c14, (0, 1560), _c14)
except Exception:
    pass
layout["District_of_Columbia"] = [0, 1560, 1440, 1692]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/15_icon_Miami.png
try:
    _c15 = get_crop(15, 1440, 132)
    canvas.paste(_c15, (0, 1200), _c15)
except Exception:
    pass
layout["Miami"] = [0, 1200, 1440, 1332]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/16_icon_Philadelphia.png
try:
    _c16 = get_crop(16, 1440, 132)
    canvas.paste(_c16, (0, 1920), _c16)
except Exception:
    pass
layout["Philadelphia"] = [0, 1920, 1440, 2052]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/17_icon_Nearby.png
try:
    _c17 = get_crop(17, 415, 114)
    canvas.paste(_c17, (48, 465), _c17)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 53, 65)
    canvas.paste(_c18, (382, 1), _c18)
except Exception:
    pass
layout["icon_18"] = [382, 1, 435, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/19_text_4.50.png
try:
    _c19 = get_crop(19, 89, 43)
    canvas.paste(_c19, (22, 17), _c19)
except Exception:
    pass
layout["4.50"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/20_text_Los_Angeles.png
try:
    _c20 = get_crop(20, 1344, 129)
    canvas.paste(_c20, (48, 264), _c20)
except Exception:
    pass
layout["Los_Angeles"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/21_text_Online_events.png
try:
    _c21 = get_crop(21, 452, 114)
    canvas.paste(_c21, (511, 465), _c21)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/22_text_Virtual_attendance.png
try:
    _c22 = get_crop(22, 452, 114)
    canvas.paste(_c22, (511, 465), _c22)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/23_text_Found_locations.png
try:
    _c23 = get_crop(23, 311, 50)
    canvas.paste(_c23, (44, 740), _c23)
except Exception:
    pass
layout["Found_locations"] = [44, 740, 355, 790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/24_text_New_York.png
try:
    _c24 = get_crop(24, 212, 55)
    canvas.paste(_c24, (44, 2288), _c24)
except Exception:
    pass
layout["New_York"] = [44, 2288, 256, 2343]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/25_text_New_York.png
try:
    _c25 = get_crop(25, 154, 38)
    canvas.paste(_c25, (47, 2353), _c25)
except Exception:
    pass
layout["New_York"] = [47, 2353, 201, 2391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/26_text_Atlanta.png
try:
    _c26 = get_crop(26, 163, 52)
    canvas.paste(_c26, (44, 2468), _c26)
except Exception:
    pass
layout["Atlanta"] = [44, 2468, 207, 2520]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/27_text_Georgia.png
try:
    _c27 = get_crop(27, 133, 43)
    canvas.paste(_c27, (45, 2533), _c27)
except Exception:
    pass
layout["Georgia"] = [45, 2533, 178, 2576]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/28_clickable_New_York.png
try:
    _c28 = get_crop(28, 1440, 132)
    canvas.paste(_c28, (0, 2280), _c28)
except Exception:
    pass
layout["New_York"] = [0, 2280, 1440, 2412]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_05_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-7/29_clickable_Atlanta.png
try:
    _c29 = get_crop(29, 1440, 132)
    canvas.paste(_c29, (0, 2460), _c29)
except Exception:
    pass
layout["Atlanta"] = [0, 2460, 1440, 2592]
