# page_id: page_eventbrite_03837235ef8649c7821b415a8d3b0093_05
# screenshot: 2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7.png
# step_index: 5/8
# task: Open Eventbrite. Locate the 'Conference' category. Filter the results to only show virtual events. Choose the first event from the results. What is the duration of this event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint background and structural UI elements for the mobile Eventbrite page
# Available objects: canvas (1440x2960 RGB Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
bg_color = (250, 251, 253)        # subtle off-white background
status_bar_color = (34, 34, 34)   # dark status bar
header_divider = (227, 230, 234)  # light divider lines
muted_card = (245, 247, 249)      # muted card background
image_placeholder = (40, 44, 60)  # dark image placeholder
image_placeholder_soft = (60, 66, 86)
nav_top_border = (220, 224, 228)
card_outline = (234, 236, 239)

W, H = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (W, H)], fill=bg_color)

# Status bar area (top)
status_h = 96
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# Header / search area (under status bar)
header_top = status_h
header_h = 200
draw.rectangle([(0, header_top), (W, header_top + header_h)], fill=bg_color)
# subtle bottom divider under header/search area
draw.line([(48, header_top + header_h - 8), (W - 48, header_top + header_h - 8)], fill=header_divider, width=2)

# Small thin divider below the filter row area
filter_div_y = 460
draw.line([(24, filter_div_y), (W - 24, filter_div_y)], fill=header_divider, width=2)

# First event image placeholder (rounded)
first_img_x1, first_img_y1 = 48, 676
first_img_x2 = first_img_x1 + 1344
first_img_y2 = first_img_y1 + 360
draw.rounded_rectangle([(first_img_x1, first_img_y1), (first_img_x2, first_img_y2)],
                       radius=28, fill=image_placeholder)

# Subtle overlay band at bottom of first image to mimic visual emphasis (no icons/text)
overlay_band_h = 84
draw.rectangle([(first_img_x1, first_img_y2 - overlay_band_h), (first_img_x2, first_img_y2)],
               fill=image_placeholder_soft)

# Card area for first event metadata (white card background)
meta1_y1 = first_img_y2 + 24
meta1_y2 = meta1_y1 + 120
draw.rounded_rectangle([(first_img_x1, meta1_y1), (first_img_x2, meta1_y2)],
                       radius=16, fill=(255, 255, 255), outline=card_outline, width=1)

# Divider between first event and next content
draw.line([(24, meta1_y2 + 28), (W - 24, meta1_y2 + 28)], fill=header_divider, width=1)

# Second event image placeholder (rounded)
second_img_x1, second_img_y1 = 48, 1192
second_img_x2 = second_img_x1 + 1344
second_img_y2 = second_img_y1 + 420
draw.rounded_rectangle([(second_img_x1, second_img_y1), (second_img_x2, second_img_y2)],
                       radius=20, fill=muted_card)

# Small soft highlight rectangle inside second image area (no content, just background)
highlight_w = int((second_img_x2 - second_img_x1) * 0.28)
draw.rectangle([(second_img_x1 + 24, second_img_y1 + 24),
                (second_img_x1 + 24 + highlight_w, second_img_y1 + 24 + 24)],
               fill=(255, 255, 255, 30))

# Card area for second event metadata (white card background)
meta2_y1 = second_img_y2 + 24
meta2_y2 = meta2_y1 + 160
draw.rounded_rectangle([(second_img_x1, meta2_y1), (second_img_x2, meta2_y2)],
                       radius=16, fill=(255, 255, 255), outline=card_outline, width=1)

# Large content area further down (placeholder background where detailed listings appear)
content_block_y1 = meta2_y2 + 24
content_block_y2 = 2600
draw.rectangle([(24, content_block_y1), (W - 24, content_block_y2)], fill=bg_color)
# subtle separators within content area
sep_y = content_block_y1 + 220
while sep_y < content_block_y2 - 200:
    draw.line([(48, sep_y), (W - 48, sep_y)], fill=header_divider, width=1)
    sep_y += 260

# Bottom navigation bar background
nav_h = 156
nav_y1 = H - nav_h
draw.rectangle([(0, nav_y1), (W, H)], fill=(255, 255, 255))
# top border of nav bar
draw.line([(0, nav_y1), (W, nav_y1)], fill=nav_top_border, width=1)

# Light shadow under top header for separation
draw.line([(0, header_top + header_h), (W, header_top + header_h)], fill=card_outline, width=1)

# Left and right safe margins vertical guides (very subtle, not content)
draw.line([(24, header_top + 8), (24, H - 8)], fill=(255, 255, 255, 10))
draw.line([(W - 24, header_top + 8), (W - 24, H - 8)], fill=(255, 255, 255, 10))

# End of structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (850, 410), _c0)
except Exception:
    pass
layout["Music"] = [850, 410, 1037, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1049, 410), _c1)
except Exception:
    pass
layout["Business"] = [1049, 410, 1290, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (438, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [438, 410, 838, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/04_icon_Fo.png
try:
    _c4 = get_crop(4, 136, 111)
    canvas.paste(_c4, (1295, 406), _c4)
except Exception:
    pass
layout["Fo("] = [1295, 406, 1431, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2336), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2336, 1236, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2336), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2336, 1380, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/07_icon_Fi.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1092, 1192), _c7)
except Exception:
    pass
layout["Fi"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/08_icon_4.41.png
try:
    _c8 = get_crop(8, 128, 115)
    canvas.paste(_c8, (54, 114), _c8)
except Exception:
    pass
layout["4.41"] = [54, 114, 182, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/09_icon_Advanced_Clinical_Supervisor_Next_Level.png
try:
    _c9 = get_crop(9, 1344, 996)
    canvas.paste(_c9, (48, 1820), _c9)
except Exception:
    pass
layout["Advanced_Clinical_Supervi"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/10_icon_Search_forae.png
try:
    _c10 = get_crop(10, 69, 64)
    canvas.paste(_c10, (307, 0), _c10)
except Exception:
    pass
layout["Search_forae"] = [307, 0, 376, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/11_icon_4.41.png
try:
    _c11 = get_crop(11, 65, 65)
    canvas.paste(_c11, (111, 0), _c11)
except Exception:
    pass
layout["4.41"] = [111, 0, 176, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/12_icon_4.41.png
try:
    _c12 = get_crop(12, 62, 64)
    canvas.paste(_c12, (180, 0), _c12)
except Exception:
    pass
layout["4.41"] = [180, 0, 242, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/13_icon_clal_Adu.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1236, 1192), _c13)
except Exception:
    pass
layout["clal_Adu"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 56, 64)
    canvas.paste(_c14, (246, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [246, 0, 302, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/15_icon_8.30_AM_EDT.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (576, 2804), _c15)
except Exception:
    pass
layout["8.30_AM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 75, 60)
    canvas.paste(_c16, (1208, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1208, 0, 1283, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 65, 59)
    canvas.paste(_c17, (1315, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1315, 0, 1380, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/18_icon_Search_forae.png
try:
    _c18 = get_crop(18, 1344, 191)
    canvas.paste(_c18, (48, 72), _c18)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/19_icon_Its_Hard_For_You_To_Uncover_The_Secrets_.png
try:
    _c19 = get_crop(19, 1344, 1096)
    canvas.paste(_c19, (48, 676), _c19)
except Exception:
    pass
layout["Its_Hard_For_You_To_Uncov"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/20_icon_Advanced_Clinical_Supervisor_Next_Level.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (864, 2804), _c20)
except Exception:
    pass
layout["Advanced_Clinical_Supervi"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/21_icon_8.30_AM_EDT.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["8.30_AM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/22_icon_Search_forae.png
try:
    _c22 = get_crop(22, 52, 62)
    canvas.paste(_c22, (383, 2), _c22)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 435, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/23_icon_San_Francisco.png
try:
    _c23 = get_crop(23, 536, 144)
    canvas.paste(_c23, (0, 259), _c23)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/24_icon_4.41.png
try:
    _c24 = get_crop(24, 97, 64)
    canvas.paste(_c24, (8, 0), _c24)
except Exception:
    pass
layout["4.41"] = [8, 0, 105, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 40, 61)
    canvas.paste(_c25, (1274, 0), _c25)
except Exception:
    pass
layout["icon_25"] = [1274, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/26_icon_Retirement_Revolution.png
try:
    _c26 = get_crop(26, 1344, 1096)
    canvas.paste(_c26, (48, 676), _c26)
except Exception:
    pass
layout["Retirement_Revolution"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/27_icon_Retirement_Revolution.png
try:
    _c27 = get_crop(27, 1344, 1096)
    canvas.paste(_c27, (48, 676), _c27)
except Exception:
    pass
layout["Retirement_Revolution"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/28_icon_More.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (1152, 2804), _c28)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/29_icon_Advanced_Clinical_Supervisor_Next_Level.png
try:
    _c29 = get_crop(29, 1344, 996)
    canvas.paste(_c29, (48, 1820), _c29)
except Exception:
    pass
layout["Advanced_Clinical_Supervi"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/30_icon_Promoted.png
try:
    _c30 = get_crop(30, 245, 66)
    canvas.paste(_c30, (83, 1664), _c30)
except Exception:
    pass
layout["Promoted"] = [83, 1664, 328, 1730]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/31_text_378_events.png
try:
    _c31 = get_crop(31, 372, 103)
    canvas.paste(_c31, (54, 410), _c31)
except Exception:
    pass
layout["378_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/32_text_Online.png
try:
    _c32 = get_crop(32, 131, 48)
    canvas.paste(_c32, (90, 1607), _c32)
except Exception:
    pass
layout["Online"] = [90, 1607, 221, 1655]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/33_text_Online.png
try:
    _c33 = get_crop(33, 131, 50)
    canvas.paste(_c33, (90, 2745), _c33)
except Exception:
    pass
layout["Online"] = [90, 2745, 221, 2795]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/03837235ef8649c7821b415a8d3b0093/step_05_2024_4_24_16_40_03837235ef8649c7821b415a8d3b0093-7/34_clickable_Home.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (0, 2804), _c34)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
