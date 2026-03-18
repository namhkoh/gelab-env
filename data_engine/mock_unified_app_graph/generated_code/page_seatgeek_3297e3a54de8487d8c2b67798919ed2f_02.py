# page_id: page_seatgeek_3297e3a54de8487d8c2b67798919ed2f_02
# screenshot: 2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5.png
# step_index: 2/11
# task: Open SeatGeek. Search "Comedy Show in Los Angeles". Find the top recommendation. When is the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level UI background and structural elements for the provided canvas.
# Canvas: PIL Image 1440x2960 (RGB), draw: ImageDraw, fonts: provided.

# Colors
BG = (255, 255, 255)            # overall background (white)
STATUS_BG = (245, 246, 247)     # very light gray for status bar
HEADER_BG = (255, 255, 255)     # header white
DIVIDER = (230, 230, 230)       # subtle divider
CARD_FILL = (250, 250, 250)     # light card background
CARD_BORDER = (235, 235, 235)   # card border
INPUT_FILL = (255, 255, 255)    # input background (white)
SUBTLE_BG = (245, 245, 245)     # slightly off-white for large containers
DARK_OVERLAY = (60, 60, 60)     # dark content area/background
BOTTOM_BAR = (255, 255, 255)    # bottom nav background
SHADOW = (240, 240, 240)

W, H = canvas.size

# Fill overall background
draw.rectangle([0, 0, W, H], fill=BG)

# Status bar area (~0-56px)
status_h = 56
draw.rectangle([0, 0, W, status_h], fill=STATUS_BG)

# Thin divider below status bar
draw.line([(0, status_h), (W, status_h)], fill=DIVIDER, width=1)

# Header / toolbar area (~56-140)
header_top = status_h
header_bottom = 140
draw.rectangle([0, header_top, W, header_bottom], fill=HEADER_BG)
# bottom divider for header
draw.line([(24, header_bottom), (W-24, header_bottom)], fill=DIVIDER, width=1)

# Main search/filter panel card (rounded)
search_card_pad = 24
search_card_top = header_bottom + 16
search_card_bottom = 1120
search_card_rect = [search_card_pad, search_card_top, W - search_card_pad, search_card_bottom]
try:
    draw.rounded_rectangle(search_card_rect, radius=18, fill=CARD_FILL, outline=CARD_BORDER, width=1)
except Exception:
    # Fallback if rounded_rectangle not available
    draw.rectangle(search_card_rect, fill=CARD_FILL, outline=CARD_BORDER)

# Inner subtle separators inside search card to divide sections (no text)
# Separator under Location input
sep1_y = search_card_top + 240
draw.line([(search_card_rect[0] + 16, sep1_y), (search_card_rect[2] - 16, sep1_y)], fill=DIVIDER, width=1)

# Separator under Date area (approx)
sep2_y = search_card_top + 460
draw.line([(search_card_rect[0] + 16, sep2_y), (search_card_rect[2] - 16, sep2_y)], fill=DIVIDER, width=1)

# Location input background (rounded rect) - background only (no icons/text)
loc_box = [search_card_rect[0] + 24, search_card_top + 80, search_card_rect[2] - 24, search_card_top + 160]
try:
    draw.rounded_rectangle(loc_box, radius=12, fill=INPUT_FILL, outline=CARD_BORDER, width=1)
except Exception:
    draw.rectangle(loc_box, fill=INPUT_FILL, outline=CARD_BORDER)

# Date selection container background (larger rounded area behind segmented control)
date_box = [search_card_rect[0] + 18, search_card_top + 180, search_card_rect[2] - 18, search_card_top + 340]
try:
    draw.rounded_rectangle(date_box, radius=12, fill=SUBTLE_BG, outline=(245,245,245), width=1)
except Exception:
    draw.rectangle(date_box, fill=SUBTLE_BG, outline=(245,245,245))

# A subtle horizontal rule inside date box (to match UI grouping)
draw.line([(date_box[0] + 12, (date_box[1]+date_box[3])//2), (date_box[2] - 12, (date_box[1]+date_box[3])//2)], fill=(235,235,235), width=1)

# "Set custom date" row background (thin)
custom_row = [date_box[0] + 12, date_box[1] + (date_box[3]-date_box[1])//2 + 8, date_box[2] - 12, date_box[3] - 12]
# leave it mostly blank but provide faint divider above (already added) and rounded edges hint
try:
    draw.rounded_rectangle(custom_row, radius=8, fill=SUBTLE_BG, outline=None)
except Exception:
    draw.rectangle(custom_row, fill=SUBTLE_BG)

# Divider between search card and content below
draw.line([(24, search_card_bottom + 6), (W - 24, search_card_bottom + 6)], fill=DIVIDER, width=1)

# "Just for you" horizontal card area background (slightly darkened area behind thumbnails)
just_for_you_top = search_card_bottom + 16
just_for_you_bottom = just_for_you_top + 420
# This region in the original screenshot appears dimmed (an overlay). We'll put a subtle panel.
try:
    draw.rectangle([0, just_for_you_top, W, just_for_you_bottom], fill=DARK_OVERLAY)
except Exception:
    draw.rectangle([0, just_for_you_top, W, just_for_you_bottom], fill=DARK_OVERLAY)

# Within that area, draw three rounded thumbnail placeholders (background canvases).
thumb_w = 360
thumb_h = 240
thumb_gap = 32
thumb_y = just_for_you_top + 48
thumb_x_start = 42
for i in range(3):
    x0 = thumb_x_start + i * (thumb_w + thumb_gap)
    x1 = x0 + thumb_w
    y0 = thumb_y
    y1 = y0 + thumb_h
    # rounded rect backgrounds for thumbnails (they will be overplotted by actual images)
    try:
        draw.rounded_rectangle([x0, y0, x1, y1], radius=14, fill=(45,45,45), outline=(70,70,70))
    except Exception:
        draw.rectangle([x0, y0, x1, y1], fill=(45,45,45), outline=(70,70,70))

# Separator line below just-for-you thumbnails
separator_y = just_for_you_bottom + 8
draw.line([(24, separator_y), (W - 24, separator_y)], fill=DIVIDER, width=1)

# Trending events list background (light)
trending_top = separator_y + 16
trending_bottom = trending_top + 740
try:
    draw.rectangle([24, trending_top, W - 24, trending_bottom], fill=BG)
except Exception:
    draw.rectangle([24, trending_top, W - 24, trending_bottom], fill=BG)

# Draw list item separators for trending list (3 items visible)
list_item_h = 140
for i in range(3):
    y = trending_top + i * list_item_h
    # subtle icon background circle placeholder on left
    circle_radius = 36
    cx = 64
    cy = y + 60
    draw.ellipse([(cx - circle_radius, cy - circle_radius), (cx + circle_radius, cy + circle_radius)], fill=(255,245,245))
    # horizontal separator line under each item
    draw.line([(48, y + list_item_h - 6), (W - 48, y + list_item_h - 6)], fill=DIVIDER, width=1)

# Darkened content overlay for the lower portion of the screen (modal/overlay effect)
overlay_top = just_for_you_top
overlay_bottom = trending_bottom + 40
# Use a darker rectangle to emulate the dim overlay shown in screenshot (will be under pasted elements)
draw.rectangle([0, overlay_top, W, overlay_bottom], fill=(55, 55, 55))

# Bottom navigation bar
bottom_bar_h = 140
bottom_top = H - bottom_bar_h
draw.rectangle([0, bottom_top, W, H], fill=BOTTOM_BAR)
# Top divider for bottom bar
draw.line([(0, bottom_top), (W, bottom_top)], fill=DIVIDER, width=1)

# Inner navigation icon placeholders (only backgrounds/shapes, actual icons will be pasted)
nav_count = 5
nav_spacing = W // nav_count
icon_y_center = bottom_top + bottom_bar_h // 2
for i in range(nav_count):
    cx = nav_spacing * i + nav_spacing // 2
    # draw small circular touch target background for each nav item (light)
    r = 34
    draw.ellipse([(cx - r, icon_y_center - r), (cx + r, icon_y_center + r)], outline=(245,245,245), width=1, fill=(255,255,255))

# Final subtle top shadow under header area for depth
for offset in range(3):
    alpha_y = header_bottom + offset
    draw.line([(0, alpha_y), (W, alpha_y)], fill=(245 - offset, 245 - offset, 245 - offset), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/00_icon_Tomorrow.png
try:
    _c0 = get_crop(0, 1344, 153)
    canvas.paste(_c0, (48, 505), _c0)
except Exception:
    pass
layout["Tomorrow"] = [48, 505, 1392, 658]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/01_icon_S94.png
try:
    _c1 = get_crop(1, 472, 313)
    canvas.paste(_c1, (539, 1432), _c1)
except Exception:
    pass
layout["S94+"] = [539, 1432, 1011, 1745]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/02_icon_BOOK_OF.png
try:
    _c2 = get_crop(2, 470, 318)
    canvas.paste(_c2, (42, 1432), _c2)
except Exception:
    pass
layout["BOOK_OF"] = [42, 1432, 512, 1750]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 48, 69)
    canvas.paste(_c3, (1153, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1153, 0, 1201, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/04_icon_New_York_NY.png
try:
    _c4 = get_crop(4, 60, 57)
    canvas.paste(_c4, (243, 6), _c4)
except Exception:
    pass
layout["New_York,_NY"] = [243, 6, 303, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 96, 142)
    canvas.paste(_c5, (1344, 2469), _c5)
except Exception:
    pass
layout["icon_5"] = [1344, 2469, 1440, 2611]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/06_icon_View_all.png
try:
    _c6 = get_crop(6, 93, 143)
    canvas.paste(_c6, (1347, 2229), _c6)
except Exception:
    pass
layout["View_all"] = [1347, 2229, 1440, 2372]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 56, 56)
    canvas.paste(_c7, (313, 7), _c7)
except Exception:
    pass
layout["icon_7"] = [313, 7, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/08_icon_7_10_my.png
try:
    _c8 = get_crop(8, 44, 56)
    canvas.paste(_c8, (187, 6), _c8)
except Exception:
    pass
layout["7:10_my"] = [187, 6, 231, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 52, 67)
    canvas.paste(_c9, (1320, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1320, 0, 1372, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/10_icon_7_10_my.png
try:
    _c10 = get_crop(10, 53, 58)
    canvas.paste(_c10, (116, 4), _c10)
except Exception:
    pass
layout["7:10_my"] = [116, 4, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 102, 68)
    canvas.paste(_c11, (1213, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1213, 0, 1315, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/12_icon_Yankee_Stadium.png
try:
    _c12 = get_crop(12, 288, 168)
    canvas.paste(_c12, (576, 2792), _c12)
except Exception:
    pass
layout["Yankee_Stadium"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/13_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c13 = get_crop(13, 288, 168)
    canvas.paste(_c13, (288, 2792), _c13)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/14_icon_S116.png
try:
    _c14 = get_crop(14, 406, 312)
    canvas.paste(_c14, (1034, 1433), _c14)
except Exception:
    pass
layout["S116+"] = [1034, 1433, 1440, 1745]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/15_icon_Tracking.png
try:
    _c15 = get_crop(15, 288, 168)
    canvas.paste(_c15, (864, 2792), _c15)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 106, 120)
    canvas.paste(_c16, (1140, 2487), _c16)
except Exception:
    pass
layout["icon_16"] = [1140, 2487, 1246, 2607]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/17_icon_Browse.png
try:
    _c17 = get_crop(17, 288, 162)
    canvas.paste(_c17, (0, 2792), _c17)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/18_icon_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c18 = get_crop(18, 288, 168)
    canvas.paste(_c18, (576, 2792), _c18)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/19_icon_Close.png
try:
    _c19 = get_crop(19, 144, 240)
    canvas.paste(_c19, (1260, 72), _c19)
except Exception:
    pass
layout["Close"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/20_icon_Account.png
try:
    _c20 = get_crop(20, 288, 168)
    canvas.paste(_c20, (1152, 2792), _c20)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/21_text_New_York_NY.png
try:
    _c21 = get_crop(21, 382, 68)
    canvas.paste(_c21, (48, 133), _c21)
except Exception:
    pass
layout["New_York,_NY"] = [48, 133, 430, 201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/22_text_date.png
try:
    _c22 = get_crop(22, 117, 52)
    canvas.paste(_c22, (134, 208), _c22)
except Exception:
    pass
layout["date"] = [134, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/23_text_Location.png
try:
    _c23 = get_crop(23, 235, 54)
    canvas.paste(_c23, (44, 382), _c23)
except Exception:
    pass
layout["Location"] = [44, 382, 279, 436]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/24_text_Date.png
try:
    _c24 = get_crop(24, 140, 60)
    canvas.paste(_c24, (42, 775), _c24)
except Exception:
    pass
layout["Date"] = [42, 775, 182, 835]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/25_text_Clear.png
try:
    _c25 = get_crop(25, 264, 149)
    canvas.paste(_c25, (1176, 730), _c25)
except Exception:
    pass
layout["Clear"] = [1176, 730, 1440, 879]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/26_text_Today.png
try:
    _c26 = get_crop(26, 448, 149)
    canvas.paste(_c26, (48, 901), _c26)
except Exception:
    pass
layout["Today"] = [48, 901, 496, 1050]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/27_text_Tomorrow.png
try:
    _c27 = get_crop(27, 448, 149)
    canvas.paste(_c27, (496, 901), _c27)
except Exception:
    pass
layout["Tomorrow"] = [496, 901, 944, 1050]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/28_text_Weekend.png
try:
    _c28 = get_crop(28, 448, 149)
    canvas.paste(_c28, (944, 901), _c28)
except Exception:
    pass
layout["Weekend"] = [944, 901, 1392, 1050]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/29_text_Set_custom_date.png
try:
    _c29 = get_crop(29, 492, 149)
    canvas.paste(_c29, (474, 1052), _c29)
except Exception:
    pass
layout["Set_custom_date"] = [474, 1052, 966, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/30_text_Just_for_you.png
try:
    _c30 = get_crop(30, 306, 66)
    canvas.paste(_c30, (38, 1310), _c30)
except Exception:
    pass
layout["Just_for_you"] = [38, 1310, 344, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/31_text_View_all.png
try:
    _c31 = get_crop(31, 170, 49)
    canvas.paste(_c31, (1223, 1314), _c31)
except Exception:
    pass
layout["View_all"] = [1223, 1314, 1393, 1363]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/32_text_The_Book_of_Mormon.png
try:
    _c32 = get_crop(32, 454, 50)
    canvas.paste(_c32, (42, 1785), _c32)
except Exception:
    pass
layout["The_Book_of_Mormon"] = [42, 1785, 496, 1835]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/33_text_Andrew_Schulz.png
try:
    _c33 = get_crop(33, 322, 52)
    canvas.paste(_c33, (544, 1783), _c33)
except Exception:
    pass
layout["Andrew_Schulz"] = [544, 1783, 866, 1835]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/34_text_Matt_Rife.png
try:
    _c34 = get_crop(34, 204, 50)
    canvas.paste(_c34, (1041, 1785), _c34)
except Exception:
    pass
layout["Matt_Rife"] = [1041, 1785, 1245, 1835]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/35_text_Sat.png
try:
    _c35 = get_crop(35, 94, 51)
    canvas.paste(_c35, (43, 1854), _c35)
except Exception:
    pass
layout["Sat,"] = [43, 1854, 137, 1905]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/36_text_27_7_PM.png
try:
    _c36 = get_crop(36, 174, 48)
    canvas.paste(_c36, (213, 1852), _c36)
except Exception:
    pass
layout["27,7_PM"] = [213, 1852, 387, 1900]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/37_text_Sat.png
try:
    _c37 = get_crop(37, 94, 51)
    canvas.paste(_c37, (540, 1854), _c37)
except Exception:
    pass
layout["Sat,"] = [540, 1854, 634, 1905]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/38_text_4_8_PM.png
try:
    _c38 = get_crop(38, 153, 50)
    canvas.paste(_c38, (729, 1852), _c38)
except Exception:
    pass
layout["4,8_PM"] = [729, 1852, 882, 1902]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/39_text_Sat.png
try:
    _c39 = get_crop(39, 94, 51)
    canvas.paste(_c39, (1037, 1854), _c39)
except Exception:
    pass
layout["Sat,"] = [1037, 1854, 1131, 1905]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/40_text_27_7_PM.png
try:
    _c40 = get_crop(40, 174, 48)
    canvas.paste(_c40, (1210, 1852), _c40)
except Exception:
    pass
layout["27,7_PM"] = [1210, 1852, 1384, 1900]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/41_text_Trending_events.png
try:
    _c41 = get_crop(41, 424, 81)
    canvas.paste(_c41, (38, 2054), _c41)
except Exception:
    pass
layout["Trending_events"] = [38, 2054, 462, 2135]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/42_text_View_all.png
try:
    _c42 = get_crop(42, 165, 43)
    canvas.paste(_c42, (1227, 2071), _c42)
except Exception:
    pass
layout["View_all"] = [1227, 2071, 1392, 2114]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/43_text_Cabaret_at_the_Kit_Kat_Club.png
try:
    _c43 = get_crop(43, 598, 52)
    canvas.paste(_c43, (229, 2239), _c43)
except Exception:
    pass
layout["Cabaret_at_the_Kit_Kat_Cl"] = [229, 2239, 827, 2291]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/44_text_May_1.png
try:
    _c44 = get_crop(44, 130, 61)
    canvas.paste(_c44, (228, 2309), _c44)
except Exception:
    pass
layout["May_1"] = [228, 2309, 358, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/45_text_August_Wilson_Theatre.png
try:
    _c45 = get_crop(45, 489, 61)
    canvas.paste(_c45, (380, 2311), _c45)
except Exception:
    pass
layout["August_Wilson_Theatre"] = [380, 2311, 869, 2372]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_02_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-5/46_text_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c46 = get_crop(46, 288, 168)
    canvas.paste(_c46, (576, 2792), _c46)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [576, 2792, 864, 2960]
