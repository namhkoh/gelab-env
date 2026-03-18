# page_id: page_seatgeek_3297e3a54de8487d8c2b67798919ed2f_07
# screenshot: 2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10.png
# step_index: 7/11
# task: Open SeatGeek. Search "Comedy Show in Los Angeles". Find the top recommendation. When is the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint UI background and structural elements for the provided canvas.
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Overall page background (very light off-white to match screenshot)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FCFCFC")

# Status bar area (top strip where time/signal live)
STATUS_H = 72
draw.rectangle([(0, 0), (1440, STATUS_H)], fill="#EFEFEF")
# subtle bottom divider for status bar
draw.line([(24, STATUS_H), (1440-24, STATUS_H)], fill="#DFDFDF", width=1)

# Header / toolbar area (contains location + filter icon)
HEADER_TOP = STATUS_H
HEADER_BOTTOM = 220
draw.rectangle([(0, HEADER_TOP), (1440, HEADER_BOTTOM)], fill="#FFFFFF")
# header bottom divider / soft shadow
draw.line([(24, HEADER_BOTTOM), (1440-24, HEADER_BOTTOM)], fill="#E6E6E6", width=2)
# faint extra shadow line
draw.line([(24, HEADER_BOTTOM+2), (1440-24, HEADER_BOTTOM+2)], fill="#F5F5F5", width=1)

# "Recently viewed events" - three rounded image card backgrounds (placeholders behind images)
# Left card (clickable area)
rv_left_box = (48, 495, 48 + 462, 495 + 533)  # (x1,y1,x2,y2)
draw.rounded_rectangle(rv_left_box, radius=24, fill="#111217")  # dark placeholder for thumbnails
# Middle card
rv_mid_box = (546, 495, 546 + 462, 495 + 533)
draw.rounded_rectangle(rv_mid_box, radius=24, fill="#111217")
# Right card (slightly different height according to detections)
rv_right_box = (1044, 495, 1044 + 396, 495 + 519)
draw.rounded_rectangle(rv_right_box, radius=24, fill="#111217")

# Divider under recently viewed cards
rv_div_y = max(rv_left_box[3], rv_mid_box[3], rv_right_box[3]) + 20
draw.line([(24, rv_div_y), (1440 - 24, rv_div_y)], fill="#E9E9E9", width=1)

# Browse by category row - three dark rounded cards (backgrounds for category icons)
cat_y = 1263
cat_h = 312
cat_radius = 20
# Left category
cat_left = (48, cat_y, 48 + 462, cat_y + cat_h)
draw.rounded_rectangle(cat_left, radius=cat_radius, fill="#0B0B0C")
# Middle category
cat_mid = (546, cat_y, 546 + 462, cat_y + cat_h)
draw.rounded_rectangle(cat_mid, radius=cat_radius, fill="#0B0B0C")
# Right category
cat_right = (1044, cat_y, 1044 + 396, cat_y + cat_h)
draw.rounded_rectangle(cat_right, radius=cat_radius, fill="#0B0B0C")

# Light border around category cards to separate from white background
draw.rounded_rectangle(cat_left, radius=cat_radius, outline="#151515", width=1)
draw.rounded_rectangle(cat_mid, radius=cat_radius, outline="#151515", width=1)
draw.rounded_rectangle(cat_right, radius=cat_radius, outline="#151515", width=1)

# Divider under categories
cat_div_y = cat_y + cat_h + 28
draw.line([(24, cat_div_y), (1440 - 24, cat_div_y)], fill="#E9E9E9", width=1)

# "Just announced" section - small thumbnail background on left
# (Use a blue placeholder behind poster image)
ja_thumb = (48, 1834, 48 + 220, 1834 + 150)
draw.rounded_rectangle(ja_thumb, radius=16, fill="#2B66D6")
# subtle inner shadow/top highlight to make it read as a card
draw.line([(ja_thumb[0]+8, ja_thumb[1]+8), (ja_thumb[2]-8, ja_thumb[1]+8)], fill="#3b77e8", width=2)

# Divider under Just announced block
ja_div_y = 2050
draw.line([(24, ja_div_y), (1440 - 24, ja_div_y)], fill="#ECECEC", width=1)

# Sports / later content area separators
# thin separators to break sections visually
draw.line([(24, 2360), (1440 - 24, 2360)], fill="#F0F0F0", width=1)
draw.line([(24, 2580), (1440 - 24, 2580)], fill="#F0F0F0", width=1)

# Bottom navigation bar background and top divider
nav_top = 2792
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#FFFFFF")
draw.line([(0, nav_top), (1440, nav_top)], fill="#E6E6E6", width=2)
# slight top shadow to lift the nav bar
draw.line([(0, nav_top+2), (1440, nav_top+2)], fill="#F7F7F7", width=1)

# Subtle overall vignette/shadow under header and above major sections for depth
# (drawn as thin horizontal fades using multiple light lines)
for i, offset in enumerate([0, 2, 4]):
    alpha_shade = 230 - i * 30
    # can't use alpha directly on RGB canvas, so pick progressively lighter grays
    shade = int(240 - i * 6)
    shade_color = (shade, shade, shade)
    y = HEADER_BOTTOM + offset
    draw.line([(24, y), (1440 - 24, y)], fill=shade_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/00_icon_Sports.png
try:
    _c0 = get_crop(0, 462, 312)
    canvas.paste(_c0, (48, 1263), _c0)
except Exception:
    pass
layout["Sports"] = [48, 1263, 510, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/01_icon_Concerts.png
try:
    _c1 = get_crop(1, 462, 312)
    canvas.paste(_c1, (546, 1263), _c1)
except Exception:
    pass
layout["Concerts"] = [546, 1263, 1008, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/02_icon_Broadway.png
try:
    _c2 = get_crop(2, 396, 312)
    canvas.paste(_c2, (1044, 1263), _c2)
except Exception:
    pass
layout["Broadway"] = [1044, 1263, 1440, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/03_icon_E_Conf_Ist_Rnd.png
try:
    _c3 = get_crop(3, 462, 533)
    canvas.paste(_c3, (546, 495), _c3)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:"] = [546, 495, 1008, 1028]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/04_icon_Tickets.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (576, 2792), _c4)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/05_icon_888.png
try:
    _c5 = get_crop(5, 95, 60)
    canvas.paste(_c5, (1218, 3), _c5)
except Exception:
    pass
layout["888"] = [1218, 3, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/06_icon_7_10_W.png
try:
    _c6 = get_crop(6, 51, 54)
    canvas.paste(_c6, (117, 7), _c6)
except Exception:
    pass
layout["7:10_W"] = [117, 7, 168, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 48, 56)
    canvas.paste(_c7, (1322, 5), _c7)
except Exception:
    pass
layout["icon_7"] = [1322, 5, 1370, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/08_icon_Los_Angeles_CA.png
try:
    _c8 = get_crop(8, 60, 58)
    canvas.paste(_c8, (243, 4), _c8)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [243, 4, 303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/09_icon_Tracking.png
try:
    _c9 = get_crop(9, 288, 168)
    canvas.paste(_c9, (864, 2792), _c9)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/10_icon_MORM.png
try:
    _c10 = get_crop(10, 396, 519)
    canvas.paste(_c10, (1044, 495), _c10)
except Exception:
    pass
layout["MORM"] = [1044, 495, 1440, 1014]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/11_icon_7_10_W.png
try:
    _c11 = get_crop(11, 45, 56)
    canvas.paste(_c11, (187, 5), _c11)
except Exception:
    pass
layout["7:10_W"] = [187, 5, 232, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/12_icon_888.png
try:
    _c12 = get_crop(12, 144, 240)
    canvas.paste(_c12, (1260, 72), _c12)
except Exception:
    pass
layout["888"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 45, 60)
    canvas.paste(_c13, (1155, 4), _c13)
except Exception:
    pass
layout["icon_13"] = [1155, 4, 1200, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 55, 59)
    canvas.paste(_c14, (314, 4), _c14)
except Exception:
    pass
layout["icon_14"] = [314, 4, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/15_icon_Account.png
try:
    _c15 = get_crop(15, 288, 168)
    canvas.paste(_c15, (1152, 2792), _c15)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/16_icon_Search.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (288, 2792), _c16)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/17_text_Los_Angeles_CA.png
try:
    _c17 = get_crop(17, 458, 80)
    canvas.paste(_c17, (42, 132), _c17)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [42, 132, 500, 212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/18_text_Recently_viewed_events.png
try:
    _c18 = get_crop(18, 72, 72)
    canvas.paste(_c18, (408, 519), _c18)
except Exception:
    pass
layout["Recently_viewed_events"] = [408, 519, 480, 591]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/19_text_View_all.png
try:
    _c19 = get_crop(19, 264, 183)
    canvas.paste(_c19, (1176, 312), _c19)
except Exception:
    pass
layout["View_all"] = [1176, 312, 1440, 495]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/20_text_Browse_by_category.png
try:
    _c20 = get_crop(20, 462, 312)
    canvas.paste(_c20, (48, 1263), _c20)
except Exception:
    pass
layout["Browse_by_category"] = [48, 1263, 510, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/21_text_View_all.png
try:
    _c21 = get_crop(21, 264, 183)
    canvas.paste(_c21, (1176, 1080), _c21)
except Exception:
    pass
layout["View_all"] = [1176, 1080, 1440, 1263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/22_text_Just_announced.png
try:
    _c22 = get_crop(22, 72, 72)
    canvas.paste(_c22, (408, 1834), _c22)
except Exception:
    pass
layout["Just_announced"] = [408, 1834, 480, 1906]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/23_text_View_all.png
try:
    _c23 = get_crop(23, 264, 183)
    canvas.paste(_c23, (1176, 1627), _c23)
except Exception:
    pass
layout["View_all"] = [1176, 1627, 1440, 1810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/24_text_S52.png
try:
    _c24 = get_crop(24, 114, 52)
    canvas.paste(_c24, (95, 2037), _c24)
except Exception:
    pass
layout["S52+"] = [95, 2037, 209, 2089]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/25_text_Andrew_Schulz.png
try:
    _c25 = get_crop(25, 462, 519)
    canvas.paste(_c25, (48, 1810), _c25)
except Exception:
    pass
layout["Andrew_Schulz"] = [48, 1810, 510, 2329]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/26_text_Thu.png
try:
    _c26 = get_crop(26, 85, 45)
    canvas.paste(_c26, (45, 2235), _c26)
except Exception:
    pass
layout["Thu;"] = [45, 2235, 130, 2280]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/27_text_9_7.30_PM.png
try:
    _c27 = get_crop(27, 212, 45)
    canvas.paste(_c27, (235, 2233), _c27)
except Exception:
    pass
layout["9,7.30_PM"] = [235, 2233, 447, 2278]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/28_text_Sports.png
try:
    _c28 = get_crop(28, 179, 68)
    canvas.paste(_c28, (41, 2446), _c28)
except Exception:
    pass
layout["Sports"] = [41, 2446, 220, 2514]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/29_text_View_all.png
try:
    _c29 = get_crop(29, 264, 183)
    canvas.paste(_c29, (1176, 2381), _c29)
except Exception:
    pass
layout["View_all"] = [1176, 2381, 1440, 2564]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/30_clickable_Tracking.png
try:
    _c30 = get_crop(30, 462, 533)
    canvas.paste(_c30, (48, 495), _c30)
except Exception:
    pass
layout["Tracking"] = [48, 495, 510, 1028]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/31_clickable_Tracking.png
try:
    _c31 = get_crop(31, 72, 72)
    canvas.paste(_c31, (906, 519), _c31)
except Exception:
    pass
layout["Tracking"] = [906, 519, 978, 591]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/32_clickable_Tracking.png
try:
    _c32 = get_crop(32, 72, 72)
    canvas.paste(_c32, (408, 2588), _c32)
except Exception:
    pass
layout["Tracking"] = [408, 2588, 480, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/33_clickable_Tracking.png
try:
    _c33 = get_crop(33, 72, 72)
    canvas.paste(_c33, (906, 2588), _c33)
except Exception:
    pass
layout["Tracking"] = [906, 2588, 978, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/3297e3a54de8487d8c2b67798919ed2f/step_07_2024_4_22_19_9_3297e3a54de8487d8c2b67798919ed2f-10/34_clickable_Browse.png
try:
    _c34 = get_crop(34, 288, 162)
    canvas.paste(_c34, (0, 2792), _c34)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]
