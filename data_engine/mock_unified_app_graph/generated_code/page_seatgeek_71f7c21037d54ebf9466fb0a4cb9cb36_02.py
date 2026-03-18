# page_id: page_seatgeek_71f7c21037d54ebf9466fb0a4cb9cb36_02
# screenshot: 2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5.png
# step_index: 2/4
# task: Open SeatGeek. Search for concerts in "New York City". Filter by "pop" genre. What is the second recommendation?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint overall background
draw.rectangle([(0, 0), (1440, 2960)], fill="#F7F7F7")

# Status bar area (top ~80px) - subtle gray
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill="#EFEFEF")

# Header / toolbar background (white) with subtle bottom divider/shadow
header_top = status_h
header_bottom = 160
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
draw.line([(24, header_bottom), (1440 - 24, header_bottom)], fill="#E6E6E6", width=1)

# Main list card background (rounded white card behind top list items)
card_x0, card_y0 = 48, 180
card_x1, card_y1 = 1440 - 48, 900
draw.rounded_rectangle([(card_x0, card_y0), (card_x1, card_y1)], radius=14, fill="#FFFFFF")

# Dividers between list rows inside the card (keep left padding so badges can overlay)
sep_x0, sep_x1 = card_x0 + 16, card_x1 - 16
sep_color = "#EDEDED"
# approximate three rows separators
sep_ys = [card_y0 + 200, card_y0 + 400, card_y0 + 600]
for y in sep_ys:
    draw.line([(sep_x0, y), (sep_x1, y)], fill=sep_color, width=1)

# Recently viewed section background (white band) with top divider
recent_top = card_y1 + 30
recent_bottom = 1400
draw.rectangle([(0, recent_top), (1440, recent_bottom)], fill="#FFFFFF")
draw.line([(24, recent_top), (1440 - 24, recent_top)], fill="#E6E6E6", width=1)

# Recently viewed thumbnails (rounded rect backgrounds) - leave content to be pasted
thumbs = [
    (48, 1283, 48 + 462, 1283 + 533),   # left thumbnail
    (546, 1283, 546 + 462, 1283 + 519), # middle thumbnail
    (1044, 1283, 1044 + 396, 1283 + 533) # right thumbnail
]
for x0, y0, x1, y1 in thumbs:
    draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=20, fill="#F2F2F2")
    # subtle inner border to define thumbnail area
    draw.rounded_rectangle([(x0+2, y0+2), (x1-2, y1-2)], radius=18, outline="#E7E7E7", width=1)

# Separator line between Recently viewed and Browse by category
browse_top = 1860
draw.line([(24, browse_top), (1440 - 24, browse_top)], fill="#E6E6E6", width=1)

# "Browse by category" tiles (dark rounded rectangles as backgrounds)
category_tiles = [
    (48, 2051, 48 + 462, 2051 + 312),   # Sports
    (546, 2051, 546 + 462, 2051 + 312), # Concerts
    (1044, 2051, 1044 + 396, 2051 + 312) # Broadway
]
for x0, y0, x1, y1 in category_tiles:
    draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=22, fill="#0A0A0A")
    # very subtle inner highlight at top to mimic slight sheen
    highlight_h = 12
    draw.rectangle([(x0+6, y0+6), (x1-6, y0+6+highlight_h)], fill=(255,255,255,10))

# Divider above Just announced section
just_announced_top = 2380
draw.line([(24, just_announced_top), (1440 - 24, just_announced_top)], fill="#E6E6E6", width=1)

# Just announced thumbnails (light rounded backgrounds) - placeholders only
ja_thumbs = [
    (48, 2460, 48 + 420, 2460 + 240),
    (540, 2460, 540 + 420, 2460 + 240),
    (1040, 2460, 1040 + 360, 2460 + 240)
]
for x0, y0, x1, y1 in ja_thumbs:
    draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=14, fill="#F6F6F6")
    draw.rounded_rectangle([(x0+2, y0+2), (x1-2, y1-2)], radius=12, outline="#ECECEC", width=1)

# Bottom navigation bar background and top divider
nav_top = 2792
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#FFFFFF")
draw.line([(24, nav_top), (1440 - 24, nav_top)], fill="#E6E6E6", width=1)

# Small shadow under major cards to give slight elevation (thin translucent lines)
# simulated with slightly darker thin lines
draw.line([(card_x0+6, card_y1+2), (card_x1-6, card_y1+2)], fill="#F0F0F0", width=2)
draw.line([(48+6, recent_bottom+2), (1440-48-6, recent_bottom+2)], fill="#F0F0F0", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/00_icon_Broadway.png
try:
    _c0 = get_crop(0, 396, 312)
    canvas.paste(_c0, (1044, 2051), _c0)
except Exception:
    pass
layout["Broadway"] = [1044, 2051, 1440, 2363]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/01_icon_Concerts.png
try:
    _c1 = get_crop(1, 462, 312)
    canvas.paste(_c1, (546, 2051), _c1)
except Exception:
    pass
layout["Concerts"] = [546, 2051, 1008, 2363]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/02_icon_Yankee_Stadium.png
try:
    _c2 = get_crop(2, 1309, 236)
    canvas.paste(_c2, (0, 568), _c2)
except Exception:
    pass
layout["Yankee_Stadium"] = [0, 568, 1309, 804]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/03_icon_Sports.png
try:
    _c3 = get_crop(3, 462, 312)
    canvas.paste(_c3, (48, 2051), _c3)
except Exception:
    pass
layout["Sports"] = [48, 2051, 510, 2363]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/04_icon_S252.png
try:
    _c4 = get_crop(4, 396, 533)
    canvas.paste(_c4, (1044, 1283), _c4)
except Exception:
    pass
layout["S252+"] = [1044, 1283, 1440, 1816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/05_icon_View_all.png
try:
    _c5 = get_crop(5, 105, 147)
    canvas.paste(_c5, (1335, 850), _c5)
except Exception:
    pass
layout["View_all"] = [1335, 850, 1440, 997]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/06_icon_062.png
try:
    _c6 = get_crop(6, 101, 149)
    canvas.paste(_c6, (1339, 612), _c6)
except Exception:
    pass
layout["062"] = [1339, 612, 1440, 761]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/07_icon_884.png
try:
    _c7 = get_crop(7, 99, 149)
    canvas.paste(_c7, (1341, 377), _c7)
except Exception:
    pass
layout["884"] = [1341, 377, 1440, 526]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/08_icon_The_Book_of_Mormon.png
try:
    _c8 = get_crop(8, 462, 533)
    canvas.paste(_c8, (48, 1283), _c8)
except Exception:
    pass
layout["The_Book_of_Mormon"] = [48, 1283, 510, 1816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/09_icon_Tickets.png
try:
    _c9 = get_crop(9, 288, 168)
    canvas.paste(_c9, (576, 2792), _c9)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/10_icon_New_York_NY.png
try:
    _c10 = get_crop(10, 61, 56)
    canvas.paste(_c10, (243, 6), _c10)
except Exception:
    pass
layout["New_York,_NY"] = [243, 6, 304, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/11_icon_August_Wilson_Theatre.png
try:
    _c11 = get_crop(11, 1309, 236)
    canvas.paste(_c11, (0, 332), _c11)
except Exception:
    pass
layout["August_Wilson_Theatre"] = [0, 332, 1309, 568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/12_icon_7_03_my.png
try:
    _c12 = get_crop(12, 54, 53)
    canvas.paste(_c12, (115, 8), _c12)
except Exception:
    pass
layout["7:03_my"] = [115, 8, 169, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 56, 56)
    canvas.paste(_c13, (313, 7), _c13)
except Exception:
    pass
layout["icon_13"] = [313, 7, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/14_icon_7_03_my.png
try:
    _c14 = get_crop(14, 46, 54)
    canvas.paste(_c14, (186, 7), _c14)
except Exception:
    pass
layout["7:03_my"] = [186, 7, 232, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 49, 56)
    canvas.paste(_c15, (1321, 5), _c15)
except Exception:
    pass
layout["icon_15"] = [1321, 5, 1370, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/16_icon_884.png
try:
    _c16 = get_crop(16, 97, 61)
    canvas.paste(_c16, (1216, 2), _c16)
except Exception:
    pass
layout["884"] = [1216, 2, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 46, 64)
    canvas.paste(_c17, (1155, 2), _c17)
except Exception:
    pass
layout["icon_17"] = [1155, 2, 1201, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/18_icon_THE.png
try:
    _c18 = get_crop(18, 288, 162)
    canvas.paste(_c18, (0, 2792), _c18)
except Exception:
    pass
layout["THE"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/19_icon_884.png
try:
    _c19 = get_crop(19, 144, 240)
    canvas.paste(_c19, (1260, 72), _c19)
except Exception:
    pass
layout["884"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/20_icon_Tracking.png
try:
    _c20 = get_crop(20, 288, 168)
    canvas.paste(_c20, (864, 2792), _c20)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/21_icon_BOOK_OF.png
try:
    _c21 = get_crop(21, 462, 519)
    canvas.paste(_c21, (546, 1283), _c21)
except Exception:
    pass
layout["BOOK_OF"] = [546, 1283, 1008, 1802]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/22_icon_BOOK_OF.png
try:
    _c22 = get_crop(22, 288, 162)
    canvas.paste(_c22, (0, 2792), _c22)
except Exception:
    pass
layout["BOOK_OF"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/23_icon_View_all.png
try:
    _c23 = get_crop(23, 50, 82)
    canvas.paste(_c23, (1390, 1309), _c23)
except Exception:
    pass
layout["View_all"] = [1390, 1309, 1440, 1391]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/24_icon_062.png
try:
    _c24 = get_crop(24, 117, 129)
    canvas.paste(_c24, (1137, 632), _c24)
except Exception:
    pass
layout["062"] = [1137, 632, 1254, 761]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/25_icon_Account.png
try:
    _c25 = get_crop(25, 288, 168)
    canvas.paste(_c25, (1152, 2792), _c25)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/26_icon_MORM.png
try:
    _c26 = get_crop(26, 288, 168)
    canvas.paste(_c26, (288, 2792), _c26)
except Exception:
    pass
layout["MORM"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/27_text_New_York_NY.png
try:
    _c27 = get_crop(27, 382, 68)
    canvas.paste(_c27, (48, 133), _c27)
except Exception:
    pass
layout["New_York,_NY"] = [48, 133, 430, 201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/28_text_date.png
try:
    _c28 = get_crop(28, 117, 52)
    canvas.paste(_c28, (134, 208), _c28)
except Exception:
    pass
layout["date"] = [134, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/29_text_E_Conf_Ist_Rnd_76ers_at_Knicks_Gm_2_H.png
try:
    _c29 = get_crop(29, 1309, 234)
    canvas.paste(_c29, (0, 804), _c29)
except Exception:
    pass
layout["E_Conf_Ist_Rnd:_76ers_at_"] = [0, 804, 1309, 1038]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/30_text_Apr_22.png
try:
    _c30 = get_crop(30, 159, 62)
    canvas.paste(_c30, (225, 931), _c30)
except Exception:
    pass
layout["Apr_22"] = [225, 931, 384, 993]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/31_text_Madison_Square_Garden.png
try:
    _c31 = get_crop(31, 1309, 234)
    canvas.paste(_c31, (0, 804), _c31)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 804, 1309, 1038]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/32_text_Recently_viewed_events.png
try:
    _c32 = get_crop(32, 72, 72)
    canvas.paste(_c32, (408, 1307), _c32)
except Exception:
    pass
layout["Recently_viewed_events"] = [408, 1307, 480, 1379]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/33_text_View_all.png
try:
    _c33 = get_crop(33, 264, 183)
    canvas.paste(_c33, (1176, 1100), _c33)
except Exception:
    pass
layout["View_all"] = [1176, 1100, 1440, 1283]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/34_text_Browse_by_category.png
try:
    _c34 = get_crop(34, 462, 312)
    canvas.paste(_c34, (48, 2051), _c34)
except Exception:
    pass
layout["Browse_by_category"] = [48, 2051, 510, 2363]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/35_text_View_all.png
try:
    _c35 = get_crop(35, 264, 183)
    canvas.paste(_c35, (1176, 1868), _c35)
except Exception:
    pass
layout["View_all"] = [1176, 1868, 1440, 2051]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/36_text_Just_announced.png
try:
    _c36 = get_crop(36, 72, 72)
    canvas.paste(_c36, (408, 2622), _c36)
except Exception:
    pass
layout["Just_announced"] = [408, 2622, 480, 2694]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/37_text_View_all.png
try:
    _c37 = get_crop(37, 264, 183)
    canvas.paste(_c37, (1176, 2415), _c37)
except Exception:
    pass
layout["View_all"] = [1176, 2415, 1440, 2598]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/38_clickable_Tracking.png
try:
    _c38 = get_crop(38, 72, 72)
    canvas.paste(_c38, (906, 1307), _c38)
except Exception:
    pass
layout["Tracking"] = [906, 1307, 978, 1379]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/71f7c21037d54ebf9466fb0a4cb9cb36/step_02_2024_4_22_19_2_71f7c21037d54ebf9466fb0a4cb9cb36-5/39_clickable_Tracking.png
try:
    _c39 = get_crop(39, 72, 72)
    canvas.paste(_c39, (906, 2622), _c39)
except Exception:
    pass
layout["Tracking"] = [906, 2622, 978, 2694]
