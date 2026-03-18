# page_id: page_seatgeek_68e3462c14734440a7ace3fed432a10d_07
# screenshot: 2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10.png
# step_index: 7/13
# task: Open SeatGeek and change the current location to Los Angeles. Then find the first concert show and track its performer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for SeatGeek-like page
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
status_bar_color = "#f3f4f5"   # light gray status bar
header_bg = "#ffffff"          # header white
page_bg = "#ffffff"            # main background
divider = "#e9eaeb"            # subtle divider lines
card_bg = "#ffffff"            # card background (white)
card_shadow = "#e6e6e6"        # shadow for cards
muted_bg = "#fafafa"           # slightly off-white section background
category_card_bg = "#0c0c0c"   # dark background used for category tiles
bottom_nav_bg = "#ffffff"      # bottom nav background
top_strip_shadow = "#e8e8e8"

# Helper: rounded rect
def rr(xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)

# 1) Status bar background (top ~72px)
status_h = 72
draw.rectangle([0, 0, w, status_h], fill=status_bar_color)

# subtle top divider/shadow under status bar
draw.line([(0, status_h), (w, status_h)], fill=top_strip_shadow, width=1)

# 2) Header / toolbar area
header_top = status_h
header_bottom = 176
draw.rectangle([0, header_top, w, header_bottom], fill=header_bg)

# subtle bottom border under header
draw.line([(24, header_bottom), (w-24, header_bottom)], fill=divider, width=1)

# 3) Main content background (slight off-white bands for sections)
content_top = header_bottom
draw.rectangle([0, content_top, w, h - 200], fill=page_bg)

# 4) Recently viewed section cards (rounded white cards with light shadow)
# Left card
rv_left = (48, 495, 48+462, 495+533)
# card shadow
draw.rounded_rectangle([rv_left[0]+4, rv_left[1]+6, rv_left[2]+4, rv_left[3]+6],
                       radius=18, fill=card_shadow)
# card background
draw.rounded_rectangle([rv_left[0], rv_left[1], rv_left[2], rv_left[3]],
                       radius=18, fill=card_bg)

# Middle card
rv_mid = (546, 495, 546+462, 495+533)
draw.rounded_rectangle([rv_mid[0]+4, rv_mid[1]+6, rv_mid[2]+4, rv_mid[3]+6],
                       radius=18, fill=card_shadow)
draw.rounded_rectangle([rv_mid[0], rv_mid[1], rv_mid[2], rv_mid[3]],
                       radius=18, fill=card_bg)

# Right card
rv_right = (1044, 495, 1044+396, 495+519)
draw.rounded_rectangle([rv_right[0]+4, rv_right[1]+6, rv_right[2]+4, rv_right[3]+6],
                       radius=18, fill=card_shadow)
draw.rounded_rectangle([rv_right[0], rv_right[1], rv_right[2], rv_right[3]],
                       radius=18, fill=card_bg)

# Divider under Recently viewed section
sep_y_1 = 1080
draw.line([(24, sep_y_1), (w-24, sep_y_1)], fill=divider, width=1)

# 5) "Browse by category" tiles - dark rounded rectangles
cat_y = 1263
cat_h = 312
# Left category
cat_left = (48, cat_y, 48+462, cat_y+cat_h)
draw.rounded_rectangle([cat_left[0], cat_left[1], cat_left[2], cat_left[3]],
                       radius=20, fill=category_card_bg)
# Middle category
cat_mid = (546, cat_y, 546+462, cat_y+cat_h)
draw.rounded_rectangle([cat_mid[0], cat_mid[1], cat_mid[2], cat_mid[3]],
                       radius=20, fill=category_card_bg)
# Right category
cat_right = (1044, cat_y, 1044+396, cat_y+cat_h)
draw.rounded_rectangle([cat_right[0], cat_right[1], cat_right[2], cat_right[3]],
                       radius=20, fill=category_card_bg)

# subtle divider under browse categories
sep_y_2 = 1627
draw.line([(24, sep_y_2), (w-24, sep_y_2)], fill=divider, width=1)

# 6) Just announced section card (single small rounded card)
ja_card = (48, 1810, 48+462, 1810+519)
draw.rounded_rectangle([ja_card[0]+4, ja_card[1]+6, ja_card[2]+4, ja_card[3]+6],
                       radius=18, fill=card_shadow)
draw.rounded_rectangle([ja_card[0], ja_card[1], ja_card[2], ja_card[3]],
                       radius=18, fill=card_bg)

# Divider under Just announced
sep_y_3 = 2060
draw.line([(24, sep_y_3), (w-24, sep_y_3)], fill=divider, width=1)

# 7) Sports horizontal strip area (placeholder background behind upcoming horizontal list)
sports_strip_y = 2080
sports_strip_h = 420
draw.rectangle([0, sports_strip_y, w, sports_strip_y + sports_strip_h], fill=page_bg)

# Draw faint left & right margins' carved separators for horizontal scroller
draw.line([(24, sports_strip_y+10), (24, sports_strip_y + sports_strip_h - 10)], fill=muted_bg, width=1)
draw.line([(w-24, sports_strip_y+10), (w-24, sports_strip_y + sports_strip_h - 10)], fill=muted_bg, width=1)

# 8) Bottom navigation bar background and top divider
nav_top = 2792
nav_h = h - nav_top
draw.rectangle([0, nav_top, w, h], fill=bottom_nav_bg)
draw.line([(0, nav_top), (w, nav_top)], fill=divider, width=1)

# 9) Floating subtle shadow line above nav for depth
draw.line([(24, nav_top+2), (w-24, nav_top+2)], fill=card_shadow, width=1)

# 10) Light vertical guide lines at edges of content to match layout margins
margin_x = 24
draw.line([(margin_x, header_bottom), (margin_x, h - nav_h - 1)], fill=muted_bg, width=1)
draw.line([(w - margin_x, header_bottom), (w - margin_x, h - nav_h - 1)], fill=muted_bg, width=1)

# 11) Small pill-shaped search/filter background in header (only background shape, no icons/text)
pill_x0 = w - 120
pill_y0 = header_top + 30
pill_w = 88
pill_h = 48
draw.rounded_rectangle([pill_x0, pill_y0, pill_x0 + pill_w, pill_y0 + pill_h],
                       radius=12, fill="#ffffff", outline=divider)

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/00_icon_Sports.png
try:
    _c0 = get_crop(0, 462, 312)
    canvas.paste(_c0, (48, 1263), _c0)
except Exception:
    pass
layout["Sports"] = [48, 1263, 510, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/01_icon_Concerts.png
try:
    _c1 = get_crop(1, 462, 312)
    canvas.paste(_c1, (546, 1263), _c1)
except Exception:
    pass
layout["Concerts"] = [546, 1263, 1008, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/02_icon_Broadway.png
try:
    _c2 = get_crop(2, 396, 312)
    canvas.paste(_c2, (1044, 1263), _c2)
except Exception:
    pass
layout["Broadway"] = [1044, 1263, 1440, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/03_icon_Tickets.png
try:
    _c3 = get_crop(3, 288, 168)
    canvas.paste(_c3, (576, 2792), _c3)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/04_icon_Drake_Rescheduled.png
try:
    _c4 = get_crop(4, 462, 533)
    canvas.paste(_c4, (546, 495), _c4)
except Exception:
    pass
layout["Drake_(Rescheduled"] = [546, 495, 1008, 1028]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/05_icon_888.png
try:
    _c5 = get_crop(5, 97, 59)
    canvas.paste(_c5, (1217, 2), _c5)
except Exception:
    pass
layout["888"] = [1217, 2, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/06_icon_8.30_my.png
try:
    _c6 = get_crop(6, 52, 57)
    canvas.paste(_c6, (116, 4), _c6)
except Exception:
    pass
layout["8.30_my"] = [116, 4, 168, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 49, 55)
    canvas.paste(_c7, (1321, 5), _c7)
except Exception:
    pass
layout["icon_7"] = [1321, 5, 1370, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/08_icon_8.30_my.png
try:
    _c8 = get_crop(8, 55, 58)
    canvas.paste(_c8, (182, 4), _c8)
except Exception:
    pass
layout["8.30_my"] = [182, 4, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/09_icon_888.png
try:
    _c9 = get_crop(9, 144, 240)
    canvas.paste(_c9, (1260, 72), _c9)
except Exception:
    pass
layout["888"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/10_icon_Tracking.png
try:
    _c10 = get_crop(10, 288, 168)
    canvas.paste(_c10, (864, 2792), _c10)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 47, 61)
    canvas.paste(_c11, (1154, 3), _c11)
except Exception:
    pass
layout["icon_11"] = [1154, 3, 1201, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/12_icon_Sports.png
try:
    _c12 = get_crop(12, 288, 162)
    canvas.paste(_c12, (0, 2792), _c12)
except Exception:
    pass
layout["Sports"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/13_icon_Account.png
try:
    _c13 = get_crop(13, 288, 168)
    canvas.paste(_c13, (1152, 2792), _c13)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 51, 55)
    canvas.paste(_c14, (316, 6), _c14)
except Exception:
    pass
layout["icon_14"] = [316, 6, 367, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/15_icon_Search.png
try:
    _c15 = get_crop(15, 288, 168)
    canvas.paste(_c15, (288, 2792), _c15)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 49, 57)
    canvas.paste(_c16, (383, 4), _c16)
except Exception:
    pass
layout["icon_16"] = [383, 4, 432, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/17_icon_Los_Angeles_CA.png
try:
    _c17 = get_crop(17, 57, 58)
    canvas.paste(_c17, (246, 4), _c17)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [246, 4, 303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/18_icon_S273.png
try:
    _c18 = get_crop(18, 462, 533)
    canvas.paste(_c18, (546, 495), _c18)
except Exception:
    pass
layout["S273+"] = [546, 495, 1008, 1028]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/19_icon_Browse.png
try:
    _c19 = get_crop(19, 288, 162)
    canvas.paste(_c19, (0, 2792), _c19)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/20_text_Los_Angeles_CA.png
try:
    _c20 = get_crop(20, 459, 80)
    canvas.paste(_c20, (44, 132), _c20)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [44, 132, 503, 212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/21_text_Recently_viewed_events.png
try:
    _c21 = get_crop(21, 72, 72)
    canvas.paste(_c21, (408, 519), _c21)
except Exception:
    pass
layout["Recently_viewed_events"] = [408, 519, 480, 591]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/22_text_View_all.png
try:
    _c22 = get_crop(22, 264, 183)
    canvas.paste(_c22, (1176, 312), _c22)
except Exception:
    pass
layout["View_all"] = [1176, 312, 1440, 495]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/23_text_Browse_by_category.png
try:
    _c23 = get_crop(23, 462, 312)
    canvas.paste(_c23, (48, 1263), _c23)
except Exception:
    pass
layout["Browse_by_category"] = [48, 1263, 510, 1575]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/24_text_View_all.png
try:
    _c24 = get_crop(24, 264, 183)
    canvas.paste(_c24, (1176, 1080), _c24)
except Exception:
    pass
layout["View_all"] = [1176, 1080, 1440, 1263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/25_text_Just_announced.png
try:
    _c25 = get_crop(25, 72, 72)
    canvas.paste(_c25, (408, 1834), _c25)
except Exception:
    pass
layout["Just_announced"] = [408, 1834, 480, 1906]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/26_text_View_all.png
try:
    _c26 = get_crop(26, 264, 183)
    canvas.paste(_c26, (1176, 1627), _c26)
except Exception:
    pass
layout["View_all"] = [1176, 1627, 1440, 1810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/27_text_S46.png
try:
    _c27 = get_crop(27, 119, 52)
    canvas.paste(_c27, (95, 2037), _c27)
except Exception:
    pass
layout["S46+"] = [95, 2037, 214, 2089]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/28_text_Andrew_Schulz.png
try:
    _c28 = get_crop(28, 462, 519)
    canvas.paste(_c28, (48, 1810), _c28)
except Exception:
    pass
layout["Andrew_Schulz"] = [48, 1810, 510, 2329]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/29_text_Thu.png
try:
    _c29 = get_crop(29, 85, 45)
    canvas.paste(_c29, (45, 2235), _c29)
except Exception:
    pass
layout["Thu;"] = [45, 2235, 130, 2280]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/30_text_9_7.30_PM.png
try:
    _c30 = get_crop(30, 212, 45)
    canvas.paste(_c30, (235, 2233), _c30)
except Exception:
    pass
layout["9,7.30_PM"] = [235, 2233, 447, 2278]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/31_text_Sports.png
try:
    _c31 = get_crop(31, 179, 68)
    canvas.paste(_c31, (41, 2446), _c31)
except Exception:
    pass
layout["Sports"] = [41, 2446, 220, 2514]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/32_text_View_all.png
try:
    _c32 = get_crop(32, 264, 183)
    canvas.paste(_c32, (1176, 2381), _c32)
except Exception:
    pass
layout["View_all"] = [1176, 2381, 1440, 2564]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/33_clickable_Tracking.png
try:
    _c33 = get_crop(33, 462, 519)
    canvas.paste(_c33, (48, 495), _c33)
except Exception:
    pass
layout["Tracking"] = [48, 495, 510, 1014]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/34_clickable_Tracking.png
try:
    _c34 = get_crop(34, 396, 519)
    canvas.paste(_c34, (1044, 495), _c34)
except Exception:
    pass
layout["Tracking"] = [1044, 495, 1440, 1014]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/35_clickable_Tracking.png
try:
    _c35 = get_crop(35, 72, 72)
    canvas.paste(_c35, (906, 519), _c35)
except Exception:
    pass
layout["Tracking"] = [906, 519, 978, 591]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/36_clickable_Tracking.png
try:
    _c36 = get_crop(36, 72, 72)
    canvas.paste(_c36, (408, 2588), _c36)
except Exception:
    pass
layout["Tracking"] = [408, 2588, 480, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_07_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-10/37_clickable_Tracking.png
try:
    _c37 = get_crop(37, 72, 72)
    canvas.paste(_c37, (906, 2588), _c37)
except Exception:
    pass
layout["Tracking"] = [906, 2588, 978, 2660]
