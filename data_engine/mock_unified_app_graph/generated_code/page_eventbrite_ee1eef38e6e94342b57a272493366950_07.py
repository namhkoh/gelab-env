# page_id: page_eventbrite_ee1eef38e6e94342b57a272493366950_07
# screenshot: 2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9.png
# step_index: 7/10
# task: Open Eventbrite. Open "Fashion" category. Apply filter for free events. From the list, select the first non-promoted event and add it to your favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background & structural UI elements for the mobile Eventbrite-like page.
# Assumes: canvas (1440x2960 RGB PIL Image) and draw (ImageDraw) and fonts are available.

# Colors
status_bar_color = (196, 196, 196)      # light gray status bar
page_bg = (255, 255, 255)               # white page bg (canvas already white)
search_bg = (247, 249, 252)             # very light bluish/gray for search field
search_border = (225, 227, 233)         # subtle border under search
chip_bg = (236, 246, 255)               # pale chip background (not drawing chips themselves)
divider_color = (226, 226, 230)         # thin separators
card_shadow = (240, 241, 245)           # subtle card shadow
card_bg = (255, 255, 255)               # card fill (white)
bottom_nav_bg = (250, 250, 250)         # bottom nav background
bottom_nav_border = (210, 210, 210)     # top border for bottom nav

W, H = canvas.size

# Helper: safe rounded rectangle drawing
def rrect(rect, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline, width=width)

# 1) Status bar area (approx top 56px)
draw.rectangle([(0,0),(W,56)], fill=status_bar_color)

# subtle inner line under status bar
draw.line([(0,56),(W,56)], fill=divider_color, width=1)

# 2) Search bar background (rounded) - sits under status bar
search_rect = (40, 68, W-40, 160)
rrect(search_rect, radius=14, fill=search_bg, outline=search_border, width=2)

# subtle thin divider line below search area
draw.line([(40, 164),(W-40, 164)], fill=divider_color, width=1)

# 3) Filter/controls row separator (a light horizontal gap area)
# Draw a faint horizontal rule below chips area to separate header from list
draw.line([(24, 420),(W-24, 420)], fill=divider_color, width=1)

# 4) Event list card backgrounds (two stacked cards)
# Draw subtle drop shadows then white rounded card backgrounds.
# Card 1 (first event list item)
card1_shadow = (36, 480, W-36, 1060)
card1_rect = (30, 476, W-30, 1056)
# shadow
draw.rectangle([(card1_shadow[0]+4, card1_shadow[1]+6),(card1_shadow[2]+4, card1_shadow[3]+6)], fill=card_shadow)
# card background
rrect(card1_rect, radius=18, fill=card_bg, outline=None)

# small divider under first card to separate from next section
draw.line([(36, 1068),(W-36, 1068)], fill=divider_color, width=1)

# Card 2 (second event list item)
card2_shadow = (36, 1120, W-36, 1900)
card2_rect = (30, 1116, W-30, 1896)
draw.rectangle([(card2_shadow[0]+4, card2_shadow[1]+6),(card2_shadow[2]+4, card2_shadow[3]+6)], fill=card_shadow)
rrect(card2_rect, radius=18, fill=card_bg, outline=None)

# 5) Thin separators between content regions
draw.line([(36, 1908),(W-36, 1908)], fill=divider_color, width=1)
draw.line([(36, 2388),(W-36, 2388)], fill=divider_color, width=1)

# 6) Content image backgrounds inside cards (subtle darker area behind images)
# These are non-textural placeholders for image areas (images/icons will be pasted on top).
# First image block (top portion of card1)
img1_rect = (48, 520, W-48, 820)
rrect(img1_rect, radius=8, fill=(245,245,248), outline=None)

# Second image block (top portion of card2)
img2_rect = (48, 1160, W-48, 1460)
rrect(img2_rect, radius=8, fill=(245,245,248), outline=None)

# 7) "Free" tag background placeholders near event titles (light pill shapes).
# Draw only faint pill shapes (no text).
pill1 = (48, 840, 120, 880)
draw.rounded_rectangle(pill1, radius=10, fill=(226,240,232), outline=None)
pill2 = (48, 1484, 120, 1524)
draw.rounded_rectangle(pill2, radius=10, fill=(226,240,232), outline=None)

# 8) Bottom navigation bar
nav_h = 120
nav_top = H - nav_h
draw.rectangle([(0, nav_top), (W, H)], fill=bottom_nav_bg)
# top border for nav
draw.line([(0, nav_top), (W, nav_top)], fill=bottom_nav_border, width=1)

# lightly indicate 5 nav slots as faint circles (no icons/text)
slot_w = W // 5
for i in range(5):
    cx = slot_w * i + slot_w // 2
    cy = nav_top + nav_h // 2
    r = 22
    draw.ellipse([(cx-r, cy-r),(cx+r, cy+r)], outline=(220,220,220), width=2, fill=None)

# 9) Final subtle overall vignette/top shadow under header to give depth
draw.rectangle([(0, 156),(W,160)], fill=(250,250,250))

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/00_icon_Anytime.png
try:
    _c0 = get_crop(0, 400, 135)
    canvas.paste(_c0, (458, 390), _c0)
except Exception:
    pass
layout["Anytime"] = [458, 390, 858, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/01_icon_Fashion.png
try:
    _c1 = get_crop(1, 220, 135)
    canvas.paste(_c1, (870, 390), _c1)
except Exception:
    pass
layout["Fashion"] = [870, 390, 1090, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/02_icon_2_Filters.png
try:
    _c2 = get_crop(2, 392, 135)
    canvas.paste(_c2, (54, 390), _c2)
except Exception:
    pass
layout["2_Filters"] = [54, 390, 446, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/03_icon_Bugston.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 2336), _c3)
except Exception:
    pass
layout["Bugston"] = [1092, 2336, 1236, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/04_icon_Bugston.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1236, 2336), _c4)
except Exception:
    pass
layout["Bugston"] = [1236, 2336, 1380, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/05_icon_5.28.png
try:
    _c5 = get_crop(5, 123, 113)
    canvas.paste(_c5, (56, 114), _c5)
except Exception:
    pass
layout["5.28"] = [56, 114, 179, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1192), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/07_icon_5.28.png
try:
    _c7 = get_crop(7, 61, 64)
    canvas.paste(_c7, (180, 0), _c7)
except Exception:
    pass
layout["5.28"] = [180, 0, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/09_icon_Search_forae.png
try:
    _c9 = get_crop(9, 69, 63)
    canvas.paste(_c9, (307, 0), _c9)
except Exception:
    pass
layout["Search_forae"] = [307, 0, 376, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 55, 64)
    canvas.paste(_c10, (246, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [246, 0, 301, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/11_icon_5.28.png
try:
    _c11 = get_crop(11, 61, 66)
    canvas.paste(_c11, (114, 0), _c11)
except Exception:
    pass
layout["5.28"] = [114, 0, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 63, 59)
    canvas.paste(_c12, (1317, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1317, 0, 1380, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/13_icon_Recognise_Respond_Refer.png
try:
    _c13 = get_crop(13, 1344, 996)
    canvas.paste(_c13, (48, 1820), _c13)
except Exception:
    pass
layout["Recognise,_Respond_&_Refe"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 78, 61)
    canvas.paste(_c14, (1207, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1207, 0, 1285, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/15_icon_PROTECT_YOUR_BUSINESS_AND_FAMILY.png
try:
    _c15 = get_crop(15, 1344, 1096)
    canvas.paste(_c15, (48, 676), _c15)
except Exception:
    pass
layout["PROTECT_YOUR_BUSINESS_AND"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/16_icon_Online.png
try:
    _c16 = get_crop(16, 377, 144)
    canvas.paste(_c16, (0, 259), _c16)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/17_icon_Search_forae.png
try:
    _c17 = get_crop(17, 1344, 191)
    canvas.paste(_c17, (48, 72), _c17)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/18_icon_Sun_Mav_5_._8.00_PM_EDT.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (288, 2804), _c18)
except Exception:
    pass
layout["Sun,_Mav_5_._8.00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/19_icon_Search_forae.png
try:
    _c19 = get_crop(19, 51, 61)
    canvas.paste(_c19, (383, 2), _c19)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/20_icon_Recognise_Respond_Refer.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (864, 2804), _c20)
except Exception:
    pass
layout["Recognise,_Respond_&_Refe"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 61, 55)
    canvas.paste(_c21, (37, 821), _c21)
except Exception:
    pass
layout["icon_21"] = [37, 821, 98, 876]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/22_icon_Online_Professional_Development.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (576, 2804), _c22)
except Exception:
    pass
layout["Online_Professional_Devel"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 40, 61)
    canvas.paste(_c23, (1274, 0), _c23)
except Exception:
    pass
layout["icon_23"] = [1274, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/24_icon_Sun_Mav_5_._8.00_PM_EDT.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (0, 2804), _c24)
except Exception:
    pass
layout["Sun,_Mav_5_._8.00_PM_EDT"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/25_icon_Recognise_Respond_Refer.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (1152, 2804), _c25)
except Exception:
    pass
layout["Recognise,_Respond_&_Refe"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/26_icon_Promoted.png
try:
    _c26 = get_crop(26, 251, 66)
    canvas.paste(_c26, (80, 1664), _c26)
except Exception:
    pass
layout["Promoted"] = [80, 1664, 331, 1730]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/27_icon_Pow.png
try:
    _c27 = get_crop(27, 48, 186)
    canvas.paste(_c27, (1352, 683), _c27)
except Exception:
    pass
layout["Pow"] = [1352, 683, 1400, 869]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/28_text_5.28.png
try:
    _c28 = get_crop(28, 91, 45)
    canvas.paste(_c28, (20, 15), _c28)
except Exception:
    pass
layout["5.28"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/29_text_423_events.png
try:
    _c29 = get_crop(29, 392, 135)
    canvas.paste(_c29, (54, 390), _c29)
except Exception:
    pass
layout["423_events"] = [54, 390, 446, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/ee1eef38e6e94342b57a272493366950/step_07_2024_4_24_17_26_ee1eef38e6e94342b57a272493366950-9/30_text_Online.png
try:
    _c30 = get_crop(30, 131, 48)
    canvas.paste(_c30, (90, 1607), _c30)
except Exception:
    pass
layout["Online"] = [90, 1607, 221, 1655]
