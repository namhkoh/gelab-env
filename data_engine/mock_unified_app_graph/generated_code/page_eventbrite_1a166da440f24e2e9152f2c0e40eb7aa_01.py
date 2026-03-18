# page_id: page_eventbrite_1a166da440f24e2e9152f2c0e40eb7aa_01
# screenshot: 2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3.png
# step_index: 1/16
# task: Open Eventbrite. Check "Sports" category. Filter events happening next month. Add the first event to your wishlist.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided. Fonts: font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
BG = "#FFFFFF"
STATUS_BAR = "#D0D0D0"
HEADER_DIVIDER = "#E9E7EB"
CARD_BG = "#FFFFFF"
CARD_SHADOW = "#F6F5F7"
THUMB_BG = "#F2F2F4"
SEPARATOR = "#EFEFF1"
NAV_BG = "#FFFFFF"
NAV_TOP_BORDER = "#ECEBED"

# Fill background (dominant color)
draw.rectangle([(0, 0), (W, H)], fill=BG)

# Status bar area (approx ~50-64px high)
status_h = 64
draw.rectangle([(0, 0), (W, status_h)], fill=STATUS_BAR)
# subtle bottom line under status bar
draw.line([(0, status_h), (W, status_h)], fill=HEADER_DIVIDER, width=1)

# Header / toolbar background area (below status bar)
header_top = status_h
header_h = 160
draw.rectangle([(0, header_top), (W, header_top + header_h)], fill=BG)
# divider below header
draw.line([(0, header_top + header_h), (W, header_top + header_h)], fill=HEADER_DIVIDER, width=1)

# Main content area subtle background (keep white, but add a faint overall tint shadow band to indicate content area)
content_top = header_top + header_h + 16
# faint band to visually separate header from content
draw.rectangle([(0, content_top - 8), (W, content_top + 8)], fill=CARD_SHADOW)

# Event rows: draw subtle rounded card backgrounds and thumbnail placeholders
rows = [
    (48, 490, 1344, 396),
    (48, 886, 1344, 396),
    (48, 1282, 1344, 396),
    (48, 1678, 1344, 396),
    (48, 2074, 1344, 396),
    (48, 2470, 1344, 346),
]

# Helper to compute rectangle coords from given x,y,w,h
def rect_from_xywh(x, y, w, h):
    return (x, y, x + w, y + h)

for x, y, w, h in rows:
    # card background (rounded)
    card_bbox = rect_from_xywh(x, y, w, h)
    # draw a subtle shadow rectangle behind for elevation
    shadow_bbox = (card_bbox[0] + 0, card_bbox[1] + 6, card_bbox[2] + 0, card_bbox[3] + 6)
    draw.rounded_rectangle(shadow_bbox, radius=16, fill=CARD_SHADOW)
    # main card (same color as page but with slight border feel)
    draw.rounded_rectangle(card_bbox, radius=12, fill=CARD_BG, outline=SEPARATOR)
    # thumbnail placeholder on the left (do not draw any icons/text)
    thumb_w = 180
    thumb_h = 180
    thumb_x = x + 0
    thumb_y = y + 30
    thumb_bbox = (thumb_x, thumb_y, thumb_x + thumb_w, thumb_y + thumb_h)
    draw.rounded_rectangle(thumb_bbox, radius=8, fill=THUMB_BG, outline="#E6E6E8")
    # subtle divider line under this row
    sep_y = y + h - 1
    draw.line([(x + 0, sep_y), (x + w, sep_y)], fill=SEPARATOR, width=1)

# Bottom navigation bar background
nav_top = 2804
draw.rectangle([(0, nav_top), (W, H)], fill=NAV_BG)
# top border for nav
draw.line([(0, nav_top), (W, nav_top)], fill=NAV_TOP_BORDER, width=1)

# Small subtle left/right padding guides (non-intrusive, very light) to match content margins
pad_x = 48
draw.line([(pad_x, header_top + header_h + 8), (pad_x, H - 200)], fill="#FFFFFF00")  # effectively invisible placeholder

# Add faint vertical separators between content area and the right edge to suggest layout boundaries
right_margin = 48
draw.line([(W - right_margin, header_top + header_h + 8), (W - right_margin, H - 200)], fill="#FFFFFF00")  # placeholder (transparent)

# Note: All interactive icons, text and images will be pasted on top of these backgrounds.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/00_icon_Online.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 490), _c0)
except Exception:
    pass
layout["Online"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/01_icon_Online.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 2074), _c1)
except Exception:
    pass
layout["Online"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/02_icon_Online.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1678), _c2)
except Exception:
    pass
layout["Online"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/03_icon_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 125)
    canvas.paste(_c4, (1140, 2345), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1140, 2345, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 125)
    canvas.paste(_c5, (1140, 1949), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1949, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 125)
    canvas.paste(_c6, (1284, 2345), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2345, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 139)
    canvas.paste(_c7, (1284, 1539), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/08_icon_5.30.png
try:
    _c8 = get_crop(8, 108, 102)
    canvas.paste(_c8, (38, 120), _c8)
except Exception:
    pass
layout["5.30"] = [38, 120, 146, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/09_icon_On..png
try:
    _c9 = get_crop(9, 288, 156)
    canvas.paste(_c9, (288, 2804), _c9)
except Exception:
    pass
layout["On."] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 139)
    canvas.paste(_c10, (1284, 747), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 125)
    canvas.paste(_c11, (1284, 1949), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1949, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 139)
    canvas.paste(_c12, (1284, 1143), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/13_icon_Home.png
try:
    _c13 = get_crop(13, 288, 156)
    canvas.paste(_c13, (0, 2804), _c13)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/14_icon_Online_events.png
try:
    _c14 = get_crop(14, 586, 117)
    canvas.paste(_c14, (427, 2651), _c14)
except Exception:
    pass
layout["Online_events"] = [427, 2651, 1013, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 60, 58)
    canvas.paste(_c15, (312, 3), _c15)
except Exception:
    pass
layout["icon_15"] = [312, 3, 372, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/16_icon_5.30.png
try:
    _c16 = get_crop(16, 55, 59)
    canvas.paste(_c16, (183, 3), _c16)
except Exception:
    pass
layout["5.30"] = [183, 3, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/17_icon_Art_for_Grief_and_Loss.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 1282), _c17)
except Exception:
    pass
layout["Art_for_Grief_and_Loss"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 51, 59)
    canvas.paste(_c18, (248, 3), _c18)
except Exception:
    pass
layout["icon_18"] = [248, 3, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/19_icon_Favorite_button.png
try:
    _c19 = get_crop(19, 144, 139)
    canvas.paste(_c19, (1140, 747), _c19)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/20_icon_Favorite_button.png
try:
    _c20 = get_crop(20, 144, 139)
    canvas.paste(_c20, (1140, 1143), _c20)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/21_icon_Favorite_button.png
try:
    _c21 = get_crop(21, 144, 139)
    canvas.paste(_c21, (1140, 1539), _c21)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 48, 53)
    canvas.paste(_c22, (1321, 7), _c22)
except Exception:
    pass
layout["icon_22"] = [1321, 7, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/23_icon_Tickets.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/24_icon_Working_with_Grief_and_Loss.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 490), _c24)
except Exception:
    pass
layout["Working_with_Grief_and_Lo"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 67, 60)
    canvas.paste(_c25, (1212, 3), _c25)
except Exception:
    pass
layout["icon_25"] = [1212, 3, 1279, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/26_icon_S_00_AM_EDT.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 1678), _c26)
except Exception:
    pass
layout["S:00_AM_EDT"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/27_icon_5_O0_AM_EDT.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 2074), _c27)
except Exception:
    pass
layout["5:O0_AM_EDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 44, 56)
    canvas.paste(_c28, (385, 7), _c28)
except Exception:
    pass
layout["icon_28"] = [385, 7, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/29_icon_5.30.png
try:
    _c29 = get_crop(29, 57, 61)
    canvas.paste(_c29, (116, 2), _c29)
except Exception:
    pass
layout["5.30"] = [116, 2, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/30_icon_suppoloyed_Orilee_herapeeticrarard_Outh_.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 1282), _c30)
except Exception:
    pass
layout["suppoloyed_Orilee__herape"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/31_icon_Understanding_Grief_and_Loss.png
try:
    _c31 = get_crop(31, 1344, 396)
    canvas.paste(_c31, (48, 886), _c31)
except Exception:
    pass
layout["Understanding_Grief_and_L"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/32_icon_icon_32.png
try:
    _c32 = get_crop(32, 42, 56)
    canvas.paste(_c32, (1272, 5), _c32)
except Exception:
    pass
layout["icon_32"] = [1272, 5, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/33_icon_Online.png
try:
    _c33 = get_crop(33, 112, 53)
    canvas.paste(_c33, (390, 1496), _c33)
except Exception:
    pass
layout["Online"] = [390, 1496, 502, 1549]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/34_icon_Online.png
try:
    _c34 = get_crop(34, 112, 54)
    canvas.paste(_c34, (390, 703), _c34)
except Exception:
    pass
layout["Online"] = [390, 703, 502, 757]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/35_icon_Art_for_Grief_and_Loss.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 1282), _c35)
except Exception:
    pass
layout["Art_for_Grief_and_Loss"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/36_icon_9_2273_creator_followers.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (576, 2804), _c36)
except Exception:
    pass
layout["9_2273_creator_followers"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/37_text_5.30.png
try:
    _c37 = get_crop(37, 91, 45)
    canvas.paste(_c37, (20, 15), _c37)
except Exception:
    pass
layout["5.30"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/38_text_More_events_you_II_love.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 490), _c38)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/39_text_Thu_May_2.png
try:
    _c39 = get_crop(39, 195, 48)
    canvas.paste(_c39, (389, 2522), _c39)
except Exception:
    pass
layout["Thu,_May_2"] = [389, 2522, 584, 2570]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/40_text_6_00_PM_EDT.png
try:
    _c40 = get_crop(40, 1344, 346)
    canvas.paste(_c40, (48, 2470), _c40)
except Exception:
    pass
layout["6:00_PM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/41_text_Free.png
try:
    _c41 = get_crop(41, 78, 38)
    canvas.paste(_c41, (274, 2561), _c41)
except Exception:
    pass
layout["Free"] = [274, 2561, 352, 2599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/42_text_How_to_Break_Into_Tech_Learn_to_Code_wit.png
try:
    _c42 = get_crop(42, 1344, 346)
    canvas.paste(_c42, (48, 2470), _c42)
except Exception:
    pass
layout["How_to_Break_Into_Tech:_L"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_01_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-3/43_clickable_More.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (1152, 2804), _c43)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
