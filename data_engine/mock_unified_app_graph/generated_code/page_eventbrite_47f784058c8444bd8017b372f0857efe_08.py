# page_id: page_eventbrite_47f784058c8444bd8017b372f0857efe_08
# screenshot: 2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10.png
# step_index: 8/11
# task: Open Eventbrite. Explore local events scheduled for this weekend. Select the first event from the 'Science' category. Read details of the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top/bottom/background, header, cards, dividers for Eventbrite-style UI
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
bg_color = (246, 247, 251)        # very light cool background
status_bar_color = (191, 191, 191)  # gray status bar
header_color = (255, 255, 255)    # white header/toolbar
divider_color = (230, 231, 235)   # subtle divider
card_shadow = (232, 236, 242)     # card shadow / subtle gray
card_bg = (255, 255, 255)         # card background white
accent_banner = (255, 246, 210)   # pale yellow banner for content area
accent_strip = (203, 130, 255)    # pale purple accent

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar (top area)
status_h = 56
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# Header / search area background
header_top = status_h
header_bottom = 320
draw.rectangle([(0, header_top), (w, header_bottom)], fill=header_color)

# Header inner subtle bottom divider
draw.line([(32, header_bottom), (w - 32, header_bottom)], fill=divider_color, width=2)

# Light horizontal rule below the filter/search area
hr_y = 520
draw.line([(24, hr_y), (w - 24, hr_y)], fill=divider_color, width=2)

# Main event card 1 container with rounded corners and subtle shadow
card_margin_x = 48
card1_top = 620
card1_bottom = 1720
card1_box = (card_margin_x, card1_top, w - card_margin_x, card1_bottom)
shadow_offset = 8

# Shadow (simple offset rectangle filled lightly)
draw.rounded_rectangle(
    [card1_box[0] + shadow_offset, card1_box[1] + shadow_offset, card1_box[2] + shadow_offset, card1_box[3] + shadow_offset],
    radius=28, fill=card_shadow
)
# Card background
draw.rounded_rectangle(card1_box, radius=28, fill=card_bg)

# Subtle divider inside the card to separate image area from text area
# We won't draw image content (that will be pasted), only a faint separator line where image would transition.
img_height_est = 420  # estimated image area within card
sep_y = card1_top + img_height_est
draw.line([(card1_box[0] + 24, sep_y), (card1_box[2] - 24, sep_y)], fill=divider_color, width=1)

# A small soft inner shadow near top edge of image area for depth (not duplicating actual image)
inner_shadow_top = card1_top + 6
draw.line([(card1_box[0] + 20, inner_shadow_top), (card1_box[2] - 20, inner_shadow_top)], fill=(245,246,249), width=2)

# Main event card 2 container (second listing) with rounded corners and shadow
card2_top = 1728
card2_bottom = 2816  # leave some space above bottom nav
card2_box = (card_margin_x, card2_top, w - card_margin_x, card2_bottom)

draw.rounded_rectangle(
    [card2_box[0] + shadow_offset, card2_box[1] + shadow_offset, card2_box[2] + shadow_offset, card2_box[3] + shadow_offset],
    radius=28, fill=card_shadow
)
draw.rounded_rectangle(card2_box, radius=28, fill=card_bg)

# Decorative colored banner area behind image region of second card (will be covered by pasted image but gives background)
banner_h = 260
banner_box = (card2_box[0] + 12, card2_box[1] + 12, card2_box[2] - 12, card2_box[1] + 12 + banner_h)
draw.rectangle(banner_box, fill=accent_banner)
# Accent strip at top of banner
acc_strip_h = 48
draw.rectangle([banner_box[0], banner_box[1], banner_box[2], banner_box[1] + acc_strip_h], fill=accent_strip)

# Divider lines between the two cards area and the following content
draw.line([(24, card1_bottom + 24), (w - 24, card1_bottom + 24)], fill=divider_color, width=1)

# Content separators (subtle) around the page
separator_positions = [440, 640, 900, 1680, 2800]
for y in separator_positions:
    # avoid drawing heavy separators over the header area; keep them faint
    if 0 < y < h - 120:
        draw.line([(24, y), (w - 24, y)], fill=(245, 246, 249), width=1)

# Bottom navigation bar background and top border
nav_h = 120
nav_top = h - nav_h
draw.rectangle([(0, nav_top), (w, h)], fill=card_bg)
draw.line([(0, nav_top), (w, nav_top)], fill=divider_color, width=2)

# Soft shadow above bottom nav for separation
for i, alpha in enumerate([230, 200, 170], start=1):
    y = nav_top - i * 2
    draw.line([(0, y), (w, y)], fill=(240, 241, 245), width=1)

# Left vertical margin guide (visual only, very faint) - helps define content area; extremely subtle so it won't conflict with icons/text
draw.line([(card_margin_x - 8, header_top + 8), (card_margin_x - 8, nav_top - 8)], fill=(250, 250, 251), width=1)

# Right vertical margin guide (very faint)
draw.line([(w - card_margin_x + 8, header_top + 8), (w - card_margin_x + 8, nav_top - 8)], fill=(250, 250, 251), width=1)

# Light bottom padding area under last card (so pasted content doesn't touch nav directly)
pad_y = card2_bottom + 12
if pad_y < nav_top - 12:
    draw.rectangle([(card2_box[0], pad_y), (card2_box[2], nav_top - 12)], fill=(250, 250, 251))

# Done - background, headers, cards, dividers drawn.
# (All icons/text/images will be pasted on top at their detected positions.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/00_icon_This_Weekend.png
try:
    _c0 = get_crop(0, 504, 135)
    canvas.paste(_c0, (458, 390), _c0)
except Exception:
    pass
layout["This_Weekend"] = [458, 390, 962, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/01_icon_2_Filters.png
try:
    _c1 = get_crop(1, 392, 135)
    canvas.paste(_c1, (54, 390), _c1)
except Exception:
    pass
layout["2_Filters"] = [54, 390, 446, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/02_icon_Science_Tech.png
try:
    _c2 = get_crop(2, 361, 135)
    canvas.paste(_c2, (974, 390), _c2)
except Exception:
    pass
layout["Science_&_Tech"] = [974, 390, 1335, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/03_icon_April.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1236, 2269), _c3)
except Exception:
    pass
layout["April"] = [1236, 2269, 1380, 2413]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/04_icon_Area_Bioengineering_Symposium_BABS.png
try:
    _c4 = get_crop(4, 1344, 1029)
    canvas.paste(_c4, (48, 676), _c4)
except Exception:
    pass
layout["Area_Bioengineering_Sympo"] = [48, 676, 1392, 1705]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/05_icon_7.58.png
try:
    _c5 = get_crop(5, 115, 111)
    canvas.paste(_c5, (60, 115), _c5)
except Exception:
    pass
layout["7.58"] = [60, 115, 175, 226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1192), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/08_icon_7.58.png
try:
    _c8 = get_crop(8, 61, 65)
    canvas.paste(_c8, (180, 0), _c8)
except Exception:
    pass
layout["7.58"] = [180, 0, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/09_icon_Search_forae.png
try:
    _c9 = get_crop(9, 66, 62)
    canvas.paste(_c9, (308, 1), _c9)
except Exception:
    pass
layout["Search_forae"] = [308, 1, 374, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/10_icon_7.58.png
try:
    _c10 = get_crop(10, 61, 66)
    canvas.paste(_c10, (114, 0), _c10)
except Exception:
    pass
layout["7.58"] = [114, 0, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 54, 64)
    canvas.paste(_c11, (246, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [246, 0, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/12_icon_April.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1092, 2269), _c12)
except Exception:
    pass
layout["April"] = [1092, 2269, 1236, 2413]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 56, 61)
    canvas.paste(_c13, (1317, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1317, 0, 1373, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 61, 62)
    canvas.paste(_c14, (1212, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1212, 0, 1273, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 40, 62)
    canvas.paste(_c15, (1273, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1273, 0, 1313, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/16_icon_Search_forae.png
try:
    _c16 = get_crop(16, 1344, 191)
    canvas.paste(_c16, (48, 72), _c16)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/17_icon_ALL_EDUCATORS.png
try:
    _c17 = get_crop(17, 1344, 1063)
    canvas.paste(_c17, (48, 1753), _c17)
except Exception:
    pass
layout["ALL_EDUCATORS"] = [48, 1753, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/18_icon_Search_forae.png
try:
    _c18 = get_crop(18, 49, 63)
    canvas.paste(_c18, (383, 1), _c18)
except Exception:
    pass
layout["Search_forae"] = [383, 1, 432, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/19_icon_1I_00_AM_PDT.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["1I:00_AM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/20_icon_1I_00_AM_PDT.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (576, 2804), _c20)
except Exception:
    pass
layout["1I:00_AM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/21_icon_April.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (864, 2804), _c21)
except Exception:
    pass
layout["April"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/22_icon_San_Francisco.png
try:
    _c22 = get_crop(22, 536, 144)
    canvas.paste(_c22, (0, 259), _c22)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/23_icon_Bio-Link_Depot.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (0, 2804), _c23)
except Exception:
    pass
layout["Bio-Link_Depot"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/24_icon_7.58.png
try:
    _c24 = get_crop(24, 142, 63)
    canvas.paste(_c24, (10, 1), _c24)
except Exception:
    pass
layout["7.58"] = [10, 1, 152, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/25_text_66_events.png
try:
    _c25 = get_crop(25, 392, 135)
    canvas.paste(_c25, (54, 390), _c25)
except Exception:
    pass
layout["66_events"] = [54, 390, 446, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/26_text_Sun_Apr_28.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Sun,_Apr_28"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/27_text_1I_00_AM_PDT.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (288, 2804), _c27)
except Exception:
    pass
layout["1I:00_AM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_08_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-10/28_clickable_More.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (1152, 2804), _c28)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
