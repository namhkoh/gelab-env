# page_id: page_eventbrite_e1a6a0d0e93c4b71830358b28372ec21_09
# screenshot: 2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11.png
# step_index: 9/9
# task: Open Eventbrite. Search for "Language Learning". Filter only online events. Note how many events are available for "Spanish".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw structural UI elements on the provided canvas using the provided draw object.
# Assumes variables provided in the environment: canvas (PIL Image), draw (ImageDraw), fonts.

# Dimensions
width, height = canvas.size

# Colors
status_bar_color = (158, 158, 158)      # grey status bar
status_bar_sep = (255, 255, 255, 60)    # subtle top highlight (unused alpha fallback)
header_bg = (255, 255, 255)             # header background (white)
divider_color = (224, 224, 224)         # light divider lines
card1_color = (30, 136, 229)            # blue card background (approx)
card2_color = (236, 239, 241)           # light grey card background
card_shadow = (200, 200, 200)           # shadow for cards
bottom_nav_bg = (255, 255, 255)         # bottom nav background
bottom_nav_border = (230, 230, 230)

# Status bar (top area)
status_bar_h = 96
draw.rectangle([0, 0, width, status_bar_h], fill=status_bar_color)

# Thin divider under status bar
draw.line([(0, status_bar_h), (width, status_bar_h)], fill=divider_color, width=1)

# Header area (search/title area)
header_top = status_bar_h
header_h = 220
draw.rectangle([0, header_top, width, header_h], fill=header_bg)

# Subtle bottom divider under header
draw.line([(48, header_h), (width-48, header_h)], fill=divider_color, width=2)

# Additional faint horizontal rule below filters area (separator)
filters_sep_y = header_h + 120
draw.line([(48, filters_sep_y), (width-48, filters_sep_y)], fill=divider_color, width=1)

# Event card 1 background (blue banner area) - positioned to match detected crop
card1_x, card1_y = 48, 676
card1_w, card1_h = 1344, 1024
card1_rect = [card1_x, card1_y, card1_x + card1_w, card1_y + card1_h]

# Shadow for card1
shadow_offset = 8
draw.rounded_rectangle(
    [card1_rect[0] + shadow_offset, card1_rect[1] + shadow_offset,
     card1_rect[2] + shadow_offset, card1_rect[3] + shadow_offset],
    radius=28, fill=card_shadow
)

# Main rounded card1 background
draw.rounded_rectangle(card1_rect, radius=28, fill=card1_color)

# Thin inner border highlight on top edge of card1
draw.line([(card1_x + 12, card1_y + 8), (card1_x + card1_w - 12, card1_y + 8)], fill=(255,255,255,40), width=2)

# Separator (space) between cards
sep_y = card1_y + card1_h + 36
draw.line([(48, sep_y), (width-48, sep_y)], fill=divider_color, width=1)

# Event card 2 background (light grey banner area)
card2_x, card2_y = 48, 1748
card2_w, card2_h = 1344, 1048
card2_rect = [card2_x, card2_y, card2_x + card2_w, card2_y + card2_h]

# Shadow for card2
draw.rounded_rectangle(
    [card2_rect[0] + shadow_offset, card2_rect[1] + shadow_offset,
     card2_rect[2] + shadow_offset, card2_rect[3] + shadow_offset],
    radius=28, fill=card_shadow
)

# Main rounded card2 background
draw.rounded_rectangle(card2_rect, radius=28, fill=card2_color)

# Subtle inner top highlight for card2
draw.line([(card2_x + 12, card2_y + 8), (card2_x + card2_w - 12, card2_y + 8)], fill=(255,255,255,90), width=2)

# Content separators and structure lines for list area
list_start_y = sep_y + 24
draw.line([(48, list_start_y), (width-48, list_start_y)], fill=divider_color, width=1)

# Card content bottom separator by the time/metadata area (below card1)
meta_sep_y = card1_y + card1_h + 12
draw.line([(48, meta_sep_y), (width-48, meta_sep_y)], fill=divider_color, width=1)

# Bottom navigation bar background and top border
bottom_nav_h = 120
bottom_nav_y = height - bottom_nav_h
draw.rectangle([0, bottom_nav_y, width, height], fill=bottom_nav_bg)
draw.line([(0, bottom_nav_y), (width, bottom_nav_y)], fill=bottom_nav_border, width=2)

# Small top divider for the entire page (thin)
draw.line([(0, header_top), (width, header_top)], fill=divider_color, width=1)

# Safety padding separators (left/right margins) vertical guide lines (very faint)
margin_x = 48
draw.line([(margin_x, 0), (margin_x, height)], fill=(245,245,245), width=1)
draw.line([(width - margin_x, 0), (width - margin_x, height)], fill=(245,245,245), width=1)

# Done drawing structural elements.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (850, 410), _c0)
except Exception:
    pass
layout["Music"] = [850, 410, 1037, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["Anytime"] = [438, 410, 838, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1049, 410), _c2)
except Exception:
    pass
layout["Business"] = [1049, 410, 1290, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2264), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2264, 1236, 2408]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/05_icon_Overflow_menu_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 2264), _c5)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2264, 1380, 2408]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/06_icon_Fo.png
try:
    _c6 = get_crop(6, 138, 111)
    canvas.paste(_c6, (1295, 406), _c6)
except Exception:
    pass
layout["Fo("] = [1295, 406, 1433, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/07_icon_LIVE.png
try:
    _c7 = get_crop(7, 1344, 1024)
    canvas.paste(_c7, (48, 676), _c7)
except Exception:
    pass
layout["LIVE"] = [48, 676, 1392, 1700]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/08_icon_5.18.png
try:
    _c8 = get_crop(8, 127, 119)
    canvas.paste(_c8, (53, 111), _c8)
except Exception:
    pass
layout["5.18"] = [53, 111, 180, 230]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/09_icon_Close_current_screen.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/10_icon_Favorite_button.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1092, 1192), _c10)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1236, 1192), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/12_icon_5.18.png
try:
    _c12 = get_crop(12, 58, 62)
    canvas.paste(_c12, (181, 1), _c12)
except Exception:
    pass
layout["5.18"] = [181, 1, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 101, 62)
    canvas.paste(_c13, (1209, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1209, 0, 1310, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 62, 61)
    canvas.paste(_c14, (310, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [310, 1, 372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/15_icon_Language_Learning.png
try:
    _c15 = get_crop(15, 1344, 191)
    canvas.paste(_c15, (48, 72), _c15)
except Exception:
    pass
layout["Language_Learning"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/16_icon_5.18.png
try:
    _c16 = get_crop(16, 57, 65)
    canvas.paste(_c16, (116, 0), _c16)
except Exception:
    pass
layout["5.18"] = [116, 0, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 51, 62)
    canvas.paste(_c17, (247, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [247, 1, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 63, 60)
    canvas.paste(_c18, (1318, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1318, 0, 1381, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/19_icon_7_00_PM_EDT.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (576, 2804), _c19)
except Exception:
    pass
layout["7:00_PM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/20_icon_Online.png
try:
    _c20 = get_crop(20, 377, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/21_icon_7_00_PM_EDT.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["7:00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 47, 60)
    canvas.paste(_c22, (384, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [384, 3, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/23_icon_More.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (1152, 2804), _c23)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/24_icon_7_00_PM_EDT.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (864, 2804), _c24)
except Exception:
    pass
layout["7:00_PM_EDT"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/25_icon_Online_Spanish_Tutoring.png
try:
    _c25 = get_crop(25, 1344, 1048)
    canvas.paste(_c25, (48, 1748), _c25)
except Exception:
    pass
layout["Online_Spanish_Tutoring"] = [48, 1748, 1392, 2796]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/26_icon_Redefining_English_Teaching_with_AI.png
try:
    _c26 = get_crop(26, 1344, 1024)
    canvas.paste(_c26, (48, 676), _c26)
except Exception:
    pass
layout["Redefining_English_Teachi"] = [48, 676, 1392, 1700]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/27_icon_It_s_all_aboutyou.png
try:
    _c27 = get_crop(27, 1344, 1048)
    canvas.paste(_c27, (48, 1748), _c27)
except Exception:
    pass
layout["It's_all_aboutyou!"] = [48, 1748, 1392, 2796]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/28_icon_Online.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Online"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/29_text_5.18.png
try:
    _c29 = get_crop(29, 89, 43)
    canvas.paste(_c29, (22, 17), _c29)
except Exception:
    pass
layout["5.18"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/30_text_4events.png
try:
    _c30 = get_crop(30, 372, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["4events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/31_text_Online.png
try:
    _c31 = get_crop(31, 131, 50)
    canvas.paste(_c31, (90, 1600), _c31)
except Exception:
    pass
layout["Online"] = [90, 1600, 221, 1650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/32_text_ONLINE.png
try:
    _c32 = get_crop(32, 308, 77)
    canvas.paste(_c32, (283, 1772), _c32)
except Exception:
    pass
layout["ONLINE"] = [283, 1772, 591, 1849]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/33_text_SPANISH.png
try:
    _c33 = get_crop(33, 351, 77)
    canvas.paste(_c33, (623, 1772), _c33)
except Exception:
    pass
layout["SPANISH"] = [623, 1772, 974, 1849]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/34_text_CLASSES.png
try:
    _c34 = get_crop(34, 362, 86)
    canvas.paste(_c34, (1005, 1767), _c34)
except Exception:
    pass
layout["CLASSES"] = [1005, 1767, 1367, 1853]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/35_text_Whip_up_a_customised_language_class_that.png
try:
    _c35 = get_crop(35, 1344, 1048)
    canvas.paste(_c35, (48, 1748), _c35)
except Exception:
    pass
layout["Whip_up_a_customised_lang"] = [48, 1748, 1392, 2796]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_09_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-11/36_text_fityour.png
try:
    _c36 = get_crop(36, 137, 67)
    canvas.paste(_c36, (429, 2016), _c36)
except Exception:
    pass
layout["fityour"] = [429, 2016, 566, 2083]
