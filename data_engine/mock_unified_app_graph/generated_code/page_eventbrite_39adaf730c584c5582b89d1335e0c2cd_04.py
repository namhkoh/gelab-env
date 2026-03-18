# page_id: page_eventbrite_39adaf730c584c5582b89d1335e0c2cd_04
# screenshot: 2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6.png
# step_index: 4/6
# task: Open Eventbrite. Search for 'food and drink' events. Follow the organizer of the first event in listing.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural layout for the Eventbrite "Food and Drink" listing UI.
# Uses provided canvas (1440x2960) and draw objects.

# Colors
BG = (255, 255, 255)            # main background (white)
STATUS_BG = (189, 189, 189)     # status bar grey
HEADER_DIV = (230, 230, 230)    # header divider
CARD_SHADOW = (235, 235, 240)   # subtle card shadow
CARD_BG = (255, 255, 255)       # card background (white)
NAV_BORDER = (224, 224, 224)    # top border of bottom nav
NAV_BG = (255, 255, 255)        # nav bar background
SECTION_DIV = (245, 245, 247)   # faint section divider / background

w, h = canvas.size

# Fill full background (explicit)
draw.rectangle([(0, 0), (w, h)], fill=BG)

# Status bar (top area)
status_h = 80
draw.rectangle([(0, 0), (w, status_h)], fill=STATUS_BG)

# Header area (search title + location + small divider)
header_top = status_h
# Keep header white to let pasted text/icons stand out; draw a faint area to emphasize separation
draw.rectangle([(0, header_top), (w, header_top + 200)], fill=BG)
# bottom divider under header
header_bottom_y = header_top + 195
draw.line([(32, header_bottom_y), (w - 32, header_bottom_y)], fill=HEADER_DIV, width=2)

# Minor section background band for filters row (subtle)
filters_band_top = header_bottom_y + 24
filters_band_bottom = filters_band_top + 110
draw.rectangle([(0, filters_band_top), (w, filters_band_bottom)], fill=SECTION_DIV)

# Card container positions (two main event cards)
card_left = 48
card_right = w - 48
card_radius = 28

# First card
card1_top = 520
card1_height = 700
card1_bottom = card1_top + card1_height

# Card shadow (offset)
shadow_offset = 10
draw.rounded_rectangle(
    [(card_left + shadow_offset, card1_top + shadow_offset),
     (card_right + shadow_offset, card1_bottom + shadow_offset)],
    radius=card_radius, fill=CARD_SHADOW
)

# Card background
draw.rounded_rectangle(
    [(card_left, card1_top), (card_right, card1_bottom)],
    radius=card_radius, fill=CARD_BG, outline=(245,245,245)
)

# Subtle thin divider inside card (to separate image area from details)
# We approximate image area height (top portion) - leave space for pasted image overlays
image_area_height1 = int(card1_top + card1_height * 0.45)
draw.line([(card_left + 8, image_area_height1), (card_right - 8, image_area_height1)], fill=HEADER_DIV, width=1)

# Second card
card2_top = card1_bottom + 80
card2_height = 760
card2_bottom = card2_top + card2_height

# Shadow and background for second card
draw.rounded_rectangle(
    [(card_left + shadow_offset, card2_top + shadow_offset),
     (card_right + shadow_offset, card2_bottom + shadow_offset)],
    radius=card_radius, fill=CARD_SHADOW
)
draw.rounded_rectangle(
    [(card_left, card2_top), (card_right, card2_bottom)],
    radius=card_radius, fill=CARD_BG, outline=(245,245,245)
)

# Image area divider for second card
image_area_height2 = int(card2_top + card2_height * 0.48)
draw.line([(card_left + 8, image_area_height2), (card_right - 8, image_area_height2)], fill=HEADER_DIV, width=1)

# Section separator line between listing and lower content
separator_y = card2_bottom + 40
draw.line([(32, separator_y), (w - 32, separator_y)], fill=HEADER_DIV, width=1)

# Bottom navigation bar background and top border
nav_h = 200
nav_top = h - nav_h
draw.line([(0, nav_top), (w, nav_top)], fill=NAV_BORDER, width=2)
draw.rectangle([(0, nav_top), (w, h)], fill=NAV_BG)

# Small subtle horizontal guideline above nav for extra separation
draw.line([(24, nav_top + 6), (w - 24, nav_top + 6)], fill=(250,250,250), width=1)

# Additional faint separators to delineate list sections (non-intrusive)
sep_positions = [card1_bottom + 40, card1_bottom + 200, card2_bottom + 20]
for y_pos in sep_positions:
    draw.line([(48, y_pos), (w - 48, y_pos)], fill=(248,248,249), width=1)

# Done - leave textual content and icons to be pasted afterwards.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/04_icon_VE.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2355), _c4)
except Exception:
    pass
layout["VE"] = [1092, 2355, 1236, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/05_icon_Foo.png
try:
    _c5 = get_crop(5, 149, 110)
    canvas.paste(_c5, (1283, 406), _c5)
except Exception:
    pass
layout["Foo"] = [1283, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1192), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/07_icon_VE.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 2355), _c7)
except Exception:
    pass
layout["VE"] = [1236, 2355, 1380, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/09_icon_Foo.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/10_icon_7.44.png
try:
    _c10 = get_crop(10, 123, 113)
    canvas.paste(_c10, (56, 115), _c10)
except Exception:
    pass
layout["7.44"] = [56, 115, 179, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 104, 61)
    canvas.paste(_c11, (1206, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1206, 0, 1310, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/12_icon_7.44.png
try:
    _c12 = get_crop(12, 58, 63)
    canvas.paste(_c12, (182, 0), _c12)
except Exception:
    pass
layout["7.44"] = [182, 0, 240, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 66, 62)
    canvas.paste(_c13, (308, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [308, 1, 374, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/14_icon_7.44.png
try:
    _c14 = get_crop(14, 61, 64)
    canvas.paste(_c14, (113, 0), _c14)
except Exception:
    pass
layout["7.44"] = [113, 0, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 50, 61)
    canvas.paste(_c15, (249, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [249, 1, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/16_icon_Help_for_Dieting_Eating_Disorders_and_Fo.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (576, 2804), _c16)
except Exception:
    pass
layout["Help_for_Dieting,_Eating_"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 60, 61)
    canvas.paste(_c17, (1318, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1318, 0, 1378, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/18_icon_Los_Angeles.png
try:
    _c18 = get_crop(18, 492, 144)
    canvas.paste(_c18, (0, 259), _c18)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/19_icon_Food_and_Drink.png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["Food_and_Drink"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/20_icon_7.44.png
try:
    _c20 = get_crop(20, 95, 63)
    canvas.paste(_c20, (12, 0), _c20)
except Exception:
    pass
layout["7.44"] = [12, 0, 107, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/21_icon_Help_for_Dieting_Eating_Disorders_and_Fo.png
try:
    _c21 = get_crop(21, 1344, 977)
    canvas.paste(_c21, (48, 1839), _c21)
except Exception:
    pass
layout["Help_for_Dieting,_Eating_"] = [48, 1839, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/22_icon_Wed_May_1_._6.00_PM_EDT.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["Wed,_May_1_._6.00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/23_icon_Help_for_Dieting_Eating_Disorders_and_Fo.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["Help_for_Dieting,_Eating_"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/24_icon_Ticket_sales_end_soon.png
try:
    _c24 = get_crop(24, 1344, 1115)
    canvas.paste(_c24, (48, 676), _c24)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [48, 676, 1392, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/25_icon_Food_and_Drink.png
try:
    _c25 = get_crop(25, 48, 61)
    canvas.paste(_c25, (384, 2), _c25)
except Exception:
    pass
layout["Food_and_Drink"] = [384, 2, 432, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/26_icon_Help_for_Dieting_Eating_Disorders_and_Fo.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (1152, 2804), _c26)
except Exception:
    pass
layout["Help_for_Dieting,_Eating_"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/27_icon_sip_paint.png
try:
    _c27 = get_crop(27, 290, 79)
    canvas.paste(_c27, (86, 1464), _c27)
except Exception:
    pass
layout["sip_&_paint"] = [86, 1464, 376, 1543]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/28_icon_Promoted.png
try:
    _c28 = get_crop(28, 253, 68)
    canvas.paste(_c28, (79, 1684), _c28)
except Exception:
    pass
layout["Promoted"] = [79, 1684, 332, 1752]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/29_text_4_699_events.png
try:
    _c29 = get_crop(29, 359, 103)
    canvas.paste(_c29, (54, 410), _c29)
except Exception:
    pass
layout["4,699_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/30_text_Ticket_sales_end_soon.png
try:
    _c30 = get_crop(30, 415, 49)
    canvas.paste(_c30, (125, 1388), _c30)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [125, 1388, 540, 1437]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/31_text_Wed_Apr_24.png
try:
    _c31 = get_crop(31, 246, 50)
    canvas.paste(_c31, (95, 1561), _c31)
except Exception:
    pass
layout["Wed,_Apr_24"] = [95, 1561, 341, 1611]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/32_text_5.30_PM_PDT.png
try:
    _c32 = get_crop(32, 254, 45)
    canvas.paste(_c32, (357, 1560), _c32)
except Exception:
    pass
layout["5.30_PM_PDT"] = [357, 1560, 611, 1605]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/33_text_7_Kingdoms.png
try:
    _c33 = get_crop(33, 227, 54)
    canvas.paste(_c33, (93, 1626), _c33)
except Exception:
    pass
layout["7_Kingdoms"] = [93, 1626, 320, 1680]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_04_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-6/34_text_Online.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (0, 2804), _c34)
except Exception:
    pass
layout["Online"] = [0, 2804, 288, 2960]
