# page_id: page_eventbrite_86c0bd1901f44c94916665f4058f9b6d_07
# screenshot: 2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9.png
# step_index: 7/11
# task: Open Eventbrite. Set the city to Los Angeles. Select the 'Food & Drink' category. What's the date of the first event that is not promoted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw page background and structural chrome for the Eventbrite UI
w, h = canvas.size

# Colors
bg = (250, 251, 252)           # overall page background (very light)
status_bar_col = (150, 150, 150)  # status bar (darker grey)
search_area_col = (255, 255, 255) # search/header area (white)
divider_col = (225, 226, 229)   # light divider lines
card_fill = (255, 255, 255)     # card background
card_border = (235, 236, 238)   # card border
card_shadow = (237, 238, 240)   # subtle shadow behind cards
nav_bg = (255, 255, 255)        # bottom navigation background

# Fill canvas background
draw.rectangle([(0, 0), (w, h)], fill=bg)

# Status bar (approx. top 64px)
status_h = 64
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_col)

# Header / search area (just below status bar)
search_top = status_h
search_bottom = 160
draw.rectangle([(0, search_top), (w, search_bottom)], fill=search_area_col)
# subtle divider under search field
draw.line([(48, search_bottom), (w-48, search_bottom)], fill=divider_col, width=2)

# Location row / filters row background area (keep neutral background)
loc_top = search_bottom
loc_bottom = 420
draw.rectangle([(0, loc_top), (w, loc_bottom)], fill=search_area_col)
# divider under location & filter chips area
draw.line([(24, loc_bottom), (w-24, loc_bottom)], fill=divider_col, width=2)

# Light horizontal spacer area for "10,000 events" region (do not draw text)
list_header_top = loc_bottom + 8
list_header_bottom = list_header_top + 80
# keep same background but add a faint bottom divider
draw.rectangle([(0, list_header_top), (w, list_header_bottom)], fill=bg)
draw.line([(48, list_header_bottom), (w-48, list_header_bottom)], fill=divider_col, width=1)

# Draw first event card background (rounded rectangle + subtle shadow)
# Event image/content will be pasted on top; this is just the card chrome.
card1_left = 48
card1_right = card1_left + 1344
card1_top = 640
card1_bottom = 1880
# shadow
shadow_offset = 8
draw.rounded_rectangle(
    [(card1_left + shadow_offset, card1_top + shadow_offset),
     (card1_right + shadow_offset, card1_bottom + shadow_offset)],
    radius=22, fill=card_shadow, outline=None
)
# card body
draw.rounded_rectangle(
    [(card1_left, card1_top), (card1_right, card1_bottom)],
    radius=22, fill=card_fill, outline=card_border, width=1
)

# Divider between event cards (subtle)
divider_y = card1_bottom + 12
draw.line([(48, divider_y), (card1_right, divider_y)], fill=divider_col, width=1)

# Draw second event card background (rounded rectangle + subtle shadow)
card2_left = 48
card2_right = card2_left + 1344
# second event image detection top at ~1899 -> give a bit of padding
card2_top = 1860
card2_bottom = min(h - 120, card2_top + 980)  # keep inside canvas
# shadow
draw.rounded_rectangle(
    [(card2_left + shadow_offset, card2_top + shadow_offset),
     (card2_right + shadow_offset, card2_bottom + shadow_offset)],
    radius=22, fill=card_shadow, outline=None
)
# card body
draw.rounded_rectangle(
    [(card2_left, card2_top), (card2_right, card2_bottom)],
    radius=22, fill=card_fill, outline=card_border, width=1
)

# Promoted/label area background (do not draw text; just ensure space)
# Provide a faint rounded pill behind where promoted label would appear (no icon/text)
promoted_pill_bbox = (80, 1720, 330, 1786)  # leaves room for promoted badge above second card
draw.rounded_rectangle(promoted_pill_bbox, radius=14, fill=(245,246,248), outline=None)

# Bottom navigation background and top divider
nav_top = h - 200  # leave ~200px for bottom nav + safe area
draw.line([(24, nav_top), (w-24, nav_top)], fill=divider_col, width=1)
draw.rectangle([(0, nav_top), (w, h)], fill=nav_bg)

# Subtle top toolbar under status bar (header strip where small icons sit) - keep neutral
toolbar_top = status_h
toolbar_bottom = status_h + 44
draw.rectangle([(0, toolbar_top), (w, toolbar_bottom)], fill=(248,249,250))

# Final subtle vertical gutters (left and right margins)
gutters_w = 24
draw.rectangle([(0, 0), (gutters_w, h)], fill=bg)
draw.rectangle([(w-gutters_w, 0), (w, h)], fill=bg)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/04_icon_Foo.png
try:
    _c4 = get_crop(4, 149, 110)
    canvas.paste(_c4, (1283, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1283, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/06_icon_JNDRE_Roto.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2415), _c6)
except Exception:
    pass
layout["JNDRE_Roto"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/08_icon_JNDRE_Roto.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2415), _c8)
except Exception:
    pass
layout["JNDRE_Roto"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/09_icon_Tom_Everhart_at_the_Grand_Opening_of_Cho.png
try:
    _c9 = get_crop(9, 1344, 1175)
    canvas.paste(_c9, (48, 676), _c9)
except Exception:
    pass
layout["Tom_Everhart_at_the_Grand"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/10_icon_7.13.png
try:
    _c10 = get_crop(10, 125, 111)
    canvas.paste(_c10, (55, 117), _c10)
except Exception:
    pass
layout["7.13"] = [55, 117, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/11_icon_Search_forae.png
try:
    _c11 = get_crop(11, 1344, 191)
    canvas.paste(_c11, (48, 72), _c11)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 53, 64)
    canvas.paste(_c12, (1151, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1151, 0, 1204, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/13_icon_Search_forae.png
try:
    _c13 = get_crop(13, 69, 63)
    canvas.paste(_c13, (307, 0), _c13)
except Exception:
    pass
layout["Search_forae"] = [307, 0, 376, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 54, 64)
    canvas.paste(_c14, (246, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [246, 0, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 91, 61)
    canvas.paste(_c15, (1212, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1212, 0, 1303, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/16_icon_7.13.png
try:
    _c16 = get_crop(16, 61, 63)
    canvas.paste(_c16, (181, 0), _c16)
except Exception:
    pass
layout["7.13"] = [181, 0, 242, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/17_icon_7.13.png
try:
    _c17 = get_crop(17, 62, 65)
    canvas.paste(_c17, (114, 0), _c17)
except Exception:
    pass
layout["7.13"] = [114, 0, 176, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 59, 59)
    canvas.paste(_c18, (1317, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1317, 0, 1376, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/19_icon_Sun_Apr_28_._5.00_PM_PDT.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (576, 2804), _c19)
except Exception:
    pass
layout["Sun,_Apr_28_._5.00_PM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/20_icon_Los_Angeles.png
try:
    _c20 = get_crop(20, 492, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/21_icon_Iet.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (864, 2804), _c21)
except Exception:
    pass
layout["Iet"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/22_icon_Search_forae.png
try:
    _c22 = get_crop(22, 53, 61)
    canvas.paste(_c22, (383, 2), _c22)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 436, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/23_icon_Regal_LA_Live.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (288, 2804), _c23)
except Exception:
    pass
layout["Regal_LA_Live"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/24_icon_JNDRE_Roto.png
try:
    _c24 = get_crop(24, 147, 153)
    canvas.paste(_c24, (957, 2393), _c24)
except Exception:
    pass
layout["JNDRE_Roto"] = [957, 2393, 1104, 2546]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/25_icon_Iet.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (1152, 2804), _c25)
except Exception:
    pass
layout["Iet"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/26_icon_Tom_Everhart_at_the_Grand_Opening_of_Cho.png
try:
    _c26 = get_crop(26, 1344, 1175)
    canvas.paste(_c26, (48, 676), _c26)
except Exception:
    pass
layout["Tom_Everhart_at_the_Grand"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/27_icon_Promoted.png
try:
    _c27 = get_crop(27, 242, 66)
    canvas.paste(_c27, (85, 1744), _c27)
except Exception:
    pass
layout["Promoted"] = [85, 1744, 327, 1810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/28_icon_Regal_LA_Live.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Regal_LA_Live"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/29_icon_7.13.png
try:
    _c29 = get_crop(29, 139, 64)
    canvas.paste(_c29, (6, 0), _c29)
except Exception:
    pass
layout["7.13"] = [6, 0, 145, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/30_icon_InTERNATIONAL.png
try:
    _c30 = get_crop(30, 1344, 917)
    canvas.paste(_c30, (48, 1899), _c30)
except Exception:
    pass
layout["InTERNATIONAL"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/31_icon_Beyond_Hollywood_Int_I_Film_Festival_202.png
try:
    _c31 = get_crop(31, 1344, 917)
    canvas.paste(_c31, (48, 1899), _c31)
except Exception:
    pass
layout["Beyond_Hollywood_Int'I_Fi"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/32_icon_icon_32.png
try:
    _c32 = get_crop(32, 40, 60)
    canvas.paste(_c32, (1274, 0), _c32)
except Exception:
    pass
layout["icon_32"] = [1274, 0, 1314, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/33_text_10_000_events.png
try:
    _c33 = get_crop(33, 359, 103)
    canvas.paste(_c33, (54, 410), _c33)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/34_text_313_N_Beverly_Dr.png
try:
    _c34 = get_crop(34, 323, 55)
    canvas.paste(_c34, (90, 1686), _c34)
except Exception:
    pass
layout["313_N_Beverly_Dr"] = [90, 1686, 413, 1741]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/35_text_Thu_Apr_25.png
try:
    _c35 = get_crop(35, 230, 52)
    canvas.paste(_c35, (93, 2680), _c35)
except Exception:
    pass
layout["Thu,_Apr_25"] = [93, 2680, 323, 2732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/36_text_Sun_Apr_28_._5.00_PM_PDT.png
try:
    _c36 = get_crop(36, 1344, 917)
    canvas.paste(_c36, (48, 1899), _c36)
except Exception:
    pass
layout["Sun,_Apr_28_._5.00_PM_PDT"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_07_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-9/37_text_Regal_LA_Live.png
try:
    _c37 = get_crop(37, 288, 156)
    canvas.paste(_c37, (0, 2804), _c37)
except Exception:
    pass
layout["Regal_LA_Live"] = [0, 2804, 288, 2960]
