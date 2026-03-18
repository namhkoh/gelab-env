# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_12
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-14.png
# step_index: 12/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint UI background and structural elements for a 1440x2960 canvas (variables provided: canvas, draw)

# Fill overall background with a very light neutral color
draw.rectangle([(0, 0), (1440, 2960)], fill="#F6F7F9")

# Status bar (top area)
STATUS_H = 96
draw.rectangle([(0, 0), (1440, STATUS_H)], fill="#D0D0D0")

# Top toolbar / header background (below status bar)
HEADER_H = 200
draw.rectangle([(0, STATUS_H), (1440, HEADER_H)], fill="#FFFFFF")
# Header bottom divider
draw.line([(32, HEADER_H), (1408, HEADER_H)], fill="#E6E6E6", width=2)

# Main content area top divider under filters/search area
FILTER_AREA_BOTTOM = 460
draw.line([(24, FILTER_AREA_BOTTOM), (1416, FILTER_AREA_BOTTOM)], fill="#ECEDEE", width=1)

# First event card shadow + card background
card1_x0, card1_y0 = 48, 220
card1_x1, card1_y1 = 1392, 1180
shadow_offset = 8
draw.rounded_rectangle(
    [(card1_x0 + shadow_offset, card1_y0 + shadow_offset),
     (card1_x1 + shadow_offset, card1_y1 + shadow_offset)],
    radius=28,
    fill="#E9EAEC"
)
draw.rounded_rectangle(
    [(card1_x0, card1_y0), (card1_x1, card1_y1)],
    radius=24,
    fill="#FFFFFF"
)

# Image area background inside first card (kept neutral; actual image will be pasted on top)
img1_y0 = card1_y0 + 16
img1_y1 = card1_y0 + 420
draw.rounded_rectangle(
    [(card1_x0 + 16, img1_y0), (card1_x1 - 16, img1_y1)],
    radius=16,
    fill="#F2F4F6"
)
# divider below image in card1
draw.line(
    [(card1_x0 + 16, img1_y1 + 14), (card1_x1 - 16, img1_y1 + 14)],
    fill="#F0F1F3",
    width=1
)

# Subtle separated area for metadata/title within card1 (no text drawn)
meta_top = img1_y1 + 30
draw.rectangle(
    [(card1_x0 + 16, meta_top), (card1_x1 - 16, card1_y1 - 20)],
    fill="#FFFFFF"
)

# Second event/promoted card shadow + card background
card2_x0, card2_y0 = 48, 1260
card2_x1, card2_y1 = 1392, 1860
draw.rounded_rectangle(
    [(card2_x0 + shadow_offset, card2_y0 + shadow_offset),
     (card2_x1 + shadow_offset, card2_y1 + shadow_offset)],
    radius=28,
    fill="#E9EAEC"
)
draw.rounded_rectangle(
    [(card2_x0, card2_y0), (card2_x1, card2_y1)],
    radius=24,
    fill="#FFFFFF"
)

# Banner image area inside second card (kept neutral)
img2_y0 = card2_y0 + 30
img2_y1 = card2_y0 + 420
draw.rounded_rectangle(
    [(card2_x0 + 16, img2_y0), (card2_x1 - 16, img2_y1)],
    radius=16,
    fill="#F7FBFC"
)
# divider below image in card2
draw.line(
    [(card2_x0 + 16, img2_y1 + 14), (card2_x1 - 16, img2_y1 + 14)],
    fill="#F0F1F3",
    width=1
)

# Small promoted badge background placeholder area (no text/icon)
badge_w, badge_h = 140, 56
badge_x = card2_x0 + 36
badge_y = img2_y1 + 30
draw.rounded_rectangle(
    [(badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h)],
    radius=12,
    fill="#F5F7F8"
)

# Thin separator between cards / sections
draw.line([(24, card1_y1 + 30), (1416, card1_y1 + 30)], fill="#EFEFF1", width=1)

# Bottom navigation bar background and top divider
NAV_H = 200
nav_top = 2960 - NAV_H
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#FFFFFF")
draw.line([(0, nav_top), (1440, nav_top)], fill="#E6E6E6", width=2)

# Add a faint left gutter guide and right gutter guide (visual structure only)
gutter_x = 48
draw.line([(gutter_x, HEADER_H + 12), (gutter_x, nav_top - 12)], fill="#FCFCFD", width=2)
draw.line([(1440 - gutter_x, HEADER_H + 12), (1440 - gutter_x, nav_top - 12)], fill="#FCFCFD", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (850, 410), _c0)
except Exception:
    pass
layout["Music"] = [850, 410, 1037, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["Anytime"] = [438, 410, 838, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1049, 410), _c2)
except Exception:
    pass
layout["Business"] = [1049, 410, 1290, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/05_icon_EEL.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2415), _c5)
except Exception:
    pass
layout["EEL"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/06_icon_Fo.png
try:
    _c6 = get_crop(6, 136, 111)
    canvas.paste(_c6, (1295, 406), _c6)
except Exception:
    pass
layout["Fo("] = [1295, 406, 1431, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2415), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/09_icon_7.29.png
try:
    _c9 = get_crop(9, 127, 114)
    canvas.paste(_c9, (55, 113), _c9)
except Exception:
    pass
layout["7.29"] = [55, 113, 182, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/10_icon_Close_current_screen.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 96), _c10)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 53, 66)
    canvas.paste(_c11, (1151, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1151, 0, 1204, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/12_icon_Yoga_session.png
try:
    _c12 = get_crop(12, 67, 62)
    canvas.paste(_c12, (307, 1), _c12)
except Exception:
    pass
layout["Yoga_session"] = [307, 1, 374, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 99, 64)
    canvas.paste(_c13, (1212, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1212, 0, 1311, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/14_icon_7.29.png
try:
    _c14 = get_crop(14, 60, 63)
    canvas.paste(_c14, (181, 0), _c14)
except Exception:
    pass
layout["7.29"] = [181, 0, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 51, 62)
    canvas.paste(_c15, (249, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [249, 1, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/16_icon_7.29.png
try:
    _c16 = get_crop(16, 61, 65)
    canvas.paste(_c16, (114, 0), _c16)
except Exception:
    pass
layout["7.29"] = [114, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/17_icon_Find_Joy_Ancient_Wisdom_for_Modern_Times.png
try:
    _c17 = get_crop(17, 1344, 1175)
    canvas.paste(_c17, (48, 676), _c17)
except Exception:
    pass
layout["Find_Joy:_Ancient_Wisdom_"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 56, 63)
    canvas.paste(_c18, (1318, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1318, 0, 1374, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/19_icon_Yoga_session.png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["Yoga_session"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/20_icon_Sat_Mav_11_._11.30_AMEDT.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (288, 2804), _c20)
except Exception:
    pass
layout["Sat,_Mav_11_._11.30_AMEDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/21_icon_New_York.png
try:
    _c21 = get_crop(21, 434, 144)
    canvas.paste(_c21, (0, 259), _c21)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/22_icon_Yoga_session.png
try:
    _c22 = get_crop(22, 50, 62)
    canvas.paste(_c22, (383, 2), _c22)
except Exception:
    pass
layout["Yoga_session"] = [383, 2, 433, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/23_icon_Promoted.png
try:
    _c23 = get_crop(23, 243, 67)
    canvas.paste(_c23, (84, 1743), _c23)
except Exception:
    pass
layout["Promoted"] = [84, 1743, 327, 1810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/24_icon_Day.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (864, 2804), _c24)
except Exception:
    pass
layout["Day"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/25_icon_Sat_Mav_11_._11.30_AMEDT.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["Sat,_Mav_11_._11.30_AMEDT"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/26_icon_Day.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (576, 2804), _c26)
except Exception:
    pass
layout["Day"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/27_icon_More.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (1152, 2804), _c27)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/28_icon_Z-00_PM.png
try:
    _c28 = get_crop(28, 1344, 917)
    canvas.paste(_c28, (48, 1899), _c28)
except Exception:
    pass
layout["Z-00_PM"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/29_text_7.29.png
try:
    _c29 = get_crop(29, 91, 45)
    canvas.paste(_c29, (20, 15), _c29)
except Exception:
    pass
layout["7.29"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/30_text_1_678_events.png
try:
    _c30 = get_crop(30, 372, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["1,678_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/31_text_Wed_Apr_24.png
try:
    _c31 = get_crop(31, 248, 54)
    canvas.paste(_c31, (93, 1619), _c31)
except Exception:
    pass
layout["Wed,_Apr_24"] = [93, 1619, 341, 1673]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/32_text_7_30AM_EDT.png
try:
    _c32 = get_crop(32, 254, 45)
    canvas.paste(_c32, (357, 1620), _c32)
except Exception:
    pass
layout["7:30AM_EDT"] = [357, 1620, 611, 1665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/33_text_450_Park_Ave_S.png
try:
    _c33 = get_crop(33, 284, 45)
    canvas.paste(_c33, (91, 1687), _c33)
except Exception:
    pass
layout["450_Park_Ave_S"] = [91, 1687, 375, 1732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/34_text_Explore.png
try:
    _c34 = get_crop(34, 172, 60)
    canvas.paste(_c34, (184, 1919), _c34)
except Exception:
    pass
layout["Explore"] = [184, 1919, 356, 1979]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_12_2024_4_23_19_27_45f56b06f31541079045047b6d542613-14/35_text_Schools.png
try:
    _c35 = get_crop(35, 177, 52)
    canvas.paste(_c35, (183, 1980), _c35)
except Exception:
    pass
layout["Schools"] = [183, 1980, 360, 2032]
