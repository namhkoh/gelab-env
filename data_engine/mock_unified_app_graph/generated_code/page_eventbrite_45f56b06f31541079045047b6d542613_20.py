# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_20
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-22.png
# step_index: 20/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint overall background (slightly off-white to match screenshot)
draw.rectangle([(0, 0), canvas.size], fill="#FBFCFD")

w, h = canvas.size

# Status bar background (top ~96px)
status_h = 96
draw.rectangle([(0, 0), (w, status_h)], fill="#BFBFBF")

# Subtle top hairline and a faint inner highlight for status bar
draw.line([(0, status_h), (w, status_h)], fill="#E9E9E9", width=1)

# Header / toolbar area (search title area)
header_top = status_h
header_bottom = 220
draw.rectangle([(0, header_top), (w, header_bottom)], fill="#FFFFFF")
# Divider under header
draw.line([(48, header_bottom), (w - 48, header_bottom)], fill="#E6E6E6", width=2)

# Filters row divider (separating filter chips area from results count)
filters_div_y = 520
draw.line([(48, filters_div_y), (w - 48, filters_div_y)], fill="#F0F0F0", width=1)

# Draw first event card background with subtle shadow
# (placed behind the image + metadata; image and overlays will be pasted later)
card1_x = 48
card1_y = 640
card1_w = 1344
card1_h = 1175 + 220  # include some extra area for text block below image
card1_box = (card1_x, card1_y, card1_x + card1_w, card1_y + card1_h)

# Shadow: slightly larger rounded rectangle behind card
shadow_offset = 12
draw.rounded_rectangle(
    [
        (card1_box[0] + shadow_offset, card1_box[1] + shadow_offset),
        (card1_box[2] + shadow_offset, card1_box[3] + shadow_offset)
    ],
    radius=28,
    fill="#EDEFF1"
)

# Card background (white)
draw.rounded_rectangle([ (card1_box[0], card1_box[1]), (card1_box[2], card1_box[3]) ],
                       radius=24, fill="#FFFFFF")

# Separator line between image area and text inside the card (subtle)
img_bottom_y = card1_y + 520  # approximate image height area within card
draw.line([(card1_x + 24, img_bottom_y), (card1_x + card1_w - 24, img_bottom_y)], fill="#F5F6F7", width=1)

# Draw second event card background (further down the page)
card2_x = 48
card2_y = 1888
card2_w = 1344
card2_h = 920
card2_box = (card2_x, card2_y, card2_x + card2_w, card2_y + card2_h)

# Shadow for second card
draw.rounded_rectangle(
    [
        (card2_box[0] + shadow_offset, card2_box[1] + shadow_offset),
        (card2_box[2] + shadow_offset, card2_box[3] + shadow_offset)
    ],
    radius=22,
    fill="#ECEFF2"
)

draw.rounded_rectangle([ (card2_box[0], card2_box[1]), (card2_box[2], card2_box[3]) ],
                       radius=20, fill="#FFFFFF")

# Small light divider between cards and following content
between_y = card2_box[1] + card2_h + 24
draw.line([(48, between_y), (w - 48, between_y)], fill="#EFEFF1", width=1)

# Floating content banner backgrounds (subtle pastel bars behind categories)
# Placeholders only for non-detected decorative banners (kept minimal and not overlapping detected chip positions)
banner_w = 420
banner_h = 56
# Example banner left of first card's top area (kept out of detected icon bounds)
draw.rounded_rectangle([(60, card1_y - 80), (60 + banner_w, card1_y - 80 + banner_h)],
                       radius=16, fill="#F5FAFF")
draw.line([(60, card1_y - 24), (60 + banner_w, card1_y - 24)], fill="#E2F0FF", width=1)

# Main content area separator line before bottom navigation
nav_sep_y = h - 160
draw.line([(0, nav_sep_y), (w, nav_sep_y)], fill="#E6E6E6", width=2)

# Bottom navigation bar background
nav_h = h - nav_sep_y
draw.rectangle([(0, nav_sep_y), (w, h)], fill="#FFFFFF")

# Slight top shadow for nav bar (to separate from content)
draw.line([(48, nav_sep_y), (w - 48, nav_sep_y)], fill="#DDDDDD", width=1)

# Final subtle global vignette / soft edges (very light)
# Draw faint rectangles at left and right edges to mimic slight canvas depth
edge_w = 24
draw.rectangle([(0, 0), (edge_w, h)], fill="#FBFBFC")
draw.rectangle([(w - edge_w, 0), (w, h)], fill="#FBFBFC")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (1054, 410), _c0)
except Exception:
    pass
layout["Music"] = [1054, 410, 1241, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/01_icon_May_03_-_06_2024.png
try:
    _c1 = get_crop(1, 584, 103)
    canvas.paste(_c1, (458, 410), _c1)
except Exception:
    pass
layout["May_03_-_06,_2024"] = [458, 410, 1042, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/02_icon_2_Filters.png
try:
    _c2 = get_crop(2, 392, 103)
    canvas.paste(_c2, (54, 410), _c2)
except Exception:
    pass
layout["2_Filters"] = [54, 410, 446, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/03_icon_Favorite_button.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 1192), _c3)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/04_icon_Busin.png
try:
    _c4 = get_crop(4, 139, 103)
    canvas.paste(_c4, (1253, 410), _c4)
except Exception:
    pass
layout["Busin"] = [1253, 410, 1392, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/05_icon_Overflow_menu_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 1192), _c5)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/06_icon_Interactive_Live_Music_and_Jam_Session_a.png
try:
    _c6 = get_crop(6, 1344, 1175)
    canvas.paste(_c6, (48, 676), _c6)
except Exception:
    pass
layout["Interactive_Live_Music_an"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 2415), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/08_icon_Close_current_screen.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1248, 96), _c8)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/09_icon_7.29.png
try:
    _c9 = get_crop(9, 125, 117)
    canvas.paste(_c9, (55, 111), _c9)
except Exception:
    pass
layout["7.29"] = [55, 111, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 48, 66)
    canvas.paste(_c10, (1154, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1154, 0, 1202, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 65, 63)
    canvas.paste(_c11, (308, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [308, 1, 373, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/12_icon_7.29.png
try:
    _c12 = get_crop(12, 58, 63)
    canvas.paste(_c12, (181, 1), _c12)
except Exception:
    pass
layout["7.29"] = [181, 1, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/13_icon_Favorite_button.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1092, 2415), _c13)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/14_icon_7.29.png
try:
    _c14 = get_crop(14, 58, 64)
    canvas.paste(_c14, (115, 0), _c14)
except Exception:
    pass
layout["7.29"] = [115, 0, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 97, 66)
    canvas.paste(_c15, (1213, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1213, 0, 1310, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 48, 62)
    canvas.paste(_c16, (251, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [251, 1, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 52, 65)
    canvas.paste(_c17, (1319, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1319, 0, 1371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/18_icon_A_Monumental.png
try:
    _c18 = get_crop(18, 1344, 917)
    canvas.paste(_c18, (48, 1899), _c18)
except Exception:
    pass
layout["A_Monumental"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/19_icon_Breakthrouah_in_Mental_Health.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["Breakthrouah_in_Mental_He"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/20_icon_Yoga_session.png
try:
    _c20 = get_crop(20, 1344, 191)
    canvas.paste(_c20, (48, 72), _c20)
except Exception:
    pass
layout["Yoga_session"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/21_icon_Yoga_session.png
try:
    _c21 = get_crop(21, 47, 63)
    canvas.paste(_c21, (384, 1), _c21)
except Exception:
    pass
layout["Yoga_session"] = [384, 1, 431, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/22_icon_Free.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (0, 2804), _c22)
except Exception:
    pass
layout["Free"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/23_icon_New_York.png
try:
    _c23 = get_crop(23, 434, 144)
    canvas.paste(_c23, (0, 259), _c23)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/24_icon_Free.png
try:
    _c24 = get_crop(24, 128, 77)
    canvas.paste(_c24, (90, 2592), _c24)
except Exception:
    pass
layout["Free"] = [90, 2592, 218, 2669]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/25_icon_Breakthrouah_in_Mental_Health.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["Breakthrouah_in_Mental_He"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/26_icon_Promoted.png
try:
    _c26 = get_crop(26, 242, 65)
    canvas.paste(_c26, (85, 1744), _c26)
except Exception:
    pass
layout["Promoted"] = [85, 1744, 327, 1809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/27_icon_A_Monumental.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (864, 2804), _c27)
except Exception:
    pass
layout["A_Monumental"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/28_icon_pinkFROG_cafe.png
try:
    _c28 = get_crop(28, 285, 62)
    canvas.paste(_c28, (88, 1679), _c28)
except Exception:
    pass
layout["pinkFROG_cafe"] = [88, 1679, 373, 1741]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/29_icon_More.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (1152, 2804), _c29)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/30_icon_7.29.png
try:
    _c30 = get_crop(30, 104, 64)
    canvas.paste(_c30, (10, 0), _c30)
except Exception:
    pass
layout["7.29"] = [10, 0, 114, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_20_2024_4_23_19_27_45f56b06f31541079045047b6d542613-22/31_text_409_events.png
try:
    _c31 = get_crop(31, 392, 103)
    canvas.paste(_c31, (54, 410), _c31)
except Exception:
    pass
layout["409_events"] = [54, 410, 446, 513]
