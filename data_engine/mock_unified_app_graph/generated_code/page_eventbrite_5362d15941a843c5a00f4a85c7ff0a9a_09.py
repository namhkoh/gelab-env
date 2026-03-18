# page_id: page_eventbrite_5362d15941a843c5a00f4a85c7ff0a9a_09
# screenshot: 2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11.png
# step_index: 9/12
# task: Open Eventbrite. Set the city to 'Los Angeles'. Search 'Business'. Filter 'French' speaking events. Add the first event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill (dominant very-light background)
draw.rectangle([(0, 0), (1440, 2960)], fill="#fbfbfc")

# Status bar area (top band, do NOT draw icons/text)
status_h = 96
draw.rectangle([(0, 0), (1440, status_h)], fill="#cfcfcf")

# Header / toolbar background (under the status bar)
header_top = status_h
header_bottom = 220
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#ffffff")
# subtle bottom divider for header
draw.line([(32, header_bottom), (1408, header_bottom)], fill="#e6e6e8", width=2)

# Light area under header for filter strip (background only, no pills/icons)
filter_strip_top = header_bottom + 12
filter_strip_bottom = filter_strip_top + 96
draw.rectangle([(0, filter_strip_top), (1440, filter_strip_bottom)], fill="#fbfdff")
# subtle separator under filters
draw.line([(32, filter_strip_bottom), (1408, filter_strip_bottom)], fill="#f0f0f2", width=1)

# Content separators
sep_y1 = 340
sep_y2 = 1160
sep_y3 = 1888
draw.line([(24, sep_y1), (1416, sep_y1)], fill="#f0f0f2", width=1)
draw.line([(24, sep_y2), (1416, sep_y2)], fill="#f0f0f2", width=1)
draw.line([(24, sep_y3), (1416, sep_y3)], fill="#f0f0f2", width=1)

# First event card background (rounded rectangle with subtle shadow)
card1_x0, card1_y0 = 48, 360
card1_x1, card1_y1 = 1392, 880
shadow_offset = 8
# shadow
draw.rounded_rectangle(
    [(card1_x0 + shadow_offset, card1_y0 + shadow_offset),
     (card1_x1 + shadow_offset, card1_y1 + shadow_offset)],
    radius=28, fill="#eef0f2"
)
# card background
draw.rounded_rectangle(
    [(card1_x0, card1_y0), (card1_x1, card1_y1)],
    radius=24, fill="#ffffff", outline="#e6e7eb", width=1
)

# Second event card background (rounded rectangle with subtle shadow)
card2_x0, card2_y0 = 48, 1240
card2_x1, card2_y1 = 1392, 1880
# shadow
draw.rounded_rectangle(
    [(card2_x0 + shadow_offset, card2_y0 + shadow_offset),
     (card2_x1 + shadow_offset, card2_y1 + shadow_offset)],
    radius=28, fill="#eef0f2"
)
# card background
draw.rounded_rectangle(
    [(card2_x0, card2_y0), (card2_x1, card2_y1)],
    radius=24, fill="#ffffff", outline="#e6e7eb", width=1
)

# Smaller promoted/label background area (sub-card) around mid-content (decorative only)
promo_x0, promo_y0 = 64, 1000
promo_x1, promo_y1 = 420, 1068
draw.rounded_rectangle([(promo_x0, promo_y0), (promo_x1, promo_y1)], radius=12, fill="#f6f8f7", outline="#e6e7eb")

# Large banner/background area behind second image (subtle band)
banner_x0, banner_y0 = 48, 1888
banner_x1, banner_y1 = 1392, 2360
draw.rectangle([(banner_x0, banner_y0), (banner_x1, banner_y1)], fill="#ffffff")
draw.line([(banner_x0 + 12, banner_y0), (banner_x1 - 12, banner_y0)], fill="#e8e9ec", width=1)

# Footer / bottom navigation background (do NOT draw icons)
nav_top = 2820
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#ffffff")
# top divider for nav
draw.line([(24, nav_top), (1416, nav_top)], fill="#e6e7eb", width=1)

# Subtle left and right page margins as visual guides (thin)
draw.line([(24, header_top), (24, 2960)], fill="#f3f4f6", width=1)
draw.line([(1416, header_top), (1416, 2960)], fill="#f3f4f6", width=1)

# Decorative subtle vertical rhythm lines between major sections (very faint)
for y in (520, 920, 1360, 1680, 2200):
    draw.line([(48, y), (1392, y)], fill="#fbfcfd", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 103)
    canvas.paste(_c1, (425, 410), _c1)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1036, 410), _c2)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/04_icon_6_00_to_B_00_PM_EST.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2415), _c4)
except Exception:
    pass
layout["6:00_to_B:00_PM_EST"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/05_icon_6_00_to_B_00_PM_EST.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 2415), _c5)
except Exception:
    pass
layout["6:00_to_B:00_PM_EST"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/06_icon_Foo.png
try:
    _c6 = get_crop(6, 136, 110)
    canvas.paste(_c6, (1284, 406), _c6)
except Exception:
    pass
layout["Foo"] = [1284, 406, 1420, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/07_icon_EcoMmcR.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1092, 1192), _c7)
except Exception:
    pass
layout["EcoMmcR"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/08_icon_EcoMmcR.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["EcoMmcR"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/09_icon_Close_current_screen.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 52, 64)
    canvas.paste(_c10, (1152, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1152, 1, 1204, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/11_icon_8.02.png
try:
    _c11 = get_crop(11, 122, 116)
    canvas.paste(_c11, (55, 113), _c11)
except Exception:
    pass
layout["8.02"] = [55, 113, 177, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/12_icon_Business.png
try:
    _c12 = get_crop(12, 66, 61)
    canvas.paste(_c12, (308, 1), _c12)
except Exception:
    pass
layout["Business"] = [308, 1, 374, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 100, 63)
    canvas.paste(_c13, (1212, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1212, 0, 1312, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/14_icon_8.02.png
try:
    _c14 = get_crop(14, 58, 62)
    canvas.paste(_c14, (182, 1), _c14)
except Exception:
    pass
layout["8.02"] = [182, 1, 240, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 52, 62)
    canvas.paste(_c15, (247, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [247, 1, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/16_icon_8.02.png
try:
    _c16 = get_crop(16, 56, 63)
    canvas.paste(_c16, (117, 0), _c16)
except Exception:
    pass
layout["8.02"] = [117, 0, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 54, 60)
    canvas.paste(_c17, (1319, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [1319, 1, 1373, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/18_icon_EcoMMERCE_TracK.png
try:
    _c18 = get_crop(18, 1344, 1175)
    canvas.paste(_c18, (48, 676), _c18)
except Exception:
    pass
layout["EcoMMERCE_TracK"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/19_icon_Los_Angeles.png
try:
    _c19 = get_crop(19, 492, 144)
    canvas.paste(_c19, (0, 259), _c19)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/20_icon_Business.png
try:
    _c20 = get_crop(20, 1344, 191)
    canvas.paste(_c20, (48, 72), _c20)
except Exception:
    pass
layout["Business"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/21_icon_6_00_PM_EDT.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["6:00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 49, 59)
    canvas.paste(_c22, (384, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [384, 3, 433, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/23_icon_Register_TODAY.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["Register_TODAY"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/24_icon_Tue_Apr_23.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (0, 2804), _c24)
except Exception:
    pass
layout["Tue,_Apr_23"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/25_icon_6_00_PM_EDT.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["6:00_PM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/26_icon_Promoted.png
try:
    _c26 = get_crop(26, 240, 67)
    canvas.paste(_c26, (87, 1742), _c26)
except Exception:
    pass
layout["Promoted"] = [87, 1742, 327, 1809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/27_icon_Register_TODAY.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (1152, 2804), _c27)
except Exception:
    pass
layout["Register_TODAY"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/28_icon_April_23_2024.png
try:
    _c28 = get_crop(28, 1344, 917)
    canvas.paste(_c28, (48, 1899), _c28)
except Exception:
    pass
layout["April_23,2024"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/29_text_8.02.png
try:
    _c29 = get_crop(29, 91, 43)
    canvas.paste(_c29, (20, 17), _c29)
except Exception:
    pass
layout["8.02"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/30_text_7_110_events.png
try:
    _c30 = get_crop(30, 359, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["7,110_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/31_text_MERcE_TrACK_JUNI.png
try:
    _c31 = get_crop(31, 307, 39)
    canvas.paste(_c31, (114, 704), _c31)
except Exception:
    pass
layout["MERcE_TrACK_JUNI"] = [114, 704, 421, 743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/32_text_Ecommerce_TracK_JUNI.png
try:
    _c32 = get_crop(32, 400, 103)
    canvas.paste(_c32, (425, 410), _c32)
except Exception:
    pass
layout["Ecommerce_TracK_JUNI"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/33_text_Ecommercetrack.png
try:
    _c33 = get_crop(33, 263, 30)
    canvas.paste(_c33, (849, 707), _c33)
except Exception:
    pass
layout["Ecommercetrack"] = [849, 707, 1112, 737]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/34_text_JUNI.png
try:
    _c34 = get_crop(34, 122, 39)
    canvas.paste(_c34, (1125, 704), _c34)
except Exception:
    pass
layout["JUNI"] = [1125, 704, 1247, 743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/35_text_FcO.png
try:
    _c35 = get_crop(35, 64, 30)
    canvas.paste(_c35, (1260, 707), _c35)
except Exception:
    pass
layout["FcO"] = [1260, 707, 1324, 737]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/36_text_Online.png
try:
    _c36 = get_crop(36, 129, 45)
    canvas.paste(_c36, (91, 1687), _c36)
except Exception:
    pass
layout["Online"] = [91, 1687, 220, 1732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/37_text_Free.png
try:
    _c37 = get_crop(37, 77, 39)
    canvas.paste(_c37, (117, 2614), _c37)
except Exception:
    pass
layout["Free"] = [117, 2614, 194, 2653]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/38_text_Level_Up_Your_Small_Business.png
try:
    _c38 = get_crop(38, 1344, 917)
    canvas.paste(_c38, (48, 1899), _c38)
except Exception:
    pass
layout["Level_Up_Your_Small_Busin"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/39_text_Tue_Apr_23.png
try:
    _c39 = get_crop(39, 288, 156)
    canvas.paste(_c39, (0, 2804), _c39)
except Exception:
    pass
layout["Tue,_Apr_23"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_09_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-11/40_text_6_00_PM_EDT.png
try:
    _c40 = get_crop(40, 288, 156)
    canvas.paste(_c40, (288, 2804), _c40)
except Exception:
    pass
layout["6:00_PM_EDT"] = [288, 2804, 576, 2960]
