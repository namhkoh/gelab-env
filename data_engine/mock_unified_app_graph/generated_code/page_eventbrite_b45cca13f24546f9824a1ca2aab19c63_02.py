# page_id: page_eventbrite_b45cca13f24546f9824a1ca2aab19c63_02
# screenshot: 2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4.png
# step_index: 2/11
# task: Open Eventbrite. Search for "Art". Filter for events in New York. Select first recommended event. Save it to wishlist. What is the duration of the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill=(143, 143, 143))  # muted grey status bar

# subtle inner line under status bar
draw.line([(0, status_h), (1440, status_h)], fill=(200, 200, 200), width=1)

# Header side gutters around the search crop area (avoid drawing inside detected search crop at x=48..1392, y=72..263)
# left gutter strip
draw.rectangle([(0, 72), (48, 263)], fill=(255, 255, 255))
# right gutter strip
draw.rectangle([(1392, 72), (1440, 263)], fill=(255, 255, 255))

# A thin subtle background band under the search area to visually separate header from content (below search crop)
band_y0 = 263
band_y1 = 300
draw.rectangle([(0, band_y0), (1440, band_y1)], fill=(250, 250, 250))

# Light divider line under the band
draw.line([(48, band_y1), (1392, band_y1)], fill=(230, 230, 230), width=1)

# Content area subtle left margin guideline (visual structure only, narrow and light)
draw.line([(48, band_y1 + 24), (48, 2600)], fill=(245, 245, 245), width=2)

# Right content margin guideline
draw.line([(1392, band_y1 + 24), (1392, 2600)], fill=(245, 245, 245), width=2)

# Large whitespace area remains white (don't draw over detected list item bounding boxes)

# Bottom navigation bar background (detected nav icons are pasted on top; draw the bar and its separators)
nav_top = 2804
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill=(255, 255, 255))

# top border of nav bar
draw.line([(0, nav_top), (1440, nav_top)], fill=(226, 226, 226), width=2)

# subtle shadow above nav bar for depth
for i, a in enumerate([48, 32, 16, 8]):
    y = nav_top - (i + 1) * 2
    alpha = a
    # emulate semi-transparent shadow by mixing with white background (lighter grey)
    shade = 240 - i * 6
    draw.line([(0, y), (1440, y)], fill=(shade, shade, shade), width=1)

# Bottom-most hairline
draw.line([(0, nav_bottom-1), (1440, nav_bottom-1)], fill=(230, 230, 230), width=1)

# Left and right outer container thin borders to frame the UI
draw.line([(0, 0), (0, 2960)], fill=(245,245,245), width=1)
draw.line([(1439, 0), (1439, 2960)], fill=(245,245,245), width=1)

# Small rounded card placeholders in lower content area (well below detected rows) to suggest additional sections
card_w = 1344
card_x0 = 48
card_x1 = card_x0 + card_w
card_h = 140
card_y = 1840
draw.rounded_rectangle([(card_x0, card_y), (card_x1, card_y + card_h)], radius=12, outline=(235,235,235), width=1, fill=(255,255,255))

card_y += card_h + 28
draw.rounded_rectangle([(card_x0, card_y), (card_x1, card_y + card_h)], radius=12, outline=(235,235,235), width=1, fill=(255,255,255))

# small guiding divider near top of content (but outside detected search crop)
guideline_y = band_y1 + 12
draw.line([(60, guideline_y), (1380, guideline_y)], fill=(230,230,230), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/00_icon_7.04.png
try:
    _c0 = get_crop(0, 58, 63)
    canvas.paste(_c0, (181, 0), _c0)
except Exception:
    pass
layout["7.04"] = [181, 0, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/01_icon_7.04.png
try:
    _c1 = get_crop(1, 60, 63)
    canvas.paste(_c1, (114, 1), _c1)
except Exception:
    pass
layout["7.04"] = [114, 1, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/02_icon_Search_forae.png
try:
    _c2 = get_crop(2, 64, 64)
    canvas.paste(_c2, (309, 1), _c2)
except Exception:
    pass
layout["Search_forae"] = [309, 1, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 50, 62)
    canvas.paste(_c3, (248, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [248, 2, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/04_icon_Food_Drink.png
try:
    _c4 = get_crop(4, 1344, 144)
    canvas.paste(_c4, (48, 534), _c4)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 58, 62)
    canvas.paste(_c5, (1316, 0), _c5)
except Exception:
    pass
layout["Cancel"] = [1316, 0, 1374, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/06_icon_Cancel.png
try:
    _c6 = get_crop(6, 99, 62)
    canvas.paste(_c6, (1212, 0), _c6)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1311, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/07_icon_Cancel.png
try:
    _c7 = get_crop(7, 149, 144)
    canvas.paste(_c7, (1243, 97), _c7)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/08_icon_Food_Drink.png
try:
    _c8 = get_crop(8, 1344, 144)
    canvas.paste(_c8, (48, 678), _c8)
except Exception:
    pass
layout["Food_&_Drink"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/09_icon_Search_forae.png
try:
    _c9 = get_crop(9, 1344, 144)
    canvas.paste(_c9, (48, 390), _c9)
except Exception:
    pass
layout["Search_forae"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/10_icon_Close_current_screen.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 822), _c10)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/11_icon_7.04.png
try:
    _c11 = get_crop(11, 126, 113)
    canvas.paste(_c11, (53, 115), _c11)
except Exception:
    pass
layout["7.04"] = [53, 115, 179, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/12_icon_Favorites.png
try:
    _c12 = get_crop(12, 288, 156)
    canvas.paste(_c12, (576, 2804), _c12)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 1254), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 534), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/15_icon_Close_current_screen.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1248, 1110), _c15)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 678), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 1398), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/18_icon_Tickets.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (864, 2804), _c18)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/19_icon_Cancel.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 390), _c19)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 1686), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/21_icon_Science_Tech.png
try:
    _c21 = get_crop(21, 1344, 144)
    canvas.paste(_c21, (48, 822), _c21)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/22_icon_Search_forae.png
try:
    _c22 = get_crop(22, 48, 65)
    canvas.paste(_c22, (383, 2), _c22)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 431, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/23_icon_Close_current_screen.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1248, 1542), _c23)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/24_icon_Home.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (0, 2804), _c24)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/25_icon_Close_current_screen.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1248, 966), _c25)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/26_icon_Search_events.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (288, 2804), _c26)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/27_icon_Science_Tech.png
try:
    _c27 = get_crop(27, 1344, 144)
    canvas.paste(_c27, (48, 966), _c27)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/28_icon_Search_forae.png
try:
    _c28 = get_crop(28, 1344, 191)
    canvas.paste(_c28, (48, 72), _c28)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/29_icon_More.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (1152, 2804), _c29)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/30_icon_Basketball.png
try:
    _c30 = get_crop(30, 1344, 144)
    canvas.paste(_c30, (48, 1110), _c30)
except Exception:
    pass
layout["Basketball"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/31_icon_Taylor_Swift.png
try:
    _c31 = get_crop(31, 1344, 144)
    canvas.paste(_c31, (48, 1542), _c31)
except Exception:
    pass
layout["Taylor_Swift"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/32_icon_Exhibition.png
try:
    _c32 = get_crop(32, 1344, 144)
    canvas.paste(_c32, (48, 1254), _c32)
except Exception:
    pass
layout["Exhibition"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/33_icon_Taylor_Swift.png
try:
    _c33 = get_crop(33, 1344, 144)
    canvas.paste(_c33, (48, 1398), _c33)
except Exception:
    pass
layout["Taylor_Swift"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/34_icon_Talkshow.png
try:
    _c34 = get_crop(34, 117, 130)
    canvas.paste(_c34, (25, 1696), _c34)
except Exception:
    pass
layout["Talkshow"] = [25, 1696, 142, 1826]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/35_text_Recent.png
try:
    _c35 = get_crop(35, 200, 56)
    canvas.paste(_c35, (46, 301), _c35)
except Exception:
    pass
layout["Recent"] = [46, 301, 246, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/36_text_Talkshow.png
try:
    _c36 = get_crop(36, 177, 45)
    canvas.paste(_c36, (165, 1738), _c36)
except Exception:
    pass
layout["Talkshow"] = [165, 1738, 342, 1783]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_02_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-4/37_clickable_Talkshow.png
try:
    _c37 = get_crop(37, 1344, 144)
    canvas.paste(_c37, (48, 1686), _c37)
except Exception:
    pass
layout["Talkshow"] = [48, 1686, 1392, 1830]
