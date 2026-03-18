# page_id: page_eventbrite_5f8371b476c64aeeb04a6d8281b87f4d_02
# screenshot: 2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4.png
# step_index: 2/7
# task: Open Eventbrite. Search Science & Tech event. Select the first one that is not promoted. If it is free, add it to Favorites. If it is not free, record its price in Google Keep Notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (canvas already white, but ensure fill)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Top status bar
status_h = 64
draw.rectangle([(0, 0), (1440, status_h)], fill=(189, 189, 189))  # muted grey status bar

# Header / search area background (subtle off-white to separate from page)
header_top = status_h
header_bottom = 160
padding_h = 48
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(250, 250, 250))

# Search field container (rounded) spanning content width, behind auto-pasted search elements
content_left = 48
content_right = 1440 - 48
search_top = header_top + 8
search_bottom = header_top + 120
draw.rounded_rectangle(
    [(content_left, search_top), (content_right, search_bottom)],
    radius=12,
    fill=(255, 255, 255),
    outline=None
)

# Blue underline below search field (prominent accent)
underline_y = search_bottom + 0
draw.line([(content_left, underline_y), (content_right, underline_y)], fill=(47, 91, 232), width=4)

# Subtle divider under the header area
draw.line([(0, header_bottom), (1440, header_bottom)], fill=(230, 230, 230), width=1)

# Section card background for the "Recent" list area (very light)
recent_top = header_bottom + 16
# keep it white but use a faint off-white band to indicate grouping
draw.rectangle([(0, recent_top), (1440, recent_top + 48)], fill=(255, 255, 255))

# Draw separators between list rows (use the detected clickable rows positions, draw thin lines across content width)
row_info = [
    (48, 390, 1344, 144),
    (48, 534, 1344, 144),
    (48, 678, 1344, 144),
    (48, 822, 1344, 144),
    (48, 966, 1344, 144),
    (48, 1110, 1344, 144),
    (48, 1254, 1344, 144),
    (48, 1398, 1344, 144),
]
sep_color = (230, 230, 235)
for (x, y, w, h) in row_info:
    sep_y = y + h - 1
    draw.line([(content_left, sep_y), (content_right, sep_y)], fill=sep_color, width=1)

# Add light left edge guides for rows (subtle vertical line to align content area)
draw.line([(content_left, recent_top), (content_left, 1600)], fill=(245, 245, 245), width=1)
draw.line([(content_right, recent_top), (content_right, 1600)], fill=(245, 245, 245), width=1)

# Bottom navigation bar background and top divider
nav_top = 2804
draw.line([(0, nav_top), (1440, nav_top)], fill=(225, 225, 230), width=1)
draw.rectangle([(0, nav_top), (1440, 2960)], fill=(250, 250, 252))

# Slight subtle shadow above the nav bar (thin)
draw.line([(0, nav_top - 4), (1440, nav_top - 4)], fill=(240, 240, 242), width=1)

# Final faint vertical rhythm lines to suggest card/list spacing (very subtle)
for y in range(220, 1600, 144):
    # do not draw strong lines over content, keep them extremely light
    draw.line([(content_left + 4, y), (content_right - 4, y)], fill=(248, 248, 249), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/00_icon_9.37.png
try:
    _c0 = get_crop(0, 58, 64)
    canvas.paste(_c0, (181, 0), _c0)
except Exception:
    pass
layout["9.37"] = [181, 0, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/01_icon_9.37.png
try:
    _c1 = get_crop(1, 57, 63)
    canvas.paste(_c1, (113, 1), _c1)
except Exception:
    pass
layout["9.37"] = [113, 1, 170, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 58, 62)
    canvas.paste(_c2, (245, 2), _c2)
except Exception:
    pass
layout["icon_2"] = [245, 2, 303, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/03_icon_Cancel.png
try:
    _c3 = get_crop(3, 99, 62)
    canvas.paste(_c3, (1212, 0), _c3)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1311, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/04_icon_Search_forae.png
try:
    _c4 = get_crop(4, 55, 62)
    canvas.paste(_c4, (315, 2), _c4)
except Exception:
    pass
layout["Search_forae"] = [315, 2, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 56, 62)
    canvas.paste(_c5, (1317, 0), _c5)
except Exception:
    pass
layout["Cancel"] = [1317, 0, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/06_icon_9.37.png
try:
    _c6 = get_crop(6, 128, 114)
    canvas.paste(_c6, (51, 115), _c6)
except Exception:
    pass
layout["9.37"] = [51, 115, 179, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/07_icon_Basketball.png
try:
    _c7 = get_crop(7, 1344, 144)
    canvas.paste(_c7, (48, 534), _c7)
except Exception:
    pass
layout["Basketball"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/08_icon_Cancel.png
try:
    _c8 = get_crop(8, 149, 144)
    canvas.paste(_c8, (1243, 97), _c8)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/09_icon_Search_forae.png
try:
    _c9 = get_crop(9, 49, 63)
    canvas.paste(_c9, (383, 1), _c9)
except Exception:
    pass
layout["Search_forae"] = [383, 1, 432, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/10_icon_Close_current_screen.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 822), _c10)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/11_icon_Favorites.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (576, 2804), _c11)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/12_icon_Search_forae.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 390), _c12)
except Exception:
    pass
layout["Search_forae"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 534), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 678), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/15_icon_Exhibition.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 678), _c15)
except Exception:
    pass
layout["Exhibition"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/16_icon_Tickets.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (864, 2804), _c16)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 1254), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/18_icon_Close_current_screen.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 1398), _c18)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/19_icon_Home.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (0, 2804), _c19)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/20_icon_Search_events.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (288, 2804), _c20)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/21_icon_Cancel.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1248, 390), _c21)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/22_icon_Close_current_screen.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1248, 1110), _c22)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/23_icon_Close_current_screen.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1248, 966), _c23)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/24_icon_Search_forae.png
try:
    _c24 = get_crop(24, 1344, 191)
    canvas.paste(_c24, (48, 72), _c24)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/25_icon_More.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (1152, 2804), _c25)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/26_text_9.37.png
try:
    _c26 = get_crop(26, 89, 43)
    canvas.paste(_c26, (20, 17), _c26)
except Exception:
    pass
layout["9.37"] = [20, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/27_text_Recent.png
try:
    _c27 = get_crop(27, 200, 56)
    canvas.paste(_c27, (46, 301), _c27)
except Exception:
    pass
layout["Recent"] = [46, 301, 246, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/28_text_Festival.png
try:
    _c28 = get_crop(28, 156, 50)
    canvas.paste(_c28, (162, 872), _c28)
except Exception:
    pass
layout["Festival"] = [162, 872, 318, 922]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/29_text_Taylor_Swift.png
try:
    _c29 = get_crop(29, 229, 57)
    canvas.paste(_c29, (161, 1015), _c29)
except Exception:
    pass
layout["Taylor_Swift"] = [161, 1015, 390, 1072]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/30_text_Talkshow.png
try:
    _c30 = get_crop(30, 181, 52)
    canvas.paste(_c30, (163, 1160), _c30)
except Exception:
    pass
layout["Talkshow"] = [163, 1160, 344, 1212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/31_text_Broadway.png
try:
    _c31 = get_crop(31, 195, 56)
    canvas.paste(_c31, (163, 1303), _c31)
except Exception:
    pass
layout["Broadway"] = [163, 1303, 358, 1359]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/32_text_Football.png
try:
    _c32 = get_crop(32, 159, 43)
    canvas.paste(_c32, (165, 1451), _c32)
except Exception:
    pass
layout["Football"] = [165, 1451, 324, 1494]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/33_clickable_Festival.png
try:
    _c33 = get_crop(33, 1344, 144)
    canvas.paste(_c33, (48, 822), _c33)
except Exception:
    pass
layout["Festival"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/34_clickable_Taylor_Swift.png
try:
    _c34 = get_crop(34, 1344, 144)
    canvas.paste(_c34, (48, 966), _c34)
except Exception:
    pass
layout["Taylor_Swift"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/35_clickable_Talkshow.png
try:
    _c35 = get_crop(35, 1344, 144)
    canvas.paste(_c35, (48, 1110), _c35)
except Exception:
    pass
layout["Talkshow"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/36_clickable_Broadway.png
try:
    _c36 = get_crop(36, 1344, 144)
    canvas.paste(_c36, (48, 1254), _c36)
except Exception:
    pass
layout["Broadway"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_02_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-4/37_clickable_Football.png
try:
    _c37 = get_crop(37, 1344, 144)
    canvas.paste(_c37, (48, 1398), _c37)
except Exception:
    pass
layout["Football"] = [48, 1398, 1392, 1542]
