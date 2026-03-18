# page_id: page_eventbrite_cee80d2c889b45cd86e30248baf590b8_07
# screenshot: 2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9.png
# step_index: 7/13
# task: Open Eventbrite. Search Food & Drink party event in New York. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the page
# Available variables: canvas (PIL Image), draw (PIL ImageDraw), font_sm, font_md, font_lg, font_xl

# Full white background (canvas already white, but ensure)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar (top ~56px) - muted grey
status_h = 56
draw.rectangle([(0, 0), (1440, status_h)], fill=(150, 150, 150))
# subtle divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill=(130, 130, 130), width=1)

# Header / Search area backdrop (keeps it visually separate from status bar)
header_top = status_h
header_bottom = 200
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))
# Blue underline for the search field (prominent accent)
underline_y = 160
draw.line([(48, underline_y), (1392, underline_y)], fill=(26, 115, 232), width=4)

# Thin subtle top border for header (to separate from status bar)
draw.line([(0, header_top), (1440, header_top)], fill=(200, 200, 200), width=1)

# Section area background (a very subtle card area behind the "Recent" list)
section_x0 = 32
section_x1 = 1408
section_y0 = 260
section_y1 = 1700
# soft off-white card with slight rounded corners
draw.rounded_rectangle([(section_x0, section_y0), (section_x1, section_y1)],
                       radius=8, fill=(250, 250, 251), outline=None)

# Separator lines between list items (use detected clickable item positions)
item_tops = [534, 678, 822, 966, 1110, 1254, 1398, 1542]
item_height = 144
sep_color = (233, 233, 236)
for top in item_tops:
    # draw bottom separator for each item
    y = top + item_height
    # ensure separator is inside the section card bounds
    if section_y0 < y < section_y1:
        draw.line([(48, y), (1392, y)], fill=sep_color, width=1)

# Left column guide (subtle) to align list item icons/text visually
draw.line([(48, section_y0 + 8), (48, section_y1 - 8)], fill=(245, 245, 247), width=1)

# Right edge subtle column guide (to visually align the remove icons on the right)
draw.line([(1248, section_y0 + 8), (1248, section_y1 - 8)], fill=(245, 245, 247), width=1)

# Bottom navigation bar area (approx 2804px - bottom)
nav_top = 2804
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill=(247, 247, 248))
# top divider of nav bar
draw.line([(0, nav_top), (1440, nav_top)], fill=(220, 220, 222), width=1)

# Subtle shadow above nav bar to lift it visually
for i, alpha in enumerate([18, 12, 6], start=0):
    y = nav_top - (i + 1)
    draw.line([(0, y), (1440, y)], fill=(0, 0, 0, 0), width=1)  # kept as no-op color to avoid drawing icons

# Decorative faint vertical divider near the left edge for content rhythm
draw.line([(140, section_y0 + 20), (140, section_y1 - 20)], fill=(250, 250, 252), width=1)

# Accent small rounded rectangle behind the "Recent" header area (no text drawn)
recent_card = (40, 280, 420, 340)
draw.rounded_rectangle([recent_card[0:2], recent_card[2:4]], radius=6, fill=(255, 255, 255), outline=(238, 238, 241))

# A subtle large empty content area (placeholder background) below the list to indicate scrollable space
content_bg_top = section_y1 + 10
content_bg_bottom = nav_top - 10
draw.rectangle([(48, content_bg_top), (1392, content_bg_bottom)], fill=(255, 255, 255))

# Final thin outer frame to give the screen slight edge
draw.rectangle([(0, 0), (1439, 2959)], outline=(240, 240, 243), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 49, 67)
    canvas.paste(_c0, (1154, 1), _c0)
except Exception:
    pass
layout["icon_0"] = [1154, 1, 1203, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/01_icon_9.44.png
try:
    _c1 = get_crop(1, 57, 61)
    canvas.paste(_c1, (180, 2), _c1)
except Exception:
    pass
layout["9.44"] = [180, 2, 237, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/02_icon_9.44.png
try:
    _c2 = get_crop(2, 53, 63)
    canvas.paste(_c2, (116, 1), _c2)
except Exception:
    pass
layout["9.44"] = [116, 1, 169, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/03_icon_Cancel.png
try:
    _c3 = get_crop(3, 99, 64)
    canvas.paste(_c3, (1212, 1), _c3)
except Exception:
    pass
layout["Cancel"] = [1212, 1, 1311, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 56, 60)
    canvas.paste(_c4, (246, 3), _c4)
except Exception:
    pass
layout["icon_4"] = [246, 3, 302, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 54, 62)
    canvas.paste(_c5, (1318, 1), _c5)
except Exception:
    pass
layout["Cancel"] = [1318, 1, 1372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 54, 61)
    canvas.paste(_c6, (315, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [315, 3, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/07_icon_9.44.png
try:
    _c7 = get_crop(7, 126, 108)
    canvas.paste(_c7, (49, 116), _c7)
except Exception:
    pass
layout["9.44"] = [49, 116, 175, 224]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/08_icon_Science_Tech.png
try:
    _c8 = get_crop(8, 1344, 144)
    canvas.paste(_c8, (48, 534), _c8)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/09_icon_Cancel.png
try:
    _c9 = get_crop(9, 149, 144)
    canvas.paste(_c9, (1243, 97), _c9)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/10_icon_FSearch_fora..png
try:
    _c10 = get_crop(10, 1344, 191)
    canvas.paste(_c10, (48, 72), _c10)
except Exception:
    pass
layout["FSearch_fora."] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/11_icon_FSearch_fora..png
try:
    _c11 = get_crop(11, 48, 62)
    canvas.paste(_c11, (383, 2), _c11)
except Exception:
    pass
layout["FSearch_fora."] = [383, 2, 431, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/12_icon_Close_current_screen.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 822), _c12)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 534), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 678), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/15_icon_Favorites.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (576, 2804), _c15)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 1254), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 1110), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/18_icon_Tickets.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (864, 2804), _c18)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/19_icon_Close_current_screen.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 1542), _c19)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/20_icon_Cancel.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 390), _c20)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/21_icon_Close_current_screen.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1248, 1398), _c21)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/22_icon_Search_events.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/23_icon_Home.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (0, 2804), _c23)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/24_icon_Close_current_screen.png
try:
    _c24 = get_crop(24, 144, 144)
    canvas.paste(_c24, (1248, 966), _c24)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/25_icon_More.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (1152, 2804), _c25)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/26_icon_Science_Tech.png
try:
    _c26 = get_crop(26, 1344, 144)
    canvas.paste(_c26, (48, 390), _c26)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/27_icon_Basketball.png
try:
    _c27 = get_crop(27, 1344, 144)
    canvas.paste(_c27, (48, 678), _c27)
except Exception:
    pass
layout["Basketball"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/28_icon_Science_Tech.png
try:
    _c28 = get_crop(28, 1344, 144)
    canvas.paste(_c28, (48, 390), _c28)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/29_text_9.44.png
try:
    _c29 = get_crop(29, 94, 43)
    canvas.paste(_c29, (20, 15), _c29)
except Exception:
    pass
layout["9.44"] = [20, 15, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/30_text_Recent.png
try:
    _c30 = get_crop(30, 203, 62)
    canvas.paste(_c30, (45, 299), _c30)
except Exception:
    pass
layout["Recent"] = [45, 299, 248, 361]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/31_text_Exhibition.png
try:
    _c31 = get_crop(31, 191, 50)
    canvas.paste(_c31, (164, 872), _c31)
except Exception:
    pass
layout["Exhibition"] = [164, 872, 355, 922]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/32_text_Festival.png
try:
    _c32 = get_crop(32, 157, 52)
    canvas.paste(_c32, (161, 1015), _c32)
except Exception:
    pass
layout["Festival"] = [161, 1015, 318, 1067]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/33_text_Taylor_Swift.png
try:
    _c33 = get_crop(33, 229, 57)
    canvas.paste(_c33, (161, 1158), _c33)
except Exception:
    pass
layout["Taylor_Swift"] = [161, 1158, 390, 1215]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/34_text_Talkshow.png
try:
    _c34 = get_crop(34, 179, 48)
    canvas.paste(_c34, (164, 1304), _c34)
except Exception:
    pass
layout["Talkshow"] = [164, 1304, 343, 1352]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/35_text_Broadway.png
try:
    _c35 = get_crop(35, 195, 57)
    canvas.paste(_c35, (163, 1446), _c35)
except Exception:
    pass
layout["Broadway"] = [163, 1446, 358, 1503]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/36_text_Football.png
try:
    _c36 = get_crop(36, 159, 43)
    canvas.paste(_c36, (165, 1597), _c36)
except Exception:
    pass
layout["Football"] = [165, 1597, 324, 1640]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/37_clickable_Exhibition.png
try:
    _c37 = get_crop(37, 1344, 144)
    canvas.paste(_c37, (48, 822), _c37)
except Exception:
    pass
layout["Exhibition"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/38_clickable_Festival.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 966), _c38)
except Exception:
    pass
layout["Festival"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/39_clickable_Taylor_Swift.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 1110), _c39)
except Exception:
    pass
layout["Taylor_Swift"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/40_clickable_Talkshow.png
try:
    _c40 = get_crop(40, 1344, 144)
    canvas.paste(_c40, (48, 1254), _c40)
except Exception:
    pass
layout["Talkshow"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/41_clickable_Broadway.png
try:
    _c41 = get_crop(41, 1344, 144)
    canvas.paste(_c41, (48, 1398), _c41)
except Exception:
    pass
layout["Broadway"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_07_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-9/42_clickable_Football.png
try:
    _c42 = get_crop(42, 1344, 144)
    canvas.paste(_c42, (48, 1542), _c42)
except Exception:
    pass
layout["Football"] = [48, 1542, 1392, 1686]
