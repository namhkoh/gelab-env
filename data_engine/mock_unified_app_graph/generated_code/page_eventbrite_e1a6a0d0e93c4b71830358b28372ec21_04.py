# page_id: page_eventbrite_e1a6a0d0e93c4b71830358b28372ec21_04
# screenshot: 2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6.png
# step_index: 4/9
# task: Open Eventbrite. Search for "Language Learning". Filter only online events. Note how many events are available for "Spanish".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the mobile UI page
# Available variables: canvas (PIL Image), draw (PIL ImageDraw), font_sm, font_md, font_lg, font_xl

# Page background
draw.rectangle([(0, 0), (1440, 2960)], fill=(247, 249, 252))  # very light cool gray background

# Status bar (top ~50px)
status_h = 56
draw.rectangle([(0, 0), (1440, status_h)], fill=(173, 173, 173))  # muted gray status bar
draw.line([(0, status_h), (1440, status_h)], fill=(200, 200, 200), width=1)  # bottom border of status bar

# Header / Title area (below status bar)
header_top = status_h
header_bottom = 180
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))  # white header background
# subtle divider under header
draw.line([(32, header_bottom), (1408, header_bottom)], fill=(225, 226, 230), width=1)

# Light search / content divider line (thin)
divider_y = 220
draw.line([(24, divider_y), (1416, divider_y)], fill=(234, 235, 238), width=1)

# Filters area background band (behind filter chips — chips themselves will be pasted later)
filters_band_top = 360
filters_band_bottom = 520
draw.rectangle([(0, filters_band_top), (1440, filters_band_bottom)], fill=(250, 252, 255))  # very subtle bluish band
# faint top and bottom separators for the filters band
draw.line([(24, filters_band_top), (1416, filters_band_top)], fill=(240, 241, 243), width=1)
draw.line([(24, filters_band_bottom), (1416, filters_band_bottom)], fill=(240, 241, 243), width=1)

# Section separator under chips
sep_y = 520
draw.line([(24, sep_y), (1416, sep_y)], fill=(230, 231, 234), width=1)

# Card 1 background (rounded with subtle shadow)
card1_x0, card1_y0 = 40, 660
card1_x1, card1_y1 = 1400, 1760  # covers image + title region (image and text will be pasted on top)
shadow_offset = 6
# shadow
draw.rounded_rectangle(
    [(card1_x0 + shadow_offset, card1_y0 + shadow_offset), (card1_x1 + shadow_offset, card1_y1 + shadow_offset)],
    radius=28, fill=(220, 222, 226)
)
# card
draw.rounded_rectangle([(card1_x0, card1_y0), (card1_x1, card1_y1)], radius=24, fill=(255, 255, 255))

# Separator under first card (subtle)
draw.line([(24, card1_y1 + 16), (1416, card1_y1 + 16)], fill=(235, 236, 239), width=1)

# Card 2 background (rounded with subtle shadow)
# Based on detected second content block starting around y ~1815
card2_x0, card2_y0 = 40, 1790
card2_x1, card2_y1 = 1400, 2820
# shadow
draw.rounded_rectangle(
    [(card2_x0 + shadow_offset, card2_y0 + shadow_offset), (card2_x1 + shadow_offset, card2_y1 + shadow_offset)],
    radius=28, fill=(220, 222, 226)
)
# card
draw.rounded_rectangle([(card2_x0, card2_y0), (card2_x1, card2_y1)], radius=24, fill=(255, 255, 255))

# Subtle horizontal separators between list items (below cards)
draw.line([(24, card2_y1 + 12), (1416, card2_y1 + 12)], fill=(240, 241, 244), width=1)

# Bottom navigation bar area
nav_top = 2750
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill=(255, 255, 255))  # white nav background
draw.line([(0, nav_top), (1440, nav_top)], fill=(220, 221, 224), width=1)  # top border of nav bar

# Small safe-area gap at very bottom
draw.rectangle([(0, nav_bottom - 16), (1440, nav_bottom)], fill=(248, 249, 251))

# Additional subtle left/right margins guides (non-intrusive, very faint)
margin_color = (245, 246, 248)
draw.line([(24, status_h), (24, nav_top)], fill=margin_color, width=1)
draw.line([(1416, status_h), (1416, nav_top)], fill=margin_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/04_icon_21_June_2024.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["21_June_2024"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/05_icon_Foo.png
try:
    _c5 = get_crop(5, 150, 110)
    canvas.paste(_c5, (1282, 406), _c5)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/06_icon_21_June_2024.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["21_June_2024"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/07_icon_ISth.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1092, 2331), _c7)
except Exception:
    pass
layout["ISth,"] = [1092, 2331, 1236, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/08_icon_ISth.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2331), _c8)
except Exception:
    pass
layout["ISth,"] = [1236, 2331, 1380, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/09_icon_Close_current_screen.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/10_icon_Language_Learning.png
try:
    _c10 = get_crop(10, 1344, 191)
    canvas.paste(_c10, (48, 72), _c10)
except Exception:
    pass
layout["Language_Learning"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/11_icon_5.18.png
try:
    _c11 = get_crop(11, 126, 116)
    canvas.paste(_c11, (55, 113), _c11)
except Exception:
    pass
layout["5.18"] = [55, 113, 181, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 104, 61)
    canvas.paste(_c12, (1207, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1207, 0, 1311, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 66, 62)
    canvas.paste(_c13, (308, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [308, 1, 374, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/14_icon_5.18.png
try:
    _c14 = get_crop(14, 60, 63)
    canvas.paste(_c14, (181, 0), _c14)
except Exception:
    pass
layout["5.18"] = [181, 0, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/15_icon_Arenuc_Chlc_Z0_Illinois.png
try:
    _c15 = get_crop(15, 1344, 1001)
    canvas.paste(_c15, (48, 1815), _c15)
except Exception:
    pass
layout["Arenuc_Chlc'Z0,_Illinois"] = [48, 1815, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 51, 61)
    canvas.paste(_c16, (249, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [249, 1, 300, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/17_icon_5.18.png
try:
    _c17 = get_crop(17, 62, 65)
    canvas.paste(_c17, (114, 0), _c17)
except Exception:
    pass
layout["5.18"] = [114, 0, 176, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/18_icon_Chicago.png
try:
    _c18 = get_crop(18, 417, 144)
    canvas.paste(_c18, (0, 259), _c18)
except Exception:
    pass
layout["Chicago"] = [0, 259, 417, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 60, 61)
    canvas.paste(_c19, (1318, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1318, 0, 1378, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 50, 60)
    canvas.paste(_c20, (384, 3), _c20)
except Exception:
    pass
layout["icon_20"] = [384, 3, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/21_icon_15_._5.00_PM_CDT.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["15_._5.00_PM_CDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/22_icon_Iimei.png
try:
    _c22 = get_crop(22, 1344, 1091)
    canvas.paste(_c22, (48, 676), _c22)
except Exception:
    pass
layout["Iimei"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/23_icon_Wine_and_Unwind_DIY_Body_Oil_and_Candle.png
try:
    _c23 = get_crop(23, 1344, 1091)
    canvas.paste(_c23, (48, 676), _c23)
except Exception:
    pass
layout["Wine_and_Unwind:_DIY_Body"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/24_icon_Promoted.png
try:
    _c24 = get_crop(24, 42, 60)
    canvas.paste(_c24, (285, 1662), _c24)
except Exception:
    pass
layout["Promoted"] = [285, 1662, 327, 1722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/25_icon_15_._5.00_PM_CDT.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["15_._5.00_PM_CDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/26_icon_Wed.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Wed,"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/27_icon_May.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (864, 2804), _c27)
except Exception:
    pass
layout["May"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/28_text_5.18.png
try:
    _c28 = get_crop(28, 89, 43)
    canvas.paste(_c28, (22, 17), _c28)
except Exception:
    pass
layout["5.18"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/29_text_6_167_events.png
try:
    _c29 = get_crop(29, 359, 103)
    canvas.paste(_c29, (54, 410), _c29)
except Exception:
    pass
layout["6,167_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/30_text_Fri.png
try:
    _c30 = get_crop(30, 76, 51)
    canvas.paste(_c30, (90, 1536), _c30)
except Exception:
    pass
layout["Fri,"] = [90, 1536, 166, 1587]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/31_text_17_._6_00_PM_CDT.png
try:
    _c31 = get_crop(31, 337, 45)
    canvas.paste(_c31, (253, 1537), _c31)
except Exception:
    pass
layout["17_._6:00_PM_CDT"] = [253, 1537, 590, 1582]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/32_text_CLEAR.png
try:
    _c32 = get_crop(32, 110, 36)
    canvas.paste(_c32, (1181, 1951), _c32)
except Exception:
    pass
layout["CLEAR"] = [1181, 1951, 1291, 1987]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/33_text_Wed.png
try:
    _c33 = get_crop(33, 105, 50)
    canvas.paste(_c33, (93, 2759), _c33)
except Exception:
    pass
layout["Wed,"] = [93, 2759, 198, 2809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/34_text_15_._5.00_PM_CDT.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (288, 2804), _c34)
except Exception:
    pass
layout["15_._5.00_PM_CDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_04_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-6/35_clickable_More.png
try:
    _c35 = get_crop(35, 288, 156)
    canvas.paste(_c35, (1152, 2804), _c35)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
