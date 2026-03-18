# page_id: page_eventbrite_5059e61823f845cf95cbd1d8d54254c6_16
# screenshot: 2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18.png
# step_index: 16/19
# task: Open Eventbrite. Look for 'Education' in Los Angeles happening on May 4. Filter to show only free events. How many events are posted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level UI structural drawing for Event list page (PIL ImageDraw)
# Assumes variables provided in environment:
# - canvas: PIL Image (1440x2960 RGB)
# - draw: PIL ImageDraw.Draw(canvas)
# - font_sm, font_md, font_lg, font_xl

# Background fill (soft off-white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#F6F7F9")

# Status bar (top ~72px) - dark muted gray like Android status bar
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#9E9E9E")

# Header / Toolbar area (white) under status bar
header_y1 = status_h
header_y2 = 152
draw.rectangle([(0, header_y1), (1440, header_y2)], fill="#FFFFFF")

# Left accent behind title (purely decorative, not text)
draw.rounded_rectangle([(28, header_y1 + 18), (260, header_y1 + 86)], radius=18, fill="#2F6BE6")

# Bottom divider under header
draw.line([(28, header_y2), (1412, header_y2)], fill="#E6E7EA", width=2)

# Filter / pill area background band (subtle white band with extra spacing)
filter_band_y1 = 220
filter_band_y2 = 480
draw.rectangle([(0, filter_band_y1), (1440, filter_band_y2)], fill="#FFFFFF")
# subtle top/bottom hairlines for the band
draw.line([(24, filter_band_y1), (1416, filter_band_y1)], fill="#F0F1F4", width=1)
draw.line([(24, filter_band_y2), (1416, filter_band_y2)], fill="#F0F1F4", width=1)

# Primary event list card (rounded white card behind the first event block)
# Slightly larger than the image area so pasted image/icons sit on top
card1 = (24, 628, 1416, 1768)
draw.rounded_rectangle(card1, radius=28, fill="#FFFFFF", outline="#E9EAEE", width=1)
# subtle drop shadow line below card1
draw.line([(card1[0]+8, card1[3]+2), (card1[2]-8, card1[3]+2)], fill="#ECEFF3", width=3)

# Secondary event list card (rounded white card behind the second event block)
card2 = (24, 1688, 1416, 2460)
draw.rounded_rectangle(card2, radius=28, fill="#FFFFFF", outline="#E9EAEE", width=1)
# subtle drop shadow line below card2
draw.line([(card2[0]+8, card2[3]+2), (card2[2]-8, card2[3]+2)], fill="#ECEFF3", width=3)

# Thin separators between list items / sections
sep_x1 = 40
sep_x2 = 1400
separators_y = [460, 620, 1700, 2468]
for y in separators_y:
    draw.line([(sep_x1, y), (sep_x2, y)], fill="#F0F1F4", width=1)

# Large content area background behind images (dark banner strip) - decorative,
# but placed inside cards so actual images/icons will be pasted on top.
# These are intentionally subtle and act as background bands rather than duplicating detected images.
# First image band (keeps margins for pasted image)
img_band1 = (48, 676, 48 + 1344, 676 + 360)  # top portion band behind image area
draw.rectangle(img_band1, fill="#0B0B0B")

# Second image band (top portion)
img_band2 = (48, 1708, 48 + 1344, 1708 + 360)
draw.rectangle(img_band2, fill="#0B0B0B")

# Bottom navigation bar background (fixed)
nav_h = 120
draw.rectangle([(0, 2960 - nav_h), (1440, 2960)], fill="#FFFFFF")
# top border of nav
draw.line([(24, 2960 - nav_h), (1416, 2960 - nav_h)], fill="#E6E7EA", width=2)

# Subtle left/right padding guideline bars (visual alignment helpers)
draw.line([(40, 160), (40, 2800)], fill="#F3F4F6", width=1)
draw.line([(1400, 160), (1400, 2800)], fill="#F3F4F6", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (951, 410), _c0)
except Exception:
    pass
layout["Music"] = [951, 410, 1138, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/01_icon_May_04_2024.png
try:
    _c1 = get_crop(1, 501, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["May_04,_2024"] = [438, 410, 939, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1150, 410), _c2)
except Exception:
    pass
layout["Business"] = [1150, 410, 1391, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/04_icon_REAL_ESTATE_your.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2269), _c4)
except Exception:
    pass
layout["REAL_ESTATE_your"] = [1092, 2269, 1236, 2413]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/05_icon_WIN_by_making.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 2269), _c5)
except Exception:
    pass
layout["WIN_by_making"] = [1236, 2269, 1380, 2413]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/06_icon_CEC.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1192), _c6)
except Exception:
    pass
layout["CEC"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 49, 66)
    canvas.paste(_c7, (1153, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1153, 0, 1202, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/08_icon_Close_current_screen.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1248, 96), _c8)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/09_icon_Ist_Nature-Based_Education_Summit.png
try:
    _c9 = get_crop(9, 1344, 1029)
    canvas.paste(_c9, (48, 676), _c9)
except Exception:
    pass
layout["Ist_Nature-Based_Educatio"] = [48, 676, 1392, 1705]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/10_icon_Education.png
try:
    _c10 = get_crop(10, 64, 61)
    canvas.paste(_c10, (309, 1), _c10)
except Exception:
    pass
layout["Education"] = [309, 1, 373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/11_icon_UCLA_Lab_School.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1236, 1192), _c11)
except Exception:
    pass
layout["UCLA_Lab_School"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/12_icon_7.36.png
try:
    _c12 = get_crop(12, 58, 63)
    canvas.paste(_c12, (181, 1), _c12)
except Exception:
    pass
layout["7.36"] = [181, 1, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/13_icon_7.36.png
try:
    _c13 = get_crop(13, 57, 64)
    canvas.paste(_c13, (116, 0), _c13)
except Exception:
    pass
layout["7.36"] = [116, 0, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/14_icon_7.36.png
try:
    _c14 = get_crop(14, 113, 112)
    canvas.paste(_c14, (60, 115), _c14)
except Exception:
    pass
layout["7.36"] = [60, 115, 173, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 48, 61)
    canvas.paste(_c15, (251, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [251, 1, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 83, 65)
    canvas.paste(_c16, (1213, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1213, 0, 1296, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 51, 63)
    canvas.paste(_c17, (1320, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1320, 0, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/18_icon_Education.png
try:
    _c18 = get_crop(18, 48, 61)
    canvas.paste(_c18, (384, 2), _c18)
except Exception:
    pass
layout["Education"] = [384, 2, 432, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/19_icon_Education.png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["Education"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/20_icon_Sat_May_4_._7_00_AM_PDT.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (288, 2804), _c20)
except Exception:
    pass
layout["Sat,_May_4_._7:00_AM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/21_icon_Los_Angeles.png
try:
    _c21 = get_crop(21, 492, 144)
    canvas.paste(_c21, (0, 259), _c21)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/22_icon_Free.png
try:
    _c22 = get_crop(22, 124, 77)
    canvas.paste(_c22, (91, 2446), _c22)
except Exception:
    pass
layout["Free"] = [91, 2446, 215, 2523]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/23_icon_Introduction_To_Our_Nationwide_Communitv.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["Introduction_To_Our_Natio"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/24_icon_Introduction_To_Our_Nationwide_Communitv.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["Introduction_To_Our_Natio"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/25_icon_Introduction_To_Our_Nationwide_Communitv.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["Introduction_To_Our_Natio"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/26_icon_Ist_Nature-Based_Education_Summit.png
try:
    _c26 = get_crop(26, 1344, 1029)
    canvas.paste(_c26, (48, 676), _c26)
except Exception:
    pass
layout["Ist_Nature-Based_Educatio"] = [48, 676, 1392, 1705]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/27_icon_7.36.png
try:
    _c27 = get_crop(27, 94, 63)
    canvas.paste(_c27, (13, 1), _c27)
except Exception:
    pass
layout["7.36"] = [13, 1, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/28_icon_ONLINE_Event.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["ONLINE_Event"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 42, 63)
    canvas.paste(_c29, (1273, 0), _c29)
except Exception:
    pass
layout["icon_29"] = [1273, 0, 1315, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/30_text_242_events.png
try:
    _c30 = get_crop(30, 372, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["242_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/31_text_NCOME.png
try:
    _c31 = get_crop(31, 158, 43)
    canvas.paste(_c31, (168, 1867), _c31)
except Exception:
    pass
layout["NCOME"] = [168, 1867, 326, 1910]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/32_text_REAL_ESTATE.png
try:
    _c32 = get_crop(32, 504, 68)
    canvas.paste(_c32, (469, 1840), _c32)
except Exception:
    pass
layout["REAL_ESTATE"] = [469, 1840, 973, 1908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/33_text_Discover.png
try:
    _c33 = get_crop(33, 268, 66)
    canvas.paste(_c33, (1048, 1857), _c33)
except Exception:
    pass
layout["Discover"] = [1048, 1857, 1316, 1923]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/34_text_DEDUCTIONS.png
try:
    _c34 = get_crop(34, 270, 43)
    canvas.paste(_c34, (156, 1962), _c34)
except Exception:
    pass
layout["DEDUCTIONS"] = [156, 1962, 426, 2005]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/35_text_IS.png
try:
    _c35 = get_crop(35, 84, 64)
    canvas.paste(_c35, (857, 1932), _c35)
except Exception:
    pass
layout["IS"] = [857, 1932, 941, 1996]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/36_text_The_SYSTEM.png
try:
    _c36 = get_crop(36, 366, 64)
    canvas.paste(_c36, (998, 1939), _c36)
except Exception:
    pass
layout["The_SYSTEM"] = [998, 1939, 1364, 2003]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/37_text_EQUITY.png
try:
    _c37 = get_crop(37, 161, 52)
    canvas.paste(_c37, (150, 2056), _c37)
except Exception:
    pass
layout["EQUITY"] = [150, 2056, 311, 2108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/38_text_The_Blueprint_to.png
try:
    _c38 = get_crop(38, 297, 49)
    canvas.paste(_c38, (995, 2068), _c38)
except Exception:
    pass
layout["The_Blueprint_to"] = [995, 2068, 1292, 2117]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/39_text_APPRECIATION.png
try:
    _c39 = get_crop(39, 305, 50)
    canvas.paste(_c39, (152, 2152), _c39)
except Exception:
    pass
layout["APPRECIATION"] = [152, 2152, 457, 2202]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/40_text_you.png
try:
    _c40 = get_crop(40, 80, 41)
    canvas.paste(_c40, (993, 2175), _c40)
except Exception:
    pass
layout["you"] = [993, 2175, 1073, 2216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/41_text_WIN_by_making.png
try:
    _c41 = get_crop(41, 144, 144)
    canvas.paste(_c41, (1092, 2269), _c41)
except Exception:
    pass
layout["WIN_by_making"] = [1092, 2269, 1236, 2413]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/42_text_LEVERAGE.png
try:
    _c42 = get_crop(42, 214, 43)
    canvas.paste(_c42, (154, 2251), _c42)
except Exception:
    pass
layout["LEVERAGE"] = [154, 2251, 368, 2294]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/43_text_CD_EAL.png
try:
    _c43 = get_crop(43, 1344, 1063)
    canvas.paste(_c43, (48, 1753), _c43)
except Exception:
    pass
layout["CD_EAL"] = [48, 1753, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/44_text_The_Path_to_Wealth_Through_Education.png
try:
    _c44 = get_crop(44, 1344, 1063)
    canvas.paste(_c44, (48, 1753), _c44)
except Exception:
    pass
layout["The_Path_to_Wealth_Throug"] = [48, 1753, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/45_text_Pacoima.png
try:
    _c45 = get_crop(45, 246, 63)
    canvas.paste(_c45, (92, 2612), _c45)
except Exception:
    pass
layout["Pacoima"] = [92, 2612, 338, 2675]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/46_text_Sat_May_4_._7_00_AM_PDT.png
try:
    _c46 = get_crop(46, 288, 156)
    canvas.paste(_c46, (288, 2804), _c46)
except Exception:
    pass
layout["Sat,_May_4_._7:00_AM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_16_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-18/47_text_ONLINE_Event.png
try:
    _c47 = get_crop(47, 288, 156)
    canvas.paste(_c47, (0, 2804), _c47)
except Exception:
    pass
layout["ONLINE_Event"] = [0, 2804, 288, 2960]
