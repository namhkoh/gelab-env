# page_id: page_eventbrite_5059e61823f845cf95cbd1d8d54254c6_19
# screenshot: 2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21.png
# step_index: 19/19
# task: Open Eventbrite. Look for 'Education' in Los Angeles happening on May 4. Filter to show only free events. How many events are posted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background
bg_color = (247, 249, 251)  # very light gray page background
draw.rectangle([0, 0, 1440, 2960], fill=bg_color)

# Status bar (top ~50px area, darker gray)
status_h = 50
status_color = (150, 150, 150)
draw.rectangle([0, 0, 1440, status_h], fill=status_color)

# Header area (large white header under status bar)
header_top = status_h
header_bottom = 260
header_bg = (255, 255, 255)
draw.rectangle([0, header_top, 1440, header_bottom], fill=header_bg)

# Thin divider under header
divider_color = (220, 224, 229)
draw.line([(48, header_bottom), (1392, header_bottom)], fill=divider_color, width=2)

# Subtle search/filters background row (behind the filter pills but not drawing pills)
filters_row_top = header_bottom + 8
filters_row_bottom = 420
filters_bg = (255, 255, 255)
draw.rectangle([0, filters_row_top, 1440, filters_row_bottom], fill=filters_bg)
# slight shadow line below filters row
draw.line([(48, filters_row_bottom), (1392, filters_row_bottom)], fill=(235,238,242), width=1)

# Draw large rounded image/card background for first event (matches screenshot image area)
# Detection: pos=(48,676) size=1344x1029
first_img_x, first_img_y = 48, 676
first_img_w, first_img_h = 1344, 1029
first_img_rect = [first_img_x, first_img_y, first_img_x + first_img_w, first_img_y + first_img_h]
image_bg = (26, 30, 22)  # dark image placeholder background (deep greenish/black)
draw.rounded_rectangle(first_img_rect, radius=20, fill=image_bg)

# Subtle shadow under the first image (a muted gray rectangle offset)
shadow_color = (230, 233, 236)
shadow_offset = 8
draw.rounded_rectangle(
    [first_img_x + shadow_offset, first_img_y + first_img_h + 8, first_img_x + first_img_w + shadow_offset, first_img_y + first_img_h + 18],
    radius=6, fill=shadow_color
)

# White card area under first image for title/meta background (do not draw text)
first_card_meta_top = first_img_y + first_img_h + 28
first_card_meta_bottom = first_card_meta_top + 220
draw.rectangle([48, first_card_meta_top, 1392, first_card_meta_bottom], fill=(255,255,255))
# subtle top border for this card area
draw.line([(48, first_card_meta_top), (1392, first_card_meta_top)], fill=(236,239,243), width=1)

# Separator between first and second event
sep_y = first_card_meta_bottom + 18
draw.line([(48, sep_y), (1392, sep_y)], fill=(235,238,242), width=1)

# Draw large rounded image/card background for second event
# Detection: pos=(48,1753) size=1344x1063
second_img_x, second_img_y = 48, 1753
second_img_w, second_img_h = 1344, 1063
second_img_rect = [second_img_x, second_img_y, second_img_x + second_img_w, second_img_y + second_img_h]
# second image has an orange/black graphic in screenshot; use dark neutral background so pasted image will show
second_image_bg = (12, 12, 12)
draw.rounded_rectangle(second_img_rect, radius=20, fill=second_image_bg)

# Shadow under the second image
draw.rounded_rectangle(
    [second_img_x + shadow_offset, second_img_y + second_img_h + 8, second_img_x + second_img_w + shadow_offset, second_img_y + second_img_h + 18],
    radius=6, fill=shadow_color
)

# White card area under second image for title/meta background (clipped near bottom)
second_card_meta_top = second_img_y + second_img_h + 28
second_card_meta_bottom = min(second_card_meta_top + 220, 2960 - 110)  # leave space for bottom nav
draw.rectangle([48, second_card_meta_top, 1392, second_card_meta_bottom], fill=(255,255,255))
draw.line([(48, second_card_meta_top), (1392, second_card_meta_top)], fill=(236,239,243), width=1)

# Bottom navigation bar background
nav_h = 96
nav_top = 2960 - nav_h
nav_bg = (255, 255, 255)
draw.rectangle([0, nav_top, 1440, 2960], fill=nav_bg)
# top divider for nav
draw.line([(0, nav_top), (1440, nav_top)], fill=(225, 228, 232), width=1)

# Small rounded card backgrounds for potential "Free" badges area (only background shapes, not badges themselves)
# We'll draw faint, very pale rounded rectangles where badges might sit, but keep them neutral so icons/text will replace them.
# Note: Avoid drawing exact badge artwork; just a faint soft background suggestion removed from exact badge positions.
badge_hint_color = (245, 247, 248)
draw.rounded_rectangle([60, first_img_y + 12, 160, first_img_y + 60], radius=10, fill=badge_hint_color)
draw.rounded_rectangle([60, second_img_y + 12, 188, second_img_y + 60], radius=10, fill=badge_hint_color)

# Subtle vertical margins and page edges (rounded corners clipped visually)
edge_panel_color = (250, 251, 252)
# left and right gutters to simulate padding (very subtle)
draw.rectangle([0, 0, 48, 2960], fill=edge_panel_color)
draw.rectangle([1392, 0, 1440, 2960], fill=edge_panel_color)

# Final thin horizontal separators at a few logical spots (non-intrusive)
for y in (header_bottom + 2, filters_row_bottom + 2, sep_y + 2, second_card_meta_bottom + 2):
    draw.line([(48, y), (1392, y)], fill=(245,247,249), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/00_icon_May_04_2024.png
try:
    _c0 = get_crop(0, 501, 103)
    canvas.paste(_c0, (458, 410), _c0)
except Exception:
    pass
layout["May_04,_2024"] = [458, 410, 959, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/01_icon_Music.png
try:
    _c1 = get_crop(1, 187, 103)
    canvas.paste(_c1, (971, 410), _c1)
except Exception:
    pass
layout["Music"] = [971, 410, 1158, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/02_icon_2_Filters.png
try:
    _c2 = get_crop(2, 392, 103)
    canvas.paste(_c2, (54, 410), _c2)
except Exception:
    pass
layout["2_Filters"] = [54, 410, 446, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/03_icon_Business.png
try:
    _c3 = get_crop(3, 222, 103)
    canvas.paste(_c3, (1170, 410), _c3)
except Exception:
    pass
layout["Business"] = [1170, 410, 1392, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/04_icon_REAL_ESTATEyour.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2269), _c4)
except Exception:
    pass
layout["REAL_ESTATEyour"] = [1092, 2269, 1236, 2413]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/05_icon_REAL_ESTATEyour.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 2269), _c5)
except Exception:
    pass
layout["REAL_ESTATEyour"] = [1236, 2269, 1380, 2413]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/06_icon_CEC.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1192), _c6)
except Exception:
    pass
layout["CEC"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 48, 67)
    canvas.paste(_c7, (1154, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1154, 0, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/08_icon_Close_current_screen.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1248, 96), _c8)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/09_icon_UCLA_Lab_School.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1236, 1192), _c9)
except Exception:
    pass
layout["UCLA_Lab_School"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/10_icon_Education.png
try:
    _c10 = get_crop(10, 64, 62)
    canvas.paste(_c10, (309, 1), _c10)
except Exception:
    pass
layout["Education"] = [309, 1, 373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/11_icon_7.36.png
try:
    _c11 = get_crop(11, 58, 63)
    canvas.paste(_c11, (181, 1), _c11)
except Exception:
    pass
layout["7.36"] = [181, 1, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/12_icon_7.36.png
try:
    _c12 = get_crop(12, 57, 65)
    canvas.paste(_c12, (116, 0), _c12)
except Exception:
    pass
layout["7.36"] = [116, 0, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/13_icon_Ist_Nature-Based_Education_Summit.png
try:
    _c13 = get_crop(13, 1344, 1029)
    canvas.paste(_c13, (48, 676), _c13)
except Exception:
    pass
layout["Ist_Nature-Based_Educatio"] = [48, 676, 1392, 1705]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/14_icon_7.36.png
try:
    _c14 = get_crop(14, 113, 112)
    canvas.paste(_c14, (60, 115), _c14)
except Exception:
    pass
layout["7.36"] = [60, 115, 173, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 52, 62)
    canvas.paste(_c15, (248, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [248, 1, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 66, 66)
    canvas.paste(_c16, (1213, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1213, 0, 1279, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 51, 64)
    canvas.paste(_c17, (1319, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1319, 0, 1370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/18_icon_Education.png
try:
    _c18 = get_crop(18, 48, 61)
    canvas.paste(_c18, (384, 2), _c18)
except Exception:
    pass
layout["Education"] = [384, 2, 432, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/19_icon_Sat_May_4_._7_00_AM_PDT.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["Sat,_May_4_._7:00_AM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/20_icon_Free.png
try:
    _c20 = get_crop(20, 124, 77)
    canvas.paste(_c20, (91, 2446), _c20)
except Exception:
    pass
layout["Free"] = [91, 2446, 215, 2523]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/21_icon_Education.png
try:
    _c21 = get_crop(21, 1344, 191)
    canvas.paste(_c21, (48, 72), _c21)
except Exception:
    pass
layout["Education"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/22_icon_Los_Angeles.png
try:
    _c22 = get_crop(22, 492, 144)
    canvas.paste(_c22, (0, 259), _c22)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 43, 63)
    canvas.paste(_c23, (1272, 1), _c23)
except Exception:
    pass
layout["icon_23"] = [1272, 1, 1315, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/24_icon_Introduction_To_Our_Nationwide_Communitv.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["Introduction_To_Our_Natio"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/25_icon_Ist_Nature-Based_Education_Summit.png
try:
    _c25 = get_crop(25, 1344, 1029)
    canvas.paste(_c25, (48, 676), _c25)
except Exception:
    pass
layout["Ist_Nature-Based_Educatio"] = [48, 676, 1392, 1705]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/26_icon_Introduction_To_Our_Nationwide_Communitv.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (576, 2804), _c26)
except Exception:
    pass
layout["Introduction_To_Our_Natio"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/27_icon_Introduction_To_Our_Nationwide_Communitv.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (864, 2804), _c27)
except Exception:
    pass
layout["Introduction_To_Our_Natio"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/28_icon_Free.png
try:
    _c28 = get_crop(28, 127, 78)
    canvas.paste(_c28, (89, 1369), _c28)
except Exception:
    pass
layout["Free"] = [89, 1369, 216, 1447]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/29_icon_7.36.png
try:
    _c29 = get_crop(29, 94, 65)
    canvas.paste(_c29, (13, 0), _c29)
except Exception:
    pass
layout["7.36"] = [13, 0, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/30_icon_ONLINE_Event.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (0, 2804), _c30)
except Exception:
    pass
layout["ONLINE_Event"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/31_text_153_events.png
try:
    _c31 = get_crop(31, 392, 103)
    canvas.paste(_c31, (54, 410), _c31)
except Exception:
    pass
layout["153_events"] = [54, 410, 446, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/32_text_NCOME.png
try:
    _c32 = get_crop(32, 161, 45)
    canvas.paste(_c32, (165, 1865), _c32)
except Exception:
    pass
layout["NCOME"] = [165, 1865, 326, 1910]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/33_text_REAL_ESTATE.png
try:
    _c33 = get_crop(33, 504, 68)
    canvas.paste(_c33, (469, 1840), _c33)
except Exception:
    pass
layout["REAL_ESTATE"] = [469, 1840, 973, 1908]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/34_text_Discover.png
try:
    _c34 = get_crop(34, 267, 61)
    canvas.paste(_c34, (1049, 1858), _c34)
except Exception:
    pass
layout["Discover"] = [1049, 1858, 1316, 1919]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/35_text_DEDUCTIONS.png
try:
    _c35 = get_crop(35, 272, 45)
    canvas.paste(_c35, (154, 1960), _c35)
except Exception:
    pass
layout["DEDUCTIONS"] = [154, 1960, 426, 2005]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/36_text_IS.png
try:
    _c36 = get_crop(36, 84, 64)
    canvas.paste(_c36, (857, 1932), _c36)
except Exception:
    pass
layout["IS"] = [857, 1932, 941, 1996]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/37_text_The_SYSTEM.png
try:
    _c37 = get_crop(37, 368, 64)
    canvas.paste(_c37, (996, 1939), _c37)
except Exception:
    pass
layout["The_SYSTEM"] = [996, 1939, 1364, 2003]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/38_text_EQUITY.png
try:
    _c38 = get_crop(38, 158, 52)
    canvas.paste(_c38, (153, 2056), _c38)
except Exception:
    pass
layout["EQUITY"] = [153, 2056, 311, 2108]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/39_text_The_Blueprint_to.png
try:
    _c39 = get_crop(39, 297, 49)
    canvas.paste(_c39, (995, 2068), _c39)
except Exception:
    pass
layout["The_Blueprint_to"] = [995, 2068, 1292, 2117]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/40_text_APPRECIATION.png
try:
    _c40 = get_crop(40, 307, 49)
    canvas.paste(_c40, (150, 2151), _c40)
except Exception:
    pass
layout["APPRECIATION"] = [150, 2151, 457, 2200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/41_text_you-_WIN_by_making.png
try:
    _c41 = get_crop(41, 144, 144)
    canvas.paste(_c41, (1092, 2269), _c41)
except Exception:
    pass
layout["you-_WIN_by_making"] = [1092, 2269, 1236, 2413]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/42_text_LEVERAGE.png
try:
    _c42 = get_crop(42, 217, 45)
    canvas.paste(_c42, (151, 2249), _c42)
except Exception:
    pass
layout["LEVERAGE"] = [151, 2249, 368, 2294]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/43_text_ED_E4L.png
try:
    _c43 = get_crop(43, 1344, 1063)
    canvas.paste(_c43, (48, 1753), _c43)
except Exception:
    pass
layout["ED_E4L"] = [48, 1753, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/44_text_The_Path_to_Wealth_Through_Education.png
try:
    _c44 = get_crop(44, 1344, 1063)
    canvas.paste(_c44, (48, 1753), _c44)
except Exception:
    pass
layout["The_Path_to_Wealth_Throug"] = [48, 1753, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/45_text_Pacoima.png
try:
    _c45 = get_crop(45, 246, 63)
    canvas.paste(_c45, (92, 2612), _c45)
except Exception:
    pass
layout["Pacoima"] = [92, 2612, 338, 2675]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/46_text_Sat_May_4_._7_00_AM_PDT.png
try:
    _c46 = get_crop(46, 288, 156)
    canvas.paste(_c46, (288, 2804), _c46)
except Exception:
    pass
layout["Sat,_May_4_._7:00_AM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_19_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-21/47_text_ONLINE_Event.png
try:
    _c47 = get_crop(47, 288, 156)
    canvas.paste(_c47, (0, 2804), _c47)
except Exception:
    pass
layout["ONLINE_Event"] = [0, 2804, 288, 2960]
