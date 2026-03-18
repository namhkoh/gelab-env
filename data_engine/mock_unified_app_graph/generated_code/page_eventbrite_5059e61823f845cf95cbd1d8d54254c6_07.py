# page_id: page_eventbrite_5059e61823f845cf95cbd1d8d54254c6_07
# screenshot: 2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9.png
# step_index: 7/19
# task: Open Eventbrite. Look for 'Education' in Los Angeles happening on May 4. Filter to show only free events. How many events are posted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (dominant pale off-white)
draw.rectangle((0, 0, 1440, 2960), fill=(249, 250, 252))

# Status bar (top ~72px) - darker bar to mimic device status area
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(120, 120, 120))

# Subtle overlay line under status bar
draw.line((0, status_h, 1440, status_h), fill=(200, 200, 200), width=1)

# Header / search area background (white) beneath status bar
header_top = status_h
header_bottom = 264
draw.rectangle((0, header_top, 1440, header_bottom), fill=(255, 255, 255))

# Thin divider under header
draw.line((48, header_bottom, 1392, header_bottom), fill=(230, 230, 235), width=2)

# Horizontal chips/filters row background (very subtle)
chips_top = 336
chips_bottom = 484
# keep it white but add faint band to separate area
draw.rectangle((0, chips_top, 1440, chips_bottom), fill=(250, 251, 253))
draw.line((48, chips_bottom, 1392, chips_bottom), fill=(235, 235, 240), width=1)

# Page content area (main background remains pale)
content_top = chips_bottom + 20

# Card shadows and card backgrounds for event items
card_margin_x = 36
card_width_right = 1404  # 36px margin both sides (1440-36)
card_radius = 24

# First event card shadow (slightly offset)
first_card_top = 636
first_card_bottom = 1800
shadow_offset = 8
shadow_clr = (225, 226, 230)
draw.rounded_rectangle(
    (card_margin_x + shadow_offset, first_card_top + shadow_offset,
     card_width_right + shadow_offset, first_card_bottom + shadow_offset),
    radius=card_radius, fill=shadow_clr)

# First event card (white)
draw.rounded_rectangle(
    (card_margin_x, first_card_top, card_width_right, first_card_bottom),
    radius=card_radius, fill=(255, 255, 255))

# Subtle inner divider in first card (to separate image area from metadata)
# Place a faint line where image area would end; we don't draw the image itself.
# Use the image top from detected elements: image at y=676 with height 1115 -> bottom 1791
image1_top = 676
image1_bottom = image1_top + 1115
# divider just below the image area within the card
div_y = image1_bottom + 18
draw.line((card_margin_x + 24, div_y, card_width_right - 24, div_y), fill=(240, 240, 245), width=1)

# Second event card shadow and background
second_card_top = 1800
second_card_bottom = 2920
draw.rounded_rectangle(
    (card_margin_x + shadow_offset, second_card_top + shadow_offset,
     card_width_right + shadow_offset, min(second_card_bottom + shadow_offset, 2960)),
    radius=card_radius, fill=shadow_clr)

draw.rounded_rectangle(
    (card_margin_x, second_card_top, card_width_right, min(second_card_bottom, 2960)),
    radius=card_radius, fill=(255, 255, 255))

# Divider below second card's image area (image at y=1839 height 977 -> bottom 2816)
image2_top = 1839
image2_bottom = image2_top + 977
div2_y = min(image2_bottom + 18, 2950)
draw.line((card_margin_x + 24, div2_y, card_width_right - 24, div2_y), fill=(240, 240, 245), width=1)

# Light gray band behind lists / headings (e.g., the "10,000 events" area)
events_heading_band_top = 504
events_heading_band_bottom = 600
draw.rectangle((0, events_heading_band_top, 1440, events_heading_band_bottom), fill=(249, 250, 252))

# Thin separator lines for content sections
for y in (events_heading_band_bottom, first_card_top - 8, second_card_top - 8):
    draw.line((36, y, 1404, y), fill=(235, 235, 240), width=1)

# Bottom navigation bar background with subtle top border
nav_h = 140
nav_top = 2960 - nav_h
draw.rectangle((0, nav_top, 1440, 2960), fill=(255, 255, 255))
draw.line((36, nav_top, 1404, nav_top), fill=(230, 230, 235), width=2)

# Small notch indicator area above nav (subtle)
notch_w = 120
notch_h = 6
notch_x = (1440 - notch_w) // 2
draw.rounded_rectangle((notch_x, nav_top - 18, notch_x + notch_w, nav_top - 18 + notch_h),
                       radius=3, fill=(245, 245, 247))

# Side margins vertical guides (very faint) to imply content insets
draw.line((36, 0, 36, 2960), fill=(250, 250, 252), width=1)
draw.line((1404, 0, 1404, 2960), fill=(250, 250, 252), width=1)

# Final very subtle vignette at card edges to reinforce depth
# (simple corner rectangles to suggest slightly darker corners)
corner_shade = (246, 247, 249)
draw.rectangle((card_margin_x - 4, first_card_top - 6, card_margin_x + 8, first_card_bottom + 6), fill=corner_shade)
draw.rectangle((card_width_right - 8, first_card_top - 6, card_width_right + 4, first_card_bottom + 6), fill=corner_shade)
draw.rectangle((card_margin_x - 4, second_card_top - 6, card_margin_x + 8, min(second_card_bottom + 6, 2960)), fill=corner_shade)
draw.rectangle((card_width_right - 8, second_card_top - 6, card_width_right + 4, min(second_card_bottom + 6, 2960)), fill=corner_shade)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/04_icon_Ghibli_Music_Night.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2355), _c4)
except Exception:
    pass
layout["Ghibli_Music_Night"] = [1092, 2355, 1236, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/06_icon_Foo.png
try:
    _c6 = get_crop(6, 151, 110)
    canvas.paste(_c6, (1282, 406), _c6)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1433, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2355), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2355, 1380, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/09_icon_Search_forae.png
try:
    _c9 = get_crop(9, 1344, 191)
    canvas.paste(_c9, (48, 72), _c9)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/10_icon_7.35.png
try:
    _c10 = get_crop(10, 116, 106)
    canvas.paste(_c10, (59, 118), _c10)
except Exception:
    pass
layout["7.35"] = [59, 118, 175, 224]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 53, 65)
    canvas.paste(_c11, (1152, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1152, 0, 1205, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/12_icon_Rooftop_Pool_Side_Pink_Full_Moon_Yoga_Cl.png
try:
    _c12 = get_crop(12, 1344, 1115)
    canvas.paste(_c12, (48, 676), _c12)
except Exception:
    pass
layout["Rooftop_Pool_Side_Pink_Fu"] = [48, 676, 1392, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 91, 61)
    canvas.paste(_c13, (1212, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1212, 0, 1303, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/14_icon_Search_forae.png
try:
    _c14 = get_crop(14, 69, 62)
    canvas.paste(_c14, (307, 1), _c14)
except Exception:
    pass
layout["Search_forae"] = [307, 1, 376, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/15_icon_7.35.png
try:
    _c15 = get_crop(15, 60, 63)
    canvas.paste(_c15, (181, 0), _c15)
except Exception:
    pass
layout["7.35"] = [181, 0, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/16_icon_Los_Angeles.png
try:
    _c16 = get_crop(16, 492, 144)
    canvas.paste(_c16, (0, 259), _c16)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 54, 63)
    canvas.paste(_c17, (246, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [246, 1, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/18_icon_7.35.png
try:
    _c18 = get_crop(18, 61, 65)
    canvas.paste(_c18, (114, 0), _c18)
except Exception:
    pass
layout["7.35"] = [114, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 58, 59)
    canvas.paste(_c19, (1317, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1317, 0, 1375, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/20_icon_Ghibli_Music_Night.png
try:
    _c20 = get_crop(20, 1344, 977)
    canvas.paste(_c20, (48, 1839), _c20)
except Exception:
    pass
layout["Ghibli_Music_Night"] = [48, 1839, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/21_icon_Search_forae.png
try:
    _c21 = get_crop(21, 52, 60)
    canvas.paste(_c21, (383, 3), _c21)
except Exception:
    pass
layout["Search_forae"] = [383, 3, 435, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/22_icon_lordan_Hiah_School.Atlantic_Avenue.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["lordan_Hiah_School.Atlant"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/23_icon_ona_Beach_CA.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["ona_Beach_CA"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/24_icon_USA.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["USA"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/25_icon_Promoted.png
try:
    _c25 = get_crop(25, 245, 63)
    canvas.paste(_c25, (83, 1685), _c25)
except Exception:
    pass
layout["Promoted"] = [83, 1685, 328, 1748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/26_icon_Ticket_sales_end_soon.png
try:
    _c26 = get_crop(26, 489, 86)
    canvas.paste(_c26, (89, 1369), _c26)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [89, 1369, 578, 1455]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/27_icon_ona_Beach_CA.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (576, 2804), _c27)
except Exception:
    pass
layout["ona_Beach_CA"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 40, 60)
    canvas.paste(_c28, (1274, 0), _c28)
except Exception:
    pass
layout["icon_28"] = [1274, 0, 1314, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/29_icon_7.35.png
try:
    _c29 = get_crop(29, 102, 63)
    canvas.paste(_c29, (10, 0), _c29)
except Exception:
    pass
layout["7.35"] = [10, 0, 112, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/30_icon_Sat_Apr_27.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (0, 2804), _c30)
except Exception:
    pass
layout["Sat,_Apr_27"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/31_text_10_000_events.png
try:
    _c31 = get_crop(31, 359, 103)
    canvas.paste(_c31, (54, 410), _c31)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/32_text_Fairmont_Century_Plaza.png
try:
    _c32 = get_crop(32, 438, 56)
    canvas.paste(_c32, (93, 1627), _c32)
except Exception:
    pass
layout["Fairmont_Century_Plaza"] = [93, 1627, 531, 1683]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_07_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-9/33_text_8898.png
try:
    _c33 = get_crop(33, 280, 102)
    canvas.paste(_c33, (1085, 1848), _c33)
except Exception:
    pass
layout["8898"] = [1085, 1848, 1365, 1950]
