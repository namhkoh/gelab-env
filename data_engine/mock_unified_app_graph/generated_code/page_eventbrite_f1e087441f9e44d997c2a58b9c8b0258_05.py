# page_id: page_eventbrite_f1e087441f9e44d997c2a58b9c8b0258_05
# screenshot: 2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7.png
# step_index: 5/10
# task: Open Eventbrite. Find the 'Arts' category. Select events that are available for this weekend. From the results, open the first item and add it to favorite. Follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall subtle off-white background
draw.rectangle([(0, 0), (1440, 2960)], fill="#f7f7f8")

# Status bar (top) - light gray strip
status_h = 80
draw.rectangle([(0, 0), (1440, status_h)], fill="#bfbfc0")

# Thin darker top edge for subtle depth
draw.line([(0, 0), (1440, 0)], fill="#a8a8a8", width=1)

# Search/header area separators
# Light underline under the search area
draw.line([(48, 264), (1392, 264)], fill="#e1e1e3", width=2)
# A secondary subtle divider a bit higher for visual separation
draw.line([(48, 160), (1392, 160)], fill="#efeff1", width=1)

# Large event card 1 with shadow (do not draw any icons/text inside)
card1_x0, card1_y0 = 36, 600
card1_x1, card1_y1 = 1404, 1888
# shadow
draw.rounded_rectangle([(card1_x0+0, card1_y0+10), (card1_x1+0, card1_y1+10)], radius=32, fill="#ececf1")
# white card body
draw.rounded_rectangle([(card1_x0, card1_y0), (card1_x1, card1_y1)], radius=32, fill="#ffffff", outline=None)

# Separator between cards (thin line)
draw.line([(48, 1896), (1392, 1896)], fill="#f0f0f2", width=1)

# Large event card 2 with shadow (next list item)
card2_x0, card2_y0 = 36, 1904
card2_x1, card2_y1 = 1404, 2624
draw.rounded_rectangle([(card2_x0+0, card2_y0+10), (card2_x1+0, card2_y1+10)], radius=28, fill="#ececf1")
draw.rounded_rectangle([(card2_x0, card2_y0), (card2_x1, card2_y1)], radius=28, fill="#ffffff", outline=None)

# Thin separators and subtle section dividers within content flow
div_y_positions = [520, 680, 980, 1360, 1700, 2300]
for y in div_y_positions:
    draw.line([(48, y), (1392, y)], fill="#f3f3f5", width=1)

# Bottom navigation bar area (background + top divider)
nav_top = 2760
draw.line([(0, nav_top), (1440, nav_top)], fill="#e6e6e8", width=2)
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#ffffff")

# Subtle inner highlight at top of nav to separate from content
draw.line([(24, nav_top+2), (1416, nav_top+2)], fill="#fbfbfb", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/00_icon_Anytime.png
try:
    _c0 = get_crop(0, 400, 135)
    canvas.paste(_c0, (438, 390), _c0)
except Exception:
    pass
layout["Anytime"] = [438, 390, 838, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/01_icon_1_Filter.png
try:
    _c1 = get_crop(1, 372, 135)
    canvas.paste(_c1, (54, 390), _c1)
except Exception:
    pass
layout["1_Filter"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/02_icon_Arts.png
try:
    _c2 = get_crop(2, 152, 135)
    canvas.paste(_c2, (850, 390), _c2)
except Exception:
    pass
layout["Arts"] = [850, 390, 1002, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/03_icon_Favorite_button.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 2434), _c3)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2434, 1236, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/05_icon_Overflow_menu_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 2434), _c5)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2434, 1380, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/07_icon_Spring_Dance_Workshop_Classes_2024_April.png
try:
    _c7 = get_crop(7, 1344, 1194)
    canvas.paste(_c7, (48, 676), _c7)
except Exception:
    pass
layout["Spring_Dance_Workshop_Cla"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/08_icon_4.32.png
try:
    _c8 = get_crop(8, 117, 108)
    canvas.paste(_c8, (59, 117), _c8)
except Exception:
    pass
layout["4.32"] = [59, 117, 176, 225]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/09_icon_Search_forae.png
try:
    _c9 = get_crop(9, 69, 63)
    canvas.paste(_c9, (307, 0), _c9)
except Exception:
    pass
layout["Search_forae"] = [307, 0, 376, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/10_icon_4.32.png
try:
    _c10 = get_crop(10, 61, 63)
    canvas.paste(_c10, (181, 0), _c10)
except Exception:
    pass
layout["4.32"] = [181, 0, 242, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 54, 64)
    canvas.paste(_c11, (246, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [246, 0, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/12_icon_4.32.png
try:
    _c12 = get_crop(12, 62, 65)
    canvas.paste(_c12, (114, 0), _c12)
except Exception:
    pass
layout["4.32"] = [114, 0, 176, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 63, 59)
    canvas.paste(_c13, (1317, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1317, 0, 1380, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 68, 60)
    canvas.paste(_c14, (1207, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1207, 0, 1275, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/15_icon_Search_forae.png
try:
    _c15 = get_crop(15, 1344, 191)
    canvas.paste(_c15, (48, 72), _c15)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/16_icon_San_Francisco.png
try:
    _c16 = get_crop(16, 536, 144)
    canvas.paste(_c16, (0, 259), _c16)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/17_icon_Search_forae.png
try:
    _c17 = get_crop(17, 51, 61)
    canvas.paste(_c17, (383, 2), _c17)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 62, 60)
    canvas.paste(_c18, (1251, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1251, 0, 1313, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/19_icon_Spring_Dance_Workshop_Classes_2024_April.png
try:
    _c19 = get_crop(19, 1344, 1194)
    canvas.paste(_c19, (48, 676), _c19)
except Exception:
    pass
layout["Spring_Dance_Workshop_Cla"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/20_icon_LflL_M-A.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (288, 2804), _c20)
except Exception:
    pass
layout["LflL_M-A"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/21_icon_Tickets.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (864, 2804), _c21)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/22_icon_4.32.png
try:
    _c22 = get_crop(22, 108, 63)
    canvas.paste(_c22, (10, 0), _c22)
except Exception:
    pass
layout["4.32"] = [10, 0, 118, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/23_icon_LflL_M-A.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (576, 2804), _c23)
except Exception:
    pass
layout["LflL_M-A"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/24_icon_More.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/25_icon_CHANGE_YOUR_STORY.png
try:
    _c25 = get_crop(25, 1344, 898)
    canvas.paste(_c25, (48, 1918), _c25)
except Exception:
    pass
layout["CHANGE_YOUR_STORY"] = [48, 1918, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/26_icon_Joe_Goode_Performance_Group.png
try:
    _c26 = get_crop(26, 45, 58)
    canvas.paste(_c26, (283, 1766), _c26)
except Exception:
    pass
layout["Joe_Goode_Performance_Gro"] = [283, 1766, 328, 1824]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/27_icon_Ticket_sales_end_soon.png
try:
    _c27 = get_crop(27, 489, 84)
    canvas.paste(_c27, (88, 1370), _c27)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [88, 1370, 577, 1454]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/28_icon_LflL_M-A.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (864, 2804), _c28)
except Exception:
    pass
layout["LflL_M-A"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/29_icon_Free.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["Free"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/30_icon_Promoted.png
try:
    _c30 = get_crop(30, 247, 62)
    canvas.paste(_c30, (83, 1764), _c30)
except Exception:
    pass
layout["Promoted"] = [83, 1764, 330, 1826]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/31_text_2_001_events.png
try:
    _c31 = get_crop(31, 372, 135)
    canvas.paste(_c31, (54, 390), _c31)
except Exception:
    pass
layout["2,001_events"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_05_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-7/32_text_Joe_Goode_Performance_Group.png
try:
    _c32 = get_crop(32, 575, 55)
    canvas.paste(_c32, (90, 1705), _c32)
except Exception:
    pass
layout["Joe_Goode_Performance_Gro"] = [90, 1705, 665, 1760]
