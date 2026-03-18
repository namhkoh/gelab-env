# page_id: page_eventbrite_e381830686d842d08e553d1397c2110d_01
# screenshot: 2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3.png
# step_index: 1/3
# task: Open Eventbrite. Open "Recommended". Select the third recommended event. Add it to favourites. What is the refund policy?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Status bar (top)
draw.rectangle([(0, 0), (1440, 60)], fill="#d0d0d0")

# Header / toolbar background (below status bar)
header_top = 60
header_bottom = 160
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#ffffff")
# subtle bottom divider under header
draw.line([(24, header_bottom), (1440-24, header_bottom)], fill="#e6e6e6", width=1)

# Main content subtle background (a very light warm tint to match screenshot tone)
draw.rectangle([(0, header_bottom), (1440, 2760)], fill="#ffffff")

# Define card positions based on detected list rows
cards = [
    (48, 490, 48 + 1344, 490 + 396),
    (48, 886, 48 + 1344, 886 + 396),
    (48, 1282, 48 + 1344, 1282 + 396),
    (48, 1678, 48 + 1344, 1678 + 396),
    (48, 2074, 48 + 1344, 2074 + 396),
    (48, 2470, 48 + 1344, 2470 + 346),
]

# Draw subtle card shadows and white card bodies with rounded corners and soft borders
for (x0, y0, x1, y1) in cards:
    # shadow
    sh_offset = 6
    draw.rounded_rectangle([(x0+6, y0+sh_offset), (x1+6, y1+sh_offset)], radius=16, fill="#f5f5f6")
    # card body
    draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=16, fill="#ffffff", outline="#efeff1", width=1)
    # separator line under card (subtle)
    draw.line([(x0+12, y1+14), (x1-12, y1+14)], fill="#f0f0f0", width=1)

# Additional subtle separators between items (full width, light)
sep_positions = [cards[i][3] + 28 for i in range(len(cards))]
for y in sep_positions:
    if y < 2760:
        draw.line([(36, y), (1440-36, y)], fill="#fafafa", width=1)

# Floating content area hint (a light rounded pill behind the floating "Find" control area)
# Position it low on the screen but do not overlap nav bar; keep faint shadow only
float_box = (320, 2560, 1120, 2650)
draw.rounded_rectangle([(float_box[0]+4, float_box[1]+8), (float_box[2]+4, float_box[3]+8)], radius=36, fill="#f3f3f3")
draw.rounded_rectangle([float_box[0], float_box[1], float_box[2], float_box[3]], radius=36, fill="#ffffff", outline="#e9e9ea")

# Bottom navigation bar background and top divider
nav_top = 2760
draw.rectangle([(0, nav_top), (1440, 2960)], fill="#ffffff")
draw.line([(24, nav_top), (1440-24, nav_top)], fill="#e6e6e6", width=1)

# Small active indicator bar under the left-most nav item (subtle, not an icon)
active_x_center = 144  # corresponds roughly to left-most icon center
indicator_w = 56
draw.rounded_rectangle([(active_x_center - indicator_w/2, nav_top + 8),
                        (active_x_center + indicator_w/2, nav_top + 18)],
                       radius=9, fill="#ff6b2d")

# Top-of-list big title area subtle spacing box (behind "More events you'll love")
title_bg = (48, 200, 1392, 320)
draw.rectangle([title_bg[0], title_bg[1], title_bg[2], title_bg[3]], fill="#ffffff")
# faint underline for the title area
draw.line([(title_bg[0], title_bg[3] + 6), (title_bg[2], title_bg[3] + 6)], fill="#f2f2f2", width=1)

# A subtle left edge visual guide (vertical divider) to separate thumbnails from text columns
# This is a faint vertical line aligned near the left thumbnail area to guide layout (not an icon)
thumb_div_x = 48 + 144  # thumbnails roughly 144px wide in the layout
draw.line([(thumb_div_x + 8, header_bottom + 12), (thumb_div_x + 8, nav_top - 12)], fill="#fafafa", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/00_icon_YG.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["YG"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/01_icon_Online.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 490), _c1)
except Exception:
    pass
layout["Online"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/02_icon_Online.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1678), _c2)
except Exception:
    pass
layout["Online"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/03_icon_Q_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Q_Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/04_icon_Or.png
try:
    _c4 = get_crop(4, 288, 156)
    canvas.paste(_c4, (288, 2804), _c4)
except Exception:
    pass
layout["Or,"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/05_icon_Loss.png
try:
    _c5 = get_crop(5, 144, 125)
    canvas.paste(_c5, (1140, 2345), _c5)
except Exception:
    pass
layout["Loss"] = [1140, 2345, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 125)
    canvas.paste(_c6, (1140, 1949), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1949, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/07_icon_7.02.png
try:
    _c7 = get_crop(7, 113, 106)
    canvas.paste(_c7, (35, 118), _c7)
except Exception:
    pass
layout["7.02"] = [35, 118, 148, 224]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 125)
    canvas.paste(_c8, (1284, 2345), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2345, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/09_icon_Understanding_Grief_and_Loss.png
try:
    _c9 = get_crop(9, 1344, 396)
    canvas.paste(_c9, (48, 1282), _c9)
except Exception:
    pass
layout["Understanding_Grief_and_L"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 139)
    canvas.paste(_c10, (1284, 1539), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/11_icon_Home.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (0, 2804), _c11)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 139)
    canvas.paste(_c12, (1284, 747), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 139)
    canvas.paste(_c13, (1284, 1143), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 125)
    canvas.paste(_c14, (1284, 1949), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1949, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 62, 59)
    canvas.paste(_c15, (311, 3), _c15)
except Exception:
    pass
layout["icon_15"] = [311, 3, 373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/16_icon_7.02.png
try:
    _c16 = get_crop(16, 55, 60)
    canvas.paste(_c16, (183, 3), _c16)
except Exception:
    pass
layout["7.02"] = [183, 3, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 51, 60)
    canvas.paste(_c17, (248, 2), _c17)
except Exception:
    pass
layout["icon_17"] = [248, 2, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/18_icon_Favorite_button.png
try:
    _c18 = get_crop(18, 144, 139)
    canvas.paste(_c18, (1140, 747), _c18)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/19_icon_Favorite_button.png
try:
    _c19 = get_crop(19, 144, 139)
    canvas.paste(_c19, (1140, 1539), _c19)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/20_icon_Favorite_button.png
try:
    _c20 = get_crop(20, 144, 139)
    canvas.paste(_c20, (1140, 1143), _c20)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/21_icon_Working_with_Grief_and_Loss.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 490), _c21)
except Exception:
    pass
layout["Working_with_Grief_and_Lo"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 48, 55)
    canvas.paste(_c22, (1321, 6), _c22)
except Exception:
    pass
layout["icon_22"] = [1321, 6, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/23_icon_1252_creator_followers.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 886), _c23)
except Exception:
    pass
layout["1252_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 96, 61)
    canvas.paste(_c24, (1211, 2), _c24)
except Exception:
    pass
layout["icon_24"] = [1211, 2, 1307, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/25_icon_Online_events.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["Online_events"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/26_icon_5.00AM_EST.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 1678), _c26)
except Exception:
    pass
layout["5.00AM_EST"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/27_icon_Q_Search_events.png
try:
    _c27 = get_crop(27, 44, 57)
    canvas.paste(_c27, (385, 6), _c27)
except Exception:
    pass
layout["Q_Search_events"] = [385, 6, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/28_icon_Loss.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (864, 2804), _c28)
except Exception:
    pass
layout["Loss"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/29_icon_7.02.png
try:
    _c29 = get_crop(29, 56, 61)
    canvas.paste(_c29, (116, 2), _c29)
except Exception:
    pass
layout["7.02"] = [116, 2, 172, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/30_icon_Online.png
try:
    _c30 = get_crop(30, 112, 55)
    canvas.paste(_c30, (390, 702), _c30)
except Exception:
    pass
layout["Online"] = [390, 702, 502, 757]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/31_icon_Ur.png
try:
    _c31 = get_crop(31, 60, 59)
    canvas.paste(_c31, (388, 2641), _c31)
except Exception:
    pass
layout["Ur"] = [388, 2641, 448, 2700]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/32_icon_Grief_and.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 1282), _c32)
except Exception:
    pass
layout["Grief_and"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/33_icon_Online_events.png
try:
    _c33 = get_crop(33, 586, 117)
    canvas.paste(_c33, (427, 2651), _c33)
except Exception:
    pass
layout["Online_events"] = [427, 2651, 1013, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/34_icon_Grief_and.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 886), _c34)
except Exception:
    pass
layout["Grief_and"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/35_text_7.02.png
try:
    _c35 = get_crop(35, 89, 43)
    canvas.paste(_c35, (22, 17), _c35)
except Exception:
    pass
layout["7.02"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/36_text_More_events_you_II_love.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 490), _c36)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/37_text_Sat_Jun_1_5_00_AM_EDT.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 2074), _c37)
except Exception:
    pass
layout["Sat,_Jun_1_+_5:00_AM_EDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/38_text_Understanding_your_Grief_and_Loss.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 2074), _c38)
except Exception:
    pass
layout["Understanding_your_Grief_"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/39_text_Online.png
try:
    _c39 = get_crop(39, 112, 38)
    canvas.paste(_c39, (392, 2323), _c39)
except Exception:
    pass
layout["Online"] = [392, 2323, 504, 2361]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/40_text_Sat.png
try:
    _c40 = get_crop(40, 77, 45)
    canvas.paste(_c40, (390, 2583), _c40)
except Exception:
    pass
layout["Sat,"] = [390, 2583, 467, 2628]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/41_text_5.00_AM_EDT.png
try:
    _c41 = get_crop(41, 1344, 346)
    canvas.paste(_c41, (48, 2470), _c41)
except Exception:
    pass
layout["5.00_AM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/42_text_Loss.png
try:
    _c42 = get_crop(42, 110, 57)
    canvas.paste(_c42, (1031, 2646), _c42)
except Exception:
    pass
layout["Loss"] = [1031, 2646, 1141, 2703]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e381830686d842d08e553d1397c2110d/step_01_2024_4_23_19_1_e381830686d842d08e553d1397c2110d-3/43_clickable_More.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (1152, 2804), _c43)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
