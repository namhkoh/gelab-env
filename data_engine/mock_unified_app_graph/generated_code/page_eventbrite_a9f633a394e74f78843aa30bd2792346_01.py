# page_id: page_eventbrite_a9f633a394e74f78843aa30bd2792346_01
# screenshot: 2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3.png
# step_index: 1/18
# task: Open Eventbrite. Set the city to "Los Angeles". Look for Photography workshops happening next week. What is the price of the tickets for first non-promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (1440x2960 white). Fonts: font_sm, font_md, font_lg, font_xl.

# Background fill (subtle off-white to match app background)
bg_color = (250, 250, 252)  # very light off-white
draw.rectangle([(0, 0), (1440, 2960)], fill=bg_color)

# Status bar (top ~56px) - muted gray background
status_h = 56
status_color = (187, 187, 187)
draw.rectangle([(0, 0), (1440, status_h)], fill=status_color)

# Header / toolbar area (below status bar) - white panel with subtle bottom divider
header_top = status_h
header_bottom = 220
header_color = (255, 255, 255)
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=header_color)
# subtle divider line under header
divider_color = (230, 230, 235)
draw.line([(24, header_bottom), (1440-24, header_bottom)], fill=divider_color, width=1)

# Subtle shadow line below header (very faint)
shadow_color = (240, 240, 245)
draw.line([(24, header_bottom+1), (1440-24, header_bottom+1)], fill=shadow_color, width=1)

# Section: "More events you'll love" area left as background (no text drawn)
# Draw event card backgrounds (rounded rectangles) stacked down the page
card_x = 48
card_w = 1344
card_h = 396
card_radius = 14
card_fill = (255, 255, 255)  # white cards
card_outline = (230, 230, 235)  # subtle border

# Positions inferred from detected elements: y = 490, 886, 1282, 1678, 2074, 2470
start_ys = [490, 886, 1282, 1678, 2074, 2470]
for y in start_ys:
    # main card background
    draw.rounded_rectangle(
        [(card_x, y), (card_x + card_w, y + card_h)],
        radius=card_radius,
        fill=card_fill,
        outline=card_outline,
        width=1
    )
    # subtle inner top shadow strip to give slight depth (do not draw any icons/content)
    top_strip_h = 8
    strip_color = (248, 248, 250)
    draw.rectangle([(card_x + 1, y + 1), (card_x + card_w - 1, y + top_strip_h)], fill=strip_color)

# Separator lines between cards (thin)
sep_color = (245, 245, 246)
for i in range(len(start_ys) - 1):
    y_sep = start_ys[i] + card_h + 12
    # Keep separator subtle and aligned full width within content margins
    draw.line([(card_x + 8, y_sep), (card_x + card_w - 8, y_sep)], fill=sep_color, width=1)

# Draw a larger content area band behind the list to visually group it (subtle)
content_band_top = start_ys[0] - 24
content_band_bottom = start_ys[-1] + card_h + 24
band_color = (250, 250, 252)
draw.rectangle([(0, content_band_top), (1440, content_band_bottom)], fill=band_color)

# Floating selection pill area near bottom (background only) - leave interactive content to be pasted later
# This draws the subtle rounded white pill background behind the location drop-down (do not draw text/icon)
pill_w = 360
pill_h = 84
pill_x = int((1440 - pill_w) / 2)
pill_y = 2520
pill_color = (255, 255, 255)
pill_outline = (230, 230, 235)
draw.rounded_rectangle(
    [(pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h)],
    radius=42,
    fill=pill_color,
    outline=pill_outline,
    width=1
)

# Bottom navigation bar background
nav_top = 2804
nav_bottom = 2960
nav_bg = (255, 255, 255)
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill=nav_bg)
# top divider for nav
draw.line([(0, nav_top), (1440, nav_top)], fill=divider_color, width=1)

# Highlight background under the left-most (Home) tab to indicate selection (background only)
home_tab_w = 288
home_tab_h = nav_bottom - nav_top
home_tab_x = 0
home_tab_y = nav_top
# subtle orange indicator bar above the icon area (background only)
indicator_color = (236, 95, 43)  # Eventbrite orange
indicator_h = 4
draw.rectangle([(home_tab_x + 90, nav_top + 10), (home_tab_x + 90 + 108, nav_top + 10 + indicator_h)], fill=indicator_color)

# Small faint shadows under each card to lift them slightly
shadow_color_card = (245, 245, 246)
for y in start_ys:
    sx1 = card_x + 6
    sx2 = card_x + card_w - 6
    sy = y + card_h + 6
    draw.rectangle([(sx1, sy), (sx2, sy + 2)], fill=shadow_color_card)

# End of background and structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/00_icon_Search_events.png
try:
    _c0 = get_crop(0, 1179, 144)
    canvas.paste(_c0, (195, 93), _c0)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/01_icon_Washington.png
try:
    _c1 = get_crop(1, 454, 117)
    canvas.paste(_c1, (493, 2651), _c1)
except Exception:
    pass
layout["Washington"] = [493, 2651, 947, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/02_icon_Workshop_on_Grief_and_Remembrance.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1678), _c2)
except Exception:
    pass
layout["Workshop_on_Grief_and_Rem"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/03_icon_Yoga_for_Grief_Loss.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1282), _c3)
except Exception:
    pass
layout["Yoga_for_Grief_&_Loss"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 123)
    canvas.paste(_c4, (1140, 2347), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/05_icon_Pain.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 2074), _c5)
except Exception:
    pass
layout["Pain"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 123)
    canvas.paste(_c6, (1284, 2347), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/07_icon_Rockvin.png
try:
    _c7 = get_crop(7, 288, 156)
    canvas.paste(_c7, (288, 2804), _c7)
except Exception:
    pass
layout["Rockvin"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 747), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/09_icon_Home.png
try:
    _c9 = get_crop(9, 288, 156)
    canvas.paste(_c9, (0, 2804), _c9)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/10_icon_Favorite_button.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 1951), _c10)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/11_icon_Favorite_button.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1140, 1539), _c11)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 139)
    canvas.paste(_c12, (1284, 747), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/13_icon_4.50.png
try:
    _c13 = get_crop(13, 99, 93)
    canvas.paste(_c13, (43, 124), _c13)
except Exception:
    pass
layout["4.50"] = [43, 124, 142, 217]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1284, 1143), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 139)
    canvas.paste(_c15, (1284, 1539), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 58, 57)
    canvas.paste(_c16, (313, 4), _c16)
except Exception:
    pass
layout["icon_16"] = [313, 4, 371, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/17_icon_4.50.png
try:
    _c17 = get_crop(17, 55, 59)
    canvas.paste(_c17, (183, 3), _c17)
except Exception:
    pass
layout["4.50"] = [183, 3, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 50, 58)
    canvas.paste(_c18, (248, 3), _c18)
except Exception:
    pass
layout["icon_18"] = [248, 3, 298, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/19_icon_Favorite_button.png
try:
    _c19 = get_crop(19, 144, 139)
    canvas.paste(_c19, (1140, 1143), _c19)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 47, 51)
    canvas.paste(_c20, (1321, 8), _c20)
except Exception:
    pass
layout["icon_20"] = [1321, 8, 1368, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/21_icon_Yoga_for_Grief_Loss.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 886), _c21)
except Exception:
    pass
layout["Yoga_for_Grief_&_Loss"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/22_icon_Wendt_Center_For_Loss_and_Healing.png
try:
    _c22 = get_crop(22, 1344, 396)
    canvas.paste(_c22, (48, 2074), _c22)
except Exception:
    pass
layout["Wendt_Center_For_Loss_and"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/23_icon_226_creator_followers.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 490), _c23)
except Exception:
    pass
layout["226_creator_followers"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/24_icon_Overflow_menu_button.png
try:
    _c24 = get_crop(24, 144, 123)
    canvas.paste(_c24, (1284, 1951), _c24)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/25_icon_4.50.png
try:
    _c25 = get_crop(25, 58, 60)
    canvas.paste(_c25, (115, 2), _c25)
except Exception:
    pass
layout["4.50"] = [115, 2, 173, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 41, 53)
    canvas.paste(_c26, (1272, 7), _c26)
except Exception:
    pass
layout["icon_26"] = [1272, 7, 1313, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 54, 54)
    canvas.paste(_c27, (1214, 7), _c27)
except Exception:
    pass
layout["icon_27"] = [1214, 7, 1268, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 44, 55)
    canvas.paste(_c28, (385, 7), _c28)
except Exception:
    pass
layout["icon_28"] = [385, 7, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/29_icon_Tickets.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (864, 2804), _c29)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/30_icon_1I_00AM_EDT.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 886), _c30)
except Exception:
    pass
layout["1I:00AM_EDT"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/31_icon_Partner_Loss_Grief_Knots_and_Sitting_wit.png
try:
    _c31 = get_crop(31, 1344, 396)
    canvas.paste(_c31, (48, 2074), _c31)
except Exception:
    pass
layout["Partner_Loss,_Grief_Knots"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/32_icon_Collaging_Me.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 1282), _c32)
except Exception:
    pass
layout["Collaging_Me"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/33_text_4.50.png
try:
    _c33 = get_crop(33, 89, 43)
    canvas.paste(_c33, (22, 17), _c33)
except Exception:
    pass
layout["4.50"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/34_text_More_events_you_II_love.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 490), _c34)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/35_text_Tue_Apr_30.png
try:
    _c35 = get_crop(35, 200, 43)
    canvas.paste(_c35, (390, 2525), _c35)
except Exception:
    pass
layout["Tue,_Apr_30"] = [390, 2525, 590, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/36_text_7_00_PM_EDT.png
try:
    _c36 = get_crop(36, 1344, 346)
    canvas.paste(_c36, (48, 2470), _c36)
except Exception:
    pass
layout["7:00_PM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/37_text_Free.png
try:
    _c37 = get_crop(37, 78, 38)
    canvas.paste(_c37, (274, 2561), _c37)
except Exception:
    pass
layout["Free"] = [274, 2561, 352, 2599]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/38_clickable_Favorites.png
try:
    _c38 = get_crop(38, 288, 156)
    canvas.paste(_c38, (576, 2804), _c38)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_01_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-3/39_clickable_More.png
try:
    _c39 = get_crop(39, 288, 156)
    canvas.paste(_c39, (1152, 2804), _c39)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
