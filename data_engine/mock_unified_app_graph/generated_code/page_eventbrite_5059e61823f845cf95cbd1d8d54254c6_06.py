# page_id: page_eventbrite_5059e61823f845cf95cbd1d8d54254c6_06
# screenshot: 2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8.png
# step_index: 6/19
# task: Open Eventbrite. Look for 'Education' in Los Angeles happening on May 4. Filter to show only free events. How many events are posted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background with a very light off-white
draw.rectangle((0, 0, 1440, 2960), fill=(250, 250, 252))

# Status bar area (top ~96px) - subtle gray
draw.rectangle((0, 0, 1440, 96), fill=(190, 190, 190))

# Header / toolbar background area below status bar
header_top = 96
header_bottom = 220
draw.rectangle((0, header_top, 1440, header_bottom), fill=(255, 255, 255))
# subtle divider under header
draw.line((48, header_bottom, 1392, header_bottom), fill=(235, 235, 238), width=1)

# Card positions and sizes (use rounded white cards with soft shadow)
card_x1 = 48
card_width = 1344
card_x2 = card_x1 + card_width
card_positions = [
    (48, 490, 48 + 1344, 490 + 396),
    (48, 886, 48 + 1344, 886 + 396),
    (48, 1282, 48 + 1344, 1282 + 396),
    (48, 1678, 48 + 1344, 1678 + 396),
    (48, 2074, 48 + 1344, 2074 + 396),
    (48, 2470, 48 + 1344, 2470 + 346),
]

for (x1, y1, x2, y2) in card_positions:
    # soft shadow (slightly offset)
    shadow_offset = 8
    draw.rounded_rectangle(
        (x1 + shadow_offset, y1 + shadow_offset, x2 + shadow_offset, y2 + shadow_offset),
        radius=24,
        fill=(245, 245, 247),
    )
    # card background (white)
    draw.rounded_rectangle((x1, y1, x2, y2), radius=24, fill=(255, 255, 255), outline=None)
    # subtle separator line below each card to emphasize stacking (light)
    sep_y = y2 + 20
    draw.line((x1 + 16, sep_y, x2 - 16, sep_y), fill=(238, 238, 241), width=1)

# Additional thin separators between list items (to match subtle UI dividers)
for (_, y1, _, y2) in card_positions:
    # place a very light divider a few pixels below the card bottom
    div_y = y2 + 6
    draw.line((card_x1 + 12, div_y, card_x2 - 12, div_y), fill=(245, 245, 247), width=1)

# Bottom navigation bar background (space for icons will be overlaid)
bottom_nav_top = 2804
draw.rectangle((0, bottom_nav_top, 1440, 2960), fill=(250, 250, 251))
# top divider of nav bar
draw.line((0, bottom_nav_top, 1440, bottom_nav_top), fill=(230, 230, 233), width=2)

# Floating location/controls area background hint (do not draw the actual control content)
# Draw a subtle pill shadow area center-bottom (background only)
pill_center_x = 720
pill_center_y = 2656
pill_w = 420
pill_h = 90
pill_x1 = pill_center_x - pill_w // 2
pill_y1 = pill_center_y - pill_h // 2
pill_x2 = pill_center_x + pill_w // 2
pill_y2 = pill_center_y + pill_h // 2
# shadow
draw.rounded_rectangle((pill_x1+6, pill_y1+8, pill_x2+6, pill_y2+8), radius=40, fill=(245,245,247))
# pill background (very light)
draw.rounded_rectangle((pill_x1, pill_y1, pill_x2, pill_y2), radius=40, fill=(255,255,255))

# Top-left app margin accent (subtle left edge padding visual)
draw.rectangle((0, header_bottom, 48, 2960), fill=(250,250,252))

# small decorative left gutter line (very light) to match UI rhythm
draw.line((48, header_bottom + 8, 48, bottom_nav_top - 8), fill=(247, 247, 249), width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/00_icon_NDIE.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1678), _c0)
except Exception:
    pass
layout["NDIE"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/01_icon_Q_Search_events.png
try:
    _c1 = get_crop(1, 1179, 144)
    canvas.paste(_c1, (195, 93), _c1)
except Exception:
    pass
layout["Q_Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/02_icon_FRIDAY.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 2074), _c2)
except Exception:
    pass
layout["FRIDAY"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/03_icon_REoPUNKSFRE.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 886), _c3)
except Exception:
    pass
layout["REoPUNKSFRE"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/04_icon_Los_Angeles.png
try:
    _c4 = get_crop(4, 456, 117)
    canvas.paste(_c4, (492, 2651), _c4)
except Exception:
    pass
layout["Los_Angeles"] = [492, 2651, 948, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/05_icon_NDIE_DANCEPA.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1282), _c5)
except Exception:
    pass
layout["NDIE_DANCEPA"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 123)
    canvas.paste(_c6, (1140, 763), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 123)
    canvas.paste(_c7, (1140, 1159), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 49, 65)
    canvas.paste(_c8, (1153, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [1153, 2, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/09_icon_Club_Decades.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1140, 1935), _c9)
except Exception:
    pass
layout["Club_Decades"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/10_icon_Afliccion_Perdida_y.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 2347), _c10)
except Exception:
    pass
layout["Afliccion,_Perdida_y"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 123)
    canvas.paste(_c11, (1284, 1159), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1284, 2347), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/13_icon_Favorite_button.png
try:
    _c13 = get_crop(13, 144, 139)
    canvas.paste(_c13, (1140, 1539), _c13)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1284, 1539), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 123)
    canvas.paste(_c15, (1284, 763), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/16_icon_8_59_creator_followers.png
try:
    _c16 = get_crop(16, 1344, 396)
    canvas.paste(_c16, (48, 1282), _c16)
except Exception:
    pass
layout["8_59_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/17_icon_Public_House_Los_Angeles_CA.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 490), _c17)
except Exception:
    pass
layout["Public_House_(Los_Angeles"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/18_icon_Sylmai.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (288, 2804), _c18)
except Exception:
    pass
layout["Sylmai"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/19_icon_8_21119_creator_followers.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 886), _c19)
except Exception:
    pass
layout["8_21119_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/20_icon_Overflow_menu_button.png
try:
    _c20 = get_crop(20, 144, 139)
    canvas.paste(_c20, (1284, 1935), _c20)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/21_icon_Home.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (0, 2804), _c21)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 97, 60)
    canvas.paste(_c22, (1216, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [1216, 3, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 60, 59)
    canvas.paste(_c23, (312, 3), _c23)
except Exception:
    pass
layout["icon_23"] = [312, 3, 372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/24_icon_7.35.png
try:
    _c24 = get_crop(24, 57, 60)
    canvas.paste(_c24, (182, 2), _c24)
except Exception:
    pass
layout["7.35"] = [182, 2, 239, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 52, 60)
    canvas.paste(_c25, (247, 2), _c25)
except Exception:
    pass
layout["icon_25"] = [247, 2, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/26_icon_7.35.png
try:
    _c26 = get_crop(26, 102, 99)
    canvas.paste(_c26, (41, 122), _c26)
except Exception:
    pass
layout["7.35"] = [41, 122, 143, 221]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 48, 53)
    canvas.paste(_c27, (1321, 7), _c27)
except Exception:
    pass
layout["icon_27"] = [1321, 7, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/28_icon_8_4717_creator_followers.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 1678), _c28)
except Exception:
    pass
layout["8_4717_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/29_icon_7.35.png
try:
    _c29 = get_crop(29, 59, 61)
    canvas.paste(_c29, (115, 2), _c29)
except Exception:
    pass
layout["7.35"] = [115, 2, 174, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/30_icon_Free.png
try:
    _c30 = get_crop(30, 1344, 346)
    canvas.paste(_c30, (48, 2470), _c30)
except Exception:
    pass
layout["Free"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/31_icon_Q_Search_events.png
try:
    _c31 = get_crop(31, 44, 57)
    canvas.paste(_c31, (385, 6), _c31)
except Exception:
    pass
layout["Q_Search_events"] = [385, 6, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/32_icon_ter_for_Break_Into_Tech_nowl.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 490), _c32)
except Exception:
    pass
layout["ter_for_Break_Into_Tech_n"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/33_icon_Sun_Apr_28.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 1678), _c33)
except Exception:
    pass
layout["Sun,_Apr_28"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/34_icon_Free.png
try:
    _c34 = get_crop(34, 130, 74)
    canvas.paste(_c34, (244, 1352), _c34)
except Exception:
    pass
layout["Free"] = [244, 1352, 374, 1426]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/35_icon_Tickets.png
try:
    _c35 = get_crop(35, 288, 156)
    canvas.paste(_c35, (864, 2804), _c35)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/36_icon_BIZARRE_LOVE_TRIANGLE_New_Wave_Post.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 2074), _c36)
except Exception:
    pass
layout["BIZARRE_LOVE_TRIANGLE:_Ne"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/37_icon_31_creator_followers.png
try:
    _c37 = get_crop(37, 288, 156)
    canvas.paste(_c37, (576, 2804), _c37)
except Exception:
    pass
layout["31_creator_followers"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/38_icon_Break_into_Tech_Social_Broxton_Brewery.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 490), _c38)
except Exception:
    pass
layout["Break_into_Tech_Social:_B"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/39_icon_5.30_PM_PDT.png
try:
    _c39 = get_crop(39, 1344, 346)
    canvas.paste(_c39, (48, 2470), _c39)
except Exception:
    pass
layout["5.30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/40_text_7.35.png
try:
    _c40 = get_crop(40, 94, 45)
    canvas.paste(_c40, (20, 15), _c40)
except Exception:
    pass
layout["7.35"] = [20, 15, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/41_text_More_events_you_II_love.png
try:
    _c41 = get_crop(41, 1344, 396)
    canvas.paste(_c41, (48, 490), _c41)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/42_text_Fri_May_3_._9_00_PM_PDT.png
try:
    _c42 = get_crop(42, 1344, 396)
    canvas.paste(_c42, (48, 2074), _c42)
except Exception:
    pass
layout["Fri,_May_3_._9:00_PM_PDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/43_text_Bar_Franca.png
try:
    _c43 = get_crop(43, 175, 38)
    canvas.paste(_c43, (394, 2328), _c43)
except Exception:
    pass
layout["Bar_Franca"] = [394, 2328, 569, 2366]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/44_text_202_creator_followers.png
try:
    _c44 = get_crop(44, 1344, 396)
    canvas.paste(_c44, (48, 2074), _c44)
except Exception:
    pass
layout["202_creator_followers"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/45_text_Mon_May_13.png
try:
    _c45 = get_crop(45, 222, 43)
    canvas.paste(_c45, (393, 2525), _c45)
except Exception:
    pass
layout["Mon,_May_13"] = [393, 2525, 615, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/46_text_5.30_PM_PDT.png
try:
    _c46 = get_crop(46, 1344, 346)
    canvas.paste(_c46, (48, 2470), _c46)
except Exception:
    pass
layout["5.30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/47_text_31_creator_followers.png
try:
    _c47 = get_crop(47, 1344, 346)
    canvas.paste(_c47, (48, 2470), _c47)
except Exception:
    pass
layout["31_creator_followers"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_06_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-8/48_clickable_More.png
try:
    _c48 = get_crop(48, 288, 156)
    canvas.paste(_c48, (1152, 2804), _c48)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
