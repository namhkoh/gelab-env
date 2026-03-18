# page_id: page_eventbrite_4fbf805fbd914a178f72f68b0bc03f81_01
# screenshot: 2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3.png
# step_index: 1/10
# task: Open Eventbrite. Explore "Education" events. Apply filters for events happening tomorrow. From the list, select the third event and check out its description.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw app background and structural UI elements for the Event list page.
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Full-canvas background (very light off-white to match screenshot)
draw.rectangle((0, 0, 1440, 2960), fill="#fbfbfc")

# Status bar area at top (~56px high)
draw.rectangle((0, 0, 1440, 56), fill="#cfcfcf")

# Top toolbar / header area background (behind search bar and logo)
# Provide a subtle slightly darker band under the status bar
draw.rectangle((0, 56, 1440, 220), fill="#ffffff")

# Rounded search bar background (large, centered under status bar)
search_left, search_top, search_right, search_bottom = 48, 68, 1392, 184
draw.rounded_rectangle(
    (search_left, search_top, search_right, search_bottom),
    radius=64,
    fill="#f6f6f8",
    outline="#e6e6e9",
    width=2
)

# Thin divider under header
draw.line((48, search_bottom + 12, 1392, search_bottom + 12), fill="#efedf4", width=1)

# Card rows (rounded white cards with subtle outlines and shadows)
cards = [
    # (x, y, width, height) - using detected row positions/sizes
    (48, 886, 1344, 396),
    (48, 1282, 1344, 396),
    (48, 1678, 1344, 396),
    (48, 2074, 1344, 396),
    (48, 2470, 1344, 346)  # slightly shorter card near the bottom
]

for (x, y, w, h) in cards:
    left, top, right, bottom = x, y, x + w, y + h

    # subtle shadow under each card (light translucent effect simulated with a thin filled rect)
    shadow_rect = (left + 6, bottom - 4, right + 6, bottom + 6)
    draw.rectangle(shadow_rect, fill="#f2f2f4")

    # card background (white) with soft outline
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=14,
        fill="#ffffff",
        outline="#ebe8f0",
        width=1
    )

    # subtle internal separator line to mimic list row separation (below card)
    draw.line((left + 12, bottom + 8, right - 12, bottom + 8), fill="#f0eef4", width=1)

# Additional subtle separators between the stacked card areas (across full content width)
sep_positions = [ (cards[0][1] - 12), (cards[1][1] - 12), (cards[2][1] - 12), (cards[3][1] - 12) ]
for sy in sep_positions:
    draw.line((48, sy, 1392, sy), fill="#fafafa", width=1)

# Bottom navigation bar background and top divider
nav_top = 2804
draw.rectangle((0, nav_top, 1440, 2960), fill="#ffffff")
draw.line((0, nav_top, 1440, nav_top), fill="#eae8ee", width=2)

# Slight elevation shadow above nav to set it apart
draw.rectangle((0, nav_top - 6, 1440, nav_top - 2), fill="#f4f3f6")

# Safe area left/right subtle guides (visual structure only, very faint)
draw.line((48, 220, 48, nav_top - 10), fill="#ffffff", width=1)
draw.line((1392, 220, 1392, nav_top - 10), fill="#ffffff", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/00_icon_ter_for_Break_Into_Tech_nowl.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 886), _c0)
except Exception:
    pass
layout["ter_for_Break_Into_Tech_n"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/01_icon_Q_Search_events.png
try:
    _c1 = get_crop(1, 1179, 144)
    canvas.paste(_c1, (195, 93), _c1)
except Exception:
    pass
layout["Q_Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/02_icon_FRIDAY.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 2074), _c2)
except Exception:
    pass
layout["FRIDAY"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/03_icon_NDIE.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1282), _c3)
except Exception:
    pass
layout["NDIE"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/04_icon_NDIE.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 490), _c4)
except Exception:
    pass
layout["NDIE"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/05_icon_Iaightsinel.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1678), _c5)
except Exception:
    pass
layout["Iaightsinel"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/06_icon_9_00_PM_PDT.png
try:
    _c6 = get_crop(6, 1344, 396)
    canvas.paste(_c6, (48, 490), _c6)
except Exception:
    pass
layout["9:00_PM_PDT"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/07_icon_Los_Angeles.png
try:
    _c7 = get_crop(7, 456, 117)
    canvas.paste(_c7, (492, 2651), _c7)
except Exception:
    pass
layout["Los_Angeles"] = [492, 2651, 948, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 747), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/09_icon_Favorite_button.png
try:
    _c9 = get_crop(9, 144, 123)
    canvas.paste(_c9, (1140, 1951), _c9)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/10_icon_Afliccion_Perdida_y.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 2347), _c10)
except Exception:
    pass
layout["Afliccion,_Perdida_y"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/11_icon_Favorite_button.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1140, 1539), _c11)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 139)
    canvas.paste(_c12, (1284, 747), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/13_icon_Indie_Sleaze_4_26_Club_Decades.png
try:
    _c13 = get_crop(13, 1344, 396)
    canvas.paste(_c13, (48, 1282), _c13)
except Exception:
    pass
layout["Indie_Sleaze_4_26_@_Club_"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/14_icon_Sylmai.png
try:
    _c14 = get_crop(14, 288, 156)
    canvas.paste(_c14, (288, 2804), _c14)
except Exception:
    pass
layout["Sylmai"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 123)
    canvas.paste(_c15, (1284, 2347), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/16_icon_Overflow_menu_button.png
try:
    _c16 = get_crop(16, 144, 123)
    canvas.paste(_c16, (1284, 1951), _c16)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/17_icon_8_21125_creator_followers.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 1678), _c17)
except Exception:
    pass
layout["8_21125_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/18_icon_Favorite_button.png
try:
    _c18 = get_crop(18, 144, 123)
    canvas.paste(_c18, (1140, 1159), _c18)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/19_icon_Overflow_menu_button.png
try:
    _c19 = get_crop(19, 144, 123)
    canvas.paste(_c19, (1284, 1159), _c19)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/20_icon_Overflow_menu_button.png
try:
    _c20 = get_crop(20, 144, 139)
    canvas.paste(_c20, (1284, 1539), _c20)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/21_icon_Home.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (0, 2804), _c21)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 60, 59)
    canvas.paste(_c22, (312, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [312, 3, 372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/23_icon_4.55.png
try:
    _c23 = get_crop(23, 99, 96)
    canvas.paste(_c23, (43, 123), _c23)
except Exception:
    pass
layout["4.55"] = [43, 123, 142, 219]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/24_icon_4.55.png
try:
    _c24 = get_crop(24, 56, 61)
    canvas.paste(_c24, (182, 2), _c24)
except Exception:
    pass
layout["4.55"] = [182, 2, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 50, 60)
    canvas.paste(_c25, (248, 2), _c25)
except Exception:
    pass
layout["icon_25"] = [248, 2, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 47, 53)
    canvas.paste(_c26, (1321, 7), _c26)
except Exception:
    pass
layout["icon_26"] = [1321, 7, 1368, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/27_icon_Public_House_Los_Angeles_CA.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 886), _c27)
except Exception:
    pass
layout["Public_House_(Los_Angeles"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/28_icon_4.55.png
try:
    _c28 = get_crop(28, 59, 61)
    canvas.paste(_c28, (115, 2), _c28)
except Exception:
    pass
layout["4.55"] = [115, 2, 174, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 67, 57)
    canvas.paste(_c29, (1212, 5), _c29)
except Exception:
    pass
layout["icon_29"] = [1212, 5, 1279, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/30_icon_Free.png
try:
    _c30 = get_crop(30, 1344, 346)
    canvas.paste(_c30, (48, 2470), _c30)
except Exception:
    pass
layout["Free"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/31_icon_Q_Search_events.png
try:
    _c31 = get_crop(31, 44, 56)
    canvas.paste(_c31, (385, 7), _c31)
except Exception:
    pass
layout["Q_Search_events"] = [385, 7, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/32_icon_Free.png
try:
    _c32 = get_crop(32, 126, 74)
    canvas.paste(_c32, (247, 560), _c32)
except Exception:
    pass
layout["Free"] = [247, 560, 373, 634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/33_icon_icon_33.png
try:
    _c33 = get_crop(33, 41, 55)
    canvas.paste(_c33, (1272, 6), _c33)
except Exception:
    pass
layout["icon_33"] = [1272, 6, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/34_icon_Indie_Sleaze_4_26_Club_Decades.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 1282), _c34)
except Exception:
    pass
layout["Indie_Sleaze_4_26_@_Club_"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/35_icon_Punk_Indie_Rock_Dance_Party.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 2074), _c35)
except Exception:
    pass
layout["Punk;_Indie_Rock_Dance_Pa"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/36_icon_Tickets.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (864, 2804), _c36)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/37_icon_BIZARRE_LOVE_TRIANGLE_New_Wave_Post.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 2074), _c37)
except Exception:
    pass
layout["BIZARRE_LOVE_TRIANGLE:_Ne"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/38_icon_31_creator_followers.png
try:
    _c38 = get_crop(38, 288, 156)
    canvas.paste(_c38, (576, 2804), _c38)
except Exception:
    pass
layout["31_creator_followers"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/39_icon_5.30_PM_PDT.png
try:
    _c39 = get_crop(39, 1344, 346)
    canvas.paste(_c39, (48, 2470), _c39)
except Exception:
    pass
layout["5.30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/40_text_4.55.png
try:
    _c40 = get_crop(40, 92, 43)
    canvas.paste(_c40, (22, 17), _c40)
except Exception:
    pass
layout["4.55"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/41_text_More_events_you_II_love.png
try:
    _c41 = get_crop(41, 1344, 396)
    canvas.paste(_c41, (48, 490), _c41)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/42_text_Mon_May_13.png
try:
    _c42 = get_crop(42, 222, 43)
    canvas.paste(_c42, (393, 2525), _c42)
except Exception:
    pass
layout["Mon,_May_13"] = [393, 2525, 615, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/43_text_5.30_PM_PDT.png
try:
    _c43 = get_crop(43, 1344, 346)
    canvas.paste(_c43, (48, 2470), _c43)
except Exception:
    pass
layout["5.30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/44_text_31_creator_followers.png
try:
    _c44 = get_crop(44, 1344, 346)
    canvas.paste(_c44, (48, 2470), _c44)
except Exception:
    pass
layout["31_creator_followers"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_01_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-3/45_clickable_More.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (1152, 2804), _c45)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
