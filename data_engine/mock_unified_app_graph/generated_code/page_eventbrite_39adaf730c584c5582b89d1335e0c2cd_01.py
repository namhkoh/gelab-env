# page_id: page_eventbrite_39adaf730c584c5582b89d1335e0c2cd_01
# screenshot: 2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3.png
# step_index: 1/6
# task: Open Eventbrite. Search for 'food and drink' events. Follow the organizer of the first event in listing.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background (dominant color: near-white)
draw.rectangle([0, 0, 1440, 2960], fill="#ffffff")

# Top status bar background (~56px)
draw.rectangle([0, 0, 1440, 56], fill="#d6d6d6")

# Thin divider under status/header area
draw.line([(48, 260), (1392, 260)], fill="#e6e6e6", width=1)

# Define card positions (top-left x=48, width=1344, heights ~396)
card_tops = [490, 886, 1282, 1678, 2074, 2470]
card_left = 48
card_right = card_left + 1344
card_height = 396
corner_radius = 16

# Draw subtle shadow + rounded card backgrounds for each event group
for top in card_tops:
    # shadow (slightly offset)
    shadow_box = [card_left + 4, top + 8, card_right + 4, top + card_height + 8]
    draw.rounded_rectangle(shadow_box, radius=corner_radius, fill="#f2f2f2", outline=None)
    # card background
    card_box = [card_left, top, card_right, top + card_height]
    draw.rounded_rectangle(card_box, radius=corner_radius, fill="#ffffff", outline="#f0f0f0", width=1)

    # subtle separator line below each card (to emphasize sections)
    sep_y = top + card_height + 18
    draw.line([(card_left + 8, sep_y), (card_right - 8, sep_y)], fill="#f1f1f1", width=1)

# Large content area background (if any large image/content blocks exist further down)
# We'll draw a faint tinted band behind the lower content region (not overlapping detected floating pill)
band_top = 2380
band_bottom = 2640
draw.rectangle([0, band_top, 1440, band_bottom], fill="#fbfbfc")

# Top-of-list subtle heading area (behind the "More events you'll love" heading)
heading_bg_top = 200
heading_bg_bottom = 360
draw.rectangle([48, heading_bg_top, 1392, heading_bg_bottom], fill="#ffffff")
draw.line([(48, heading_bg_bottom), (1392, heading_bg_bottom)], fill="#e9e9e9", width=1)

# Bottom navigation bar background and top divider
nav_top = 2804
nav_bottom = 2960
draw.rectangle([0, nav_top, 1440, nav_bottom], fill="#ffffff")
draw.line([(0, nav_top), (1440, nav_top)], fill="#e1e1e1", width=2)

# Left and right page gutters (subtle vertical guides)
draw.line([(40, 56), (40, nav_top - 48)], fill="#fafafa", width=2)
draw.line([(1400, 56), (1400, nav_top - 48)], fill="#fafafa", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/00_icon_Free.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["Free"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/01_icon_NDIE.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1678), _c1)
except Exception:
    pass
layout["NDIE"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/02_icon_Los_Angeles.png
try:
    _c2 = get_crop(2, 456, 117)
    canvas.paste(_c2, (492, 2651), _c2)
except Exception:
    pass
layout["Los_Angeles"] = [492, 2651, 948, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/03_icon_REoPUNKSFRE.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 886), _c3)
except Exception:
    pass
layout["REoPUNKSFRE"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/04_icon_Search_events.png
try:
    _c4 = get_crop(4, 1179, 144)
    canvas.paste(_c4, (195, 93), _c4)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/05_icon_NDIE_DANCEPA.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1282), _c5)
except Exception:
    pass
layout["NDIE_DANCEPA"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 123)
    canvas.paste(_c6, (1140, 763), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/07_icon_Afliccion_Perdida_y.png
try:
    _c7 = get_crop(7, 144, 139)
    canvas.paste(_c7, (1140, 1935), _c7)
except Exception:
    pass
layout["Afliccion,_Perdida_y"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 123)
    canvas.paste(_c8, (1284, 1159), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/09_icon_Favorite_button.png
try:
    _c9 = get_crop(9, 144, 123)
    canvas.paste(_c9, (1140, 1159), _c9)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/10_icon_Afliccion_Perdida_y.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 2347), _c10)
except Exception:
    pass
layout["Afliccion,_Perdida_y"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 123)
    canvas.paste(_c11, (1284, 2347), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 139)
    canvas.paste(_c12, (1284, 1539), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 123)
    canvas.paste(_c13, (1284, 763), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/14_icon_Favorite_button.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1140, 1539), _c14)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 139)
    canvas.paste(_c15, (1284, 1935), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/16_icon_8_59_creator_followers.png
try:
    _c16 = get_crop(16, 1344, 396)
    canvas.paste(_c16, (48, 1282), _c16)
except Exception:
    pass
layout["8_59_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/17_icon_The_Gr.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (288, 2804), _c17)
except Exception:
    pass
layout["The_Gr"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 62, 59)
    canvas.paste(_c18, (311, 3), _c18)
except Exception:
    pass
layout["icon_18"] = [311, 3, 373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/19_icon_8_21119_creator_followers.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 886), _c19)
except Exception:
    pass
layout["8_21119_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/20_icon_7.44.png
try:
    _c20 = get_crop(20, 57, 61)
    canvas.paste(_c20, (182, 2), _c20)
except Exception:
    pass
layout["7.44"] = [182, 2, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/21_icon_Public_House_Los_Angeles_CA.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 490), _c21)
except Exception:
    pass
layout["Public_House_(Los_Angeles"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/22_icon_Home.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (0, 2804), _c22)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 52, 60)
    canvas.paste(_c23, (247, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [247, 2, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/24_icon_ter_for_Break_Into_Tech_nowl.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 490), _c24)
except Exception:
    pass
layout["ter_for_Break_Into_Tech_n"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/25_icon_7.44.png
try:
    _c25 = get_crop(25, 100, 98)
    canvas.paste(_c25, (42, 122), _c25)
except Exception:
    pass
layout["7.44"] = [42, 122, 142, 220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 48, 53)
    canvas.paste(_c26, (1320, 7), _c26)
except Exception:
    pass
layout["icon_26"] = [1320, 7, 1368, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/27_icon_8_4717_creator_followers.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 1678), _c27)
except Exception:
    pass
layout["8_4717_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/28_icon_7.44.png
try:
    _c28 = get_crop(28, 58, 62)
    canvas.paste(_c28, (115, 1), _c28)
except Exception:
    pass
layout["7.44"] = [115, 1, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 85, 58)
    canvas.paste(_c29, (1212, 4), _c29)
except Exception:
    pass
layout["icon_29"] = [1212, 4, 1297, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 44, 57)
    canvas.paste(_c30, (385, 6), _c30)
except Exception:
    pass
layout["icon_30"] = [385, 6, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/31_icon_Sun_Apr_28.png
try:
    _c31 = get_crop(31, 1344, 396)
    canvas.paste(_c31, (48, 1678), _c31)
except Exception:
    pass
layout["Sun,_Apr_28"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/32_icon_7.44.png
try:
    _c32 = get_crop(32, 90, 60)
    canvas.paste(_c32, (17, 3), _c32)
except Exception:
    pass
layout["7.44"] = [17, 3, 107, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/33_icon_Grief_Loss_Resiliency.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 2074), _c33)
except Exception:
    pass
layout["Grief;_Loss,_Resiliency"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/34_icon_icon_34.png
try:
    _c34 = get_crop(34, 41, 56)
    canvas.paste(_c34, (1272, 5), _c34)
except Exception:
    pass
layout["icon_34"] = [1272, 5, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/35_icon_Break_into_Tech_Social_Broxton_Brewery.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 490), _c35)
except Exception:
    pass
layout["Break_into_Tech_Social:_B"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/36_icon_Free.png
try:
    _c36 = get_crop(36, 130, 74)
    canvas.paste(_c36, (244, 1352), _c36)
except Exception:
    pass
layout["Free"] = [244, 1352, 374, 1426]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/37_icon_8_225_creator_followers.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 490), _c37)
except Exception:
    pass
layout["8_225_creator_followers"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/38_text_More_events_you_II_love.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 490), _c38)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/39_text_2000s_NITE.png
try:
    _c39 = get_crop(39, 202, 49)
    canvas.paste(_c39, (81, 2528), _c39)
except Exception:
    pass
layout["2000s_NITE"] = [81, 2528, 283, 2577]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/40_text_Fri_May_31.png
try:
    _c40 = get_crop(40, 184, 43)
    canvas.paste(_c40, (392, 2525), _c40)
except Exception:
    pass
layout["Fri,_May_31"] = [392, 2525, 576, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/41_text_9_30_PM_PDT.png
try:
    _c41 = get_crop(41, 1344, 346)
    canvas.paste(_c41, (48, 2470), _c41)
except Exception:
    pass
layout["9:30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/42_text_UNDERGROUND_X_200Os_NITE_Dance_Partyl.png
try:
    _c42 = get_crop(42, 1344, 346)
    canvas.paste(_c42, (48, 2470), _c42)
except Exception:
    pass
layout["UNDERGROUND_X_200Os_NITE_"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/43_text_MEt.png
try:
    _c43 = get_crop(43, 79, 67)
    canvas.paste(_c43, (205, 2598), _c43)
except Exception:
    pass
layout["MEt"] = [205, 2598, 284, 2665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/44_clickable_Favorites.png
try:
    _c44 = get_crop(44, 288, 156)
    canvas.paste(_c44, (576, 2804), _c44)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/45_clickable_Tickets.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (864, 2804), _c45)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/39adaf730c584c5582b89d1335e0c2cd/step_01_2024_4_23_19_42_39adaf730c584c5582b89d1335e0c2cd-3/46_clickable_More.png
try:
    _c46 = get_crop(46, 288, 156)
    canvas.paste(_c46, (1152, 2804), _c46)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
