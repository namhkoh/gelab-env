# page_id: page_eventbrite_80babd0c0d294f03bca3ebb1c46a8a8d_01
# screenshot: 2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3.png
# step_index: 1/8
# task: Open Eventbrite. Search Art event in New York. Select the second one. Record its location and time in Google Keep Notes. Follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (slightly warm white to match screenshot)
draw.rectangle((0, 0, 1440, 2960), fill=(250, 250, 251))

# Status bar area at top (~70px) - muted grey bar
status_h = 70
draw.rectangle((0, 0, 1440, status_h), fill=(189, 189, 189))

# Subtle top divider under status bar
draw.line((0, status_h, 1440, status_h), fill=(210, 210, 210), width=1)

# Header area (contains search bar region) - keep background same as page but add a faint divider
header_top = status_h
header_bottom = 230
draw.rectangle((0, header_top, 1440, header_bottom), fill=(250, 250, 251))
draw.line((48, header_bottom, 1392, header_bottom), fill=(235, 235, 236), width=1)

# Large section title area (left empty for pasted text) - add subtle spacing background
section_title_top = header_bottom + 20
section_title_bottom = section_title_top + 80
# keep it visually identical to page but add a tiny shadow line below
draw.rectangle((0, section_title_top, 1440, section_title_bottom), fill=(250, 250, 251))
draw.line((48, section_title_bottom, 1392, section_title_bottom), fill=(245, 245, 246), width=1)

# Event row card positions (from detected positions). We'll draw soft card backgrounds and thumbnail placeholders.
card_x1 = 48
card_x2 = card_x1 + 1344
card_height = 396
card_ys = [490, 886, 1282, 1678, 2074, 2470]

for y in card_ys:
    # subtle shadow/background behind card (slightly offset)
    shadow_box = (card_x1, y + 6, card_x2, y + card_height + 6)
    draw.rounded_rectangle(shadow_box, radius=14, fill=(245, 245, 246))
    # main card (white)
    card_box = (card_x1, y, card_x2, y + card_height)
    draw.rounded_rectangle(card_box, radius=12, fill=(255, 255, 255), outline=(235, 235, 236))
    # thin divider line at bottom of card
    draw.line((card_x1 + 16, y + card_height, card_x2 - 16, y + card_height), fill=(240, 240, 241), width=1)

    # left thumbnail background placeholder (rounded) - will be overlaid by actual thumbnails
    thumb_x1 = card_x1 + 16
    thumb_x2 = thumb_x1 + 180
    thumb_y1 = y + 20
    thumb_y2 = thumb_y1 + 180
    draw.rounded_rectangle((thumb_x1, thumb_y1, thumb_x2, thumb_y2), radius=10, fill=(236, 236, 240))

    # small subtle vertical separator between thumb and content area
    sep_x = thumb_x2 + 20
    draw.line((sep_x, y + 18, sep_x, y + card_height - 18), fill=(248, 248, 249), width=1)

# Additional subtle separators across the main content area between groups
for sep_y in [card_ys[0] - 20, card_ys[2] - 20, card_ys[4] - 20]:
    draw.line((48, sep_y, 1392, sep_y), fill=(245, 245, 246), width=1)

# Floating location pill area on lower content (don't draw the pill itself, but add minimal background hint)
# We will only add a faint shadow under that area so pasted pill appears elevated
pill_shadow_top = 2600
pill_shadow_left = 420
pill_shadow_right = pill_shadow_left + 500
pill_shadow_bottom = pill_shadow_top + 70
draw.rounded_rectangle((pill_shadow_left, pill_shadow_top + 6, pill_shadow_right, pill_shadow_bottom + 6),
                       radius=40, fill=(246, 246, 246))

# Bottom navigation bar background and top divider
nav_h = 156
nav_top = 2960 - nav_h
draw.rectangle((0, nav_top, 1440, 2960), fill=(255, 255, 255))
draw.line((0, nav_top, 1440, nav_top), fill=(230, 230, 230), width=1)

# Subtle indicator slot for home tab (background hint only; actual icons will be pasted)
home_indicator_w = 128
home_indicator_h = 6
home_indicator_x = 48 + (home_indicator_w // 2)
home_indicator_y = nav_top + 10
draw.rectangle((48, home_indicator_y, 48 + home_indicator_w, home_indicator_y + home_indicator_h), fill=(255, 143, 75))

# final faint overall vignette/border to ground the layout
draw.line((0, 0, 0, 2960), fill=(245, 245, 245))
draw.line((1439, 0, 1439, 2960), fill=(245, 245, 245))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/00_icon_iORk.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["iORk"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/01_icon_ZDRTTZY.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 490), _c1)
except Exception:
    pass
layout["ZDRTTZY"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/02_icon_95_HEEEYIMI_UESK_EEudooz.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 886), _c2)
except Exception:
    pass
layout["95_HEEEYIMI_UESK_EEudooz"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/03_icon_New_York.png
try:
    _c3 = get_crop(3, 405, 117)
    canvas.paste(_c3, (518, 2651), _c3)
except Exception:
    pass
layout["New_York"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/04_icon_Search_events.png
try:
    _c4 = get_crop(4, 1179, 144)
    canvas.paste(_c4, (195, 93), _c4)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/05_icon_DL_NO_COVER_ALL_NIGHT.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 490), _c5)
except Exception:
    pass
layout["DL_(NO_COVER_ALL_NIGHT)"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/06_icon_8_8609_creator_followers.png
try:
    _c6 = get_crop(6, 1344, 396)
    canvas.paste(_c6, (48, 1282), _c6)
except Exception:
    pass
layout["8_8609_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 123)
    canvas.paste(_c7, (1140, 1159), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/08_icon_The_DL.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 1935), _c8)
except Exception:
    pass
layout["The_DL"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/09_icon_Favorite_button.png
try:
    _c9 = get_crop(9, 144, 123)
    canvas.paste(_c9, (1140, 763), _c9)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/10_icon_The_DL.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 2347), _c10)
except Exception:
    pass
layout["The_DL"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 123)
    canvas.paste(_c11, (1284, 1159), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/12_icon_The_DL.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1284, 2347), _c12)
except Exception:
    pass
layout["The_DL"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 139)
    canvas.paste(_c13, (1284, 1539), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/14_icon_9.25.png
try:
    _c14 = get_crop(14, 53, 58)
    canvas.paste(_c14, (183, 3), _c14)
except Exception:
    pass
layout["9.25"] = [183, 3, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 123)
    canvas.paste(_c15, (1284, 763), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/16_icon_Favorite_button.png
try:
    _c16 = get_crop(16, 144, 139)
    canvas.paste(_c16, (1140, 1539), _c16)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/17_icon_INDIE.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 1282), _c17)
except Exception:
    pass
layout["INDIE"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/18_icon_The_DL.png
try:
    _c18 = get_crop(18, 144, 139)
    canvas.paste(_c18, (1284, 1935), _c18)
except Exception:
    pass
layout["The_DL"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 56, 56)
    canvas.paste(_c19, (247, 5), _c19)
except Exception:
    pass
layout["icon_19"] = [247, 5, 303, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/20_icon_The_DL.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (288, 2804), _c20)
except Exception:
    pass
layout["The_DL"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/21_icon_DilaIcTt.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (0, 2804), _c21)
except Exception:
    pass
layout["DilaIcTt"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/22_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c22 = get_crop(22, 1344, 396)
    canvas.paste(_c22, (48, 886), _c22)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/23_icon_free_Grief_and_Loss_Healing_Circle.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 1678), _c23)
except Exception:
    pass
layout["(free)_Grief_and_Loss_Hea"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 48, 52)
    canvas.paste(_c24, (1320, 7), _c24)
except Exception:
    pass
layout["icon_24"] = [1320, 7, 1368, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 73, 59)
    canvas.paste(_c25, (1211, 4), _c25)
except Exception:
    pass
layout["icon_25"] = [1211, 4, 1284, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/26_icon_9.25.png
try:
    _c26 = get_crop(26, 97, 105)
    canvas.paste(_c26, (44, 118), _c26)
except Exception:
    pass
layout["9.25"] = [44, 118, 141, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 62, 58)
    canvas.paste(_c27, (311, 5), _c27)
except Exception:
    pass
layout["icon_27"] = [311, 5, 373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 48, 56)
    canvas.paste(_c28, (383, 7), _c28)
except Exception:
    pass
layout["icon_28"] = [383, 7, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/29_icon_Fireworks_July_Ath_Rooftop_Party.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 2074), _c29)
except Exception:
    pass
layout["Fireworks_July_Ath_Roofto"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/30_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 2074), _c30)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 42, 57)
    canvas.paste(_c31, (1272, 5), _c31)
except Exception:
    pass
layout["icon_31"] = [1272, 5, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/32_icon_free_Grief_and_Loss_Healing_Circle.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 1678), _c32)
except Exception:
    pass
layout["(free)_Grief_and_Loss_Hea"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/33_icon_Free.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 1678), _c33)
except Exception:
    pass
layout["Free"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/34_icon_Free.png
try:
    _c34 = get_crop(34, 130, 75)
    canvas.paste(_c34, (244, 1749), _c34)
except Exception:
    pass
layout["Free"] = [244, 1749, 374, 1824]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/35_text_9.25.png
try:
    _c35 = get_crop(35, 94, 43)
    canvas.paste(_c35, (20, 17), _c35)
except Exception:
    pass
layout["9.25"] = [20, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/36_text_More_events_you_II_love.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 490), _c36)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/37_text_Fri_Mar_22.png
try:
    _c37 = get_crop(37, 186, 41)
    canvas.paste(_c37, (392, 2526), _c37)
except Exception:
    pass
layout["Fri,_Mar_22"] = [392, 2526, 578, 2567]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/38_text_1O_00_PM_EDT.png
try:
    _c38 = get_crop(38, 1344, 346)
    canvas.paste(_c38, (48, 2470), _c38)
except Exception:
    pass
layout["1O:00_PM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/39_text_OFFICIAL_PARTY.png
try:
    _c39 = get_crop(39, 1344, 346)
    canvas.paste(_c39, (48, 2470), _c39)
except Exception:
    pass
layout["OFFICIAL_PARTY"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/40_text_THEDL_ROOFTOP_Every.png
try:
    _c40 = get_crop(40, 1344, 346)
    canvas.paste(_c40, (48, 2470), _c40)
except Exception:
    pass
layout["THEDL_ROOFTOP_(Every"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/41_text_DilaIcTt.png
try:
    _c41 = get_crop(41, 65, 18)
    canvas.paste(_c41, (130, 2737), _c41)
except Exception:
    pass
layout["DilaIcTt"] = [130, 2737, 195, 2755]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/42_text_Jupla.png
try:
    _c42 = get_crop(42, 46, 15)
    canvas.paste(_c42, (211, 2737), _c42)
except Exception:
    pass
layout["Jupla"] = [211, 2737, 257, 2752]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/43_text_The_DL.png
try:
    _c43 = get_crop(43, 115, 38)
    canvas.paste(_c43, (394, 2723), _c43)
except Exception:
    pass
layout["The_DL"] = [394, 2723, 509, 2761]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/44_clickable_Favorites.png
try:
    _c44 = get_crop(44, 288, 156)
    canvas.paste(_c44, (576, 2804), _c44)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/45_clickable_Tickets.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (864, 2804), _c45)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/80babd0c0d294f03bca3ebb1c46a8a8d/step_01_2024_3_20_17_24_80babd0c0d294f03bca3ebb1c46a8a8d-3/46_clickable_More.png
try:
    _c46 = get_crop(46, 288, 156)
    canvas.paste(_c46, (1152, 2804), _c46)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
