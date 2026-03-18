# page_id: page_eventbrite_c36ef2ec281b4d40839b2bf8f321a58b_03
# screenshot: 2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5.png
# step_index: 3/8
# task: Open Eventbrite. Set the city to "Chicago". Select the "Fashion" category and view the recommended events. See the date of the first play and its venue.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw background and structural UI elements for the page
bg_color = (250, 250, 252)       # very light off-white background
status_bar_color = (200, 200, 200)  # gray status bar
header_color = (255, 255, 255)   # white app header area
divider_color = (226, 226, 230)  # subtle divider color
card_shadow = (235, 237, 240)    # light shadow for cards
card_fill = (255, 255, 255)      # card background
card_border = (230, 230, 234)    # subtle card border
nav_bg = (255, 255, 255)         # bottom nav background
nav_divider = (222, 222, 226)    # top divider for nav

W, H = canvas.size

# full background
draw.rectangle([0, 0, W, H], fill=bg_color)

# status bar (approx 56px)
status_h = 56
draw.rectangle([0, 0, W, status_h], fill=status_bar_color)

# app header / toolbar area (just the background and bottom divider)
header_top = status_h
header_h = 94
draw.rectangle([0, header_top, W, header_top + header_h], fill=header_color)
draw.line([24, header_top + header_h - 1, W - 24, header_top + header_h - 1], fill=divider_color, width=1)

# main content vertical padding start
content_x = 48
content_w = 1344

# Cards data based on detected positions/sizes (do not draw any icons/text inside)
cards = [
    (48, 490, 1344, 396),
    (48, 886, 1344, 396),
    (48, 1282, 1344, 396),
    (48, 1678, 1344, 396),
    (48, 2074, 1344, 396),
    (48, 2470, 1344, 346),
]

radius = 18

for (x, y, w, h) in cards:
    # shadow (offset)
    shadow_offset = 8
    sx0, sy0 = x + shadow_offset, y + shadow_offset
    sx1, sy1 = x + w + shadow_offset, y + h + shadow_offset
    draw.rounded_rectangle([sx0, sy0, sx1, sy1], radius=radius, fill=card_shadow)

    # main card
    x0, y0 = x, y
    x1, y1 = x + w, y + h
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=card_fill, outline=card_border, width=1)

    # subtle separator line under card
    sep_y = y1 + 22
    draw.line([x0 + 12, sep_y, x1 - 12, sep_y], fill=divider_color, width=1)

# A subtle large content banner background area behind the list title (do not draw text)
title_banner_y = 420
title_banner_h = 80
# keep it minimal (slightly transparent-looking via a very light tint)
draw.rectangle([content_x, title_banner_y, content_x + content_w, title_banner_y + title_banner_h], fill=bg_color)

# Floating location pill area background is auto-pasted; avoid drawing the pill itself.
# Draw only a faint area to hint layering under that (a soft shadow)
pill_shadow_x = 420
pill_shadow_y = 2580
pill_shadow_w = 500
pill_shadow_h = 140
draw.rounded_rectangle([pill_shadow_x + 6, pill_shadow_y + 8, pill_shadow_x + pill_shadow_w + 6, pill_shadow_y + pill_shadow_h + 8],
                       radius=40, fill=(240,240,243))

# Bottom navigation bar background and top divider
nav_h = 120
nav_y0 = H - nav_h
draw.rectangle([0, nav_y0, W, H], fill=nav_bg)
draw.line([24, nav_y0, W - 24, nav_y0], fill=nav_divider, width=1)

# subtle left & right padding indicators on nav (visual structure only)
pad_x = 72
draw.line([pad_x, nav_y0 + 10, pad_x, H - 10], fill=(255,255,255,0))
draw.line([W - pad_x, nav_y0 + 10, W - pad_x, H - 10], fill=(255,255,255,0))

# final subtle overall vignette edges to ground the UI (very light)
edge_strip = 20
draw.rectangle([0, 0, edge_strip, H], fill=bg_color)
draw.rectangle([W - edge_strip, 0, W, H], fill=bg_color)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/00_icon_Chicago.png
try:
    _c0 = get_crop(0, 388, 117)
    canvas.paste(_c0, (526, 2651), _c0)
except Exception:
    pass
layout["Chicago"] = [526, 2651, 914, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/01_icon_CyPo6.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1678), _c1)
except Exception:
    pass
layout["CyPo6"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/02_icon_ripg_-_LeaTG_Atans.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 2074), _c2)
except Exception:
    pass
layout["ripg_-_LeaTG_Atans"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/03_icon_Okstore.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1282), _c3)
except Exception:
    pass
layout["Okstore"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/04_icon_Q_Search_events.png
try:
    _c4 = get_crop(4, 1179, 144)
    canvas.paste(_c4, (195, 93), _c4)
except Exception:
    pass
layout["Q_Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/05_icon_Sat_Oct_19.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 490), _c5)
except Exception:
    pass
layout["Sat,_Oct_19"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/06_icon_Dovetail_Brewery.png
try:
    _c6 = get_crop(6, 144, 139)
    canvas.paste(_c6, (1140, 1935), _c6)
except Exception:
    pass
layout["Dovetail_Brewery"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 123)
    canvas.paste(_c7, (1140, 2347), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 1539), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1284, 1935), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1284, 2347), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/11_icon_Favorite_button.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1140, 1143), _c11)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/12_icon_Favorite_button.png
try:
    _c12 = get_crop(12, 144, 125)
    canvas.paste(_c12, (1140, 761), _c12)
except Exception:
    pass
layout["Favorite_button"] = [1140, 761, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/13_icon_7940_Wolcott_Ave_apt_2_Chicago_IL_USA.png
try:
    _c13 = get_crop(13, 1344, 396)
    canvas.paste(_c13, (48, 490), _c13)
except Exception:
    pass
layout["7940_$_Wolcott_Ave_apt_2,"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 125)
    canvas.paste(_c14, (1284, 761), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 761, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/15_icon_Joliet.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (288, 2804), _c15)
except Exception:
    pass
layout["Joliet"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/16_icon_Overflow_menu_button.png
try:
    _c16 = get_crop(16, 144, 139)
    canvas.paste(_c16, (1284, 1539), _c16)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/17_icon_5.12.png
try:
    _c17 = get_crop(17, 105, 100)
    canvas.paste(_c17, (40, 122), _c17)
except Exception:
    pass
layout["5.12"] = [40, 122, 145, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/18_icon_through_thc_chi.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (0, 2804), _c18)
except Exception:
    pass
layout["through_thc_chi"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/19_icon_5.12.png
try:
    _c19 = get_crop(19, 55, 60)
    canvas.paste(_c19, (183, 2), _c19)
except Exception:
    pass
layout["5.12"] = [183, 2, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/20_icon_ON.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 886), _c20)
except Exception:
    pass
layout["ON"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/21_icon_Overflow_menu_button.png
try:
    _c21 = get_crop(21, 144, 139)
    canvas.paste(_c21, (1284, 1143), _c21)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 60, 58)
    canvas.paste(_c22, (312, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [312, 3, 372, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/23_icon_49_creator_followers.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 886), _c23)
except Exception:
    pass
layout["49_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 50, 59)
    canvas.paste(_c24, (248, 2), _c24)
except Exception:
    pass
layout["icon_24"] = [248, 2, 298, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/25_icon_Indie_Bookstore_Day_at_Goblin_Market.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 1282), _c25)
except Exception:
    pass
layout["Indie_Bookstore_Day_at_Go"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 48, 53)
    canvas.paste(_c26, (1321, 7), _c26)
except Exception:
    pass
layout["icon_26"] = [1321, 7, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/27_icon_Planting_Seeds_bilingual.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 2074), _c27)
except Exception:
    pass
layout["Planting_Seeds_(bilingual"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 59, 58)
    canvas.paste(_c28, (1212, 4), _c28)
except Exception:
    pass
layout["icon_28"] = [1212, 4, 1271, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/29_icon_Q_Search_events.png
try:
    _c29 = get_crop(29, 44, 56)
    canvas.paste(_c29, (385, 6), _c29)
except Exception:
    pass
layout["Q_Search_events"] = [385, 6, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 41, 55)
    canvas.paste(_c30, (1272, 6), _c30)
except Exception:
    pass
layout["icon_30"] = [1272, 6, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/31_icon_73_creator_followers.png
try:
    _c31 = get_crop(31, 1344, 396)
    canvas.paste(_c31, (48, 1678), _c31)
except Exception:
    pass
layout["73_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/32_icon_5.12.png
try:
    _c32 = get_crop(32, 57, 60)
    canvas.paste(_c32, (116, 2), _c32)
except Exception:
    pass
layout["5.12"] = [116, 2, 173, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/33_icon_Self-Love_in_Nature_Releasing_Grief_thro.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 2074), _c33)
except Exception:
    pass
layout["Self-Love_in_Nature:_Rele"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/34_icon_Grief_R.png
try:
    _c34 = get_crop(34, 1344, 346)
    canvas.paste(_c34, (48, 2470), _c34)
except Exception:
    pass
layout["Grief_R"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/35_icon_6_00_PM_CDT.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 1678), _c35)
except Exception:
    pass
layout["6:00_PM_CDT"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/36_icon_Dovetail_Brewery.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 1678), _c36)
except Exception:
    pass
layout["Dovetail_Brewery"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/37_icon_Discover_Your_Path_To_Healing_With_Our_G.png
try:
    _c37 = get_crop(37, 1344, 346)
    canvas.paste(_c37, (48, 2470), _c37)
except Exception:
    pass
layout["Discover_Your_Path_To_Hea"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/38_text_5.12.png
try:
    _c38 = get_crop(38, 89, 43)
    canvas.paste(_c38, (22, 17), _c38)
except Exception:
    pass
layout["5.12"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/39_text_More_events_you_II_love.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 490), _c39)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/40_text_Tue_May_7.png
try:
    _c40 = get_crop(40, 191, 43)
    canvas.paste(_c40, (390, 2525), _c40)
except Exception:
    pass
layout["Tue,_May_7"] = [390, 2525, 581, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/41_text_6_00_PM_CDT.png
try:
    _c41 = get_crop(41, 1344, 346)
    canvas.paste(_c41, (48, 2470), _c41)
except Exception:
    pass
layout["6:00_PM_CDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/42_text_Joliet.png
try:
    _c42 = get_crop(42, 96, 38)
    canvas.paste(_c42, (390, 2723), _c42)
except Exception:
    pass
layout["Joliet"] = [390, 2723, 486, 2761]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/43_clickable_Favorites.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (576, 2804), _c43)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/44_clickable_Tickets.png
try:
    _c44 = get_crop(44, 288, 156)
    canvas.paste(_c44, (864, 2804), _c44)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c36ef2ec281b4d40839b2bf8f321a58b/step_03_2024_4_24_17_11_c36ef2ec281b4d40839b2bf8f321a58b-5/45_clickable_More.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (1152, 2804), _c45)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
