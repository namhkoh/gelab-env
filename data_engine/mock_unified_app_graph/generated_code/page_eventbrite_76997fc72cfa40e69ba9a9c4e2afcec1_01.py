# page_id: page_eventbrite_76997fc72cfa40e69ba9a9c4e2afcec1_01
# screenshot: 2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3.png
# step_index: 1/3
# task: Open Eventbrite. Open favorite tab and remove the second event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structure drawing for Eventbrite-like mobile UI
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Overall canvas fill (very light off-white / subtle warm white)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FBFBFE")

# Status bar (top ~60px) - light grey bar to match screenshot status area
status_h = 60
draw.rectangle([(0, 0), (1440, status_h)], fill="#CFCFCF")
# thin bottom divider line under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#BFBFBF", width=1)

# App header background area below status bar (contains search field area that will be pasted on top)
header_top = status_h
header_h = 140
draw.rectangle([(0, header_top), (1440, header_top + header_h)], fill="#FFFFFF")
# subtle bottom shadow / divider under header
draw.line([(0, header_top + header_h), (1440, header_top + header_h)], fill="#E9E6EF", width=1)

# List of event card bounding boxes (as rounded white cards) based on detected positions.
cards = [
    # (x, y, width, height)
    (48, 490, 1344, 396),
    (48, 886, 1344, 396),
    (48, 1282, 1344, 396),
    (48, 1678, 1344, 396),
    (48, 2074, 1344, 396),
    (48, 2470, 1344, 346)  # last card slightly shorter
]

# Draw each card with a subtle shadow and border
for (x, y, w, h) in cards:
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    # shadow (offset down-right)
    shadow_offset = 6
    shadow_bbox = [x1, y1 + shadow_offset, x2, y2 + shadow_offset]
    draw.rounded_rectangle(shadow_bbox, radius=14, fill="#F2F2F4")

    # card background (white)
    card_bbox = [x1, y1, x2, y2]
    draw.rounded_rectangle(card_bbox, radius=12, fill="#FFFFFF", outline="#ECE9F2", width=1)

    # subtle separator line under each card (spanning content width)
    sep_y = y2 + 8
    draw.line([(x1 + 12, sep_y), (x2 - 12, sep_y)], fill="#F0EDF4", width=1)

# Content area background for the main feed (a very faint tint to separate from pure white cards)
feed_top = header_top + header_h + 28
feed_bottom = cards[-1][1] + cards[-1][3] + 48
draw.rectangle([(0, feed_top), (1440, feed_bottom)], fill="#FBF8FC")

# Bottom navigation bar background (approx 156px high)
nav_top = 2804
nav_h = 156
draw.rectangle([(0, nav_top), (1440, nav_top + nav_h)], fill="#FFFFFF")
# top divider for nav bar
draw.line([(0, nav_top), (1440, nav_top)], fill="#E7E4EB", width=1)

# Floating location pill background area placeholder (do not draw the content—only a soft backdrop)
# Place it low on the screen but ensure not to duplicate detected element exact visuals; draw only a soft blur-like rounded rect
pill_x, pill_y, pill_w, pill_h = 420, 2700, 600, 110
draw.rounded_rectangle(
    [pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
    radius=36,
    fill="#FFFFFF",
    outline="#EEF0F4",
    width=1
)
# subtle shadow for the pill
draw.rounded_rectangle(
    [pill_x, pill_y + 6, pill_x + pill_w, pill_y + pill_h + 6],
    radius=36,
    fill="#F6F6F8"
)

# Top-left app accent area (background behind logo / search area) - subtle pale orange circle accent (background element only)
accent_cx, accent_cy, accent_r = 84, header_top + 64, 34
draw.ellipse([(accent_cx - accent_r, accent_cy - accent_r), (accent_cx + accent_r, accent_cy + accent_r)], fill="#FFF3EE")

# Final subtle overall vignette / edge line at very bottom to ground the page
draw.line([(0, 2959), (1440, 2959)], fill="#EDECF0", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/00_icon_Free.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["Free"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/01_icon_NDIE.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1678), _c1)
except Exception:
    pass
layout["NDIE"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/02_icon_Los_Angeles.png
try:
    _c2 = get_crop(2, 456, 117)
    canvas.paste(_c2, (492, 2651), _c2)
except Exception:
    pass
layout["Los_Angeles"] = [492, 2651, 948, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/03_icon_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/04_icon_REoPUNKSFRE.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 886), _c4)
except Exception:
    pass
layout["REoPUNKSFRE"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/05_icon_NDIE_DANCEPA.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1282), _c5)
except Exception:
    pass
layout["NDIE_DANCEPA"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 123)
    canvas.paste(_c6, (1140, 763), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/07_icon_Afliccion_Perdida_y.png
try:
    _c7 = get_crop(7, 144, 139)
    canvas.paste(_c7, (1140, 1935), _c7)
except Exception:
    pass
layout["Afliccion,_Perdida_y"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 123)
    canvas.paste(_c8, (1284, 1159), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/09_icon_Favorite_button.png
try:
    _c9 = get_crop(9, 144, 123)
    canvas.paste(_c9, (1140, 1159), _c9)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/10_icon_Afliccion_Perdida_y.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 2347), _c10)
except Exception:
    pass
layout["Afliccion,_Perdida_y"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 123)
    canvas.paste(_c11, (1284, 2347), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 139)
    canvas.paste(_c12, (1284, 1539), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 123)
    canvas.paste(_c13, (1284, 763), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/14_icon_Favorite_button.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1140, 1539), _c14)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/15_icon_8_60_creator_followers.png
try:
    _c15 = get_crop(15, 1344, 396)
    canvas.paste(_c15, (48, 1282), _c15)
except Exception:
    pass
layout["8_60_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/16_icon_Overflow_menu_button.png
try:
    _c16 = get_crop(16, 144, 139)
    canvas.paste(_c16, (1284, 1935), _c16)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/17_icon_The_Gr.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (288, 2804), _c17)
except Exception:
    pass
layout["The_Gr"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/18_icon_Public_House_Los_Angeles_CA.png
try:
    _c18 = get_crop(18, 1344, 396)
    canvas.paste(_c18, (48, 490), _c18)
except Exception:
    pass
layout["Public_House_(Los_Angeles"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/19_icon_8_21126_creator_followers.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 886), _c19)
except Exception:
    pass
layout["8_21126_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 60, 59)
    canvas.paste(_c20, (312, 3), _c20)
except Exception:
    pass
layout["icon_20"] = [312, 3, 372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/21_icon_Home.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (0, 2804), _c21)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/22_icon_5.36.png
try:
    _c22 = get_crop(22, 57, 61)
    canvas.paste(_c22, (182, 2), _c22)
except Exception:
    pass
layout["5.36"] = [182, 2, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/23_icon_5.36.png
try:
    _c23 = get_crop(23, 102, 98)
    canvas.paste(_c23, (41, 122), _c23)
except Exception:
    pass
layout["5.36"] = [41, 122, 143, 220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 52, 60)
    canvas.paste(_c24, (247, 2), _c24)
except Exception:
    pass
layout["icon_24"] = [247, 2, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/25_icon_8_4722_creator_followers.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 1678), _c25)
except Exception:
    pass
layout["8_4722_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 48, 52)
    canvas.paste(_c26, (1320, 7), _c26)
except Exception:
    pass
layout["icon_26"] = [1320, 7, 1368, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/27_icon_ter_for_Break_Into_Tech_nowl.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 490), _c27)
except Exception:
    pass
layout["ter_for_Break_Into_Tech_n"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 84, 58)
    canvas.paste(_c28, (1212, 4), _c28)
except Exception:
    pass
layout["icon_28"] = [1212, 4, 1296, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/29_icon_5.36.png
try:
    _c29 = get_crop(29, 59, 62)
    canvas.paste(_c29, (115, 1), _c29)
except Exception:
    pass
layout["5.36"] = [115, 1, 174, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 44, 57)
    canvas.paste(_c30, (385, 6), _c30)
except Exception:
    pass
layout["icon_30"] = [385, 6, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/31_icon_Grief_Loss_Resiliency.png
try:
    _c31 = get_crop(31, 1344, 396)
    canvas.paste(_c31, (48, 2074), _c31)
except Exception:
    pass
layout["Grief;_Loss,_Resiliency"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/32_icon_Sun_Apr_28.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 1678), _c32)
except Exception:
    pass
layout["Sun,_Apr_28"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/33_icon_icon_33.png
try:
    _c33 = get_crop(33, 41, 55)
    canvas.paste(_c33, (1272, 6), _c33)
except Exception:
    pass
layout["icon_33"] = [1272, 6, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/34_icon_Break_into_Tech_Social_Broxton_Brewery.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 490), _c34)
except Exception:
    pass
layout["Break_into_Tech_Social:_B"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/35_icon_Free.png
try:
    _c35 = get_crop(35, 130, 74)
    canvas.paste(_c35, (244, 1352), _c35)
except Exception:
    pass
layout["Free"] = [244, 1352, 374, 1426]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/36_icon_The_Virgil.png
try:
    _c36 = get_crop(36, 158, 51)
    canvas.paste(_c36, (390, 1131), _c36)
except Exception:
    pass
layout["The_Virgil"] = [390, 1131, 548, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/37_text_5.36.png
try:
    _c37 = get_crop(37, 91, 45)
    canvas.paste(_c37, (20, 15), _c37)
except Exception:
    pass
layout["5.36"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/38_text_More_events_you_II_love.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 490), _c38)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/39_text_2000s_NITE.png
try:
    _c39 = get_crop(39, 202, 49)
    canvas.paste(_c39, (81, 2528), _c39)
except Exception:
    pass
layout["2000s_NITE"] = [81, 2528, 283, 2577]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/40_text_Fri_May_31.png
try:
    _c40 = get_crop(40, 184, 43)
    canvas.paste(_c40, (392, 2525), _c40)
except Exception:
    pass
layout["Fri,_May_31"] = [392, 2525, 576, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/41_text_9_30_PM_PDT.png
try:
    _c41 = get_crop(41, 1344, 346)
    canvas.paste(_c41, (48, 2470), _c41)
except Exception:
    pass
layout["9:30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/42_text_UNDERGROUND_X_200Os_NITE_Dance_Partyl.png
try:
    _c42 = get_crop(42, 1344, 346)
    canvas.paste(_c42, (48, 2470), _c42)
except Exception:
    pass
layout["UNDERGROUND_X_200Os_NITE_"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/43_text_MEt.png
try:
    _c43 = get_crop(43, 79, 67)
    canvas.paste(_c43, (205, 2598), _c43)
except Exception:
    pass
layout["MEt"] = [205, 2598, 284, 2665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/44_clickable_Favorites.png
try:
    _c44 = get_crop(44, 288, 156)
    canvas.paste(_c44, (576, 2804), _c44)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/45_clickable_Tickets.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (864, 2804), _c45)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_01_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-3/46_clickable_More.png
try:
    _c46 = get_crop(46, 288, 156)
    canvas.paste(_c46, (1152, 2804), _c46)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
