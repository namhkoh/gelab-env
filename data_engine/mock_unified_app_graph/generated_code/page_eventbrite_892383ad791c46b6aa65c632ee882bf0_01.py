# page_id: page_eventbrite_892383ad791c46b6aa65c632ee882bf0_01
# screenshot: 2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3.png
# step_index: 1/12
# task: Open Eventbrite. Search for online "Music" events happening next weekend.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided. Fonts: font_sm, font_md, font_lg, font_xl
# Draw background and structural UI elements only.

# Colors
bg_color = "#fbfbfc"         # overall page background (very light)
status_color = "#cfcfcf"     # status bar background
header_bg = "#ffffff"        # header/toolbar background
search_bar_bg = "#ffffff"    # (we won't draw inner search content)
card_bg = "#ffffff"          # card backgrounds
card_outline = "#eef0f3"     # subtle card outline / shadow
thumb_bg = "#eef3f5"         # thumbnail placeholder background
divider_color = "#e9e9ee"    # section separators
nav_bg = "#ffffff"           # bottom nav background
nav_top_border = "#e6e6ea"

W, H = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (W, H)], fill=bg_color)

# Status bar area (top ~56px)
status_h = 56
draw.rectangle([(0, 0), (W, status_h)], fill=status_color)
# thin divider under status bar
draw.line([(0, status_h), (W, status_h)], fill="#bfbfbf", width=1)

# Header / toolbar background area (beneath status bar)
header_y0 = status_h
header_y1 = 200
draw.rectangle([(0, header_y0), (W, header_y1)], fill=header_bg)
# subtle bottom divider
draw.line([(48, header_y1), (W-48, header_y1)], fill=divider_color, width=1)

# Draw subtle large rounded search bar background behind where the app will paste the search UI.
# Keep it minimal so as not to duplicate icons/text.
search_x0 = 48
search_x1 = W - 48
search_y0 = status_h + 16
search_y1 = search_y0 + 96
draw.rounded_rectangle([(search_x0, search_y0), (search_x1, search_y1)],
                       radius=48, fill=search_bar_bg, outline=card_outline, width=1)

# Content area: list cards background blocks (rounded rects)
# Use detected list-group positions (x=48, width=1344). We'll draw neutral card backgrounds and thumbnail placeholders.
card_x0 = 48
card_w = 1344
card_x1 = card_x0 + card_w

# Detected card top positions (approx). We'll create card blocks at these y values:
card_tops = [490, 886, 1282, 1678, 2074, 2470]  # include several rows
card_height = 160  # visual card height
card_radius = 12

for y in card_tops:
    cy0 = y - 10
    cy1 = cy0 + card_height
    # card shadow / outline (subtle)
    draw.rounded_rectangle([(card_x0+2, cy0+4), (card_x1+2, cy1+4)],
                           radius=card_radius, fill=card_outline)
    # main card background
    draw.rounded_rectangle([(card_x0, cy0), (card_x1, cy1)],
                           radius=card_radius, fill=card_bg)
    # thumbnail placeholder on the left (behind real image that will be pasted)
    thumb_x0 = card_x0 + 8
    thumb_x1 = thumb_x0 + 160
    thumb_y0 = cy0 + 10
    thumb_y1 = thumb_y0 + 140
    draw.rectangle([(thumb_x0, thumb_y0), (thumb_x1, thumb_y1)], fill=thumb_bg)
    # subtle separator line at bottom of card
    sep_y = cy1 + 10
    draw.line([(card_x0+8, sep_y), (card_x1-8, sep_y)], fill=divider_color, width=1)

# Full-width separators between groups / sections
# Example: top section title separator under header
draw.line([(48, 440), (W-48, 440)], fill=divider_color, width=1)

# Additional subtle horizontal separators aligned with content rhythm
extra_seps = [1050, 1430, 1810, 2190]
for sy in extra_seps:
    draw.line([(48, sy), (W-48, sy)], fill="#f2f3f6", width=1)

# Bottom navigation bar background and divider
nav_h = 156
nav_y0 = H - nav_h
draw.rectangle([(0, nav_y0), (W, H)], fill=nav_bg)
# top border line for nav
draw.line([(0, nav_y0), (W, nav_y0)], fill=nav_top_border, width=1)

# Small elevated white panel background for a possible floating filter pill area (we keep it subtle and behind)
# Place it slightly above nav but avoid overlapping detected exact pill area too heavily.
pill_w = 620
pill_h = 96
pill_x0 = (W - pill_w) // 2 - 60
pill_y0 = nav_y0 - 160
pill_x1 = pill_x0 + pill_w
pill_y1 = pill_y0 + pill_h
draw.rounded_rectangle([(pill_x0, pill_y0), (pill_x1, pill_y1)], radius=48,
                       fill="#ffffff", outline="#eceff2", width=1)

# Top-of-content faint large banner area (not duplicating posted images/text)
banner_h = 36
draw.rectangle([(48, 360), (W-48, 360+banner_h)], fill="#ffffff")
draw.line([(48, 360+banner_h), (W-48, 360+banner_h)], fill=divider_color, width=1)

# Final subtle vignette/shadow at very bottom to ground the nav (thin)
draw.line([(0, H-1), (W, H-1)], fill="#e8e8ea", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/00_icon_Online.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 490), _c0)
except Exception:
    pass
layout["Online"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/01_icon_Online.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 2074), _c1)
except Exception:
    pass
layout["Online"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/02_icon_Q_Search_events.png
try:
    _c2 = get_crop(2, 1179, 144)
    canvas.paste(_c2, (195, 93), _c2)
except Exception:
    pass
layout["Q_Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/03_icon_Or.png
try:
    _c3 = get_crop(3, 288, 156)
    canvas.paste(_c3, (288, 2804), _c3)
except Exception:
    pass
layout["Or,"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/04_icon_Understanding_Grief_and_Loss.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 1282), _c4)
except Exception:
    pass
layout["Understanding_Grief_and_L"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/05_icon_Loss.png
try:
    _c5 = get_crop(5, 144, 125)
    canvas.paste(_c5, (1140, 2345), _c5)
except Exception:
    pass
layout["Loss"] = [1140, 2345, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 125)
    canvas.paste(_c6, (1284, 2345), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2345, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/07_icon_Home.png
try:
    _c7 = get_crop(7, 288, 156)
    canvas.paste(_c7, (0, 2804), _c7)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/08_icon_5.22.png
try:
    _c8 = get_crop(8, 108, 103)
    canvas.paste(_c8, (38, 120), _c8)
except Exception:
    pass
layout["5.22"] = [38, 120, 146, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1284, 1539), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/10_icon_Favorite_button.png
try:
    _c10 = get_crop(10, 144, 139)
    canvas.paste(_c10, (1140, 1935), _c10)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1284, 1935), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 139)
    canvas.paste(_c12, (1284, 747), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/13_icon_5.22.png
try:
    _c13 = get_crop(13, 54, 61)
    canvas.paste(_c13, (184, 2), _c13)
except Exception:
    pass
layout["5.22"] = [184, 2, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 60, 59)
    canvas.paste(_c14, (312, 3), _c14)
except Exception:
    pass
layout["icon_14"] = [312, 3, 372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 139)
    canvas.paste(_c15, (1284, 1143), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 51, 59)
    canvas.paste(_c16, (248, 3), _c16)
except Exception:
    pass
layout["icon_16"] = [248, 3, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/17_icon_Working_with_Grief_and_Loss.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 490), _c17)
except Exception:
    pass
layout["Working_with_Grief_and_Lo"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/18_icon_Favorite_button.png
try:
    _c18 = get_crop(18, 144, 139)
    canvas.paste(_c18, (1140, 747), _c18)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 48, 54)
    canvas.paste(_c19, (1321, 7), _c19)
except Exception:
    pass
layout["icon_19"] = [1321, 7, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/20_icon_1253_creator_followers.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 886), _c20)
except Exception:
    pass
layout["1253_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/21_icon_Favorite_button.png
try:
    _c21 = get_crop(21, 144, 139)
    canvas.paste(_c21, (1140, 1143), _c21)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 92, 60)
    canvas.paste(_c22, (1212, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [1212, 3, 1304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/23_icon_Online_events.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (576, 2804), _c23)
except Exception:
    pass
layout["Online_events"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/24_icon_Favorite_button.png
try:
    _c24 = get_crop(24, 144, 139)
    canvas.paste(_c24, (1140, 1539), _c24)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/25_icon_5.22.png
try:
    _c25 = get_crop(25, 57, 60)
    canvas.paste(_c25, (116, 3), _c25)
except Exception:
    pass
layout["5.22"] = [116, 3, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/26_icon_Art_for_Grief_and_Loss.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 1678), _c26)
except Exception:
    pass
layout["Art_for_Grief_and_Loss"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/27_icon_Weeruy_se55.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 1678), _c27)
except Exception:
    pass
layout["Weeruy_se55"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/28_icon_Q_Search_events.png
try:
    _c28 = get_crop(28, 44, 56)
    canvas.paste(_c28, (385, 7), _c28)
except Exception:
    pass
layout["Q_Search_events"] = [385, 7, 429, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/29_icon_Loss.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (864, 2804), _c29)
except Exception:
    pass
layout["Loss"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/30_icon_Online_events.png
try:
    _c30 = get_crop(30, 586, 117)
    canvas.paste(_c30, (427, 2651), _c30)
except Exception:
    pass
layout["Online_events"] = [427, 2651, 1013, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/31_icon_Online.png
try:
    _c31 = get_crop(31, 112, 54)
    canvas.paste(_c31, (390, 703), _c31)
except Exception:
    pass
layout["Online"] = [390, 703, 502, 757]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/32_icon_S_00_AM_EDT.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 2074), _c32)
except Exception:
    pass
layout["S:00_AM_EDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/33_icon_Ur.png
try:
    _c33 = get_crop(33, 59, 59)
    canvas.paste(_c33, (388, 2641), _c33)
except Exception:
    pass
layout["Ur"] = [388, 2641, 447, 2700]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/34_icon_icon_34.png
try:
    _c34 = get_crop(34, 42, 57)
    canvas.paste(_c34, (1272, 5), _c34)
except Exception:
    pass
layout["icon_34"] = [1272, 5, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/35_text_5.22.png
try:
    _c35 = get_crop(35, 89, 43)
    canvas.paste(_c35, (22, 17), _c35)
except Exception:
    pass
layout["5.22"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/36_text_More_events_you_II_love.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 490), _c36)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/37_text_Sat_Oct_5.png
try:
    _c37 = get_crop(37, 177, 45)
    canvas.paste(_c37, (390, 2583), _c37)
except Exception:
    pass
layout["Sat,_Oct_5"] = [390, 2583, 567, 2628]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/38_text_5_00_AM_EDT.png
try:
    _c38 = get_crop(38, 1344, 346)
    canvas.paste(_c38, (48, 2470), _c38)
except Exception:
    pass
layout["5:00_AM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/39_text_Loss.png
try:
    _c39 = get_crop(39, 110, 57)
    canvas.paste(_c39, (1031, 2646), _c39)
except Exception:
    pass
layout["Loss"] = [1031, 2646, 1141, 2703]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/892383ad791c46b6aa65c632ee882bf0/step_01_2024_4_24_17_21_892383ad791c46b6aa65c632ee882bf0-3/40_clickable_More.png
try:
    _c40 = get_crop(40, 288, 156)
    canvas.paste(_c40, (1152, 2804), _c40)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
