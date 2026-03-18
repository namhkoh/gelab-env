# page_id: page_eventbrite_97cc452004d64d67bb4c5b51ffb8f6d4_01
# screenshot: 2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3.png
# step_index: 1/7
# task: Open Eventbrite. Search Business event. Select the first one that is not promoted. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the described mobile page.
# Available variables: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
bg_color = (250, 250, 252)        # very light off-white page background
status_bar_color = (210, 210, 210)  # light grey status bar
divider_color = (235, 235, 239)    # subtle divider
card_shadow = (235, 235, 238)      # subtle shadow / lift
card_border = (245, 245, 247)      # very light border for cards
pill_shadow = (220, 220, 224)
bottom_bar_color = (255, 255, 255)

# Fill overall background
draw.rectangle([(0, 0), (W, H)], fill=bg_color)

# Status bar area (approx ~50-80px tall)
status_h = 80
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# Top header / toolbar area (below status bar). Keep it light; search widget will be pasted on top.
header_top = status_h
header_bottom = 260
draw.rectangle([(0, header_top), (W, header_bottom)], fill=bg_color)
# subtle bottom divider under header
draw.line([(24, header_bottom), (W-24, header_bottom)], fill=divider_color, width=1)

# Main content area: list card backgrounds
card_x = 48
card_w = 1344
card_h = 396
card_radius = 18

# Y positions of detected list item blocks (from provided detected elements)
card_ys = [490, 886, 1282, 1678, 2074, 2470]

for y in card_ys:
    x1 = card_x
    y1 = y
    x2 = card_x + card_w
    y2 = y + card_h

    # shadow (light)
    shadow_offset = 6
    draw.rounded_rectangle(
        [(x1, y1 + shadow_offset), (x2, y2 + shadow_offset)],
        radius=card_radius,
        fill=card_shadow,
        outline=None
    )

    # card background (white-ish)
    draw.rounded_rectangle(
        [(x1, y1), (x2, y2)],
        radius=card_radius,
        fill=(255, 255, 255),
        outline=card_border,
        width=1
    )

    # subtle divider between cards (space below each card)
    sep_y = y2 + 12
    draw.line([(x1 + 8, sep_y), (x2 - 8, sep_y)], fill=divider_color, width=1)

# Small separators between major sections (above the first card area)
# a faint line right above the first card group to define the "More events you'll love" header area
first_card_top = card_ys[0]
draw.line([(24, first_card_top - 26), (W-24, first_card_top - 26)], fill=divider_color, width=1)

# Floating location pill/background (behind detected "New York" label)
# Determine approximate center from detection: detected text centered roughly at x ~ 720, y ~ 2651
pill_center_x = 720
pill_center_y = 2651
pill_w = 540
pill_h = 100
pill_x1 = pill_center_x - pill_w // 2
pill_y1 = pill_center_y - pill_h // 2
pill_x2 = pill_center_x + pill_w // 2
pill_y2 = pill_center_y + pill_h // 2
pill_radius = 60

# pill shadow
draw.rounded_rectangle([(pill_x1+4, pill_y1+6), (pill_x2+4, pill_y2+6)], radius=pill_radius, fill=pill_shadow)
# pill background (white)
draw.rounded_rectangle([(pill_x1, pill_y1), (pill_x2, pill_y2)], radius=pill_radius, fill=(255,255,255), outline=card_border)

# Bottom navigation bar background
bottom_h = 120
bottom_top = H - bottom_h
draw.rectangle([(0, bottom_top), (W, H)], fill=bottom_bar_color)
# top divider of bottom nav
draw.line([(24, bottom_top), (W-24, bottom_top)], fill=divider_color, width=1)

# Additional subtle structural accents:
# Left page margin guide (very faint) and right margin guide to frame the content region
margin_x_left = 24
margin_x_right = W - 24
draw.line([(margin_x_left, header_bottom + 8), (margin_x_left, H - bottom_h - 8)], fill=divider_color, width=1)
draw.line([(margin_x_right, header_bottom + 8), (margin_x_right, H - bottom_h - 8)], fill=divider_color, width=1)

# Small topmost fine divider under status bar (to separate indicators from header)
draw.line([(0, status_h), (W, status_h)], fill=divider_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/00_icon_iORk.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["iORk"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/01_icon_ZDRTTZY.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 490), _c1)
except Exception:
    pass
layout["ZDRTTZY"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/02_icon_95_HEEEYIMI_UESK_EEudooz.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 886), _c2)
except Exception:
    pass
layout["95_HEEEYIMI_UESK_EEudooz"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/03_icon_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/04_icon_DL_NO_COVER_ALL_NIGHT.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 490), _c4)
except Exception:
    pass
layout["DL_(NO_COVER_ALL_NIGHT)"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/05_icon_free_Grief_and_Loss_Healing_Circle.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1282), _c5)
except Exception:
    pass
layout["(free)_Grief_and_Loss_Hea"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/06_icon_The_DL.png
try:
    _c6 = get_crop(6, 144, 123)
    canvas.paste(_c6, (1140, 1951), _c6)
except Exception:
    pass
layout["The_DL"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/07_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c7 = get_crop(7, 1344, 396)
    canvas.paste(_c7, (48, 1678), _c7)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 1539), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/09_icon_Favorite_button.png
try:
    _c9 = get_crop(9, 144, 123)
    canvas.paste(_c9, (1140, 763), _c9)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/10_icon_The_DL.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 2347), _c10)
except Exception:
    pass
layout["The_DL"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 52, 57)
    canvas.paste(_c11, (183, 4), _c11)
except Exception:
    pass
layout["icon_11"] = [183, 4, 235, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/12_icon_The_DL.png
try:
    _c12 = get_crop(12, 288, 156)
    canvas.paste(_c12, (288, 2804), _c12)
except Exception:
    pass
layout["The_DL"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/13_icon_The_DL.png
try:
    _c13 = get_crop(13, 144, 123)
    canvas.paste(_c13, (1284, 1951), _c13)
except Exception:
    pass
layout["The_DL"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1284, 1539), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 123)
    canvas.paste(_c15, (1284, 1159), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/16_icon_The_DL.png
try:
    _c16 = get_crop(16, 144, 123)
    canvas.paste(_c16, (1284, 2347), _c16)
except Exception:
    pass
layout["The_DL"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/17_icon_dtLaIct.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 1678), _c17)
except Exception:
    pass
layout["dtLaIct"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 55, 56)
    canvas.paste(_c18, (247, 5), _c18)
except Exception:
    pass
layout["icon_18"] = [247, 5, 302, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/19_icon_Favorite_button.png
try:
    _c19 = get_crop(19, 144, 123)
    canvas.paste(_c19, (1140, 1159), _c19)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/20_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 886), _c20)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/21_icon_Overflow_menu_button.png
try:
    _c21 = get_crop(21, 144, 123)
    canvas.paste(_c21, (1284, 763), _c21)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/22_icon_Ary.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (0, 2804), _c22)
except Exception:
    pass
layout["Ary"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 47, 52)
    canvas.paste(_c23, (1321, 7), _c23)
except Exception:
    pass
layout["icon_23"] = [1321, 7, 1368, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/24_icon_New_York.png
try:
    _c24 = get_crop(24, 405, 117)
    canvas.paste(_c24, (518, 2651), _c24)
except Exception:
    pass
layout["New_York"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/25_icon_9.39.png
try:
    _c25 = get_crop(25, 92, 100)
    canvas.paste(_c25, (47, 120), _c25)
except Exception:
    pass
layout["9.39"] = [47, 120, 139, 220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 63, 59)
    canvas.paste(_c26, (1211, 4), _c26)
except Exception:
    pass
layout["icon_26"] = [1211, 4, 1274, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 62, 58)
    canvas.paste(_c27, (311, 5), _c27)
except Exception:
    pass
layout["icon_27"] = [311, 5, 373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 48, 56)
    canvas.paste(_c28, (383, 7), _c28)
except Exception:
    pass
layout["icon_28"] = [383, 7, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 42, 56)
    canvas.paste(_c29, (1272, 5), _c29)
except Exception:
    pass
layout["icon_29"] = [1272, 5, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/30_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 2074), _c30)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/31_icon_TUmU_5i0.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (576, 2804), _c31)
except Exception:
    pass
layout["TUmU'5i0"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/32_icon_Fireworks_July_Ath_Rooftop_Party.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 2074), _c32)
except Exception:
    pass
layout["Fireworks_July_Ath_Roofto"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/33_icon_icon_33.png
try:
    _c33 = get_crop(33, 31, 48)
    canvas.paste(_c33, (913, 2687), _c33)
except Exception:
    pass
layout["icon_33"] = [913, 2687, 944, 2735]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/34_text_9.39.png
try:
    _c34 = get_crop(34, 94, 45)
    canvas.paste(_c34, (17, 15), _c34)
except Exception:
    pass
layout["9.39"] = [17, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/35_text_More_events_you_II_love.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 490), _c35)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/36_text_Sun_Jun_23.png
try:
    _c36 = get_crop(36, 205, 49)
    canvas.paste(_c36, (388, 2554), _c36)
except Exception:
    pass
layout["Sun,_Jun_23"] = [388, 2554, 593, 2603]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/37_text_3_00_PM_EDT.png
try:
    _c37 = get_crop(37, 1344, 346)
    canvas.paste(_c37, (48, 2470), _c37)
except Exception:
    pass
layout["3:00_PM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/38_text_The_DL_Rooftop.png
try:
    _c38 = get_crop(38, 144, 123)
    canvas.paste(_c38, (1140, 2347), _c38)
except Exception:
    pass
layout["The_DL_Rooftop"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/39_text_Ary.png
try:
    _c39 = get_crop(39, 1344, 346)
    canvas.paste(_c39, (48, 2470), _c39)
except Exception:
    pass
layout["Ary"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/40_text_The_DL.png
try:
    _c40 = get_crop(40, 115, 38)
    canvas.paste(_c40, (394, 2693), _c40)
except Exception:
    pass
layout["The_DL"] = [394, 2693, 509, 2731]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/41_text_TUmU_5i0.png
try:
    _c41 = get_crop(41, 405, 117)
    canvas.paste(_c41, (518, 2651), _c41)
except Exception:
    pass
layout["TUmU'5i0"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/42_clickable_Tickets.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (864, 2804), _c42)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_01_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-3/43_clickable_More.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (1152, 2804), _c43)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
