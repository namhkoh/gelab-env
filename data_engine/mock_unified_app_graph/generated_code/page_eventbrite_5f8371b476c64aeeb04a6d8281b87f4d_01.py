# page_id: page_eventbrite_5f8371b476c64aeeb04a6d8281b87f4d_01
# screenshot: 2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3.png
# step_index: 1/7
# task: Open Eventbrite. Search Science & Tech event. Select the first one that is not promoted. If it is free, add it to Favorites. If it is not free, record its price in Google Keep Notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the provided canvas.
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
status_bar_color = (210, 210, 210)     # light grey status bar
status_divider = (185, 185, 185)       # divider under status bar
toolbar_shadow = (235, 235, 235)       # subtle toolbar bottom shadow
card_bg = (250, 250, 252)              # very light off-white card background
card_border = (230, 230, 235)          # card border / outline
thumb_bg = (45, 28, 64)                # dark purple thumbnail background (placeholder)
separator = (240, 240, 242)            # list separator
nav_bar_top = (230, 230, 235)          # top border for bottom nav
nav_bg = (255, 255, 255)               # nav bar background
floating_shadow = (220, 220, 225)      # shadow under floating pill

W, H = canvas.size

# 1) Status bar area (top ~84px)
status_h = 84
draw.rectangle([0, 0, W, status_h], fill=status_bar_color)
# small divider under status bar
draw.line([(0, status_h), (W, status_h)], fill=status_divider, width=1)

# 2) Header / toolbar area shadow (search bar region sits under this; don't draw the search control)
toolbar_top = status_h
toolbar_bottom = 200
# Draw subtle bottom shadow to separate header from content
draw.rectangle([0, toolbar_top, W, toolbar_bottom], fill=(255,255,255))
draw.line([(0, toolbar_bottom), (W, toolbar_bottom)], fill=toolbar_shadow, width=2)

# 3) Section card backgrounds (rounded rectangles for each event row)
# Detected row positions and sizes (use these to place card backgrounds)
rows = [
    (48, 490, 48+1344, 490+396),
    (48, 886, 48+1344, 886+396),
    (48, 1282, 48+1344, 1282+396),
    (48, 1678, 48+1344, 1678+396),
    (48, 2074, 48+1344, 2074+396),
    (48, 2470, 48+1344, 2470+396),
]
card_radius = 18

for (x1, y1, x2, y2) in rows:
    # subtle shadow under card (a thin soft line)
    shadow_y = y2 + 6
    draw.rectangle([x1+6, shadow_y, x2-6, shadow_y+2], fill=(240,240,245))
    # card background (slightly off-white to separate from page)
    draw.rounded_rectangle([x1, y1, x2, y2], radius=card_radius, fill=card_bg, outline=card_border, width=1)

    # 3a) Thumbnail/background area on left side of each card (placeholder background only)
    # Thumbnail sizing: square, vertically centered within the card
    thumb_w = 232
    thumb_h = 232
    thumb_x1 = x1 + 0
    thumb_y1 = y1 + ( (y2 - y1) - thumb_h) // 2
    thumb_x2 = thumb_x1 + thumb_w
    thumb_y2 = thumb_y1 + thumb_h
    # draw thumbnail background (rounded)
    draw.rounded_rectangle([thumb_x1+12, thumb_y1, thumb_x2+12, thumb_y2], radius=12, fill=thumb_bg)

# 4) Separator lines between rows (faint)
for (_, y1, _, y2) in rows:
    # draw a faint separator just below each row (except last)
    sep_y = y2 + 2
    draw.line([(48, sep_y), (48+1344, sep_y)], fill=separator, width=1)

# 5) Floating location pill shadow (behind detected floating control; do not draw the pill itself)
# Detected floating pill area: pos=(518,2651) size=405x117 -> create soft shadow ellipse behind it
pill_x, pill_y, pill_w, pill_h = 518, 2651, 405, 117
shadow_box = [pill_x-18, pill_y+8, pill_x+pill_w+18, pill_y+pill_h+18]
# Draw an oval shadow (soft by using a slightly larger filled rounded rectangle)
draw.rounded_rectangle(shadow_box, radius=60, fill=floating_shadow)

# 6) Bottom navigation bar background and top border
nav_h = 120
nav_top = H - nav_h
draw.rectangle([0, nav_top, W, H], fill=nav_bg)
draw.line([(0, nav_top), (W, nav_top)], fill=nav_bar_top, width=2)

# 7) Additional subtle full-width separators for visual rhythm (don't overlap detected content)
# A faint divider under the header area (between toolbar and "More events" heading)
divider_y = toolbar_bottom + 28
draw.line([(48, divider_y), (W-48, divider_y)], fill=(245,245,247), width=1)

# End of structural drawing.
# (Do not draw text, icons, or any detected elements — they will be pasted on top.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/00_icon_iORk.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["iORk"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/01_icon_ZDRTTZY.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 490), _c1)
except Exception:
    pass
layout["ZDRTTZY"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/02_icon_95_HEEEYIMI_UESK_EEudooz.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 886), _c2)
except Exception:
    pass
layout["95_HEEEYIMI_UESK_EEudooz"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/03_icon_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/04_icon_DL_NO_COVER_ALL_NIGHT.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 490), _c4)
except Exception:
    pass
layout["DL_(NO_COVER_ALL_NIGHT)"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/05_icon_free_Grief_and_Loss_Healing_Circle.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1282), _c5)
except Exception:
    pass
layout["(free)_Grief_and_Loss_Hea"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/06_icon_The_DL.png
try:
    _c6 = get_crop(6, 144, 123)
    canvas.paste(_c6, (1140, 1951), _c6)
except Exception:
    pass
layout["The_DL"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/07_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c7 = get_crop(7, 1344, 396)
    canvas.paste(_c7, (48, 1678), _c7)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 1539), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 53, 57)
    canvas.paste(_c9, (183, 4), _c9)
except Exception:
    pass
layout["icon_9"] = [183, 4, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/10_icon_Favorite_button.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 763), _c10)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/11_icon_The_DL.png
try:
    _c11 = get_crop(11, 144, 123)
    canvas.paste(_c11, (1140, 2347), _c11)
except Exception:
    pass
layout["The_DL"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/12_icon_The_DL.png
try:
    _c12 = get_crop(12, 288, 156)
    canvas.paste(_c12, (288, 2804), _c12)
except Exception:
    pass
layout["The_DL"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/13_icon_The_DL.png
try:
    _c13 = get_crop(13, 144, 123)
    canvas.paste(_c13, (1284, 1951), _c13)
except Exception:
    pass
layout["The_DL"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1284, 1539), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 123)
    canvas.paste(_c15, (1284, 1159), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/16_icon_The_DL.png
try:
    _c16 = get_crop(16, 144, 123)
    canvas.paste(_c16, (1284, 2347), _c16)
except Exception:
    pass
layout["The_DL"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/17_icon_dtLaIct.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 1678), _c17)
except Exception:
    pass
layout["dtLaIct"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 55, 56)
    canvas.paste(_c18, (247, 5), _c18)
except Exception:
    pass
layout["icon_18"] = [247, 5, 302, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/19_icon_Favorite_button.png
try:
    _c19 = get_crop(19, 144, 123)
    canvas.paste(_c19, (1140, 1159), _c19)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/20_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 886), _c20)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/21_icon_Overflow_menu_button.png
try:
    _c21 = get_crop(21, 144, 123)
    canvas.paste(_c21, (1284, 763), _c21)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/22_icon_Ary.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (0, 2804), _c22)
except Exception:
    pass
layout["Ary"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 47, 52)
    canvas.paste(_c23, (1321, 7), _c23)
except Exception:
    pass
layout["icon_23"] = [1321, 7, 1368, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/24_icon_New_York.png
try:
    _c24 = get_crop(24, 405, 117)
    canvas.paste(_c24, (518, 2651), _c24)
except Exception:
    pass
layout["New_York"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/25_icon_9.37.png
try:
    _c25 = get_crop(25, 92, 100)
    canvas.paste(_c25, (47, 120), _c25)
except Exception:
    pass
layout["9.37"] = [47, 120, 139, 220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 63, 59)
    canvas.paste(_c26, (1211, 4), _c26)
except Exception:
    pass
layout["icon_26"] = [1211, 4, 1274, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 62, 58)
    canvas.paste(_c27, (311, 5), _c27)
except Exception:
    pass
layout["icon_27"] = [311, 5, 373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 48, 56)
    canvas.paste(_c28, (383, 7), _c28)
except Exception:
    pass
layout["icon_28"] = [383, 7, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/29_icon_9.37.png
try:
    _c29 = get_crop(29, 53, 57)
    canvas.paste(_c29, (115, 4), _c29)
except Exception:
    pass
layout["9.37"] = [115, 4, 168, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 42, 56)
    canvas.paste(_c30, (1272, 5), _c30)
except Exception:
    pass
layout["icon_30"] = [1272, 5, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/31_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c31 = get_crop(31, 1344, 396)
    canvas.paste(_c31, (48, 2074), _c31)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/32_icon_TUmU_5i0.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (576, 2804), _c32)
except Exception:
    pass
layout["TUmU'5i0"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/33_icon_Fireworks_July_Ath_Rooftop_Party.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 2074), _c33)
except Exception:
    pass
layout["Fireworks_July_Ath_Roofto"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/34_icon_icon_34.png
try:
    _c34 = get_crop(34, 31, 48)
    canvas.paste(_c34, (913, 2687), _c34)
except Exception:
    pass
layout["icon_34"] = [913, 2687, 944, 2735]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/35_text_9.37.png
try:
    _c35 = get_crop(35, 89, 43)
    canvas.paste(_c35, (20, 17), _c35)
except Exception:
    pass
layout["9.37"] = [20, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/36_text_More_events_you_II_love.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 490), _c36)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/37_text_Sun_Jun_23.png
try:
    _c37 = get_crop(37, 205, 49)
    canvas.paste(_c37, (388, 2554), _c37)
except Exception:
    pass
layout["Sun,_Jun_23"] = [388, 2554, 593, 2603]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/38_text_3_00_PM_EDT.png
try:
    _c38 = get_crop(38, 1344, 346)
    canvas.paste(_c38, (48, 2470), _c38)
except Exception:
    pass
layout["3:00_PM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/39_text_The_DL_Rooftop.png
try:
    _c39 = get_crop(39, 144, 123)
    canvas.paste(_c39, (1140, 2347), _c39)
except Exception:
    pass
layout["The_DL_Rooftop"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/40_text_Ary.png
try:
    _c40 = get_crop(40, 1344, 346)
    canvas.paste(_c40, (48, 2470), _c40)
except Exception:
    pass
layout["Ary"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/41_text_The_DL.png
try:
    _c41 = get_crop(41, 115, 38)
    canvas.paste(_c41, (394, 2693), _c41)
except Exception:
    pass
layout["The_DL"] = [394, 2693, 509, 2731]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/42_text_TUmU_5i0.png
try:
    _c42 = get_crop(42, 405, 117)
    canvas.paste(_c42, (518, 2651), _c42)
except Exception:
    pass
layout["TUmU'5i0"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/43_clickable_Tickets.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (864, 2804), _c43)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_01_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-3/44_clickable_More.png
try:
    _c44 = get_crop(44, 288, 156)
    canvas.paste(_c44, (1152, 2804), _c44)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
