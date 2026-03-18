# page_id: page_eventbrite_cee80d2c889b45cd86e30248baf590b8_06
# screenshot: 2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8.png
# step_index: 6/13
# task: Open Eventbrite. Search Food & Drink party event in New York. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw UI background and structure for Eventbrite-like list page
# Uses provided variables: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Canvas size
W, H = canvas.size

# Colors
dominant_bg = "#FFFFFF"            # overall background
status_bar_bg = "#CFCFCF"         # status bar gray
divider_color = "#ECEBF0"         # faint divider/separator
card_shadow = "#F3F3F5"           # subtle shadow under cards
card_bg = "#FFFFFF"               # card background (white)
header_divider = "#EDE9F1"        # header bottom divider
nav_divider = "#EAEAF0"           # nav top divider
content_tint = "#FAFAFB"          # slightly off-white tint for larger content areas

# Fill overall background
draw.rectangle([(0,0),(W,H)], fill=dominant_bg)

# Status bar area (~0 - 56px)
status_h = 56
draw.rectangle([(0,0),(W,status_h)], fill=status_bar_bg)
# thin bottom divider under status bar
draw.line([(0,status_h),(W,status_h)], fill=divider_color, width=1)

# Header/toolbar area (just below status bar)
header_top = status_h
header_bottom = 200
draw.rectangle([(0,header_top),(W,header_bottom)], fill=dominant_bg)
# header bottom divider
draw.line([(0,header_bottom),(W,header_bottom)], fill=header_divider, width=1)

# Content area background (slight tint behind list)
content_top = header_bottom + 24
draw.rectangle([(0,content_top),(W,H)], fill=dominant_bg)

# Define the list "cards" (x, y, width, height) matching detected rows
rows = [
    (48, 490, 1344, 396),
    (48, 886, 1344, 396),
    (48, 1282, 1344, 396),
    (48, 1678, 1344, 396),
    (48, 2074, 1344, 396),
    (48, 2470, 1344, 346)
]

# Draw each card background with subtle shadow and rounded corners
for (x, y, w, h) in rows:
    # shadow (offset)
    shadow_offset = 6
    shadow_box = [ (x+shadow_offset, y+shadow_offset), (x+w+shadow_offset, y+h+shadow_offset) ]
    draw.rounded_rectangle(shadow_box, radius=16, fill=card_shadow)
    # main card
    card_box = [ (x, y), (x+w, y+h) ]
    draw.rounded_rectangle(card_box, radius=16, fill=card_bg)
    # subtle top highlight line inside card
    highlight_y = y + 1
    draw.line([(x+8, highlight_y), (x+w-8, highlight_y)], fill="#FFFFFF", width=1)
    # bottom separator line (match the card bottom to separate from next)
    bottom_y = y + h
    draw.line([(x+8, bottom_y), (x+w-8, bottom_y)], fill=divider_color, width=1)

# Additional separators between list items (in case of overlapping/adjacent)
for i in range(len(rows)-1):
    _, y, _, h = rows[i]
    sep_y = y + h
    draw.line([(48, sep_y), (48+1344, sep_y)], fill=divider_color, width=1)

# Floating location pill area (background shape behind detected "New York" widget)
# We draw only a subtle rounded pill background (no text/icon)
pill_w, pill_h = 420, 110
pill_x = (W - pill_w) // 2
pill_y = 2651  # align with detected y of New York (pos=(518,2651) size=405x117)
pill_box = [(pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h)]
draw.rounded_rectangle(pill_box, radius=56, fill="#FFFFFF", outline=divider_color, )

# Bottom navigation bar background and top divider
nav_h = 120
nav_top = H - nav_h
draw.rectangle([(0, nav_top), (W, H)], fill=dominant_bg)
draw.line([(0, nav_top), (W, nav_top)], fill=nav_divider, width=1)

# Safe-area inner shadow for nav to separate from content slightly
draw.line([(0, nav_top+2), (W, nav_top+2)], fill="#F6F6F8", width=1)

# Small left edge vertical guideline (subtle) to visually align list content
draw.line([(48, header_bottom+12), (48, H - nav_h - 12)], fill="#F5F5F7", width=1)

# End of background & structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/00_icon_iORk.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["iORk"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/01_icon_ZDRTTZY.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 490), _c1)
except Exception:
    pass
layout["ZDRTTZY"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/02_icon_95_HEEEYIMI_UESK_EEudooz.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 886), _c2)
except Exception:
    pass
layout["95_HEEEYIMI_UESK_EEudooz"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/03_icon_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/04_icon_DL_NO_COVER_ALL_NIGHT.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 490), _c4)
except Exception:
    pass
layout["DL_(NO_COVER_ALL_NIGHT)"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/05_icon_free_Grief_and_Loss_Healing_Circle.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1282), _c5)
except Exception:
    pass
layout["(free)_Grief_and_Loss_Hea"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/06_icon_The_DL.png
try:
    _c6 = get_crop(6, 144, 123)
    canvas.paste(_c6, (1140, 1951), _c6)
except Exception:
    pass
layout["The_DL"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/07_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c7 = get_crop(7, 1344, 396)
    canvas.paste(_c7, (48, 1678), _c7)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 50, 64)
    canvas.paste(_c8, (1153, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [1153, 2, 1203, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/09_icon_Favorite_button.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1140, 1539), _c9)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/10_icon_Favorite_button.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 763), _c10)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/11_icon_The_DL.png
try:
    _c11 = get_crop(11, 144, 123)
    canvas.paste(_c11, (1140, 2347), _c11)
except Exception:
    pass
layout["The_DL"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 53, 57)
    canvas.paste(_c12, (183, 4), _c12)
except Exception:
    pass
layout["icon_12"] = [183, 4, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/13_icon_The_DL.png
try:
    _c13 = get_crop(13, 288, 156)
    canvas.paste(_c13, (288, 2804), _c13)
except Exception:
    pass
layout["The_DL"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/14_icon_The_DL.png
try:
    _c14 = get_crop(14, 144, 123)
    canvas.paste(_c14, (1284, 1951), _c14)
except Exception:
    pass
layout["The_DL"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 139)
    canvas.paste(_c15, (1284, 1539), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/16_icon_Overflow_menu_button.png
try:
    _c16 = get_crop(16, 144, 123)
    canvas.paste(_c16, (1284, 1159), _c16)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/17_icon_The_DL.png
try:
    _c17 = get_crop(17, 144, 123)
    canvas.paste(_c17, (1284, 2347), _c17)
except Exception:
    pass
layout["The_DL"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 56, 56)
    canvas.paste(_c18, (247, 5), _c18)
except Exception:
    pass
layout["icon_18"] = [247, 5, 303, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/19_icon_Favorite_button.png
try:
    _c19 = get_crop(19, 144, 123)
    canvas.paste(_c19, (1140, 1159), _c19)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/20_icon_dtLaIct.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 1678), _c20)
except Exception:
    pass
layout["dtLaIct"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/21_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 886), _c21)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/22_icon_Overflow_menu_button.png
try:
    _c22 = get_crop(22, 144, 123)
    canvas.paste(_c22, (1284, 763), _c22)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 97, 59)
    canvas.paste(_c23, (1216, 4), _c23)
except Exception:
    pass
layout["icon_23"] = [1216, 4, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/24_icon_Ary.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (0, 2804), _c24)
except Exception:
    pass
layout["Ary"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 47, 52)
    canvas.paste(_c25, (1321, 7), _c25)
except Exception:
    pass
layout["icon_25"] = [1321, 7, 1368, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/26_icon_New_York.png
try:
    _c26 = get_crop(26, 405, 117)
    canvas.paste(_c26, (518, 2651), _c26)
except Exception:
    pass
layout["New_York"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/27_icon_9.44.png
try:
    _c27 = get_crop(27, 93, 100)
    canvas.paste(_c27, (46, 120), _c27)
except Exception:
    pass
layout["9.44"] = [46, 120, 139, 220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 62, 58)
    canvas.paste(_c28, (311, 5), _c28)
except Exception:
    pass
layout["icon_28"] = [311, 5, 373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 48, 56)
    canvas.paste(_c29, (383, 7), _c29)
except Exception:
    pass
layout["icon_29"] = [383, 7, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/30_icon_Best_Rooftop_Lounge_NYC.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 2074), _c30)
except Exception:
    pass
layout["Best_Rooftop_Lounge_NYC"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/31_icon_TUmU_5i0.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (576, 2804), _c31)
except Exception:
    pass
layout["TUmU'5i0"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/32_icon_Fireworks_July_Ath_Rooftop_Party.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 2074), _c32)
except Exception:
    pass
layout["Fireworks_July_Ath_Roofto"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/33_icon_icon_33.png
try:
    _c33 = get_crop(33, 31, 48)
    canvas.paste(_c33, (913, 2687), _c33)
except Exception:
    pass
layout["icon_33"] = [913, 2687, 944, 2735]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/34_icon_9.44.png
try:
    _c34 = get_crop(34, 57, 58)
    canvas.paste(_c34, (112, 4), _c34)
except Exception:
    pass
layout["9.44"] = [112, 4, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/35_text_More_events_you_II_love.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 490), _c35)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/36_text_Sun_Jun_23.png
try:
    _c36 = get_crop(36, 205, 49)
    canvas.paste(_c36, (388, 2554), _c36)
except Exception:
    pass
layout["Sun,_Jun_23"] = [388, 2554, 593, 2603]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/37_text_3_00_PM_EDT.png
try:
    _c37 = get_crop(37, 1344, 346)
    canvas.paste(_c37, (48, 2470), _c37)
except Exception:
    pass
layout["3:00_PM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/38_text_The_DL_Rooftop.png
try:
    _c38 = get_crop(38, 144, 123)
    canvas.paste(_c38, (1140, 2347), _c38)
except Exception:
    pass
layout["The_DL_Rooftop"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/39_text_Ary.png
try:
    _c39 = get_crop(39, 1344, 346)
    canvas.paste(_c39, (48, 2470), _c39)
except Exception:
    pass
layout["Ary"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/40_text_The_DL.png
try:
    _c40 = get_crop(40, 115, 38)
    canvas.paste(_c40, (394, 2693), _c40)
except Exception:
    pass
layout["The_DL"] = [394, 2693, 509, 2731]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/41_text_TUmU_5i0.png
try:
    _c41 = get_crop(41, 405, 117)
    canvas.paste(_c41, (518, 2651), _c41)
except Exception:
    pass
layout["TUmU'5i0"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/42_clickable_Tickets.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (864, 2804), _c42)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_06_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-8/43_clickable_More.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (1152, 2804), _c43)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
