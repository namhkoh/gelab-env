# page_id: page_eventbrite_5362d15941a843c5a00f4a85c7ff0a9a_06
# screenshot: 2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8.png
# step_index: 6/12
# task: Open Eventbrite. Set the city to 'Los Angeles'. Search 'Business'. Filter 'French' speaking events. Add the first event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background & structure painting for Eventbrite-like UI
# Uses provided variables: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
bg_color = (250, 250, 251)         # very light off-white background
status_color = (189, 189, 189)     # light grey status bar
toolbar_color = (255, 255, 255)    # white toolbar area
card_color = (255, 255, 255)       # white cards
card_border = (238, 239, 241)      # subtle card border
divider = (236, 236, 236)          # light divider lines
nav_bg = (255, 255, 255)           # bottom navigation background
floating_shadow = (235, 236, 240)  # floating pill shadow

w, h = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar area (approx ~50-60px high)
status_h = 60
draw.rectangle([(0, 0), (w, status_h)], fill=status_color)
# subtle bottom border for status bar
draw.line([(0, status_h - 1), (w, status_h - 1)], fill=(170, 170, 170), width=1)

# Toolbar / header area behind search and logo
toolbar_top = status_h
toolbar_bottom = 180
draw.rectangle([(0, toolbar_top), (w, toolbar_bottom)], fill=toolbar_color)
# separator under toolbar
draw.line([(48, toolbar_bottom), (w - 48, toolbar_bottom)], fill=divider, width=1)

# Main content area - draw section background (keeps overall white; subtle top padding area)
content_top = toolbar_bottom + 24
# A very subtle horizontal bar behind the section title area to ground the heading
draw.rectangle([(0, content_top), (w, content_top + 80)], fill=(250, 250, 251))

# Card list: Draw rounded white cards behind each event list item.
card_x0 = 48
card_x1 = card_x0 + 1344  # matches detected widths
card_width = card_x1 - card_x0
card_height = 396
card_radius = 14

card_tops = [490, 886, 1282, 1678, 2074, 2470]
for top in card_tops:
    bottom = top + card_height
    # Draw card background
    draw.rounded_rectangle([(card_x0, top), (card_x1, bottom)],
                           radius=card_radius,
                           fill=card_color,
                           outline=card_border,
                           width=1)
    # subtle inner top highlight
    highlight_y = top + 8
    draw.line([(card_x0 + 8, highlight_y), (card_x1 - 8, highlight_y)], fill=(250,250,250), width=1)
    # bottom divider shadow to separate cards
    draw.line([(card_x0 + 12, bottom + 1), (card_x1 - 12, bottom + 1)], fill=(245,245,246), width=1)

# Additional subtle separators between list groups (outside card bounds)
for sep_y in [card_tops[0] - 28, card_tops[2] - 28, card_tops[4] - 28]:
    draw.line([(48, sep_y), (w - 48, sep_y)], fill=divider, width=1)

# Floating location pill shadow (background only; the actual pill will be pasted)
# Detected Los Angeles pill at (492,2651) size 456x117
pill_x0 = 492 - 24
pill_y0 = 2651 - 24
pill_x1 = 492 + 456 + 24 - 0
pill_y1 = 2651 + 117 + 24 - 0
# draw a subtle rounded shadow behind the pill
draw.rounded_rectangle([(pill_x0, pill_y0), (pill_x1, pill_y1)],
                       radius=60, fill=floating_shadow)

# Bottom navigation bar background
nav_top = 2804
nav_bottom = h
draw.rectangle([(0, nav_top), (w, nav_bottom)], fill=nav_bg)
# top border for the nav area
draw.line([(0, nav_top), (w, nav_top)], fill=divider, width=1)

# Small home indicator/background at bottom center (background only; icon will be pasted)
indicator_w = 160
indicator_h = 6
ind_x0 = (w - indicator_w) // 2
ind_x1 = ind_x0 + indicator_w
ind_y0 = nav_top + 14
ind_y1 = ind_y0 + indicator_h
draw.rounded_rectangle([(ind_x0, ind_y0), (ind_x1, ind_y1)], radius=3, fill=(245,245,247))

# Final subtle vignette on edges (very light) to match screenshot feel
edge_strip = 10
# left
draw.rectangle([(0, 0), (edge_strip, h)], fill=(253,253,253))
# right
draw.rectangle([(w - edge_strip, 0), (w, h)], fill=(253,253,253))
# done - leave text/icons/buttons to be pasted by downstream process

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/00_icon_Free.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["Free"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/01_icon_NDIE_DANCEPA.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1678), _c1)
except Exception:
    pass
layout["NDIE_DANCEPA"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/02_icon_Ibaigktsinel.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1282), _c2)
except Exception:
    pass
layout["Ibaigktsinel"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/03_icon_Q_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Q_Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/04_icon_Los_Angeles.png
try:
    _c4 = get_crop(4, 456, 117)
    canvas.paste(_c4, (492, 2651), _c4)
except Exception:
    pass
layout["Los_Angeles"] = [492, 2651, 948, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/05_icon_NDIE.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 886), _c5)
except Exception:
    pass
layout["NDIE"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/06_icon_Afliccion_Perdida_y.png
try:
    _c6 = get_crop(6, 144, 139)
    canvas.paste(_c6, (1140, 1935), _c6)
except Exception:
    pass
layout["Afliccion,_Perdida_y"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/07_icon_ter_for_Break_Into_Tech_nowl.png
try:
    _c7 = get_crop(7, 1344, 396)
    canvas.paste(_c7, (48, 490), _c7)
except Exception:
    pass
layout["ter_for_Break_Into_Tech_n"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 123)
    canvas.paste(_c8, (1140, 1555), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1555, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1284, 1935), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 49, 65)
    canvas.paste(_c10, (1153, 2), _c10)
except Exception:
    pass
layout["icon_10"] = [1153, 2, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 123)
    canvas.paste(_c11, (1284, 1555), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1555, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/12_icon_Afliccion_Perdida_y.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1140, 2347), _c12)
except Exception:
    pass
layout["Afliccion,_Perdida_y"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 123)
    canvas.paste(_c13, (1284, 2347), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1284, 1143), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/15_icon_Club_Decades.png
try:
    _c15 = get_crop(15, 144, 139)
    canvas.paste(_c15, (1140, 1143), _c15)
except Exception:
    pass
layout["Club_Decades"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/16_icon_The_Gr.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (288, 2804), _c16)
except Exception:
    pass
layout["The_Gr"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/17_icon_Favorite_button.png
try:
    _c17 = get_crop(17, 144, 123)
    canvas.paste(_c17, (1140, 763), _c17)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 98, 60)
    canvas.paste(_c18, (1215, 3), _c18)
except Exception:
    pass
layout["icon_18"] = [1215, 3, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/19_icon_Overflow_menu_button.png
try:
    _c19 = get_crop(19, 144, 123)
    canvas.paste(_c19, (1284, 763), _c19)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 62, 59)
    canvas.paste(_c20, (311, 3), _c20)
except Exception:
    pass
layout["icon_20"] = [311, 3, 373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/21_icon_Home.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (0, 2804), _c21)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/22_icon_8.02.png
try:
    _c22 = get_crop(22, 57, 60)
    canvas.paste(_c22, (182, 2), _c22)
except Exception:
    pass
layout["8.02"] = [182, 2, 239, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/23_icon_8_4717_creator_followers.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 886), _c23)
except Exception:
    pass
layout["8_4717_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 52, 60)
    canvas.paste(_c24, (247, 2), _c24)
except Exception:
    pass
layout["icon_24"] = [247, 2, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/25_icon_59_creator_followers.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 1678), _c25)
except Exception:
    pass
layout["59_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/26_icon_8.02.png
try:
    _c26 = get_crop(26, 104, 100)
    canvas.paste(_c26, (40, 121), _c26)
except Exception:
    pass
layout["8.02"] = [40, 121, 144, 221]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 49, 54)
    canvas.paste(_c27, (1320, 6), _c27)
except Exception:
    pass
layout["icon_27"] = [1320, 6, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/28_icon_Public_House_Los_Angeles_CA.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 490), _c28)
except Exception:
    pass
layout["Public_House_(Los_Angeles"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/29_icon_8.02.png
try:
    _c29 = get_crop(29, 58, 62)
    canvas.paste(_c29, (115, 1), _c29)
except Exception:
    pass
layout["8.02"] = [115, 1, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/30_icon_8_21119_creator_followers.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 1282), _c30)
except Exception:
    pass
layout["8_21119_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/31_icon_Q_Search_events.png
try:
    _c31 = get_crop(31, 45, 57)
    canvas.paste(_c31, (385, 6), _c31)
except Exception:
    pass
layout["Q_Search_events"] = [385, 6, 430, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/32_icon_Grief_Loss_Resiliency.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 2074), _c32)
except Exception:
    pass
layout["Grief;_Loss,_Resiliency"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/33_icon_9.30_PM_PDT.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 886), _c33)
except Exception:
    pass
layout["9.30_PM_PDT"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/34_icon_Free.png
try:
    _c34 = get_crop(34, 127, 74)
    canvas.paste(_c34, (246, 1748), _c34)
except Exception:
    pass
layout["Free"] = [246, 1748, 373, 1822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/35_icon_YEAH_YEAH_YAS_Queer_Indie_Dance_Party_LA.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 1678), _c35)
except Exception:
    pass
layout["YEAH_YEAH_YAS:_Queer_Indi"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/36_icon_8_21119_creator_followers.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 1282), _c36)
except Exception:
    pass
layout["8_21119_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/37_icon_8_4717_creator_followers.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 886), _c37)
except Exception:
    pass
layout["8_4717_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/38_text_8.02.png
try:
    _c38 = get_crop(38, 91, 43)
    canvas.paste(_c38, (20, 17), _c38)
except Exception:
    pass
layout["8.02"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/39_text_More_events_you_II_love.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 490), _c39)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/40_text_2000s_NITE.png
try:
    _c40 = get_crop(40, 202, 49)
    canvas.paste(_c40, (81, 2528), _c40)
except Exception:
    pass
layout["2000s_NITE"] = [81, 2528, 283, 2577]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/41_text_Fri_May_31.png
try:
    _c41 = get_crop(41, 184, 43)
    canvas.paste(_c41, (392, 2525), _c41)
except Exception:
    pass
layout["Fri,_May_31"] = [392, 2525, 576, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/42_text_9_30_PM_PDT.png
try:
    _c42 = get_crop(42, 1344, 346)
    canvas.paste(_c42, (48, 2470), _c42)
except Exception:
    pass
layout["9:30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/43_text_UNDERGROUND_X_200Os_NITE_Dance_Partyl.png
try:
    _c43 = get_crop(43, 1344, 346)
    canvas.paste(_c43, (48, 2470), _c43)
except Exception:
    pass
layout["UNDERGROUND_X_200Os_NITE_"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/44_text_MEt.png
try:
    _c44 = get_crop(44, 79, 67)
    canvas.paste(_c44, (205, 2598), _c44)
except Exception:
    pass
layout["MEt"] = [205, 2598, 284, 2665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/45_clickable_Favorites.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (576, 2804), _c45)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/46_clickable_Tickets.png
try:
    _c46 = get_crop(46, 288, 156)
    canvas.paste(_c46, (864, 2804), _c46)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5362d15941a843c5a00f4a85c7ff0a9a/step_06_2024_4_23_20_1_5362d15941a843c5a00f4a85c7ff0a9a-8/47_clickable_More.png
try:
    _c47 = get_crop(47, 288, 156)
    canvas.paste(_c47, (1152, 2804), _c47)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
