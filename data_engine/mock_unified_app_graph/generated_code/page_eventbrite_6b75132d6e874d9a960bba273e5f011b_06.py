# page_id: page_eventbrite_6b75132d6e874d9a960bba273e5f011b_06
# screenshot: 2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8.png
# step_index: 6/11
# task: Open Eventbrite. Set the city to 'San Francisco'. Search 'Outdoor'. Select an event starting after 5 PM. Check the ticket price.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (canvas: PIL Image 1440x2960 RGB, draw: ImageDraw)
# Fonts provided: font_sm, font_md, font_lg, font_xl

# Colors
bg_color = (255, 255, 255)            # main background (white)
status_bar_color = (200, 200, 200)    # light gray status bar
header_bg = (255, 255, 255)           # header area (white)
search_fill = (245, 247, 250)         # search box background
search_outline = (226, 230, 235)      # search box border
card_bg = (255, 255, 255)             # card background (white)
card_border = (236, 239, 241)         # subtle card border
thumb_bg = (245, 245, 246)            # thumbnail placeholder background
divider = (238, 240, 242)             # separators
bottom_bar = (255, 255, 255)          # bottom navigation background
shadow_color = (220, 220, 220)        # subtle shadow

# Fill overall background
draw.rectangle([(0, 0), (1440, 2960)], fill=bg_color)

# Status bar (top ~0-86px)
status_h = 86
draw.rectangle([(0, 0), (1440, status_h)], fill=status_bar_color)

# Header area under status bar
header_top = status_h
header_bottom = 280
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=header_bg)

# Search field background (do not draw text/icons)
search_x, search_y = 195, 93
search_w, search_h = 1179, 144
search_bbox = (search_x, search_y, search_x + search_w, search_y + search_h)
# Rounded rectangle for search field
try:
    draw.rounded_rectangle(search_bbox, radius=72, fill=search_fill, outline=search_outline, width=2)
except Exception:
    # fallback if rounded_rectangle not available
    draw.rectangle(search_bbox, fill=search_fill, outline=search_outline)

# Subtle header bottom divider
draw.line([(48, header_bottom), (1392, header_bottom)], fill=divider, width=1)

# Event rows (background cards). Use the detected vertical positions and sizes.
rows = [
    (48, 490, 48 + 1344, 490 + 396),
    (48, 886, 48 + 1344, 886 + 396),
    (48, 1282, 48 + 1344, 1282 + 396),
    (48, 1678, 48 + 1344, 1678 + 396),
    (48, 2074, 48 + 1344, 2074 + 396),
    (48, 2470, 48 + 1344, 2470 + 346)  # last row slightly shorter
]

# Draw each card with a subtle shadow and border (background only)
for (x1, y1, x2, y2) in rows:
    radius = 12
    # shadow (slightly offset)
    sh_offset = 6
    shadow_bbox = (x1, y1 + sh_offset, x2, y2 + sh_offset)
    try:
        draw.rounded_rectangle(shadow_bbox, radius=radius, fill=shadow_color)
    except Exception:
        draw.rectangle(shadow_bbox, fill=shadow_color)
    # card background (on top of shadow)
    card_bbox = (x1, y1, x2, y2)
    try:
        draw.rounded_rectangle(card_bbox, radius=radius, fill=card_bg, outline=card_border, width=1)
    except Exception:
        draw.rectangle(card_bbox, fill=card_bg, outline=card_border)

    # Left thumbnail background (placeholder shape behind image)
    # Typical thumbnail size ~150x150, vertically centered within card
    thumb_w, thumb_h = 150, 150
    thumb_x = x1 + 0
    thumb_y = y1 + ( (y2 - y1) - thumb_h ) // 2
    thumb_bbox = (thumb_x, thumb_y, thumb_x + thumb_w, thumb_y + thumb_h)
    try:
        draw.rounded_rectangle(thumb_bbox, radius=8, fill=thumb_bg)
    except Exception:
        draw.rectangle(thumb_bbox, fill=thumb_bg)

    # Right-side small divider between card content and right edge (visual structure)
    right_div_x = x2 - 8
    draw.line([(right_div_x, y1 + 12), (right_div_x, y2 - 12)], fill=divider, width=1)

# Horizontal separators between cards (subtle)
for (x1, y1, x2, y2) in rows:
    sep_y = y2 + 16
    if sep_y < 2800:
        draw.line([(48, sep_y), (1392, sep_y)], fill=divider, width=1)

# Floating "San Francisco" pill is a detected element; do NOT draw it.
# Instead ensure area behind it (main canvas) is clean. We leave it as-is.

# Bottom navigation bar background and top divider (do not draw icons)
bottom_top = 2804
draw.line([(0, bottom_top), (1440, bottom_top)], fill=divider, width=1)
draw.rectangle([(0, bottom_top), (1440, 2960)], fill=bottom_bar)

# Subtle notch/shadow above bottom bar to match screenshot feel
draw.rectangle([(0, bottom_top - 6), (1440, bottom_top)], fill=(250, 250, 251))

# Final subtle vignette at the very top of content area (under header) for depth
vignette_y = header_bottom + 2
draw.line([(48, vignette_y), (1392, vignette_y)], fill=(245, 245, 246), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/00_icon_ering_to_soothe_the_brokel.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1678), _c0)
except Exception:
    pass
layout["ering_to_soothe_the_broke"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/01_icon_Spring-Zing_Happy_Hour.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 2074), _c1)
except Exception:
    pass
layout["Spring-Zing_Happy_Hour"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/02_icon_NDIE.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 490), _c2)
except Exception:
    pass
layout["NDIE"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/03_icon_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/04_icon_Sat.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 886), _c4)
except Exception:
    pass
layout["Sat,"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/05_icon_San_Francisco.png
try:
    _c5 = get_crop(5, 495, 117)
    canvas.paste(_c5, (473, 2651), _c5)
except Exception:
    pass
layout["San_Francisco"] = [473, 2651, 968, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/06_icon_Spring-Zing_Happy.png
try:
    _c6 = get_crop(6, 144, 123)
    canvas.paste(_c6, (1140, 1951), _c6)
except Exception:
    pass
layout["Spring-Zing_Happy"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 48, 65)
    canvas.paste(_c7, (1154, 2), _c7)
except Exception:
    pass
layout["icon_7"] = [1154, 2, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 747), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/09_icon_City.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1140, 1539), _c9)
except Exception:
    pass
layout["City"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/10_icon_Bissa.png
try:
    _c10 = get_crop(10, 288, 156)
    canvas.paste(_c10, (288, 2804), _c10)
except Exception:
    pass
layout["Bissa}"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/11_icon_Reggaeton.png
try:
    _c11 = get_crop(11, 144, 123)
    canvas.paste(_c11, (1140, 2347), _c11)
except Exception:
    pass
layout["Reggaeton__"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/12_icon_RIEF_MEDICIN.png
try:
    _c12 = get_crop(12, 1344, 396)
    canvas.paste(_c12, (48, 1282), _c12)
except Exception:
    pass
layout["RIEF_MEDICIN"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/13_icon_Spring-Zing_Happy.png
try:
    _c13 = get_crop(13, 144, 123)
    canvas.paste(_c13, (1284, 1951), _c13)
except Exception:
    pass
layout["Spring-Zing_Happy"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1284, 747), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/15_icon_8.11.png
try:
    _c15 = get_crop(15, 108, 101)
    canvas.paste(_c15, (38, 121), _c15)
except Exception:
    pass
layout["8.11"] = [38, 121, 146, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/16_icon_City.png
try:
    _c16 = get_crop(16, 144, 139)
    canvas.paste(_c16, (1284, 1539), _c16)
except Exception:
    pass
layout["City"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/17_icon_Reggaeton.png
try:
    _c17 = get_crop(17, 144, 123)
    canvas.paste(_c17, (1284, 2347), _c17)
except Exception:
    pass
layout["Reggaeton__"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/18_icon_8.11.png
try:
    _c18 = get_crop(18, 54, 60)
    canvas.paste(_c18, (184, 2), _c18)
except Exception:
    pass
layout["8.11"] = [184, 2, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/19_icon_SatvaonG.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (0, 2804), _c19)
except Exception:
    pass
layout["SatvaonG"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/20_icon_City.png
try:
    _c20 = get_crop(20, 144, 139)
    canvas.paste(_c20, (1140, 1143), _c20)
except Exception:
    pass
layout["City"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/21_icon_PDO_Thread_Training.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1282), _c21)
except Exception:
    pass
layout["PDO_Thread_Training_|"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/22_icon_Overflow_menu_button.png
try:
    _c22 = get_crop(22, 144, 139)
    canvas.paste(_c22, (1284, 1143), _c22)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 58, 57)
    canvas.paste(_c23, (313, 4), _c23)
except Exception:
    pass
layout["icon_23"] = [313, 4, 371, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 97, 59)
    canvas.paste(_c24, (1216, 4), _c24)
except Exception:
    pass
layout["icon_24"] = [1216, 4, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/25_icon_8.11.png
try:
    _c25 = get_crop(25, 57, 59)
    canvas.paste(_c25, (114, 3), _c25)
except Exception:
    pass
layout["8.11"] = [114, 3, 171, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 49, 58)
    canvas.paste(_c26, (248, 3), _c26)
except Exception:
    pass
layout["icon_26"] = [248, 3, 297, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 47, 52)
    canvas.paste(_c27, (1321, 8), _c27)
except Exception:
    pass
layout["icon_27"] = [1321, 8, 1368, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/28_icon_8_29_creator_followers.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 886), _c28)
except Exception:
    pass
layout["8_29_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/29_icon_59_creator_followers.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 490), _c29)
except Exception:
    pass
layout["59_creator_followers"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/30_icon_8_100_creator_followers.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 1678), _c30)
except Exception:
    pass
layout["8_100_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/31_icon_Free.png
try:
    _c31 = get_crop(31, 125, 73)
    canvas.paste(_c31, (248, 561), _c31)
except Exception:
    pass
layout["Free"] = [248, 561, 373, 634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/32_icon_Area.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 2074), _c32)
except Exception:
    pass
layout["Area"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/33_icon_Salsa.png
try:
    _c33 = get_crop(33, 1344, 346)
    canvas.paste(_c33, (48, 2470), _c33)
except Exception:
    pass
layout["Salsa"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/34_icon_icon_34.png
try:
    _c34 = get_crop(34, 43, 55)
    canvas.paste(_c34, (385, 7), _c34)
except Exception:
    pass
layout["icon_34"] = [385, 7, 428, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/35_icon_Grief_Medicine_A_Gathering_to_Soothe_the.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 1678), _c35)
except Exception:
    pass
layout["Grief_Medicine:_A_Gatheri"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/36_icon_Processing_Grief_Self-Care_for_Loss.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 886), _c36)
except Exception:
    pass
layout["Processing_Grief:_Self-Ca"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/37_icon_Yggae.png
try:
    _c37 = get_crop(37, 150, 68)
    canvas.paste(_c37, (933, 2643), _c37)
except Exception:
    pass
layout["Yggae"] = [933, 2643, 1083, 2711]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/38_icon_8_100_creator_followers.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 1678), _c38)
except Exception:
    pass
layout["8_100_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/39_text_8.11.png
try:
    _c39 = get_crop(39, 89, 41)
    canvas.paste(_c39, (20, 17), _c39)
except Exception:
    pass
layout["8.11"] = [20, 17, 109, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/40_text_More_events_you_II_love.png
try:
    _c40 = get_crop(40, 1344, 396)
    canvas.paste(_c40, (48, 490), _c40)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/41_text_Mon.png
try:
    _c41 = get_crop(41, 92, 43)
    canvas.paste(_c41, (393, 2129), _c41)
except Exception:
    pass
layout["Mon,"] = [393, 2129, 485, 2172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/42_text_13.png
try:
    _c42 = get_crop(42, 54, 38)
    canvas.paste(_c42, (561, 2129), _c42)
except Exception:
    pass
layout["13"] = [561, 2129, 615, 2167]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/43_text_5_00_PM_PDT.png
try:
    _c43 = get_crop(43, 1344, 396)
    canvas.paste(_c43, (48, 2074), _c43)
except Exception:
    pass
layout["5:00_PM_PDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/44_text_Hour_The_Lookout.png
try:
    _c44 = get_crop(44, 1344, 396)
    canvas.paste(_c44, (48, 2074), _c44)
except Exception:
    pass
layout["Hour_@_The_Lookout"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/45_text_3600_16th_St.png
try:
    _c45 = get_crop(45, 223, 38)
    canvas.paste(_c45, (392, 2328), _c45)
except Exception:
    pass
layout["3600_16th_St"] = [392, 2328, 615, 2366]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/46_text_8_9266_creator_followers.png
try:
    _c46 = get_crop(46, 1344, 396)
    canvas.paste(_c46, (48, 2074), _c46)
except Exception:
    pass
layout["8_9266_creator_followers"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/47_text_Aanonananal.png
try:
    _c47 = get_crop(47, 194, 14)
    canvas.paste(_c47, (98, 2542), _c47)
except Exception:
    pass
layout["Aanonananal"] = [98, 2542, 292, 2556]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/48_text_Sat_May_4.png
try:
    _c48 = get_crop(48, 186, 43)
    canvas.paste(_c48, (392, 2525), _c48)
except Exception:
    pass
layout["Sat,_May_4"] = [392, 2525, 578, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/49_text_IO_00_AM_PDT.png
try:
    _c49 = get_crop(49, 1344, 346)
    canvas.paste(_c49, (48, 2470), _c49)
except Exception:
    pass
layout["IO:00_AM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/50_text_hellaGood.png
try:
    _c50 = get_crop(50, 186, 41)
    canvas.paste(_c50, (101, 2556), _c50)
except Exception:
    pass
layout["hellaGood"] = [101, 2556, 287, 2597]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/51_text_ssan.png
try:
    _c51 = get_crop(51, 25, 9)
    canvas.paste(_c51, (252, 2636), _c51)
except Exception:
    pass
layout["ssan"] = [252, 2636, 277, 2645]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/52_text_featuring-.png
try:
    _c52 = get_crop(52, 43, 15)
    canvas.paste(_c52, (215, 2650), _c52)
except Exception:
    pass
layout["'featuring-"] = [215, 2650, 258, 2665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/53_text_Jah_Wafeidk_SHELTER.png
try:
    _c53 = get_crop(53, 129, 13)
    canvas.paste(_c53, (142, 2702), _c53)
except Exception:
    pass
layout["Jah_Wafeidk_SHELTER"] = [142, 2702, 271, 2715]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/54_text_DJGREENB_DJAGANA_DJMALIIGZ.png
try:
    _c54 = get_crop(54, 215, 18)
    canvas.paste(_c54, (91, 2718), _c54)
except Exception:
    pass
layout["DJGREENB_DJAGANA_DJMALIIG"] = [91, 2718, 306, 2736]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/55_text_Log_AETa.png
try:
    _c55 = get_crop(55, 41, 6)
    canvas.paste(_c55, (111, 2738), _c55)
except Exception:
    pass
layout["Log__AETa"] = [111, 2738, 152, 2744]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/56_text_atrobcats.png
try:
    _c56 = get_crop(56, 43, 13)
    canvas.paste(_c56, (156, 2746), _c56)
except Exception:
    pass
layout["atrobcats"] = [156, 2746, 199, 2759]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/57_text_nalceuani.png
try:
    _c57 = get_crop(57, 37, 7)
    canvas.paste(_c57, (212, 2742), _c57)
except Exception:
    pass
layout["nalceuani"] = [212, 2742, 249, 2749]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/58_text_Lrocaa_Rnrae.png
try:
    _c58 = get_crop(58, 53, 9)
    canvas.paste(_c58, (240, 2763), _c58)
except Exception:
    pass
layout["Lrocaa_Rnrae"] = [240, 2763, 293, 2772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/59_text_SatvaonG.png
try:
    _c59 = get_crop(59, 60, 29)
    canvas.paste(_c59, (92, 2761), _c59)
except Exception:
    pass
layout["SatvaonG"] = [92, 2761, 152, 2790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/60_text_t0ph.png
try:
    _c60 = get_crop(60, 32, 15)
    canvas.paste(_c60, (158, 2767), _c60)
except Exception:
    pass
layout["t0ph"] = [158, 2767, 190, 2782]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/61_text_Z44.png
try:
    _c61 = get_crop(61, 23, 15)
    canvas.paste(_c61, (197, 2767), _c61)
except Exception:
    pass
layout["Z44"] = [197, 2767, 220, 2782]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/62_text_71J_Nissiom_St.st.png
try:
    _c62 = get_crop(62, 74, 13)
    canvas.paste(_c62, (232, 2774), _c62)
except Exception:
    pass
layout["{71J_Nissiom_St.st"] = [232, 2774, 306, 2787]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/63_clickable_Favorites.png
try:
    _c63 = get_crop(63, 288, 156)
    canvas.paste(_c63, (576, 2804), _c63)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/64_clickable_Tickets.png
try:
    _c64 = get_crop(64, 288, 156)
    canvas.paste(_c64, (864, 2804), _c64)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/6b75132d6e874d9a960bba273e5f011b/step_06_2024_4_23_20_10_6b75132d6e874d9a960bba273e5f011b-8/65_clickable_More.png
try:
    _c65 = get_crop(65, 288, 156)
    canvas.paste(_c65, (1152, 2804), _c65)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
