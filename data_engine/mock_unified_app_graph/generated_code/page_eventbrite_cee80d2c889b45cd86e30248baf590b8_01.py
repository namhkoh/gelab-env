# page_id: page_eventbrite_cee80d2c889b45cd86e30248baf590b8_01
# screenshot: 2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3.png
# step_index: 1/13
# task: Open Eventbrite. Search Food & Drink party event in New York. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (canvas already provided)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar (top ~72px) - darker strip
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(158, 158, 158))

# Header / toolbar area under status bar
header_top = status_h
header_bottom = 176
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))
# subtle bottom divider for header
draw.line([(0, header_bottom), (1440, header_bottom)], fill=(230, 230, 235), width=2)

# Content area: subtle page background tint (still very light)
draw.rectangle([(0, header_bottom), (1440, 2960)], fill=(255, 255, 255))

# Row bounding Y positions (detected row boxes)
row_tops = [490, 886, 1282, 1678, 2074, 2347, 2651]
row_height = 396

# Colors for thumbnail placeholders (varied but muted so they won't be mistaken for exact content)
thumb_colors = [
    (238, 240, 243),  # light gray
    (25, 25, 25),     # dark (for image-like posts)
    (60, 40, 80),     # deep muted purple
    (45, 45, 45),     # dark gray
    (230, 235, 240),  # pale blue-gray
    (220, 200, 230),  # pale lavender
    (245, 230, 220)   # pale peach
]

# Draw row card backgrounds (subtle rounded rectangles) and thumbnail placeholders
for i, top in enumerate(row_tops):
    card_left = 32
    card_right = 1408
    card_top = top + 8
    card_bottom = top + row_height - 8
    # Card background (very subtle off-white with light border)
    draw.rounded_rectangle(
        [(card_left, card_top), (card_right, card_bottom)],
        radius=14,
        fill=(255, 255, 255),
        outline=(235, 235, 239),
        width=1
    )

    # Thumbnail placeholder on the left
    thumb_x = 48
    thumb_y = top + 22
    thumb_size = 150
    color = thumb_colors[i % len(thumb_colors)]
    draw.rounded_rectangle(
        [(thumb_x, thumb_y), (thumb_x + thumb_size, thumb_y + thumb_size)],
        radius=10,
        fill=color,
        outline=(210, 210, 215),
        width=1
    )

    # A subtle inner "image content" shape to suggest imagery without drawing any real content
    inner_pad = 10
    draw.rounded_rectangle(
        [
            (thumb_x + inner_pad, thumb_y + inner_pad),
            (thumb_x + thumb_size - inner_pad, thumb_y + thumb_size - inner_pad)
        ],
        radius=6,
        fill=(0, 0, 0, 0),
        outline=(200, 200, 205),
        width=1
    )

    # Divider line below each card
    divider_y = card_bottom + 12
    draw.line([(48, divider_y), (1392, divider_y)], fill=(240, 240, 244), width=1)

# Floating location pill area (do not draw the actual map pin or text) - draw only a subtle backdrop
pill_center_y = 2526  # approximate location from detected elements
pill_w = 420
pill_h = 84
pill_x = (1440 - pill_w) // 2
pill_y = pill_center_y - pill_h // 2
draw.rounded_rectangle(
    [(pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h)],
    radius=42,
    fill=(255, 255, 255),
    outline=(220, 220, 225),
    width=1
)

# Bottom navigation bar background and top divider
nav_top = 2804
draw.rectangle([(0, nav_top), (1440, 2960)], fill=(255, 255, 255))
draw.line([(0, nav_top), (1440, nav_top)], fill=(230, 230, 235), width=2)

# Optional subtle shadows under some cards to separate layers (very faint)
for top in row_tops:
    shadow_box = (40, top + row_height - 2, 1400, top + row_height + 6)
    # simulate a faint shadow by drawing several translucent lines (approximate using lighter grays)
    for offset, alpha in enumerate([12, 9, 6], start=0):
        y = shadow_box[1] + offset
        draw.line([(shadow_box[0] + offset, y), (shadow_box[2] - offset, y)], fill=(240 - offset*3, 240 - offset*3, 244 - offset*2), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/00_icon_Free.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["Free"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/01_icon_Free.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1678), _c1)
except Exception:
    pass
layout["Free"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/02_icon_Los_Angeles.png
try:
    _c2 = get_crop(2, 456, 117)
    canvas.paste(_c2, (492, 2651), _c2)
except Exception:
    pass
layout["Los_Angeles"] = [492, 2651, 948, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/03_icon_NDIE.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1282), _c3)
except Exception:
    pass
layout["NDIE"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/04_icon_Afliccion_Perdida_y.png
try:
    _c4 = get_crop(4, 144, 123)
    canvas.paste(_c4, (1140, 1951), _c4)
except Exception:
    pass
layout["Afliccion,_Perdida_y"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/05_icon_REoPUNKSFRE.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 886), _c5)
except Exception:
    pass
layout["REoPUNKSFRE"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/06_icon_8Os_vs_Indie.png
try:
    _c6 = get_crop(6, 1344, 396)
    canvas.paste(_c6, (48, 1678), _c6)
except Exception:
    pass
layout["8Os_vs_Indie"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/07_icon_Indie_Sleaze_4_26_Club_Decades.png
try:
    _c7 = get_crop(7, 1344, 396)
    canvas.paste(_c7, (48, 1282), _c7)
except Exception:
    pass
layout["Indie_Sleaze_4_26_@_Club_"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/08_icon_Search_events.png
try:
    _c8 = get_crop(8, 1179, 144)
    canvas.paste(_c8, (195, 93), _c8)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/09_icon_Thu_Mar_28.png
try:
    _c9 = get_crop(9, 1344, 396)
    canvas.paste(_c9, (48, 490), _c9)
except Exception:
    pass
layout["Thu,_Mar_28"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/10_icon_Afliccion_Perdida_y.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 2347), _c10)
except Exception:
    pass
layout["Afliccion,_Perdida_y"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 123)
    canvas.paste(_c11, (1284, 1951), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/12_icon_Favorite_button.png
try:
    _c12 = get_crop(12, 144, 139)
    canvas.paste(_c12, (1140, 1539), _c12)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 123)
    canvas.paste(_c13, (1284, 1159), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/14_icon_Favorite_button.png
try:
    _c14 = get_crop(14, 144, 123)
    canvas.paste(_c14, (1140, 763), _c14)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 123)
    canvas.paste(_c15, (1284, 2347), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/16_icon_9.44.png
try:
    _c16 = get_crop(16, 53, 58)
    canvas.paste(_c16, (183, 3), _c16)
except Exception:
    pass
layout["9.44"] = [183, 3, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/17_icon_Apartn.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (288, 2804), _c17)
except Exception:
    pass
layout["Apartn"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/18_icon_Favorite_button.png
try:
    _c18 = get_crop(18, 144, 123)
    canvas.paste(_c18, (1140, 1159), _c18)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 56, 57)
    canvas.paste(_c19, (247, 4), _c19)
except Exception:
    pass
layout["icon_19"] = [247, 4, 303, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/20_icon_Overflow_menu_button.png
try:
    _c20 = get_crop(20, 144, 139)
    canvas.paste(_c20, (1284, 1539), _c20)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/21_icon_Overflow_menu_button.png
try:
    _c21 = get_crop(21, 144, 123)
    canvas.paste(_c21, (1284, 763), _c21)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/22_icon_8_20599_creator_followers.png
try:
    _c22 = get_crop(22, 1344, 396)
    canvas.paste(_c22, (48, 886), _c22)
except Exception:
    pass
layout["8_20599_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/23_icon_Home.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (0, 2804), _c23)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 49, 54)
    canvas.paste(_c24, (1320, 6), _c24)
except Exception:
    pass
layout["icon_24"] = [1320, 6, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 83, 61)
    canvas.paste(_c25, (1211, 3), _c25)
except Exception:
    pass
layout["icon_25"] = [1211, 3, 1294, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/26_icon_Traumatic_Loss_Conference.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 490), _c26)
except Exception:
    pass
layout["Traumatic_Loss_Conference"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/27_icon_9.44.png
try:
    _c27 = get_crop(27, 101, 105)
    canvas.paste(_c27, (41, 118), _c27)
except Exception:
    pass
layout["9.44"] = [41, 118, 142, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/28_icon_Indie_Sleaze_4_26_Club_Decades.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 1282), _c28)
except Exception:
    pass
layout["Indie_Sleaze_4_26_@_Club_"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/29_icon_Grief_Loss_Resiliency.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 2074), _c29)
except Exception:
    pass
layout["Grief;_Loss,_Resiliency"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 48, 56)
    canvas.paste(_c30, (383, 7), _c30)
except Exception:
    pass
layout["icon_30"] = [383, 7, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 62, 61)
    canvas.paste(_c31, (311, 4), _c31)
except Exception:
    pass
layout["icon_31"] = [311, 4, 373, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/32_icon_Tickets.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (864, 2804), _c32)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/33_icon_8_90_creator_followers.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 490), _c33)
except Exception:
    pass
layout["8_90_creator_followers"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/34_icon_Blue_Mondays_vs_Rock_it_Fridays_Ziings.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 1678), _c34)
except Exception:
    pass
layout["Blue_Mondays_vs_Rock_it!_"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/35_icon_The_Virgil.png
try:
    _c35 = get_crop(35, 157, 52)
    canvas.paste(_c35, (391, 1130), _c35)
except Exception:
    pass
layout["The_Virgil"] = [391, 1130, 548, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/36_icon_icon_36.png
try:
    _c36 = get_crop(36, 42, 58)
    canvas.paste(_c36, (1272, 4), _c36)
except Exception:
    pass
layout["icon_36"] = [1272, 4, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/37_icon_8_430_creator_followers.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 1678), _c37)
except Exception:
    pass
layout["8_430_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/38_text_9.44.png
try:
    _c38 = get_crop(38, 94, 43)
    canvas.paste(_c38, (20, 15), _c38)
except Exception:
    pass
layout["9.44"] = [20, 15, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/39_text_More_events_you_II_love.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 490), _c39)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/40_text_SAURUA.png
try:
    _c40 = get_crop(40, 57, 12)
    canvas.paste(_c40, (170, 2535), _c40)
except Exception:
    pass
layout["SAURUA;"] = [170, 2535, 227, 2547]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/41_text_Sat_Apr_13_-.png
try:
    _c41 = get_crop(41, 196, 43)
    canvas.paste(_c41, (392, 2526), _c41)
except Exception:
    pass
layout["Sat,_Apr_13_-"] = [392, 2526, 588, 2569]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/42_text_1O_00_PM_PDT.png
try:
    _c42 = get_crop(42, 1344, 346)
    canvas.paste(_c42, (48, 2470), _c42)
except Exception:
    pass
layout["1O:00_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/43_text_Lovers.png
try:
    _c43 = get_crop(43, 168, 41)
    canvas.paste(_c43, (112, 2598), _c43)
except Exception:
    pass
layout["Lovers_&"] = [112, 2598, 280, 2639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/44_text_Rooftop_Party.png
try:
    _c44 = get_crop(44, 88, 16)
    canvas.paste(_c44, (153, 2672), _c44)
except Exception:
    pass
layout["Rooftop_Party"] = [153, 2672, 241, 2688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/45_text_2027_creator_followers.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (576, 2804), _c45)
except Exception:
    pass
layout["2027_creator_followers"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/cee80d2c889b45cd86e30248baf590b8/step_01_2024_3_20_17_43_cee80d2c889b45cd86e30248baf590b8-3/46_clickable_More.png
try:
    _c46 = get_crop(46, 288, 156)
    canvas.paste(_c46, (1152, 2804), _c46)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
