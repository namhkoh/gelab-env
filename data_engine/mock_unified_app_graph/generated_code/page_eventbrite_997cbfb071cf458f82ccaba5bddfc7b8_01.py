# page_id: page_eventbrite_997cbfb071cf458f82ccaba5bddfc7b8_01
# screenshot: 2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3.png
# step_index: 1/15
# task: Open Eventbrite. Search Online free events. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the Event list screen.
# Uses provided: canvas (PIL Image 1440x2960), draw (ImageDraw), font_* variables.

# Colors
bg_color = "#FBF9FC"            # very light off-white background
status_bar_color = "#D9D9D9"    # light gray status bar
search_bar_fill = "#FFFFFF"     # white search field
search_bar_border = "#E6E2E8"   # subtle border for search field
header_bg = "#FFFFFF"           # header background (below status bar)
card_shadow = "#F3F1F6"         # subtle card shadow
card_fill = "#FFFFFF"           # card background
card_border = "#F0EDF4"         # faint card border
thumb_bg = "#EFEEF3"            # thumbnail placeholder background
divider_color = "#ECE8EF"       # separators between sections
bottom_nav_bg = "#FFFFFF"       # bottom navigation background
pill_shadow = "#F0EEF6"         # floating pill shadow

W, H = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (W, H)], fill=bg_color)

# Status bar area (approx 50-66px high)
status_h = 66
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# Header background area below status bar
header_top = status_h
header_bottom = 220
draw.rectangle([(0, header_top), (W, header_bottom)], fill=header_bg)

# Search field background (rounded rectangle) - leave content inside to be pasted later
search_left = 48
search_top = 84
search_right = W - 48
search_bottom = 228
search_radius = 72
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=search_radius,
    fill=search_bar_fill,
    outline=search_bar_border,
    width=2
)

# Subtle divider below header
draw.line([(48, header_bottom + 6), (W - 48, header_bottom + 6)], fill=divider_color, width=1)

# Card positions (derived from detected elements)
card_positions = [
    (48, 490, 1392, 490 + 396),
    (48, 886, 1392, 886 + 396),
    (48, 1282, 1392, 1282 + 396),
    (48, 1678, 1392, 1678 + 396),
    (48, 2074, 1392, 2074 + 396),
    (48, 2470, 1392, 2470 + 346)
]

card_radius = 14
shadow_offset = 8

for (x1, y1, x2, y2) in card_positions:
    # Draw shadow slightly below the card
    shadow_box = (x1, y1 + shadow_offset, x2, y2 + shadow_offset)
    draw.rounded_rectangle([shadow_box[0], shadow_box[1], shadow_box[2], shadow_box[3]],
                           radius=card_radius + 1, fill=card_shadow)
    # Draw card background
    draw.rounded_rectangle([x1, y1, x2, y2], radius=card_radius, fill=card_fill, outline=card_border, width=1)
    # Thumbnail placeholder area on left (will be covered by pasted thumbnails)
    thumb_margin = 24
    thumb_size = y2 - y1 - 2 * thumb_margin
    thumb_x1 = x1 + thumb_margin
    thumb_y1 = y1 + thumb_margin
    thumb_x2 = thumb_x1 + thumb_size
    thumb_y2 = thumb_y1 + thumb_size
    draw.rectangle([(thumb_x1, thumb_y1), (thumb_x2, thumb_y2)], fill=thumb_bg, outline=card_border)

    # Right-side divider line (vertical subtle guide near heart/overflow icons)
    guide_x = x2 - 180
    draw.line([(guide_x, y1 + 20), (guide_x, y2 - 20)], fill=divider_color, width=1)

    # Horizontal separator line under card (soft)
    sep_y = y2 + 12
    draw.line([(x1 + 8, sep_y), (x2 - 8, sep_y)], fill=divider_color, width=1)

# Large section title area (no text drawn) - create a subtle area for the "More events you'll love" heading
title_block_top = 360
title_block_bottom = 480
draw.rectangle([(48, title_block_top), (W - 48, title_block_bottom)], fill=bg_color)

# Floating location pill shadow area (behind pasted 'Los Angeles' pill)
pill_w = 456
pill_h = 117
pill_x1 = (W - pill_w) // 2
pill_y1 = 2651
pill_x2 = pill_x1 + pill_w
pill_y2 = pill_y1 + pill_h
# draw a very soft rounded shadow
draw.rounded_rectangle([(pill_x1 - 6, pill_y1 + 8), (pill_x2 + 6, pill_y2 + 8)], radius=36, fill=pill_shadow)

# Bottom navigation bar background and top divider
nav_top = 2804
draw.rectangle([(0, nav_top), (W, H)], fill=bottom_nav_bg)
draw.line([(0, nav_top), (W, nav_top)], fill=divider_color, width=1)

# Draw small rounded card for the floating location selector anchor (background only; content will be pasted)
anchor_w = 520
anchor_h = 92
anchor_x1 = (W - anchor_w) // 2
anchor_y1 = 2530
anchor_x2 = anchor_x1 + anchor_w
anchor_y2 = anchor_y1 + anchor_h
draw.rounded_rectangle([(anchor_x1, anchor_y1), (anchor_x2, anchor_y2)], radius=46, fill="#FFFFFF", outline=card_border, width=1)
# anchor shadow
draw.rounded_rectangle([(anchor_x1, anchor_y1 + 6), (anchor_x2, anchor_y2 + 6)], radius=46, fill=pill_shadow)

# Final subtle top padding divider beneath status/search area
draw.line([(24, header_bottom + 20), (W - 24, header_bottom + 20)], fill="#F5F3F7", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/00_icon_Free.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2074), _c0)
except Exception:
    pass
layout["Free"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/01_icon_Search_events.png
try:
    _c1 = get_crop(1, 1179, 144)
    canvas.paste(_c1, (195, 93), _c1)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/02_icon_Favorite_button.png
try:
    _c2 = get_crop(2, 144, 123)
    canvas.paste(_c2, (1140, 1951), _c2)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/03_icon_Afliccion_Perdida_y.png
try:
    _c3 = get_crop(3, 144, 123)
    canvas.paste(_c3, (1140, 2347), _c3)
except Exception:
    pass
layout["Afliccion,_Perdida_y"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 139)
    canvas.paste(_c4, (1140, 1539), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/05_icon_NDIE.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1282), _c5)
except Exception:
    pass
layout["NDIE"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 123)
    canvas.paste(_c6, (1284, 1951), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 123)
    canvas.paste(_c7, (1284, 1159), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/08_icon_Los_Angeles.png
try:
    _c8 = get_crop(8, 456, 117)
    canvas.paste(_c8, (492, 2651), _c8)
except Exception:
    pass
layout["Los_Angeles"] = [492, 2651, 948, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 123)
    canvas.paste(_c9, (1284, 2347), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/10_icon_Favorite_button.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 763), _c10)
except Exception:
    pass
layout["Favorite_button"] = [1140, 763, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/11_icon_Sylmai.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (288, 2804), _c11)
except Exception:
    pass
layout["Sylmai"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 139)
    canvas.paste(_c12, (1284, 1539), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/13_icon_Free.png
try:
    _c13 = get_crop(13, 1344, 346)
    canvas.paste(_c13, (48, 2470), _c13)
except Exception:
    pass
layout["Free"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/14_icon_Apartment_503_Nightclub.png
try:
    _c14 = get_crop(14, 1344, 396)
    canvas.paste(_c14, (48, 1678), _c14)
except Exception:
    pass
layout["Apartment_503_Nightclub"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/15_icon_Indie_Sleaze_4_26_Club_Decades.png
try:
    _c15 = get_crop(15, 1344, 396)
    canvas.paste(_c15, (48, 1282), _c15)
except Exception:
    pass
layout["Indie_Sleaze_4_26_@_Club_"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/16_icon_Favorite_button.png
try:
    _c16 = get_crop(16, 144, 123)
    canvas.paste(_c16, (1140, 1159), _c16)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/17_icon_REoPUNKSFRE.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 886), _c17)
except Exception:
    pass
layout["REoPUNKSFRE"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/18_icon_Overflow_menu_button.png
try:
    _c18 = get_crop(18, 144, 123)
    canvas.paste(_c18, (1284, 763), _c18)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 763, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 56, 57)
    canvas.paste(_c19, (247, 4), _c19)
except Exception:
    pass
layout["icon_19"] = [247, 4, 303, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/20_icon_8_20599_creator_followers.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 886), _c20)
except Exception:
    pass
layout["8_20599_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/21_icon_Rooftop_Party.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1678), _c21)
except Exception:
    pass
layout["Rooftop_Party"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/22_icon_Home.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (0, 2804), _c22)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/23_icon_9.15.png
try:
    _c23 = get_crop(23, 53, 58)
    canvas.paste(_c23, (183, 3), _c23)
except Exception:
    pass
layout["9.15"] = [183, 3, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/24_icon_18_8Os_vs_Indie_2_rooms.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 2074), _c24)
except Exception:
    pass
layout["18+_:_8Os_vs_Indie_!_2_ro"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/25_icon_Traumatic_Loss_Conference.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 490), _c25)
except Exception:
    pass
layout["Traumatic_Loss_Conference"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 47, 54)
    canvas.paste(_c26, (1321, 6), _c26)
except Exception:
    pass
layout["icon_26"] = [1321, 6, 1368, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 89, 59)
    canvas.paste(_c27, (1211, 4), _c27)
except Exception:
    pass
layout["icon_27"] = [1211, 4, 1300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/28_icon_Thu_Mar_28.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 490), _c28)
except Exception:
    pass
layout["Thu,_Mar_28"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/29_icon_9.15.png
try:
    _c29 = get_crop(29, 97, 102)
    canvas.paste(_c29, (43, 119), _c29)
except Exception:
    pass
layout["9.15"] = [43, 119, 140, 221]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/30_icon_5.30_PM_PDT.png
try:
    _c30 = get_crop(30, 1344, 346)
    canvas.paste(_c30, (48, 2470), _c30)
except Exception:
    pass
layout["5.30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 60, 61)
    canvas.paste(_c31, (312, 4), _c31)
except Exception:
    pass
layout["icon_31"] = [312, 4, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/32_icon_icon_32.png
try:
    _c32 = get_crop(32, 48, 56)
    canvas.paste(_c32, (383, 7), _c32)
except Exception:
    pass
layout["icon_32"] = [383, 7, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/33_icon_Indie_Sleaze_4_26_Club_Decades.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 1282), _c33)
except Exception:
    pass
layout["Indie_Sleaze_4_26_@_Club_"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/34_icon_Blue_Mondays_vs_Rock_it_Fridays_Ziings.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 2074), _c34)
except Exception:
    pass
layout["Blue_Mondays_vs_Rock_it!_"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/35_icon_Tickets.png
try:
    _c35 = get_crop(35, 288, 156)
    canvas.paste(_c35, (864, 2804), _c35)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/36_icon_The_Virgil.png
try:
    _c36 = get_crop(36, 157, 51)
    canvas.paste(_c36, (391, 1131), _c36)
except Exception:
    pass
layout["The_Virgil"] = [391, 1131, 548, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/37_icon_19creator_followers.png
try:
    _c37 = get_crop(37, 288, 156)
    canvas.paste(_c37, (576, 2804), _c37)
except Exception:
    pass
layout["19creator_followers"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/38_icon_8_90_creator_followers.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 490), _c38)
except Exception:
    pass
layout["8_90_creator_followers"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/39_icon_icon_39.png
try:
    _c39 = get_crop(39, 42, 58)
    canvas.paste(_c39, (1272, 4), _c39)
except Exception:
    pass
layout["icon_39"] = [1272, 4, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/40_icon_VIBE_Lovers_Friends_Rooftop_Party_21_in.png
try:
    _c40 = get_crop(40, 1344, 396)
    canvas.paste(_c40, (48, 1678), _c40)
except Exception:
    pass
layout["VIBE:_Lovers_&_Friends'_R"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/41_text_9.15.png
try:
    _c41 = get_crop(41, 94, 41)
    canvas.paste(_c41, (20, 17), _c41)
except Exception:
    pass
layout["9.15"] = [20, 17, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/42_text_More_events_you_II_love.png
try:
    _c42 = get_crop(42, 1344, 396)
    canvas.paste(_c42, (48, 490), _c42)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/43_text_Mon.png
try:
    _c43 = get_crop(43, 92, 43)
    canvas.paste(_c43, (393, 2525), _c43)
except Exception:
    pass
layout["Mon,"] = [393, 2525, 485, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/44_text_13.png
try:
    _c44 = get_crop(44, 54, 36)
    canvas.paste(_c44, (561, 2526), _c44)
except Exception:
    pass
layout["13"] = [561, 2526, 615, 2562]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/45_text_5.30_PM_PDT.png
try:
    _c45 = get_crop(45, 1344, 346)
    canvas.paste(_c45, (48, 2470), _c45)
except Exception:
    pass
layout["5.30_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/46_text_19creator_followers.png
try:
    _c46 = get_crop(46, 288, 156)
    canvas.paste(_c46, (576, 2804), _c46)
except Exception:
    pass
layout["19creator_followers"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/997cbfb071cf458f82ccaba5bddfc7b8/step_01_2024_3_20_17_14_997cbfb071cf458f82ccaba5bddfc7b8-3/47_clickable_More.png
try:
    _c47 = get_crop(47, 288, 156)
    canvas.paste(_c47, (1152, 2804), _c47)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
