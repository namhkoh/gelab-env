# page_id: page_eventbrite_c7c81d1bf6744774b99294e9f124dda3_01
# screenshot: 2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3.png
# step_index: 1/10
# task: Open Eventbrite. Search for "Fitness". Select the events in the location "Chicago". What is the price of the first event in listing?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for a 1440x2960 mobile canvas using provided
# canvas (PIL Image) and draw (ImageDraw) objects and fonts.
w, h = canvas.size

# Colors
bg_color = (250, 250, 252)         # very light off-white page background
status_color = (189, 189, 189)     # light grey status bar
search_border = (221, 217, 235)    # soft pale purple/grey border for search pill
search_fill = (255, 255, 255)      # white for search pill
card_shadow = (235, 235, 238)      # subtle shadow under cards
card_fill = (255, 255, 255)        # card background (white)
card_outline = (242, 242, 245)     # very light outline for cards
divider_color = (236, 236, 239)    # thin dividers between sections
toolbar_bg = (255, 255, 255)       # toolbar / content background (white)
nav_top_border = (226, 226, 229)   # top border for bottom nav

# Fill overall background
draw.rectangle((0, 0, w, h), fill=bg_color)

# Status bar area (~50-72px tall). Keep slightly taller for safe spacing.
status_h = 72
draw.rectangle((0, 0, w, status_h), fill=status_color)

# Toolbar / search area
# Use detected search area: pos=(195,93) size=(1179,144)
search_x, search_y, search_w, search_h = 195, 93, 1179, 144
search_radius = int(search_h / 2)
search_bbox = (search_x, search_y, search_x + search_w, search_y + search_h)
# draw subtle drop shadow for the search pill
shadow_offset = 4
draw.rounded_rectangle(
    (search_bbox[0] + shadow_offset, search_bbox[1] + shadow_offset,
     search_bbox[2] + shadow_offset, search_bbox[3] + shadow_offset),
    radius=search_radius, fill=card_shadow
)
# draw search pill body (no icons/text)
draw.rounded_rectangle(search_bbox, radius=search_radius, fill=search_fill, outline=search_border, width=2)

# Thin divider line below toolbar
toolbar_div_y = search_y + search_h + 28
draw.line((48, toolbar_div_y, w - 48, toolbar_div_y), fill=divider_color, width=1)

# Event "cards" background areas (rounded rects with subtle shadows)
# Use detected card positions / sizes (x=48, widths=1344)
card_x = 48
card_w = 1344
card_positions = [
    (48, 490, 1344, 396),
    (48, 886, 1344, 396),
    (48, 1282, 1344, 396),
    (48, 1678, 1344, 396),
    (48, 2074, 1344, 396),
    (48, 2470, 1344, 346)  # last card slightly shorter
]
# Convert these to (x,y,width,height) already; we'll draw at x positions from list
for entry in card_positions:
    cx, cy, cwidth, cheight = entry
    x0 = cx
    y0 = cy
    x1 = cx + cwidth
    y1 = cy + cheight
    radius = 14

    # shadow
    shadow_off = 6
    draw.rounded_rectangle((x0 + shadow_off, y0 + shadow_off, x1 + shadow_off, y1 + shadow_off),
                           radius=radius, fill=card_shadow)

    # main card
    draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=card_fill, outline=card_outline, width=1)

    # subtle horizontal separators inside the card area to suggest content rows (but avoid drawing text/icons)
    # draw very light horizontal guide lines (not text)
    sep_y1 = y0 + int(cheight * 0.55)
    draw.line((x0 + 12, sep_y1, x1 - 12, sep_y1), fill=(249,249,251), width=1)

# Additional section separators between major groups (thin lines)
section_dividers = [
    450,   # below hero heading
    818,
    1214,
    1610,
    2006,
    2402
]
for sy in section_dividers:
    draw.line((48, sy, w - 48, sy), fill=divider_color, width=1)

# Draw a subtle floating location pill shadow area near the bottom center where the location control will be placed
# Detected floating location is around y ~ 2780 and width ~ 360; we draw only a soft shadow (no pill)
pill_shadow_w = 420
pill_shadow_h = 84
pill_center_x = w // 2
pill_center_y = 2796
ps_x0 = pill_center_x - pill_shadow_w // 2
ps_y0 = pill_center_y - pill_shadow_h // 2
ps_x1 = pill_center_x + pill_shadow_w // 2
ps_y1 = pill_center_y + pill_shadow_h // 2
draw.rounded_rectangle((ps_x0 + 6, ps_y0 + 6, ps_x1 + 6, ps_y1 + 6), radius=44, fill=(240, 240, 243))

# Bottom navigation bar area (draw background and top border)
nav_y0 = 2804
nav_h = 156
draw.rectangle((0, nav_y0, w, nav_y0 + nav_h), fill=toolbar_bg)
draw.line((48, nav_y0, w - 48, nav_y0), fill=nav_top_border, width=1)

# Small subtle shadow above page bottom area for depth
draw.line((0, nav_y0 + 1, w, nav_y0 + 1), fill=(247, 247, 249), width=1)

# Optional small left margin vertical guide (very subtle) to frame content (not text)
draw.line((48, status_h + 24, 48, nav_y0 - 24), fill=(248, 248, 249), width=1)
draw.line((w - 48, status_h + 24, w - 48, nav_y0 - 24), fill=(248, 248, 249), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/00_icon_New_York.png
try:
    _c0 = get_crop(0, 405, 117)
    canvas.paste(_c0, (518, 2651), _c0)
except Exception:
    pass
layout["New_York"] = [518, 2651, 923, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/01_icon_City.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 886), _c1)
except Exception:
    pass
layout["City,"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/02_icon_VOSCHINO.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1678), _c2)
except Exception:
    pass
layout["VOSCHINO"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/03_icon_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/04_icon_Free.png
try:
    _c4 = get_crop(4, 1344, 396)
    canvas.paste(_c4, (48, 2074), _c4)
except Exception:
    pass
layout["Free"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/05_icon_Free.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1282), _c5)
except Exception:
    pass
layout["Free"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/06_icon_New_York.png
try:
    _c6 = get_crop(6, 144, 139)
    canvas.paste(_c6, (1140, 747), _c6)
except Exception:
    pass
layout["New_York"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/07_icon_City.png
try:
    _c7 = get_crop(7, 144, 139)
    canvas.paste(_c7, (1140, 1935), _c7)
except Exception:
    pass
layout["City"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/08_icon_Conference_Connections.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 2331), _c8)
except Exception:
    pass
layout["Conference_Connections"] = [1140, 2331, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/09_icon_New_York.png
try:
    _c9 = get_crop(9, 144, 123)
    canvas.paste(_c9, (1140, 1159), _c9)
except Exception:
    pass
layout["New_York"] = [1140, 1159, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/10_icon_Free.png
try:
    _c10 = get_crop(10, 1344, 396)
    canvas.paste(_c10, (48, 490), _c10)
except Exception:
    pass
layout["Free"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1284, 2331), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2331, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/12_icon_New_York.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1284, 1159), _c12)
except Exception:
    pass
layout["New_York"] = [1284, 1159, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/13_icon_New_York.png
try:
    _c13 = get_crop(13, 144, 139)
    canvas.paste(_c13, (1284, 747), _c13)
except Exception:
    pass
layout["New_York"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1284, 1935), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 123)
    canvas.paste(_c15, (1284, 1555), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1555, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/16_icon_Primary.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (288, 2804), _c16)
except Exception:
    pass
layout["Primary"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/17_icon_Favorite_button.png
try:
    _c17 = get_crop(17, 144, 123)
    canvas.paste(_c17, (1140, 1555), _c17)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1555, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/18_icon_7.09.png
try:
    _c18 = get_crop(18, 103, 97)
    canvas.paste(_c18, (41, 123), _c18)
except Exception:
    pass
layout["7.09"] = [41, 123, 144, 220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 60, 58)
    canvas.paste(_c19, (312, 3), _c19)
except Exception:
    pass
layout["icon_19"] = [312, 3, 372, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/20_icon_free_Grief_and_Loss_Healing_Circle.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 490), _c20)
except Exception:
    pass
layout["(free)_Grief_and_Loss_Hea"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/21_icon_7.09.png
try:
    _c21 = get_crop(21, 57, 60)
    canvas.paste(_c21, (182, 2), _c21)
except Exception:
    pass
layout["7.09"] = [182, 2, 239, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 50, 59)
    canvas.paste(_c22, (248, 2), _c22)
except Exception:
    pass
layout["icon_22"] = [248, 2, 298, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/23_icon_139_creator_followers.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 1282), _c23)
except Exception:
    pass
layout["139_creator_followers"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 47, 52)
    canvas.paste(_c24, (1321, 7), _c24)
except Exception:
    pass
layout["icon_24"] = [1321, 7, 1368, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/25_icon_Medical_Hair_Loss_Therapy_Training.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 886), _c25)
except Exception:
    pass
layout["Medical_Hair_Loss_Therapy"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/26_icon_Good_Afternoon_New_York.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 1678), _c26)
except Exception:
    pass
layout["Good_Afternoon_New_York"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/27_icon_and_Primary_Venture_Parti.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (0, 2804), _c27)
except Exception:
    pass
layout["and_Primary_Venture_Parti"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 55, 56)
    canvas.paste(_c28, (1213, 6), _c28)
except Exception:
    pass
layout["icon_28"] = [1213, 6, 1268, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 41, 53)
    canvas.paste(_c29, (1272, 7), _c29)
except Exception:
    pass
layout["icon_29"] = [1272, 7, 1313, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/30_icon_7.09.png
try:
    _c30 = get_crop(30, 58, 60)
    canvas.paste(_c30, (115, 2), _c30)
except Exception:
    pass
layout["7.09"] = [115, 2, 173, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/31_icon_Free.png
try:
    _c31 = get_crop(31, 130, 74)
    canvas.paste(_c31, (244, 560), _c31)
except Exception:
    pass
layout["Free"] = [244, 560, 374, 634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/32_icon_icon_32.png
try:
    _c32 = get_crop(32, 44, 56)
    canvas.paste(_c32, (385, 6), _c32)
except Exception:
    pass
layout["icon_32"] = [385, 6, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/33_icon_8_1646_creator_followers.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 886), _c33)
except Exception:
    pass
layout["8_1646_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/34_icon_free_Grief_and_Loss_Healing_Circle.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 490), _c34)
except Exception:
    pass
layout["(free)_Grief_and_Loss_Hea"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/35_icon_8_15225_creator_followers.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 2074), _c35)
except Exception:
    pass
layout["8_15225_creator_followers"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/36_icon_City.png
try:
    _c36 = get_crop(36, 1344, 346)
    canvas.paste(_c36, (48, 2470), _c36)
except Exception:
    pass
layout["City"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/37_icon_Tickets.png
try:
    _c37 = get_crop(37, 288, 156)
    canvas.paste(_c37, (864, 2804), _c37)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/38_icon_10_creator_followers.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 1678), _c38)
except Exception:
    pass
layout["10_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/39_icon_15.png
try:
    _c39 = get_crop(39, 1344, 346)
    canvas.paste(_c39, (48, 2470), _c39)
except Exception:
    pass
layout["15"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/40_text_7.09.png
try:
    _c40 = get_crop(40, 91, 45)
    canvas.paste(_c40, (20, 15), _c40)
except Exception:
    pass
layout["7.09"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/41_text_More_events_you_II_love.png
try:
    _c41 = get_crop(41, 1344, 396)
    canvas.paste(_c41, (48, 490), _c41)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/42_text_IN_PERSON_EVENT.png
try:
    _c42 = get_crop(42, 133, 16)
    canvas.paste(_c42, (133, 2561), _c42)
except Exception:
    pass
layout["IN_PERSON_EVENT"] = [133, 2561, 266, 2577]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/43_text_WEDNESDAY_May.png
try:
    _c43 = get_crop(43, 118, 16)
    canvas.paste(_c43, (135, 2584), _c43)
except Exception:
    pass
layout["WEDNESDAY,_May"] = [135, 2584, 253, 2600]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/44_text_2024.png
try:
    _c44 = get_crop(44, 32, 16)
    canvas.paste(_c44, (267, 2584), _c44)
except Exception:
    pass
layout["2024"] = [267, 2584, 299, 2600]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/45_text_10.0_AM_ET.png
try:
    _c45 = get_crop(45, 76, 16)
    canvas.paste(_c45, (153, 2600), _c45)
except Exception:
    pass
layout["10.0_AM_ET"] = [153, 2600, 229, 2616]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/46_text_Wip_CONFERENCE_CON.png
try:
    _c46 = get_crop(46, 215, 20)
    canvas.paste(_c46, (137, 2635), _c46)
except Exception:
    pass
layout["Wip_CONFERENCE_CON"] = [137, 2635, 352, 2655]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/47_text_MEN.png
try:
    _c47 = get_crop(47, 44, 18)
    canvas.paste(_c47, (49, 2658), _c47)
except Exception:
    pass
layout["MEN"] = [49, 2658, 93, 2676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/48_text_Product_Breakfast.png
try:
    _c48 = get_crop(48, 152, 16)
    canvas.paste(_c48, (135, 2660), _c48)
except Exception:
    pass
layout["Product_Breakfast"] = [135, 2660, 287, 2676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/49_text_Presented_by.png
try:
    _c49 = get_crop(49, 115, 18)
    canvas.paste(_c49, (135, 2702), _c49)
except Exception:
    pass
layout["Presented_by:"] = [135, 2702, 250, 2720]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/50_text_Women_In_Product_New.png
try:
    _c50 = get_crop(50, 201, 16)
    canvas.paste(_c50, (135, 2723), _c50)
except Exception:
    pass
layout["Women_In_Product_New"] = [135, 2723, 336, 2739]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/51_text_and_Primary_Venture_Parti.png
try:
    _c51 = get_crop(51, 217, 21)
    canvas.paste(_c51, (135, 2741), _c51)
except Exception:
    pass
layout["and_Primary_Venture_Parti"] = [135, 2741, 352, 2762]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/52_clickable_Favorites.png
try:
    _c52 = get_crop(52, 288, 156)
    canvas.paste(_c52, (576, 2804), _c52)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_01_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-3/53_clickable_More.png
try:
    _c53 = get_crop(53, 288, 156)
    canvas.paste(_c53, (1152, 2804), _c53)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
