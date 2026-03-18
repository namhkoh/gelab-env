# page_id: page_eventbrite_e794243d416840069b0e5f15aefc4a34_01
# screenshot: 2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3.png
# step_index: 1/7
# task: Open Eventbrite. Open "Business Seminar". Select the first event. Note the contact details of the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the provided canvas and draw objects.
# Uses available variables: canvas (1440x2960 PIL Image) and draw (ImageDraw).
# Available fonts: font_sm, font_md, font_lg, font_xl

# Colors
bg_color = (250, 250, 253)        # very light off-white background
status_bar_color = (210, 210, 210)  # light gray status bar
toolbar_color = (255, 255, 255)     # white toolbar area
card_bg = (255, 255, 255)           # card white
card_border = (235, 235, 240)       # subtle card border / shadow
thumb_bg = (245, 246, 250)          # thumbnail placeholder background
divider_color = (240, 240, 245)     # thin divider lines
bottom_nav_bg = (255, 255, 255)     # bottom nav background
shadow_color = (230, 230, 235)      # light shadow lines

W, H = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (W, H)], fill=bg_color)

# Status bar (top ~60px)
status_h = 60
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# Thin inner highlight at bottom of status bar (subtle)
draw.line([(0, status_h), (W, status_h)], fill=shadow_color, width=1)

# Top toolbar area (contains search bar region but we don't draw the search control)
toolbar_top = status_h
toolbar_bottom = 160
draw.rectangle([(0, toolbar_top), (W, toolbar_bottom)], fill=toolbar_color)

# Subtle shadow under toolbar
draw.line([(0, toolbar_bottom), (W, toolbar_bottom)], fill=shadow_color, width=1)
draw.line([(0, toolbar_bottom+1), (W, toolbar_bottom+1)], fill=(245,245,246), width=1)

# Define event card positions (based on detected crop positions)
card_positions = [
    490,  # first visible card
    886,
    1282,
    1678,
    2074,
    2470  # additional lower card area
]

card_x = 48
card_width = 1344
card_height = 396
card_radius = 14

# Draw card containers and thumbnail placeholders
for y in card_positions:
    x1 = card_x
    y1 = y
    x2 = x1 + card_width
    y2 = y1 + card_height

    # Draw a subtle drop shadow / border behind the card
    shadow_rect = [x1+2, y1+6, x2+2, y2+6]
    draw.rounded_rectangle(shadow_rect, radius=card_radius, fill=card_border)

    # Card background (rounded white rectangle)
    draw.rounded_rectangle([x1, y1, x2, y2], radius=card_radius, fill=card_bg)

    # Thumbnail placeholder on the left (do not draw any thumbnails or content)
    thumb_x1 = x1
    thumb_y1 = y1 + 16
    thumb_x2 = thumb_x1 + 180
    thumb_y2 = thumb_y1 + 180
    draw.rounded_rectangle([thumb_x1, thumb_y1, thumb_x2, thumb_y2], radius=10, fill=thumb_bg)

    # Vertical divider line to separate thumbnail area from content area (subtle)
    div_x = thumb_x2 + 18
    draw.line([(div_x, y1 + 12), (div_x, y2 - 12)], fill=divider_color, width=1)

    # Thin divider under each card
    sep_y = y2 + 22
    draw.line([(x1 + 0, sep_y), (x2, sep_y)], fill=divider_color, width=1)

# Floating content area hint (do not draw text) - subtle rounded pill shape near lower content area
pill_w = 520
pill_h = 84
pill_x = (W - pill_w) // 2
pill_y = 2380
draw.rounded_rectangle([pill_x, pill_y, pill_x + pill_w, pill_y + pill_h], radius=42, fill=(255,255,255))
# pill shadow
draw.ellipse([(pill_x + pill_w - 60, pill_y + pill_h - 8), (pill_x + pill_w - 24, pill_y + pill_h + 20)], fill=(250,250,252))

# Bottom navigation bar background
bottom_nav_top = 2804
draw.rectangle([(0, bottom_nav_top), (W, H)], fill=bottom_nav_bg)
# Top border for nav
draw.line([(0, bottom_nav_top), (W, bottom_nav_top)], fill=shadow_color, width=1)
draw.line([(0, bottom_nav_top+1), (W, bottom_nav_top+1)], fill=(245,245,246), width=1)

# Add subtle separators for typical nav item slots (do not draw icons)
nav_slot_w = W // 5
for i in range(1, 5):
    nx = i * nav_slot_w
    # very faint vertical hints (not visible as UI elements, just structure)
    draw.line([(nx, bottom_nav_top + 12), (nx, H - 12)], fill=(255,255,255,0))

# Final subtle vignette along left/right edges (very light)
edge_shade = (248, 248, 250)
edge_width = 14
draw.rectangle([(0, 0), (edge_width, H)], fill=edge_shade)
draw.rectangle([(W - edge_width, 0), (W, H)], fill=edge_shade)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/00_icon_Online.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 490), _c0)
except Exception:
    pass
layout["Online"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/01_icon_Online.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 2074), _c1)
except Exception:
    pass
layout["Online"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/02_icon_Online.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1678), _c2)
except Exception:
    pass
layout["Online"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/03_icon_Search_events.png
try:
    _c3 = get_crop(3, 1179, 144)
    canvas.paste(_c3, (195, 93), _c3)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 125)
    canvas.paste(_c4, (1140, 2345), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1140, 2345, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 125)
    canvas.paste(_c5, (1140, 1949), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1949, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 125)
    canvas.paste(_c6, (1284, 2345), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2345, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/07_icon_On.png
try:
    _c7 = get_crop(7, 288, 156)
    canvas.paste(_c7, (288, 2804), _c7)
except Exception:
    pass
layout["On"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/08_icon_Home.png
try:
    _c8 = get_crop(8, 288, 156)
    canvas.paste(_c8, (0, 2804), _c8)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/09_icon_5.20.png
try:
    _c9 = get_crop(9, 108, 102)
    canvas.paste(_c9, (38, 120), _c9)
except Exception:
    pass
layout["5.20"] = [38, 120, 146, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 139)
    canvas.paste(_c10, (1284, 1539), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/11_icon_Overflow_menu_button.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1284, 747), _c11)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 125)
    canvas.paste(_c12, (1284, 1949), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1949, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 139)
    canvas.paste(_c13, (1284, 1143), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 60, 58)
    canvas.paste(_c14, (312, 3), _c14)
except Exception:
    pass
layout["icon_14"] = [312, 3, 372, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/15_icon_5.20.png
try:
    _c15 = get_crop(15, 55, 59)
    canvas.paste(_c15, (183, 3), _c15)
except Exception:
    pass
layout["5.20"] = [183, 3, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/16_icon_Online_events.png
try:
    _c16 = get_crop(16, 586, 117)
    canvas.paste(_c16, (427, 2651), _c16)
except Exception:
    pass
layout["Online_events"] = [427, 2651, 1013, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/17_icon_Art_for_Grief_and_Loss.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 1282), _c17)
except Exception:
    pass
layout["Art_for_Grief_and_Loss"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/18_icon_Favorite_button.png
try:
    _c18 = get_crop(18, 144, 139)
    canvas.paste(_c18, (1140, 747), _c18)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 50, 59)
    canvas.paste(_c19, (248, 3), _c19)
except Exception:
    pass
layout["icon_19"] = [248, 3, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/20_icon_Favorite_button.png
try:
    _c20 = get_crop(20, 144, 139)
    canvas.paste(_c20, (1140, 1143), _c20)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/21_icon_Favorite_button.png
try:
    _c21 = get_crop(21, 144, 139)
    canvas.paste(_c21, (1140, 1539), _c21)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 48, 53)
    canvas.paste(_c22, (1321, 7), _c22)
except Exception:
    pass
layout["icon_22"] = [1321, 7, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/23_icon_Working_with_Grief_and_Loss.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 490), _c23)
except Exception:
    pass
layout["Working_with_Grief_and_Lo"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/24_icon_Tickets.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (864, 2804), _c24)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/25_icon_S_00_AM_EDT.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 1678), _c25)
except Exception:
    pass
layout["S:00_AM_EDT"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/26_icon_Tr.png
try:
    _c26 = get_crop(26, 53, 54)
    canvas.paste(_c26, (392, 2647), _c26)
except Exception:
    pass
layout["Tr"] = [392, 2647, 445, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 65, 59)
    canvas.paste(_c27, (1212, 4), _c27)
except Exception:
    pass
layout["icon_27"] = [1212, 4, 1277, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/28_icon_5_O0_AM_EDT.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 2074), _c28)
except Exception:
    pass
layout["5:O0_AM_EDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 44, 55)
    canvas.paste(_c29, (385, 7), _c29)
except Exception:
    pass
layout["icon_29"] = [385, 7, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 42, 55)
    canvas.paste(_c30, (1272, 6), _c30)
except Exception:
    pass
layout["icon_30"] = [1272, 6, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/31_icon_5.20.png
try:
    _c31 = get_crop(31, 57, 61)
    canvas.paste(_c31, (116, 2), _c31)
except Exception:
    pass
layout["5.20"] = [116, 2, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/32_icon_Understanding_Grief_and_Loss.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 886), _c32)
except Exception:
    pass
layout["Understanding_Grief_and_L"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/33_icon_Online.png
try:
    _c33 = get_crop(33, 112, 53)
    canvas.paste(_c33, (390, 1496), _c33)
except Exception:
    pass
layout["Online"] = [390, 1496, 502, 1549]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/34_icon_suppoloyed_Orilee_herapeeticrarard_Outh_.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 1282), _c34)
except Exception:
    pass
layout["suppoloyed_Orilee__herape"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/35_icon_Online.png
try:
    _c35 = get_crop(35, 112, 54)
    canvas.paste(_c35, (390, 703), _c35)
except Exception:
    pass
layout["Online"] = [390, 703, 502, 757]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/36_icon_Grief_and.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 886), _c36)
except Exception:
    pass
layout["Grief_and"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/37_icon_Art_for_Grief_and_Loss.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 1282), _c37)
except Exception:
    pass
layout["Art_for_Grief_and_Loss"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/38_text_5.20.png
try:
    _c38 = get_crop(38, 89, 43)
    canvas.paste(_c38, (22, 17), _c38)
except Exception:
    pass
layout["5.20"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/39_text_More_events_you_II_love.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 490), _c39)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/40_text_Fri_May_31.png
try:
    _c40 = get_crop(40, 184, 43)
    canvas.paste(_c40, (392, 2525), _c40)
except Exception:
    pass
layout["Fri,_May_31"] = [392, 2525, 576, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/41_text_9_00_AM_EDT.png
try:
    _c41 = get_crop(41, 1344, 346)
    canvas.paste(_c41, (48, 2470), _c41)
except Exception:
    pass
layout["9:00_AM_EDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/42_clickable_Favorites.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (576, 2804), _c42)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_01_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-3/43_clickable_More.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (1152, 2804), _c43)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
