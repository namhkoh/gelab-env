# page_id: page_eventbrite_8efde6fd9e974d7e804c40fab58deb06_03
# screenshot: 2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5.png
# step_index: 3/8
# task: Open Eventbrite. Search for "Education". Filter only online events. Note how many events are available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (1440x2960 white). Draw background and structural UI elements.

# Colors
bg_color = (255, 255, 255)            # main background (white)
status_bar_color = (200, 200, 200)    # light gray status bar
header_divider_blue = (38, 84, 255)   # vibrant blue underline in header
header_bg = (255, 255, 255)           # header white
card_bg = (255, 255, 255)             # card white
card_border = (230, 230, 230)         # subtle card border / shadow
thumb_bg = (245, 245, 247)            # thumbnail placeholder background
section_divider = (230, 230, 235)     # light separator lines
bottom_bar_bg = (255, 255, 255)       # bottom navigation bar background
bottom_bar_top_border = (220, 220, 225)

w, h = canvas.size

# Fill overall background (in case not already)
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar area (approx ~56px high)
status_h = 56
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# Header area (search header). Place under status bar.
header_top = status_h
header_bottom = 152
draw.rectangle([(0, header_top), (w, header_bottom)], fill=header_bg)

# Blue underline below header (thin)
underline_h = 6
draw.rectangle([(48, header_bottom - underline_h), (w - 48, header_bottom)], fill=header_divider_blue)

# Subtle shadow line under header
draw.line([(0, header_bottom), (w, header_bottom)], fill=(240, 240, 245), width=1)

# Section separator between "Popular" list and "Events" area
# From screenshot the Events section title sits around y ~1026, draw a separator a little above.
events_separator_y = 1026
draw.line([(48, events_separator_y), (w - 48, events_separator_y)], fill=section_divider, width=2)

# Draw large rounded "card" backgrounds for main event list items.
# Use detected group positions (approximate) for subtle card grouping.
event_cards = [
    (48, 1117, 48 + 1344, 1117 + 396),
    (48, 1513, 48 + 1344, 1513 + 396),
    (48, 1909, 48 + 1344, 1909 + 396),
    (48, 2305, 48 + 1344, 2305 + 396),
]
card_radius = 12
for (x1, y1, x2, y2) in event_cards:
    # outer subtle border/shadow (very light)
    shadow_rect = [(x1+2, y1+4), (x2+2, y2+4)]
    draw.rounded_rectangle(shadow_rect, radius=card_radius, fill=(250,250,250))
    # card background
    draw.rounded_rectangle([(x1, y1), (x2, y2)], radius=card_radius, fill=card_bg, outline=card_border, width=1)
    # left thumbnail background inside card (subtle placeholder, will be covered by pasted image)
    thumb_x = x1 + 12
    thumb_y = y1 + 12
    thumb_w = 180
    thumb_h = 120
    draw.rounded_rectangle([(thumb_x, thumb_y), (thumb_x + thumb_w, thumb_y + thumb_h)], radius=8, fill=thumb_bg, outline=(235,235,240))

    # right side small thin divider to visually separate thumbnail area (very subtle)
    divider_x = thumb_x + thumb_w + 18
    draw.line([(divider_x, y1 + 16), (divider_x, y2 - 16)], fill=(245,245,247), width=1)

# Additional subtle separators between list rows (where items end), placed between cards
for _, y1, _, y2 in event_cards:
    # light line below each card (except last maybe)
    draw.line([(48, y2 + 8), (w - 48, y2 + 8)], fill=section_divider, width=1)

# Large content area / banner placeholders (example: image blocks further down)
# There are small promotional tiles near bottom; draw a low-contrast banner row background behind them
promo_y = 2720
promo_h = 140
draw.rectangle([(0, promo_y), (w, promo_y + promo_h)], fill=(250, 250, 252))

# Bottom navigation bar background and top border
bottom_bar_top = 2804  # derived from detected bottom icons
bottom_bar_height = h - bottom_bar_top
draw.rectangle([(0, bottom_bar_top), (w, h)], fill=bottom_bar_bg)
# Top border of bottom bar
draw.line([(0, bottom_bar_top), (w, bottom_bar_top)], fill=bottom_bar_top_border, width=2)

# Small horizontal separator lines for list sections near top (Popular area)
# Popular title sits around y ~298; draw a faint dividing line under the search header area
popular_div_y = 298 + 90  # slightly below the "Popular" title area
draw.line([(48, popular_div_y), (w - 48, popular_div_y)], fill=section_divider, width=1)

# Final subtle overall vignette lines to match clean app UI (thin)
draw.line([(48, header_bottom + 8), (w - 48, header_bottom + 8)], fill=(245,245,250), width=1)
draw.line([(48, popular_div_y + 180), (w - 48, popular_div_y + 180)], fill=(245,245,250), width=1)

# End of structural/background drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/00_icon_Events.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1117), _c0)
except Exception:
    pass
layout["Events"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/01_icon_JANL.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 2305), _c1)
except Exception:
    pass
layout["JANL"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/02_icon_8_56_creator_followers.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1513), _c2)
except Exception:
    pass
layout["8_56_creator_followers"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/03_icon_Educational_professionals.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1117), _c3)
except Exception:
    pass
layout["Educational_professionals"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/04_icon_Education.png
try:
    _c4 = get_crop(4, 54, 57)
    canvas.paste(_c4, (314, 5), _c4)
except Exception:
    pass
layout["Education"] = [314, 5, 368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 42, 54)
    canvas.paste(_c5, (254, 7), _c5)
except Exception:
    pass
layout["icon_5"] = [254, 7, 296, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/06_icon_6.59.png
try:
    _c6 = get_crop(6, 52, 60)
    canvas.paste(_c6, (184, 3), _c6)
except Exception:
    pass
layout["6.59"] = [184, 3, 236, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/07_icon_6.59.png
try:
    _c7 = get_crop(7, 56, 60)
    canvas.paste(_c7, (115, 4), _c7)
except Exception:
    pass
layout["6.59"] = [115, 4, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/08_icon_6.59.png
try:
    _c8 = get_crop(8, 110, 101)
    canvas.paste(_c8, (62, 120), _c8)
except Exception:
    pass
layout["6.59"] = [62, 120, 172, 221]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/09_icon_Fr.png
try:
    _c9 = get_crop(9, 288, 156)
    canvas.paste(_c9, (864, 2804), _c9)
except Exception:
    pass
layout["Fr"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/10_icon_Education.png
try:
    _c10 = get_crop(10, 1344, 191)
    canvas.paste(_c10, (48, 72), _c10)
except Exception:
    pass
layout["Education"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/11_icon_2Day_BEAUTY_EXPO_INNU_TRISTATE_AREA.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (288, 2804), _c11)
except Exception:
    pass
layout["2Day_BEAUTY_EXPO_INNU_TRI"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/12_icon_Cancel.png
try:
    _c12 = get_crop(12, 45, 59)
    canvas.paste(_c12, (1323, 3), _c12)
except Exception:
    pass
layout["Cancel"] = [1323, 3, 1368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/13_icon_8_56_creator_followers.png
try:
    _c13 = get_crop(13, 1344, 396)
    canvas.paste(_c13, (48, 1909), _c13)
except Exception:
    pass
layout["8_56_creator_followers"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/14_icon_Cancel.png
try:
    _c14 = get_crop(14, 86, 63)
    canvas.paste(_c14, (1216, 0), _c14)
except Exception:
    pass
layout["Cancel"] = [1216, 0, 1302, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/15_icon_Cancel.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1099, 96), _c15)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/16_icon_6.59.png
try:
    _c16 = get_crop(16, 89, 58)
    canvas.paste(_c16, (17, 5), _c16)
except Exception:
    pass
layout["6.59"] = [17, 5, 106, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/17_icon_new.png
try:
    _c17 = get_crop(17, 96, 96)
    canvas.paste(_c17, (31, 529), _c17)
except Exception:
    pass
layout["new"] = [31, 529, 127, 625]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/18_icon_Cancel.png
try:
    _c18 = get_crop(18, 42, 60)
    canvas.paste(_c18, (1272, 3), _c18)
except Exception:
    pass
layout["Cancel"] = [1272, 3, 1314, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/19_icon_Cancel.png
try:
    _c19 = get_crop(19, 149, 144)
    canvas.paste(_c19, (1243, 97), _c19)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/20_icon_2Day_BEAUTY_EXPO_INNU_TRISTATE_AREA.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (0, 2804), _c20)
except Exception:
    pass
layout["2Day_BEAUTY_EXPO_INNU_TRI"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/21_icon_Fading.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1513), _c21)
except Exception:
    pass
layout["Fading"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/22_icon_AllilOn_Education_Blending_and_Tapering.png
try:
    _c22 = get_crop(22, 1344, 396)
    canvas.paste(_c22, (48, 1909), _c22)
except Exception:
    pass
layout["AllilOn_Education:_Blendi"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/23_icon_Allilot.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 1513), _c23)
except Exception:
    pass
layout["Allilot"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/24_icon_new.png
try:
    _c24 = get_crop(24, 92, 98)
    canvas.paste(_c24, (36, 646), _c24)
except Exception:
    pass
layout["new"] = [36, 646, 128, 744]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/25_icon_Education_You_networking_event.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 1117), _c25)
except Exception:
    pass
layout["Education?You_networking_"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/26_icon_Education.png
try:
    _c26 = get_crop(26, 45, 59)
    canvas.paste(_c26, (385, 4), _c26)
except Exception:
    pass
layout["Education"] = [385, 4, 430, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/27_icon_Popular.png
try:
    _c27 = get_crop(27, 97, 105)
    canvas.paste(_c27, (34, 404), _c27)
except Exception:
    pass
layout["Popular"] = [34, 404, 131, 509]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/28_icon_More.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (1152, 2804), _c28)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/29_icon_new.png
try:
    _c29 = get_crop(29, 83, 87)
    canvas.paste(_c29, (40, 771), _c29)
except Exception:
    pass
layout["new"] = [40, 771, 123, 858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/30_icon_Events.png
try:
    _c30 = get_crop(30, 76, 75)
    canvas.paste(_c30, (41, 896), _c30)
except Exception:
    pass
layout["Events"] = [41, 896, 117, 971]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/31_icon_Fading.png
try:
    _c31 = get_crop(31, 1344, 396)
    canvas.paste(_c31, (48, 1513), _c31)
except Exception:
    pass
layout["Fading"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/32_text_Popular.png
try:
    _c32 = get_crop(32, 221, 78)
    canvas.paste(_c32, (44, 298), _c32)
except Exception:
    pass
layout["Popular"] = [44, 298, 265, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/33_text_education_conference.png
try:
    _c33 = get_crop(33, 1344, 120)
    canvas.paste(_c33, (48, 378), _c33)
except Exception:
    pass
layout["education_conference"] = [48, 378, 1392, 498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/34_text_education_technology.png
try:
    _c34 = get_crop(34, 1344, 120)
    canvas.paste(_c34, (48, 498), _c34)
except Exception:
    pass
layout["education_technology"] = [48, 498, 1392, 618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/35_text_new.png
try:
    _c35 = get_crop(35, 85, 36)
    canvas.paste(_c35, (165, 679), _c35)
except Exception:
    pass
layout["new"] = [165, 679, 250, 715]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/36_text_city_department_of_education.png
try:
    _c36 = get_crop(36, 1344, 120)
    canvas.paste(_c36, (48, 618), _c36)
except Exception:
    pass
layout["city_department_of_educat"] = [48, 618, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/37_text_education_computer_science.png
try:
    _c37 = get_crop(37, 1344, 120)
    canvas.paste(_c37, (48, 738), _c37)
except Exception:
    pass
layout["education_computer_scienc"] = [48, 738, 1392, 858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/38_text_department_of_education.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 858), _c38)
except Exception:
    pass
layout["department_of_education"] = [48, 858, 1392, 1002]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/39_text_Events.png
try:
    _c39 = get_crop(39, 188, 61)
    canvas.paste(_c39, (45, 1026), _c39)
except Exception:
    pass
layout["Events"] = [45, 1026, 233, 1087]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/40_text_Thu.png
try:
    _c40 = get_crop(40, 89, 50)
    canvas.paste(_c40, (389, 2420), _c40)
except Exception:
    pass
layout["Thu,"] = [389, 2420, 478, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/41_text_16_._6_00_PM_EDT.png
try:
    _c41 = get_crop(41, 1344, 396)
    canvas.paste(_c41, (48, 2305), _c41)
except Exception:
    pass
layout["16_._6:00_PM_EDT"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/42_text_Education_Through_Music_2024_GALA.png
try:
    _c42 = get_crop(42, 1344, 396)
    canvas.paste(_c42, (48, 2305), _c42)
except Exception:
    pass
layout["Education_Through_Music_2"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/43_text_583_Park_Ave.png
try:
    _c43 = get_crop(43, 219, 39)
    canvas.paste(_c43, (392, 2547), _c43)
except Exception:
    pass
layout["583_Park_Ave"] = [392, 2547, 611, 2586]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/44_text_2Day_BEAUTY_EXPO_INNU_TRISTATE_AREA.png
try:
    _c44 = get_crop(44, 288, 156)
    canvas.paste(_c44, (0, 2804), _c44)
except Exception:
    pass
layout["2Day_BEAUTY_EXPO_INNU_TRI"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/45_text_Fr.png
try:
    _c45 = get_crop(45, 39, 12)
    canvas.paste(_c45, (750, 2794), _c45)
except Exception:
    pass
layout["Fr"] = [750, 2794, 789, 2806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_03_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-5/46_clickable_Favorites.png
try:
    _c46 = get_crop(46, 288, 156)
    canvas.paste(_c46, (576, 2804), _c46)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]
