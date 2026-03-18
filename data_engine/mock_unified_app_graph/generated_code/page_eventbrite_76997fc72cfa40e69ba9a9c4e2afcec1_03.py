# page_id: page_eventbrite_76997fc72cfa40e69ba9a9c4e2afcec1_03
# screenshot: 2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5.png
# step_index: 3/3
# task: Open Eventbrite. Open favorite tab and remove the second event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background/base
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar (top ~50px)
status_h = 60
draw.rectangle((0, 0, 1440, status_h), fill=(192, 192, 192))
# subtle bottom divider for status bar
draw.line((0, status_h - 1, 1440, status_h - 1), fill=(170, 170, 170), width=1)

# Top header area (below status bar) - keep white but add soft subtle divider
header_top = status_h
header_bottom = 320
draw.rectangle((0, header_top, 1440, header_bottom), fill=(255, 255, 255))
draw.line((24, header_bottom - 1, 1440 - 24, header_bottom - 1), fill=(240, 240, 240), width=1)

# Tab underline (indicator under the left "Events" tab)
# Positioned under the tab area (approx where tabs are detected)
tab_indicator_y = 444
draw.rounded_rectangle((40, tab_indicator_y, 200, tab_indicator_y + 8), radius=4, fill=(36, 90, 255))

# Main content area background (already white) - add subtle vertical padding lines
draw.line((24, header_bottom + 8, 24, 2960 - 156), fill=(255,255,255), width=1)  # left margin mask (invisible)
draw.line((1440 - 24, header_bottom + 8, 1440 - 24, 2960 - 156), fill=(255,255,255), width=1)  # right margin mask

# Define card areas (these correspond to detected event blocks)
card_areas = [
    (0, 675, 1440, 675 + 396),
    (0, 1272, 1440, 1272 + 396),
    (0, 1869, 1440, 1869 + 396),
    (0, 2466, 1440, 2466 + 350)
]

# Draw rounded card backgrounds and subtle separators
for (l, t, r, b) in card_areas:
    # Inset the card slightly to create white gutters like the screenshot
    inset = 24
    card_l, card_t, card_r, card_b = l + inset, t + 8, r - inset, b - 8
    # card background (very slightly off-white to separate from page)
    draw.rounded_rectangle((card_l, card_t, card_r, card_b), radius=14, fill=(250,250,250), outline=(235,235,235), width=1)
    # left thumbnail background (placeholder area behind actual thumbnail that will be pasted)
    thumb_x = card_l + 16
    thumb_y = card_t + 22
    thumb_w = 140
    thumb_h = 140
    draw.rectangle((thumb_x, thumb_y, thumb_x + thumb_w, thumb_y + thumb_h), fill=(240,240,240), outline=(230,230,230))
    # subtle horizontal separator below each card
    sep_y = card_b + 12
    draw.line((card_l, sep_y, card_r, sep_y), fill=(245,245,245), width=1)
    # small orange time-banner placeholder above text area (behind time text)
    time_bar_x = thumb_x + thumb_w + 18
    time_bar_y = thumb_y - 2
    draw.rounded_rectangle((time_bar_x, time_bar_y, time_bar_x + 480, time_bar_y + 28), radius=6, fill=(214, 69, 24))

# Additional subtle separators between sections (to echo screenshot structure)
separator_ys = [560, 1160, 1755, 2345]
for y in separator_ys:
    draw.line((24, y, 1440 - 24, y), fill=(245,245,245), width=1)

# Bottom navigation bar area
nav_top = 2804
nav_bottom = 2960
draw.rectangle((0, nav_top, 1440, nav_bottom), fill=(255, 255, 255))
# top border of nav
draw.line((0, nav_top, 1440, nav_top), fill=(230, 230, 230), width=2)
# faint center background highlight for active area
active_center_w = 160
draw.rounded_rectangle((720 - active_center_w/2, nav_top + 8, 720 + active_center_w/2, nav_bottom - 20), radius=40, fill=(255,255,255))

# Floating thin bottom guideline to separate content from nav icons
draw.line((24, nav_top + 2, 1440 - 24, nav_top + 2), fill=(240,240,240), width=1)

# Subtle overall left margin vertical guide (visual structure only)
draw.line((24, header_bottom + 12, 24, nav_top - 12), fill=(255,255,255), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/00_icon_SHIPYARD.png
try:
    _c0 = get_crop(0, 1440, 396)
    canvas.paste(_c0, (0, 1272), _c0)
except Exception:
    pass
layout["SHIPYARD"] = [0, 1272, 1440, 1668]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/01_icon_Win_More_Business.png
try:
    _c1 = get_crop(1, 1440, 396)
    canvas.paste(_c1, (0, 675), _c1)
except Exception:
    pass
layout["&_Win_More_Business"] = [0, 675, 1440, 1071]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/02_icon_JACK.png
try:
    _c2 = get_crop(2, 1440, 396)
    canvas.paste(_c2, (0, 1869), _c2)
except Exception:
    pass
layout["JACK"] = [0, 1869, 1440, 2265]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/03_icon_REE.png
try:
    _c3 = get_crop(3, 288, 156)
    canvas.paste(_c3, (288, 2804), _c3)
except Exception:
    pass
layout["REE"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/04_icon_Events.png
try:
    _c4 = get_crop(4, 220, 134)
    canvas.paste(_c4, (0, 344), _c4)
except Exception:
    pass
layout["Events"] = [0, 344, 220, 478]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/05_icon_Favorites.png
try:
    _c5 = get_crop(5, 308, 144)
    canvas.paste(_c5, (217, 330), _c5)
except Exception:
    pass
layout["Favorites"] = [217, 330, 525, 474]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/06_icon_Organizers.png
try:
    _c6 = get_crop(6, 308, 144)
    canvas.paste(_c6, (217, 330), _c6)
except Exception:
    pass
layout["Organizers"] = [217, 330, 525, 474]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 61, 62)
    canvas.paste(_c7, (310, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [310, 1, 371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/08_icon_5.37.png
try:
    _c8 = get_crop(8, 61, 63)
    canvas.paste(_c8, (179, 1), _c8)
except Exception:
    pass
layout["5.37"] = [179, 1, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/09_icon_5.37.png
try:
    _c9 = get_crop(9, 62, 65)
    canvas.paste(_c9, (112, 0), _c9)
except Exception:
    pass
layout["5.37"] = [112, 0, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 49, 59)
    canvas.paste(_c10, (250, 3), _c10)
except Exception:
    pass
layout["icon_10"] = [250, 3, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/11_icon_Recoonise.png
try:
    _c11 = get_crop(11, 1440, 350)
    canvas.paste(_c11, (0, 2466), _c11)
except Exception:
    pass
layout["Recoonise"] = [0, 2466, 1440, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/12_icon_Unlike_event.png
try:
    _c12 = get_crop(12, 72, 72)
    canvas.paste(_c12, (1320, 2145), _c12)
except Exception:
    pass
layout["Unlike_event"] = [1320, 2145, 1392, 2217]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/13_icon_THE_LIFE_DEATH_OF_ART.png
try:
    _c13 = get_crop(13, 1440, 396)
    canvas.paste(_c13, (0, 1869), _c13)
except Exception:
    pass
layout["THE_LIFE_&_DEATH_OF_ART"] = [0, 1869, 1440, 2265]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 56, 71)
    canvas.paste(_c14, (1316, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1316, 0, 1372, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/15_icon_Unlike_event.png
try:
    _c15 = get_crop(15, 72, 72)
    canvas.paste(_c15, (1320, 951), _c15)
except Exception:
    pass
layout["Unlike_event"] = [1320, 951, 1392, 1023]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/16_icon_Unlike_event.png
try:
    _c16 = get_crop(16, 72, 72)
    canvas.paste(_c16, (1320, 1548), _c16)
except Exception:
    pass
layout["Unlike_event"] = [1320, 1548, 1392, 1620]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/17_icon_Share_Event.png
try:
    _c17 = get_crop(17, 72, 72)
    canvas.paste(_c17, (1200, 951), _c17)
except Exception:
    pass
layout["Share_Event"] = [1200, 951, 1272, 1023]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 91, 71)
    canvas.paste(_c18, (1210, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1210, 0, 1301, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/19_icon_Share_Event.png
try:
    _c19 = get_crop(19, 72, 72)
    canvas.paste(_c19, (1200, 2145), _c19)
except Exception:
    pass
layout["Share_Event"] = [1200, 2145, 1272, 2217]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/20_icon_Share_Event.png
try:
    _c20 = get_crop(20, 72, 72)
    canvas.paste(_c20, (1200, 1548), _c20)
except Exception:
    pass
layout["Share_Event"] = [1200, 1548, 1272, 1620]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/21_icon_REE.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (0, 2804), _c21)
except Exception:
    pass
layout["REE"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 42, 67)
    canvas.paste(_c22, (1272, 1), _c22)
except Exception:
    pass
layout["icon_22"] = [1272, 1, 1314, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/23_icon_THE_LIFE_DEATH_OF_ART.png
try:
    _c23 = get_crop(23, 1440, 396)
    canvas.paste(_c23, (0, 1869), _c23)
except Exception:
    pass
layout["THE_LIFE_&_DEATH_OF_ART"] = [0, 1869, 1440, 2265]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/24_icon_5.37.png
try:
    _c24 = get_crop(24, 93, 61)
    canvas.paste(_c24, (15, 2), _c24)
except Exception:
    pass
layout["5.37"] = [15, 2, 108, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/25_icon_THE_LIFE_DEATH_OF_ART.png
try:
    _c25 = get_crop(25, 1440, 396)
    canvas.paste(_c25, (0, 1869), _c25)
except Exception:
    pass
layout["THE_LIFE_&_DEATH_OF_ART"] = [0, 1869, 1440, 2265]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/26_icon_Sun_May_5_._8.00_PM_EDT.png
try:
    _c26 = get_crop(26, 1440, 350)
    canvas.paste(_c26, (0, 2466), _c26)
except Exception:
    pass
layout["Sun,_May_5_._8.00_PM_EDT"] = [0, 2466, 1440, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/27_icon_Realtors_Refine_Client_Appreciation.png
try:
    _c27 = get_crop(27, 1440, 396)
    canvas.paste(_c27, (0, 675), _c27)
except Exception:
    pass
layout["Realtors:_Refine_Client_A"] = [0, 675, 1440, 1071]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/28_text_Today.png
try:
    _c28 = get_crop(28, 185, 85)
    canvas.paste(_c28, (37, 576), _c28)
except Exception:
    pass
layout["Today"] = [37, 576, 222, 661]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/29_text_Wed.png
try:
    _c29 = get_crop(29, 151, 73)
    canvas.paste(_c29, (263, 582), _c29)
except Exception:
    pass
layout["Wed,"] = [263, 582, 414, 655]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/30_text_24.png
try:
    _c30 = get_crop(30, 84, 61)
    canvas.paste(_c30, (519, 584), _c30)
except Exception:
    pass
layout["24"] = [519, 584, 603, 645]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/31_text_Wed.png
try:
    _c31 = get_crop(31, 110, 54)
    canvas.paste(_c31, (390, 727), _c31)
except Exception:
    pass
layout["Wed,"] = [390, 727, 500, 781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/32_text_24_._9_00_PM_EDT.png
try:
    _c32 = get_crop(32, 1440, 396)
    canvas.paste(_c32, (0, 675), _c32)
except Exception:
    pass
layout["24_._9:00_PM_EDT"] = [0, 675, 1440, 1071]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/33_text_Strategies_to_Wow_Win_More_Business.png
try:
    _c33 = get_crop(33, 1440, 396)
    canvas.paste(_c33, (0, 675), _c33)
except Exception:
    pass
layout["Strategies_to_Wow_&_Win_M"] = [0, 675, 1440, 1071]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/34_text_JACK.png
try:
    _c34 = get_crop(34, 105, 45)
    canvas.paste(_c34, (390, 945), _c34)
except Exception:
    pass
layout["JACK"] = [390, 945, 495, 990]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/35_text_Sat_Apr_27.png
try:
    _c35 = get_crop(35, 308, 72)
    canvas.paste(_c35, (42, 1180), _c35)
except Exception:
    pass
layout["Sat,_Apr_27"] = [42, 1180, 350, 1252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/36_text_Sat_Apr_27.png
try:
    _c36 = get_crop(36, 224, 55)
    canvas.paste(_c36, (388, 1325), _c36)
except Exception:
    pass
layout["Sat,_Apr_27"] = [388, 1325, 612, 1380]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/37_text_Sun_Apr_28.png
try:
    _c37 = get_crop(37, 1440, 396)
    canvas.paste(_c37, (0, 1272), _c37)
except Exception:
    pass
layout["Sun,_Apr_28"] = [0, 1272, 1440, 1668]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/38_text_I_00AM_PDT.png
try:
    _c38 = get_crop(38, 279, 50)
    canvas.paste(_c38, (890, 1325), _c38)
except Exception:
    pass
layout["I:00AM_PDT"] = [890, 1325, 1169, 1375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/39_text_Shipyard_Open_Studios.png
try:
    _c39 = get_crop(39, 1440, 396)
    canvas.paste(_c39, (0, 1272), _c39)
except Exception:
    pass
layout["Shipyard_Open_Studios"] = [0, 1272, 1440, 1668]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/40_text_Spring_2024.png
try:
    _c40 = get_crop(40, 281, 69)
    canvas.paste(_c40, (914, 1394), _c40)
except Exception:
    pass
layout["Spring_2024"] = [914, 1394, 1195, 1463]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/41_text_Hunters_Point_Shipyard.png
try:
    _c41 = get_crop(41, 1440, 396)
    canvas.paste(_c41, (0, 1272), _c41)
except Exception:
    pass
layout["Hunters_Point_Shipyard"] = [0, 1272, 1440, 1668]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/42_text_Sun.png
try:
    _c42 = get_crop(42, 143, 77)
    canvas.paste(_c42, (39, 1773), _c42)
except Exception:
    pass
layout["Sun,"] = [39, 1773, 182, 1850]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/43_text_28.png
try:
    _c43 = get_crop(43, 84, 64)
    canvas.paste(_c43, (284, 1777), _c43)
except Exception:
    pass
layout["28"] = [284, 1777, 368, 1841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/44_text_Mon.png
try:
    _c44 = get_crop(44, 150, 73)
    canvas.paste(_c44, (44, 2371), _c44)
except Exception:
    pass
layout["Mon,"] = [44, 2371, 194, 2444]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/45_text_Hair_3R_s.png
try:
    _c45 = get_crop(45, 205, 52)
    canvas.paste(_c45, (393, 2590), _c45)
except Exception:
    pass
layout["Hair_3R's"] = [393, 2590, 598, 2642]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/46_text_Recognise_Respond_Refer_-.png
try:
    _c46 = get_crop(46, 1440, 350)
    canvas.paste(_c46, (0, 2466), _c46)
except Exception:
    pass
layout["Recognise,_Respond_&_Refe"] = [0, 2466, 1440, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/47_text_Online_Professional_Development.png
try:
    _c47 = get_crop(47, 1440, 350)
    canvas.paste(_c47, (0, 2466), _c47)
except Exception:
    pass
layout["Online_Professional_Devel"] = [0, 2466, 1440, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/48_text_USO_Warrior_and_Family_Center_at_Fort_Be.png
try:
    _c48 = get_crop(48, 1440, 350)
    canvas.paste(_c48, (0, 2466), _c48)
except Exception:
    pass
layout["USO_Warrior_and_Family_Ce"] = [0, 2466, 1440, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/49_clickable_Share_Event.png
try:
    _c49 = get_crop(49, 72, 72)
    canvas.paste(_c49, (1200, 2742), _c49)
except Exception:
    pass
layout["Share_Event"] = [1200, 2742, 1272, 2814]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/50_clickable_Unlike_event.png
try:
    _c50 = get_crop(50, 72, 72)
    canvas.paste(_c50, (1320, 2742), _c50)
except Exception:
    pass
layout["Unlike_event"] = [1320, 2742, 1392, 2814]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/51_clickable_Favorites.png
try:
    _c51 = get_crop(51, 288, 156)
    canvas.paste(_c51, (576, 2804), _c51)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/52_clickable_Tickets.png
try:
    _c52 = get_crop(52, 288, 156)
    canvas.paste(_c52, (864, 2804), _c52)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/76997fc72cfa40e69ba9a9c4e2afcec1/step_03_2024_4_24_17_36_76997fc72cfa40e69ba9a9c4e2afcec1-5/53_clickable_More.png
try:
    _c53 = get_crop(53, 288, 156)
    canvas.paste(_c53, (1152, 2804), _c53)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
