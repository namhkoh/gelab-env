# page_id: page_eventbrite_66847fb559f849b19cea93b83307fae7_02
# screenshot: 2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4.png
# step_index: 2/4
# task: Open Eventbrite. Open favorites and select the second event. Process to checkout and see what payment options it offers.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for the provided canvas and draw objects.
# Assumes variables provided: canvas (PIL.Image 1440x2960), draw (ImageDraw.Draw), font_sm, font_md, font_lg, font_xl

# Overall background (dominant white/off-white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar area at top (~72px)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill=(210, 210, 210))

# Header / toolbar divider lines
# Slight darker divider under status/header area (toolbar area assumed below status)
toolbar_bottom = 340
draw.rectangle([(0, status_h), (1440, toolbar_bottom)], fill=(255, 255, 255))
draw.line([(24, toolbar_bottom), (1440 - 24, toolbar_bottom)], fill=(230, 230, 235), width=2)

# Subtle underline for the active tab area (keeps decorative, not duplicating text/icons)
active_tab_underline_y = 420
draw.line([(120, active_tab_underline_y), (360, active_tab_underline_y)], fill=(54, 112, 255), width=4)

# Section card backgrounds (three event rows)
card_fill = (250, 250, 252)        # very subtle off-white for cards
card_border = (236, 236, 241)      # slight border
card_radius = 18
card_margin_x = 32

event_rows = [
    (0, 675, 1440, 675 + 396),
    (0, 1272, 1440, 1272 + 396),
    (0, 1869, 1440, 1869 + 396)
]

for (x0, y0, x1, y1) in event_rows:
    rx0 = x0 + card_margin_x
    rx1 = x1 - card_margin_x
    ry0 = y0 + 12   # small vertical inset so cards sit with breathing room
    ry1 = y1 - 12
    draw.rounded_rectangle([(rx0, ry0), (rx1, ry1)], radius=card_radius, fill=card_fill, outline=card_border, width=1)

    # subtle separator under each card
    sep_y = ry1 + 18
    draw.line([(rx0 + 6, sep_y), (rx1 - 6, sep_y)], fill=(242, 242, 246), width=1)

# "Find events" pills background area (light panel behind category chips)
pills_top = 2480
pills_bottom = 2796
pills_left = 24
pills_right = 1440 - 24
draw.rounded_rectangle([(pills_left, pills_top), (pills_right, pills_bottom)], radius=22, fill=(255, 255, 255), outline=(240, 240, 245), width=1)

# A faint shadow-like divider above the pills area
draw.line([(pills_left + 6, pills_top), (pills_right - 6, pills_top)], fill=(235, 235, 239), width=2)

# Bottom navigation bar background and top divider
nav_top = 2804
nav_bottom = 2960
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill=(255, 255, 255))
draw.line([(24, nav_top), (1440 - 24, nav_top)], fill=(230, 230, 235), width=2)

# Additional subtle separators to structure content flow
# Separator under the main title/intro area (above first event card)
draw.line([(24, 632), (1440 - 24, 632)], fill=(245, 245, 248), width=1)

# Light vertical guides (very subtle) to suggest content margins (do not overlap detected elements)
guide_x_left = 48
guide_x_right = 1440 - 48
draw.line([(guide_x_left, toolbar_bottom + 6), (guide_x_left, nav_top - 6)], fill=(250, 250, 252), width=1)
draw.line([(guide_x_right, toolbar_bottom + 6), (guide_x_right, nav_top - 6)], fill=(250, 250, 252), width=1)

# Decorative faint card tops for visual grouping (small colored banners behind image areas)
# (Keep very subtle and generic so they are only background structure)
banner_color = (250, 248, 246)
for (x0, y0, x1, y1) in event_rows:
    bx0 = x0 + card_margin_x + 12
    bx1 = bx0 + 220
    by0 = y0 + 28
    by1 = by0 + 128
    draw.rounded_rectangle([(bx0, by0), (bx1, by1)], radius=8, fill=banner_color, outline=None)

# Final subtle bottom edge shadow for depth
draw.line([(0, nav_bottom - 1), (1440, nav_bottom - 1)], fill=(245, 245, 247), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/00_icon_Un.png
try:
    _c0 = get_crop(0, 1440, 396)
    canvas.paste(_c0, (0, 1869), _c0)
except Exception:
    pass
layout["Un"] = [0, 1869, 1440, 2265]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/01_icon_Active.png
try:
    _c1 = get_crop(1, 237, 144)
    canvas.paste(_c1, (909, 2534), _c1)
except Exception:
    pass
layout["Active"] = [909, 2534, 1146, 2678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/02_icon_Learn.png
try:
    _c2 = get_crop(2, 224, 144)
    canvas.paste(_c2, (1146, 2534), _c2)
except Exception:
    pass
layout["Learn"] = [1146, 2534, 1370, 2678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/03_icon_Food_Drink.png
try:
    _c3 = get_crop(3, 352, 144)
    canvas.paste(_c3, (557, 2534), _c3)
except Exception:
    pass
layout["Food_&_Drink"] = [557, 2534, 909, 2678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/04_icon_Nightlife.png
try:
    _c4 = get_crop(4, 278, 144)
    canvas.paste(_c4, (279, 2534), _c4)
except Exception:
    pass
layout["Nightlife"] = [279, 2534, 557, 2678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/05_icon_Music.png
try:
    _c5 = get_crop(5, 231, 144)
    canvas.paste(_c5, (48, 2534), _c5)
except Exception:
    pass
layout["Music"] = [48, 2534, 279, 2678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/06_icon_Sales_ended.png
try:
    _c6 = get_crop(6, 1440, 396)
    canvas.paste(_c6, (0, 675), _c6)
except Exception:
    pass
layout["Sales_ended"] = [0, 675, 1440, 1071]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/07_icon_JACK.png
try:
    _c7 = get_crop(7, 1440, 396)
    canvas.paste(_c7, (0, 1272), _c7)
except Exception:
    pass
layout["JACK"] = [0, 1272, 1440, 1668]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/08_icon_Events.png
try:
    _c8 = get_crop(8, 220, 134)
    canvas.paste(_c8, (0, 343), _c8)
except Exception:
    pass
layout["Events"] = [0, 343, 220, 477]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/09_icon_Favorites.png
try:
    _c9 = get_crop(9, 308, 144)
    canvas.paste(_c9, (217, 330), _c9)
except Exception:
    pass
layout["Favorites"] = [217, 330, 525, 474]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/10_icon_Organizers.png
try:
    _c10 = get_crop(10, 308, 144)
    canvas.paste(_c10, (217, 330), _c10)
except Exception:
    pass
layout["Organizers"] = [217, 330, 525, 474]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 62, 62)
    canvas.paste(_c11, (310, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [310, 1, 372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/12_icon_7.38.png
try:
    _c12 = get_crop(12, 58, 62)
    canvas.paste(_c12, (180, 1), _c12)
except Exception:
    pass
layout["7.38"] = [180, 1, 238, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/13_icon_Nightlife.png
try:
    _c13 = get_crop(13, 288, 156)
    canvas.paste(_c13, (288, 2804), _c13)
except Exception:
    pass
layout["Nightlife"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/14_icon_7.38.png
try:
    _c14 = get_crop(14, 60, 64)
    canvas.paste(_c14, (114, 0), _c14)
except Exception:
    pass
layout["7.38"] = [114, 0, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/15_icon_THE_LIFE_DEATH_OF_ART.png
try:
    _c15 = get_crop(15, 1440, 396)
    canvas.paste(_c15, (0, 1272), _c15)
except Exception:
    pass
layout["THE_LIFE_&_DEATH_OF_ART"] = [0, 1272, 1440, 1668]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 49, 59)
    canvas.paste(_c16, (250, 3), _c16)
except Exception:
    pass
layout["icon_16"] = [250, 3, 299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/17_icon_Festival.png
try:
    _c17 = get_crop(17, 263, 138)
    canvas.paste(_c17, (48, 2678), _c17)
except Exception:
    pass
layout["Festival"] = [48, 2678, 311, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/18_icon_Festival.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (0, 2804), _c18)
except Exception:
    pass
layout["Festival"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 56, 71)
    canvas.paste(_c19, (1316, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1316, 0, 1372, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/20_icon_Unlike_event.png
try:
    _c20 = get_crop(20, 72, 72)
    canvas.paste(_c20, (1320, 2145), _c20)
except Exception:
    pass
layout["Unlike_event"] = [1320, 2145, 1392, 2217]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/21_icon_Understanding_Grief_and_Loss.png
try:
    _c21 = get_crop(21, 1440, 396)
    canvas.paste(_c21, (0, 1869), _c21)
except Exception:
    pass
layout["Understanding_Grief_and_L"] = [0, 1869, 1440, 2265]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/22_icon_Unlike_event.png
try:
    _c22 = get_crop(22, 72, 72)
    canvas.paste(_c22, (1320, 1548), _c22)
except Exception:
    pass
layout["Unlike_event"] = [1320, 1548, 1392, 1620]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/23_icon_Unlike_event.png
try:
    _c23 = get_crop(23, 72, 72)
    canvas.paste(_c23, (1320, 951), _c23)
except Exception:
    pass
layout["Unlike_event"] = [1320, 951, 1392, 1023]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 93, 71)
    canvas.paste(_c24, (1210, 0), _c24)
except Exception:
    pass
layout["icon_24"] = [1210, 0, 1303, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/25_icon_Share_Event.png
try:
    _c25 = get_crop(25, 72, 72)
    canvas.paste(_c25, (1200, 1548), _c25)
except Exception:
    pass
layout["Share_Event"] = [1200, 1548, 1272, 1620]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/26_icon_Active.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (864, 2804), _c26)
except Exception:
    pass
layout["Active"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/27_icon_Share_Event.png
try:
    _c27 = get_crop(27, 72, 72)
    canvas.paste(_c27, (1200, 951), _c27)
except Exception:
    pass
layout["Share_Event"] = [1200, 951, 1272, 1023]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/28_icon_UrbanGlass.png
try:
    _c28 = get_crop(28, 224, 62)
    canvas.paste(_c28, (390, 937), _c28)
except Exception:
    pass
layout["UrbanGlass"] = [390, 937, 614, 999]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 42, 67)
    canvas.paste(_c29, (1272, 1), _c29)
except Exception:
    pass
layout["icon_29"] = [1272, 1, 1314, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/30_icon_THE_LIFE_DEATH_OF_ART.png
try:
    _c30 = get_crop(30, 1440, 396)
    canvas.paste(_c30, (0, 1272), _c30)
except Exception:
    pass
layout["THE_LIFE_&_DEATH_OF_ART"] = [0, 1272, 1440, 1668]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/31_icon_with_Shuhei_Fujii.png
try:
    _c31 = get_crop(31, 1440, 396)
    canvas.paste(_c31, (0, 675), _c31)
except Exception:
    pass
layout["with_Shuhei_Fujii"] = [0, 675, 1440, 1071]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/32_icon_Share_Event.png
try:
    _c32 = get_crop(32, 72, 72)
    canvas.paste(_c32, (1200, 2145), _c32)
except Exception:
    pass
layout["Share_Event"] = [1200, 2145, 1272, 2217]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/33_icon_Tue_Apr_23.png
try:
    _c33 = get_crop(33, 1440, 396)
    canvas.paste(_c33, (0, 675), _c33)
except Exception:
    pass
layout["Tue,_Apr_23"] = [0, 675, 1440, 1071]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/34_icon_7.38.png
try:
    _c34 = get_crop(34, 93, 62)
    canvas.paste(_c34, (15, 1), _c34)
except Exception:
    pass
layout["7.38"] = [15, 1, 108, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/35_icon_Learn.png
try:
    _c35 = get_crop(35, 288, 156)
    canvas.paste(_c35, (1152, 2804), _c35)
except Exception:
    pass
layout["Learn"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/36_text_Today.png
try:
    _c36 = get_crop(36, 180, 80)
    canvas.paste(_c36, (39, 579), _c36)
except Exception:
    pass
layout["Today"] = [39, 579, 219, 659]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/37_text_Apr_23.png
try:
    _c37 = get_crop(37, 190, 72)
    canvas.paste(_c37, (389, 583), _c37)
except Exception:
    pass
layout["Apr_23"] = [389, 583, 579, 655]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/38_text_JACK.png
try:
    _c38 = get_crop(38, 105, 43)
    canvas.paste(_c38, (390, 1474), _c38)
except Exception:
    pass
layout["JACK"] = [390, 1474, 495, 1517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/39_text_Wed_Jun_26.png
try:
    _c39 = get_crop(39, 334, 68)
    canvas.paste(_c39, (45, 1777), _c39)
except Exception:
    pass
layout["Wed,_Jun_26"] = [45, 1777, 379, 1845]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/40_text_Wed_Jun_26.png
try:
    _c40 = get_crop(40, 244, 52)
    canvas.paste(_c40, (391, 1922), _c40)
except Exception:
    pass
layout["Wed,_Jun_26"] = [391, 1922, 635, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/41_text_I_00_PM_EDT.png
try:
    _c41 = get_crop(41, 1440, 396)
    canvas.paste(_c41, (0, 1869), _c41)
except Exception:
    pass
layout["I:00_PM_EDT"] = [0, 1869, 1440, 2265]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/42_text_Find_events.png
try:
    _c42 = get_crop(42, 231, 144)
    canvas.paste(_c42, (48, 2534), _c42)
except Exception:
    pass
layout["Find_events"] = [48, 2534, 279, 2678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/66847fb559f849b19cea93b83307fae7/step_02_2024_4_23_19_37_66847fb559f849b19cea93b83307fae7-4/43_clickable_Favorites.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (576, 2804), _c43)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]
