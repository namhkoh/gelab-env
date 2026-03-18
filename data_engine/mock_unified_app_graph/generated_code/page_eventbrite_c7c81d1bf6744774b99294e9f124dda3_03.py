# page_id: page_eventbrite_c7c81d1bf6744774b99294e9f124dda3_03
# screenshot: 2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5.png
# step_index: 3/10
# task: Open Eventbrite. Search for "Fitness". Select the events in the location "Chicago". What is the price of the first event in listing?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for a 1440x2960 canvas.
# Available variables: canvas (PIL.Image), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
status_bar_color = (190, 190, 190)        # light grey status bar
header_underline_color = (27, 92, 235)    # blue underline for search
divider_color = (230, 230, 235)           # subtle dividers
card_bg = (250, 250, 252)                 # very light card background
card_border = (220, 220, 225)             # card border
bottom_divider = (224, 224, 228)          # top divider for bottom nav

W, H = canvas.size

# Status bar
status_h = 72
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# Subtle top hairline under status bar
draw.line([(0, status_h), (W, status_h)], fill=divider_color, width=1)

# Header/search area (background is left white to match page; draw only underline)
header_top = status_h
header_bottom = 168
# Blue underline for active search (placed near bottom of header area)
underline_thickness = 6
draw.rectangle([(48, header_bottom - underline_thickness), (W - 48, header_bottom)], fill=header_underline_color)

# Light divider under header area
draw.line([(0, header_bottom), (W, header_bottom)], fill=divider_color, width=1)

# Popular section separators (subtle horizontal separators for list items)
popular_start_x = 48
popular_end_x = W - 48
popular_separators_y = [360, 430, 500, 570, 640]  # positions between "Popular" items
for y in popular_separators_y:
    draw.line([(popular_start_x, y), (popular_end_x, y)], fill=divider_color, width=1)

# Card-like backgrounds for Events list items.
# Use the detected event item bounding boxes as guides (do not draw any text or icons).
event_boxes = [
    (48, 1117, 48 + 1344, 1117 + 396),
    (48, 1513, 48 + 1344, 1513 + 396),
    (48, 1909, 48 + 1344, 1909 + 396),
    (48, 2305, 48 + 1344, 2305 + 396),
]
for (x1, y1, x2, y2) in event_boxes:
    # Slightly inset fill to create separation from edges
    inset = 0
    rx1, ry1, rx2, ry2 = x1 + inset, y1 + inset, x2 - inset, y2 - inset
    # Rounded rectangle background and border
    try:
        draw.rounded_rectangle([(rx1, ry1), (rx2, ry2)], radius=12, fill=card_bg, outline=card_border, width=1)
    except Exception:
        # Fallback if rounded_rectangle not available
        draw.rectangle([(rx1, ry1), (rx2, ry2)], fill=card_bg, outline=card_border)

    # subtle separator line below each card
    draw.line([(rx1 + 12, ry2), (rx2 - 12, ry2)], fill=divider_color, width=1)

# Small separators between individual event rows inside the event list area
# (These are thin horizontal rules to visually separate stacked content blocks)
extra_sep_ys = [ (y2 + 20) for (_, _, _, y2) in event_boxes if y2 + 20 < H - 200 ]
for y in extra_sep_ys:
    draw.line([(48, y), (W - 48, y)], fill=(245,245,247), width=1)

# Bottom navigation area (background + top divider)
bottom_nav_top = 2804
draw.rectangle([(0, bottom_nav_top), (W, H)], fill=(255, 255, 255))
draw.line([(0, bottom_nav_top), (W, bottom_nav_top)], fill=bottom_divider, width=2)

# Final subtle overall vertical rhythm guides (very faint) to help alignment (non-intrusive)
guide_color = (245, 245, 247)
for gx in (48, 312, 576, 840, 1104):
    draw.line([(gx, header_bottom + 8), (gx, bottom_nav_top - 8)], fill=guide_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/00_icon_Fitness_Equipr.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 2305), _c0)
except Exception:
    pass
layout["Fitness_Equipr"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/01_icon_ycar_olds.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1117), _c1)
except Exception:
    pass
layout["'ycar_olds"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/02_icon_Fitness.png
try:
    _c2 = get_crop(2, 1344, 191)
    canvas.paste(_c2, (48, 72), _c2)
except Exception:
    pass
layout["Fitness]"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/03_icon_8_344_creator_followers.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1117), _c3)
except Exception:
    pass
layout["8_344_creator_followers"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/04_icon_7.09.png
try:
    _c4 = get_crop(4, 128, 109)
    canvas.paste(_c4, (52, 115), _c4)
except Exception:
    pass
layout["7.09"] = [52, 115, 180, 224]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/05_icon_Fitness.png
try:
    _c5 = get_crop(5, 57, 59)
    canvas.paste(_c5, (312, 4), _c5)
except Exception:
    pass
layout["Fitness]"] = [312, 4, 369, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/06_icon_7.09.png
try:
    _c6 = get_crop(6, 53, 58)
    canvas.paste(_c6, (183, 4), _c6)
except Exception:
    pass
layout["7.09"] = [183, 4, 236, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/07_icon_7.09.png
try:
    _c7 = get_crop(7, 58, 60)
    canvas.paste(_c7, (114, 3), _c7)
except Exception:
    pass
layout["7.09"] = [114, 3, 172, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 45, 55)
    canvas.paste(_c8, (252, 6), _c8)
except Exception:
    pass
layout["icon_8"] = [252, 6, 297, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/09_icon_Frt.png
try:
    _c9 = get_crop(9, 288, 156)
    canvas.paste(_c9, (864, 2804), _c9)
except Exception:
    pass
layout["Frt"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/10_icon_Low_Impact_Fitness.png
try:
    _c10 = get_crop(10, 1344, 396)
    canvas.paste(_c10, (48, 1909), _c10)
except Exception:
    pass
layout["Low_Impact_Fitness"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/11_icon_Fort_Hamilton_Senior_Recreation_Center.png
try:
    _c11 = get_crop(11, 1344, 396)
    canvas.paste(_c11, (48, 1513), _c11)
except Exception:
    pass
layout["Fort_Hamilton_Senior_Recr"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/12_icon_Cancel.png
try:
    _c12 = get_crop(12, 95, 63)
    canvas.paste(_c12, (1215, 0), _c12)
except Exception:
    pass
layout["Cancel"] = [1215, 0, 1310, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/13_icon_Cancel.png
try:
    _c13 = get_crop(13, 48, 60)
    canvas.paste(_c13, (1322, 2), _c13)
except Exception:
    pass
layout["Cancel"] = [1322, 2, 1370, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/14_icon_Cancel.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1099, 96), _c14)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/15_icon_8_211_creator_followers.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (288, 2804), _c15)
except Exception:
    pass
layout["8_211_creator_followers"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/16_icon_free_fitness.png
try:
    _c16 = get_crop(16, 1344, 120)
    canvas.paste(_c16, (48, 618), _c16)
except Exception:
    pass
layout["free_fitness"] = [48, 618, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/17_icon_More.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (1152, 2804), _c17)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/18_icon_Intro_to_Fitness_Equipment.png
try:
    _c18 = get_crop(18, 1344, 396)
    canvas.paste(_c18, (48, 2305), _c18)
except Exception:
    pass
layout["Intro_to_Fitness_Equipmen"] = [48, 2305, 1392, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/19_icon_Home.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (0, 2804), _c19)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/20_icon_Chair_Fitness.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 1513), _c20)
except Exception:
    pass
layout["Chair_Fitness"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/21_icon_Strive_2_Fitness_Jump_into_Fitness.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1117), _c21)
except Exception:
    pass
layout["Strive_2_Fitness_Jump_int"] = [48, 1117, 1392, 1513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/22_icon_fitness_classes.png
try:
    _c22 = get_crop(22, 1344, 120)
    canvas.paste(_c22, (48, 378), _c22)
except Exception:
    pass
layout["fitness_classes"] = [48, 378, 1392, 498]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/23_icon_Cancel.png
try:
    _c23 = get_crop(23, 149, 144)
    canvas.paste(_c23, (1243, 97), _c23)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/24_icon_9_30_AM_EDT.png
try:
    _c24 = get_crop(24, 1344, 396)
    canvas.paste(_c24, (48, 1909), _c24)
except Exception:
    pass
layout["9:30_AM_EDT"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/25_icon_Fort_Hamilton_Senior_Recreation_Center.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 1513), _c25)
except Exception:
    pass
layout["Fort_Hamilton_Senior_Recr"] = [48, 1513, 1392, 1909]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/26_icon_Tiemeyer_Park.png
try:
    _c26 = get_crop(26, 233, 51)
    canvas.paste(_c26, (391, 1323), _c26)
except Exception:
    pass
layout["Tiemeyer_Park"] = [391, 1323, 624, 1374]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 46, 61)
    canvas.paste(_c27, (385, 3), _c27)
except Exception:
    pass
layout["icon_27"] = [385, 3, 431, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/28_icon_Low_Impact_Fitness.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 1909), _c28)
except Exception:
    pass
layout["Low_Impact_Fitness"] = [48, 1909, 1392, 2305]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/29_text_7.09.png
try:
    _c29 = get_crop(29, 91, 45)
    canvas.paste(_c29, (20, 15), _c29)
except Exception:
    pass
layout["7.09"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/30_text_Popular.png
try:
    _c30 = get_crop(30, 221, 78)
    canvas.paste(_c30, (44, 298), _c30)
except Exception:
    pass
layout["Popular"] = [44, 298, 265, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/31_text_fitness_expo.png
try:
    _c31 = get_crop(31, 238, 52)
    canvas.paste(_c31, (159, 550), _c31)
except Exception:
    pass
layout["fitness_expo"] = [159, 550, 397, 602]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/32_text_sports_and_fitness.png
try:
    _c32 = get_crop(32, 1344, 120)
    canvas.paste(_c32, (48, 738), _c32)
except Exception:
    pass
layout["sports_and_fitness"] = [48, 738, 1392, 858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/33_text_yoga_and_fitness_classes.png
try:
    _c33 = get_crop(33, 1344, 144)
    canvas.paste(_c33, (48, 858), _c33)
except Exception:
    pass
layout["yoga_and_fitness_classes"] = [48, 858, 1392, 1002]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/34_text_Events.png
try:
    _c34 = get_crop(34, 189, 57)
    canvas.paste(_c34, (46, 1029), _c34)
except Exception:
    pass
layout["Events"] = [46, 1029, 235, 1086]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/35_text_Frt.png
try:
    _c35 = get_crop(35, 60, 12)
    canvas.paste(_c35, (771, 2794), _c35)
except Exception:
    pass
layout["Frt"] = [771, 2794, 831, 2806]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/36_clickable_fitness_expo.png
try:
    _c36 = get_crop(36, 1344, 120)
    canvas.paste(_c36, (48, 498), _c36)
except Exception:
    pass
layout["fitness_expo"] = [48, 498, 1392, 618]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_03_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-5/37_clickable_Favorites.png
try:
    _c37 = get_crop(37, 288, 156)
    canvas.paste(_c37, (576, 2804), _c37)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]
