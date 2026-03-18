# page_id: page_eventbrite_4c12747d6cb14426aff8b07be52a617b_07
# screenshot: 2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9.png
# step_index: 7/11
# task: Open Eventbrite. Search 'Art'. Filter event type "Performance". Select the first event. Follow the organizer and save the event to favorite. What is the price of the ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# canvas: PIL Image (1440x2960 RGB)
# draw: PIL ImageDraw object
# font_sm, font_md, font_lg, font_xl available

w, h = canvas.size

# Colors
bg_white = (255, 255, 255)
status_bar_color = (190, 190, 190)        # top status bar neutral gray
header_bg = (255, 255, 255)               # header white
divider = (230, 230, 235)                 # subtle divider
muted_panel = (247, 249, 251)             # very light panel background
muted_blue_panel = (239, 249, 255)        # faint bluish panel for chip groups
shadow_color = (220, 220, 225)

# Fill overall background (ensure clean)
draw.rectangle([(0, 0), (w, h)], fill=bg_white)

# Status bar area (~0..96px)
status_h = 96
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# Header / toolbar area (below status bar)
header_top = status_h
header_bottom = 176
draw.rectangle([(0, header_top), (w, header_bottom)], fill=header_bg)

# Header bottom divider + subtle shadow line
draw.line([(0, header_bottom), (w, header_bottom)], fill=divider, width=1)
# small shadow under header
for i, a in enumerate([1, 2, 4]):
    y = header_bottom + i
    alpha_shade = max(0, shadow_color[0] - a*6)
    shade = (alpha_shade, alpha_shade, alpha_shade)
    draw.line([(0, y), (w, y)], fill=shade, width=1)

# Section background cards (rounded rectangles behind groups of chips)
# Categories group (around y ~320..520)
cat_box = (24, 320, w-24, 540)
draw.rounded_rectangle(cat_box, radius=20, fill=muted_blue_panel, outline=None)

# Event type group (big group of many chips, around 760..1260)
etype_box = (24, 760, w-24, 1260)
draw.rounded_rectangle(etype_box, radius=22, fill=muted_blue_panel, outline=None)

# Languages group (around 1620..1800)
lang_box = (24, 1620, w-24, 1830)
draw.rounded_rectangle(lang_box, radius=20, fill=muted_blue_panel, outline=None)

# Price / toggle area - subtle panel for Price section (small)
price_box = (24, 1960, w-24, 2068)
draw.rounded_rectangle(price_box, radius=16, fill=muted_panel, outline=None)

# Sort by segmented control background (subtle long rounded rectangle)
sort_box = (36, 2368, w-36, 2468)
draw.rounded_rectangle(sort_box, radius=18, fill=muted_panel, outline=None)
# inner divider between segments (approx center)
draw.line([(w/2, 2368+8), (w/2, 2468-8)], fill=divider, width=1)

# Subtle separators between major sections (to help structure)
separators_y = [
    300,   # above categories
    740,   # above event types
    1560,  # above languages
    1940,  # above price
    2328,  # above sort by
    2740   # above apply filters bar
]
for y in separators_y:
    draw.line([(36, y), (w-36, y)], fill=divider, width=1)

# Additional subtle left/right edge rails (visual margins)
rail_color = (250, 250, 251)
draw.rectangle([(0, header_bottom+1), (24, h-1)], fill=rail_color)
draw.rectangle([(w-24, header_bottom+1), (w, h-1)], fill=rail_color)

# Top-left small horizontal notch area behind status icons (slightly darker)
draw.rectangle([(0, status_h-6), (w, status_h)], fill=divider)

# Final faint top/bottom rounding for long panels to match mobile feel
# (soften corners of large event-type panel with a thin outline)
draw.rounded_rectangle(etype_box, radius=22, outline=(235,235,238), width=1)

# Ensure we do not draw or overlay any detected text or icons.
# (All drawn elements are pure backgrounds, dividers, and containers.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/00_icon_Game.png
try:
    _c0 = get_crop(0, 185, 144)
    canvas.paste(_c0, (749, 1083), _c0)
except Exception:
    pass
layout["Game"] = [749, 1083, 934, 1227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/01_icon_Food_Drink.png
try:
    _c1 = get_crop(1, 312, 144)
    canvas.paste(_c1, (512, 383), _c1)
except Exception:
    pass
layout["Food_&_Drink"] = [512, 383, 824, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/02_icon_French.png
try:
    _c2 = get_crop(2, 205, 144)
    canvas.paste(_c2, (768, 1656), _c2)
except Exception:
    pass
layout["French"] = [768, 1656, 973, 1800]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/03_icon_Tournament.png
try:
    _c3 = get_crop(3, 302, 144)
    canvas.paste(_c3, (423, 1083), _c3)
except Exception:
    pass
layout["Tournament"] = [423, 1083, 725, 1227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/04_icon_Spanish.png
try:
    _c4 = get_crop(4, 225, 144)
    canvas.paste(_c4, (519, 1656), _c4)
except Exception:
    pass
layout["Spanish"] = [519, 1656, 744, 1800]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/05_icon_Race.png
try:
    _c5 = get_crop(5, 169, 144)
    canvas.paste(_c5, (958, 1083), _c5)
except Exception:
    pass
layout["Race"] = [958, 1083, 1127, 1227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/06_icon_Community.png
try:
    _c6 = get_crop(6, 294, 144)
    canvas.paste(_c6, (848, 383), _c6)
except Exception:
    pass
layout["Community"] = [848, 383, 1142, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/07_icon_Performance.png
try:
    _c7 = get_crop(7, 320, 144)
    canvas.paste(_c7, (36, 956), _c7)
except Exception:
    pass
layout["Performance"] = [36, 956, 356, 1100]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/08_icon_Party.png
try:
    _c8 = get_crop(8, 173, 127)
    canvas.paste(_c8, (36, 1083), _c8)
except Exception:
    pass
layout["Party"] = [36, 1083, 209, 1210]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/09_icon_Rally.png
try:
    _c9 = get_crop(9, 166, 144)
    canvas.paste(_c9, (233, 1083), _c9)
except Exception:
    pass
layout["Rally"] = [233, 1083, 399, 1227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/10_icon_Class.png
try:
    _c10 = get_crop(10, 176, 144)
    canvas.paste(_c10, (856, 956), _c10)
except Exception:
    pass
layout["Class"] = [856, 956, 1032, 1100]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/11_icon_Italian.png
try:
    _c11 = get_crop(11, 191, 144)
    canvas.paste(_c11, (997, 1656), _c11)
except Exception:
    pass
layout["Italian"] = [997, 1656, 1188, 1800]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/12_icon_Expo.png
try:
    _c12 = get_crop(12, 167, 144)
    canvas.paste(_c12, (614, 829), _c12)
except Exception:
    pass
layout["Expo"] = [614, 829, 781, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/13_icon_Appearance.png
try:
    _c13 = get_crop(13, 307, 144)
    canvas.paste(_c13, (563, 1210), _c13)
except Exception:
    pass
layout["Appearance"] = [563, 1210, 870, 1354]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/14_icon_Screening.png
try:
    _c14 = get_crop(14, 269, 144)
    canvas.paste(_c14, (380, 956), _c14)
except Exception:
    pass
layout["Screening"] = [380, 956, 649, 1100]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/15_icon_Gala.png
try:
    _c15 = get_crop(15, 159, 144)
    canvas.paste(_c15, (673, 956), _c15)
except Exception:
    pass
layout["Gala"] = [673, 956, 832, 1100]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/16_icon_Convention.png
try:
    _c16 = get_crop(16, 293, 144)
    canvas.paste(_c16, (805, 829), _c16)
except Exception:
    pass
layout["Convention"] = [805, 829, 1098, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/17_icon_Festival.png
try:
    _c17 = get_crop(17, 219, 127)
    canvas.paste(_c17, (1122, 829), _c17)
except Exception:
    pass
layout["Festival"] = [1122, 829, 1341, 956]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/18_icon_German.png
try:
    _c18 = get_crop(18, 225, 135)
    canvas.paste(_c18, (270, 1656), _c18)
except Exception:
    pass
layout["German"] = [270, 1656, 495, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/19_icon_Seminar.png
try:
    _c19 = get_crop(19, 232, 144)
    canvas.paste(_c19, (358, 829), _c19)
except Exception:
    pass
layout["Seminar"] = [358, 829, 590, 973]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/20_icon_Networking.png
try:
    _c20 = get_crop(20, 296, 144)
    canvas.paste(_c20, (1056, 956), _c20)
except Exception:
    pass
layout["Networking"] = [1056, 956, 1352, 1100]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/21_icon_Music.png
try:
    _c21 = get_crop(21, 187, 135)
    canvas.paste(_c21, (36, 383), _c21)
except Exception:
    pass
layout["Music"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/22_icon_Other.png
try:
    _c22 = get_crop(22, 182, 144)
    canvas.paste(_c22, (894, 1210), _c22)
except Exception:
    pass
layout["Other"] = [894, 1210, 1076, 1354]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/23_icon_Retreat.png
try:
    _c23 = get_crop(23, 215, 135)
    canvas.paste(_c23, (324, 1210), _c23)
except Exception:
    pass
layout["Retreat"] = [324, 1210, 539, 1345]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/24_icon_Business.png
try:
    _c24 = get_crop(24, 241, 135)
    canvas.paste(_c24, (247, 383), _c24)
except Exception:
    pass
layout["Business"] = [247, 383, 488, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/25_icon_Tour.png
try:
    _c25 = get_crop(25, 156, 144)
    canvas.paste(_c25, (1151, 1083), _c25)
except Exception:
    pass
layout["Tour"] = [1151, 1083, 1307, 1227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/26_icon_Arts.png
try:
    _c26 = get_crop(26, 152, 144)
    canvas.paste(_c26, (1166, 383), _c26)
except Exception:
    pass
layout["Arts"] = [1166, 383, 1318, 527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/27_icon_Attraction.png
try:
    _c27 = get_crop(27, 264, 135)
    canvas.paste(_c27, (36, 1210), _c27)
except Exception:
    pass
layout["Attraction"] = [36, 1210, 300, 1345]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/28_icon_Conference.png
try:
    _c28 = get_crop(28, 298, 127)
    canvas.paste(_c28, (36, 829), _c28)
except Exception:
    pass
layout["Conference"] = [36, 829, 334, 956]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/29_icon_English.png
try:
    _c29 = get_crop(29, 210, 135)
    canvas.paste(_c29, (36, 1656), _c29)
except Exception:
    pass
layout["English"] = [36, 1656, 246, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/30_icon_Apply_filters_1.png
try:
    _c30 = get_crop(30, 1344, 144)
    canvas.paste(_c30, (48, 2768), _c30)
except Exception:
    pass
layout["Apply_filters_(1)"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/31_icon_Relevance.png
try:
    _c31 = get_crop(31, 660, 144)
    canvas.paste(_c31, (54, 2405), _c31)
except Exception:
    pass
layout["Relevance"] = [54, 2405, 714, 2549]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/32_icon_Date.png
try:
    _c32 = get_crop(32, 660, 144)
    canvas.paste(_c32, (726, 2405), _c32)
except Exception:
    pass
layout["Date"] = [726, 2405, 1386, 2549]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/33_icon_7.52.png
try:
    _c33 = get_crop(33, 144, 144)
    canvas.paste(_c33, (12, 72), _c33)
except Exception:
    pass
layout["7.52"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/34_icon_7.52.png
try:
    _c34 = get_crop(34, 64, 65)
    canvas.paste(_c34, (112, 1), _c34)
except Exception:
    pass
layout["7.52"] = [112, 1, 176, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/35_icon_7.52.png
try:
    _c35 = get_crop(35, 60, 63)
    canvas.paste(_c35, (180, 1), _c35)
except Exception:
    pass
layout["7.52"] = [180, 1, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/36_icon_icon_36.png
try:
    _c36 = get_crop(36, 64, 62)
    canvas.paste(_c36, (308, 3), _c36)
except Exception:
    pass
layout["icon_36"] = [308, 3, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/37_icon_Clear_all.png
try:
    _c37 = get_crop(37, 54, 66)
    canvas.paste(_c37, (1319, 0), _c37)
except Exception:
    pass
layout["Clear_all"] = [1319, 0, 1373, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/38_icon_Clear_all.png
try:
    _c38 = get_crop(38, 100, 64)
    canvas.paste(_c38, (1212, 0), _c38)
except Exception:
    pass
layout["Clear_all"] = [1212, 0, 1312, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/39_icon_icon_39.png
try:
    _c39 = get_crop(39, 51, 62)
    canvas.paste(_c39, (248, 2), _c39)
except Exception:
    pass
layout["icon_39"] = [248, 2, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/40_icon_Toggle_to_show_only_free_events.png
try:
    _c40 = get_crop(40, 144, 144)
    canvas.paste(_c40, (1248, 2110), _c40)
except Exception:
    pass
layout["Toggle_to_show_only_free_"] = [1248, 2110, 1392, 2254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/41_icon_Clear_all.png
try:
    _c41 = get_crop(41, 178, 144)
    canvas.paste(_c41, (1214, 72), _c41)
except Exception:
    pass
layout["Clear_all"] = [1214, 72, 1392, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/42_text_7.52.png
try:
    _c42 = get_crop(42, 91, 45)
    canvas.paste(_c42, (20, 15), _c42)
except Exception:
    pass
layout["7.52"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/43_text_Filters.png
try:
    _c43 = get_crop(43, 180, 66)
    canvas.paste(_c43, (631, 116), _c43)
except Exception:
    pass
layout["Filters"] = [631, 116, 811, 182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/44_text_Categories.png
try:
    _c44 = get_crop(44, 187, 135)
    canvas.paste(_c44, (36, 383), _c44)
except Exception:
    pass
layout["Categories"] = [36, 383, 223, 518]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/45_text_Show_all_categories.png
try:
    _c45 = get_crop(45, 516, 144)
    canvas.paste(_c45, (0, 518), _c45)
except Exception:
    pass
layout["Show_all_categories"] = [0, 518, 516, 662]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/46_text_Event_type.png
try:
    _c46 = get_crop(46, 298, 127)
    canvas.paste(_c46, (36, 829), _c46)
except Exception:
    pass
layout["Event_type"] = [36, 829, 334, 956]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/47_text_Show_less_event_types.png
try:
    _c47 = get_crop(47, 569, 144)
    canvas.paste(_c47, (0, 1345), _c47)
except Exception:
    pass
layout["Show_less_event_types"] = [0, 1345, 569, 1489]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/48_text_Languages.png
try:
    _c48 = get_crop(48, 210, 135)
    canvas.paste(_c48, (36, 1656), _c48)
except Exception:
    pass
layout["Languages"] = [36, 1656, 246, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/49_text_Show_all_languages.png
try:
    _c49 = get_crop(49, 511, 144)
    canvas.paste(_c49, (0, 1791), _c49)
except Exception:
    pass
layout["Show_all_languages"] = [0, 1791, 511, 1935]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/50_text_Price.png
try:
    _c50 = get_crop(50, 149, 63)
    canvas.paste(_c50, (45, 1995), _c50)
except Exception:
    pass
layout["Price"] = [45, 1995, 194, 2058]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/51_text_Only_free_events.png
try:
    _c51 = get_crop(51, 660, 144)
    canvas.paste(_c51, (54, 2405), _c51)
except Exception:
    pass
layout["Only_free_events"] = [54, 2405, 714, 2549]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4c12747d6cb14426aff8b07be52a617b/step_07_2024_4_23_19_51_4c12747d6cb14426aff8b07be52a617b-9/52_text_Sort_by.png
try:
    _c52 = get_crop(52, 204, 74)
    canvas.paste(_c52, (42, 2313), _c52)
except Exception:
    pass
layout["Sort_by"] = [42, 2313, 246, 2387]
