# page_id: page_eventbrite_f01eaa41f6284da09deb7ced3e4eea4e_01
# screenshot: 2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3.png
# step_index: 1/11
# task: Open Eventbrite. Check out 'Sports' events. Apply filters for events happening this week. Select the first event. Check similar events and add the first similar event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint background and structural elements for the mobile UI mockup
# Uses available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Overall background (slightly warm-off-white to match screenshot)
bg_color = (252, 251, 255)  # very light lavender/white
draw.rectangle([(0, 0), canvas.size], fill=bg_color)

# Status bar (top ~72px) - muted gray area
status_h = 72
status_color = (195, 195, 195)
draw.rectangle([(0, 0), (canvas.width, status_h)], fill=status_color)

# Thin separator under status bar
draw.line([(0, status_h), (canvas.width, status_h)], fill=(210, 210, 210), width=1)

# Header / toolbar area (search bar sits here in actual UI; we only draw background and divider)
header_top = status_h
header_h = 120
header_color = bg_color  # keep header matching page background
draw.rectangle([(0, header_top), (canvas.width, header_top + header_h)], fill=header_color)

# Divider under header
divider_y = header_top + header_h - 8
draw.line([(48, divider_y), (canvas.width - 48, divider_y)], fill=(235, 232, 240), width=1)

# Function to draw a card background with subtle shadow
def draw_card(x, y, w, h, radius=18):
    # shadow
    shadow_color = (241, 241, 245)
    shadow_offset = (4, 8)
    draw.rounded_rectangle(
        [(x + shadow_offset[0], y + shadow_offset[1]), (x + w + shadow_offset[0], y + h + shadow_offset[1])],
        radius=radius, fill=shadow_color
    )
    # main card
    card_color = (255, 255, 255)
    outline_color = (236, 234, 241)
    draw.rounded_rectangle([(x, y), (x + w, y + h)], radius=radius, fill=card_color, outline=outline_color, width=1)

# Event cards positions (from detected crops) - draw background cards only
cards = [
    (48, 490, 1344, 396),
    (48, 886, 1344, 396),
    (48, 1282, 1344, 396),
    (48, 1678, 1344, 396),
    (48, 2074, 1344, 396),
    (48, 2470, 1344, 346),
]

for (x, y, w, h) in cards:
    draw_card(x, y, w, h, radius=16)
    # subtle separator line under each card (keeps spacing consistent)
    sep_y = y + h + 12
    draw.line([(48, sep_y), (canvas.width - 48, sep_y)], fill=(245, 244, 247), width=1)

# Additional subtle section header band where the page title sits (visual grouping)
section_band_y = 430
band_h = 80
band_color = (250, 247, 254)  # very subtle tint to separate title area
draw.rectangle([(48, section_band_y), (canvas.width - 48, section_band_y + band_h)], fill=band_color, outline=None)
# thin bottom border for band
draw.line([(48, section_band_y + band_h), (canvas.width - 48, section_band_y + band_h)], fill=(236, 234, 241), width=1)

# Floating bottom navigation area background
nav_top = 2804
nav_color = (255, 255, 255)
draw.rectangle([(0, nav_top), (canvas.width, canvas.height)], fill=nav_color)
# top border for nav
draw.line([(0, nav_top), (canvas.width, nav_top)], fill=(230, 228, 235), width=2)

# Safe-area horizontal guidelines (subtle) to visually constrain content width - not UI elements
side_margin = 48
guide_color = (255, 255, 255, 0)  # no-op visually but kept for structure (transparent intent)

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/00_icon_ering_to_soothe_the_brokel.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1678), _c0)
except Exception:
    pass
layout["ering_to_soothe_the_broke"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/01_icon_Spring-Zing_Happy_Hour.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 2074), _c1)
except Exception:
    pass
layout["Spring-Zing_Happy_Hour"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/02_icon_NDIE.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 490), _c2)
except Exception:
    pass
layout["NDIE"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/03_icon_Sat.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 886), _c3)
except Exception:
    pass
layout["Sat,"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/04_icon_Q_Search_events.png
try:
    _c4 = get_crop(4, 1179, 144)
    canvas.paste(_c4, (195, 93), _c4)
except Exception:
    pass
layout["Q_Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/05_icon_San_Francisco.png
try:
    _c5 = get_crop(5, 495, 117)
    canvas.paste(_c5, (473, 2651), _c5)
except Exception:
    pass
layout["San_Francisco"] = [473, 2651, 968, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/06_icon_Spring-Zing_Happy.png
try:
    _c6 = get_crop(6, 144, 123)
    canvas.paste(_c6, (1140, 1951), _c6)
except Exception:
    pass
layout["Spring-Zing_Happy"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 139)
    canvas.paste(_c7, (1140, 747), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/08_icon_City.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 1539), _c8)
except Exception:
    pass
layout["City"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/09_icon_Bissa.png
try:
    _c9 = get_crop(9, 288, 156)
    canvas.paste(_c9, (288, 2804), _c9)
except Exception:
    pass
layout["Bissa}"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/10_icon_Reggaeton.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1140, 2347), _c10)
except Exception:
    pass
layout["Reggaeton__"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/11_icon_Spring-Zing_Happy.png
try:
    _c11 = get_crop(11, 144, 123)
    canvas.paste(_c11, (1284, 1951), _c11)
except Exception:
    pass
layout["Spring-Zing_Happy"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/12_icon_Overflow_menu_button.png
try:
    _c12 = get_crop(12, 144, 139)
    canvas.paste(_c12, (1284, 747), _c12)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/13_icon_4.35.png
try:
    _c13 = get_crop(13, 109, 103)
    canvas.paste(_c13, (38, 120), _c13)
except Exception:
    pass
layout["4.35"] = [38, 120, 147, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/14_icon_City.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1284, 1539), _c14)
except Exception:
    pass
layout["City"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/15_icon_Reggaeton.png
try:
    _c15 = get_crop(15, 144, 123)
    canvas.paste(_c15, (1284, 2347), _c15)
except Exception:
    pass
layout["Reggaeton__"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/16_icon_SatvaonG.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (0, 2804), _c16)
except Exception:
    pass
layout["SatvaonG"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/17_icon_4.35.png
try:
    _c17 = get_crop(17, 54, 60)
    canvas.paste(_c17, (184, 2), _c17)
except Exception:
    pass
layout["4.35"] = [184, 2, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/18_icon_City.png
try:
    _c18 = get_crop(18, 144, 139)
    canvas.paste(_c18, (1140, 1143), _c18)
except Exception:
    pass
layout["City"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 58, 57)
    canvas.paste(_c19, (313, 4), _c19)
except Exception:
    pass
layout["icon_19"] = [313, 4, 371, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/20_icon_Overflow_menu_button.png
try:
    _c20 = get_crop(20, 144, 139)
    canvas.paste(_c20, (1284, 1143), _c20)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/21_icon_PDO_Thread_Training.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1282), _c21)
except Exception:
    pass
layout["PDO_Thread_Training_|"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 47, 58)
    canvas.paste(_c22, (250, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [250, 3, 297, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 46, 52)
    canvas.paste(_c23, (1322, 8), _c23)
except Exception:
    pass
layout["icon_23"] = [1322, 8, 1368, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/24_icon_4.35.png
try:
    _c24 = get_crop(24, 58, 60)
    canvas.paste(_c24, (115, 3), _c24)
except Exception:
    pass
layout["4.35"] = [115, 3, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/25_icon_8_60_creator_followers.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 490), _c25)
except Exception:
    pass
layout["8_60_creator_followers"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/26_icon_8_30_creator_followers.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 886), _c26)
except Exception:
    pass
layout["8_30_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/27_icon_8_100_creator_followers.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 1678), _c27)
except Exception:
    pass
layout["8_100_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 56, 57)
    canvas.paste(_c28, (1213, 5), _c28)
except Exception:
    pass
layout["icon_28"] = [1213, 5, 1269, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/29_icon_Free.png
try:
    _c29 = get_crop(29, 125, 73)
    canvas.paste(_c29, (248, 561), _c29)
except Exception:
    pass
layout["Free"] = [248, 561, 373, 634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/30_icon_Sales_ended.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 1282), _c30)
except Exception:
    pass
layout["Sales_ended"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 41, 54)
    canvas.paste(_c31, (1272, 7), _c31)
except Exception:
    pass
layout["icon_31"] = [1272, 7, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/32_icon_Area.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 2074), _c32)
except Exception:
    pass
layout["Area"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/33_icon_Salsa.png
try:
    _c33 = get_crop(33, 1344, 346)
    canvas.paste(_c33, (48, 2470), _c33)
except Exception:
    pass
layout["Salsa"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/34_icon_Grief_Medicine_A_Gathering_to_Soothe_the.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 1678), _c34)
except Exception:
    pass
layout["Grief_Medicine:_A_Gatheri"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/35_icon_Q_Search_events.png
try:
    _c35 = get_crop(35, 43, 55)
    canvas.paste(_c35, (385, 7), _c35)
except Exception:
    pass
layout["Q_Search_events"] = [385, 7, 428, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/36_icon_Processing_Grief_Self-Care_for_Loss.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 886), _c36)
except Exception:
    pass
layout["Processing_Grief:_Self-Ca"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/37_icon_Yggae.png
try:
    _c37 = get_crop(37, 288, 156)
    canvas.paste(_c37, (864, 2804), _c37)
except Exception:
    pass
layout["Yggae"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/38_icon_Yggae.png
try:
    _c38 = get_crop(38, 150, 68)
    canvas.paste(_c38, (933, 2643), _c38)
except Exception:
    pass
layout["Yggae"] = [933, 2643, 1083, 2711]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/39_text_4.35.png
try:
    _c39 = get_crop(39, 92, 43)
    canvas.paste(_c39, (22, 17), _c39)
except Exception:
    pass
layout["4.35"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/40_text_More_events_you_II_love.png
try:
    _c40 = get_crop(40, 1344, 396)
    canvas.paste(_c40, (48, 490), _c40)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/41_text_Mon.png
try:
    _c41 = get_crop(41, 92, 43)
    canvas.paste(_c41, (393, 2129), _c41)
except Exception:
    pass
layout["Mon,"] = [393, 2129, 485, 2172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/42_text_13.png
try:
    _c42 = get_crop(42, 54, 38)
    canvas.paste(_c42, (561, 2129), _c42)
except Exception:
    pass
layout["13"] = [561, 2129, 615, 2167]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/43_text_5_00_PM_PDT.png
try:
    _c43 = get_crop(43, 1344, 396)
    canvas.paste(_c43, (48, 2074), _c43)
except Exception:
    pass
layout["5:00_PM_PDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/44_text_3600_16th_St.png
try:
    _c44 = get_crop(44, 223, 38)
    canvas.paste(_c44, (392, 2328), _c44)
except Exception:
    pass
layout["3600_16th_St"] = [392, 2328, 615, 2366]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/45_text_8_9294_creator_followers.png
try:
    _c45 = get_crop(45, 1344, 396)
    canvas.paste(_c45, (48, 2074), _c45)
except Exception:
    pass
layout["8_9294_creator_followers"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/46_text_Aanonananal.png
try:
    _c46 = get_crop(46, 194, 14)
    canvas.paste(_c46, (98, 2542), _c46)
except Exception:
    pass
layout["Aanonananal"] = [98, 2542, 292, 2556]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/47_text_Sat_May_4.png
try:
    _c47 = get_crop(47, 186, 43)
    canvas.paste(_c47, (392, 2525), _c47)
except Exception:
    pass
layout["Sat,_May_4"] = [392, 2525, 578, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/48_text_IO_00_AM_PDT.png
try:
    _c48 = get_crop(48, 1344, 346)
    canvas.paste(_c48, (48, 2470), _c48)
except Exception:
    pass
layout["IO:00_AM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/49_text_hellaGood.png
try:
    _c49 = get_crop(49, 186, 41)
    canvas.paste(_c49, (101, 2556), _c49)
except Exception:
    pass
layout["hellaGood"] = [101, 2556, 287, 2597]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/50_text_ssan.png
try:
    _c50 = get_crop(50, 25, 9)
    canvas.paste(_c50, (252, 2636), _c50)
except Exception:
    pass
layout["ssan"] = [252, 2636, 277, 2645]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/51_text_featuring-.png
try:
    _c51 = get_crop(51, 43, 15)
    canvas.paste(_c51, (215, 2650), _c51)
except Exception:
    pass
layout["'featuring-"] = [215, 2650, 258, 2665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/52_text_Jah_Wafeidk_SHELTER.png
try:
    _c52 = get_crop(52, 129, 13)
    canvas.paste(_c52, (142, 2702), _c52)
except Exception:
    pass
layout["Jah_Wafeidk_SHELTER"] = [142, 2702, 271, 2715]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/53_text_DJGREENB_DJAGANA_DJMALIIGZ.png
try:
    _c53 = get_crop(53, 215, 18)
    canvas.paste(_c53, (91, 2718), _c53)
except Exception:
    pass
layout["DJGREENB_DJAGANA_DJMALIIG"] = [91, 2718, 306, 2736]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/54_text_Log_AETa.png
try:
    _c54 = get_crop(54, 41, 6)
    canvas.paste(_c54, (111, 2738), _c54)
except Exception:
    pass
layout["Log__AETa"] = [111, 2738, 152, 2744]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/55_text_atrobcats.png
try:
    _c55 = get_crop(55, 43, 13)
    canvas.paste(_c55, (156, 2746), _c55)
except Exception:
    pass
layout["atrobcats"] = [156, 2746, 199, 2759]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/56_text_nalceuani.png
try:
    _c56 = get_crop(56, 37, 7)
    canvas.paste(_c56, (212, 2742), _c56)
except Exception:
    pass
layout["nalceuani"] = [212, 2742, 249, 2749]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/57_text_Lrocaa_Rnrae.png
try:
    _c57 = get_crop(57, 53, 9)
    canvas.paste(_c57, (240, 2763), _c57)
except Exception:
    pass
layout["Lrocaa_Rnrae"] = [240, 2763, 293, 2772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/58_text_SatvaonG.png
try:
    _c58 = get_crop(58, 60, 29)
    canvas.paste(_c58, (92, 2761), _c58)
except Exception:
    pass
layout["SatvaonG"] = [92, 2761, 152, 2790]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/59_text_t0ph.png
try:
    _c59 = get_crop(59, 32, 15)
    canvas.paste(_c59, (158, 2767), _c59)
except Exception:
    pass
layout["t0ph"] = [158, 2767, 190, 2782]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/60_text_Z44.png
try:
    _c60 = get_crop(60, 23, 15)
    canvas.paste(_c60, (197, 2767), _c60)
except Exception:
    pass
layout["Z44"] = [197, 2767, 220, 2782]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/61_text_71J_Nissiom_St.st.png
try:
    _c61 = get_crop(61, 74, 13)
    canvas.paste(_c61, (232, 2774), _c61)
except Exception:
    pass
layout["{71J_Nissiom_St.st"] = [232, 2774, 306, 2787]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/62_text_9_232_creator_followers.png
try:
    _c62 = get_crop(62, 1344, 346)
    canvas.paste(_c62, (48, 2470), _c62)
except Exception:
    pass
layout["9_232_creator_followers"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/63_clickable_Favorites.png
try:
    _c63 = get_crop(63, 288, 156)
    canvas.paste(_c63, (576, 2804), _c63)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_01_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-3/64_clickable_More.png
try:
    _c64 = get_crop(64, 288, 156)
    canvas.paste(_c64, (1152, 2804), _c64)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
