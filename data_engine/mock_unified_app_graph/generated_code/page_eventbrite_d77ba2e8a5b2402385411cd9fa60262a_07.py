# page_id: page_eventbrite_d77ba2e8a5b2402385411cd9fa60262a_07
# screenshot: 2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9.png
# step_index: 7/8
# task: Open Eventbrite. Search for "Music". Filter only free events. Choose the first event. When is the date and timing?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and UI structure for Eventbrite-like page
# Uses provided variables: canvas (PIL.Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
bg_color = (250, 251, 252)        # very light background
statusbar_color = (189, 189, 189) # grey status bar
toolbar_color = (255, 255, 255)   # white toolbar area
divider_color = (224, 224, 224)   # light divider lines
card_bg = (255, 255, 255)         # card white
card_shadow = (235, 236, 238)     # subtle shadow for cards
chip_area_bg = (250, 250, 253)    # slight tint behind filter chips
bottom_nav_bg = (255, 255, 255)   # bottom nav bar background

w, h = canvas.size

# Fill whole background
draw.rectangle([0, 0, w, h], fill=bg_color)

# Status bar (top bar where time/signal sit) ~96px height
status_h = 96
draw.rectangle([0, 0, w, status_h], fill=statusbar_color)

# Toolbar / Search header area below status bar
toolbar_top = status_h
toolbar_bottom = 220
# subtle drop shadow under toolbar: a thin line
draw.rectangle([0, toolbar_top, w, toolbar_bottom], fill=toolbar_color)
draw.line([(32, toolbar_bottom), (w-32, toolbar_bottom)], fill=divider_color, width=2)

# Area for location row / filter chips (we only draw background bands, not chips/text)
chip_band_top = toolbar_bottom + 20
chip_band_bottom = chip_band_top + 160
draw.rectangle([0, chip_band_top, w, chip_band_bottom], fill=chip_area_bg)
# separate the chip band from content with a subtle divider
draw.line([(48, chip_band_bottom), (w-48, chip_band_bottom)], fill=divider_color, width=1)

# Content margin and card width (matches detected elements margins)
margin_x = 48
content_width = w - 2 * margin_x

# First event card background (includes image area + title area)
# Using detected image at y=676 height ~1096, we create a card that encloses image + metadata area
card1_top = 600
card1_bottom = 1780
card1_coords = (margin_x, card1_top, margin_x + content_width, card1_bottom)
# shadow
shadow_offset = 8
draw.rounded_rectangle(
    [card1_coords[0] + shadow_offset, card1_coords[1] + shadow_offset,
     card1_coords[2] + shadow_offset, card1_coords[3] + shadow_offset],
    radius=28, fill=card_shadow)
# main card
draw.rounded_rectangle(card1_coords, radius=28, fill=card_bg)
# thin divider at bottom of card1 (separates from next content)
draw.line([(margin_x + 24, card1_bottom), (margin_x + content_width - 24, card1_bottom)], fill=divider_color, width=1)

# Second event card background (encloses second large image + metadata)
card2_top = 1800
card2_bottom = 2830
card2_coords = (margin_x, card2_top, margin_x + content_width, card2_bottom)
# shadow
draw.rounded_rectangle(
    [card2_coords[0] + shadow_offset, card2_coords[1] + shadow_offset,
     card2_coords[2] + shadow_offset, card2_coords[3] + shadow_offset],
    radius=28, fill=card_shadow)
# main card
draw.rounded_rectangle(card2_coords, radius=28, fill=card_bg)
# subtle divider between cards area (above card2)
draw.line([(margin_x + 16, card2_top - 16), (margin_x + content_width - 16, card2_top - 16)], fill=divider_color, width=1)

# Top banner / large hero area behind first page title (do not draw text)
# Provide a subtle horizontal rule under the header area
draw.line([(48, 320), (w-48, 320)], fill=divider_color, width=1)

# Bottom navigation bar background and top divider
nav_height = 120
nav_top = h - nav_height
draw.rectangle([0, nav_top, w, h], fill=bottom_nav_bg)
draw.line([(32, nav_top), (w-32, nav_top)], fill=divider_color, width=2)

# Safe area accents: small left edge separators to guide layout (non-intrusive)
# Vertical guide lines near left margin (very faint)
draw.line([(margin_x, toolbar_bottom + 6), (margin_x, h - nav_height - 6)], fill=(245,245,245), width=1)
draw.line([(w - margin_x, toolbar_bottom + 6), (w - margin_x, h - nav_height - 6)], fill=(245,245,245), width=1)

# Done drawing structural UI elements. (No icons or text are drawn here.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (850, 410), _c0)
except Exception:
    pass
layout["Music"] = [850, 410, 1037, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["Anytime"] = [438, 410, 838, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1049, 410), _c2)
except Exception:
    pass
layout["Business"] = [1049, 410, 1290, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/04_icon_Fo.png
try:
    _c4 = get_crop(4, 135, 111)
    canvas.paste(_c4, (1295, 406), _c4)
except Exception:
    pass
layout["Fo("] = [1295, 406, 1430, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2336), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2336, 1236, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2336), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2336, 1380, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/07_icon_Interactive_Live_Music_and_Jam_Session_a.png
try:
    _c7 = get_crop(7, 1344, 996)
    canvas.paste(_c7, (48, 1820), _c7)
except Exception:
    pass
layout["Interactive_Live_Music_an"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/08_icon_Close_current_screen.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1248, 96), _c8)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1236, 1192), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/10_icon_6.49.png
try:
    _c10 = get_crop(10, 60, 65)
    canvas.paste(_c10, (180, 0), _c10)
except Exception:
    pass
layout["6.49"] = [180, 0, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/11_icon_Music.png
try:
    _c11 = get_crop(11, 66, 64)
    canvas.paste(_c11, (308, 0), _c11)
except Exception:
    pass
layout["Music"] = [308, 0, 374, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/12_icon_6.49.png
try:
    _c12 = get_crop(12, 128, 117)
    canvas.paste(_c12, (52, 111), _c12)
except Exception:
    pass
layout["6.49"] = [52, 111, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/13_icon_Favorite_button.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1092, 1192), _c13)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/14_icon_6.49.png
try:
    _c14 = get_crop(14, 62, 66)
    canvas.paste(_c14, (113, 0), _c14)
except Exception:
    pass
layout["6.49"] = [113, 0, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/15_icon_Music.png
try:
    _c15 = get_crop(15, 52, 65)
    canvas.paste(_c15, (247, 0), _c15)
except Exception:
    pass
layout["Music"] = [247, 0, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 101, 62)
    canvas.paste(_c16, (1207, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1207, 0, 1308, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 58, 62)
    canvas.paste(_c17, (1319, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1319, 0, 1377, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/18_icon_Sun_Mav_5_._6.30_PM_EDT.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (288, 2804), _c18)
except Exception:
    pass
layout["Sun,_Mav_5_._6.30_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/19_icon_626.png
try:
    _c19 = get_crop(19, 1344, 1096)
    canvas.paste(_c19, (48, 676), _c19)
except Exception:
    pass
layout["626"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/20_icon_Music.png
try:
    _c20 = get_crop(20, 1344, 191)
    canvas.paste(_c20, (48, 72), _c20)
except Exception:
    pass
layout["Music"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/21_icon_New_York.png
try:
    _c21 = get_crop(21, 434, 144)
    canvas.paste(_c21, (0, 259), _c21)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/22_icon_Brooklyn.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (864, 2804), _c22)
except Exception:
    pass
layout["Brooklyn"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 49, 63)
    canvas.paste(_c23, (384, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [384, 2, 433, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/24_icon_Brooklyn.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["Brooklyn"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/25_icon_Sun_Mav_5_._6.30_PM_EDT.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["Sun,_Mav_5_._6.30_PM_EDT"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/26_icon_More.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (1152, 2804), _c26)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/27_text_6.49.png
try:
    _c27 = get_crop(27, 89, 41)
    canvas.paste(_c27, (22, 17), _c27)
except Exception:
    pass
layout["6.49"] = [22, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/28_text_1_875_events.png
try:
    _c28 = get_crop(28, 372, 103)
    canvas.paste(_c28, (54, 410), _c28)
except Exception:
    pass
layout["1,875_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/29_text_SUN.png
try:
    _c29 = get_crop(29, 156, 69)
    canvas.paste(_c29, (81, 674), _c29)
except Exception:
    pass
layout["SUN"] = [81, 674, 237, 743]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d77ba2e8a5b2402385411cd9fa60262a/step_07_2024_4_23_18_47_d77ba2e8a5b2402385411cd9fa60262a-9/30_text_Promoted.png
try:
    _c30 = get_crop(30, 193, 43)
    canvas.paste(_c30, (94, 1678), _c30)
except Exception:
    pass
layout["Promoted"] = [94, 1678, 287, 1721]
