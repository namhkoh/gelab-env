# page_id: page_eventbrite_642b2c00296046d8aea556a4767dc400_05
# screenshot: 2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7.png
# step_index: 5/12
# task: Open Eventbrite. Search free events in New York. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural layout for Eventbrite-like mobile UI
# Uses provided variables: canvas (1440x2960 PIL Image), draw (ImageDraw), fonts

w, h = canvas.size

# Colors
bg = (249, 250, 252)          # page background (very light)
status_bar_col = (149, 153, 158)  # status bar dark gray
toolbar_bg = (255, 255, 255)  # toolbar / header white
divider = (224, 226, 230)     # soft divider
chip_band = (237, 246, 255)   # pale blue band behind filter chips
card_shadow = (230, 232, 235) # card shadow
card_bg = (255, 255, 255)     # card white
bottom_bar_bg = (255, 255, 255)
soft_grey = (246, 247, 249)

# Clear canvas with page background
draw.rectangle((0, 0, w, h), fill=bg)

# Status bar (top ~60px)
status_h = 60
draw.rectangle((0, 0, w, status_h), fill=status_bar_col)

# Top toolbar / search header area
# Keep it slightly taller to provide background behind the search field
toolbar_top = status_h
toolbar_bottom = 300
draw.rectangle((0, toolbar_top, w, toolbar_bottom), fill=toolbar_bg)
# Divider under toolbar
draw.line((48, toolbar_bottom, w-48, toolbar_bottom), fill=divider, width=2)

# Pale band behind filter chips (chips themselves will be pasted later)
chip_band_top = 300
chip_band_bottom = 460
draw.rectangle((0, chip_band_top, w, chip_band_bottom), fill=chip_band)
# thin divider under chips row
draw.line((48, chip_band_bottom + 6, w-48, chip_band_bottom + 6), fill=divider, width=1)

# Heading area background (behind "10,000 events" etc.)
# subtle inset background to separate from cards
heading_band_top = chip_band_bottom + 24
heading_band_bottom = heading_band_top + 120
draw.rectangle((48, heading_band_top, w-48, heading_band_bottom), fill=soft_grey, outline=None)

# First event card (rounded rectangle background + shadow)
card1_x = 48
card1_y = 676
card1_w = 1344
card1_h = 1096
card1_box = (card1_x, card1_y, card1_x + card1_w, card1_y + card1_h)

# shadow (slightly offset)
shadow_offset = 8
draw.rounded_rectangle(
    (card1_box[0] + shadow_offset, card1_box[1] + shadow_offset,
     card1_box[2] + shadow_offset, card1_box[3] + shadow_offset),
    radius=28, fill=card_shadow
)
# card background
draw.rounded_rectangle(card1_box, radius=28, fill=card_bg)

# subtle divider across card to hint separation between image and text
# Place it roughly where the image area ends visually (leave a band near top)
image_split_y = card1_y + int(card1_h * 0.42)
draw.line((card1_x + 24, image_split_y, card1_x + card1_w - 24, image_split_y),
          fill=divider, width=2)

# Second event card (rounded rectangle background + shadow)
card2_x = 48
card2_y = 1820
card2_w = 1344
card2_h = 996
card2_box = (card2_x, card2_y, card2_x + card2_w, card2_y + card2_h)

draw.rounded_rectangle(
    (card2_box[0] + shadow_offset, card2_box[1] + shadow_offset,
     card2_box[2] + shadow_offset, card2_box[3] + shadow_offset),
    radius=28, fill=card_shadow
)
draw.rounded_rectangle(card2_box, radius=28, fill=card_bg)

# divider line above the second card to separate list items
draw.line((48, card2_y - 24, w-48, card2_y - 24), fill=divider, width=1)

# Create a faint image-area band on the second card (simulates image panel background)
# Keep this subtle and neutral so it won't conflict with pasted image content
img_band_margin = 24
img_band_height = int(card2_h * 0.45)
draw.rectangle(
    (card2_x + img_band_margin, card2_y + img_band_margin,
     card2_x + card2_w - img_band_margin, card2_y + img_band_margin + img_band_height),
    fill=(240, 245, 248)
)

# separator between list rows further down
sep_y = card2_y + card2_h + 16
draw.line((48, sep_y, w-48, sep_y), fill=divider, width=1)

# Bottom navigation bar background and top divider
bottom_bar_h = 120
bottom_bar_top = h - bottom_bar_h
draw.line((48, bottom_bar_top, w-48, bottom_bar_top), fill=divider, width=2)
draw.rectangle((0, bottom_bar_top, w, h), fill=bottom_bar_bg)

# Subtle indicator row above bottom nav to separate content from nav
small_bar_top = bottom_bar_top - 18
draw.rectangle((0, small_bar_top, w, bottom_bar_top), fill=toolbar_bg)

# Edge gutters (visual vertical guides to match card margins)
gutter_x1 = 48
gutter_x2 = w - 48
draw.line((gutter_x1, toolbar_bottom, gutter_x1, bottom_bar_top), fill=(0,0,0,0))
draw.line((gutter_x2, toolbar_bottom, gutter_x2, bottom_bar_top), fill=(0,0,0,0))

# Final subtle vignette at card bottoms (very light)
draw.rectangle((card1_x + 12, card1_box[3] - 10, card1_box[2] - 12, card1_box[3]), fill=(250,250,251))
draw.rectangle((card2_x + 12, card2_box[3] - 10, card2_box[2] - 12, card2_box[3]), fill=(250,250,251))

# Note: actual icons, text, buttons, and images will be pasted on top at their detected positions.
# This drawing provides only background, cards, separators, and structural elements.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (850, 410), _c0)
except Exception:
    pass
layout["Music"] = [850, 410, 1037, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["Anytime"] = [438, 410, 838, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1049, 410), _c2)
except Exception:
    pass
layout["Business"] = [1049, 410, 1290, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/04_icon_Fo.png
try:
    _c4 = get_crop(4, 141, 111)
    canvas.paste(_c4, (1295, 406), _c4)
except Exception:
    pass
layout["Fo("] = [1295, 406, 1436, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/05_icon_Overflow_menu_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 2336), _c5)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2336, 1380, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/06_icon_PROCESS_OF_BUILDING_YOUR_REAL_ESTATE_NEW.png
try:
    _c6 = get_crop(6, 1344, 996)
    canvas.paste(_c6, (48, 1820), _c6)
except Exception:
    pass
layout["PROCESS_OF_BUILDING_YOUR_"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/07_icon_ACCELERATOR_EVENT.png
try:
    _c7 = get_crop(7, 1344, 996)
    canvas.paste(_c7, (48, 1820), _c7)
except Exception:
    pass
layout["ACCELERATOR_EVENT"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/08_icon_9.09.png
try:
    _c8 = get_crop(8, 130, 117)
    canvas.paste(_c8, (53, 113), _c8)
except Exception:
    pass
layout["9.09"] = [53, 113, 183, 230]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/09_icon_Favorite_button.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1092, 2336), _c9)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2336, 1236, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 55, 61)
    canvas.paste(_c10, (247, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [247, 1, 302, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 96, 60)
    canvas.paste(_c11, (1207, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1207, 0, 1303, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/12_icon_Search_forae.png
try:
    _c12 = get_crop(12, 60, 62)
    canvas.paste(_c12, (312, 1), _c12)
except Exception:
    pass
layout["Search_forae"] = [312, 1, 372, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/13_icon_9.09.png
try:
    _c13 = get_crop(13, 55, 62)
    canvas.paste(_c13, (182, 0), _c13)
except Exception:
    pass
layout["9.09"] = [182, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 65, 59)
    canvas.paste(_c14, (1316, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1316, 0, 1381, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/15_icon_Search_forae.png
try:
    _c15 = get_crop(15, 1344, 191)
    canvas.paste(_c15, (48, 72), _c15)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/16_icon_6_00_PM_EDT.png
try:
    _c16 = get_crop(16, 288, 156)
    canvas.paste(_c16, (288, 2804), _c16)
except Exception:
    pass
layout["6:00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/17_icon_Overflow_menu_button.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1236, 1192), _c17)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/18_icon_9.09.png
try:
    _c18 = get_crop(18, 57, 64)
    canvas.paste(_c18, (114, 0), _c18)
except Exception:
    pass
layout["9.09"] = [114, 0, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/19_icon_New_York.png
try:
    _c19 = get_crop(19, 434, 144)
    canvas.paste(_c19, (0, 259), _c19)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/20_icon_Tequila_Artistic_Transformation.png
try:
    _c20 = get_crop(20, 1344, 1096)
    canvas.paste(_c20, (48, 676), _c20)
except Exception:
    pass
layout["Tequila_&_Artistic_Transf"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/21_icon_Search_forae.png
try:
    _c21 = get_crop(21, 50, 61)
    canvas.paste(_c21, (383, 2), _c21)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 433, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/22_icon_Free_Real_Estate_Connections_Accelerator.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (864, 2804), _c22)
except Exception:
    pass
layout["Free_Real_Estate_Connecti"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/23_icon_Thu_Mar_21.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (0, 2804), _c23)
except Exception:
    pass
layout["Thu,_Mar_21"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/24_icon_Free_Real_Estate_Connections_Accelerator.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["Free_Real_Estate_Connecti"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/25_icon_Favorite_button.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1092, 1192), _c25)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/26_icon_Promoted.png
try:
    _c26 = get_crop(26, 245, 67)
    canvas.paste(_c26, (83, 1665), _c26)
except Exception:
    pass
layout["Promoted"] = [83, 1665, 328, 1732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/27_icon_More.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (1152, 2804), _c27)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 39, 61)
    canvas.paste(_c28, (1275, 0), _c28)
except Exception:
    pass
layout["icon_28"] = [1275, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/29_text_9.09.png
try:
    _c29 = get_crop(29, 94, 45)
    canvas.paste(_c29, (17, 15), _c29)
except Exception:
    pass
layout["9.09"] = [17, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/30_text_10_000_events.png
try:
    _c30 = get_crop(30, 372, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/642b2c00296046d8aea556a4767dc400/step_05_2024_3_20_17_8_642b2c00296046d8aea556a4767dc400-7/31_text_Thu_Mar_21.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (0, 2804), _c31)
except Exception:
    pass
layout["Thu,_Mar_21"] = [0, 2804, 288, 2960]
