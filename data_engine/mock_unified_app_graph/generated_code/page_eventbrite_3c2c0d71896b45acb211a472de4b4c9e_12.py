# page_id: page_eventbrite_3c2c0d71896b45acb211a472de4b4c9e_12
# screenshot: 2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14.png
# step_index: 12/15
# task: Open Eventbrite. Search free Health event in Los Angeles. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background (dominant light off-white from the screenshot)
bg_color = (247, 248, 251)  # very light gray/blue
draw.rectangle([(0, 0), (canvas.width, canvas.height)], fill=bg_color)

# Status bar area (top strip, keep it simple and solid - icons will be pasted on top)
status_bar_h = 88
status_color = (133, 138, 146)  # muted dark gray for status area
draw.rectangle([(0, 0), (canvas.width, status_bar_h)], fill=status_color)

# Header / search area (below status bar) - light, card-like
header_y0 = status_bar_h
header_y1 = 260
header_bg = (255, 255, 255)  # white header/background for search region
draw.rectangle([(0, header_y0), (canvas.width, header_y1)], fill=header_bg)

# subtle bottom divider under header
divider_color = (224, 226, 229)
draw.line([(24, header_y1), (canvas.width - 24, header_y1)], fill=divider_color, width=2)

# Content area top spacing (location & filters area sits here; do not draw the filter pills themselves)
filters_area_y0 = header_y1 + 8
filters_area_y1 = header_y1 + 170
# keep same background but add a faint horizontal rule beneath to separate from listings
draw.line([(24, filters_area_y1), (canvas.width - 24, filters_area_y1)], fill=divider_color, width=1)

# "Events count" area background left as page background (no text drawn, text will be pasted)

# First event card background (rounded rectangle)
card_margin_x = 48
first_card_top = filters_area_y1 + 24
first_card_bottom = 1280
card_radius = 28
card_bg = (255, 255, 255)  # white card
card_border = (232, 234, 238)  # subtle border for card edge

draw.rounded_rectangle(
    [(card_margin_x, first_card_top), (canvas.width - card_margin_x, first_card_bottom)],
    radius=card_radius,
    fill=card_bg,
    outline=card_border,
    width=1
)

# image placeholder area inside first card (background only, actual image will be pasted on top)
# Keep it subtle and neutral so pasted image will cover; this is only background
image_pad = 24
img0_x0 = card_margin_x + image_pad
img0_x1 = canvas.width - card_margin_x - image_pad
img0_y0 = first_card_top + 20
img0_y1 = img0_y0 + 420  # height similar to screenshot image aspect
image_bg = (244, 246, 249)  # very light gray for image container background
draw.rectangle([(img0_x0, img0_y0), (img0_x1, img0_y1)], fill=image_bg, outline=None)

# subtle separator between image area and card body (thin shadow line)
draw.line([(img0_x0, img0_y1 + 12), (img0_x1, img0_y1 + 12)], fill=divider_color, width=1)

# Second event card background (rounded rectangle)
second_card_top = first_card_bottom + 220
second_card_bottom = 2780
draw.rounded_rectangle(
    [(card_margin_x, second_card_top), (canvas.width - card_margin_x, second_card_bottom)],
    radius=card_radius,
    fill=card_bg,
    outline=card_border,
    width=1
)

# image placeholder area inside second card (actual image will be pasted on top)
img1_x0 = card_margin_x + image_pad
img1_x1 = canvas.width - card_margin_x - image_pad
img1_y0 = second_card_top + 20
# the detected second event image is large; allow tall placeholder
img1_y1 = img1_y0 + 680
draw.rectangle([(img1_x0, img1_y0), (img1_x1, img1_y1)], fill=image_bg, outline=None)

# subtle separator under second card image area
draw.line([(img1_x0, img1_y1 + 12), (img1_x1, img1_y1 + 12)], fill=divider_color, width=1)

# Thin separators between major sections/content rows
sep_x0 = 24
sep_x1 = canvas.width - 24
draw.line([(sep_x0, first_card_top - 18), (sep_x1, first_card_top - 18)], fill=divider_color, width=1)
draw.line([(sep_x0, second_card_top - 18), (sep_x1, second_card_top - 18)], fill=divider_color, width=1)

# Bottom navigation bar background and top divider
bottom_nav_h = 120
bottom_nav_y0 = canvas.height - bottom_nav_h
bottom_nav_bg = (255, 255, 255)
draw.rectangle([(0, bottom_nav_y0), (canvas.width, canvas.height)], fill=bottom_nav_bg)
draw.line([(0, bottom_nav_y0), (canvas.width, bottom_nav_y0)], fill=divider_color, width=2)

# Small rounded highlight behind center nav area (to mimic subtle UI emphasis; icons will be pasted over)
center_nav_w = 88
center_nav_h = 88
center_nav_x = canvas.width // 2 - center_nav_w // 2
center_nav_y = bottom_nav_y0 + (bottom_nav_h - center_nav_h) // 2
nav_highlight = (255, 255, 255)  # keep white but add faint outline for lift
draw.rounded_rectangle(
    [(center_nav_x, center_nav_y), (center_nav_x + center_nav_w, center_nav_y + center_nav_h)],
    radius=22,
    fill=nav_highlight,
    outline=(243, 150, 57, 20)  # extremely faint outline (RGBA may be ignored, safe fallback)
)

# Overall subtle vignette/shadow under cards (non-intrusive lines to suggest elevation)
shadow_color = (238, 240, 244)
# under first card
draw.rectangle([
    (card_margin_x + 6, first_card_bottom + 2),
    (canvas.width - card_margin_x - 6, first_card_bottom + 6)
], fill=shadow_color)
# under second card
draw.rectangle([
    (card_margin_x + 6, second_card_bottom + 2),
    (canvas.width - card_margin_x - 6, second_card_bottom + 6)
], fill=shadow_color)

# Note: All actual icons, buttons, and text will be pasted on top of these backgrounds.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/00_icon_Health.png
try:
    _c0 = get_crop(0, 199, 135)
    canvas.paste(_c0, (870, 390), _c0)
except Exception:
    pass
layout["Health"] = [870, 390, 1069, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 135)
    canvas.paste(_c1, (458, 390), _c1)
except Exception:
    pass
layout["Anytime"] = [458, 390, 858, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/02_icon_2_Filters.png
try:
    _c2 = get_crop(2, 392, 135)
    canvas.paste(_c2, (54, 390), _c2)
except Exception:
    pass
layout["2_Filters"] = [54, 390, 446, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/03_icon_Favorite_button.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 1192), _c3)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/04_icon_Overflow_menu_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1236, 1192), _c4)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2336), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2336, 1236, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2336), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2336, 1380, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/07_icon_Free_Educational_Seminar_on_Estate_plann.png
try:
    _c7 = get_crop(7, 1344, 996)
    canvas.paste(_c7, (48, 1820), _c7)
except Exception:
    pass
layout["Free_Educational_Seminar_"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 57, 62)
    canvas.paste(_c8, (246, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [246, 1, 303, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 54, 65)
    canvas.paste(_c9, (1151, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1151, 0, 1205, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/10_icon_9.42.png
try:
    _c10 = get_crop(10, 117, 108)
    canvas.paste(_c10, (59, 117), _c10)
except Exception:
    pass
layout["9.42"] = [59, 117, 176, 225]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/11_icon_9.42.png
try:
    _c11 = get_crop(11, 55, 62)
    canvas.paste(_c11, (182, 0), _c11)
except Exception:
    pass
layout["9.42"] = [182, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/12_icon_Search_forae.png
try:
    _c12 = get_crop(12, 61, 63)
    canvas.paste(_c12, (311, 1), _c12)
except Exception:
    pass
layout["Search_forae"] = [311, 1, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 70, 61)
    canvas.paste(_c13, (1212, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1212, 0, 1282, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 59, 59)
    canvas.paste(_c14, (1317, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1317, 0, 1376, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/15_icon_9.42.png
try:
    _c15 = get_crop(15, 58, 64)
    canvas.paste(_c15, (114, 0), _c15)
except Exception:
    pass
layout["9.42"] = [114, 0, 172, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/16_icon_Search_forae.png
try:
    _c16 = get_crop(16, 1344, 191)
    canvas.paste(_c16, (48, 72), _c16)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/17_icon_Los_Angeles.png
try:
    _c17 = get_crop(17, 492, 144)
    canvas.paste(_c17, (0, 259), _c17)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/18_icon_Sun_Apr_7_._2_00_PM_PDT.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (288, 2804), _c18)
except Exception:
    pass
layout["Sun,_Apr_7_._2:00_PM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/19_icon_Promoted.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (234, 1625), _c19)
except Exception:
    pass
layout["Promoted"] = [234, 1625, 378, 1769]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/20_icon_Vinyland_Vintage_Market_Thursdays.png
try:
    _c20 = get_crop(20, 1344, 1096)
    canvas.paste(_c20, (48, 676), _c20)
except Exception:
    pass
layout["Vinyland_Vintage_Market_T"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/21_icon_Search_forae.png
try:
    _c21 = get_crop(21, 50, 62)
    canvas.paste(_c21, (383, 2), _c21)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 433, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/22_icon_Free_Educational_Seminar_on_Estate_plann.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (864, 2804), _c22)
except Exception:
    pass
layout["Free_Educational_Seminar_"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/23_icon_Free_Educational_Seminar_on_Estate_plann.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (1152, 2804), _c23)
except Exception:
    pass
layout["Free_Educational_Seminar_"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/24_icon_for_Nurses.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (0, 2804), _c24)
except Exception:
    pass
layout["for_Nurses"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 41, 60)
    canvas.paste(_c25, (1273, 0), _c25)
except Exception:
    pass
layout["icon_25"] = [1273, 0, 1314, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/26_icon_Free_Educational_Seminar_on_Estate_plann.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (576, 2804), _c26)
except Exception:
    pass
layout["Free_Educational_Seminar_"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/27_text_9.42.png
try:
    _c27 = get_crop(27, 91, 41)
    canvas.paste(_c27, (20, 17), _c27)
except Exception:
    pass
layout["9.42"] = [20, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/28_text_1_100_events.png
try:
    _c28 = get_crop(28, 392, 135)
    canvas.paste(_c28, (54, 390), _c28)
except Exception:
    pass
layout["1,100_events"] = [54, 390, 446, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/29_text_Sun_Apr_7_._2_00_PM_PDT.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (288, 2804), _c29)
except Exception:
    pass
layout["Sun,_Apr_7_._2:00_PM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_12_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-14/30_text_MaGe.png
try:
    _c30 = get_crop(30, 179, 177)
    canvas.paste(_c30, (897, 649), _c30)
except Exception:
    pass
layout["MaGe"] = [897, 649, 1076, 826]
