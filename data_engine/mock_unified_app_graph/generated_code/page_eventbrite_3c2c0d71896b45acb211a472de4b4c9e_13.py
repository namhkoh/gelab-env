# page_id: page_eventbrite_3c2c0d71896b45acb211a472de4b4c9e_13
# screenshot: 2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15.png
# step_index: 13/15
# task: Open Eventbrite. Search free Health event in Los Angeles. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint UI background and structural elements for the mobile Eventbrite screen.
# Available globals: canvas (PIL Image 1440x2960 RGB), draw (PIL.ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
bg_offwhite = (247, 248, 250)      # overall app background
status_bar_color = (62, 62, 62)    # dark status bar
header_bg = (255, 255, 255)        # header / toolbar background (white)
divider = (226, 226, 228)          # subtle dividers
card_bg = (255, 255, 255)          # card background (white)
card_border = (236, 237, 239)      # card border
muted_shadow = (240, 241, 242)     # very subtle shadow line

w, h = canvas.size

# Fill overall background
draw.rectangle((0, 0, w, h), fill=bg_offwhite)

# Status bar area (top ~80px)
status_h = 80
draw.rectangle((0, 0, w, status_h), fill=status_bar_color)

# Thin separator under status bar to blend into header
draw.line((0, status_h, w, status_h), fill=divider, width=1)

# Header / Search area
header_top = status_h
header_bottom = 280
draw.rectangle((0, header_top, w, header_bottom), fill=header_bg)

# subtle bottom divider under header/search filters
draw.line((40, header_bottom, w-40, header_bottom), fill=divider, width=2)

# Draw a soft search field background (large rounded pill behind chips/search - do not draw any icons/text)
search_bg_bbox = (48, header_top + 16, w - 48, header_top + 110)
draw.rounded_rectangle(search_bg_bbox, radius=28, fill=(250,250,251), outline=muted_shadow, width=1)

# Draw a subtle divider between "location/filter" area and results area
filter_div_y = 360
draw.line((40, filter_div_y, w-40, filter_div_y), fill=divider, width=1)

# Event card 1 background (behind first listed event text block)
# Using detected title/text bbox: pos=(48,525) size=(1344x515) -> build a rounded card behind it
card1_left = 48
card1_top = 525
card1_right = card1_left + 1344
card1_bottom = card1_top + 515
draw.rounded_rectangle((card1_left-12, card1_top-12, card1_right+12, card1_bottom+12),
                       radius=20, fill=card_bg, outline=card_border, width=1)

# Small divider below card1 to separate from next image card
sep_y1 = card1_bottom + 20
draw.line((48, sep_y1, w-48, sep_y1), fill=muted_shadow, width=1)

# Event card 2: banner/image card container (behind the blue banner image)
# Detected big banner: pos=(48,1088) size=(1344x1029)
card2_left = 48
card2_top = 1088
card2_right = card2_left + 1344
card2_bottom = card2_top + 1029
draw.rounded_rectangle((card2_left-12, card2_top-12, card2_right+12, card2_bottom+12),
                       radius=24, fill=card_bg, outline=card_border, width=1)

# Add a light inner top padding line for the card (subtle)
draw.line((card2_left, card2_top, card2_right, card2_top), fill=muted_shadow, width=1)

# Event card 3: lower event image container
# Detected event image: pos=(48,2165) size=(1344x651)
card3_left = 48
card3_top = 2165
card3_right = card3_left + 1344
card3_bottom = card3_top + 651
# Make sure this card doesn't obscure the bottom nav area; we extend slightly above and end at its natural bottom
draw.rounded_rectangle((card3_left-12, card3_top-12, card3_right+12, card3_bottom+6),
                       radius=20, fill=card_bg, outline=card_border, width=1)

# Separator lines between stacked event cards
draw.line((48, card2_bottom + 20, w-48, card2_bottom + 20), fill=divider, width=1)

# Bottom navigation bar background and top divider
nav_top = 2804
draw.rectangle((0, nav_top, w, h), fill=header_bg)
draw.line((0, nav_top, w, nav_top), fill=divider, width=1)

# Draw subtle navigation icon placeholders backgrounds (only structural rounded circles to imply hot areas,
# but kept neutral and semi-transparent so they don't replicate actual detected icons).
# We'll draw faint circular spots spaced evenly (no icons/text).
nav_item_count = 5
spacing = w // nav_item_count
for i in range(nav_item_count):
    cx = spacing//2 + i*spacing
    cy = nav_top + 78
    r = 36
    # very subtle circle (lighter than nav background)
    draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(252,252,253), outline=(245,245,246), width=1)

# Top small screen-wide shadow under header to separate from content
draw.line((0, header_bottom+2, w, header_bottom+2), fill=(243,243,244), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/00_icon_Health.png
try:
    _c0 = get_crop(0, 208, 112)
    canvas.paste(_c0, (862, 405), _c0)
except Exception:
    pass
layout["Health"] = [862, 405, 1070, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 1344, 191)
    canvas.paste(_c1, (48, 72), _c1)
except Exception:
    pass
layout["Anytime"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/02_icon_2_Filters.png
try:
    _c2 = get_crop(2, 492, 144)
    canvas.paste(_c2, (0, 259), _c2)
except Exception:
    pass
layout["2_Filters"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/03_icon_WALK.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 1604), _c3)
except Exception:
    pass
layout["WALK"] = [1092, 1604, 1236, 1748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/04_icon_Overflow_menu_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1236, 1604), _c4)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1604, 1380, 1748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 57, 61)
    canvas.paste(_c5, (246, 1), _c5)
except Exception:
    pass
layout["icon_5"] = [246, 1, 303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 51, 66)
    canvas.paste(_c6, (1153, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1153, 0, 1204, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/07_icon_9.42.png
try:
    _c7 = get_crop(7, 118, 108)
    canvas.paste(_c7, (58, 117), _c7)
except Exception:
    pass
layout["9.42"] = [58, 117, 176, 225]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 135)
    canvas.paste(_c8, (1092, 2681), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2681, 1236, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/09_icon_9.42.png
try:
    _c9 = get_crop(9, 55, 62)
    canvas.paste(_c9, (182, 0), _c9)
except Exception:
    pass
layout["9.42"] = [182, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/10_icon_Search_forae.png
try:
    _c10 = get_crop(10, 1344, 191)
    canvas.paste(_c10, (48, 72), _c10)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/11_icon_Search_forae.png
try:
    _c11 = get_crop(11, 58, 63)
    canvas.paste(_c11, (312, 1), _c11)
except Exception:
    pass
layout["Search_forae"] = [312, 1, 370, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 74, 62)
    canvas.paste(_c12, (1212, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1212, 0, 1286, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 135)
    canvas.paste(_c13, (1236, 2681), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2681, 1380, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 57, 58)
    canvas.paste(_c14, (1318, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [1318, 1, 1375, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/15_icon_9.42.png
try:
    _c15 = get_crop(15, 57, 63)
    canvas.paste(_c15, (114, 1), _c15)
except Exception:
    pass
layout["9.42"] = [114, 1, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/16_icon_Los_Angeles.png
try:
    _c16 = get_crop(16, 492, 144)
    canvas.paste(_c16, (0, 259), _c16)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/17_icon_Search_forae.png
try:
    _c17 = get_crop(17, 49, 61)
    canvas.paste(_c17, (383, 2), _c17)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 432, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/18_icon_UNIVISION_5K_Walk_Health_Fair_ELAC.png
try:
    _c18 = get_crop(18, 1344, 1029)
    canvas.paste(_c18, (48, 1088), _c18)
except Exception:
    pass
layout["UNIVISION_5K_Walk_+_Healt"] = [48, 1088, 1392, 2117]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/19_icon_Search_events.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/20_icon_Free.png
try:
    _c20 = get_crop(20, 125, 78)
    canvas.paste(_c20, (91, 558), _c20)
except Exception:
    pass
layout["Free"] = [91, 558, 216, 636]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/21_icon_Promoted.png
try:
    _c21 = get_crop(21, 43, 59)
    canvas.paste(_c21, (283, 937), _c21)
except Exception:
    pass
layout["Promoted"] = [283, 937, 326, 996]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/22_icon_Tickets.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (864, 2804), _c22)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/23_icon_Favorites.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (576, 2804), _c23)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 41, 60)
    canvas.paste(_c24, (1273, 0), _c24)
except Exception:
    pass
layout["icon_24"] = [1273, 0, 1314, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/25_icon_More.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (1152, 2804), _c25)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/26_icon_Home.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/27_text_9.42.png
try:
    _c27 = get_crop(27, 91, 41)
    canvas.paste(_c27, (20, 17), _c27)
except Exception:
    pass
layout["9.42"] = [20, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/28_text_Free_Educational_Seminar_on_Estate_plann.png
try:
    _c28 = get_crop(28, 1344, 515)
    canvas.paste(_c28, (48, 525), _c28)
except Exception:
    pass
layout["Free_Educational_Seminar_"] = [48, 525, 1392, 1040]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/29_text_for_Nurses.png
try:
    _c29 = get_crop(29, 288, 57)
    canvas.paste(_c29, (90, 726), _c29)
except Exception:
    pass
layout["for_Nurses"] = [90, 726, 378, 783]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/30_text_Sun_Apr_7.png
try:
    _c30 = get_crop(30, 211, 52)
    canvas.paste(_c30, (91, 809), _c30)
except Exception:
    pass
layout["Sun,_Apr_7"] = [91, 809, 302, 861]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/31_text_2_00_PM_PDT.png
try:
    _c31 = get_crop(31, 256, 50)
    canvas.paste(_c31, (319, 807), _c31)
except Exception:
    pass
layout["2:00_PM_PDT"] = [319, 807, 575, 857]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/32_text_655_North_Central_Avenue_Glendale_CA_USA.png
try:
    _c32 = get_crop(32, 1344, 515)
    canvas.paste(_c32, (48, 525), _c32)
except Exception:
    pass
layout["655_North_Central_Avenue,"] = [48, 525, 1392, 1040]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/33_text_HEALTH.png
try:
    _c33 = get_crop(33, 369, 108)
    canvas.paste(_c33, (340, 1193), _c33)
except Exception:
    pass
layout["HEALTH"] = [340, 1193, 709, 1301]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/34_text_WALK.png
try:
    _c34 = get_crop(34, 164, 65)
    canvas.paste(_c34, (883, 1463), _c34)
except Exception:
    pass
layout["WALK"] = [883, 1463, 1047, 1528]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/35_text_EVENT.png
try:
    _c35 = get_crop(35, 1344, 1029)
    canvas.paste(_c35, (48, 1088), _c35)
except Exception:
    pass
layout["EVENT"] = [48, 1088, 1392, 2117]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_13_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-15/36_clickable_Event_s_image.png
try:
    _c36 = get_crop(36, 1344, 651)
    canvas.paste(_c36, (48, 2165), _c36)
except Exception:
    pass
layout["Event's_image"] = [48, 2165, 1392, 2816]
