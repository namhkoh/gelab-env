# page_id: page_eventbrite_86c0bd1901f44c94916665f4058f9b6d_11
# screenshot: 2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13.png
# step_index: 11/11
# task: Open Eventbrite. Set the city to Los Angeles. Select the 'Food & Drink' category. What's the date of the first event that is not promoted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background/base
bg_color = (247, 248, 250)  # soft off-white dominant background
canvas.paste(bg_color, [0, 0, canvas.size[0], canvas.size[1]])

# Helpful coordinates
W, H = canvas.size
pad_x = 48
content_x0 = pad_x
content_x1 = W - pad_x

# Top status bar (approx ~72px high)
status_h = 72
status_color = (165, 165, 165)  # muted gray status bar
draw.rectangle([(0, 0), (W, status_h)], fill=status_color)

# Subtle status bar bottom divider
draw.line([(0, status_h), (W, status_h)], fill=(200,200,200), width=1)

# Header / search area background
header_y0 = status_h
header_y1 = 232
header_bg = (255, 255, 255)  # white header behind search
draw.rectangle([(0, header_y0), (W, header_y1)], fill=header_bg)

# Thin divider under header
draw.line([(content_x0, header_y1), (content_x1, header_y1)], fill=(225,225,230), width=2)

# Filter/controls area background (subtle band behind filter pills)
filters_y0 = 500
filters_y1 = 580
filters_bg = (250, 252, 253)
draw.rectangle([(0, filters_y0), (W, filters_y1)], fill=filters_bg)
draw.line([(content_x0, filters_y1), (content_x1, filters_y1)], fill=(230,230,235), width=1)

# Draw event-card background groups as rounded rectangles with soft shadow
def draw_card(x0, y0, x1, y1, radius=24):
    # shadow
    shadow_color = (234, 236, 240)
    shadow_offset = 6
    draw.rounded_rectangle([(x0+shadow_offset, y0+shadow_offset), (x1+shadow_offset, y1+shadow_offset)],
                           radius, fill=shadow_color)
    # main card
    card_color = (255, 255, 255)
    outline_color = (230, 232, 236)
    draw.rounded_rectangle([(x0, y0), (x1, y1)], radius, fill=card_color, outline=outline_color)

# First event card (BEATS & BBQ area)
card1_y0 = 580
card1_y1 = 900
draw_card(content_x0, card1_y0, content_x1, card1_y1, radius=20)

# Separator between cards
sep_y = card1_y1 + 24
draw.line([(content_x0, sep_y), (content_x1, sep_y)], fill=(235,235,238), width=1)

# Second main event card (LA'S BIGGEST CINCO DE MAYO...)
card2_y0 = 920
card2_y1 = 2016  # aligns before the next big poster
draw_card(content_x0, card2_y0, content_x1, card2_y1, radius=20)

# Separator between cards
sep2_y = card2_y1 + 24
draw.line([(content_x0, sep2_y), (content_x1, sep2_y)], fill=(235,235,238), width=1)

# Third event/poster card (ESPOLON banner)
card3_y0 = 2048
card3_y1 = 2800  # ends above the bottom nav area
draw_card(content_x0, card3_y0, content_x1, card3_y1, radius=20)

# Content area dark overlay behind image previews (subtle bands)
# These mimic background zones where images will be pasted, not the images themselves.
# For first event image area (top of first card)
img1_y0 = card1_y0 + 8
img1_y1 = img1_y0 + 160
draw.rectangle([(content_x0 + 8, img1_y0), (content_x1 - 8, img1_y1)], fill=(245,245,247))

# For second event image area (inside card2 near top)
img2_y0 = card2_y0 + 12
img2_y1 = img2_y0 + 420  # large hero image area
draw.rectangle([(content_x0 + 12, img2_y0), (content_x1 - 12, img2_y1)], fill=(245,245,247))

# For third poster image area (inside card3)
img3_y0 = card3_y0 + 12
img3_y1 = img3_y0 + 700
draw.rectangle([(content_x0 + 12, img3_y0), (content_x1 - 12, img3_y1)], fill=(245,245,247))

# Bottom navigation bar background
nav_y0 = 2804
nav_y1 = H
nav_bg = (255, 255, 255)
# top divider for nav
draw.line([(0, nav_y0), (W, nav_y0)], fill=(225,225,230), width=2)
# nav rectangle
draw.rectangle([(0, nav_y0), (W, nav_y1)], fill=nav_bg)

# Subtle rounded notch on nav corners (visual)
corner_radius = 12
draw.rounded_rectangle([(0, nav_y0), (W, nav_y0 + 48)], corner_radius, fill=nav_bg, outline=None)

# Final subtle vertical guides for content width (not visible UI, but helps structure)
# very faint lines to align content (kept extremely light)
guide_color = (250,250,251)
draw.line([(content_x0, 0), (content_x0, H)], fill=guide_color, width=1)
draw.line([(content_x1, 0), (content_x1, H)], fill=guide_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 319, 111)
    canvas.paste(_c0, (844, 406), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [844, 406, 1163, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 1344, 420)
    canvas.paste(_c1, (48, 525), _c1)
except Exception:
    pass
layout["Anytime"] = [48, 525, 1392, 945]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/02_icon_1_Filter.png
try:
    _c2 = get_crop(2, 492, 144)
    canvas.paste(_c2, (0, 259), _c2)
except Exception:
    pass
layout["1_Filter"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/03_icon_CELEBRATING_25_YEARS_OF_CREATIVITY.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 2581), _c3)
except Exception:
    pass
layout["CELEBRATING_25_YEARS_OF_C"] = [1092, 2581, 1236, 2725]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/04_icon_Overflow_menu_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1236, 2581), _c4)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2581, 1380, 2725]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/05_icon_Food_Drink.png
try:
    _c5 = get_crop(5, 133, 93)
    canvas.paste(_c5, (1100, 491), _c5)
except Exception:
    pass
layout["Food_&_Drink"] = [1100, 491, 1233, 584]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1509), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1509, 1380, 1653]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 119, 97)
    canvas.paste(_c7, (1246, 489), _c7)
except Exception:
    pass
layout["icon_7"] = [1246, 489, 1365, 586]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1092, 1509), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1509, 1236, 1653]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/09_icon_7.14.png
try:
    _c9 = get_crop(9, 125, 115)
    canvas.paste(_c9, (54, 113), _c9)
except Exception:
    pass
layout["7.14"] = [54, 113, 179, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 53, 65)
    canvas.paste(_c10, (1151, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1151, 0, 1204, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/11_icon_7.14.png
try:
    _c11 = get_crop(11, 61, 64)
    canvas.paste(_c11, (180, 0), _c11)
except Exception:
    pass
layout["7.14"] = [180, 0, 241, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/12_icon_Search_forae.png
try:
    _c12 = get_crop(12, 68, 62)
    canvas.paste(_c12, (307, 0), _c12)
except Exception:
    pass
layout["Search_forae"] = [307, 0, 375, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 55, 63)
    canvas.paste(_c13, (246, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [246, 0, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/14_icon_LA_S_BIGGEST_CINCO_DE_MAYO_PARTY_WITH.png
try:
    _c14 = get_crop(14, 1344, 1024)
    canvas.paste(_c14, (48, 993), _c14)
except Exception:
    pass
layout["LA'S_BIGGEST_CINCO_DE_MAY"] = [48, 993, 1392, 2017]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 60, 61)
    canvas.paste(_c15, (1212, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1212, 0, 1272, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/16_icon_7.14.png
try:
    _c16 = get_crop(16, 59, 64)
    canvas.paste(_c16, (115, 0), _c16)
except Exception:
    pass
layout["7.14"] = [115, 0, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 60, 59)
    canvas.paste(_c17, (1317, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1317, 0, 1377, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/18_icon_Search_forae.png
try:
    _c18 = get_crop(18, 1344, 191)
    canvas.paste(_c18, (48, 72), _c18)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/19_icon_25_ANIVERSARIO.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (576, 2804), _c19)
except Exception:
    pass
layout["25_ANIVERSARIO"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/20_icon_Free.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (0, 2804), _c20)
except Exception:
    pass
layout["Free"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 45, 60)
    canvas.paste(_c21, (1268, 0), _c21)
except Exception:
    pass
layout["icon_21"] = [1268, 0, 1313, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/22_icon_Los_Angeles.png
try:
    _c22 = get_crop(22, 492, 144)
    canvas.paste(_c22, (0, 259), _c22)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/23_icon_Tickets.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/24_icon_25_ANIVERSARIO.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (288, 2804), _c24)
except Exception:
    pass
layout["25_ANIVERSARIO"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/25_icon_More.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (1152, 2804), _c25)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/26_icon_Free.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Free"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/27_icon_LA_S_BIGGEST_CINCO_DE_MAYO_PARTY_WITH.png
try:
    _c27 = get_crop(27, 1344, 1024)
    canvas.paste(_c27, (48, 993), _c27)
except Exception:
    pass
layout["LA'S_BIGGEST_CINCO_DE_MAY"] = [48, 993, 1392, 2017]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/28_icon_CELEBRATING_25_YEARS_OF_CREATIVITY.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (864, 2804), _c28)
except Exception:
    pass
layout["CELEBRATING_25_YEARS_OF_C"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/29_icon_Search_forae.png
try:
    _c29 = get_crop(29, 50, 60)
    canvas.paste(_c29, (383, 2), _c29)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 433, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/30_icon_7.14.png
try:
    _c30 = get_crop(30, 97, 63)
    canvas.paste(_c30, (13, 0), _c30)
except Exception:
    pass
layout["7.14"] = [13, 0, 110, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/31_icon_5010S_La_Brea_Ave.png
try:
    _c31 = get_crop(31, 42, 59)
    canvas.paste(_c31, (285, 841), _c31)
except Exception:
    pass
layout["5010S_La_Brea_Ave"] = [285, 841, 327, 900]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/32_text_BEATS_BBQ.png
try:
    _c32 = get_crop(32, 361, 68)
    canvas.paste(_c32, (94, 628), _c32)
except Exception:
    pass
layout["BEATS_&_BBQ"] = [94, 628, 455, 696]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/33_text_Sun.png
try:
    _c33 = get_crop(33, 101, 54)
    canvas.paste(_c33, (90, 713), _c33)
except Exception:
    pass
layout["Sun,"] = [90, 713, 191, 767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/34_text_12_00_PM_PDT.png
try:
    _c34 = get_crop(34, 277, 48)
    canvas.paste(_c34, (335, 712), _c34)
except Exception:
    pass
layout["12:00_PM_PDT"] = [335, 712, 612, 760]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/35_text_5010S_La_Brea_Ave.png
try:
    _c35 = get_crop(35, 364, 52)
    canvas.paste(_c35, (90, 779), _c35)
except Exception:
    pass
layout["5010S_La_Brea_Ave"] = [90, 779, 454, 831]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/36_text_CINCo_O_M_Yo.png
try:
    _c36 = get_crop(36, 1344, 1024)
    canvas.paste(_c36, (48, 993), _c36)
except Exception:
    pass
layout["CINCo_O_M:_Yo"] = [48, 993, 1392, 2017]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/37_text_Skybar_Los_Angeles.png
try:
    _c37 = get_crop(37, 367, 57)
    canvas.paste(_c37, (90, 1919), _c37)
except Exception:
    pass
layout["Skybar_Los_Angeles"] = [90, 1919, 457, 1976]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/38_text_ESPOLONE.png
try:
    _c38 = get_crop(38, 140, 51)
    canvas.paste(_c38, (287, 2364), _c38)
except Exception:
    pass
layout["[ESPOLONE"] = [287, 2364, 427, 2415]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/39_text_ESPOLOH.png
try:
    _c39 = get_crop(39, 1344, 751)
    canvas.paste(_c39, (48, 2065), _c39)
except Exception:
    pass
layout["ESPOLOH"] = [48, 2065, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/40_text_25.png
try:
    _c40 = get_crop(40, 86, 76)
    canvas.paste(_c40, (317, 2473), _c40)
except Exception:
    pass
layout["25"] = [317, 2473, 403, 2549]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/41_text_CELEBRATING_25_YEARS_OF_CREATIVITY.png
try:
    _c41 = get_crop(41, 1344, 751)
    canvas.paste(_c41, (48, 2065), _c41)
except Exception:
    pass
layout["CELEBRATING_25_YEARS_OF_C"] = [48, 2065, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_11_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-13/42_text_25_ANIVERSARIO.png
try:
    _c42 = get_crop(42, 303, 74)
    canvas.paste(_c42, (211, 2606), _c42)
except Exception:
    pass
layout["25_ANIVERSARIO"] = [211, 2606, 514, 2680]
