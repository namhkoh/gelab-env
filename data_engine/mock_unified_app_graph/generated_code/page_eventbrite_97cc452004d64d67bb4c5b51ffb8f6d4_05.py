# page_id: page_eventbrite_97cc452004d64d67bb4c5b51ffb8f6d4_05
# screenshot: 2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7.png
# step_index: 5/7
# task: Open Eventbrite. Search Business event. Select the first one that is not promoted. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile UI page.
# Uses provided 'canvas' (PIL Image) and 'draw' (ImageDraw.Draw), plus fonts if needed.

width, height = canvas.size

# Colors (approximate to screenshot)
bg_color = (248, 249, 250)        # page background (very light gray)
status_bar_color = (190, 190, 190) # top status bar (light muted gray)
divider_color = (220, 221, 224)    # subtle dividers
card_shadow = (230, 232, 235)      # shadow/elevation color
card_bg = (255, 255, 255)          # card white
card_border = (240, 241, 243)      # card faint border
bottom_nav_bg = (255, 255, 255)    # bottom navigation background

# Fill overall background
draw.rectangle([(0, 0), (width, height)], fill=bg_color)

# Status bar area (top)
status_h = 72
draw.rectangle([(0, 0), (width, status_h)], fill=status_bar_color)

# Leave search area clear (detected search area will be pasted). Add a thin divider under it.
# Detected search area: (48,72) size (1344x191) => bottom y = 72+191 = 263
search_bottom = 72 + 191
draw.line([(32, search_bottom), (width - 32, search_bottom)], fill=divider_color, width=2)

# Header subtle separator above search area (thin)
draw.line([(32, status_h), (width - 32, status_h)], fill=divider_color, width=1)

# Draw first event card background (rounded rectangle with shadow)
card_radius = 36
card_left = 48
card_right = width - 48
# Place the first card starting slightly below the search area
first_card_top = search_bottom + 36   # leaves breathing room under search area
# rough height to include image area and text (keeps below second image start)
first_card_bottom = 1800

# Shadow (offset)
shadow_offset = 10
draw.rounded_rectangle(
    [(card_left, first_card_top + shadow_offset), (card_right, first_card_bottom + shadow_offset)],
    radius=card_radius, fill=card_shadow
)
# Card surface
draw.rounded_rectangle(
    [(card_left, first_card_top), (card_right, first_card_bottom)],
    radius=card_radius, fill=card_bg, outline=card_border, width=1
)

# Separator line under first card content area (subtle)
sep_y = first_card_bottom + 12
draw.line([(card_left + 12, sep_y), (card_right - 12, sep_y)], fill=divider_color, width=1)

# Draw second event card background (rounded rectangle with shadow)
second_card_top = first_card_bottom + 40
second_card_bottom = height - 260  # leave room for bottom nav and footers

draw.rounded_rectangle(
    [(card_left, second_card_top + shadow_offset), (card_right, second_card_bottom + shadow_offset)],
    radius=card_radius, fill=card_shadow
)
draw.rounded_rectangle(
    [(card_left, second_card_top), (card_right, second_card_bottom)],
    radius=card_radius, fill=card_bg, outline=card_border, width=1
)

# Add faint dividing line between the two cards (above second card)
draw.line([(card_left + 12, second_card_top - 20), (card_right - 12, second_card_top - 20)], fill=divider_color, width=1)

# Decorative content-area backgrounds:
# Dark image background band for the large hero image area inside the first card (behind actual image)
# Detected image for first card is at (48,676) size (1344x1115). We'll draw a dark band matching that region's placement,
# but slightly inset so we only draw background not the image contents.
img1_left = 48 + 12
img1_top = 676 - 6    # slight lift to align visually
img1_right = img1_left + 1344 - 24
img1_bottom = img1_top + 1115 + 12
dark_band_color = (30, 34, 49)  # deep navy-ish used as a neutral background behind image
draw.rectangle([(img1_left, img1_top), (img1_right, img1_bottom)], fill=dark_band_color, outline=None)

# Light colored label background placeholders (these are backgrounds only; labels/icons will be pasted)
# For example, the small pill behind "Just added" (avoid drawing text)
pill_w, pill_h = 260, 56
pill_x = card_left + 48
pill_y = img1_bottom + 24
pill_radius = 18
pill_color = (226, 241, 237)  # pale greenish
draw.rounded_rectangle([(pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h)], radius=pill_radius, fill=pill_color)

# Another small tag placeholder near second card image (left side date panel)
# Detected second card image region is at (48,1839) size (1344x977).
img2_left = 48 + 12
img2_top = 1839 - 6
img2_right = img2_left + 1344 - 24
img2_bottom = img2_top + 977 + 12
# Light muted panel on the left of the image area (background for date/time chip)
panel_w = 200
panel_color = (245, 246, 248)
draw.rounded_rectangle([(img2_left, img2_top), (img2_left + panel_w, img2_bottom)], radius=28, fill=panel_color)

# Bottom navigation bar background and top divider
nav_top = height - 160
draw.rectangle([(0, nav_top), (width, height)], fill=bottom_nav_bg)
draw.line([(32, nav_top), (width - 32, nav_top)], fill=divider_color, width=2)

# Slight elevated background for nav icons area (subtle)
nav_elev_h = 80
nav_elev_top = height - nav_elev_h - 28
draw.rectangle([(0, nav_elev_top), (width, height)], fill=(255,255,255))

# Final subtle horizontal guideline near top of page (under status/search) for visual structure
draw.line([(24, search_bottom + 6), (width - 24, search_bottom + 6)], fill=divider_color, width=1)

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/00_icon_Business.png
try:
    _c0 = get_crop(0, 241, 135)
    canvas.paste(_c0, (850, 390), _c0)
except Exception:
    pass
layout["Business"] = [850, 390, 1091, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 135)
    canvas.paste(_c1, (438, 390), _c1)
except Exception:
    pass
layout["Anytime"] = [438, 390, 838, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/02_icon_1_Filter.png
try:
    _c2 = get_crop(2, 372, 135)
    canvas.paste(_c2, (54, 390), _c2)
except Exception:
    pass
layout["1_Filter"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/03_icon_Favorite_button.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 2355), _c3)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2355, 1236, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/04_icon_Overflow_menu_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1236, 2355), _c4)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2355, 1380, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/07_icon_9.39.png
try:
    _c7 = get_crop(7, 123, 113)
    canvas.paste(_c7, (56, 114), _c7)
except Exception:
    pass
layout["9.39"] = [56, 114, 179, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 55, 61)
    canvas.paste(_c8, (247, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [247, 1, 302, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/09_icon_9.39.png
try:
    _c9 = get_crop(9, 55, 62)
    canvas.paste(_c9, (182, 0), _c9)
except Exception:
    pass
layout["9.39"] = [182, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/10_icon_Search_forae.png
try:
    _c10 = get_crop(10, 59, 62)
    canvas.paste(_c10, (312, 1), _c10)
except Exception:
    pass
layout["Search_forae"] = [312, 1, 371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/11_icon_SUSTAINABLE.png
try:
    _c11 = get_crop(11, 1344, 1115)
    canvas.paste(_c11, (48, 676), _c11)
except Exception:
    pass
layout["SUSTAINABLE"] = [48, 676, 1392, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 89, 60)
    canvas.paste(_c12, (1207, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1207, 0, 1296, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 64, 59)
    canvas.paste(_c13, (1317, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1317, 0, 1381, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/14_icon_9.39.png
try:
    _c14 = get_crop(14, 60, 64)
    canvas.paste(_c14, (112, 0), _c14)
except Exception:
    pass
layout["9.39"] = [112, 0, 172, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/15_icon_Search_forae.png
try:
    _c15 = get_crop(15, 50, 62)
    canvas.paste(_c15, (383, 1), _c15)
except Exception:
    pass
layout["Search_forae"] = [383, 1, 433, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/16_icon_Search_forae.png
try:
    _c16 = get_crop(16, 1344, 191)
    canvas.paste(_c16, (48, 72), _c16)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/17_icon_One_Pierrenont_Plaza.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (288, 2804), _c17)
except Exception:
    pass
layout["One_Pierrenont_Plaza"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/18_icon_BUYER_BREAKFAST.png
try:
    _c18 = get_crop(18, 1344, 977)
    canvas.paste(_c18, (48, 1839), _c18)
except Exception:
    pass
layout["BUYER_BREAKFAST"] = [48, 1839, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/19_icon_Just_addedl.png
try:
    _c19 = get_crop(19, 313, 123)
    canvas.paste(_c19, (96, 1340), _c19)
except Exception:
    pass
layout["Just_addedl"] = [96, 1340, 409, 1463]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/20_icon_New_York.png
try:
    _c20 = get_crop(20, 434, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/21_icon_Brooklvn.NY_USA.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (864, 2804), _c21)
except Exception:
    pass
layout["Brooklvn.NY_USA"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/22_icon_Brooklvn.NY_USA.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (1152, 2804), _c22)
except Exception:
    pass
layout["Brooklvn.NY_USA"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 40, 61)
    canvas.paste(_c23, (1274, 0), _c23)
except Exception:
    pass
layout["icon_23"] = [1274, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/24_icon_Cadman_Plaza_West.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["Cadman_Plaza_West"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/25_icon_Sat_Mar_23.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (0, 2804), _c25)
except Exception:
    pass
layout["Sat,_Mar_23"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/26_icon_Promoted.png
try:
    _c26 = get_crop(26, 245, 65)
    canvas.paste(_c26, (85, 1684), _c26)
except Exception:
    pass
layout["Promoted"] = [85, 1684, 330, 1749]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/27_icon_United_Nations_Headquarters.png
try:
    _c27 = get_crop(27, 47, 59)
    canvas.paste(_c27, (281, 1689), _c27)
except Exception:
    pass
layout["United_Nations_Headquarte"] = [281, 1689, 328, 1748]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/28_text_9.39.png
try:
    _c28 = get_crop(28, 94, 45)
    canvas.paste(_c28, (17, 15), _c28)
except Exception:
    pass
layout["9.39"] = [17, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/29_text_6_690_events.png
try:
    _c29 = get_crop(29, 372, 135)
    canvas.paste(_c29, (54, 390), _c29)
except Exception:
    pass
layout["6,690_events"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/30_text_The_Youth_s_Vision_For_A_Sustainable_Fut.png
try:
    _c30 = get_crop(30, 1344, 1115)
    canvas.paste(_c30, (48, 676), _c30)
except Exception:
    pass
layout["The_Youth's_Vision_For_A_"] = [48, 676, 1392, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/31_text_Fri.png
try:
    _c31 = get_crop(31, 75, 50)
    canvas.paste(_c31, (91, 1560), _c31)
except Exception:
    pass
layout["Fri,"] = [91, 1560, 166, 1610]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/32_text_26.png
try:
    _c32 = get_crop(32, 64, 45)
    canvas.paste(_c32, (237, 1560), _c32)
except Exception:
    pass
layout["26"] = [237, 1560, 301, 1605]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/33_text_1O_00_AM_EDT.png
try:
    _c33 = get_crop(33, 276, 45)
    canvas.paste(_c33, (323, 1560), _c33)
except Exception:
    pass
layout["1O:00_AM_EDT"] = [323, 1560, 599, 1605]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/34_text_MARCH23.png
try:
    _c34 = get_crop(34, 209, 49)
    canvas.paste(_c34, (51, 1980), _c34)
except Exception:
    pass
layout["MARCH23"] = [51, 1980, 260, 2029]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_05_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-7/35_text_One_Pierrenont_Plaza.png
try:
    _c35 = get_crop(35, 288, 156)
    canvas.paste(_c35, (288, 2804), _c35)
except Exception:
    pass
layout["One_Pierrenont_Plaza"] = [288, 2804, 576, 2960]
