# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_09
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-11.png
# step_index: 9/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structure for 1440x2960 canvas using PIL draw and canvas provided.

w, h = canvas.size

# Overall background (dominant color: near-white)
draw.rectangle([(0, 0), (w, h)], fill="#FFFFFF")

# Status bar area at top (~96px) - light gray bar
status_h = 96
draw.rectangle([(0, 0), (w, status_h)], fill="#D0D0D0")

# Subtle divider under the status bar
draw.line([(0, status_h), (w, status_h)], fill="#C0C0C0", width=1)

# Header / Toolbar area (search area and large title region)
header_top = status_h
header_bottom = 320
draw.rectangle([(0, header_top), (w, header_bottom)], fill="#FFFFFF")
# Bottom divider of header
draw.line([(24, header_bottom), (w-24, header_bottom)], fill="#E6E6E6", width=2)

# Location / filter row background subtle (keeps white but add a faint divider above chips)
filters_row_top = header_bottom
filters_row_bottom = 460
# faint top divider
draw.line([(24, filters_row_top+8), (w-24, filters_row_top+8)], fill="#F0F0F0", width=1)
# faint bottom divider to separate from content
draw.line([(24, filters_row_bottom), (w-24, filters_row_bottom)], fill="#EDEDED", width=1)

# Large content container area (event list background — keep white but add subtle left/right margins)
content_top = filters_row_bottom + 16
content_left = 24
content_right = w - 24
content_bottom = h - 180  # leave space for bottom nav
draw.rectangle([(content_left, content_top), (content_right, content_bottom)], fill="#FFFFFF")

# Card-style white rounded backgrounds for event items (just the card panels, images/text will be pasted on top)
card_radius = 28
card_spacing = 48

# First event card background
card1_top = content_top + 80
card1_left = 48
card1_right = w - 48
card1_height = 520
card1_bottom = card1_top + card1_height
draw.rounded_rectangle(
    [(card1_left, card1_top), (card1_right, card1_bottom)],
    radius=card_radius,
    fill="#FFFFFF",
    outline="#ECECEC",
    width=1
)

# Divider / shadow line under first card
draw.line([(card1_left+8, card1_bottom+12), (card1_right-8, card1_bottom+12)], fill="#F2F2F2", width=2)

# Second event card background (further down)
card2_top = card1_bottom + 72
card2_left = 48
card2_right = w - 48
card2_height = 520
card2_bottom = card2_top + card2_height
draw.rounded_rectangle(
    [(card2_left, card2_top), (card2_right, card2_bottom)],
    radius=card_radius,
    fill="#FFFFFF",
    outline="#ECECEC",
    width=1
)

# Additional subtle separators between list items (thin lines)
sep_y = card2_bottom + 24
draw.line([(24, sep_y), (w-24, sep_y)], fill="#F5F5F5", width=1)

# Provide a faint background band for promoted/label areas (but do not draw chips or text)
promo_band_top = card1_bottom + 6
promo_band_bottom = promo_band_top + 34
draw.rectangle([(card1_left+16, promo_band_top), (card1_right-16, promo_band_bottom)], fill="#FCFCFC")

# Content section darker banner behind images (used as placeholder backdrop but images will be pasted above)
# Note: we keep these subtle and neutral so they won't duplicate image content
image_back_h = 300
image_back_left = card1_left + 16
image_back_right = card1_right - 16
image_back_top = card1_top + 16
image_back_bottom = image_back_top + image_back_h
draw.rounded_rectangle(
    [(image_back_left, image_back_top), (image_back_right, image_back_bottom)],
    radius=18,
    fill="#F7F7F9",
    outline=None
)

# Another image backdrop for second card
image2_back_top = card2_top + 16
image2_back_bottom = image2_back_top + image_back_h
draw.rounded_rectangle(
    [(image_back_left, image2_back_top), (image_back_right, image2_back_bottom)],
    radius=18,
    fill="#F7F7F9",
    outline=None
)

# Bottom navigation bar background (height matches detected clickable area ~156px)
nav_h = 156
nav_top = h - nav_h
draw.rectangle([(0, nav_top), (w, h)], fill="#FFFFFF")
# Top divider line for nav
draw.line([(0, nav_top), (w, nav_top)], fill="#E6E6E6", width=1)

# Slight shadow/inner line to give depth above nav bar
draw.line([(0, nav_top+2), (w, nav_top+2)], fill="#F8F8F8", width=1)

# Left and right safe margins vertical guides (subtle) to mimic app layout grid
draw.line([(24, 0), (24, h)], fill="#FFFFFF", width=0)
draw.line([(w-24, 0), (w-24, h)], fill="#FFFFFF", width=0)

# End of structural drawing. UI elements (icons, images, text) will be pasted on top separately.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2415), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/06_icon_Foo.png
try:
    _c6 = get_crop(6, 148, 110)
    canvas.paste(_c6, (1283, 406), _c6)
except Exception:
    pass
layout["Foo"] = [1283, 406, 1431, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2415), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/09_icon_Interactive_Live_Music_and_Jam_Session_a.png
try:
    _c9 = get_crop(9, 1344, 1175)
    canvas.paste(_c9, (48, 676), _c9)
except Exception:
    pass
layout["Interactive_Live_Music_an"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/10_icon_7.29.png
try:
    _c10 = get_crop(10, 130, 115)
    canvas.paste(_c10, (53, 113), _c10)
except Exception:
    pass
layout["7.29"] = [53, 113, 183, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/11_icon_Foo.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 96), _c11)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 54, 65)
    canvas.paste(_c12, (1151, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1151, 0, 1205, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/13_icon_Yoga_session.png
try:
    _c13 = get_crop(13, 69, 63)
    canvas.paste(_c13, (307, 0), _c13)
except Exception:
    pass
layout["Yoga_session"] = [307, 0, 376, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 101, 63)
    canvas.paste(_c14, (1211, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1211, 0, 1312, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 52, 63)
    canvas.paste(_c15, (247, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [247, 1, 299, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/16_icon_7.29.png
try:
    _c16 = get_crop(16, 60, 63)
    canvas.paste(_c16, (181, 0), _c16)
except Exception:
    pass
layout["7.29"] = [181, 0, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/17_icon_Yoga_session.png
try:
    _c17 = get_crop(17, 1344, 191)
    canvas.paste(_c17, (48, 72), _c17)
except Exception:
    pass
layout["Yoga_session"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 57, 61)
    canvas.paste(_c18, (1318, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1318, 0, 1375, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/19_icon_7.29.png
try:
    _c19 = get_crop(19, 60, 64)
    canvas.paste(_c19, (115, 0), _c19)
except Exception:
    pass
layout["7.29"] = [115, 0, 175, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/20_icon_New_York.png
try:
    _c20 = get_crop(20, 434, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/21_icon_Promoted.png
try:
    _c21 = get_crop(21, 265, 68)
    canvas.paste(_c21, (64, 1742), _c21)
except Exception:
    pass
layout["Promoted"] = [64, 1742, 329, 1810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/22_icon_4_._8.00_PM_EDT.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["4_._8.00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/23_icon_Yoga_session.png
try:
    _c23 = get_crop(23, 51, 61)
    canvas.paste(_c23, (384, 2), _c23)
except Exception:
    pass
layout["Yoga_session"] = [384, 2, 435, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/24_icon_Brooklyn.png
try:
    _c24 = get_crop(24, 1344, 1175)
    canvas.paste(_c24, (48, 676), _c24)
except Exception:
    pass
layout["Brooklyn"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/25_icon_pinkFROG_cafe.png
try:
    _c25 = get_crop(25, 284, 63)
    canvas.paste(_c25, (89, 1677), _c25)
except Exception:
    pass
layout["pinkFROG_cafe"] = [89, 1677, 373, 1740]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/26_icon_4_._8.00_PM_EDT.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (864, 2804), _c26)
except Exception:
    pass
layout["4_._8.00_PM_EDT"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/27_icon_4_._8.00_PM_EDT.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (576, 2804), _c27)
except Exception:
    pass
layout["4_._8.00_PM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/28_icon_Ypphur.png
try:
    _c28 = get_crop(28, 1344, 917)
    canvas.paste(_c28, (48, 1899), _c28)
except Exception:
    pass
layout["Ypphur"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/29_icon_More.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (1152, 2804), _c29)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/30_icon_7.29.png
try:
    _c30 = get_crop(30, 146, 63)
    canvas.paste(_c30, (5, 0), _c30)
except Exception:
    pass
layout["7.29"] = [5, 0, 151, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/31_text_8_379events.png
try:
    _c31 = get_crop(31, 359, 103)
    canvas.paste(_c31, (54, 410), _c31)
except Exception:
    pass
layout["8,379events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_09_2024_4_23_19_27_45f56b06f31541079045047b6d542613-11/32_clickable_Home.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (0, 2804), _c32)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
