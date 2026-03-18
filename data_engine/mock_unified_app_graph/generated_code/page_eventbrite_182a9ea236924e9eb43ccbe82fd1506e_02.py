# page_id: page_eventbrite_182a9ea236924e9eb43ccbe82fd1506e_02
# screenshot: 2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4.png
# step_index: 2/13
# task: Open Eventbrite. Set time to tomorrow. Clear all search filters. Select the third one in New York. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI drawing for Eventbrite-like page
# Assumes: canvas (1440x2960 RGB PIL Image) and draw (ImageDraw) are provided

# Colors
bg_color = (247, 246, 248)        # overall very light background
status_bar_color = (189, 189, 189) # top status bar gray
search_bg = (255, 255, 255)       # white search bar
search_outline = (230, 230, 230)
chip_band = (237, 249, 255)       # pale blue band behind chips
card_bg = (255, 255, 255)         # white cards
card_shadow = (232, 232, 232)
divider = (220, 220, 222)
image_placeholder = (22, 24, 27)  # dark area behind large poster images
bottom_nav_bg = (255, 255, 255)
muted_line = (240, 240, 241)

W, H = canvas.size

# Fill overall background
draw.rectangle([(0, 0), (W, H)], fill=bg_color)

# Status bar (top area)
status_h = 84
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# Subtle bottom line under status bar
draw.line([(0, status_h), (W, status_h)], fill=divider, width=1)

# Search bar area (rounded)
search_left = 48
search_right = W - 48
search_top = status_h + 12
search_height = 176
search_bottom = search_top + search_height
search_radius = 36
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=search_radius,
    fill=search_bg,
    outline=search_outline,
    width=1
)

# Thin divider under the search bar
divider_y = search_bottom + 8
draw.line([(48, divider_y), (W - 48, divider_y)], fill=muted_line, width=1)

# Location row separator (subtle)
loc_row_top = divider_y + 12
draw.line([(48, loc_row_top + 92), (W - 48, loc_row_top + 92)], fill=divider, width=1)

# Chips background band (behind selectable chips)
chips_band_top = loc_row_top + 72
chips_band_bottom = chips_band_top + 120
chips_left = 48
chips_right = W - 48
draw.rounded_rectangle(
    [(chips_left, chips_band_top), (chips_right, chips_band_bottom)],
    radius=60,
    fill=chip_band,
    outline=None
)

# Section title separator (above the event count/title area)
title_sep_y = chips_band_bottom + 28
draw.line([(48, title_sep_y), (W - 48, title_sep_y)], fill=muted_line, width=1)

# Large carousel / hero image background (placeholder area behind pasted images)
carousel_top = title_sep_y + 28
carousel_bottom = carousel_top + 260
carousel_left = 48
carousel_right = W - 48
# shadow
draw.rectangle(
    [(carousel_left + 6, carousel_top + 8), (carousel_right + 6, carousel_bottom + 8)],
    fill=card_shadow
)
# background card
draw.rectangle([(carousel_left, carousel_top), (carousel_right, carousel_bottom)], fill=card_bg)

# subtle divider under carousel
draw.line([(48, carousel_bottom + 18), (W - 48, carousel_bottom + 18)], fill=divider, width=1)

# First event card (rounded white card background)
card1_top = carousel_bottom + 36
card1_bottom = card1_top + 420
card_margin = 24
card1_left = card_margin
card1_right = W - card_margin
card_radius = 24
# shadow
draw.rounded_rectangle(
    [(card1_left + 6, card1_top + 8), (card1_right + 6, card1_bottom + 8)],
    radius=card_radius,
    fill=card_shadow
)
# card
draw.rounded_rectangle(
    [(card1_left, card1_top), (card1_right, card1_bottom)],
    radius=card_radius,
    fill=card_bg,
    outline=None
)

# thin separator line within first card (to separate image area from text area)
inner_sep_y = card1_top + 160
draw.line([(card1_left + 24, inner_sep_y), (card1_right - 24, inner_sep_y)], fill=muted_line, width=1)

# Second poster area (dark image background placeholder)
poster_top = card1_bottom + 36
poster_left = 48
poster_right = W - 48
poster_height = 420
# shadow for poster
draw.rectangle([(poster_left + 6, poster_top + 8), (poster_right + 6, poster_top + poster_height + 8)], fill=card_shadow)
# dark placeholder behind the poster image
draw.rounded_rectangle(
    [(poster_left, poster_top), (poster_right, poster_top + poster_height)],
    radius=20,
    fill=image_placeholder
)

# thin divider after poster
poster_div_y = poster_top + poster_height + 28
draw.line([(48, poster_div_y), (W - 48, poster_div_y)], fill=divider, width=1)

# Third event card area (rounded white card)
card3_top = poster_div_y + 24
card3_bottom = card3_top + 400
card3_left = 24
card3_right = W - 24
# shadow
draw.rounded_rectangle(
    [(card3_left + 6, card3_top + 8), (card3_right + 6, card3_bottom + 8)],
    radius=20,
    fill=card_shadow
)
# card
draw.rounded_rectangle(
    [(card3_left, card3_top), (card3_right, card3_bottom)],
    radius=20,
    fill=card_bg
)

# separators between list items (subtle lines)
sep_y = card3_bottom + 20
draw.line([(48, sep_y), (W - 48, sep_y)], fill=muted_line, width=1)

# Bottom navigation bar background with top border
nav_height = 120
nav_top = H - nav_height
draw.rectangle([(0, nav_top), (W, H)], fill=bottom_nav_bg)
# top divider
draw.line([(0, nav_top), (W, nav_top)], fill=divider, width=1)

# Small rounded pill at left used as subtle handle in some UIs (purely decorative)
handle_w = 120
handle_h = 8
handle_x = (W - handle_w) // 2
handle_y = status_h - 18
draw.rounded_rectangle([(handle_x, handle_y), (handle_x + handle_w, handle_y + handle_h)], radius=4, fill=muted_line)

# subtle left and right page margins vertical guides (very faint)
draw.line([(24, nav_top), (24, H)], fill=(250, 250, 250), width=1)
draw.line([(W - 24, nav_top), (W - 24, H)], fill=(250, 250, 250), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/04_icon_Foo.png
try:
    _c4 = get_crop(4, 154, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1436, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/05_icon_IMETHOD.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2336), _c5)
except Exception:
    pass
layout["IMETHOD"] = [1092, 2336, 1236, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/06_icon_IMETHOD.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2336), _c6)
except Exception:
    pass
layout["IMETHOD"] = [1236, 2336, 1380, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/07_icon_9.31.png
try:
    _c7 = get_crop(7, 128, 116)
    canvas.paste(_c7, (54, 114), _c7)
except Exception:
    pass
layout["9.31"] = [54, 114, 182, 230]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 56, 61)
    canvas.paste(_c8, (247, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [247, 1, 303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/09_icon_Search_forae.png
try:
    _c9 = get_crop(9, 63, 63)
    canvas.paste(_c9, (311, 1), _c9)
except Exception:
    pass
layout["Search_forae"] = [311, 1, 374, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/10_icon_9.31.png
try:
    _c10 = get_crop(10, 56, 62)
    canvas.paste(_c10, (182, 0), _c10)
except Exception:
    pass
layout["9.31"] = [182, 0, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/11_icon_9.31.png
try:
    _c11 = get_crop(11, 62, 64)
    canvas.paste(_c11, (111, 0), _c11)
except Exception:
    pass
layout["9.31"] = [111, 0, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/12_icon_New_York.png
try:
    _c12 = get_crop(12, 434, 144)
    canvas.paste(_c12, (0, 259), _c12)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 102, 60)
    canvas.paste(_c13, (1205, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1205, 0, 1307, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 66, 59)
    canvas.paste(_c14, (1314, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1314, 0, 1380, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1236, 1192), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/16_icon_GLABING_BLOOM_SOUND_COLLECTIVE.png
try:
    _c16 = get_crop(16, 1344, 996)
    canvas.paste(_c16, (48, 1820), _c16)
except Exception:
    pass
layout["GLABING;_BLOOM_SOUND_COLL"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/17_icon_Search_forae.png
try:
    _c17 = get_crop(17, 51, 61)
    canvas.paste(_c17, (383, 2), _c17)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/18_icon_Search_forae.png
try:
    _c18 = get_crop(18, 1344, 191)
    canvas.paste(_c18, (48, 72), _c18)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/19_icon_The_Snace_at_Irondale.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["The_Snace_at_Irondale"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/20_icon_slO_2Lo.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (864, 2804), _c20)
except Exception:
    pass
layout["slO_2Lo"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/21_icon_Favorite_button.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1092, 1192), _c21)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/22_icon_Promoted.png
try:
    _c22 = get_crop(22, 244, 66)
    canvas.paste(_c22, (84, 1665), _c22)
except Exception:
    pass
layout["Promoted"] = [84, 1665, 328, 1731]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/23_icon_Tequila_Artistic_Transformation.png
try:
    _c23 = get_crop(23, 1344, 1096)
    canvas.paste(_c23, (48, 676), _c23)
except Exception:
    pass
layout["Tequila_&_Artistic_Transf"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/24_icon_slO_2Lo.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["slO_2Lo"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/25_icon_6.30_PM_EDT.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["6.30_PM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/26_icon_Anytime.png
try:
    _c26 = get_crop(26, 210, 292)
    canvas.paste(_c26, (477, 670), _c26)
except Exception:
    pass
layout["Anytime"] = [477, 670, 687, 962]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/27_icon_10_000_events.png
try:
    _c27 = get_crop(27, 213, 295)
    canvas.paste(_c27, (217, 669), _c27)
except Exception:
    pass
layout["10,000_events"] = [217, 669, 430, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/28_icon_Wed_Mar_20.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Wed,_Mar_20"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/29_text_9.31.png
try:
    _c29 = get_crop(29, 89, 43)
    canvas.paste(_c29, (20, 17), _c29)
except Exception:
    pass
layout["9.31"] = [20, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/30_text_10_000_events.png
try:
    _c30 = get_crop(30, 359, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/31_text_3.20.24.png
try:
    _c31 = get_crop(31, 172, 40)
    canvas.paste(_c31, (649, 1819), _c31)
except Exception:
    pass
layout["3.20.24"] = [649, 1819, 821, 1859]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/32_text_Wed_Mar_20.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (0, 2804), _c32)
except Exception:
    pass
layout["Wed,_Mar_20"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/33_text_6.30_PM_EDT.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (288, 2804), _c33)
except Exception:
    pass
layout["6.30_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_02_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-4/34_text_The_Snace_at_Irondale.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (288, 2804), _c34)
except Exception:
    pass
layout["The_Snace_at_Irondale"] = [288, 2804, 576, 2960]
