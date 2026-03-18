# page_id: page_eventbrite_182a9ea236924e9eb43ccbe82fd1506e_05
# screenshot: 2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7.png
# step_index: 5/13
# task: Open Eventbrite. Set time to tomorrow. Clear all search filters. Select the third one in New York. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the Eventbrite-like page.
# Variables available: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm/font_md/font_lg/font_xl

# Canvas size
W, H = canvas.size

# Colors
status_bar_color = (196, 196, 196)        # soft gray for status bar
toolbar_bg = (247, 246, 249)              # very light lavender/white for header area
divider_color = (220, 220, 225)           # subtle divider line
card_bg = (255, 255, 255)                 # card white
muted_bg = (250, 250, 252)                # very slight off-white band
image_placeholder_dark = (40, 20, 45)     # deep plum for poster/image background
nav_bar_bg = (255, 255, 255)              # bottom nav background (white)
subtle_shadow = (235, 235, 238)           # used as faint borders

# Status bar (top ~72px)
status_h = 72
draw.rectangle([0, 0, W, status_h], fill=status_bar_color)

# Header / toolbar area under status bar (~72px to ~220px)
toolbar_top = status_h
toolbar_bottom = 220
draw.rectangle([0, toolbar_top, W, toolbar_bottom], fill=toolbar_bg)

# Thin divider line under the main search/header area
divider_y = toolbar_bottom + 12
draw.line([(48, divider_y), (W - 48, divider_y)], fill=divider_color, width=2)

# Light band for filter/category area (where pills will be pasted on top)
filters_band_top = divider_y + 12
filters_band_bottom = filters_band_top + 120
draw.rectangle([0, filters_band_top, W, filters_band_bottom], fill=muted_bg)

# A subtle horizontal rule below the filter band
draw.line([(48, filters_band_bottom + 6), (W - 48, filters_band_bottom + 6)], fill=subtle_shadow, width=1)

# Large rounded card area for the first list item (carousel + title area)
card1_x0, card1_y0 = 32, filters_band_bottom + 48
card1_x1, card1_y1 = W - 32, card1_y0 + 520
draw.rounded_rectangle([card1_x0, card1_y0, card1_x1, card1_y1], radius=20, fill=card_bg, outline=subtle_shadow, width=1)

# Separator line below the first card (thin)
sep1_y = card1_y1 + 20
draw.line([(48, sep1_y), (W - 48, sep1_y)], fill=divider_color, width=2)

# Second card area (holds poster image + event details)
card2_x0, card2_y0 = 32, sep1_y + 20
card2_x1, card2_y1 = W - 32, card2_y0 + 920
draw.rounded_rectangle([card2_x0, card2_y0, card2_x1, card2_y1], radius=20, fill=card_bg, outline=subtle_shadow, width=1)

# Within second card, add a darker poster/image background block (image will be pasted on top)
poster_x0 = card2_x0 + 16
poster_y0 = card2_y0 + 28
poster_x1 = card2_x1 - 16
poster_y1 = poster_y0 + 420
draw.rounded_rectangle([poster_x0, poster_y0, poster_x1, poster_y1], radius=18, fill=image_placeholder_dark)

# Thin divider between poster and event details inside the card
draw.line([(poster_x0, poster_y1 + 18), (poster_x1, poster_y1 + 18)], fill=subtle_shadow, width=1)

# Another content card lower on the page for promoted/featured section
promo_card_x0, promo_card_y0 = 48, card2_y1 + 32
promo_card_x1, promo_card_y1 = W - 48, promo_card_y0 + 360
draw.rounded_rectangle([promo_card_x0, promo_card_y0, promo_card_x1, promo_card_y1], radius=16, fill=card_bg, outline=subtle_shadow, width=1)

# Large full-width image area further down (placeholder background)
large_img_x0, large_img_y0 = 48, promo_card_y1 + 24
large_img_x1, large_img_y1 = W - 48, large_img_y0 + 980
draw.rectangle([large_img_x0, large_img_y0, large_img_x1, large_img_y1], fill=image_placeholder_dark)

# Horizontal separators for the list (thin lines)
for y in (card1_y1 + 160, card1_y1 + 340, card2_y1 + 480):
    if y < H - 200:
        draw.line([(48, y), (W - 48, y)], fill=divider_color, width=1)

# Bottom navigation bar area (~120px high)
nav_h = 120
nav_y0 = H - nav_h
draw.rectangle([0, nav_y0, W, H], fill=nav_bar_bg)
# Top hairline for nav bar
draw.line([(24, nav_y0), (W - 24, nav_y0)], fill=subtle_shadow, width=1)

# Slight left/right padding bars to imply safe area edges (visual structure)
edge_strip_w = 12
draw.rectangle([0, status_h, edge_strip_w, H - nav_h], fill=toolbar_bg)
draw.rectangle([W - edge_strip_w, status_h, W, H - nav_h], fill=toolbar_bg)

# Final subtle drop line under status bar to separate from header
draw.line([(0, status_h), (W, status_h)], fill=subtle_shadow, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/04_icon_Foo.png
try:
    _c4 = get_crop(4, 154, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1436, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/05_icon_IMETHOD.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2336), _c5)
except Exception:
    pass
layout["IMETHOD"] = [1092, 2336, 1236, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/06_icon_IMETHOD.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2336), _c6)
except Exception:
    pass
layout["IMETHOD"] = [1236, 2336, 1380, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/07_icon_9.31.png
try:
    _c7 = get_crop(7, 128, 116)
    canvas.paste(_c7, (54, 114), _c7)
except Exception:
    pass
layout["9.31"] = [54, 114, 182, 230]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 56, 61)
    canvas.paste(_c8, (247, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [247, 1, 303, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/09_icon_Search_forae.png
try:
    _c9 = get_crop(9, 63, 63)
    canvas.paste(_c9, (311, 1), _c9)
except Exception:
    pass
layout["Search_forae"] = [311, 1, 374, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/10_icon_9.31.png
try:
    _c10 = get_crop(10, 56, 62)
    canvas.paste(_c10, (182, 0), _c10)
except Exception:
    pass
layout["9.31"] = [182, 0, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/11_icon_9.31.png
try:
    _c11 = get_crop(11, 62, 64)
    canvas.paste(_c11, (111, 0), _c11)
except Exception:
    pass
layout["9.31"] = [111, 0, 173, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/12_icon_New_York.png
try:
    _c12 = get_crop(12, 434, 144)
    canvas.paste(_c12, (0, 259), _c12)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 102, 60)
    canvas.paste(_c13, (1205, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1205, 0, 1307, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 66, 59)
    canvas.paste(_c14, (1314, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1314, 0, 1380, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1236, 1192), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/16_icon_GLABING_BLOOM_SOUND_COLLECTIVE.png
try:
    _c16 = get_crop(16, 1344, 996)
    canvas.paste(_c16, (48, 1820), _c16)
except Exception:
    pass
layout["GLABING;_BLOOM_SOUND_COLL"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/17_icon_Search_forae.png
try:
    _c17 = get_crop(17, 51, 61)
    canvas.paste(_c17, (383, 2), _c17)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/18_icon_Search_forae.png
try:
    _c18 = get_crop(18, 1344, 191)
    canvas.paste(_c18, (48, 72), _c18)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/19_icon_The_Snace_at_Irondale.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["The_Snace_at_Irondale"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/20_icon_slO_2Lo.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (864, 2804), _c20)
except Exception:
    pass
layout["slO_2Lo"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/21_icon_Favorite_button.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1092, 1192), _c21)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/22_icon_Promoted.png
try:
    _c22 = get_crop(22, 244, 66)
    canvas.paste(_c22, (84, 1665), _c22)
except Exception:
    pass
layout["Promoted"] = [84, 1665, 328, 1731]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/23_icon_Tequila_Artistic_Transformation.png
try:
    _c23 = get_crop(23, 1344, 1096)
    canvas.paste(_c23, (48, 676), _c23)
except Exception:
    pass
layout["Tequila_&_Artistic_Transf"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/24_icon_slO_2Lo.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["slO_2Lo"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/25_icon_6.30_PM_EDT.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (576, 2804), _c25)
except Exception:
    pass
layout["6.30_PM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/26_icon_Anytime.png
try:
    _c26 = get_crop(26, 210, 292)
    canvas.paste(_c26, (477, 670), _c26)
except Exception:
    pass
layout["Anytime"] = [477, 670, 687, 962]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/27_icon_10_000_events.png
try:
    _c27 = get_crop(27, 213, 295)
    canvas.paste(_c27, (217, 669), _c27)
except Exception:
    pass
layout["10,000_events"] = [217, 669, 430, 964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/28_icon_Wed_Mar_20.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Wed,_Mar_20"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/29_text_9.31.png
try:
    _c29 = get_crop(29, 89, 43)
    canvas.paste(_c29, (20, 17), _c29)
except Exception:
    pass
layout["9.31"] = [20, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/30_text_10_000_events.png
try:
    _c30 = get_crop(30, 359, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/31_text_3.20.24.png
try:
    _c31 = get_crop(31, 172, 40)
    canvas.paste(_c31, (649, 1819), _c31)
except Exception:
    pass
layout["3.20.24"] = [649, 1819, 821, 1859]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/32_text_Wed_Mar_20.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (0, 2804), _c32)
except Exception:
    pass
layout["Wed,_Mar_20"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/33_text_6.30_PM_EDT.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (288, 2804), _c33)
except Exception:
    pass
layout["6.30_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/182a9ea236924e9eb43ccbe82fd1506e/step_05_2024_3_20_17_30_182a9ea236924e9eb43ccbe82fd1506e-7/34_text_The_Snace_at_Irondale.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (288, 2804), _c34)
except Exception:
    pass
layout["The_Snace_at_Irondale"] = [288, 2804, 576, 2960]
