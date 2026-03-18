# page_id: page_eventbrite_9fdb2ee43d5a49adac5304bdd5dacfc2_04
# screenshot: 2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6.png
# step_index: 4/8
# task: Open Eventbrite. Look up 'Pet' events. Filter by events happening this weekend. Select the third non-promoted event from the results - how much are the tickets for the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (1440x2960 RGB)
w, h = canvas.size

# Colors
bg = "#FBFCFD"            # overall very light background
status_bar = "#BDBDBE"    # top status bar gray
divider = "#E6E7E9"       # subtle divider lines
card_bg = "#FFFFFF"       # card background white
card_shadow = "#ECEFF3"   # shadow for elevation
chip_bg = "#EAF6FF"       # not used for chips themselves (chips are pasted), but for subtle band
nav_bg = "#FFFFFF"        # bottom nav background
muted = "#F4F6F8"         # very subtle separators

# Fill background
draw.rectangle((0, 0, w, h), fill=bg)

# Status bar at the very top (~72px)
status_h = 72
draw.rectangle((0, 0, w, status_h), fill=status_bar)

# Header / toolbar area under status bar
header_y0 = status_h
header_y1 = 180
draw.rectangle((0, header_y0, w, header_y1), fill=card_bg)
# header bottom divider
draw.line((32, header_y1 - 1, w - 32, header_y1 - 1), fill=divider, width=1)

# Search field background (rounded) - behind the pasted search text/icon
search_x0, search_x1 = 48, w - 48
search_y0, search_y1 = header_y0 + 20, header_y0 + 84
draw.rounded_rectangle((search_x0, search_y0, search_x1, search_y1),
                       radius=12, fill=card_bg, outline=divider, width=1)

# Location / filters band area (keeps a clean white area for chips to be pasted)
loc_band_y0 = header_y1
loc_band_y1 = 460
draw.rectangle((0, loc_band_y0, w, loc_band_y1), fill=card_bg)
# light divider below filters
draw.line((24, loc_band_y1 - 1, w - 24, loc_band_y1 - 1), fill=muted, width=1)

# Content area background (slightly different to separate from header)
content_y0 = loc_band_y1
content_y1 = h - 140
draw.rectangle((0, content_y0, w, content_y1), fill=bg)

# First event card background with subtle shadow and rounded corners
card1_x0, card1_x1 = 48, w - 48
card1_y0 = content_y0 + 60
card1_h = 620
# shadow
draw.rounded_rectangle((card1_x0 + 6, card1_y0 + 8, card1_x1 + 6, card1_y0 + card1_h + 8),
                       radius=28, fill=card_shadow)
# card
draw.rounded_rectangle((card1_x0, card1_y0, card1_x1, card1_y0 + card1_h),
                       radius=28, fill=card_bg, outline=divider, width=1)

# Divider separating image area (top portion) and info area (bottom portion) of first card
# Note: actual image and text will be pasted on top by the pipeline; we only provide structure.
img_area_height = int(card1_h * 0.45)
draw.line((card1_x0 + 20, card1_y0 + img_area_height, card1_x1 - 20, card1_y0 + img_area_height),
          fill=muted, width=1)

# Subtle tag area under image where small labels may be pasted (we leave it blank but show slight band)
tag_band_y0 = card1_y0 + img_area_height + 12
tag_band_y1 = tag_band_y0 + 42
draw.rectangle((card1_x0 + 28, tag_band_y0, card1_x1 - 28, tag_band_y1), fill=bg)

# Second content card / promoted banner (structure only)
card2_y0 = card1_y0 + card1_h + 48
card2_h = 760
# shadow
draw.rounded_rectangle((card1_x0 + 6, card2_y0 + 8, card1_x1 + 6, card2_y0 + card2_h + 8),
                       radius=20, fill=card_shadow)
# card
draw.rounded_rectangle((card1_x0, card2_y0, card1_x1, card2_y0 + card2_h),
                       radius=20, fill=card_bg, outline=divider, width=1)

# small separator between list items
sep_y = card2_y0 + card2_h + 36
draw.line((24, sep_y, w - 24, sep_y), fill=muted, width=1)

# Bottom safe area / navigation bar
nav_h = 100
nav_y0 = h - nav_h
draw.rectangle((0, nav_y0, w, h), fill=nav_bg)
# top divider of nav
draw.line((0, nav_y0, w, nav_y0), fill=divider, width=1)

# Floating subtle line to indicate content end above nav
draw.line((24, nav_y0 - 12, w - 24, nav_y0 - 12), fill="#F0F1F3", width=1)

# Optional: subtle left/right margins guide lines (very faint) to help alignment of pasted elements
guide_color = "#FFFFFF00"  # fully transparent placeholder (no visible guides)
# (intentionally left as invisible so nothing is accidentally duplicated)

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/04_icon_Foo.png
try:
    _c4 = get_crop(4, 150, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/07_icon_Emall_SHERRYLEAK_GmmLgOL.png
try:
    _c7 = get_crop(7, 1344, 977)
    canvas.paste(_c7, (48, 1839), _c7)
except Exception:
    pass
layout["Emall;_SHERRYLEAK@GmmLgOL"] = [48, 1839, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/08_icon_Waslina_laxr_anb.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1092, 2355), _c8)
except Exception:
    pass
layout["Waslina_(laxr_(anb"] = [1092, 2355, 1236, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/09_icon_Foo.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/10_icon_4.47.png
try:
    _c10 = get_crop(10, 123, 112)
    canvas.paste(_c10, (55, 115), _c10)
except Exception:
    pass
layout["4.47"] = [55, 115, 178, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/11_icon_Pets.png
try:
    _c11 = get_crop(11, 66, 63)
    canvas.paste(_c11, (308, 1), _c11)
except Exception:
    pass
layout["Pets"] = [308, 1, 374, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/12_icon_4.47.png
try:
    _c12 = get_crop(12, 58, 65)
    canvas.paste(_c12, (114, 0), _c12)
except Exception:
    pass
layout["4.47"] = [114, 0, 172, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/13_icon_4.47.png
try:
    _c13 = get_crop(13, 57, 64)
    canvas.paste(_c13, (182, 0), _c13)
except Exception:
    pass
layout["4.47"] = [182, 0, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 103, 61)
    canvas.paste(_c14, (1207, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1207, 0, 1310, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/15_icon_Washington.png
try:
    _c15 = get_crop(15, 493, 144)
    canvas.paste(_c15, (0, 259), _c15)
except Exception:
    pass
layout["Washington"] = [0, 259, 493, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/16_icon_Waslina_laxr_anb.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1236, 2355), _c16)
except Exception:
    pass
layout["Waslina_(laxr_(anb"] = [1236, 2355, 1380, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/17_icon_Wedding_Place_Cards_Trade_shows.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (576, 2804), _c17)
except Exception:
    pass
layout["Wedding_Place_Cards_&_Tra"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/18_icon_Pets.png
try:
    _c18 = get_crop(18, 51, 63)
    canvas.paste(_c18, (247, 1), _c18)
except Exception:
    pass
layout["Pets"] = [247, 1, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/19_icon_Pets.png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["Pets"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 58, 61)
    canvas.paste(_c20, (1319, 0), _c20)
except Exception:
    pass
layout["icon_20"] = [1319, 0, 1377, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/21_icon_7_00_PM_EDT.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["7:00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/22_icon_Digital_Caricatures_drawn_from_photos_fo.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (864, 2804), _c22)
except Exception:
    pass
layout["Digital_Caricatures_drawn"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/23_icon_Waslina_laxr_anb.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (1152, 2804), _c23)
except Exception:
    pass
layout["Waslina_(laxr_(anb"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 50, 61)
    canvas.paste(_c24, (384, 3), _c24)
except Exception:
    pass
layout["icon_24"] = [384, 3, 434, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/25_icon_4_00_PM_EDT.png
try:
    _c25 = get_crop(25, 1344, 1115)
    canvas.paste(_c25, (48, 676), _c25)
except Exception:
    pass
layout["4:00_PM_EDT"] = [48, 676, 1392, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/26_icon_4.47.png
try:
    _c26 = get_crop(26, 95, 63)
    canvas.paste(_c26, (12, 0), _c26)
except Exception:
    pass
layout["4.47"] = [12, 0, 107, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/27_icon_tickets_left.png
try:
    _c27 = get_crop(27, 1344, 1115)
    canvas.paste(_c27, (48, 676), _c27)
except Exception:
    pass
layout["tickets_left"] = [48, 676, 1392, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/28_icon_Shrimp_Blast_2024.png
try:
    _c28 = get_crop(28, 513, 81)
    canvas.paste(_c28, (83, 1462), _c28)
except Exception:
    pass
layout["Shrimp_Blast_2024"] = [83, 1462, 596, 1543]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/29_icon_Online.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["Online"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/30_text_216_events.png
try:
    _c30 = get_crop(30, 359, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["216_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/31_text_Wedding_Bridesmaids_Groomsmen_Caricature.png
try:
    _c31 = get_crop(31, 1344, 977)
    canvas.paste(_c31, (48, 1839), _c31)
except Exception:
    pass
layout["Wedding;_Bridesmaids;_&_G"] = [48, 1839, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/32_text_Mon_Apr_29.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (0, 2804), _c32)
except Exception:
    pass
layout["Mon,_Apr_29"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/33_text_7_00_PM_EDT.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (288, 2804), _c33)
except Exception:
    pass
layout["7:00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/9fdb2ee43d5a49adac5304bdd5dacfc2/step_04_2024_4_24_16_46_9fdb2ee43d5a49adac5304bdd5dacfc2-6/34_text_Online.png
try:
    _c34 = get_crop(34, 288, 156)
    canvas.paste(_c34, (0, 2804), _c34)
except Exception:
    pass
layout["Online"] = [0, 2804, 288, 2960]
