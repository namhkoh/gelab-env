# page_id: page_eventbrite_c7c81d1bf6744774b99294e9f124dda3_09
# screenshot: 2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11.png
# step_index: 9/10
# task: Open Eventbrite. Search for "Fitness". Select the events in the location "Chicago". What is the price of the first event in listing?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (1440x2960)
# Colors (picked to match the screenshot's dominant tones)
bg_color = (250, 251, 253)        # very light off-white background
status_bar_color = (189, 189, 189) # muted grey for status bar
header_bg = (255, 255, 255)        # header is white
divider_color = (225, 228, 233)    # subtle divider
chip_band_color = (244, 249, 255)  # very pale blue behind filter chips
card_shadow = (232, 235, 240)      # light shadow for cards
card_bg = (255, 255, 255)          # card background white
image_placeholder = (240, 243, 246) # pale image area (will be covered by pasted images)
bottom_nav_bg = (255, 255, 255)
accent_divider = (238, 241, 246)

W, H = canvas.size

# Fill overall background
draw.rectangle((0, 0, W, H), fill=bg_color)

# Status bar area (top ~96px)
status_h = 96
draw.rectangle((0, 0, W, status_h), fill=status_bar_color)

# Header / toolbar area below status bar (~96-168)
header_top = status_h
header_bottom = 168
draw.rectangle((0, header_top, W, header_bottom), fill=header_bg)

# Subtle divider under header
draw.line((48, header_bottom, W - 48, header_bottom), fill=divider_color, width=2)

# Filter chips background band (behind the pill chips row)
chips_top = 360
chips_bottom = 460
# wide pale band to give chips a soft background area
draw.rectangle((24, chips_top, W - 24, chips_bottom), fill=chip_band_color)
# light inner divider under chip band
draw.line((24, chips_bottom + 8, W - 24, chips_bottom + 8), fill=accent_divider, width=1)

# First event card background with shadow (position from detections)
card1_x0, card1_y0 = 48, 676
card1_x1, card1_y1 = 1392, 1870  # 48 + 1344 = 1392 ; height from detection
card_radius = 28

# shadow
shadow_offset = 10
draw.rounded_rectangle(
    (card1_x0, card1_y0 + shadow_offset, card1_x1, card1_y1 + shadow_offset),
    radius=card_radius, fill=card_shadow
)
# card background
draw.rounded_rectangle((card1_x0, card1_y0, card1_x1, card1_y1), radius=card_radius, fill=card_bg, outline=None)

# Image area placeholder inside first card (top portion)
# Keep it subtle since actual image will be pasted over this area
img1_h = 420
draw.rounded_rectangle(
    (card1_x0 + 8, card1_y0 + 8, card1_x1 - 8, card1_y0 + img1_h),
    radius=18, fill=image_placeholder
)

# Thin divider between image area and text area inside card (subtle)
draw.line((card1_x0 + 20, card1_y0 + img1_h + 6, card1_x1 - 20, card1_y0 + img1_h + 6), fill=accent_divider, width=1)

# Second event card background (position from detections)
card2_x0, card2_y0 = 48, 1918
card2_x1, card2_y1 = 1392, 2816  # 48 + 1344, 1918 + 898 = 2816
# shadow
draw.rounded_rectangle(
    (card2_x0, card2_y0 + shadow_offset, card2_x1, card2_y1 + shadow_offset),
    radius=card_radius, fill=card_shadow
)
# card
draw.rounded_rectangle((card2_x0, card2_y0, card2_x1, card2_y1), radius=card_radius, fill=card_bg, outline=None)

# Image area placeholder in second card (top portion)
img2_h = 420
draw.rounded_rectangle(
    (card2_x0 + 8, card2_y0 + 8, card2_x1 - 8, card2_y0 + img2_h),
    radius=18, fill=image_placeholder
)
draw.line((card2_x0 + 20, card2_y0 + img2_h + 6, card2_x1 - 20, card2_y0 + img2_h + 6), fill=accent_divider, width=1)

# Separator line between list and bottom nav
bottom_nav_top = 2804
draw.line((24, bottom_nav_top, W - 24, bottom_nav_top), fill=divider_color, width=2)

# Bottom navigation background area
draw.rectangle((0, bottom_nav_top, W, H), fill=bottom_nav_bg)

# Subtle horizontal guideline near top of page (under search header)
draw.line((24, header_bottom + 8, W - 24, header_bottom + 8), fill=accent_divider, width=1)

# Left edge vertical margin guide (very subtle) to match layout margins
draw.line((48, header_bottom + 12, 48, H - 200), fill=(245, 247, 249), width=1)
draw.line((W - 48, header_bottom + 12, W - 48, H - 200), fill=(245, 247, 249), width=1)

# Done: Background, status bar, header dividers, card backgrounds, separators and bottom nav.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 2434), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2434, 1236, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/05_icon_Overflow_menu_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 2434), _c5)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2434, 1380, 2578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/06_icon_Foo.png
try:
    _c6 = get_crop(6, 148, 110)
    canvas.paste(_c6, (1283, 406), _c6)
except Exception:
    pass
layout["Foo"] = [1283, 406, 1431, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1092, 1192), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/09_icon_Foo.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 54, 66)
    canvas.paste(_c10, (1151, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1151, 0, 1205, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/11_icon_7.10.png
try:
    _c11 = get_crop(11, 126, 117)
    canvas.paste(_c11, (53, 112), _c11)
except Exception:
    pass
layout["7.10"] = [53, 112, 179, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/12_icon_7.10.png
try:
    _c12 = get_crop(12, 60, 65)
    canvas.paste(_c12, (180, 0), _c12)
except Exception:
    pass
layout["7.10"] = [180, 0, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/13_icon_Fitness.png
try:
    _c13 = get_crop(13, 68, 64)
    canvas.paste(_c13, (307, 0), _c13)
except Exception:
    pass
layout["Fitness"] = [307, 0, 375, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 99, 63)
    canvas.paste(_c14, (1212, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1212, 0, 1311, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/15_icon_Fitness.png
try:
    _c15 = get_crop(15, 54, 65)
    canvas.paste(_c15, (246, 0), _c15)
except Exception:
    pass
layout["Fitness"] = [246, 0, 300, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/16_icon_SATURDAYZ_MASADA_Afrobeats.png
try:
    _c16 = get_crop(16, 1344, 1194)
    canvas.paste(_c16, (48, 676), _c16)
except Exception:
    pass
layout["SATURDAYZ_@_MASADA:_Afrob"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 55, 62)
    canvas.paste(_c17, (1319, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1319, 0, 1374, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/18_icon_SATURDAYZ_MASADA_Afrobeats.png
try:
    _c18 = get_crop(18, 1344, 1194)
    canvas.paste(_c18, (48, 676), _c18)
except Exception:
    pass
layout["SATURDAYZ_@_MASADA:_Afrob"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/19_icon_7.10.png
try:
    _c19 = get_crop(19, 60, 66)
    canvas.paste(_c19, (115, 0), _c19)
except Exception:
    pass
layout["7.10"] = [115, 0, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/20_icon_Chicago.png
try:
    _c20 = get_crop(20, 417, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["Chicago"] = [0, 259, 417, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/21_icon_Fitness.png
try:
    _c21 = get_crop(21, 1344, 191)
    canvas.paste(_c21, (48, 72), _c21)
except Exception:
    pass
layout["Fitness"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 51, 62)
    canvas.paste(_c22, (384, 2), _c22)
except Exception:
    pass
layout["icon_22"] = [384, 2, 435, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/23_icon_Taproom.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (288, 2804), _c23)
except Exception:
    pass
layout["Taproom"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/24_icon_Taproom.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["Taproom"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/25_icon_BiG_RQVE.png
try:
    _c25 = get_crop(25, 1344, 898)
    canvas.paste(_c25, (48, 1918), _c25)
except Exception:
    pass
layout["BiG_RQVE"] = [48, 1918, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/26_icon_Tickets.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (864, 2804), _c26)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/27_icon_Taproom.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (864, 2804), _c27)
except Exception:
    pass
layout["Taproom"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/28_icon_Bia_Grove_Brewerv.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Bia_Grove_Brewerv"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/29_icon_10_00_PM_CDT.png
try:
    _c29 = get_crop(29, 1344, 1194)
    canvas.paste(_c29, (48, 676), _c29)
except Exception:
    pass
layout["10:00_PM_CDT"] = [48, 676, 1392, 1870]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/30_text_7.10.png
try:
    _c30 = get_crop(30, 89, 41)
    canvas.paste(_c30, (22, 17), _c30)
except Exception:
    pass
layout["7.10"] = [22, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/31_text_873_events.png
try:
    _c31 = get_crop(31, 359, 103)
    canvas.paste(_c31, (54, 410), _c31)
except Exception:
    pass
layout["873_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_09_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-11/32_clickable_More.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (1152, 2804), _c32)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
