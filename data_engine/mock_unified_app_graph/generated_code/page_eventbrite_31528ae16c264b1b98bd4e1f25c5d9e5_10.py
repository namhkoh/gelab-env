# page_id: page_eventbrite_31528ae16c264b1b98bd4e1f25c5d9e5_10
# screenshot: 2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12.png
# step_index: 10/11
# task: Open Eventbrite. Search 'Fitness'. Filter for free events. Browse and select any 'Yoga' event. Note the location.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for Eventbrite-like mobile page
# Available variables:
# - canvas: PIL Image (1440x2960 RGB)
# - draw: PIL ImageDraw object
# - font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Overall page background (very light off-white)
bg_color = (248, 249, 250)   # subtle off-white
draw.rectangle((0, 0, W, H), fill=bg_color)

# Status bar area at top (approx ~96px high)
status_h = 96
status_color = (189, 189, 189)  # muted gray bar
draw.rectangle((0, 0, W, status_h), fill=status_color)

# Header / toolbar area (search header)
header_top = status_h
header_bottom = 256  # sits above filter chips row
header_color = (255, 255, 255)
draw.rectangle((0, header_top, W, header_bottom), fill=header_color)

# Subtle bottom divider under header
divider_color = (226, 229, 233)
draw.line((48, header_bottom, W-48, header_bottom), fill=divider_color, width=2)

# Filter chips area background (keeps white but slightly distinct)
chips_top = header_bottom
chips_bottom = 420
draw.rectangle((0, chips_top, W, chips_bottom), fill=header_color)

# Light separator under chips
draw.line((48, chips_bottom, W-48, chips_bottom), fill=divider_color, width=1)

# Helper to draw a card with subtle shadow and rounded corners
def draw_card(x, y, w, h, radius=24, fill=(255,255,255), shadow_offset=(8,10)):
    # shadow
    sx = x + shadow_offset[0]
    sy = y + shadow_offset[1]
    draw.rounded_rectangle((sx, sy, sx + w, sy + h), radius=radius, fill=(238,240,242))
    # card
    draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=(235,237,240))

# First event card (behind image and text; images/icons will be pasted on top)
card_x = 48
card_w = 1344
# Place first card so that its image (detected at y=525) sits nicely within
card1_y = 440
card1_h = 720
draw_card(card_x, card1_y, card_w, card1_h, radius=20)

# Add a subtle internal separator between the image area and the textual area on the card
# (Positioned so it will be under the image; not drawing any text)
img_area_height = 300
sep_y = card1_y + img_area_height
draw.line((card_x + 24, sep_y, card_x + card_w - 24, sep_y), fill=(244,246,248), width=1)

# Second event card (lower on the page)
card2_y = 1480
card2_h = 720
draw_card(card_x, card2_y, card_w, card2_h, radius=20)

# Internal separator for second card
sep2_y = card2_y + img_area_height
draw.line((card_x + 24, sep2_y, card_x + card_w - 24, sep2_y), fill=(244,246,248), width=1)

# Additional faint horizontal separators to suggest item boundaries down the feed
for y in (card1_y + card1_h + 24, card2_y + card2_h + 24, 2400):
    if y < H - 200:
        draw.line((48, y, W-48, y), fill=(245,247,249), width=1)

# Bottom navigation bar background
nav_top = 2804
nav_color = (255, 255, 255)
draw.rectangle((0, nav_top, W, H), fill=nav_color)
# Divider above nav
draw.line((0, nav_top, W, nav_top), fill=divider_color, width=2)

# Small left & right gutters / safe areas (vertical faint lines)
gutter_color = (250, 251, 252)
draw.rectangle((0, 0, 48, H), fill=gutter_color)
draw.rectangle((W-48, 0, W, H), fill=gutter_color)

# Light page-wide vertical guide line (very faint) to echo content alignment (non-functional)
draw.line((48, header_bottom + 8, 48, nav_top - 8), fill=(250,250,251), width=1)

# End of structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/00_icon_Anytime.png
try:
    _c0 = get_crop(0, 1344, 191)
    canvas.paste(_c0, (48, 72), _c0)
except Exception:
    pass
layout["Anytime"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/01_icon_Music.png
try:
    _c1 = get_crop(1, 198, 110)
    canvas.paste(_c1, (843, 406), _c1)
except Exception:
    pass
layout["Music"] = [843, 406, 1041, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/02_icon_Business.png
try:
    _c2 = get_crop(2, 251, 111)
    canvas.paste(_c2, (1042, 405), _c2)
except Exception:
    pass
layout["Business"] = [1042, 405, 1293, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 536, 144)
    canvas.paste(_c3, (0, 259), _c3)
except Exception:
    pass
layout["1_Filter"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 961), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 961, 1236, 1105]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2117), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2117, 1236, 2261]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/06_icon_Fo.png
try:
    _c6 = get_crop(6, 139, 110)
    canvas.paste(_c6, (1296, 406), _c6)
except Exception:
    pass
layout["Fo("] = [1296, 406, 1435, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/07_icon_Nindasw_Flow.png
try:
    _c7 = get_crop(7, 1344, 1029)
    canvas.paste(_c7, (48, 1601), _c7)
except Exception:
    pass
layout["Nindasw_Flow"] = [48, 1601, 1392, 2630]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 961), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 961, 1380, 1105]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1236, 2117), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2117, 1380, 2261]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/10_icon_HIIT_Bodyweight_Pilates_Weekly_Fitness.png
try:
    _c10 = get_crop(10, 1344, 1028)
    canvas.paste(_c10, (48, 525), _c10)
except Exception:
    pass
layout["HIIT_Bodyweight_+_Pilates"] = [48, 525, 1392, 1553]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/11_icon_7.55.png
try:
    _c11 = get_crop(11, 118, 110)
    canvas.paste(_c11, (57, 115), _c11)
except Exception:
    pass
layout["7.55"] = [57, 115, 175, 225]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/12_icon_Close_current_screen.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 96), _c12)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/13_icon_Fitness.png
try:
    _c13 = get_crop(13, 66, 64)
    canvas.paste(_c13, (308, 0), _c13)
except Exception:
    pass
layout["Fitness"] = [308, 0, 374, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/14_icon_7.55.png
try:
    _c14 = get_crop(14, 60, 65)
    canvas.paste(_c14, (180, 0), _c14)
except Exception:
    pass
layout["7.55"] = [180, 0, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/15_icon_Fitness.png
try:
    _c15 = get_crop(15, 53, 65)
    canvas.paste(_c15, (247, 0), _c15)
except Exception:
    pass
layout["Fitness"] = [247, 0, 300, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/16_icon_7.55.png
try:
    _c16 = get_crop(16, 60, 66)
    canvas.paste(_c16, (114, 0), _c16)
except Exception:
    pass
layout["7.55"] = [114, 0, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 58, 61)
    canvas.paste(_c17, (1317, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1317, 0, 1375, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/18_icon_Tickets.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (864, 2804), _c18)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 79, 61)
    canvas.paste(_c19, (1208, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1208, 0, 1287, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/20_icon_Fitness.png
try:
    _c20 = get_crop(20, 1344, 191)
    canvas.paste(_c20, (48, 72), _c20)
except Exception:
    pass
layout["Fitness"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/21_icon_Free.png
try:
    _c21 = get_crop(21, 126, 77)
    canvas.paste(_c21, (90, 1138), _c21)
except Exception:
    pass
layout["Free"] = [90, 1138, 216, 1215]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/22_icon_Search_events.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 49, 62)
    canvas.paste(_c23, (384, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [384, 2, 433, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/24_icon_San_Francisco.png
try:
    _c24 = get_crop(24, 536, 144)
    canvas.paste(_c24, (0, 259), _c24)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/25_icon_More.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (1152, 2804), _c25)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/26_icon_icon_26.png
try:
    _c26 = get_crop(26, 42, 62)
    canvas.paste(_c26, (1273, 0), _c26)
except Exception:
    pass
layout["icon_26"] = [1273, 0, 1315, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/27_icon_Home.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (0, 2804), _c27)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/28_icon_Home.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/29_icon_HIIT_Bodyweight_Pilates_Weekly_Fitness.png
try:
    _c29 = get_crop(29, 1344, 1028)
    canvas.paste(_c29, (48, 525), _c29)
except Exception:
    pass
layout["HIIT_Bodyweight_+_Pilates"] = [48, 525, 1392, 1553]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/30_text_7.55.png
try:
    _c30 = get_crop(30, 92, 43)
    canvas.paste(_c30, (22, 17), _c30)
except Exception:
    pass
layout["7.55"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/31_text_Thrive.png
try:
    _c31 = get_crop(31, 121, 43)
    canvas.paste(_c31, (94, 1458), _c31)
except Exception:
    pass
layout["Thrive"] = [94, 1458, 215, 1501]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/32_text_lululemon_Broadway_Oakland_CA_USA.png
try:
    _c32 = get_crop(32, 1344, 1029)
    canvas.paste(_c32, (48, 1601), _c32)
except Exception:
    pass
layout["lululemon;_Broadway;_Oakl"] = [48, 1601, 1392, 2630]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/31528ae16c264b1b98bd4e1f25c5d9e5/step_10_2024_4_23_19_54_31528ae16c264b1b98bd4e1f25c5d9e5-12/33_clickable_Favorites.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (576, 2804), _c33)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]
