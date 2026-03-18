# page_id: page_eventbrite_8efde6fd9e974d7e804c40fab58deb06_08
# screenshot: 2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10.png
# step_index: 8/8
# task: Open Eventbrite. Search for "Education". Filter only online events. Note how many events are available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI elements for Eventbrite "Education" search results
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Fill overall background (dominant color: white/off-white)
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# Status bar area at top (~50-96px)
STATUS_H = 96
draw.rectangle((0, 0, 1440, STATUS_H), fill="#d0d0d0")
# subtle glossy line under status bar
draw.line((0, STATUS_H, 1440, STATUS_H), fill="#c2c2c2", width=1)

# Header / toolbar area (search + page title background)
HEADER_TOP = STATUS_H
HEADER_BOTTOM = 184
draw.rectangle((0, HEADER_TOP, 1440, HEADER_BOTTOM), fill="#FFFFFF")
# bottom divider under header
draw.line((48, HEADER_BOTTOM, 1392, HEADER_BOTTOM), fill="#e6e7ea", width=2)

# Light search field background (behind pasted search icon/text)
SEARCH_Y = HEADER_TOP + 28
draw.rounded_rectangle((48, SEARCH_Y, 1392, SEARCH_Y + 72), radius=12, fill="#f6f8fb", outline=None)

# Filter / chips separator line (between header and results filter chips)
FILTER_DIV_Y = 300
draw.line((48, FILTER_DIV_Y, 1392, FILTER_DIV_Y), fill="#f0f1f4", width=1)

# Large section heading area background (keeps text legible over background)
# (leave actual text to be pasted)
HEADING_AREA_Y = FILTER_DIV_Y + 24
draw.rectangle((48, HEADING_AREA_Y, 1392, HEADING_AREA_Y + 64), fill="#FFFFFF")

# First event card: image background (rounded) and card body
CARD_X = 48
CARD_W = 1344

# First image placeholder area (subtle pale-blue background)
FIRST_IMAGE_Y = 520
FIRST_IMAGE_H = 360
draw.rounded_rectangle(
    (CARD_X, FIRST_IMAGE_Y, CARD_X + CARD_W, FIRST_IMAGE_Y + FIRST_IMAGE_H),
    radius=20,
    fill="#eef6ff",
    outline="#e0eefc"
)

# White card body under the first image
FIRST_BODY_Y = FIRST_IMAGE_Y + FIRST_IMAGE_H + 20
FIRST_BODY_H = 220
draw.rounded_rectangle(
    (CARD_X, FIRST_BODY_Y, CARD_X + CARD_W, FIRST_BODY_Y + FIRST_BODY_H),
    radius=12,
    fill="#FFFFFF",
    outline="#e9eaed"
)
# subtle top divider to separate image and body
draw.line((CARD_X + 12, FIRST_BODY_Y, CARD_X + CARD_W - 12, FIRST_BODY_Y), fill="#eeeeee", width=1)

# Separator line after first card
SEP1_Y = FIRST_BODY_Y + FIRST_BODY_H + 24
draw.line((48, SEP1_Y, 1392, SEP1_Y), fill="#efeff2", width=1)

# Second event card: image background (rounded) and card body
SECOND_IMAGE_Y = SEP1_Y + 40
SECOND_IMAGE_H = 340
draw.rounded_rectangle(
    (CARD_X, SECOND_IMAGE_Y, CARD_X + CARD_W, SECOND_IMAGE_Y + SECOND_IMAGE_H),
    radius=20,
    fill="#fff7f2",
    outline="#fde7da"
)

# White card body under the second image
SECOND_BODY_Y = SECOND_IMAGE_Y + SECOND_IMAGE_H + 20
SECOND_BODY_H = 220
draw.rounded_rectangle(
    (CARD_X, SECOND_BODY_Y, CARD_X + CARD_W, SECOND_BODY_Y + SECOND_BODY_H),
    radius=12,
    fill="#FFFFFF",
    outline="#e9eaed"
)
# subtle top divider to separate second image and body
draw.line((CARD_X + 12, SECOND_BODY_Y, CARD_X + CARD_W - 12, SECOND_BODY_Y), fill="#eeeeee", width=1)

# Separator lines between content sections
draw.line((48, SECOND_BODY_Y + SECOND_BODY_H + 28, 1392, SECOND_BODY_Y + SECOND_BODY_H + 28), fill="#efeff2", width=1)

# Bottom navigation bar background area (reserve space for icons pasted later)
NAV_TOP = 2804
draw.rectangle((0, NAV_TOP, 1440, 2960), fill="#FFFFFF")
# top border of nav bar
draw.line((0, NAV_TOP, 1440, NAV_TOP), fill="#dcdcdc", width=2)

# Extra subtle dividing guide lines for layout flow (do not draw any text/icons)
# vertical margins guides (very faint)
draw.line((48, 0, 48, 2960), fill="#f7f7f8", width=1)
draw.line((1392, 0, 1392, 2960), fill="#f7f7f8", width=1)

# Finished structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/05_icon_Foo.png
try:
    _c5 = get_crop(5, 150, 110)
    canvas.paste(_c5, (1282, 406), _c5)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/07_icon_Foo.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1248, 96), _c7)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/08_icon_7.00.png
try:
    _c8 = get_crop(8, 119, 113)
    canvas.paste(_c8, (58, 115), _c8)
except Exception:
    pass
layout["7.00"] = [58, 115, 177, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/09_icon_7.00.png
try:
    _c9 = get_crop(9, 61, 65)
    canvas.paste(_c9, (180, 0), _c9)
except Exception:
    pass
layout["7.00"] = [180, 0, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/10_icon_Education.png
try:
    _c10 = get_crop(10, 69, 64)
    canvas.paste(_c10, (307, 0), _c10)
except Exception:
    pass
layout["Education"] = [307, 0, 376, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 104, 61)
    canvas.paste(_c11, (1206, 0), _c11)
except Exception:
    pass
layout["icon_11"] = [1206, 0, 1310, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/12_icon_Online.png
try:
    _c12 = get_crop(12, 377, 144)
    canvas.paste(_c12, (0, 259), _c12)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 54, 64)
    canvas.paste(_c13, (246, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [246, 0, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/14_icon_5.00_PM_EDT.png
try:
    _c14 = get_crop(14, 288, 156)
    canvas.paste(_c14, (576, 2804), _c14)
except Exception:
    pass
layout["5.00_PM_EDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/15_icon_7.00.png
try:
    _c15 = get_crop(15, 60, 66)
    canvas.paste(_c15, (115, 0), _c15)
except Exception:
    pass
layout["7.00"] = [115, 0, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/16_icon_Education.png
try:
    _c16 = get_crop(16, 1344, 191)
    canvas.paste(_c16, (48, 72), _c16)
except Exception:
    pass
layout["Education"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 59, 61)
    canvas.paste(_c17, (1318, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1318, 0, 1377, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/18_icon_Favorite_button.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1092, 2336), _c18)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2336, 1236, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/19_icon_Free_Zoom_ABCs_of_Retiring_in_Portugal.png
try:
    _c19 = get_crop(19, 1344, 1096)
    canvas.paste(_c19, (48, 676), _c19)
except Exception:
    pass
layout["Free_Zoom:_ABCs_of_Retiri"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/20_icon_4WEEK_SERIES_OF_ONLINE_CHILDBIRTH_PREPAR.png
try:
    _c20 = get_crop(20, 1344, 996)
    canvas.paste(_c20, (48, 1820), _c20)
except Exception:
    pass
layout["4WEEK_SERIES_OF_ONLINE_CH"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/21_icon_Education.png
try:
    _c21 = get_crop(21, 50, 62)
    canvas.paste(_c21, (384, 2), _c21)
except Exception:
    pass
layout["Education"] = [384, 2, 434, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/22_icon_Overflow_menu_button.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1236, 2336), _c22)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2336, 1380, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/23_icon_EMPOWERED_BIRTH_CHOICES_CHILDBIRTH.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["EMPOWERED_BIRTH_CHOICES_C"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/24_icon_5.00_PM_EDT.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (288, 2804), _c24)
except Exception:
    pass
layout["5.00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/25_icon_EMPOWERED_BIRTH_CHOICES_CHILDBIRTH.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (1152, 2804), _c25)
except Exception:
    pass
layout["EMPOWERED_BIRTH_CHOICES_C"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/26_text_7.00.png
try:
    _c26 = get_crop(26, 89, 41)
    canvas.paste(_c26, (22, 17), _c26)
except Exception:
    pass
layout["7.00"] = [22, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/27_text_10_000_events.png
try:
    _c27 = get_crop(27, 359, 103)
    canvas.paste(_c27, (54, 410), _c27)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/28_text_Promoted.png
try:
    _c28 = get_crop(28, 193, 43)
    canvas.paste(_c28, (94, 1678), _c28)
except Exception:
    pass
layout["Promoted"] = [94, 1678, 287, 1721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/29_text_Empowered_Birth_Choices.png
try:
    _c29 = get_crop(29, 1344, 996)
    canvas.paste(_c29, (48, 1820), _c29)
except Exception:
    pass
layout["Empowered_Birth_Choices"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_08_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-10/30_clickable_Home.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (0, 2804), _c30)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
