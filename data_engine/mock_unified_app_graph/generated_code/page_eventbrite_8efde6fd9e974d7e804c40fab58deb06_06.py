# page_id: page_eventbrite_8efde6fd9e974d7e804c40fab58deb06_06
# screenshot: 2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8.png
# step_index: 6/8
# task: Open Eventbrite. Search for "Education". Filter only online events. Note how many events are available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
draw.rectangle((0, 0, 1440, 2960), fill="#ffffff")

# Status bar area (top ~96px)
draw.rectangle((0, 0, 1440, 96), fill="#bdbdbd")

# Header / toolbar area (white to contrast status bar)
draw.rectangle((0, 96, 1440, 280), fill="#ffffff")

# Subtle shadow under header
draw.rectangle((0, 276, 1440, 284), fill="#e9ecf0")

# Thin divider line under header
draw.line((48, 280, 1392, 280), fill="#e6e6e9", width=2)

# Main content background subtle band (very light)
draw.rectangle((0, 280, 1440, 2960), fill="#ffffff")

# First event card shadow and background (pos=(48,676), size=1344x1115)
card1_x, card1_y = 48, 676
card1_w, card1_h = 1344, 1115
shadow_offset = 12
# shadow
draw.rounded_rectangle(
    (card1_x, card1_y + shadow_offset, card1_x + card1_w, card1_y + card1_h + shadow_offset),
    radius=28, fill="#e9edf5"
)
# card background
draw.rounded_rectangle(
    (card1_x, card1_y, card1_x + card1_w, card1_y + card1_h),
    radius=28, fill="#ffffff", outline="#f1f3f6", width=1
)

# Small separator under first card area (to separate title block and next content)
sep_y1 = card1_y + card1_h + 24
draw.line((48, sep_y1, 1392, sep_y1), fill="#f0f1f4", width=1)

# Second event card shadow and background (pos=(48,1839), size=1344x977)
card2_x, card2_y = 48, 1839
card2_w, card2_h = 1344, 977
# shadow
draw.rounded_rectangle(
    (card2_x, card2_y + shadow_offset, card2_x + card2_w, card2_y + card2_h + shadow_offset),
    radius=28, fill="#e9edf5"
)
# card background
draw.rounded_rectangle(
    (card2_x, card2_y, card2_x + card2_w, card2_y + card2_h),
    radius=28, fill="#ffffff", outline="#f1f3f6", width=1
)

# Separator lines between major sections (light)
draw.line((48, 560, 1392, 560), fill="#f2f3f5", width=1)
draw.line((48, 1480, 1392, 1480), fill="#f2f3f5", width=1)

# Bottom navigation bar background and top border
bottom_nav_top = 2840
draw.rectangle((0, bottom_nav_top, 1440, 2960), fill="#ffffff")
draw.line((0, bottom_nav_top, 1440, bottom_nav_top), fill="#e6e6e9", width=2)

# Subtle left/right page margins guide (very faint)
draw.line((48, 96, 48, 2960), fill="#fbfbfc", width=1)
draw.line((1392, 96, 1392, 2960), fill="#fbfbfc", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 103)
    canvas.paste(_c1, (425, 410), _c1)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1036, 410), _c2)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/04_icon_Foo.png
try:
    _c4 = get_crop(4, 150, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1092, 2355), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2355, 1236, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2355), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2355, 1380, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/09_icon_Close_current_screen.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/10_icon_7.00.png
try:
    _c10 = get_crop(10, 123, 117)
    canvas.paste(_c10, (54, 112), _c10)
except Exception:
    pass
layout["7.00"] = [54, 112, 177, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/11_icon_Education.png
try:
    _c11 = get_crop(11, 68, 62)
    canvas.paste(_c11, (307, 1), _c11)
except Exception:
    pass
layout["Education"] = [307, 1, 375, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/12_icon_7.00.png
try:
    _c12 = get_crop(12, 60, 63)
    canvas.paste(_c12, (180, 1), _c12)
except Exception:
    pass
layout["7.00"] = [180, 1, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 102, 61)
    canvas.paste(_c13, (1207, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1207, 0, 1309, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 51, 61)
    canvas.paste(_c14, (249, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [249, 1, 300, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 63, 60)
    canvas.paste(_c15, (1318, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1318, 0, 1381, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/16_icon_7.00.png
try:
    _c16 = get_crop(16, 58, 65)
    canvas.paste(_c16, (116, 0), _c16)
except Exception:
    pass
layout["7.00"] = [116, 0, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/17_icon_Education.png
try:
    _c17 = get_crop(17, 1344, 191)
    canvas.paste(_c17, (48, 72), _c17)
except Exception:
    pass
layout["Education"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/18_icon_New_York.png
try:
    _c18 = get_crop(18, 434, 144)
    canvas.paste(_c18, (0, 259), _c18)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/19_icon_Education.png
try:
    _c19 = get_crop(19, 50, 61)
    canvas.paste(_c19, (384, 2), _c19)
except Exception:
    pass
layout["Education"] = [384, 2, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/20_icon_11.30AM_EDT.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (288, 2804), _c20)
except Exception:
    pass
layout["11.30AM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/21_icon_Mama_Mingle.png
try:
    _c21 = get_crop(21, 1344, 1115)
    canvas.paste(_c21, (48, 676), _c21)
except Exception:
    pass
layout["Mama_Mingle"] = [48, 676, 1392, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/22_icon_Day.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (864, 2804), _c22)
except Exception:
    pass
layout["Day"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/23_icon_1I_O0AM_EDT.png
try:
    _c23 = get_crop(23, 1344, 1115)
    canvas.paste(_c23, (48, 676), _c23)
except Exception:
    pass
layout["1I:O0AM_EDT"] = [48, 676, 1392, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/24_icon_Day.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["Day"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/25_icon_2-00_PM.png
try:
    _c25 = get_crop(25, 1344, 977)
    canvas.paste(_c25, (48, 1839), _c25)
except Exception:
    pass
layout["'2-00_PM"] = [48, 1839, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/26_icon_Day.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (1152, 2804), _c26)
except Exception:
    pass
layout["Day"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/27_icon_Mama_Mingle.png
try:
    _c27 = get_crop(27, 374, 77)
    canvas.paste(_c27, (90, 1466), _c27)
except Exception:
    pass
layout["Mama_Mingle"] = [90, 1466, 464, 1543]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/28_icon_Albee_Sauare.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Albee_Sauare"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/29_text_7.00.png
try:
    _c29 = get_crop(29, 89, 41)
    canvas.paste(_c29, (22, 17), _c29)
except Exception:
    pass
layout["7.00"] = [22, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_06_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-8/30_text_4_290_events.png
try:
    _c30 = get_crop(30, 359, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["4,290_events"] = [54, 410, 413, 513]
