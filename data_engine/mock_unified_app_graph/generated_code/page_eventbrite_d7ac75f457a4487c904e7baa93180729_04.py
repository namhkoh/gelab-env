# page_id: page_eventbrite_d7ac75f457a4487c904e7baa93180729_04
# screenshot: 2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6.png
# step_index: 4/11
# task: Open Eventbrite. Search for 'Cooking' classes. Filter to only show free events that occur in the weekend. Select the first event and proceed to checkout.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background with a very light off-white to match the screenshot's dominant tone
draw.rectangle((0, 0, 1440, 2960), fill="#fbfcfe")

# Status bar (top area)
status_h = 96
draw.rectangle((0, 0, 1440, status_h), fill="#bfbfc3")

# Header area (below status bar). Keep it light/white but add a subtle bottom divider.
header_y0 = status_h
header_y1 = 220
draw.rectangle((0, header_y0, 1440, header_y1), fill="#ffffff")
draw.line((48, header_y1, 1392, header_y1), fill="#e0e0e3", width=2)

# Subtle separator under filters row (approx where filter chips sit)
filters_sep_y = 460
draw.line((48, filters_sep_y, 1392, filters_sep_y), fill="#eceef1", width=1)

# Large hero/banner card background (rounded) -- dark blue base as in screenshot
hero_x0, hero_y0 = 48, 360
hero_x1, hero_y1 = 1392, 780
# shadow
draw.rounded_rectangle((hero_x0+8, hero_y0+10, hero_x1+8, hero_y1+10), radius=20, fill="#e6e9ef")
# main hero background (will be overlaid by actual image/content)
draw.rounded_rectangle((hero_x0, hero_y0, hero_x1, hero_y1), radius=20, fill="#0b274a")

# Thin divider between hero and next content
draw.line((48, hero_y1 + 32, 1392, hero_y1 + 32), fill="#f0f1f4", width=1)

# Event card (portrait-style) background lower on the page
card2_x0, card2_y0 = 48, 1160
card2_x1, card2_y1 = 1392, 1680
# shadow for the card
draw.rounded_rectangle((card2_x0+6, card2_y0+8, card2_x1+6, card2_y1+8), radius=20, fill="#e9e9ed")
# white card body (image area will be pasted over this)
draw.rounded_rectangle((card2_x0, card2_y0, card2_x1, card2_y1), radius=20, fill="#ffffff")

# Separator lines for list sections further down
draw.line((48, card2_y1 + 40, 1392, card2_y1 + 40), fill="#eceef1", width=1)
draw.line((48, card2_y1 + 120, 1392, card2_y1 + 120), fill="#f6f7f9", width=1)

# Bottom navigation bar area
nav_h = 140
nav_y0 = 2960 - nav_h
draw.rectangle((0, nav_y0, 1440, 2960), fill="#ffffff")
# top divider for nav
draw.line((0, nav_y0, 1440, nav_y0), fill="#e6e7ea", width=2)
# subtle nav shadow
draw.rectangle((0, nav_y0-6, 1440, nav_y0), fill="#fbfbfc")

# Small subtle page edge padding indicators (left/right) to mimic card spacing
left_edge_x = 24
right_edge_x = 1440 - 24
draw.line((left_edge_x, status_h, left_edge_x, 2800), fill="#fbfcfe", width=2)
draw.line((right_edge_x, status_h, right_edge_x, 2800), fill="#fbfcfe", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/05_icon_Overflow_menu_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 1192), _c5)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2336), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2336, 1236, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/07_icon_Foo.png
try:
    _c7 = get_crop(7, 149, 110)
    canvas.paste(_c7, (1282, 406), _c7)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1431, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2336), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2336, 1380, 2480]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/09_icon_Cooking.png
try:
    _c9 = get_crop(9, 1344, 191)
    canvas.paste(_c9, (48, 72), _c9)
except Exception:
    pass
layout["Cooking"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/10_icon_Close_current_screen.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 96), _c10)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/11_icon_4.38.png
try:
    _c11 = get_crop(11, 120, 110)
    canvas.paste(_c11, (57, 116), _c11)
except Exception:
    pass
layout["4.38"] = [57, 116, 177, 226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/12_icon_Cooking.png
try:
    _c12 = get_crop(12, 67, 63)
    canvas.paste(_c12, (308, 0), _c12)
except Exception:
    pass
layout["Cooking"] = [308, 0, 375, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/13_icon_4.38.png
try:
    _c13 = get_crop(13, 59, 63)
    canvas.paste(_c13, (182, 0), _c13)
except Exception:
    pass
layout["4.38"] = [182, 0, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 105, 61)
    canvas.paste(_c14, (1206, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1206, 0, 1311, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 52, 62)
    canvas.paste(_c15, (247, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [247, 1, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/16_icon_4.38.png
try:
    _c16 = get_crop(16, 59, 64)
    canvas.paste(_c16, (115, 0), _c16)
except Exception:
    pass
layout["4.38"] = [115, 0, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/17_icon_Wood_Fired_Master_Class.png
try:
    _c17 = get_crop(17, 1344, 996)
    canvas.paste(_c17, (48, 1820), _c17)
except Exception:
    pass
layout["Wood_Fired_Master_Class"] = [48, 1820, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 59, 61)
    canvas.paste(_c18, (1318, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1318, 0, 1377, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/19_icon_San_Francisco.png
try:
    _c19 = get_crop(19, 536, 144)
    canvas.paste(_c19, (0, 259), _c19)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 49, 60)
    canvas.paste(_c20, (384, 3), _c20)
except Exception:
    pass
layout["icon_20"] = [384, 3, 433, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/21_icon_The_Forno_Piombo_Garden.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["The_Forno_Piombo_Garden"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/22_icon_Separating_the_Al_hype_from_the_real_val.png
try:
    _c22 = get_crop(22, 1344, 1096)
    canvas.paste(_c22, (48, 676), _c22)
except Exception:
    pass
layout["Separating_the_Al_hype_fr"] = [48, 676, 1392, 1772]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/23_icon_4.38.png
try:
    _c23 = get_crop(23, 96, 63)
    canvas.paste(_c23, (10, 0), _c23)
except Exception:
    pass
layout["4.38"] = [10, 0, 106, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/24_icon_Wood_Fired_Master_Class.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["Wood_Fired_Master_Class"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/25_icon_Promoted.png
try:
    _c25 = get_crop(25, 248, 66)
    canvas.paste(_c25, (82, 1664), _c25)
except Exception:
    pass
layout["Promoted"] = [82, 1664, 330, 1730]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/26_text_376_events.png
try:
    _c26 = get_crop(26, 359, 103)
    canvas.paste(_c26, (54, 410), _c26)
except Exception:
    pass
layout["376_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/27_text_Paaer.png
try:
    _c27 = get_crop(27, 55, 16)
    canvas.paste(_c27, (84, 741), _c27)
except Exception:
    pass
layout["Paaer"] = [84, 741, 139, 757]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/28_text_Sat_Apr_27.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Sat,_Apr_27"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/29_text_4_00_PM_PDT.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (288, 2804), _c29)
except Exception:
    pass
layout["4:00_PM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/30_text_The_Forno_Piombo_Garden.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (288, 2804), _c30)
except Exception:
    pass
layout["The_Forno_Piombo_Garden"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/31_clickable_Tickets.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (864, 2804), _c31)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/d7ac75f457a4487c904e7baa93180729/step_04_2024_4_24_16_37_d7ac75f457a4487c904e7baa93180729-6/32_clickable_More.png
try:
    _c32 = get_crop(32, 288, 156)
    canvas.paste(_c32, (1152, 2804), _c32)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
