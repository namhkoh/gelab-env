# page_id: page_eventbrite_1a166da440f24e2e9152f2c0e40eb7aa_02
# screenshot: 2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4.png
# step_index: 2/16
# task: Open Eventbrite. Check "Sports" category. Filter events happening next month. Add the first event to your wishlist.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
draw.rectangle((0, 0, 1440, 2960), fill="#f6f7f9")

# Status bar (top area)
status_h = 64
draw.rectangle((0, 0, 1440, status_h), fill="#bdbfc1")
# Thin bottom divider under status bar
draw.line((0, status_h, 1440, status_h), fill="#a9a9ac", width=1)

# Header / Search area
header_top = status_h
header_bottom = 168
draw.rectangle((0, header_top, 1440, header_bottom), fill="#ffffff")
# subtle shadow/divider under header
draw.line((48, header_bottom, 1392, header_bottom), fill="#e1e1e6", width=2)

# A faint horizontal rule under the chips/search filters area
filters_rule_y = 260
draw.line((48, filters_rule_y, 1392, filters_rule_y), fill="#ececf0", width=1)

# Content card parameters
left_margin = 48
right_margin = 1392
card_width = right_margin - left_margin
card_radius = 28
shadow_offset = 8

# Helper to draw a card with shadow and an image/banner area
def draw_card(top_y, height):
    bottom_y = top_y + height
    # shadow (subtle)
    draw.rounded_rectangle(
        (left_margin + shadow_offset, top_y + shadow_offset, right_margin + shadow_offset, bottom_y + shadow_offset),
        radius=card_radius,
        fill="#e9e9ee"
    )
    # card background
    draw.rounded_rectangle(
        (left_margin, top_y, right_margin, bottom_y),
        radius=card_radius,
        fill="#ffffff"
    )
    # image/banner area inside card (rounded corners at top only)
    img_top = top_y + 24
    img_bottom = img_top + int(height * 0.42)
    # draw a darker rectangular banner to represent the image background
    draw.rounded_rectangle(
        (left_margin + 24, img_top, right_margin - 24, img_bottom),
        radius=18,
        fill="#273a48"
    )
    # subtle highlight band across top of image (to mimic lighting)
    highlight_h = 18
    draw.rectangle(
        (left_margin + 24, img_top, right_margin - 24, img_top + highlight_h),
        fill="#2f4a5b"
    )
    # divider between image and content area
    div_y = img_bottom + 18
    draw.line((left_margin + 32, div_y, right_margin - 32, div_y), fill="#f0f0f3", width=1)
    return bottom_y

# Draw a sequence of cards spaced down the page
y = 220
card_h = 540
y = draw_card(y, card_h) + 48
y = draw_card(y, card_h) + 48
y = draw_card(y, card_h) + 48

# Large full-width section background near the lower content area (subtle tint)
section_top = y
section_bottom = section_top + 420
draw.rectangle((0, section_top, 1440, section_bottom), fill="#fafbfd")
# separator above this section
draw.line((48, section_top, 1392, section_top), fill="#e7e7ea", width=1)

# Bottom navigation bar background
nav_top = 2840
draw.rectangle((0, nav_top, 1440, 2960), fill="#ffffff")
# top divider for nav
draw.line((0, nav_top, 1440, nav_top), fill="#e0e0e3", width=2)

# Small top accent line under header and above first card cluster
draw.line((48, 200, 1392, 200), fill="#ecebf0", width=1)

# Final subtle vertical guides (margins) to match layout spacing (non-intrusive)
draw.line((left_margin, 0, left_margin, 2960), fill="#ffffff", width=1)
draw.line((right_margin, 0, right_margin, 2960), fill="#ffffff", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2415), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2415), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/08_icon_Foo.png
try:
    _c8 = get_crop(8, 146, 110)
    canvas.paste(_c8, (1283, 406), _c8)
except Exception:
    pass
layout["Foo"] = [1283, 406, 1429, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/09_icon_5.30.png
try:
    _c9 = get_crop(9, 128, 114)
    canvas.paste(_c9, (54, 115), _c9)
except Exception:
    pass
layout["5.30"] = [54, 115, 182, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/10_icon_5.30.png
try:
    _c10 = get_crop(10, 62, 65)
    canvas.paste(_c10, (179, 0), _c10)
except Exception:
    pass
layout["5.30"] = [179, 0, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/11_icon_Search_forae.png
try:
    _c11 = get_crop(11, 69, 63)
    canvas.paste(_c11, (307, 0), _c11)
except Exception:
    pass
layout["Search_forae"] = [307, 0, 376, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 54, 64)
    canvas.paste(_c12, (246, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [246, 0, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 64, 59)
    canvas.paste(_c13, (1315, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1315, 0, 1379, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 76, 60)
    canvas.paste(_c14, (1208, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1208, 0, 1284, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/15_icon_5.30.png
try:
    _c15 = get_crop(15, 60, 66)
    canvas.paste(_c15, (115, 0), _c15)
except Exception:
    pass
layout["5.30"] = [115, 0, 175, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/16_icon_Search_forae.png
try:
    _c16 = get_crop(16, 1344, 191)
    canvas.paste(_c16, (48, 72), _c16)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/17_icon_Tech_to_Franchise_Trailblazer_Mapping_Yo.png
try:
    _c17 = get_crop(17, 1344, 917)
    canvas.paste(_c17, (48, 1899), _c17)
except Exception:
    pass
layout["Tech_to_Franchise_Trailbl"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/18_icon_SSCG_Lunch_Learn_Looking_to_Launch.png
try:
    _c18 = get_crop(18, 1344, 1175)
    canvas.paste(_c18, (48, 676), _c18)
except Exception:
    pass
layout["SSCG_Lunch_&_Learn:_Looki"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/19_icon_Promoted.png
try:
    _c19 = get_crop(19, 248, 68)
    canvas.paste(_c19, (81, 1742), _c19)
except Exception:
    pass
layout["Promoted"] = [81, 1742, 329, 1810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/20_icon_Online.png
try:
    _c20 = get_crop(20, 377, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/21_icon_Path_to_Ownershin.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (288, 2804), _c21)
except Exception:
    pass
layout["Path_to_Ownershin"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/22_icon_Tech_to_Franchise_Trailblazer_Mapping_Yo.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (864, 2804), _c22)
except Exception:
    pass
layout["Tech_to_Franchise_Trailbl"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/23_icon_Free.png
try:
    _c23 = get_crop(23, 127, 78)
    canvas.paste(_c23, (91, 2592), _c23)
except Exception:
    pass
layout["Free"] = [91, 2592, 218, 2670]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/24_icon_SSCG_Lunch_Learn_Looking_to_Launch.png
try:
    _c24 = get_crop(24, 1344, 1175)
    canvas.paste(_c24, (48, 676), _c24)
except Exception:
    pass
layout["SSCG_Lunch_&_Learn:_Looki"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/25_icon_Search_forae.png
try:
    _c25 = get_crop(25, 52, 61)
    canvas.paste(_c25, (383, 2), _c25)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 435, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/26_icon_Path_to_Ownershin.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Path_to_Ownershin"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/27_icon_Tech_to_Franchise_Trailblazer_Mapping_Yo.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (1152, 2804), _c27)
except Exception:
    pass
layout["Tech_to_Franchise_Trailbl"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/28_icon_Tech_to_Franchise_Trailblazer_Mapping_Yo.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (576, 2804), _c28)
except Exception:
    pass
layout["Tech_to_Franchise_Trailbl"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 39, 60)
    canvas.paste(_c29, (1275, 0), _c29)
except Exception:
    pass
layout["icon_29"] = [1275, 0, 1314, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/30_text_5.30.png
try:
    _c30 = get_crop(30, 91, 45)
    canvas.paste(_c30, (20, 15), _c30)
except Exception:
    pass
layout["5.30"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/31_text_10_000_events.png
try:
    _c31 = get_crop(31, 359, 103)
    canvas.paste(_c31, (54, 410), _c31)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/32_text_Join.png
try:
    _c32 = get_crop(32, 87, 27)
    canvas.paste(_c32, (518, 740), _c32)
except Exception:
    pass
layout["Join"] = [518, 740, 605, 767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/33_text_Ws.png
try:
    _c33 = get_crop(33, 50, 27)
    canvas.paste(_c33, (615, 740), _c33)
except Exception:
    pass
layout["Ws"] = [615, 740, 665, 767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/34_text_oR.png
try:
    _c34 = get_crop(34, 73, 27)
    canvas.paste(_c34, (673, 740), _c34)
except Exception:
    pass
layout["oR"] = [673, 740, 746, 767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/35_text_OUR_SMALL_BUSINESS.png
try:
    _c35 = get_crop(35, 400, 103)
    canvas.paste(_c35, (425, 410), _c35)
except Exception:
    pass
layout["OUR_SMALL_BUSINESS"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/36_text_Sscu.png
try:
    _c36 = get_crop(36, 208, 79)
    canvas.paste(_c36, (107, 826), _c36)
except Exception:
    pass
layout["Sscu"] = [107, 826, 315, 905]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/37_text_2024.png
try:
    _c37 = get_crop(37, 198, 89)
    canvas.paste(_c37, (1162, 811), _c37)
except Exception:
    pass
layout["2024"] = [1162, 811, 1360, 900]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/38_text_Learm.png
try:
    _c38 = get_crop(38, 103, 28)
    canvas.paste(_c38, (386, 885), _c38)
except Exception:
    pass
layout["Learm"] = [386, 885, 489, 913]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/39_text_hom.png
try:
    _c39 = get_crop(39, 78, 24)
    canvas.paste(_c39, (497, 886), _c39)
except Exception:
    pass
layout["hom"] = [497, 886, 575, 910]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/40_text_Laurch.png
try:
    _c40 = get_crop(40, 131, 27)
    canvas.paste(_c40, (636, 886), _c40)
except Exception:
    pass
layout["Laurch"] = [636, 886, 767, 913]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/41_text_0_Scale.png
try:
    _c41 = get_crop(41, 135, 25)
    canvas.paste(_c41, (775, 885), _c41)
except Exception:
    pass
layout["0_Scale"] = [775, 885, 910, 910]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/42_text_Growthi.png
try:
    _c42 = get_crop(42, 152, 27)
    canvas.paste(_c42, (571, 934), _c42)
except Exception:
    pass
layout["Growthi"] = [571, 934, 723, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_02_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-4/43_text_Online.png
try:
    _c43 = get_crop(43, 129, 45)
    canvas.paste(_c43, (91, 1687), _c43)
except Exception:
    pass
layout["Online"] = [91, 1687, 220, 1732]
