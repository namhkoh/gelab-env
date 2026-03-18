# page_id: page_eventbrite_b45cca13f24546f9824a1ca2aab19c63_04
# screenshot: 2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6.png
# step_index: 4/11
# task: Open Eventbrite. Search for "Art". Filter for events in New York. Select first recommended event. Save it to wishlist. What is the duration of the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
draw.rectangle([(0, 0), (1440, 2960)], fill="#fbfcfd")

# Status bar (top area)
status_h = 96
draw.rectangle([(0, 0), (1440, status_h)], fill="#9aa2a6")

# Header / toolbar area beneath status bar
header_top = status_h
header_bottom = 160
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#ffffff")
# subtle bottom divider under header
draw.line([(24, header_bottom), (1440-24, header_bottom)], fill="#e6e7ea", width=2)

# Large thin separator under filters/search area
# (approx where filter chips occupy; keep it subtle)
sep_y = 520
draw.line([(24, sep_y), (1440-24, sep_y)], fill="#f0f1f4", width=2)

# Event card 1 container (rounded rectangle behind image + details)
card1_left = 40
card1_right = 1440 - 40
card1_top = 656
card1_bottom = 1880
card_radius = 24
# subtle shadow: a faint grey strip below card
draw.rectangle([(card1_left+6, card1_bottom+6), (card1_right+6, card1_bottom+10)], fill="#eceff1")
draw.rounded_rectangle([(card1_left, card1_top), (card1_right, card1_bottom)], radius=card_radius, fill="#ffffff", outline="#eef0f3", width=1)

# Divider between first and second card
divider_y = card1_bottom + 20
draw.line([(card1_left+8, divider_y), (card1_right-8, divider_y)], fill="#eceff1", width=1)

# Event card 2 container (rounded rectangle behind image + details)
card2_left = card1_left
card2_right = card1_right
card2_top = divider_y + 24
card2_bottom = 2840
# shadow for second card
draw.rectangle([(card2_left+6, card2_bottom+6), (card2_right+6, card2_bottom+10)], fill="#eceff1")
draw.rounded_rectangle([(card2_left, card2_top), (card2_right, card2_bottom)], radius=card_radius, fill="#ffffff", outline="#eef0f3", width=1)

# Thin separators between image area and text areas inside cards
# (approx positions; keep subtle and not overlapping detected text/images)
# For card1: image sits roughly starting near y ~676 (detected). Draw a separator below image area margin.
card1_image_bottom = 676 + 1175  # detected image pos y + height
sep1 = card1_image_bottom + 10
draw.line([(card1_left+12, sep1), (card1_right-12, sep1)], fill="#f3f4f6", width=1)

# For card2: image detected at y=1899 with height 917
card2_image_bottom = 1899 + 917
sep2 = card2_image_bottom + 10
# Ensure sep2 is within card2 bounds
if sep2 < card2_bottom - 12:
    draw.line([(card2_left+12, sep2), (card2_right-12, sep2)], fill="#f3f4f6", width=1)

# Content area banner background (subtle tinted stripe behind filter chips area)
banner_top = 340
banner_bottom = 440
draw.rectangle([(24, banner_top), (1440-24, banner_bottom)], fill="#f1f8ff", outline=None)

# Bottom navigation bar background and top divider
nav_top = 2876
nav_bottom = 2960
draw.line([(0, nav_top), (1440, nav_top)], fill="#e7e8ea", width=1)
draw.rectangle([(0, nav_top), (1440, nav_bottom)], fill="#ffffff")

# Small subtle left edge rule to visually separate content margin
draw.line([(24, header_bottom+8), (24, nav_top-8)], fill="#fbfbfc", width=2)

# Final subtle global vignette-ish corners (very light) to match screenshot spacing
corner_radius = 16
# top-left corner rounding visual (light)
draw.rectangle([(0, 0), (24, 24)], fill="#fbfcfd")
draw.rectangle([(1440-24, 0), (1440, 24)], fill="#fbfcfd")
draw.rectangle([(0, 2960-24), (24, 2960)], fill="#fbfcfd")
draw.rectangle([(1440-24, 2960-24), (1440, 2960)], fill="#fbfcfd")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/04_icon_Foo.png
try:
    _c4 = get_crop(4, 150, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/06_icon_APRIL_24_2024.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2415), _c6)
except Exception:
    pass
layout["APRIL_24,2024"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 1192), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/08_icon_Foo.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1248, 96), _c8)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/09_icon_Introduction_to_the_Art_of_Living_Intuit.png
try:
    _c9 = get_crop(9, 1344, 1175)
    canvas.paste(_c9, (48, 676), _c9)
except Exception:
    pass
layout["Introduction_to_the_Art_o"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1236, 2415), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/11_icon_7.05.png
try:
    _c11 = get_crop(11, 125, 115)
    canvas.paste(_c11, (53, 113), _c11)
except Exception:
    pass
layout["7.05"] = [53, 113, 178, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/12_icon_Art.png
try:
    _c12 = get_crop(12, 70, 64)
    canvas.paste(_c12, (307, 0), _c12)
except Exception:
    pass
layout["Art"] = [307, 0, 377, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/13_icon_7.05.png
try:
    _c13 = get_crop(13, 63, 66)
    canvas.paste(_c13, (179, 0), _c13)
except Exception:
    pass
layout["7.05"] = [179, 0, 242, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/14_icon_Art.png
try:
    _c14 = get_crop(14, 56, 65)
    canvas.paste(_c14, (246, 0), _c14)
except Exception:
    pass
layout["Art"] = [246, 0, 302, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/15_icon_7.05.png
try:
    _c15 = get_crop(15, 62, 67)
    canvas.paste(_c15, (114, 0), _c15)
except Exception:
    pass
layout["7.05"] = [114, 0, 176, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 102, 61)
    canvas.paste(_c16, (1205, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1205, 0, 1307, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/17_icon_Promoted.png
try:
    _c17 = get_crop(17, 262, 68)
    canvas.paste(_c17, (68, 1742), _c17)
except Exception:
    pass
layout["Promoted"] = [68, 1742, 330, 1810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 64, 60)
    canvas.paste(_c18, (1318, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1318, 0, 1382, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/19_icon_Wild_Gnosis_the_Art_of_Attuned_Empathy_a.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (576, 2804), _c19)
except Exception:
    pass
layout["Wild_Gnosis,_the_Art_of_A"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/20_icon_Online.png
try:
    _c20 = get_crop(20, 377, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/21_icon_Wild_Gnosis_the_Art_of_Attuned_Empathy_a.png
try:
    _c21 = get_crop(21, 1344, 917)
    canvas.paste(_c21, (48, 1899), _c21)
except Exception:
    pass
layout["Wild_Gnosis,_the_Art_of_A"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/22_icon_6.00_PM_EDT.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["6.00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/23_icon_Wild_Gnosis_the_Art_of_Attuned_Empathy_a.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (1152, 2804), _c23)
except Exception:
    pass
layout["Wild_Gnosis,_the_Art_of_A"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/24_icon_Wild_Gnosis_the_Art_of_Attuned_Empathy_a.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (864, 2804), _c24)
except Exception:
    pass
layout["Wild_Gnosis,_the_Art_of_A"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 52, 61)
    canvas.paste(_c25, (383, 2), _c25)
except Exception:
    pass
layout["icon_25"] = [383, 2, 435, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/26_icon_Art.png
try:
    _c26 = get_crop(26, 181, 112)
    canvas.paste(_c26, (172, 114), _c26)
except Exception:
    pass
layout["Art"] = [172, 114, 353, 226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/27_icon_Wed_Apr_24.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (0, 2804), _c27)
except Exception:
    pass
layout["Wed,_Apr_24"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/28_text_7.05.png
try:
    _c28 = get_crop(28, 92, 41)
    canvas.paste(_c28, (22, 17), _c28)
except Exception:
    pass
layout["7.05"] = [22, 17, 114, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/29_text_10_000_events.png
try:
    _c29 = get_crop(29, 359, 103)
    canvas.paste(_c29, (54, 410), _c29)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/30_text_IntumuonProcess.png
try:
    _c30 = get_crop(30, 400, 103)
    canvas.paste(_c30, (425, 410), _c30)
except Exception:
    pass
layout["IntumuonProcess"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/31_text_Tumt_Oruvng.png
try:
    _c31 = get_crop(31, 252, 29)
    canvas.paste(_c31, (69, 740), _c31)
except Exception:
    pass
layout["Tumt_Oruvng"] = [69, 740, 321, 769]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/32_text_Online.png
try:
    _c32 = get_crop(32, 129, 45)
    canvas.paste(_c32, (91, 1687), _c32)
except Exception:
    pass
layout["Online"] = [91, 1687, 220, 1732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/33_text_AKASHIC_AWAKENING_MASTERCLASS.png
try:
    _c33 = get_crop(33, 1344, 917)
    canvas.paste(_c33, (48, 1899), _c33)
except Exception:
    pass
layout["AKASHIC_AWAKENING_MASTERC"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b45cca13f24546f9824a1ca2aab19c63/step_04_2024_4_23_19_3_b45cca13f24546f9824a1ca2aab19c63-6/34_clickable_Art.png
try:
    _c34 = get_crop(34, 1344, 191)
    canvas.paste(_c34, (48, 72), _c34)
except Exception:
    pass
layout["Art"] = [48, 72, 1392, 263]
