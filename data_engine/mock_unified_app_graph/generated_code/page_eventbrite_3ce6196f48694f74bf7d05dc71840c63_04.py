# page_id: page_eventbrite_3ce6196f48694f74bf7d05dc71840c63_04
# screenshot: 2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6.png
# step_index: 4/9
# task: Open Eventbrite. Search for 'coding workshop'. Sort the results by date. Where is the location of the soonest event that is not promoted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background/base fill
draw.rectangle([(0, 0), (1440, 2960)], fill="#ffffff")

# Status bar (top ~50px)
status_h = 50
draw.rectangle([(0, 0), (1440, status_h)], fill="#bdbdbd")

# Header / toolbar area under status bar
header_top = status_h
header_bottom = 160
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#ffffff")
# header bottom divider
draw.line([(36, header_bottom), (1404, header_bottom)], fill="#e0e0e0", width=2)

# Search/title emphasis area (no text drawn)
# subtle underline below title
draw.line([(36, header_bottom + 6), (1404, header_bottom + 6)], fill="#f0f0f0", width=1)

# Chips row background (light blue rounded band behind filter chips)
chips_top = 330
chips_bottom = 460
draw.rounded_rectangle([(24, chips_top), (1416, chips_bottom)], radius=36, fill="#f2fbff", outline=None)

# Thin separator under chips
draw.line([(36, chips_bottom + 6), (1404, chips_bottom + 6)], fill="#e6e6e6", width=1)

# First event card container background (rounded white card with subtle border)
card1_left = 36
card1_top = 640
card1_right = 1404
card1_bottom = 1780
draw.rounded_rectangle(
    [(card1_left, card1_top), (card1_right, card1_bottom)],
    radius=20,
    fill="#ffffff",
    outline="#e6e6e6",
    width=2
)

# Slight top shadow for first card (very subtle, thin)
draw.rectangle([(card1_left + 4, card1_bottom + 2), (card1_right - 4, card1_bottom + 6)], fill="#f5f5f7")

# Divider between first and second card area
divider_y = card1_bottom + 18
draw.line([(36, divider_y), (1404, divider_y)], fill="#ebeef0", width=1)

# Second event card container (surrounding the darker image/banner)
card2_left = 36
card2_top = 1790
card2_right = 1404
card2_bottom = 2640
draw.rounded_rectangle(
    [(card2_left, card2_top), (card2_right, card2_bottom)],
    radius=20,
    fill="#ffffff",
    outline="#e6e6e6",
    width=2
)

# Background banner behind the second event's image area (dark bluish strip that will sit under the pasted image)
# We draw it slightly inset so it acts as a backdrop but does not duplicate foreground content
banner_left = 54
banner_top = 1815
banner_right = 1386
banner_bottom = 2230
draw.rounded_rectangle(
    [(banner_left, banner_top), (banner_right, banner_bottom)],
    radius=14,
    fill="#12384a",
    outline=None
)

# Small label background (like the "Ticket sales end soon" pill) — draw as subtle rounded rectangle behind where it will appear
pill_left = 54
pill_top = 2260
pill_right = 320
pill_bottom = 2304
draw.rounded_rectangle([(pill_left, pill_top), (pill_right, pill_bottom)], radius=14, fill="#f3e9ff", outline=None)

# Content separators (thin lines) to structure list
# Under first card (above next content)
draw.line([(36, card1_bottom + 4), (1404, card1_bottom + 4)], fill="#f0f0f2", width=1)
# Under second card (above bottom nav)
draw.line([(36, card2_bottom + 4), (1404, card2_bottom + 4)], fill="#f0f0f2", width=1)

# Bottom navigation bar background (do not draw icons)
bottom_nav_top = 2804
bottom_nav_bottom = 2960
draw.rectangle([(0, bottom_nav_top), (1440, bottom_nav_bottom)], fill="#ffffff")
# top divider of nav
draw.line([(0, bottom_nav_top), (1440, bottom_nav_top)], fill="#e6e6e6", width=2)

# Accent highlight under active nav area (simulate small orange underline under the search/home area)
accent_x = 360
draw.line([(accent_x - 60, bottom_nav_top + 6), (accent_x + 60, bottom_nav_top + 6)], fill="#ff6b2d", width=4)

# Final subtle vignette/shadow at very bottom
draw.rectangle([(0, bottom_nav_bottom - 8), (1440, bottom_nav_bottom)], fill="#fafafa")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/04_icon_Foo.png
try:
    _c4 = get_crop(4, 150, 110)
    canvas.paste(_c4, (1283, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1283, 406, 1433, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2331), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2331, 1236, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2331), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2331, 1380, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/07_icon_The_Linkedln_Mastery_Workshop_Unlock_the.png
try:
    _c7 = get_crop(7, 1344, 1091)
    canvas.paste(_c7, (48, 676), _c7)
except Exception:
    pass
layout["The_Linkedln_Mastery_Work"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/08_icon_Foo.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1248, 96), _c8)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/09_icon_7.24.png
try:
    _c9 = get_crop(9, 126, 115)
    canvas.paste(_c9, (54, 114), _c9)
except Exception:
    pass
layout["7.24"] = [54, 114, 180, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 105, 61)
    canvas.paste(_c10, (1206, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1206, 0, 1311, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 68, 62)
    canvas.paste(_c11, (307, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [307, 1, 375, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/12_icon_7.24.png
try:
    _c12 = get_crop(12, 62, 63)
    canvas.paste(_c12, (180, 0), _c12)
except Exception:
    pass
layout["7.24"] = [180, 0, 242, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 52, 62)
    canvas.paste(_c13, (248, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [248, 1, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/14_icon_7.24.png
try:
    _c14 = get_crop(14, 62, 64)
    canvas.paste(_c14, (113, 0), _c14)
except Exception:
    pass
layout["7.24"] = [113, 0, 175, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/15_icon_REGISTERAOW.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1092, 1192), _c15)
except Exception:
    pass
layout["REGISTERAOW"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/16_icon_Coding_Workshop.png
try:
    _c16 = get_crop(16, 1344, 191)
    canvas.paste(_c16, (48, 72), _c16)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 65, 60)
    canvas.paste(_c17, (1317, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1317, 0, 1382, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 51, 62)
    canvas.paste(_c18, (383, 2), _c18)
except Exception:
    pass
layout["icon_18"] = [383, 2, 434, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/19_icon_7.24.png
try:
    _c19 = get_crop(19, 98, 63)
    canvas.paste(_c19, (8, 0), _c19)
except Exception:
    pass
layout["7.24"] = [8, 0, 106, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/20_icon_San_Francisco.png
try:
    _c20 = get_crop(20, 536, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/21_icon_The_Linkedln_Mastery_Workshop_Unlock_the.png
try:
    _c21 = get_crop(21, 1344, 1091)
    canvas.paste(_c21, (48, 676), _c21)
except Exception:
    pass
layout["The_Linkedln_Mastery_Work"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/22_icon_10.20_AAA_EDT.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["10.20_AAA_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/23_icon_FREE_Webinar_Unlock_Financial_Freedom.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (864, 2804), _c23)
except Exception:
    pass
layout["[FREE_Webinar]_Unlock_Fin"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/24_icon_with_Our_Short_Term_Rental_Workshop.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["with_Our_Short_Term_Renta"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/25_icon_Workshop.png
try:
    _c25 = get_crop(25, 1344, 1001)
    canvas.paste(_c25, (48, 1815), _c25)
except Exception:
    pass
layout["Workshop"] = [48, 1815, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/26_icon_REGISTERAOW.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (1236, 1192), _c26)
except Exception:
    pass
layout["REGISTERAOW"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/27_icon_Ticket_sales_end_soon.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (288, 2804), _c27)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/28_icon_FREE_Webinar_Unlock_Financial_Freedom.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (1152, 2804), _c28)
except Exception:
    pass
layout["[FREE_Webinar]_Unlock_Fin"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/29_text_8_468_events.png
try:
    _c29 = get_crop(29, 359, 103)
    canvas.paste(_c29, (54, 410), _c29)
except Exception:
    pass
layout["8,468_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/30_text_XXX_Unlock_thc_Power_6_Your_Profile.png
try:
    _c30 = get_crop(30, 1344, 1091)
    canvas.paste(_c30, (48, 676), _c30)
except Exception:
    pass
layout["XXX_Unlock_thc_Power_%6_Y"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/31_text_Online.png
try:
    _c31 = get_crop(31, 129, 45)
    canvas.paste(_c31, (91, 1604), _c31)
except Exception:
    pass
layout["Online"] = [91, 1604, 220, 1649]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/32_text_Promoted.png
try:
    _c32 = get_crop(32, 193, 43)
    canvas.paste(_c32, (94, 1673), _c32)
except Exception:
    pass
layout["Promoted"] = [94, 1673, 287, 1716]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/33_text_Short-Term.png
try:
    _c33 = get_crop(33, 1344, 1001)
    canvas.paste(_c33, (48, 1815), _c33)
except Exception:
    pass
layout["Short-Term"] = [48, 1815, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/34_text_Rental.png
try:
    _c34 = get_crop(34, 1344, 1001)
    canvas.paste(_c34, (48, 1815), _c34)
except Exception:
    pass
layout["Rental"] = [48, 1815, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/35_text_Cat.png
try:
    _c35 = get_crop(35, 66, 30)
    canvas.paste(_c35, (95, 2779), _c35)
except Exception:
    pass
layout["Cat"] = [95, 2779, 161, 2809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/36_text_Anr97.png
try:
    _c36 = get_crop(36, 135, 30)
    canvas.paste(_c36, (183, 2779), _c36)
except Exception:
    pass
layout["Anr97"] = [183, 2779, 318, 2809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/37_text_10.20_AAA_EDT.png
try:
    _c37 = get_crop(37, 288, 156)
    canvas.paste(_c37, (288, 2804), _c37)
except Exception:
    pass
layout["10.20_AAA_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_04_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-6/38_clickable_Home.png
try:
    _c38 = get_crop(38, 288, 156)
    canvas.paste(_c38, (0, 2804), _c38)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
