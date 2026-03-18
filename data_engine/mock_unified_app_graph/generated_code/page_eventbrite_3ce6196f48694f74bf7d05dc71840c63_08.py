# page_id: page_eventbrite_3ce6196f48694f74bf7d05dc71840c63_08
# screenshot: 2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10.png
# step_index: 8/9
# task: Open Eventbrite. Search for 'coding workshop'. Sort the results by date. Where is the location of the soonest event that is not promoted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill
draw.rectangle((0, 0, 1440, 2960), fill="#F7F8FA")

# Status bar (top area)
status_h = 96
draw.rectangle((0, 0, 1440, status_h), fill="#BDBDBD")

# Header / toolbar background (below status bar)
header_top = status_h
header_bottom = 259
draw.rectangle((0, header_top, 1440, header_bottom), fill="#FFFFFF")
# divider under header
draw.line((48, header_bottom, 1392, header_bottom), fill="#E3E6EA", width=2)

# Main content separators
# subtle horizontal rule below filter chips area (just under header divider)
draw.line((48, header_bottom + 150, 1392, header_bottom + 150), fill="#F0F1F4", width=1)

# Card A (first event) - shadow + rounded white card background
card_margin_x = 36
card_a_top = 300
card_a_bottom = 1200
card_radius = 20
shadow_offset = 8
# shadow
draw.rounded_rectangle(
    (card_margin_x + shadow_offset, card_a_top + shadow_offset,
     1440 - card_margin_x + shadow_offset, card_a_bottom + shadow_offset),
    radius=card_radius + 2, fill="#E6E7EA"
)
# white card
draw.rounded_rectangle(
    (card_margin_x, card_a_top, 1440 - card_margin_x, card_a_bottom),
    radius=card_radius, fill="#FFFFFF"
)
# light divider under Card A
draw.line((card_margin_x + 12, card_a_bottom - 12, 1440 - card_margin_x - 12, card_a_bottom - 12), fill="#F2F3F5", width=1)

# Card B (second event) - shadow + rounded white card background
card_b_top = 1200
card_b_bottom = 2000
# shadow
draw.rounded_rectangle(
    (card_margin_x + shadow_offset, card_b_top + shadow_offset,
     1440 - card_margin_x + shadow_offset, card_b_bottom + shadow_offset),
    radius=card_radius + 2, fill="#E6E7EA"
)
# white card
draw.rounded_rectangle(
    (card_margin_x, card_b_top, 1440 - card_margin_x, card_b_bottom),
    radius=card_radius, fill="#FFFFFF"
)
# light divider under Card B
draw.line((card_margin_x + 12, card_b_bottom - 12, 1440 - card_margin_x - 12, card_b_bottom - 12), fill="#F2F3F5", width=1)

# Card C (promoted / banner area) - shadow + rounded white card background
card_c_top = 2000
card_c_bottom = 2700
# shadow
draw.rounded_rectangle(
    (card_margin_x + shadow_offset, card_c_top + shadow_offset,
     1440 - card_margin_x + shadow_offset, card_c_bottom + shadow_offset),
    radius=card_radius + 2, fill="#E6E7EA"
)
# white card
draw.rounded_rectangle(
    (card_margin_x, card_c_top, 1440 - card_margin_x, card_c_bottom),
    radius=card_radius, fill="#FFFFFF"
)

# Subtle global separators between main sections (to match screenshot rhythm)
draw.line((48, 960, 1392, 960), fill="#F4F5F7", width=1)
draw.line((48, 1680, 1392, 1680), fill="#F4F5F7", width=1)

# Bottom navigation background and top border
nav_top = 2804
draw.rectangle((0, nav_top, 1440, 2960), fill="#FFFFFF")
draw.line((0, nav_top, 1440, nav_top), fill="#E6E7EA", width=2)

# Small left content vertical guide (not content, just subtle visual margin)
draw.line((48, header_bottom, 48, nav_top), fill="#FAFAFB", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/00_icon_Business.png
try:
    _c0 = get_crop(0, 251, 112)
    canvas.paste(_c0, (1042, 405), _c0)
except Exception:
    pass
layout["Business"] = [1042, 405, 1293, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 408, 114)
    canvas.paste(_c1, (431, 405), _c1)
except Exception:
    pass
layout["Anytime"] = [431, 405, 839, 519]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/02_icon_Music.png
try:
    _c2 = get_crop(2, 198, 111)
    canvas.paste(_c2, (843, 406), _c2)
except Exception:
    pass
layout["Music"] = [843, 406, 1041, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 536, 144)
    canvas.paste(_c3, (0, 259), _c3)
except Exception:
    pass
layout["1_Filter"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/04_icon_Business.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 585), _c4)
except Exception:
    pass
layout["Business"] = [1092, 585, 1236, 729]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/05_icon_Fo.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 585), _c5)
except Exception:
    pass
layout["Fo("] = [1236, 585, 1380, 729]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1827), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1827, 1236, 1971]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/07_icon_Fo.png
try:
    _c7 = get_crop(7, 138, 110)
    canvas.paste(_c7, (1296, 406), _c7)
except Exception:
    pass
layout["Fo("] = [1296, 406, 1434, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1827), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1827, 1380, 1971]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/09_icon_THE_TRUTH.png
try:
    _c9 = get_crop(9, 1344, 512)
    canvas.paste(_c9, (48, 2304), _c9)
except Exception:
    pass
layout["THE_TRUTH"] = [48, 2304, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/10_icon_7.25.png
try:
    _c10 = get_crop(10, 119, 113)
    canvas.paste(_c10, (57, 114), _c10)
except Exception:
    pass
layout["7.25"] = [57, 114, 176, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/11_icon_Close_current_screen.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 96), _c11)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 64, 61)
    canvas.paste(_c12, (308, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [308, 1, 372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/13_icon_7.25.png
try:
    _c13 = get_crop(13, 60, 62)
    canvas.paste(_c13, (181, 1), _c13)
except Exception:
    pass
layout["7.25"] = [181, 1, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/14_icon_Coding_Workshop.png
try:
    _c14 = get_crop(14, 1344, 191)
    canvas.paste(_c14, (48, 72), _c14)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 53, 62)
    canvas.paste(_c15, (247, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [247, 1, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/16_icon_PRNSA_Field_Institute_Gift_Certificates.png
try:
    _c16 = get_crop(16, 1344, 945)
    canvas.paste(_c16, (48, 1311), _c16)
except Exception:
    pass
layout["PRNSA_Field_Institute_Gif"] = [48, 1311, 1392, 2256]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/17_icon_7.25.png
try:
    _c17 = get_crop(17, 57, 63)
    canvas.paste(_c17, (116, 0), _c17)
except Exception:
    pass
layout["7.25"] = [116, 0, 173, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 95, 61)
    canvas.paste(_c18, (1209, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1209, 0, 1304, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 59, 62)
    canvas.paste(_c19, (1317, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1317, 0, 1376, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/20_icon_San_Francisco.png
try:
    _c20 = get_crop(20, 536, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/21_icon_Tickets.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (864, 2804), _c21)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/22_icon_Search_events.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 48, 60)
    canvas.paste(_c23, (383, 2), _c23)
except Exception:
    pass
layout["icon_23"] = [383, 2, 431, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/24_icon_More.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 41, 63)
    canvas.paste(_c25, (1273, 0), _c25)
except Exception:
    pass
layout["icon_25"] = [1273, 0, 1314, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/26_icon_Promoted.png
try:
    _c26 = get_crop(26, 243, 66)
    canvas.paste(_c26, (85, 1155), _c26)
except Exception:
    pass
layout["Promoted"] = [85, 1155, 328, 1221]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/27_icon_ABOUT_LIES.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (576, 2804), _c27)
except Exception:
    pass
layout["ABOUT_LIES"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/28_icon_Home.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/29_icon_Wolrktshop.png
try:
    _c29 = get_crop(29, 1344, 738)
    canvas.paste(_c29, (48, 525), _c29)
except Exception:
    pass
layout["Wolrktshop"] = [48, 525, 1392, 1263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/30_icon_ABOUT_LIES.png
try:
    _c30 = get_crop(30, 1344, 512)
    canvas.paste(_c30, (48, 2304), _c30)
except Exception:
    pass
layout["ABOUT_LIES"] = [48, 2304, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/31_text_7.25.png
try:
    _c31 = get_crop(31, 92, 43)
    canvas.paste(_c31, (22, 17), _c31)
except Exception:
    pass
layout["7.25"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/32_text_Ticket_sales_end_soon.png
try:
    _c32 = get_crop(32, 413, 51)
    canvas.paste(_c32, (127, 782), _c32)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [127, 782, 540, 833]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/33_text_FREE_Webinar_Unlock_Financial_Freedom.png
try:
    _c33 = get_crop(33, 1344, 738)
    canvas.paste(_c33, (48, 525), _c33)
except Exception:
    pass
layout["[FREE_Webinar]_Unlock_Fin"] = [48, 525, 1392, 1263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/34_text_Sat.png
try:
    _c34 = get_crop(34, 90, 53)
    canvas.paste(_c34, (90, 1032), _c34)
except Exception:
    pass
layout["Sat,"] = [90, 1032, 180, 1085]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/35_text_27.png
try:
    _c35 = get_crop(35, 64, 43)
    canvas.paste(_c35, (253, 1035), _c35)
except Exception:
    pass
layout["27"] = [253, 1035, 317, 1078]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/36_text_10.30AM_EDT.png
try:
    _c36 = get_crop(36, 277, 45)
    canvas.paste(_c36, (336, 1033), _c36)
except Exception:
    pass
layout["10.30AM_EDT"] = [336, 1033, 613, 1078]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/37_text_Online.png
try:
    _c37 = get_crop(37, 129, 45)
    canvas.paste(_c37, (91, 1100), _c37)
except Exception:
    pass
layout["Online"] = [91, 1100, 220, 1145]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/38_text_F_LULC_LICL.png
try:
    _c38 = get_crop(38, 177, 25)
    canvas.paste(_c38, (60, 2391), _c38)
except Exception:
    pass
layout["F'LULC_LICL"] = [60, 2391, 237, 2416]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/39_text_EXPONENTIAL_INTELLIGENCE.png
try:
    _c39 = get_crop(39, 1344, 512)
    canvas.paste(_c39, (48, 2304), _c39)
except Exception:
    pass
layout["EXPONENTIAL_INTELLIGENCE"] = [48, 2304, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_08_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-10/40_text_Digital_Mindfulness.png
try:
    _c40 = get_crop(40, 531, 57)
    canvas.paste(_c40, (837, 2419), _c40)
except Exception:
    pass
layout["&_Digital_Mindfulness"] = [837, 2419, 1368, 2476]
