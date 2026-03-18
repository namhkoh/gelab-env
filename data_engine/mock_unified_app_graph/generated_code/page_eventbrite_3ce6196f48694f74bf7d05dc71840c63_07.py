# page_id: page_eventbrite_3ce6196f48694f74bf7d05dc71840c63_07
# screenshot: 2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9.png
# step_index: 7/9
# task: Open Eventbrite. Search for 'coding workshop'. Sort the results by date. Where is the location of the soonest event that is not promoted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (match screenshot's dominant white)
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Top status bar area (dark gray strip)
status_height = 96
draw.rectangle((0, 0, 1440, status_height), fill=(189, 189, 189))

# Header area (search/title area) just below status bar
header_top = status_height
header_bottom = 240
draw.rectangle((0, header_top, 1440, header_bottom), fill=(255, 255, 255))

# Subtle divider under header
draw.line((32, header_bottom, 1408, header_bottom), fill=(220, 221, 224), width=2)

# Light separator line above filters area (space for filters at y~410)
filters_sep_y = 540
draw.line((24, filters_sep_y, 1416, filters_sep_y), fill=(237, 238, 241), width=2)

# Small background pill behind location selector area (do not draw icons/text)
# location detected at (0,259) size (536x144) -- draw gentle rounded rect behind it
loc_x1, loc_y1 = 24, 260
loc_x2, loc_y2 = 24 + 520, 260 + 132
draw.rounded_rectangle((loc_x1, loc_y1, loc_x2, loc_y2), radius=18, fill=(249, 250, 252), outline=(235,236,240))

# Main content card 1 background (rounded card with shadow)
c1_x1, c1_y1 = 48, 676
c1_x2, c1_y2 = c1_x1 + 1344, c1_y1 + 1091
# shadow
draw.rounded_rectangle((c1_x1 + 10, c1_y1 + 12, c1_x2 + 10, c1_y2 + 12), radius=28, fill=(235, 236, 240))
# card body
draw.rounded_rectangle((c1_x1, c1_y1, c1_x2, c1_y2), radius=24, fill=(255, 255, 255), outline=(224, 225, 228), width=1)

# Separator line between first card and next content
sep_y_after_c1 = c1_y2 + 36
draw.line((32, sep_y_after_c1, 1408, sep_y_after_c1), fill=(241, 242, 244), width=1)

# Main content card 2 background (rounded card with shadow)
c2_x1, c2_y1 = 48, 1815
c2_x2, c2_y2 = c2_x1 + 1344, c2_y1 + 1001
# shadow
draw.rounded_rectangle((c2_x1 + 10, c2_y1 + 12, c2_x2 + 10, c2_y2 + 12), radius=28, fill=(235, 236, 240))
# card body
draw.rounded_rectangle((c2_x1, c2_y1, c2_x2, c2_y2), radius=24, fill=(255, 255, 255), outline=(224, 225, 228), width=1)

# Thin separator lines for list spacing (between items)
draw.line((48, c2_y2 + 12, 1392, c2_y2 + 12), fill=(243, 244, 246), width=1)

# Bottom navigation bar background and top divider
nav_top = 2804
draw.line((0, nav_top, 1440, nav_top), fill=(224, 225, 228), width=2)
draw.rectangle((0, nav_top, 1440, 2960), fill=(255, 255, 255))

# Subtle shadow under bottom nav (a faint top glow)
draw.line((0, nav_top + 2, 1440, nav_top + 2), fill=(245, 245, 247), width=1)

# Additional subtle visual accents: faint vertical left margin guide and right margin guide
draw.line((24, header_bottom + 8, 24, nav_top - 8), fill=(250, 250, 251), width=1)
draw.line((1416, header_bottom + 8, 1416, nav_top - 8), fill=(250, 250, 251), width=1)

# Final light divider near top content count area (under filter chips, but not drawing chips themselves)
count_div_y = 512
draw.line((48, count_div_y, 1392, count_div_y), fill=(236, 237, 239), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (850, 410), _c0)
except Exception:
    pass
layout["Music"] = [850, 410, 1037, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["Anytime"] = [438, 410, 838, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/02_icon_Business.png
try:
    _c2 = get_crop(2, 241, 103)
    canvas.paste(_c2, (1049, 410), _c2)
except Exception:
    pass
layout["Business"] = [1049, 410, 1290, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/03_icon_1_Filter.png
try:
    _c3 = get_crop(3, 372, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/04_icon_Fo.png
try:
    _c4 = get_crop(4, 137, 111)
    canvas.paste(_c4, (1296, 406), _c4)
except Exception:
    pass
layout["Fo("] = [1296, 406, 1433, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2331), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2331, 1236, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 2331), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2331, 1380, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/07_icon_Close_current_screen.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1248, 96), _c7)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/08_icon_7.24.png
try:
    _c8 = get_crop(8, 127, 116)
    canvas.paste(_c8, (54, 113), _c8)
except Exception:
    pass
layout["7.24"] = [54, 113, 181, 229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 68, 63)
    canvas.paste(_c9, (307, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [307, 1, 375, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 102, 62)
    canvas.paste(_c10, (1208, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1208, 0, 1310, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/11_icon_7.24.png
try:
    _c11 = get_crop(11, 61, 64)
    canvas.paste(_c11, (181, 0), _c11)
except Exception:
    pass
layout["7.24"] = [181, 0, 242, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 51, 62)
    canvas.paste(_c12, (249, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [249, 1, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/13_icon_7.24.png
try:
    _c13 = get_crop(13, 62, 65)
    canvas.paste(_c13, (113, 0), _c13)
except Exception:
    pass
layout["7.24"] = [113, 0, 175, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/14_icon_REGISTERAOW.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1092, 1192), _c14)
except Exception:
    pass
layout["REGISTERAOW"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/15_icon_The_Linkedln_Mastery_Workshop_Unlock_the.png
try:
    _c15 = get_crop(15, 1344, 1091)
    canvas.paste(_c15, (48, 676), _c15)
except Exception:
    pass
layout["The_Linkedln_Mastery_Work"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 64, 61)
    canvas.paste(_c16, (1317, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1317, 0, 1381, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 50, 63)
    canvas.paste(_c17, (383, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [383, 1, 433, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/18_icon_Coding_Workshop.png
try:
    _c18 = get_crop(18, 1344, 191)
    canvas.paste(_c18, (48, 72), _c18)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/19_icon_7.24.png
try:
    _c19 = get_crop(19, 97, 63)
    canvas.paste(_c19, (9, 0), _c19)
except Exception:
    pass
layout["7.24"] = [9, 0, 106, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/20_icon_The_Linkedln_Mastery_Workshop_Unlock_the.png
try:
    _c20 = get_crop(20, 1344, 1091)
    canvas.paste(_c20, (48, 676), _c20)
except Exception:
    pass
layout["The_Linkedln_Mastery_Work"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/21_icon_San_Francisco.png
try:
    _c21 = get_crop(21, 536, 144)
    canvas.paste(_c21, (0, 259), _c21)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/22_icon_10.20_AAA_EDT.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["10.20_AAA_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/23_icon_Workshop.png
try:
    _c23 = get_crop(23, 1344, 1001)
    canvas.paste(_c23, (48, 1815), _c23)
except Exception:
    pass
layout["Workshop"] = [48, 1815, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/24_icon_with_Our_Short_Term_Rental_Workshop.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["with_Our_Short_Term_Renta"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/25_icon_REGISTERAOW.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (1236, 1192), _c25)
except Exception:
    pass
layout["REGISTERAOW"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/26_icon_FREE_Webinar_Unlock_Financial_Freedom.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (864, 2804), _c26)
except Exception:
    pass
layout["[FREE_Webinar]_Unlock_Fin"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/27_icon_Tickets.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (864, 2804), _c27)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/28_icon_Ticket_sales_end_soon.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (288, 2804), _c28)
except Exception:
    pass
layout["Ticket_sales_end_soon"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/29_icon_FREE_Webinar_Unlock_Financial_Freedom.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (1152, 2804), _c29)
except Exception:
    pass
layout["[FREE_Webinar]_Unlock_Fin"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/30_text_8_101_events.png
try:
    _c30 = get_crop(30, 372, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["8,101_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/31_text_XXX_Unlock_thc_Power_6_Your_Profile.png
try:
    _c31 = get_crop(31, 1344, 1091)
    canvas.paste(_c31, (48, 676), _c31)
except Exception:
    pass
layout["XXX_Unlock_thc_Power_%6_Y"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/32_text_Online.png
try:
    _c32 = get_crop(32, 129, 45)
    canvas.paste(_c32, (91, 1604), _c32)
except Exception:
    pass
layout["Online"] = [91, 1604, 220, 1649]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/33_text_Promoted.png
try:
    _c33 = get_crop(33, 193, 43)
    canvas.paste(_c33, (94, 1673), _c33)
except Exception:
    pass
layout["Promoted"] = [94, 1673, 287, 1716]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/34_text_Short-Term.png
try:
    _c34 = get_crop(34, 1344, 1001)
    canvas.paste(_c34, (48, 1815), _c34)
except Exception:
    pass
layout["Short-Term"] = [48, 1815, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/35_text_Rental.png
try:
    _c35 = get_crop(35, 1344, 1001)
    canvas.paste(_c35, (48, 1815), _c35)
except Exception:
    pass
layout["Rental"] = [48, 1815, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/36_text_with_Our_Short_Term_Rental_Workshop.png
try:
    _c36 = get_crop(36, 1344, 1001)
    canvas.paste(_c36, (48, 1815), _c36)
except Exception:
    pass
layout["with_Our_Short_Term_Renta"] = [48, 1815, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/37_text_Cat.png
try:
    _c37 = get_crop(37, 66, 30)
    canvas.paste(_c37, (95, 2779), _c37)
except Exception:
    pass
layout["Cat"] = [95, 2779, 161, 2809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/38_text_Anr97.png
try:
    _c38 = get_crop(38, 135, 30)
    canvas.paste(_c38, (183, 2779), _c38)
except Exception:
    pass
layout["Anr97"] = [183, 2779, 318, 2809]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/39_text_10.20_AAA_EDT.png
try:
    _c39 = get_crop(39, 288, 156)
    canvas.paste(_c39, (288, 2804), _c39)
except Exception:
    pass
layout["10.20_AAA_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_07_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-9/40_clickable_Home.png
try:
    _c40 = get_crop(40, 288, 156)
    canvas.paste(_c40, (0, 2804), _c40)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
