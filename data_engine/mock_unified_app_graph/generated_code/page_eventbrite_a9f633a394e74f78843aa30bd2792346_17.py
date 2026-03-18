# page_id: page_eventbrite_a9f633a394e74f78843aa30bd2792346_17
# screenshot: 2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19.png
# step_index: 17/18
# task: Open Eventbrite. Set the city to "Los Angeles". Look for Photography workshops happening next week. What is the price of the tickets for first non-promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background
draw.rectangle((0, 0, 1440, 2960), fill="#FBFBFD")

# Status bar (top)
status_h = 96
draw.rectangle((0, 0, 1440, status_h), fill="#D1D3D4")

# Header / toolbar area (below status bar)
header_y0 = status_h
header_y1 = 256
draw.rectangle((0, header_y0, 1440, header_y1), fill="#FFFFFF")

# Header bottom divider
draw.line((24, header_y1, 1440-24, header_y1), fill="#E6EAEE", width=2)

# Light horizontal separator under filter row area (approx)
draw.line((24, 500, 1440-24, 500), fill="#F0F2F5", width=1)

# First event card background (rounded rectangle behind main event image)
card1_x0, card1_y0 = 36, 660
card1_x1, card1_y1 = 1404, 1720
draw.rounded_rectangle((card1_x0, card1_y0, card1_x1, card1_y1),
                       radius=24, fill="#FFFFFF", outline="#E6E9EE", width=1)

# Subtle inner shadow strip at top edge of card1 to ground it
draw.line((card1_x0+6, card1_y0+6, card1_x1-6, card1_y0+6), fill="#F2F4F6", width=2)

# Separator between first card and the next section
sep_y = card1_y1 + 12
draw.line((24, sep_y, 1440-24, sep_y), fill="#EFEFF1", width=1)

# Second large banner/card background (rounded rectangle behind Aura banner)
card2_x0, card2_y0 = 36, 1745
card2_x1, card2_y1 = 1404, 2824
draw.rounded_rectangle((card2_x0, card2_y0, card2_x1, card2_y1),
                       radius=24, fill="#FFF6F8", outline="#F2E8EE", width=1)

# Soft horizontal divider within second card (visual subtle band)
band_y = card2_y0 + 120
draw.rectangle((card2_x0+24, band_y-10, card2_x1-24, band_y+10), fill="#FFF8FB")

# Small badge background behind "Just added" (only the colored pill, not text)
badge_x0, badge_y0 = 70, 2400
badge_x1, badge_y1 = 430, 2504
draw.rounded_rectangle((badge_x0, badge_y0, badge_x1, badge_y1),
                       radius=24, fill="#DCEEE4", outline=None)

# Thin divider above the bottom navigation
nav_top = 2804
draw.line((0, nav_top, 1440, nav_top), fill="#E6E6E6", width=1)
draw.rectangle((0, nav_top, 1440, 2960), fill="#FFFFFF")

# Slight shadow line to separate content from nav (soft)
draw.line((24, nav_top-8, 1440-24, nav_top-8), fill="#F6F7F8", width=1)

# Additional subtle separators between content groups
draw.line((24, 1160, 1440-24, 1160), fill="#F3F5F7", width=1)
draw.line((24, 2000, 1440-24, 2000), fill="#F3F5F7", width=1)

# Decorative left gutter vertical guide (very subtle)
draw.line((24, header_y1+12, 24, nav_top-12), fill="#FBFCFD", width=8)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (1111, 410), _c0)
except Exception:
    pass
layout["Music"] = [1111, 410, 1298, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/01_icon_Apr_28_-_May_04_2024.png
try:
    _c1 = get_crop(1, 661, 103)
    canvas.paste(_c1, (438, 410), _c1)
except Exception:
    pass
layout["Apr_28_-_May_04,_2024"] = [438, 410, 1099, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/02_icon_1_Filter.png
try:
    _c2 = get_crop(2, 372, 103)
    canvas.paste(_c2, (54, 410), _c2)
except Exception:
    pass
layout["1_Filter"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/03_icon_Favorite_button.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 1192), _c3)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/04_icon_Overflow_menu_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1236, 1192), _c4)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/05_icon_tc.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 2269), _c5)
except Exception:
    pass
layout["tc"] = [1236, 2269, 1380, 2413]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/06_icon_tc.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2269), _c6)
except Exception:
    pass
layout["tc"] = [1092, 2269, 1236, 2413]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/07_icon_Close_current_screen.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1248, 96), _c7)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/08_icon_Product_Photography_Workshop.png
try:
    _c8 = get_crop(8, 1344, 1029)
    canvas.paste(_c8, (48, 676), _c8)
except Exception:
    pass
layout["Product_Photography_Works"] = [48, 676, 1392, 1705]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/09_icon_4.52.png
try:
    _c9 = get_crop(9, 114, 110)
    canvas.paste(_c9, (60, 116), _c9)
except Exception:
    pass
layout["4.52"] = [60, 116, 174, 226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 48, 66)
    canvas.paste(_c10, (1154, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1154, 0, 1202, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/11_icon_4.52.png
try:
    _c11 = get_crop(11, 56, 63)
    canvas.paste(_c11, (182, 1), _c11)
except Exception:
    pass
layout["4.52"] = [182, 1, 238, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 63, 63)
    canvas.paste(_c12, (310, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [310, 0, 373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/13_icon_Photography.png
try:
    _c13 = get_crop(13, 1344, 191)
    canvas.paste(_c13, (48, 72), _c13)
except Exception:
    pass
layout["Photography"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/14_icon_4.52.png
try:
    _c14 = get_crop(14, 56, 64)
    canvas.paste(_c14, (116, 0), _c14)
except Exception:
    pass
layout["4.52"] = [116, 0, 172, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 98, 65)
    canvas.paste(_c15, (1213, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1213, 0, 1311, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/16_icon_Bu.png
try:
    _c16 = get_crop(16, 114, 111)
    canvas.paste(_c16, (1305, 406), _c16)
except Exception:
    pass
layout["Bu"] = [1305, 406, 1419, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 48, 62)
    canvas.paste(_c17, (249, 1), _c17)
except Exception:
    pass
layout["icon_17"] = [249, 1, 297, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 52, 64)
    canvas.paste(_c18, (1319, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1319, 0, 1371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/19_icon_4_._12.00_PM_PDT.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (288, 2804), _c19)
except Exception:
    pass
layout["4_._12.00_PM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/20_icon_Photography.png
try:
    _c20 = get_crop(20, 48, 62)
    canvas.paste(_c20, (384, 2), _c20)
except Exception:
    pass
layout["Photography"] = [384, 2, 432, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/21_icon_Just_addedl.png
try:
    _c21 = get_crop(21, 313, 123)
    canvas.paste(_c21, (96, 2417), _c21)
except Exception:
    pass
layout["Just_addedl"] = [96, 2417, 409, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/22_icon_Los_Angeles.png
try:
    _c22 = get_crop(22, 492, 144)
    canvas.paste(_c22, (0, 259), _c22)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/23_icon_4_._12.00_PM_PDT.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (576, 2804), _c23)
except Exception:
    pass
layout["4_._12.00_PM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/24_icon_4.52.png
try:
    _c24 = get_crop(24, 102, 63)
    canvas.paste(_c24, (11, 0), _c24)
except Exception:
    pass
layout["4.52"] = [11, 0, 113, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/25_icon_Hollywood.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (864, 2804), _c25)
except Exception:
    pass
layout["Hollywood"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/26_icon_Futurstc.png
try:
    _c26 = get_crop(26, 196, 199)
    canvas.paste(_c26, (1199, 1995), _c26)
except Exception:
    pass
layout["Futurstc"] = [1199, 1995, 1395, 2194]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/27_text_129_events.png
try:
    _c27 = get_crop(27, 372, 103)
    canvas.paste(_c27, (54, 410), _c27)
except Exception:
    pass
layout["129_events"] = [54, 410, 426, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/28_text_What_do_your_Aura_colors_say.png
try:
    _c28 = get_crop(28, 1344, 1063)
    canvas.paste(_c28, (48, 1753), _c28)
except Exception:
    pass
layout["What_do_your_Aura_colors_"] = [48, 1753, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/29_text_Passlonate.png
try:
    _c29 = get_crop(29, 138, 30)
    canvas.paste(_c29, (67, 2002), _c29)
except Exception:
    pass
layout["Passlonate"] = [67, 2002, 205, 2032]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/30_text_Adventurous.png
try:
    _c30 = get_crop(30, 167, 37)
    canvas.paste(_c30, (248, 2000), _c30)
except Exception:
    pass
layout["Adventurous"] = [248, 2000, 415, 2037]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/31_text_Optlmistlc.png
try:
    _c31 = get_crop(31, 134, 32)
    canvas.paste(_c31, (460, 2002), _c31)
except Exception:
    pass
layout["Optlmistlc"] = [460, 2002, 594, 2034]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/32_text_Balanced.png
try:
    _c32 = get_crop(32, 119, 30)
    canvas.paste(_c32, (659, 2002), _c32)
except Exception:
    pass
layout["Balanced"] = [659, 2002, 778, 2032]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/33_text_Peaceful.png
try:
    _c33 = get_crop(33, 115, 30)
    canvas.paste(_c33, (855, 2002), _c33)
except Exception:
    pass
layout["Peaceful"] = [855, 2002, 970, 2032]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/34_text_Intultive.png
try:
    _c34 = get_crop(34, 113, 32)
    canvas.paste(_c34, (1045, 2000), _c34)
except Exception:
    pass
layout["Intultive"] = [1045, 2000, 1158, 2032]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/35_text_Bookyour_appointment_at.png
try:
    _c35 = get_crop(35, 1344, 1063)
    canvas.paste(_c35, (48, 1753), _c35)
except Exception:
    pass
layout["Bookyour_appointment_at"] = [48, 1753, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/36_text_AURA.png
try:
    _c36 = get_crop(36, 164, 68)
    canvas.paste(_c36, (903, 2304), _c36)
except Exception:
    pass
layout["AURA"] = [903, 2304, 1067, 2372]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/37_text_Aura_Photography_Los_Angeles.png
try:
    _c37 = get_crop(37, 1344, 1063)
    canvas.paste(_c37, (48, 1753), _c37)
except Exception:
    pass
layout["Aura_Photography_Los_Ange"] = [48, 1753, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/38_text_Hollywood.png
try:
    _c38 = get_crop(38, 301, 72)
    canvas.paste(_c38, (956, 2551), _c38)
except Exception:
    pass
layout["Hollywood"] = [956, 2551, 1257, 2623]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/39_text_Aura.png
try:
    _c39 = get_crop(39, 129, 57)
    canvas.paste(_c39, (97, 2634), _c39)
except Exception:
    pass
layout["Aura"] = [97, 2634, 226, 2691]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/40_text_Sat.png
try:
    _c40 = get_crop(40, 90, 52)
    canvas.paste(_c40, (90, 2716), _c40)
except Exception:
    pass
layout["Sat,"] = [90, 2716, 180, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/41_text_4_._12.00_PM_PDT.png
try:
    _c41 = get_crop(41, 288, 156)
    canvas.paste(_c41, (288, 2804), _c41)
except Exception:
    pass
layout["4_._12.00_PM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/42_clickable_Home.png
try:
    _c42 = get_crop(42, 288, 156)
    canvas.paste(_c42, (0, 2804), _c42)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/a9f633a394e74f78843aa30bd2792346/step_17_2024_4_24_16_49_a9f633a394e74f78843aa30bd2792346-19/43_clickable_More.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (1152, 2804), _c43)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
