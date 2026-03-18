# page_id: page_eventbrite_5f8371b476c64aeeb04a6d8281b87f4d_03
# screenshot: 2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5.png
# step_index: 3/7
# task: Open Eventbrite. Search Science & Tech event. Select the first one that is not promoted. If it is free, add it to Favorites. If it is not free, record its price in Google Keep Notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (slightly warm white to match screenshot)
draw.rectangle((0, 0, 1440, 2960), fill="#FFFFFF")

# Status bar area (top ~56px)
status_h = 56
draw.rectangle((0, 0, 1440, status_h), fill="#CFCFCF")

# Header underline (accent blue bar under the search/title area)
# Positioned under the header region as a thin accent line
header_underline_y = 136
draw.rectangle((48, header_underline_y, 1392, header_underline_y + 4), fill="#2B6BFF")

# Section title divider (subtle line under "Events" area)
events_div_y = 340
draw.line((48, events_div_y, 1392, events_div_y), fill="#EFEFEF", width=1)

# Card-like backgrounds for each event entry (rounded rectangles)
card_left = 48
card_right = 1392
card_width = card_right - card_left
card_height = 396
card_radius = 12
card_outline = "#F2F2F4"
card_fill = "#FFFFFF"

card_tops = [390, 786, 1182, 1578, 1974]
for top in card_tops:
    bottom = top + card_height
    # main card background
    draw.rounded_rectangle(
        (card_left, top, card_right, bottom),
        radius=card_radius,
        fill=card_fill,
        outline=card_outline,
        width=1
    )
    # subtle separator line at bottom of card (inset to align with layout)
    draw.line((card_left + 8, bottom, card_right - 8, bottom), fill="#EAEAEA", width=1)

# Additional horizontal separators between stacked groups (safety)
for sep_y in [786, 1182, 1578, 1974, 2370]:
    draw.line((48, sep_y, 1392, sep_y), fill="#F0F0F0", width=1)

# Large content/banner background example (behind the first card image area)
# NOTE: This is a background block only (no icons/text). It aligns to the left image area.
thumb_w = 216  # approximate thumbnail width used by layout
thumb_padding_left = card_left + 8
for top in card_tops:
    thumb_rect = (thumb_padding_left, top + 12, thumb_padding_left + thumb_w, top + card_height - 12)
    draw.rectangle(thumb_rect, fill="#FAFAFA", outline="#EFEFEF")

# Bottom navigation bar area background and top divider
nav_top = 2804
draw.rectangle((0, nav_top, 1440, 2960), fill="#FFFFFF")
draw.line((0, nav_top, 1440, nav_top), fill="#E6E6E6", width=2)

# subtle bottom edge line
draw.line((0, 2958, 1440, 2958), fill="#EFEFEF", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/00_icon_Keeping.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1182), _c0)
except Exception:
    pass
layout["Keeping"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/01_icon_IN-PERSON_OR_Zoom.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1578), _c1)
except Exception:
    pass
layout["IN-PERSON_OR_Zoom"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/02_icon_8_327_creator_followers.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 786), _c2)
except Exception:
    pass
layout["8_327_creator_followers"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/03_icon_feuided_tetrad_heat.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 390), _c3)
except Exception:
    pass
layout["'feuided_tetrad_heat"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/04_icon_Online.png
try:
    _c4 = get_crop(4, 111, 49)
    canvas.paste(_c4, (391, 1420), _c4)
except Exception:
    pass
layout["Online"] = [391, 1420, 502, 1469]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/05_icon_JAY_SUITES.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1974), _c5)
except Exception:
    pass
layout["JAY_SUITES"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 52, 57)
    canvas.paste(_c6, (249, 5), _c6)
except Exception:
    pass
layout["icon_6"] = [249, 5, 301, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 52, 56)
    canvas.paste(_c7, (316, 6), _c7)
except Exception:
    pass
layout["icon_7"] = [316, 6, 368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/08_icon_9.37.png
try:
    _c8 = get_crop(8, 53, 60)
    canvas.paste(_c8, (183, 2), _c8)
except Exception:
    pass
layout["9.37"] = [183, 2, 236, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/09_icon_Online.png
try:
    _c9 = get_crop(9, 113, 51)
    canvas.paste(_c9, (389, 1024), _c9)
except Exception:
    pass
layout["Online"] = [389, 1024, 502, 1075]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/10_icon_Science_Tech.png
try:
    _c10 = get_crop(10, 1344, 191)
    canvas.paste(_c10, (48, 72), _c10)
except Exception:
    pass
layout["Science_&_Tech"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/11_icon_9.37.png
try:
    _c11 = get_crop(11, 56, 59)
    canvas.paste(_c11, (113, 3), _c11)
except Exception:
    pass
layout["9.37"] = [113, 3, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/12_icon_Online.png
try:
    _c12 = get_crop(12, 1344, 396)
    canvas.paste(_c12, (48, 786), _c12)
except Exception:
    pass
layout["Online"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/13_icon_9.37.png
try:
    _c13 = get_crop(13, 128, 112)
    canvas.paste(_c13, (50, 114), _c13)
except Exception:
    pass
layout["9.37"] = [50, 114, 178, 226]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/14_icon_Keeping_Science_Practical.png
try:
    _c14 = get_crop(14, 1344, 396)
    canvas.paste(_c14, (48, 1182), _c14)
except Exception:
    pass
layout["Keeping_Science_Practical"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/15_icon_The_Social_Hub.png
try:
    _c15 = get_crop(15, 242, 51)
    canvas.paste(_c15, (391, 627), _c15)
except Exception:
    pass
layout["The_Social_Hub"] = [391, 627, 633, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/16_icon_Cancel.png
try:
    _c16 = get_crop(16, 80, 61)
    canvas.paste(_c16, (1215, 1), _c16)
except Exception:
    pass
layout["Cancel"] = [1215, 1, 1295, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/17_icon_4th_Global_Conference_on_Information.png
try:
    _c17 = get_crop(17, 1344, 396)
    canvas.paste(_c17, (48, 1974), _c17)
except Exception:
    pass
layout["4th_Global_Conference_on_"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/18_icon_Tickets.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (864, 2804), _c18)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/19_icon_Cancel.png
try:
    _c19 = get_crop(19, 52, 62)
    canvas.paste(_c19, (1319, 1), _c19)
except Exception:
    pass
layout["Cancel"] = [1319, 1, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/20_icon_Cancel.png
try:
    _c20 = get_crop(20, 149, 144)
    canvas.paste(_c20, (1243, 97), _c20)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/21_icon_Professionals_The_Social_Hub.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 390), _c21)
except Exception:
    pass
layout["Professionals_@_The_Socia"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 46, 58)
    canvas.paste(_c22, (384, 4), _c22)
except Exception:
    pass
layout["icon_22"] = [384, 4, 430, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/23_icon_Cancel.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1099, 96), _c23)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/24_icon_Search_events.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (288, 2804), _c24)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/25_icon_Cancel.png
try:
    _c25 = get_crop(25, 41, 63)
    canvas.paste(_c25, (1272, 1), _c25)
except Exception:
    pass
layout["Cancel"] = [1272, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/26_icon_Home.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/27_icon_Favorites.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (576, 2804), _c27)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/28_icon_More.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (1152, 2804), _c28)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/29_icon_The_Bio-Inspired_Green_BIG_Science.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 1578), _c29)
except Exception:
    pass
layout["The_Bio-Inspired_Green_(B"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/30_icon_TechTalk_Mastering_Computer_Science.png
try:
    _c30 = get_crop(30, 1344, 396)
    canvas.paste(_c30, (48, 786), _c30)
except Exception:
    pass
layout["TechTalk:_Mastering_Compu"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/31_text_9.37.png
try:
    _c31 = get_crop(31, 89, 43)
    canvas.paste(_c31, (20, 17), _c31)
except Exception:
    pass
layout["9.37"] = [20, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/32_text_Events.png
try:
    _c32 = get_crop(32, 186, 56)
    canvas.paste(_c32, (46, 301), _c32)
except Exception:
    pass
layout["Events"] = [46, 301, 232, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/33_text_Thu.png
try:
    _c33 = get_crop(33, 80, 41)
    canvas.paste(_c33, (394, 1636), _c33)
except Exception:
    pass
layout["Thu,"] = [394, 1636, 474, 1677]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/34_text_18_-_Fri.png
try:
    _c34 = get_crop(34, 129, 41)
    canvas.paste(_c34, (540, 1636), _c34)
except Exception:
    pass
layout["18_-_Fri,"] = [540, 1636, 669, 1677]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/35_text_19.png
try:
    _c35 = get_crop(35, 55, 38)
    canvas.paste(_c35, (734, 1636), _c35)
except Exception:
    pass
layout["19"] = [734, 1636, 789, 1674]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/36_text_9_O0_AM_EDT.png
try:
    _c36 = get_crop(36, 221, 38)
    canvas.paste(_c36, (804, 1636), _c36)
except Exception:
    pass
layout["9:O0_AM_EDT"] = [804, 1636, 1025, 1674]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/37_text_CUNY_Advanced_Science_Research_Center.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 1578), _c37)
except Exception:
    pass
layout["CUNY_Advanced_Science_Res"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/38_text_8_21_creator_followers.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 1578), _c38)
except Exception:
    pass
layout["8_21_creator_followers"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5f8371b476c64aeeb04a6d8281b87f4d/step_03_2024_3_20_17_36_5f8371b476c64aeeb04a6d8281b87f4d-5/39_text_8_1501_creator_followers.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 1974), _c39)
except Exception:
    pass
layout["8_1501_creator_followers"] = [48, 1974, 1392, 2370]
