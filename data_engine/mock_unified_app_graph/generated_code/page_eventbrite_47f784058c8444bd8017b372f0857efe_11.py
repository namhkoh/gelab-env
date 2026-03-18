# page_id: page_eventbrite_47f784058c8444bd8017b372f0857efe_11
# screenshot: 2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13.png
# step_index: 11/11
# task: Open Eventbrite. Explore local events scheduled for this weekend. Select the first event from the 'Science' category. Read details of the event.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the mobile event page.
# Uses provided canvas (1440x2960) and draw, font_* variables.

# Background base (very light off-white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(250, 250, 253))

# Status bar area (top subtle gray)
status_h = 88
draw.rectangle([(0, 0), (1440, status_h)], fill=(200, 200, 200))

# Slight darker top line to mimic phone outline
draw.line([(0, status_h-1), (1440, status_h-1)], fill=(190, 190, 190), width=1)

# Header / toolbar area (nav bar below status)
header_top = status_h
header_bottom = 220
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))

# Subtle shadow under header
draw.rectangle([(0, header_bottom), (1440, header_bottom+2)], fill=(235, 235, 238))

# Content card background (large rounded rectangle for the event content)
card_left = 24
card_right = 1440 - 24
card_top = header_bottom + 18
card_bottom = 2760  # leave space above bottom sticky bar
card_radius = 24

# Soft drop shadow for the content card (slightly offset)
shadow_offset = 8
shadow_box = [card_left + shadow_offset, card_top + shadow_offset, card_right + shadow_offset, card_bottom + shadow_offset]
draw.rounded_rectangle(shadow_box, radius=card_radius+2, fill=(242, 242, 245))

# Main card (white)
draw.rounded_rectangle([card_left, card_top, card_right, card_bottom], radius=card_radius, fill=(255, 255, 255))

# Inner subtle border for the card
draw.rounded_rectangle([card_left+1, card_top+1, card_right-1, card_bottom-1], radius=card_radius-1, outline=(240, 240, 243), width=1)

# Section separators inside content (thin lines to break sections)
# 1) Divider under the About section (approx where "Read less" ends)
sep_y1 = 2268
draw.line([(card_left + 8, sep_y1), (card_right - 8, sep_y1)], fill=(238, 238, 241), width=2)

# 2) A lighter divider further down (above agenda)
sep_y2 = 2520
draw.line([(card_left + 8, sep_y2), (card_right - 8, sep_y2)], fill=(245, 245, 247), width=1)

# Light section header background hint (subtle pale band behind headers, avoid drawing any text)
band_h = 84
# put a pale band near the top of the card for the "About this event" header area
band_top = card_top + 8
band_bottom = band_top + band_h
draw.rectangle([(card_left + 8, band_top), (card_right - 8, band_bottom)], fill=(250, 250, 252))

# Small accent rounded card behind potential meta area (e.g., category area) but keep it subtle and not overlapping detected badge location
# Place it slightly right/down so it doesn't conflict with auto-pasted pill (which will be applied later).
accent_box = [card_left + 28, band_bottom + 12, card_left + 380, band_bottom + 72]
draw.rounded_rectangle(accent_box, radius=36, fill=(245, 246, 249))

# Thin horizontal divider below band
draw.line([(card_left + 8, band_bottom + 88), (card_right - 8, band_bottom + 88)], fill=(238, 238, 241), width=1)

# Bottom sticky bar background (leave space for the Get tickets button which will be pasted later)
bottom_bar_top = 2768
bottom_bar_bottom = 2960
draw.rectangle([(0, bottom_bar_top), (1440, bottom_bar_bottom)], fill=(248, 247, 250))

# Subtle border line above bottom bar
draw.line([(0, bottom_bar_top), (1440, bottom_bar_top)], fill=(232, 230, 235), width=2)

# Left "Free" item area background hint (subtle pill behind price text area)
price_area = [24, bottom_bar_top + 22, 420, bottom_bar_bottom - 22]
draw.rectangle(price_area, fill=(248, 247, 250))

# Right side inset for the "Get tickets" control area (background area for button, the button graphic will be pasted on top)
cta_area = [560, bottom_bar_top + 20, 1420, bottom_bar_bottom - 20]
# draw a faint rounded background for the CTA area (so the pasted button will sit on a matching surface)
draw.rounded_rectangle(cta_area, radius=12, fill=(248, 247, 250), outline=(235, 115, 58, 0))

# Small visual divider between content and bottom bar
draw.line([(card_left + 8, bottom_bar_top - 10), (card_right - 8, bottom_bar_top - 10)], fill=(242, 242, 245), width=1)

# Upper-left subtle page title underline (thin accent under nav title area)
title_uline_y = header_bottom - 12
draw.line([(card_left + 64, title_uline_y), (card_left + 420, title_uline_y)], fill=(245, 245, 247), width=4)

# Add faint vertical guides/margins for visual structure (do not overlap icons/text locations)
margin_x = card_left + 40
draw.line([(margin_x, card_top + 12), (margin_x, card_bottom - 12)], fill=(250, 250, 251), width=1)

# End of background/structure drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/00_icon_More.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1116, 108), _c0)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/01_icon_Get_tickets.png
try:
    _c1 = get_crop(1, 570, 144)
    canvas.paste(_c1, (822, 2768), _c1)
except Exception:
    pass
layout["Get_tickets"] = [822, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/02_icon_Share.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1260, 108), _c2)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/03_icon_Science_Technology_._Biotech.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (36, 108), _c3)
except Exception:
    pass
layout["Science_&_Technology_._Bi"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/04_icon_7.59.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (36, 108), _c4)
except Exception:
    pass
layout["7.59"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 44, 54)
    canvas.paste(_c5, (252, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [252, 6, 296, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 54, 56)
    canvas.paste(_c6, (314, 5), _c6)
except Exception:
    pass
layout["icon_6"] = [314, 5, 368, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/07_icon_7.59.png
try:
    _c7 = get_crop(7, 55, 56)
    canvas.paste(_c7, (116, 6), _c7)
except Exception:
    pass
layout["7.59"] = [116, 6, 171, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 53, 57)
    canvas.paste(_c8, (184, 4), _c8)
except Exception:
    pass
layout["icon_8"] = [184, 4, 237, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/09_icon_Area_Bioengineer.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (36, 108), _c9)
except Exception:
    pass
layout["Area_Bioengineer__"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 44, 57)
    canvas.paste(_c10, (1325, 4), _c10)
except Exception:
    pass
layout["icon_10"] = [1325, 4, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 96, 60)
    canvas.paste(_c11, (1216, 1), _c11)
except Exception:
    pass
layout["icon_11"] = [1216, 1, 1312, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/12_icon_Read_less.png
try:
    _c12 = get_crop(12, 206, 144)
    canvas.paste(_c12, (48, 2060), _c12)
except Exception:
    pass
layout["Read_less"] = [48, 2060, 254, 2204]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 43, 56)
    canvas.paste(_c13, (386, 6), _c13)
except Exception:
    pass
layout["icon_13"] = [386, 6, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/14_icon_7.59.png
try:
    _c14 = get_crop(14, 87, 58)
    canvas.paste(_c14, (18, 5), _c14)
except Exception:
    pass
layout["7.59"] = [18, 5, 105, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/15_text_About_this_event.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (36, 108), _c15)
except Exception:
    pass
layout["About_this_event"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/16_text_Join_us_for_an_in-depth_dive_into_the_bi.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1116, 108), _c16)
except Exception:
    pass
layout["Join_us_for_an_in-depth_d"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/17_text_Areal.png
try:
    _c17 = get_crop(17, 119, 50)
    canvas.paste(_c17, (742, 673), _c17)
except Exception:
    pass
layout["Areal"] = [742, 673, 861, 723]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/18_text_Welcome_to_the.png
try:
    _c18 = get_crop(18, 336, 52)
    canvas.paste(_c18, (44, 798), _c18)
except Exception:
    pass
layout["Welcome_to_the"] = [44, 798, 380, 850]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/19_text_Area_Bioengineering_Symposiuml.png
try:
    _c19 = get_crop(19, 717, 73)
    canvas.paste(_c19, (464, 793), _c19)
except Exception:
    pass
layout["Area_Bioengineering_Sympo"] = [464, 793, 1181, 866]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/20_text_Date_Saturday_April_27th_2024.png
try:
    _c20 = get_crop(20, 690, 68)
    canvas.paste(_c20, (43, 918), _c20)
except Exception:
    pass
layout["Date:_Saturday,_April_27t"] = [43, 918, 733, 986]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/21_text_Location_Hearst_Memorial_Mining_Building.png
try:
    _c21 = get_crop(21, 1182, 73)
    canvas.paste(_c21, (42, 1043), _c21)
except Exception:
    pass
layout["Location:_Hearst_Memorial"] = [42, 1043, 1224, 1116]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/22_text_Join_us_for_a_day_filled_with.png
try:
    _c22 = get_crop(22, 554, 63)
    canvas.paste(_c22, (38, 1174), _c22)
except Exception:
    pass
layout["Join_us_for_a_day_filled_"] = [38, 1174, 592, 1237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/23_text_Whether.png
try:
    _c23 = get_crop(23, 195, 55)
    canvas.paste(_c23, (43, 1301), _c23)
except Exception:
    pass
layout["Whether"] = [43, 1301, 238, 1356]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/24_text_re_a_student_researcher_or_industry_prof.png
try:
    _c24 = get_crop(24, 1037, 68)
    canvas.paste(_c24, (321, 1297), _c24)
except Exception:
    pass
layout["'re_a_student;_researcher"] = [321, 1297, 1358, 1365]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/25_text_intersection_of_biology_and_engineering_.png
try:
    _c25 = get_crop(25, 206, 144)
    canvas.paste(_c25, (48, 2060), _c25)
except Exception:
    pass
layout["intersection_of_biology_a"] = [48, 2060, 254, 2204]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/26_text_like-minded_individuals._Don_t_miss_out_.png
try:
    _c26 = get_crop(26, 206, 144)
    canvas.paste(_c26, (48, 2060), _c26)
except Exception:
    pass
layout["like-minded_individuals._"] = [48, 2060, 254, 2204]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/27_text_opportunity_to_expand_your_knowledge_and.png
try:
    _c27 = get_crop(27, 206, 144)
    canvas.paste(_c27, (48, 2060), _c27)
except Exception:
    pass
layout["opportunity_to_expand_you"] = [48, 2060, 254, 2204]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/28_text_Ifyou_wish_to.png
try:
    _c28 = get_crop(28, 283, 61)
    canvas.paste(_c28, (43, 1868), _c28)
except Exception:
    pass
layout["Ifyou_wish_to"] = [43, 1868, 326, 1929]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/29_text_additional_form..png
try:
    _c29 = get_crop(29, 206, 144)
    canvas.paste(_c29, (48, 2060), _c29)
except Exception:
    pass
layout["additional_form."] = [48, 2060, 254, 2204]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/30_text_Agenda.png
try:
    _c30 = get_crop(30, 227, 75)
    canvas.paste(_c30, (42, 2322), _c30)
except Exception:
    pass
layout["Agenda"] = [42, 2322, 269, 2397]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/47f784058c8444bd8017b372f0857efe/step_11_2024_4_23_19_57_47f784058c8444bd8017b372f0857efe-13/31_text_Free.png
try:
    _c31 = get_crop(31, 110, 55)
    canvas.paste(_c31, (89, 2816), _c31)
except Exception:
    pass
layout["Free"] = [89, 2816, 199, 2871]
