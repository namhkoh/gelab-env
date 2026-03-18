# page_id: page_eventbrite_f1e087441f9e44d997c2a58b9c8b0258_01
# screenshot: 2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3.png
# step_index: 1/10
# task: Open Eventbrite. Find the 'Arts' category. Select events that are available for this weekend. From the results, open the first item and add it to favorite. Follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Fill overall background
draw.rectangle((0, 0, 1440, 2960), fill="#FBFBFC")

# Status bar (top)
status_h = 56
draw.rectangle((0, 0, 1440, status_h), fill="#CFCFCF")

# Subtle bottom divider under status
draw.line((0, status_h, 1440, status_h), fill="#E0E0E0", width=1)

# Header / toolbar area (behind search). Keep it neutral white with subtle shadow
header_top = status_h
header_bottom = 170
draw.rectangle((0, header_top, 1440, header_bottom), fill="#FFFFFF")
# subtle shadow under header
draw.line((0, header_bottom, 1440, header_bottom), fill="#E9E9EA", width=2)

# Soft shadow behind search area (don't draw the pill itself; this stays behind pasted element)
search_shadow_bbox = (180, 78, 1266, 172)  # slightly larger than detected crop
draw.rounded_rectangle(search_shadow_bbox, radius=48, fill="#F6F6F8")

# Section title area (space for "More events you'll love") - leave white but add accent bar left
section_top = 200
section_bottom = 280
draw.rectangle((0, section_top, 1440, section_bottom), fill="#FFFFFF")
# thin accent vertical bar to hint a section (doesn't duplicate text)
draw.rectangle((48, section_top + 18, 56, section_bottom - 18), fill="#5A2A6E")

# Prepare card positions and sizes (visual backgrounds behind event rows)
card_x1 = 48
card_x2 = 1392
cards = [
    (490, 396),
    (886, 396),
    (1282, 396),
    (1678, 396),
    (2074, 396),
    (2470, 346),
]

card_fill = "#FFFFFF"
card_shadow = "#F0F0F1"
divider = "#ECE9ED"

for top, h in cards:
    bottom = top + h
    # shadow
    shadow_bbox = (card_x1 + 0, top + 6, card_x2 + 0, bottom + 8)
    draw.rounded_rectangle(shadow_bbox, radius=12, fill=card_shadow)
    # card background
    card_bbox = (card_x1, top, card_x2, bottom)
    draw.rounded_rectangle(card_bbox, radius=12, fill=card_fill)
    # subtle divider line at card bottom (separates rows)
    draw.line((card_x1 + 12, bottom, card_x2 - 12, bottom), fill=divider, width=1)

# Additional subtle separators between groups (if any)
# Between header/section and first card
draw.line((card_x1 + 12, cards[0][0] - 20, card_x2 - 12, cards[0][0] - 20), fill="#F1EAF6", width=1)

# Bottom navigation bar background
nav_top = 2804
nav_bottom = 2960
draw.rectangle((0, nav_top, 1440, nav_bottom), fill="#FFFFFF")
# top border of nav
draw.line((0, nav_top, 1440, nav_top), fill="#E8E8EA", width=2)

# Slight elevated background for floating location pill area (shadow only, do not draw pill)
loc_shadow_bbox = (440, 2580, 990, 2690)
draw.rounded_rectangle(loc_shadow_bbox, radius=40, fill="#F2F4F6")

# Final faint overall vignette shadow at very bottom to ground content
draw.rectangle((0, 2920, 1440, 2960), fill="#FCFCFD")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/00_icon_ering_to_soothe_the_brokel.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1678), _c0)
except Exception:
    pass
layout["ering_to_soothe_the_broke"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/01_icon_NDIE.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 490), _c1)
except Exception:
    pass
layout["NDIE"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/02_icon_San_Francisco.png
try:
    _c2 = get_crop(2, 495, 117)
    canvas.paste(_c2, (473, 2651), _c2)
except Exception:
    pass
layout["San_Francisco"] = [473, 2651, 968, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/03_icon_Sat.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 886), _c3)
except Exception:
    pass
layout["Sat,"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/04_icon_Q_Search_events.png
try:
    _c4 = get_crop(4, 1179, 144)
    canvas.paste(_c4, (195, 93), _c4)
except Exception:
    pass
layout["Q_Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/05_icon_QUEEN.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 2074), _c5)
except Exception:
    pass
layout["QUEEN"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 139)
    canvas.paste(_c6, (1140, 747), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1140, 747, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/07_icon_3600.png
try:
    _c7 = get_crop(7, 288, 156)
    canvas.paste(_c7, (288, 2804), _c7)
except Exception:
    pass
layout["3600"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/08_icon_City.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 1539), _c8)
except Exception:
    pass
layout["City"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1284, 747), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 747, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/10_icon_4.32.png
try:
    _c10 = get_crop(10, 110, 103)
    canvas.paste(_c10, (37, 120), _c10)
except Exception:
    pass
layout["4.32"] = [37, 120, 147, 223]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/11_icon_City.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1284, 1539), _c11)
except Exception:
    pass
layout["City"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/12_icon_Favorite_button.png
try:
    _c12 = get_crop(12, 144, 123)
    canvas.paste(_c12, (1140, 1951), _c12)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1951, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/13_icon_Overflow_menu_button.png
try:
    _c13 = get_crop(13, 144, 123)
    canvas.paste(_c13, (1284, 1951), _c13)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1951, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/14_icon_Spring-Zing_Happy.png
try:
    _c14 = get_crop(14, 144, 139)
    canvas.paste(_c14, (1140, 2331), _c14)
except Exception:
    pass
layout["Spring-Zing_Happy"] = [1140, 2331, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/15_icon_Overflow_menu_button.png
try:
    _c15 = get_crop(15, 144, 139)
    canvas.paste(_c15, (1284, 1143), _c15)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/16_icon_4.32.png
try:
    _c16 = get_crop(16, 55, 60)
    canvas.paste(_c16, (183, 2), _c16)
except Exception:
    pass
layout["4.32"] = [183, 2, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/17_icon_City.png
try:
    _c17 = get_crop(17, 144, 139)
    canvas.paste(_c17, (1140, 1143), _c17)
except Exception:
    pass
layout["City"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/18_icon_Overflow_menu_button.png
try:
    _c18 = get_crop(18, 144, 139)
    canvas.paste(_c18, (1284, 2331), _c18)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2331, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 58, 57)
    canvas.paste(_c19, (313, 4), _c19)
except Exception:
    pass
layout["icon_19"] = [313, 4, 371, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/20_icon_PDO_Thread_Training.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 1282), _c20)
except Exception:
    pass
layout["PDO_Thread_Training_|"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 50, 59)
    canvas.paste(_c21, (248, 3), _c21)
except Exception:
    pass
layout["icon_21"] = [248, 3, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/22_icon_Register_Nowl.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (0, 2804), _c22)
except Exception:
    pass
layout["Register_Nowl"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 47, 53)
    canvas.paste(_c23, (1321, 7), _c23)
except Exception:
    pass
layout["icon_23"] = [1321, 7, 1368, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/24_icon_4.32.png
try:
    _c24 = get_crop(24, 58, 60)
    canvas.paste(_c24, (115, 2), _c24)
except Exception:
    pass
layout["4.32"] = [115, 2, 173, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/25_icon_9_00_PM_PDT.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 490), _c25)
except Exception:
    pass
layout["9:00_PM_PDT"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/26_icon_8_30_creator_followers.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 886), _c26)
except Exception:
    pass
layout["8_30_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 58, 58)
    canvas.paste(_c27, (1213, 4), _c27)
except Exception:
    pass
layout["icon_27"] = [1213, 4, 1271, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/28_icon_Free.png
try:
    _c28 = get_crop(28, 125, 73)
    canvas.paste(_c28, (248, 561), _c28)
except Exception:
    pass
layout["Free"] = [248, 561, 373, 634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/29_icon_Spring-Zing_Happy_Hour.png
try:
    _c29 = get_crop(29, 1344, 346)
    canvas.paste(_c29, (48, 2470), _c29)
except Exception:
    pass
layout["Spring-Zing_Happy_Hour"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 41, 55)
    canvas.paste(_c30, (1272, 6), _c30)
except Exception:
    pass
layout["icon_30"] = [1272, 6, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/31_icon_Queen_of_Indies_2024.png
try:
    _c31 = get_crop(31, 1344, 396)
    canvas.paste(_c31, (48, 2074), _c31)
except Exception:
    pass
layout["Queen_of_Indies_2024"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/32_icon_Sales_ended.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 1282), _c32)
except Exception:
    pass
layout["Sales_ended"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/33_icon_Grief_Medicine_A_Gathering_to_Soothe_the.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 1678), _c33)
except Exception:
    pass
layout["Grief_Medicine:_A_Gatheri"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/34_icon_Q_Search_events.png
try:
    _c34 = get_crop(34, 43, 55)
    canvas.paste(_c34, (385, 7), _c34)
except Exception:
    pass
layout["Q_Search_events"] = [385, 7, 428, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/35_icon_8_100_creator_followers.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 1678), _c35)
except Exception:
    pass
layout["8_100_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/36_icon_Tickets.png
try:
    _c36 = get_crop(36, 288, 156)
    canvas.paste(_c36, (864, 2804), _c36)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/37_icon_7_00_PM_PDT.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 2074), _c37)
except Exception:
    pass
layout["7:00_PM_PDT"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/38_icon_Processing_Grief_Self-Care_for_Loss.png
try:
    _c38 = get_crop(38, 1344, 396)
    canvas.paste(_c38, (48, 886), _c38)
except Exception:
    pass
layout["Processing_Grief:_Self-Ca"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/39_text_4.32.png
try:
    _c39 = get_crop(39, 89, 43)
    canvas.paste(_c39, (22, 17), _c39)
except Exception:
    pass
layout["4.32"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/40_text_More_events_you_II_love.png
try:
    _c40 = get_crop(40, 1344, 396)
    canvas.paste(_c40, (48, 490), _c40)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/41_text_Mon.png
try:
    _c41 = get_crop(41, 92, 43)
    canvas.paste(_c41, (393, 2525), _c41)
except Exception:
    pass
layout["Mon,"] = [393, 2525, 485, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/42_text_13_S_00_PM_PDT.png
try:
    _c42 = get_crop(42, 1344, 346)
    canvas.paste(_c42, (48, 2470), _c42)
except Exception:
    pass
layout["13_+_S:00_PM_PDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/43_text_Out_in_Tech_SF_Bay_Area.png
try:
    _c43 = get_crop(43, 1344, 346)
    canvas.paste(_c43, (48, 2470), _c43)
except Exception:
    pass
layout["Out_in_Tech_SF_Bay_Area"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/44_clickable_Favorites.png
try:
    _c44 = get_crop(44, 288, 156)
    canvas.paste(_c44, (576, 2804), _c44)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_01_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-3/45_clickable_More.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (1152, 2804), _c45)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
