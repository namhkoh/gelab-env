# page_id: page_eventbrite_b2798d8b10cc4118ab8cf6648f8a4077_12
# screenshot: 2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14.png
# step_index: 12/12
# task: Open Eventbrite. Search Music event in New York. Select the first one. Record its location and time in Google Keep Notes. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top-level background
bg_color = (250, 250, 252)            # very light off-white / lavender tint
canvas.paste(bg_color, [0, 0, canvas.size[0], canvas.size[1]])

# Status bar (top ~50-96px)
status_h = 88
status_color = (233, 233, 235)        # subtle light gray
draw.rectangle([0, 0, canvas.size[0], status_h], fill=status_color)

# Header / toolbar area under status bar
header_y1 = status_h
header_y2 = 220
header_bg = (255, 255, 255)
draw.rectangle([0, header_y1, canvas.size[0], header_y2], fill=header_bg)

# Header bottom divider
divider_color = (240, 239, 244)       # very light lavender-gray
draw.line([(24, header_y2), (canvas.size[0]-24, header_y2)], fill=divider_color, width=2)

# Large rounded container for the date/time card row (cards themselves will be pasted on top)
cards_x1 = 36
cards_x2 = canvas.size[0] - 36
cards_y1 = 260
cards_y2 = 1060
card_bg = (255, 255, 255)
card_border = (230, 228, 237)
draw.rounded_rectangle([cards_x1, cards_y1, cards_x2, cards_y2],
                       radius=22, fill=card_bg, outline=card_border, width=3)

# Subtle horizontal separator under the cards area
sep_y = cards_y2 + 20
draw.line([(36, sep_y), (canvas.size[0]-36, sep_y)], fill=divider_color, width=2)

# "About this event" section background area (group container)
about_x1 = 36
about_x2 = canvas.size[0] - 36
about_y1 = sep_y + 36
about_y2 = 1880
draw.rectangle([about_x1, about_y1, about_x2, about_y2], fill=card_bg)

# Light divider lines within content to structure subsections
draw.line([(about_x1, about_y1 + 140), (about_x2, about_y1 + 140)], fill=divider_color, width=2)
draw.line([(about_x1, about_y1 + 420), (about_x2, about_y1 + 420)], fill=divider_color, width=2)

# Location area separator (thin)
loc_sep_y = about_y1 + 680
draw.line([(36, loc_sep_y), (canvas.size[0]-36, loc_sep_y)], fill=divider_color, width=2)

# Gallery / slider progress track (subtle segmented track above reservation card)
progress_y = 2220
track_x1 = 60
track_x2 = canvas.size[0] - 60
track_h = 12
track_color = (235, 234, 238)
selected_segment_color = (63, 16, 82)  # dark purple accent for active segment
# draw background track
draw.rounded_rectangle([track_x1, progress_y, track_x2, progress_y + track_h],
                       radius=8, fill=track_color)
# draw a darker active segment on the left to indicate selection
seg_w = int((track_x2 - track_x1) * 0.12)
draw.rounded_rectangle([track_x1, progress_y, track_x1 + seg_w, progress_y + track_h],
                       radius=8, fill=selected_segment_color)

# Reservation card area (rounded white card). The actual "Reserve a spot" button is pasted later; avoid drawing button itself.
res_x1 = 60
res_x2 = canvas.size[0] - 60
res_y1 = 2340
res_y2 = 2660
res_border = (226, 223, 232)
draw.rounded_rectangle([res_x1, res_y1, res_x2, res_y2],
                       radius=18, fill=card_bg, outline=res_border, width=3)

# Thin inner divider inside reservation card (to visually separate title/desc)
draw.line([(res_x1 + 28, res_y1 + 120), (res_x2 - 28, res_y1 + 120)], fill=divider_color, width=1)

# Small accent pill (left) above reservation card to mimic subtle UI marker (not an icon or text)
pill_x1 = res_x1 + 20
pill_x2 = pill_x1 + 88
pill_y1 = res_y1 - 22
pill_y2 = pill_y1 + 12
draw.rounded_rectangle([pill_x1, pill_y1, pill_x2, pill_y2], radius=6, fill=selected_segment_color)

# Final subtle footer divider above the bottom button area (leave actual button area for pasted element)
footer_div_y = res_y2 + 40
draw.line([(36, footer_div_y), (canvas.size[0]-36, footer_div_y)], fill=divider_color, width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/00_icon_24.png
try:
    _c0 = get_crop(0, 387, 516)
    canvas.paste(_c0, (24, 518), _c0)
except Exception:
    pass
layout["24"] = [24, 518, 411, 1034]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/01_icon_More.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1116, 108), _c1)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/02_icon_31.png
try:
    _c2 = get_crop(2, 387, 516)
    canvas.paste(_c2, (411, 518), _c2)
except Exception:
    pass
layout["31"] = [411, 518, 798, 1034]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/03_icon_Share.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1260, 108), _c3)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/04_icon_Music.png
try:
    _c4 = get_crop(4, 321, 100)
    canvas.paste(_c4, (42, 1280), _c4)
except Exception:
    pass
layout["Music"] = [42, 1280, 363, 1380]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/05_icon_Reserve_a.png
try:
    _c5 = get_crop(5, 1320, 135)
    canvas.paste(_c5, (60, 2768), _c5)
except Exception:
    pass
layout["Reserve_a"] = [60, 2768, 1380, 2903]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/06_icon_14.png
try:
    _c6 = get_crop(6, 255, 516)
    canvas.paste(_c6, (1185, 518), _c6)
except Exception:
    pass
layout["14"] = [1185, 518, 1440, 1034]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/07_icon_9.21.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (36, 108), _c7)
except Exception:
    pass
layout["9.21"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/08_icon_April.png
try:
    _c8 = get_crop(8, 387, 516)
    canvas.paste(_c8, (798, 518), _c8)
except Exception:
    pass
layout["April"] = [798, 518, 1185, 1034]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 48, 66)
    canvas.paste(_c9, (1154, 2), _c9)
except Exception:
    pass
layout["icon_9"] = [1154, 2, 1202, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/10_icon_Admission_is_free_brunch_open_bar_menu.png
try:
    _c10 = get_crop(10, 1320, 408)
    canvas.paste(_c10, (60, 2315), _c10)
except Exception:
    pass
layout["Admission_is_free;_brunch"] = [60, 2315, 1380, 2723]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 98, 62)
    canvas.paste(_c11, (1213, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [1213, 2, 1311, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/12_icon_Read_more.png
try:
    _c12 = get_crop(12, 249, 72)
    canvas.paste(_c12, (1077, 2489), _c12)
except Exception:
    pass
layout["Read_more"] = [1077, 2489, 1326, 2561]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 52, 59)
    canvas.paste(_c13, (315, 4), _c13)
except Exception:
    pass
layout["icon_13"] = [315, 4, 367, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 52, 58)
    canvas.paste(_c14, (184, 3), _c14)
except Exception:
    pass
layout["icon_14"] = [184, 3, 236, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 47, 58)
    canvas.paste(_c15, (1325, 4), _c15)
except Exception:
    pass
layout["icon_15"] = [1325, 4, 1372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 54, 59)
    canvas.paste(_c16, (247, 3), _c16)
except Exception:
    pass
layout["icon_16"] = [247, 3, 301, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/17_icon_9.21.png
try:
    _c17 = get_crop(17, 53, 60)
    canvas.paste(_c17, (115, 2), _c17)
except Exception:
    pass
layout["9.21"] = [115, 2, 168, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/18_icon_Show_map.png
try:
    _c18 = get_crop(18, 226, 144)
    canvas.paste(_c18, (1166, 1759), _c18)
except Exception:
    pass
layout["Show_map"] = [1166, 1759, 1392, 1903]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/19_icon_Sabor.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (36, 108), _c19)
except Exception:
    pass
layout["Sabor"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/20_text_9.21.png
try:
    _c20 = get_crop(20, 89, 43)
    canvas.paste(_c20, (20, 17), _c20)
except Exception:
    pass
layout["9.21"] = [20, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/21_text_Select_date_and_time.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (36, 108), _c21)
except Exception:
    pass
layout["Select_date_and_time"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/22_text_Sunda.png
try:
    _c22 = get_crop(22, 140, 50)
    canvas.paste(_c22, (1300, 594), _c22)
except Exception:
    pass
layout["Sunda}"] = [1300, 594, 1440, 644]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/23_text_April.png
try:
    _c23 = get_crop(23, 96, 50)
    canvas.paste(_c23, (1332, 680), _c23)
except Exception:
    pass
layout["April"] = [1332, 680, 1428, 730]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/24_text_12.00_PE.png
try:
    _c24 = get_crop(24, 152, 43)
    canvas.paste(_c24, (1287, 903), _c24)
except Exception:
    pass
layout["12.00_PE"] = [1287, 903, 1439, 946]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/25_text_About_this_event.png
try:
    _c25 = get_crop(25, 452, 56)
    canvas.paste(_c25, (46, 1196), _c25)
except Exception:
    pass
layout["About_this_event"] = [46, 1196, 498, 1252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/26_text_Enjoy_Brunch_with_a_special_treat_Live_p.png
try:
    _c26 = get_crop(26, 234, 144)
    canvas.paste(_c26, (48, 1541), _c26)
except Exception:
    pass
layout["Enjoy_Brunch_with_a_speci"] = [48, 1541, 282, 1685]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/27_text_Read_more.png
try:
    _c27 = get_crop(27, 234, 144)
    canvas.paste(_c27, (48, 1541), _c27)
except Exception:
    pass
layout["Read_more"] = [48, 1541, 282, 1685]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/28_text_Location.png
try:
    _c28 = get_crop(28, 246, 63)
    canvas.paste(_c28, (41, 1803), _c28)
except Exception:
    pass
layout["Location"] = [41, 1803, 287, 1866]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/29_text_Mamazul.png
try:
    _c29 = get_crop(29, 207, 50)
    canvas.paste(_c29, (141, 1931), _c29)
except Exception:
    pass
layout["Mamazul"] = [141, 1931, 348, 1981]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/30_text_Mamazul_1155_Broadway_New_York_NY_10001.png
try:
    _c30 = get_crop(30, 156, 12)
    canvas.paste(_c30, (558, 2255), _c30)
except Exception:
    pass
layout["Mamazul,_1155_Broadway;_N"] = [558, 2255, 714, 2267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/31_clickable_Go_to_slide_1.png
try:
    _c31 = get_crop(31, 156, 12)
    canvas.paste(_c31, (60, 2255), _c31)
except Exception:
    pass
layout["Go_to_slide_1"] = [60, 2255, 216, 2267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/32_clickable_Go_to_slide_2.png
try:
    _c32 = get_crop(32, 156, 12)
    canvas.paste(_c32, (225, 2255), _c32)
except Exception:
    pass
layout["Go_to_slide_2"] = [225, 2255, 381, 2267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/33_clickable_Go_to_slide_3.png
try:
    _c33 = get_crop(33, 156, 12)
    canvas.paste(_c33, (393, 2255), _c33)
except Exception:
    pass
layout["Go_to_slide_3"] = [393, 2255, 549, 2267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/34_clickable_Go_to_slide_5.png
try:
    _c34 = get_crop(34, 156, 12)
    canvas.paste(_c34, (726, 2255), _c34)
except Exception:
    pass
layout["Go_to_slide_5"] = [726, 2255, 882, 2267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/35_clickable_Go_to_slide_6.png
try:
    _c35 = get_crop(35, 156, 12)
    canvas.paste(_c35, (891, 2255), _c35)
except Exception:
    pass
layout["Go_to_slide_6"] = [891, 2255, 1047, 2267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/36_clickable_Go_to_slide_7.png
try:
    _c36 = get_crop(36, 156, 12)
    canvas.paste(_c36, (1059, 2255), _c36)
except Exception:
    pass
layout["Go_to_slide_7"] = [1059, 2255, 1215, 2267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/37_clickable_Go_to_slide_8.png
try:
    _c37 = get_crop(37, 156, 12)
    canvas.paste(_c37, (1224, 2255), _c37)
except Exception:
    pass
layout["Go_to_slide_8"] = [1224, 2255, 1380, 2267]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b2798d8b10cc4118ab8cf6648f8a4077/step_12_2024_3_20_17_17_b2798d8b10cc4118ab8cf6648f8a4077-14/38_clickable_Increase.png
try:
    _c38 = get_crop(38, 96, 96)
    canvas.paste(_c38, (1230, 2369), _c38)
except Exception:
    pass
layout["Increase"] = [1230, 2369, 1326, 2465]
