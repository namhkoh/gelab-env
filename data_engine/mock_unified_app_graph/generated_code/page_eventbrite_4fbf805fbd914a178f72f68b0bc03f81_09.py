# page_id: page_eventbrite_4fbf805fbd914a178f72f68b0bc03f81_09
# screenshot: 2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11.png
# step_index: 9/10
# task: Open Eventbrite. Explore "Education" events. Apply filters for events happening tomorrow. From the list, select the third event and check out its description.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the Eventbrite-like mobile page
# Uses provided variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Colors
bg_offwhite = (250, 250, 251)
status_bar_bg = (228, 228, 230)
header_divider = (226, 226, 229)
muted_divider = (238, 238, 240)
card_border = (59, 86, 255)       # blue accent for selection / card outlines
card_outline_soft = (232, 233, 237)
section_bg = (255, 255, 255)
subtle_gray = (245, 246, 248)
shadow_color = (220, 220, 225)

# Fill overall background (dominant color)
draw.rectangle([(0, 0), (W, H)], fill=bg_offwhite)

# Status bar area (top ~56px)
status_h = 56
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_bg)
# thin divider below status bar
draw.line([(0, status_h), (W, status_h)], fill=header_divider, width=1)

# Header / toolbar background (white area under status bar)
header_h = 150
draw.rectangle([(0, status_h), (W, header_h)], fill=section_bg)
# subtle divider under toolbar
draw.line([(24, header_h), (W-24, header_h)], fill=header_divider, width=1)

# "Select date and time" container background (rounded card area)
sel_left = 36
sel_right = W - 36
sel_top = 220
sel_bottom = 520
draw.rounded_rectangle([(sel_left, sel_top), (sel_right, sel_bottom)],
                       radius=18, fill=section_bg, outline=card_outline_soft, width=2)
# light inner shadow under the selection container
draw.rectangle([(sel_left+2, sel_bottom), (sel_right-2, sel_bottom+8)], fill=shadow_color)

# Separator line below selection area
sep_y = sel_bottom + 24
draw.line([(24, sep_y), (W-24, sep_y)], fill=muted_divider, width=1)

# "About this event" section area - keep background clean, draw top padding and a divider bottom
about_top = sep_y + 28
about_bottom = 1200
# subtle background block for the section header area (slightly off-white to hint grouping)
draw.rectangle([(0, about_top), (W, about_top+72)], fill=bg_offwhite)
# divider lines to frame the about section
draw.line([(24, about_bottom), (W-24, about_bottom)], fill=muted_divider, width=1)

# Location section background (light grouping block)
loc_top = about_bottom + 24
loc_bottom = 1880
draw.rectangle([(24, loc_top), (W-24, loc_bottom)], fill=section_bg)
# subtle left border marker for the location area
draw.line([(24, loc_top+8), (24, loc_bottom-8)], fill=(243,243,247), width=6)
# divider under location content
draw.line([(24, loc_bottom), (W-24, loc_bottom)], fill=muted_divider, width=1)

# "Show map" hint area - draw a very light rounded capsule on the right side (background only)
show_map_w = 240
show_map_h = 64
show_map_x1 = W - 36 - show_map_w
show_map_y1 = loc_top + 40
show_map_x2 = show_map_x1 + show_map_w
show_map_y2 = show_map_y1 + show_map_h
draw.rounded_rectangle([(show_map_x1, show_map_y1), (show_map_x2, show_map_y2)],
                       radius=32, fill=subtle_gray, outline=None)

# Thin divider separating content and ticket/card area further down
mid_separator_y = 2040
draw.line([(24, mid_separator_y), (W-24, mid_separator_y)], fill=muted_divider, width=1)

# Reserved Seating card (rounded rectangle with blue outline)
seat_left = 36
seat_right = W - 36
seat_top = 2320
seat_bottom = 2620
# card shadow (soft)
draw.rectangle([(seat_left+4, seat_top+12), (seat_right-4, seat_bottom+16)], fill=shadow_color)
# main card
draw.rounded_rectangle([(seat_left, seat_top), (seat_right, seat_bottom)],
                       radius=20, fill=section_bg, outline=card_border, width=5)
# inner divider in the seating card (light)
inner_div_y = seat_top + 140
draw.line([(seat_left+26, inner_div_y), (seat_right-26, inner_div_y)], fill=muted_divider, width=1)

# Small "Free" chip background placeholder (left side inside the seating card)
chip_w = 120
chip_h = 56
chip_x1 = seat_left + 40
chip_y1 = seat_bottom - 80
draw.rounded_rectangle([(chip_x1, chip_y1), (chip_x1+chip_w, chip_y1+chip_h)],
                       radius=28, fill=subtle_gray, outline=None)

# Quantity control placeholders (right side inside seating card) - background shapes only, no icons/text
qty_box_size = 96
qty_x2 = seat_right - 40
qty_y1 = seat_top + 60
# minus box
draw.rounded_rectangle([(qty_x2-qty_box_size*2 - 12, qty_y1), (qty_x2-qty_box_size - 12, qty_y1+qty_box_size)],
                       radius=18, fill=subtle_gray)
# number box (transparent)
draw.rounded_rectangle([(qty_x2-qty_box_size - 6, qty_y1), (qty_x2-6, qty_y1+qty_box_size)],
                       radius=18, fill=section_bg, outline=card_outline_soft, width=2)

# Horizontal rule above the reserve button area
reserve_line_y = seat_bottom + 40
draw.line([(24, reserve_line_y), (W-24, reserve_line_y)], fill=muted_divider, width=1)

# Note: The final "Reserve a spot" button and all icons/text will be pasted on top separately.
# This code only establishes background colors, cards, and separators as required.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/00_icon_24.png
try:
    _c0 = get_crop(0, 450, 516)
    canvas.paste(_c0, (24, 527), _c0)
except Exception:
    pass
layout["24"] = [24, 527, 474, 1043]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/01_icon_25.png
try:
    _c1 = get_crop(1, 450, 516)
    canvas.paste(_c1, (474, 527), _c1)
except Exception:
    pass
layout["25"] = [474, 527, 924, 1043]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/02_icon_More.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (1116, 108), _c2)
except Exception:
    pass
layout["More"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/03_icon_Real_Estate.png
try:
    _c3 = get_crop(3, 234, 144)
    canvas.paste(_c3, (48, 1495), _c3)
except Exception:
    pass
layout["Real_Estate"] = [48, 1495, 282, 1639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/04_icon_Share.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1260, 108), _c4)
except Exception:
    pass
layout["Share"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/05_icon_Decrease.png
try:
    _c5 = get_crop(5, 99, 96)
    canvas.paste(_c5, (996, 2441), _c5)
except Exception:
    pass
layout["Decrease"] = [996, 2441, 1095, 2537]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/06_icon_Reserve_a_spot.png
try:
    _c6 = get_crop(6, 1296, 132)
    canvas.paste(_c6, (72, 2756), _c6)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/07_icon_Increase.png
try:
    _c7 = get_crop(7, 96, 96)
    canvas.paste(_c7, (1224, 2441), _c7)
except Exception:
    pass
layout["Increase"] = [1224, 2441, 1320, 2537]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 92, 102)
    canvas.paste(_c8, (1108, 2439), _c8)
except Exception:
    pass
layout["icon_8"] = [1108, 2439, 1200, 2541]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/09_icon_4.56.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (36, 108), _c9)
except Exception:
    pass
layout["4.56"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/10_icon_27.png
try:
    _c10 = get_crop(10, 450, 516)
    canvas.paste(_c10, (924, 527), _c10)
except Exception:
    pass
layout["27"] = [924, 527, 1374, 1043]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/11_icon_Community_URL_Provided_Upon_Registration.png
try:
    _c11 = get_crop(11, 99, 96)
    canvas.paste(_c11, (996, 2441), _c11)
except Exception:
    pass
layout["Community,_URL_Provided_U"] = [996, 2441, 1095, 2537]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 54, 55)
    canvas.paste(_c12, (314, 6), _c12)
except Exception:
    pass
layout["icon_12"] = [314, 6, 368, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 55, 56)
    canvas.paste(_c13, (183, 5), _c13)
except Exception:
    pass
layout["icon_13"] = [183, 5, 238, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 45, 52)
    canvas.paste(_c14, (252, 8), _c14)
except Exception:
    pass
layout["icon_14"] = [252, 8, 297, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/15_icon_Business_Professional.png
try:
    _c15 = get_crop(15, 234, 144)
    canvas.paste(_c15, (48, 1495), _c15)
except Exception:
    pass
layout["Business_&_Professional"] = [48, 1495, 282, 1639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 98, 59)
    canvas.paste(_c16, (1215, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [1215, 1, 1313, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 53, 58)
    canvas.paste(_c17, (1319, 3), _c17)
except Exception:
    pass
layout["icon_17"] = [1319, 3, 1372, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/18_icon_27.png
try:
    _c18 = get_crop(18, 450, 516)
    canvas.paste(_c18, (924, 527), _c18)
except Exception:
    pass
layout["27"] = [924, 527, 1374, 1043]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/19_icon_4.56.png
try:
    _c19 = get_crop(19, 56, 60)
    canvas.paste(_c19, (116, 3), _c19)
except Exception:
    pass
layout["4.56"] = [116, 3, 172, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/20_icon_Free.png
try:
    _c20 = get_crop(20, 75, 72)
    canvas.paste(_c20, (249, 2585), _c20)
except Exception:
    pass
layout["Free"] = [249, 2585, 324, 2657]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/21_icon_By_Invitation_Only.png
try:
    _c21 = get_crop(21, 99, 96)
    canvas.paste(_c21, (996, 2441), _c21)
except Exception:
    pass
layout["By_Invitation_Only"] = [996, 2441, 1095, 2537]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/22_icon_Show_map.png
try:
    _c22 = get_crop(22, 226, 144)
    canvas.paste(_c22, (1166, 1713), _c22)
except Exception:
    pass
layout["Show_map"] = [1166, 1713, 1392, 1857]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 47, 57)
    canvas.paste(_c23, (384, 5), _c23)
except Exception:
    pass
layout["icon_23"] = [384, 5, 431, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/24_text_4.56.png
try:
    _c24 = get_crop(24, 92, 43)
    canvas.paste(_c24, (22, 17), _c24)
except Exception:
    pass
layout["4.56"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/25_text_The_Path_to_Wealth_T_..png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (36, 108), _c25)
except Exception:
    pass
layout["The_Path_to_Wealth_T_."] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/26_text_Select_date_and_time.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (36, 108), _c26)
except Exception:
    pass
layout["Select_date_and_time"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/27_text_About_this_event.png
try:
    _c27 = get_crop(27, 453, 65)
    canvas.paste(_c27, (44, 1200), _c27)
except Exception:
    pass
layout["About_this_event"] = [44, 1200, 497, 1265]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/4fbf805fbd914a178f72f68b0bc03f81/step_09_2024_4_24_16_55_4fbf805fbd914a178f72f68b0bc03f81-11/28_text_Location.png
try:
    _c28 = get_crop(28, 246, 63)
    canvas.paste(_c28, (41, 1757), _c28)
except Exception:
    pass
layout["Location"] = [41, 1757, 287, 1820]
