# page_id: page_eventbrite_bbe931c7fbf3446c80521f77d3e413aa_15
# screenshot: 2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17.png
# step_index: 15/20
# task: Open Eventbrite. Search free events in Los Angeles. Select the first one. Follow the organizer. Read more about the event. Add it to Favorites.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Status bar
status_bar_h = 96
status_color = "#E9E9EA"  # light grey status bar
draw.rectangle([(0, 0), (1440, status_bar_h)], fill=status_color)

# Header / toolbar background (under status bar)
header_h = 220
header_bg_color = "#FFFFFF"
draw.rectangle([(0, status_bar_h), (1440, header_h)], fill=header_bg_color)

# Subtle divider / shadow under header
divider_y = header_h + 2
draw.line([(24, divider_y), (1440-24, divider_y)], fill="#E6E6E9", width=2)

# Large content background (main white area - mostly the canvas is already white,
# but draw a very slightly off-white panel to hint separation)
content_panel_left = 24
content_panel_right = 1440 - 24
content_panel_top = header_h + 18
content_panel_bottom = 2360
draw.rectangle([(content_panel_left, content_panel_top), (content_panel_right, content_panel_bottom)], fill="#FFFFFF")

# Separator lines between major sections (thin greys)
sep_positions = [content_panel_top + 240, content_panel_top + 520, content_panel_top + 820, content_panel_top + 1120]
for y in sep_positions:
    draw.line([(content_panel_left, y), (content_panel_right, y)], fill="#F0F0F2", width=1)

# Ticket selection card (rounded rectangle with colored outline)
# NOTE: Do not draw any icons/text inside this card (they will be pasted later).
ticket_x1 = 48
ticket_x2 = 1440 - 48
ticket_y1 = 2380
ticket_y2 = 2560
ticket_radius = 20
ticket_border_color = "#3B4BD8"  # bluish outline seen around ticket card
ticket_border_width = 8

# Outer border (filled)
draw.rounded_rectangle([(ticket_x1, ticket_y1), (ticket_x2, ticket_y2)],
                       radius=ticket_radius, fill=ticket_border_color)

# Inner fill (slightly inset to leave border visible)
inset = ticket_border_width
draw.rounded_rectangle([(ticket_x1+inset, ticket_y1+inset), (ticket_x2-inset, ticket_y2-inset)],
                       radius=max(1, ticket_radius-inset), fill="#FFFFFF")

# Inner subtle divider inside ticket card (to separate title area from controls)
inner_div_y = ticket_y1 + 86
draw.line([(ticket_x1+24, inner_div_y), (ticket_x2-24, inner_div_y)], fill="#EEF0FF", width=2)

# Small hint of shadow under the ticket card
shadow_y_top = ticket_y2
for i, alpha in enumerate([220, 200, 180]):
    y = shadow_y_top + i
    shade = "#E9E9EF"
    draw.line([(ticket_x1+8, y), (ticket_x2-8, y)], fill=shade)

# Thin section separator above bottom area (helps visually separate content from controls)
bottom_sep_y = ticket_y2 + 68
draw.line([(24, bottom_sep_y), (1440-24, bottom_sep_y)], fill="#F2F2F4", width=1)

# (Reserve button area is detected and will be pasted automatically; do NOT draw it here.)
# Instead, draw a faint rounded container behind where the reserve button sits to provide structure,
# but keep it very subtle so it won't duplicate the detected button.
reserve_container_x1 = 48
reserve_container_x2 = 1440 - 48
reserve_container_y1 = 2720
reserve_container_y2 = 2920
draw.rounded_rectangle([(reserve_container_x1, reserve_container_y1), (reserve_container_x2, reserve_container_y2)],
                       radius=12, outline="#FFFFFF", width=1, fill="#FFFFFF")

# Overall left/right content margins (visual guides)
margin_x = 36
draw.line([(margin_x, content_panel_top), (margin_x, content_panel_bottom)], fill="#FFFFFF", width=1)
draw.line([(1440-margin_x, content_panel_top), (1440-margin_x, content_panel_bottom)], fill="#FFFFFF", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/00_icon_top.png
try:
    _c0 = get_crop(0, 144, 144)
    canvas.paste(_c0, (1116, 108), _c0)
except Exception:
    pass
layout["top"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/01_icon_top.png
try:
    _c1 = get_crop(1, 144, 144)
    canvas.paste(_c1, (1260, 108), _c1)
except Exception:
    pass
layout["top"] = [1260, 108, 1404, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/02_icon_Decrease.png
try:
    _c2 = get_crop(2, 99, 96)
    canvas.paste(_c2, (996, 2444), _c2)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/03_icon_9.13.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (36, 108), _c3)
except Exception:
    pass
layout["9.13"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/04_icon_Reserve_a_spot.png
try:
    _c4 = get_crop(4, 1296, 132)
    canvas.paste(_c4, (72, 2756), _c4)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/05_icon_Increase.png
try:
    _c5 = get_crop(5, 96, 96)
    canvas.paste(_c5, (1224, 2444), _c5)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 93, 104)
    canvas.paste(_c6, (1108, 2441), _c6)
except Exception:
    pass
layout["icon_6"] = [1108, 2441, 1201, 2545]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 46, 64)
    canvas.paste(_c7, (1155, 2), _c7)
except Exception:
    pass
layout["icon_7"] = [1155, 2, 1201, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 51, 57)
    canvas.paste(_c8, (316, 6), _c8)
except Exception:
    pass
layout["icon_8"] = [316, 6, 367, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 96, 61)
    canvas.paste(_c9, (1216, 2), _c9)
except Exception:
    pass
layout["icon_9"] = [1216, 2, 1312, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 52, 57)
    canvas.paste(_c10, (249, 5), _c10)
except Exception:
    pass
layout["icon_10"] = [249, 5, 301, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 47, 58)
    canvas.paste(_c11, (1325, 4), _c11)
except Exception:
    pass
layout["icon_11"] = [1325, 4, 1372, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 52, 56)
    canvas.paste(_c12, (183, 5), _c12)
except Exception:
    pass
layout["icon_12"] = [183, 5, 235, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/13_icon_9.13.png
try:
    _c13 = get_crop(13, 52, 58)
    canvas.paste(_c13, (117, 4), _c13)
except Exception:
    pass
layout["9.13"] = [117, 4, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/14_icon_Free.png
try:
    _c14 = get_crop(14, 141, 99)
    canvas.paste(_c14, (95, 2577), _c14)
except Exception:
    pass
layout["Free"] = [95, 2577, 236, 2676]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 49, 58)
    canvas.paste(_c15, (383, 4), _c15)
except Exception:
    pass
layout["icon_15"] = [383, 4, 432, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/16_icon_Free.png
try:
    _c16 = get_crop(16, 75, 72)
    canvas.paste(_c16, (249, 2588), _c16)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/17_text_9.13.png
try:
    _c17 = get_crop(17, 91, 43)
    canvas.paste(_c17, (20, 17), _c17)
except Exception:
    pass
layout["9.13"] = [20, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/18_text_Minorities_Building_..png
try:
    _c18 = get_crop(18, 560, 87)
    canvas.paste(_c18, (249, 146), _c18)
except Exception:
    pass
layout["Minorities_Building_."] = [249, 146, 809, 233]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/19_text_We_invite_you_to_a_value-packed_educatio.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (36, 108), _c19)
except Exception:
    pass
layout["We_invite_you_to_a_value-"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/20_text_on_how_Minorities_can_profit_and_build_g.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1116, 108), _c20)
except Exception:
    pass
layout["on_how_Minorities_can_pro"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/21_text_During_this_1_hour_session_you_will_lear.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1116, 108), _c21)
except Exception:
    pass
layout["During_this_1_hour_sessio"] = [1116, 108, 1260, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/22_text_affects_growth_access_to_capital_and_the.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (36, 108), _c22)
except Exception:
    pass
layout["affects_growth;_access_to"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/23_text_environment..png
try:
    _c23 = get_crop(23, 288, 50)
    canvas.paste(_c23, (42, 800), _c23)
except Exception:
    pass
layout["environment."] = [42, 800, 330, 850]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/24_text_Date_Saturday_April_13th_2024at_I_00_PM.png
try:
    _c24 = get_crop(24, 928, 68)
    canvas.paste(_c24, (43, 918), _c24)
except Exception:
    pass
layout["Date:_Saturday,_April_13t"] = [43, 918, 971, 986]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/25_text_Location.png
try:
    _c25 = get_crop(25, 205, 52)
    canvas.paste(_c25, (44, 1052), _c25)
except Exception:
    pass
layout["Location:"] = [44, 1052, 249, 1104]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/26_text_Los_Angeles_Convention_Center.png
try:
    _c26 = get_crop(26, 671, 66)
    canvas.paste(_c26, (43, 1176), _c26)
except Exception:
    pass
layout["Los_Angeles_Convention_Ce"] = [43, 1176, 714, 1242]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/27_text_West_HallA.png
try:
    _c27 = get_crop(27, 239, 52)
    canvas.paste(_c27, (44, 1304), _c27)
except Exception:
    pass
layout["West_HallA"] = [44, 1304, 283, 1356]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/28_text_1201_South_Figueroa_Street.png
try:
    _c28 = get_crop(28, 597, 72)
    canvas.paste(_c28, (44, 1364), _c28)
except Exception:
    pass
layout["1201_South_Figueroa_Stree"] = [44, 1364, 641, 1436]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/29_text_Room_Show_Floor_Theatre_1.png
try:
    _c29 = get_crop(29, 590, 61)
    canvas.paste(_c29, (41, 1554), _c29)
except Exception:
    pass
layout["Room:_Show_Floor_Theatre_"] = [41, 1554, 631, 1615]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/30_text_Benefits_of.png
try:
    _c30 = get_crop(30, 237, 52)
    canvas.paste(_c30, (44, 1746), _c30)
except Exception:
    pass
layout["Benefits_of"] = [44, 1746, 281, 1798]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/31_text_Learn_the_top.png
try:
    _c31 = get_crop(31, 287, 57)
    canvas.paste(_c31, (70, 1873), _c31)
except Exception:
    pass
layout["Learn_the_top"] = [70, 1873, 357, 1930]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/32_text_to_success_in.png
try:
    _c32 = get_crop(32, 281, 52)
    canvas.paste(_c32, (458, 1873), _c32)
except Exception:
    pass
layout["to_success_in"] = [458, 1873, 739, 1925]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/33_text_Learn_how_to.png
try:
    _c33 = get_crop(33, 276, 50)
    canvas.paste(_c33, (72, 2000), _c33)
except Exception:
    pass
layout["Learn_how_to"] = [72, 2000, 348, 2050]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/34_text_access_to_capital.png
try:
    _c34 = get_crop(34, 362, 57)
    canvas.paste(_c34, (428, 2000), _c34)
except Exception:
    pass
layout["access_to_capital"] = [428, 2000, 790, 2057]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/35_text_Learn_the_advantages_and_disadvantages_o.png
try:
    _c35 = get_crop(35, 99, 96)
    canvas.paste(_c35, (996, 2444), _c35)
except Exception:
    pass
layout["Learn_the_advantages_and_"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/bbe931c7fbf3446c80521f77d3e413aa/step_15_2024_3_20_17_10_bbe931c7fbf3446c80521f77d3e413aa-17/36_text_Complimentary_Access.png
try:
    _c36 = get_crop(36, 75, 72)
    canvas.paste(_c36, (249, 2588), _c36)
except Exception:
    pass
layout["Complimentary_Access"] = [249, 2588, 324, 2660]
